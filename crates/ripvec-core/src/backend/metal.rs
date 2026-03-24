//! Metal GPU embedding backend for Apple Silicon.
//!
//! Provides low-level Metal device initialization, MSL kernel compilation, and
//! compute dispatch infrastructure. Uses `objc2-metal` Rust bindings to Apple's
//! Metal framework.
//!
//! # Architecture
//!
//! The Metal backend detects the GPU family at init time ([`ChipFamily`]) and
//! stores a pre-compiled compute pipeline alongside the device and command
//! queue. All GPU resources use `StorageModeShared` (unified memory on Apple
//! Silicon) to avoid explicit CPU-GPU copies.
//!
//! Model weights are loaded via zero-copy mmap: the safetensors file is
//! memory-mapped and wrapped as a single Metal buffer using
//! `newBufferWithBytesNoCopy`. Individual tensors are addressed via
//! [`WeightRef`] offsets into this buffer.
//!
//! # Thread safety
//!
//! Metal device and command queue are thread-safe. `MetalBackend` implements
//! `Send + Sync` so it can be shared across the ring-buffer pipeline.

use std::collections::HashMap;
use std::path::Path;

use hf_hub::api::sync::Api;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSString, NSUInteger};
use objc2_metal::{
    MTLBuffer, MTLCommandQueue, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLGPUFamily, MTLLibrary, MTLResourceOptions, MTLSize,
};
use safetensors::SafeTensors;

use super::{DeviceHint, EmbedBackend, Encoding};

// ---------------------------------------------------------------------------
// CoreGraphics linkage (required for MTLCreateSystemDefaultDevice)
// ---------------------------------------------------------------------------

#[expect(unsafe_code, reason = "extern block required for CoreGraphics linkage")]
#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}

// ---------------------------------------------------------------------------
// Device + pipeline helpers
// ---------------------------------------------------------------------------

/// Create the default Metal GPU device.
fn create_device() -> crate::Result<Retained<ProtocolObject<dyn MTLDevice>>> {
    MTLCreateSystemDefaultDevice()
        .ok_or_else(|| crate::Error::Metal("no Metal device available".into()))
}

/// Create a command queue for submitting work.
fn create_queue(
    device: &ProtocolObject<dyn MTLDevice>,
) -> crate::Result<Retained<ProtocolObject<dyn MTLCommandQueue>>> {
    device
        .newCommandQueue()
        .ok_or_else(|| crate::Error::Metal("failed to create command queue".into()))
}

/// Compile MSL source code into a Metal library.
fn compile_library(
    device: &ProtocolObject<dyn MTLDevice>,
    source: &str,
) -> crate::Result<Retained<ProtocolObject<dyn MTLLibrary>>> {
    let ns_source = NSString::from_str(source);
    device
        .newLibraryWithSource_options_error(&ns_source, None)
        .map_err(|e| crate::Error::Metal(format!("MSL compilation failed: {e}")))
}

/// Create a compute pipeline state from a named kernel function.
fn create_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
    name: &str,
) -> crate::Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
    let ns_name = NSString::from_str(name);
    let function = library
        .newFunctionWithName(&ns_name)
        .ok_or_else(|| crate::Error::Metal(format!("function '{name}' not found in library")))?;
    device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|e| crate::Error::Metal(format!("pipeline creation failed: {e}")))
}

// ---------------------------------------------------------------------------
// Kernel pipelines (compiled MSL compute functions)
// ---------------------------------------------------------------------------

/// Pre-compiled Metal compute pipeline states for all BERT inference kernels.
///
/// Created once at model load time by compiling the MSL source from
/// [`super::metal_kernels::KERNELS`] and extracting each named function.
struct KernelPipelines {
    /// Embedding table lookup.
    embedding_lookup: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Add embedding table values to existing output.
    add_embeddings: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Layer normalization with threadgroup reduction.
    layer_norm: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// GELU activation (tanh approximation, in-place).
    gelu: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// `SwiGLU` activation (value * silu(gate), interleaved input layout).
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "available for future fused-weight SwiGLU optimization"
        )
    )]
    swiglu: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// `RoPE` with pre-computed cos/sin tables.
    rope_cached: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Fused scale + attention-mask + softmax.
    fused_scale_mask_softmax: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Fused residual add + layer norm.
    fused_residual_layernorm: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Fused bias + GELU activation for `ClassicBert` FFN.
    fused_bias_gelu: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Fused bias + residual add for output projections.
    fused_bias_residual: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Unified `SwiGLU` kernel handling both bias and no-bias paths (interleaved input).
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "available for future fused-weight SwiGLU optimization"
        )
    )]
    fused_swiglu: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Split QKV `[batch*seq, 3*hidden]` into Q,K,V `[batch*num_heads, seq, head_dim]`.
    qkv_split: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Reshape attention output `[batch*num_heads, seq, head_dim]` to `[batch*seq, hidden]`.
    attn_reshape: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// CLS pooling (extract row 0 per batch element).
    cls_pool: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// L2 normalize each row.
    l2_normalize: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Build float attention mask from int mask.
    build_attn_mask: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Convert FP32 to FP16.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "available for future FP16 GEMM optimization")
    )]
    f32_to_f16: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Add bias in-place: `x[idx] += bias[idx % cols]`.
    add_bias: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Head reshape: `[batch*seq, hidden]` to `[batch*num_heads, seq, head_dim]`.
    head_reshape: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Two-input `SwiGLU`: `output = value * silu(gate)` with separate buffers.
    swiglu_two_input: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// GEMM kernel using `simdgroup_matrix_multiply_accumulate`.
    gemm: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Batched GEMM: same kernel with z-dimension for batch/head index.
    gemm_batched: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl KernelPipelines {
    /// Compile all MSL kernels and create pipeline states.
    fn compile(device: &ProtocolObject<dyn MTLDevice>) -> crate::Result<Self> {
        let library = compile_library(device, super::metal_kernels::KERNELS)?;
        let p = |name: &str| create_pipeline(device, &library, name);

        // GEMM kernel is compiled from a separate source that includes the
        // simdgroup_matrix_storage header (uses #pragma METAL internals).
        let gemm_library = compile_library(device, super::metal_kernels::GEMM_KERNEL)?;

        Ok(Self {
            embedding_lookup: p("embedding_lookup_kernel")?,
            add_embeddings: p("add_embeddings_kernel")?,
            layer_norm: p("layer_norm_kernel")?,
            gelu: p("gelu_kernel")?,
            swiglu: p("swiglu_kernel")?,
            rope_cached: p("rope_cached_kernel")?,
            fused_scale_mask_softmax: p("fused_scale_mask_softmax_kernel")?,
            fused_residual_layernorm: p("fused_residual_layernorm_kernel")?,
            fused_bias_gelu: p("fused_bias_gelu_kernel")?,
            fused_bias_residual: p("fused_bias_residual_kernel")?,
            fused_swiglu: p("fused_swiglu_kernel")?,
            qkv_split: p("qkv_split_kernel")?,
            attn_reshape: p("attn_reshape_kernel")?,
            cls_pool: p("cls_pool_kernel")?,
            l2_normalize: p("l2_normalize_kernel")?,
            build_attn_mask: p("build_attn_mask_kernel")?,
            f32_to_f16: p("f32_to_f16_kernel")?,
            add_bias: p("add_bias_kernel")?,
            head_reshape: p("head_reshape_kernel")?,
            swiglu_two_input: p("swiglu_two_input_kernel")?,
            gemm: create_pipeline(device, &gemm_library, "gemm_kernel")?,
            gemm_batched: create_pipeline(device, &gemm_library, "gemm_batched_kernel")?,
        })
    }
}

// ---------------------------------------------------------------------------
// Dispatch helpers
// ---------------------------------------------------------------------------

/// Dispatch a 1D compute kernel over `n` elements.
///
/// Automatically chooses a threadgroup size based on the pipeline's maximum.
/// Uses `dispatchThreads:threadsPerThreadgroup:` for non-uniform grids.
fn dispatch_1d(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    n: usize,
) {
    let max_threads = pipeline.maxTotalThreadsPerThreadgroup();
    let threads_per_tg = max_threads.min(n).max(1);
    let grid = MTLSize {
        width: n,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: threads_per_tg,
        height: 1,
        depth: 1,
    };
    encoder.dispatchThreads_threadsPerThreadgroup(grid, tg);
}

/// Dispatch a per-row kernel using threadgroups (one threadgroup per row).
///
/// Used for kernels like `layer_norm`, `l2_normalize`, `fused_scale_mask_softmax`,
/// and `fused_residual_layernorm` that use threadgroup reductions. Each row is
/// handled by one threadgroup with up to 256 threads.
fn dispatch_rows(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    num_rows: usize,
    threads_per_row: usize,
) {
    let max_threads = pipeline.maxTotalThreadsPerThreadgroup();
    let tpg = max_threads.min(threads_per_row).clamp(1, 256);
    let grid = MTLSize {
        width: num_rows,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: tpg,
        height: 1,
        depth: 1,
    };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

/// Dispatch a GEMM kernel: C\[M,N\] = A\[M,K\] * B\[K,N\] (or B^T when `trans_b`).
///
/// Each threadgroup computes a 32x32 output tile using 16 SIMD groups (4x4
/// arrangement of 8x8 tiles). Grid is `ceil(N/32) x ceil(M/32)` threadgroups.
///
/// # Safety
///
/// Caller must ensure buffer offsets and sizes are valid.
#[expect(
    unsafe_code,
    clippy::too_many_arguments,
    clippy::borrow_as_ptr,
    reason = "Metal buffer binding requires unsafe setBuffer/setBytes calls with raw pointers"
)]
fn dispatch_gemm(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    a_buffer: &ProtocolObject<dyn MTLBuffer>,
    a_offset: usize,
    b_buffer: &ProtocolObject<dyn MTLBuffer>,
    b_offset: usize,
    c_buffer: &ProtocolObject<dyn MTLBuffer>,
    c_offset: usize,
    m: u32,
    n: u32,
    k: u32,
    trans_b: bool,
) {
    const TILE_M: usize = 32;
    const TILE_N: usize = 32;
    // 16 SIMD groups per threadgroup (4x4 arrangement), 32 threads per SIMD group
    const SIMD_GROUPS_PER_TG: usize = 16;
    const THREADS_PER_SIMD: usize = 32;

    encoder.setComputePipelineState(pipeline);

    unsafe {
        encoder.setBuffer_offset_atIndex(Some(a_buffer), a_offset, 0);
        encoder.setBuffer_offset_atIndex(Some(b_buffer), b_offset, 1);
        encoder.setBuffer_offset_atIndex(Some(c_buffer), c_offset, 2);
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::new(&m as *const u32 as *mut _).unwrap(),
            core::mem::size_of::<u32>(),
            3,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::new(&n as *const u32 as *mut _).unwrap(),
            core::mem::size_of::<u32>(),
            4,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::new(&k as *const u32 as *mut _).unwrap(),
            core::mem::size_of::<u32>(),
            5,
        );
        let trans_b_u32: u32 = if trans_b { 1 } else { 0 };
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::new(&trans_b_u32 as *const u32 as *mut _).unwrap(),
            core::mem::size_of::<u32>(),
            6,
        );
    }

    let grid_x = (n as usize).div_ceil(TILE_N);
    let grid_y = (m as usize).div_ceil(TILE_M);

    let threadgroups = MTLSize {
        width: grid_x,
        height: grid_y,
        depth: 1,
    };
    let threads_per_tg = MTLSize {
        width: SIMD_GROUPS_PER_TG * THREADS_PER_SIMD,
        height: 1,
        depth: 1,
    };

    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
}

/// Dispatch a batched GEMM: one GEMM per batch slice (z-dimension).
///
/// Replaces the per-head dispatch loop in attention. All `batch_count` GEMMs
/// run in a single dispatch, using strided offsets into A/B/C buffers.
///
/// Strides are in **elements** (not bytes).
#[expect(
    unsafe_code,
    clippy::too_many_arguments,
    clippy::borrow_as_ptr,
    reason = "Metal buffer binding requires unsafe setBuffer/setBytes calls with raw pointers"
)]
fn dispatch_gemm_batched(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    a_buffer: &ProtocolObject<dyn MTLBuffer>,
    a_offset: usize,
    b_buffer: &ProtocolObject<dyn MTLBuffer>,
    b_offset: usize,
    c_buffer: &ProtocolObject<dyn MTLBuffer>,
    c_offset: usize,
    m: u32,
    n: u32,
    k: u32,
    trans_b: bool,
    stride_a: u32,
    stride_b: u32,
    stride_c: u32,
    batch_count: u32,
) {
    const TILE_M: usize = 32;
    const TILE_N: usize = 32;
    const SIMD_GROUPS_PER_TG: usize = 16;
    const THREADS_PER_SIMD: usize = 32;

    encoder.setComputePipelineState(pipeline);

    unsafe {
        encoder.setBuffer_offset_atIndex(Some(a_buffer), a_offset, 0);
        encoder.setBuffer_offset_atIndex(Some(b_buffer), b_offset, 1);
        encoder.setBuffer_offset_atIndex(Some(c_buffer), c_offset, 2);
        let params: [u32; 7] = [
            m,
            n,
            k,
            if trans_b { 1 } else { 0 },
            stride_a,
            stride_b,
            stride_c,
        ];
        for (i, val) in params.iter().enumerate() {
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new(val as *const u32 as *mut _).unwrap(),
                core::mem::size_of::<u32>(),
                3 + i,
            );
        }
    }

    let grid_x = (n as usize).div_ceil(TILE_N);
    let grid_y = (m as usize).div_ceil(TILE_M);

    let threadgroups = MTLSize {
        width: grid_x,
        height: grid_y,
        depth: batch_count as usize,
    };
    let threads_per_tg = MTLSize {
        width: SIMD_GROUPS_PER_TG * THREADS_PER_SIMD,
        height: 1,
        depth: 1,
    };

    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
}

/// Maximum batch size for pre-allocated workspace buffers.
///
/// Workspace is allocated once for `MAX_BATCH * max_seq_len`. Batches larger
/// than this are split into sub-batches.
const MAX_BATCH: i32 = 128;

/// Maximum sequence length for workspace allocation.
const MAX_SEQ: i32 = 512;

/// Set a `constant int&` parameter at the given buffer index.
///
/// # Safety
///
/// Caller must ensure `index` does not conflict with other buffer bindings.
#[expect(
    unsafe_code,
    clippy::borrow_as_ptr,
    reason = "Metal setBytes requires unsafe FFI with raw ptr"
)]
fn set_i32_param(encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>, value: i32, index: usize) {
    unsafe {
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::new(&value as *const i32 as *mut _).expect("non-null const ptr"),
            core::mem::size_of::<i32>(),
            index,
        );
    }
}

/// Set a `constant float&` parameter at the given buffer index.
///
/// # Safety
///
/// Caller must ensure `index` does not conflict with other buffer bindings.
#[expect(
    unsafe_code,
    clippy::borrow_as_ptr,
    reason = "Metal setBytes requires unsafe FFI with raw ptr"
)]
fn set_f32_param(encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>, value: f32, index: usize) {
    unsafe {
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::new(&value as *const f32 as *mut _).expect("non-null const ptr"),
            core::mem::size_of::<f32>(),
            index,
        );
    }
}

/// Bind a Metal buffer at the given index with a byte offset.
///
/// # Safety
///
/// Caller must ensure the buffer, offset, and index are valid.
#[expect(unsafe_code, reason = "Metal setBuffer requires unsafe FFI")]
fn set_buffer(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    buffer: &ProtocolObject<dyn MTLBuffer>,
    offset: usize,
    index: usize,
) {
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(buffer), offset, index);
    }
}

// ---------------------------------------------------------------------------
// Pre-allocated workspace buffers
// ---------------------------------------------------------------------------

/// Pre-allocated Metal workspace buffers for BERT inference.
///
/// Sized at model load time for `MAX_BATCH * MAX_SEQ`. Reused across all
/// forward passes -- contents are overwritten each time.
struct MetalWorkspace {
    /// Hidden state buffer A (ping-pong with `hidden_b`).
    hidden_a: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Hidden state buffer B (ping-pong with `hidden_a`).
    hidden_b: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// QKV projection output `[batch*seq, 3*hidden]`.
    qkv: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Q after split/reshape `[batch*num_heads, seq, head_dim]`.
    q: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// K after split/reshape `[batch*num_heads, seq, head_dim]`.
    k: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// V after split/reshape `[batch*num_heads, seq, head_dim]`.
    v: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Attention scores `[batch*num_heads, seq, seq]`.
    scores: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Attention output `[batch*num_heads, seq, head_dim]`.
    attn_out: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Context after reshape `[batch*seq, hidden]`.
    context: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// FFN intermediate activations.
    ffn_inter: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// FFN activated output (for `SwiGLU`).
    activated: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Scratch buffer for intermediate results.
    scratch: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Projected output (after output projection GEMM).
    projected: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// CLS pooled embeddings `[batch, hidden]`.
    cls: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Float attention mask `[batch, seq]`.
    mask: Retained<ProtocolObject<dyn MTLBuffer>>,
}

/// Pre-computed `RoPE` cos/sin tables on Metal (`NomicBert` only).
struct MetalRopeCache {
    /// Cosine table `[max_seq, head_dim/2]`.
    cos: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Sine table `[max_seq, head_dim/2]`.
    sin: Retained<ProtocolObject<dyn MTLBuffer>>,
}

/// Allocate workspace buffers for the maximum expected batch/sequence size.
#[expect(
    clippy::cast_sign_loss,
    reason = "all dimension values are positive from validated config"
)]
fn allocate_workspace(
    device: &ProtocolObject<dyn MTLDevice>,
    config: &BertConfig,
    inter_dim: i32,
) -> crate::Result<MetalWorkspace> {
    let hd = config.hidden_size;
    let nh = config.num_attention_heads;
    let head_dim = hd / nh;
    let max_seq = config.max_position_embeddings.min(MAX_SEQ);
    let bs = MAX_BATCH * max_seq;
    let bh = MAX_BATCH * nh;

    let inter_out = match config.variant {
        ModelVariant::ClassicBert => inter_dim,
        ModelVariant::NomicBert => 2 * inter_dim,
    };

    let alloc = |n: usize| -> crate::Result<Retained<ProtocolObject<dyn MTLBuffer>>> {
        let size = (n * core::mem::size_of::<f32>()) as NSUInteger;
        device
            .newBufferWithLength_options(size, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| crate::Error::Metal(format!("workspace alloc failed ({n} floats)")))
    };

    Ok(MetalWorkspace {
        hidden_a: alloc((bs * hd) as usize)?,
        hidden_b: alloc((bs * hd) as usize)?,
        qkv: alloc((bs * 3 * hd) as usize)?,
        q: alloc((bh * max_seq * head_dim) as usize)?,
        k: alloc((bh * max_seq * head_dim) as usize)?,
        v: alloc((bh * max_seq * head_dim) as usize)?,
        scores: alloc((bh * max_seq * max_seq) as usize)?,
        attn_out: alloc((bh * max_seq * head_dim) as usize)?,
        context: alloc((bs * hd) as usize)?,
        ffn_inter: alloc((bs * inter_out) as usize)?,
        activated: alloc((bs * inter_dim) as usize)?,
        scratch: alloc((bs * hd) as usize)?,
        projected: alloc((bs * hd) as usize)?,
        cls: alloc((MAX_BATCH * hd) as usize)?,
        mask: alloc((MAX_BATCH * max_seq) as usize)?,
    })
}

/// Build the `RoPE` cos/sin cache for `NomicBert`.
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    reason = "head_dim/seq are small ints; f64->f32 truncation is intentional"
)]
fn build_rope_cache(
    device: &ProtocolObject<dyn MTLDevice>,
    config: &BertConfig,
) -> crate::Result<MetalRopeCache> {
    let head_dim = config.hidden_size / config.num_attention_heads;
    let half_dim = (head_dim / 2) as usize;
    let rope_base = f64::from(config.rotary_emb_base);
    let rope_max_seq = config.max_position_embeddings.min(MAX_SEQ) as usize;

    let mut cos_table = vec![0.0_f32; rope_max_seq * half_dim];
    let mut sin_table = vec![0.0_f32; rope_max_seq * half_dim];

    for i in 0..half_dim {
        let theta = rope_base.powf(-2.0 * i as f64 / f64::from(head_dim));
        for pos in 0..rope_max_seq {
            let angle = pos as f64 * theta;
            cos_table[pos * half_dim + i] = angle.cos() as f32;
            sin_table[pos * half_dim + i] = angle.sin() as f32;
        }
    }

    let make_buf = |data: &[f32]| -> crate::Result<Retained<ProtocolObject<dyn MTLBuffer>>> {
        let size = core::mem::size_of_val(data) as NSUInteger;
        #[expect(unsafe_code, reason = "newBufferWithBytes requires unsafe FFI")]
        unsafe {
            device.newBufferWithBytes_length_options(
                std::ptr::NonNull::new(data.as_ptr() as *mut _).expect("non-null data ptr"),
                size,
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or_else(|| crate::Error::Metal("rope cache buffer alloc failed".into()))
    };

    Ok(MetalRopeCache {
        cos: make_buf(&cos_table)?,
        sin: make_buf(&sin_table)?,
    })
}

// ---------------------------------------------------------------------------
// Chip family detection
// ---------------------------------------------------------------------------

/// Apple GPU family classification for tuning compute kernels.
///
/// Different chip families have different optimal GEMM tile sizes and
/// support different hardware features (e.g. async copies).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChipFamily {
    /// M1, M2 (Apple7/8) -- prefer async copy, 48x48 GEMM tiles.
    Apple7_8,
    /// M3+ (Apple9) -- no async copy, 32x32 GEMM tiles.
    Apple9,
}

impl ChipFamily {
    /// Detect the GPU family from the device capabilities.
    fn detect(device: &ProtocolObject<dyn MTLDevice>) -> Self {
        if device.supportsFamily(MTLGPUFamily::Apple9) {
            Self::Apple9
        } else {
            Self::Apple7_8
        }
    }
}

// ---------------------------------------------------------------------------
// Model variant detection + BertConfig (mirrors cpu.rs)
// ---------------------------------------------------------------------------

/// Which BERT variant the loaded weights correspond to.
///
/// `ClassicBert` uses learned position embeddings, GELU activation, and
/// biased QKV projections. `NomicBert` uses `RoPE`, `SwiGLU`, and unbiased
/// projections.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelVariant {
    /// Standard BERT / BGE models (e.g. `BAAI/bge-small-en-v1.5`).
    ClassicBert,
    /// `NomicBert` models (e.g. `nomic-ai/CodeRankEmbed`, `nomic-embed-text-v1.5`).
    NomicBert,
}

/// Detect the model variant by inspecting weight names.
///
/// `ClassicBert` has `embeddings.position_embeddings.weight`; `NomicBert`
/// does not (it uses rotary position encoding instead).
fn detect_variant(tensors: &SafeTensors<'_>) -> ModelVariant {
    if tensors
        .tensor("embeddings.position_embeddings.weight")
        .is_ok()
    {
        ModelVariant::ClassicBert
    } else {
        ModelVariant::NomicBert
    }
}

/// Configuration for a BERT-style encoder model.
///
/// Matches the `config.json` schema from `HuggingFace` model repos.
/// Supports both `ClassicBert` and `NomicBert` config key names.
#[derive(Debug, Clone)]
struct BertConfig {
    /// Which variant this config describes.
    variant: ModelVariant,
    /// Hidden dimension (384 for bge-small, 768 for nomic).
    hidden_size: i32,
    /// Number of transformer layers.
    num_hidden_layers: i32,
    /// Number of attention heads.
    num_attention_heads: i32,
    /// Maximum sequence length (512 for classic, 8192 for nomic).
    max_position_embeddings: i32,
    /// Base for rotary embeddings (only used by `NomicBert`).
    rotary_emb_base: f32,
    /// Layer norm epsilon.
    layer_norm_eps: f32,
}

impl BertConfig {
    /// Parse from a `config.json` value, dispatching on `variant`.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "config values are small ints/floats that fit in i32/f32"
    )]
    fn from_json(v: &serde_json::Value, variant: ModelVariant) -> crate::Result<Self> {
        let get_i32 = |key: &str| -> crate::Result<i32> {
            v.get(key)
                .and_then(serde_json::Value::as_i64)
                .map(|n| n as i32)
                .ok_or_else(|| crate::Error::Other(anyhow::anyhow!("missing config key: {key}")))
        };
        let get_f64 = |key: &str| -> crate::Result<f64> {
            v.get(key)
                .and_then(serde_json::Value::as_f64)
                .ok_or_else(|| crate::Error::Other(anyhow::anyhow!("missing config key: {key}")))
        };

        let layer_norm_eps =
            get_f64("layer_norm_epsilon").or_else(|_| get_f64("layer_norm_eps"))? as f32;

        match variant {
            ModelVariant::ClassicBert => Ok(Self {
                variant,
                hidden_size: get_i32("hidden_size")?,
                num_hidden_layers: get_i32("num_hidden_layers")?,
                num_attention_heads: get_i32("num_attention_heads")?,
                max_position_embeddings: get_i32("max_position_embeddings").unwrap_or(512),
                rotary_emb_base: 10000.0,
                layer_norm_eps,
            }),
            ModelVariant::NomicBert => {
                let hidden_size = get_i32("n_embd").or_else(|_| get_i32("hidden_size"))?;
                let num_hidden_layers =
                    get_i32("n_layer").or_else(|_| get_i32("num_hidden_layers"))?;
                let num_attention_heads =
                    get_i32("n_head").or_else(|_| get_i32("num_attention_heads"))?;
                let max_position_embeddings = get_i32("n_positions")
                    .or_else(|_| get_i32("max_position_embeddings"))
                    .unwrap_or(8192);
                let rotary_emb_base = get_f64("rotary_emb_base")
                    .map(|v| v as f32)
                    .unwrap_or(10000.0);

                Ok(Self {
                    variant,
                    hidden_size,
                    num_hidden_layers,
                    num_attention_heads,
                    max_position_embeddings,
                    rotary_emb_base,
                    layer_norm_eps,
                })
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Weight references (zero-copy offsets into mmap'd safetensors)
// ---------------------------------------------------------------------------

/// Reference to a weight tensor within the mmap'd safetensors buffer.
///
/// Instead of copying tensor data, `WeightRef` stores the byte offset and
/// shape so kernels can index directly into the Metal buffer backed by the
/// mmap'd file.
struct WeightRef {
    /// Byte offset from the start of the mmap into the safetensors file.
    offset: usize,
    /// Tensor shape (e.g. `[384, 384]`).
    shape: Vec<usize>,
    /// Number of f32 elements.
    numel: usize,
}

/// Parse safetensors header and extract byte offsets for all weight tensors.
///
/// Returns a map of tensor name to [`WeightRef`] plus the byte offset where
/// the tensor data region starts (after the JSON header).
fn extract_weight_refs(mmap: &[u8]) -> crate::Result<(HashMap<String, WeightRef>, usize)> {
    let tensors = SafeTensors::deserialize(mmap)
        .map_err(|e| crate::Error::Metal(format!("safetensors parse: {e}")))?;

    // safetensors format: [8-byte header_len][JSON header][tensor data...]
    // data_offsets() returns offsets relative to the DATA region start
    let header_len_u64 = u64::from_le_bytes(
        mmap[..8]
            .try_into()
            .map_err(|_| crate::Error::Metal("safetensors file too short".into()))?,
    );
    let header_len = usize::try_from(header_len_u64)
        .map_err(|_| crate::Error::Metal("safetensors header length overflow".into()))?;
    let data_start = 8 + header_len;

    let mmap_base = mmap.as_ptr() as usize;
    let mut refs = HashMap::new();
    for (name, view) in tensors.tensors() {
        // Compute offset by finding where view.data() sits within the mmap.
        // view.data() returns a slice into the mmap, so pointer arithmetic
        // gives us the absolute byte offset from the start of the file.
        let offset = view.data().as_ptr() as usize - mmap_base;

        let shape: Vec<usize> = view.shape().to_vec();
        let numel: usize = shape.iter().product();
        refs.insert(
            name.clone(),
            WeightRef {
                offset,
                shape,
                numel,
            },
        );
    }
    Ok((refs, data_start))
}

/// Look up a required weight tensor by name.
fn get_weight_ref<'a>(
    refs: &'a HashMap<String, WeightRef>,
    name: &str,
) -> crate::Result<&'a WeightRef> {
    refs.get(name)
        .ok_or_else(|| crate::Error::Metal(format!("missing weight: {name}")))
}

/// Look up an optional weight tensor by name.
fn try_get_weight_ref<'a>(
    refs: &'a HashMap<String, WeightRef>,
    name: &str,
) -> Option<&'a WeightRef> {
    refs.get(name)
}

// ---------------------------------------------------------------------------
// Zero-copy Metal buffer from mmap
// ---------------------------------------------------------------------------

/// Apple Silicon system page size (16KB).
const PAGE_SIZE: usize = 16384;

/// Memory-map a safetensors file and wrap it as a zero-copy Metal buffer.
///
/// The returned `Mmap` must outlive the `MTLBuffer` -- callers store the mmap
/// in a field declared AFTER the buffer to ensure correct drop order.
///
/// # Safety
///
/// Uses unsafe for `Mmap::map` (file must not be modified while mapped) and
/// `newBufferWithBytesNoCopy` (pointer must be page-aligned, which mmap
/// guarantees).
#[expect(
    unsafe_code,
    reason = "mmap + newBufferWithBytesNoCopy require unsafe FFI"
)]
fn load_weights_zero_copy(
    device: &ProtocolObject<dyn MTLDevice>,
    weights_path: &Path,
) -> crate::Result<(memmap2::Mmap, Retained<ProtocolObject<dyn MTLBuffer>>)> {
    let file = std::fs::File::open(weights_path).map_err(|e| crate::Error::Io {
        path: weights_path.display().to_string(),
        source: e,
    })?;

    let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| crate::Error::Io {
        path: weights_path.display().to_string(),
        source: e,
    })?;

    // Round up to page boundary (16KB on Apple Silicon)
    let aligned_len = mmap.len().next_multiple_of(PAGE_SIZE);

    let buffer = unsafe {
        device.newBufferWithBytesNoCopy_length_options_deallocator(
            std::ptr::NonNull::new(mmap.as_ptr() as *mut _)
                .ok_or_else(|| crate::Error::Metal("mmap returned null pointer".into()))?,
            aligned_len as NSUInteger,
            MTLResourceOptions::StorageModeShared,
            None, // Rust owns the mmap lifetime via _mmap field
        )
    }
    .ok_or_else(|| {
        crate::Error::Metal("zero-copy buffer creation failed (pointer not page-aligned?)".into())
    })?;

    Ok((mmap, buffer))
}

// ---------------------------------------------------------------------------
// Metal BERT model structs (WeightRef-based)
// ---------------------------------------------------------------------------

/// Embedding layer weight references.
struct MetalBertEmbeddings {
    /// Word embedding table `[vocab_size, hidden_size]`.
    word_embeddings: WeightRef,
    /// Learned position embeddings (`ClassicBert` only).
    position_embeddings: Option<WeightRef>,
    /// Token type embeddings (segment embeddings).
    token_type_embeddings: Option<WeightRef>,
    /// Layer norm gamma.
    layer_norm_weight: WeightRef,
    /// Layer norm beta.
    layer_norm_bias: WeightRef,
    /// Layer norm epsilon.
    layer_norm_eps: f32,
}

/// Self-attention weight references for one encoder layer.
///
/// For `ClassicBert`, Q/K/V are stored as three separate `WeightRef`s (since
/// we use zero-copy and cannot concatenate in-place). For `NomicBert`, the
/// fused QKV weight is a single tensor in the safetensors file.
struct MetalBertSelfAttention {
    /// Fused QKV weight (`NomicBert`) or Q weight (`ClassicBert`).
    qkv_weight: WeightRef,
    /// K weight (`ClassicBert` only; `None` for `NomicBert`).
    k_weight: Option<WeightRef>,
    /// V weight (`ClassicBert` only; `None` for `NomicBert`).
    v_weight: Option<WeightRef>,
    /// Fused QKV bias (`NomicBert`) or Q bias (`ClassicBert`).
    qkv_bias: Option<WeightRef>,
    /// K bias (`ClassicBert` only).
    k_bias: Option<WeightRef>,
    /// V bias (`ClassicBert` only).
    v_bias: Option<WeightRef>,
    /// Output projection weight.
    output_weight: WeightRef,
    /// Output projection bias.
    output_bias: Option<WeightRef>,
    /// Post-attention layer norm gamma.
    output_ln_weight: WeightRef,
    /// Post-attention layer norm beta.
    output_ln_bias: WeightRef,
    /// Number of attention heads (per-layer, matches backend-level `num_heads`).
    #[expect(
        dead_code,
        reason = "stored per-layer for future per-layer head count support"
    )]
    num_heads: i32,
    /// Dimension per head (per-layer, matches backend-level `head_dim`).
    #[expect(
        dead_code,
        reason = "stored per-layer for future per-layer head dim support"
    )]
    head_dim: i32,
    /// Layer norm epsilon.
    layer_norm_eps: f32,
    /// Rotary embedding base (`NomicBert` only).
    rotary_emb_base: Option<f32>,
}

/// Feed-forward network weight references for one encoder layer.
struct MetalBertFfn {
    /// Intermediate (up) projection weight.
    intermediate_weight: WeightRef,
    /// Intermediate (up) projection bias.
    intermediate_bias: Option<WeightRef>,
    /// For `NomicBert` `SwiGLU`: gate projection weight.
    gate_weight: Option<WeightRef>,
    /// Output (down) projection weight.
    output_weight: WeightRef,
    /// Output (down) projection bias.
    output_bias: Option<WeightRef>,
    /// Post-FFN layer norm gamma.
    output_ln_weight: WeightRef,
    /// Post-FFN layer norm beta.
    output_ln_bias: WeightRef,
    /// Layer norm epsilon.
    layer_norm_eps: f32,
    /// Model variant for activation function selection.
    #[expect(dead_code, reason = "stored per-layer; backend uses self.variant")]
    variant: ModelVariant,
}

/// One transformer encoder layer.
struct MetalBertLayer {
    /// Self-attention sub-layer.
    attention: MetalBertSelfAttention,
    /// Feed-forward sub-layer.
    ffn: MetalBertFfn,
}

/// Complete BERT model as weight references into a Metal buffer.
struct MetalBertModel {
    /// Embedding sub-layer.
    embeddings: MetalBertEmbeddings,
    /// Transformer encoder layers.
    layers: Vec<MetalBertLayer>,
    /// Model configuration.
    #[expect(
        dead_code,
        reason = "stored for potential future dynamic reconfiguration"
    )]
    config: BertConfig,
}

// ---------------------------------------------------------------------------
// Layer loading (WeightRef extraction)
// ---------------------------------------------------------------------------

/// Extract `ClassicBert` encoder layer weight refs.
///
/// Q/K/V are stored as three separate refs because we cannot concatenate
/// with zero-copy. The compute kernel will dispatch three separate GEMMs
/// (or use offset-based dispatch).
fn load_classic_layer_refs(
    refs: &HashMap<String, WeightRef>,
    i: i32,
    config: &BertConfig,
) -> crate::Result<(MetalBertSelfAttention, MetalBertFfn)> {
    let prefix = format!("encoder.layer.{i}");

    let attention = MetalBertSelfAttention {
        // Store Q/K/V separately (zero-copy cannot concatenate)
        qkv_weight: take_weight_ref(refs, &format!("{prefix}.attention.self.query.weight"))?,
        k_weight: Some(take_weight_ref(
            refs,
            &format!("{prefix}.attention.self.key.weight"),
        )?),
        v_weight: Some(take_weight_ref(
            refs,
            &format!("{prefix}.attention.self.value.weight"),
        )?),
        qkv_bias: try_take_weight_ref(refs, &format!("{prefix}.attention.self.query.bias")),
        k_bias: try_take_weight_ref(refs, &format!("{prefix}.attention.self.key.bias")),
        v_bias: try_take_weight_ref(refs, &format!("{prefix}.attention.self.value.bias")),
        output_weight: take_weight_ref(refs, &format!("{prefix}.attention.output.dense.weight"))?,
        output_bias: try_take_weight_ref(refs, &format!("{prefix}.attention.output.dense.bias")),
        output_ln_weight: take_weight_ref(
            refs,
            &format!("{prefix}.attention.output.LayerNorm.weight"),
        )?,
        output_ln_bias: take_weight_ref(
            refs,
            &format!("{prefix}.attention.output.LayerNorm.bias"),
        )?,
        num_heads: config.num_attention_heads,
        head_dim: config.hidden_size / config.num_attention_heads,
        layer_norm_eps: config.layer_norm_eps,
        rotary_emb_base: None,
    };

    let ffn = MetalBertFfn {
        intermediate_weight: take_weight_ref(refs, &format!("{prefix}.intermediate.dense.weight"))?,
        intermediate_bias: try_take_weight_ref(refs, &format!("{prefix}.intermediate.dense.bias")),
        gate_weight: None,
        output_weight: take_weight_ref(refs, &format!("{prefix}.output.dense.weight"))?,
        output_bias: try_take_weight_ref(refs, &format!("{prefix}.output.dense.bias")),
        output_ln_weight: take_weight_ref(refs, &format!("{prefix}.output.LayerNorm.weight"))?,
        output_ln_bias: take_weight_ref(refs, &format!("{prefix}.output.LayerNorm.bias"))?,
        layer_norm_eps: config.layer_norm_eps,
        variant: config.variant,
    };

    Ok((attention, ffn))
}

/// Extract `NomicBert` encoder layer weight refs.
fn load_nomic_layer_refs(
    refs: &HashMap<String, WeightRef>,
    i: i32,
    config: &BertConfig,
) -> crate::Result<(MetalBertSelfAttention, MetalBertFfn)> {
    let prefix = format!("encoder.layers.{i}");

    let attention = MetalBertSelfAttention {
        qkv_weight: take_weight_ref(refs, &format!("{prefix}.attn.Wqkv.weight"))?,
        k_weight: None,
        v_weight: None,
        qkv_bias: None,
        k_bias: None,
        v_bias: None,
        output_weight: take_weight_ref(refs, &format!("{prefix}.attn.out_proj.weight"))?,
        output_bias: None,
        output_ln_weight: take_weight_ref(refs, &format!("{prefix}.norm1.weight"))?,
        output_ln_bias: take_weight_ref(refs, &format!("{prefix}.norm1.bias"))?,
        num_heads: config.num_attention_heads,
        head_dim: config.hidden_size / config.num_attention_heads,
        layer_norm_eps: config.layer_norm_eps,
        rotary_emb_base: Some(config.rotary_emb_base),
    };

    // SwiGLU: fc11 = value/up, fc12 = gate, fc2 = down
    let ffn = MetalBertFfn {
        intermediate_weight: take_weight_ref(refs, &format!("{prefix}.mlp.fc11.weight"))?,
        intermediate_bias: None,
        gate_weight: Some(take_weight_ref(refs, &format!("{prefix}.mlp.fc12.weight"))?),
        output_weight: take_weight_ref(refs, &format!("{prefix}.mlp.fc2.weight"))?,
        output_bias: None,
        output_ln_weight: take_weight_ref(refs, &format!("{prefix}.norm2.weight"))?,
        output_ln_bias: take_weight_ref(refs, &format!("{prefix}.norm2.bias"))?,
        layer_norm_eps: config.layer_norm_eps,
        variant: config.variant,
    };

    Ok((attention, ffn))
}

/// Take a required weight ref by cloning its data from the map.
fn take_weight_ref(refs: &HashMap<String, WeightRef>, name: &str) -> crate::Result<WeightRef> {
    let r = get_weight_ref(refs, name)?;
    Ok(WeightRef {
        offset: r.offset,
        shape: r.shape.clone(),
        numel: r.numel,
    })
}

/// Take an optional weight ref by cloning its data from the map.
fn try_take_weight_ref(refs: &HashMap<String, WeightRef>, name: &str) -> Option<WeightRef> {
    try_get_weight_ref(refs, name).map(|r| WeightRef {
        offset: r.offset,
        shape: r.shape.clone(),
        numel: r.numel,
    })
}

// ---------------------------------------------------------------------------
// MetalBertModel construction
// ---------------------------------------------------------------------------

impl MetalBertModel {
    /// Build the model from weight refs extracted from a safetensors file.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_possible_wrap,
        reason = "hidden_size and num_layers are small positive ints from config"
    )]
    fn from_weight_refs(
        refs: &HashMap<String, WeightRef>,
        config: &BertConfig,
    ) -> crate::Result<Self> {
        let embeddings = match config.variant {
            ModelVariant::ClassicBert => MetalBertEmbeddings {
                word_embeddings: take_weight_ref(refs, "embeddings.word_embeddings.weight")?,
                position_embeddings: Some(take_weight_ref(
                    refs,
                    "embeddings.position_embeddings.weight",
                )?),
                token_type_embeddings: Some(take_weight_ref(
                    refs,
                    "embeddings.token_type_embeddings.weight",
                )?),
                layer_norm_weight: take_weight_ref(refs, "embeddings.LayerNorm.weight")?,
                layer_norm_bias: take_weight_ref(refs, "embeddings.LayerNorm.bias")?,
                layer_norm_eps: config.layer_norm_eps,
            },
            ModelVariant::NomicBert => MetalBertEmbeddings {
                word_embeddings: take_weight_ref(refs, "embeddings.word_embeddings.weight")?,
                position_embeddings: None,
                token_type_embeddings: try_take_weight_ref(
                    refs,
                    "embeddings.token_type_embeddings.weight",
                ),
                layer_norm_weight: take_weight_ref(refs, "emb_ln.weight")?,
                layer_norm_bias: take_weight_ref(refs, "emb_ln.bias")?,
                layer_norm_eps: config.layer_norm_eps,
            },
        };

        // Validate embedding dimension matches config
        if embeddings.word_embeddings.shape.len() >= 2 {
            let emb_dim = embeddings.word_embeddings.shape[1] as i32;
            if emb_dim != config.hidden_size {
                return Err(crate::Error::Metal(format!(
                    "hidden_size mismatch: config says {} but word_embeddings has dim {}",
                    config.hidden_size, emb_dim
                )));
            }
        }

        let mut layers = Vec::with_capacity(config.num_hidden_layers as usize);
        for i in 0..config.num_hidden_layers {
            let (attention, ffn) = match config.variant {
                ModelVariant::ClassicBert => load_classic_layer_refs(refs, i, config)?,
                ModelVariant::NomicBert => load_nomic_layer_refs(refs, i, config)?,
            };
            layers.push(MetalBertLayer { attention, ffn });
        }

        Ok(Self {
            embeddings,
            layers,
            config: config.clone(),
        })
    }
}

// ---------------------------------------------------------------------------
// MetalBackend
// ---------------------------------------------------------------------------

/// Metal GPU embedding backend.
///
/// Holds the device, command queue, chip family, and zero-copy model weights
/// needed to dispatch BERT inference on Apple Silicon GPUs. Weights are
/// memory-mapped from the safetensors file and wrapped as a single Metal
/// buffer via `newBufferWithBytesNoCopy`.
pub struct MetalBackend {
    /// The Metal GPU device.
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    /// Command queue for submitting compute work.
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    /// Detected GPU chip family for kernel tuning.
    #[expect(
        dead_code,
        reason = "reserved for future kernel-tuning decisions per chip family"
    )]
    chip_family: ChipFamily,
    /// Zero-copy Metal buffer wrapping the mmap'd safetensors data.
    /// Must be declared BEFORE `_mmap` so it is dropped first.
    weight_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// BERT model with weight refs indexing into `weight_buffer`.
    model: MetalBertModel,
    /// Pre-compiled compute pipeline states for all MSL kernels.
    kernels: KernelPipelines,
    /// Pre-allocated workspace buffers for intermediate results.
    workspace: MetalWorkspace,
    /// Pre-computed `RoPE` cos/sin tables (`NomicBert` only).
    rope_cache: Option<MetalRopeCache>,
    /// Hidden dimension for output vector size.
    hidden_size: i32,
    /// Number of attention heads.
    num_heads: i32,
    /// Dimension per attention head.
    head_dim: i32,
    /// Model variant.
    variant: ModelVariant,
    /// Maximum sequence length supported by the model.
    max_position_embeddings: i32,
    /// Memory-mapped safetensors file backing the Metal buffer.
    /// Must be declared AFTER `weight_buffer` to ensure correct drop order
    /// (buffer dropped before mmap).
    _mmap: memmap2::Mmap,
}

// SAFETY: Metal device, command queue, and shared-mode buffers are thread-safe
// (Apple documents MTLDevice and MTLCommandQueue as safe to use from multiple
// threads, and StorageModeShared buffers are readable from any thread).
#[expect(
    unsafe_code,
    reason = "Metal device/queue/shared buffers are documented as thread-safe"
)]
unsafe impl Send for MetalBackend {}
// SAFETY: Same rationale -- Metal's device, queue, and shared buffers are thread-safe.
#[expect(
    unsafe_code,
    reason = "Metal device/queue/shared buffers are documented as thread-safe"
)]
unsafe impl Sync for MetalBackend {}

impl MetalBackend {
    /// Load a BERT embedding model onto Metal via zero-copy mmap.
    ///
    /// Downloads `model.safetensors` and `config.json` via `hf-hub`, then
    /// memory-maps the safetensors file and wraps it as a single Metal buffer
    /// using `newBufferWithBytesNoCopy`. Weight tensors are addressed via
    /// [`WeightRef`] byte offsets. Pre-allocates workspace buffers for the
    /// maximum batch size and builds `RoPE` tables if needed.
    ///
    /// # Errors
    ///
    /// Returns an error if no Metal device is available, the model cannot
    /// be downloaded, or weight loading fails.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "config values are small ints that fit in i32"
    )]
    pub fn load(model_repo: &str, _device_hint: &DeviceHint) -> crate::Result<Self> {
        let device = create_device()?;
        let queue = create_queue(&device)?;
        let chip_family = ChipFamily::detect(&device);

        // Download model files via hf-hub
        let api = Api::new().map_err(|e| crate::Error::Download(e.to_string()))?;
        let repo = api.model(model_repo.to_string());

        let config_path = repo
            .get("config.json")
            .map_err(|e| crate::Error::Download(e.to_string()))?;
        let weights_path = repo
            .get("model.safetensors")
            .map_err(|e| crate::Error::Download(e.to_string()))?;

        // Parse config.json
        let config_str = std::fs::read_to_string(&config_path).map_err(|e| crate::Error::Io {
            path: config_path.display().to_string(),
            source: e,
        })?;
        let config_json: serde_json::Value = serde_json::from_str(&config_str)
            .map_err(|e| crate::Error::Other(anyhow::anyhow!("config parse error: {e}")))?;

        // Mmap safetensors and create zero-copy Metal buffer
        let (mmap, weight_buffer) = load_weights_zero_copy(&device, &weights_path)?;

        // Extract weight refs from the mmap'd data
        let (refs, _data_start) = extract_weight_refs(&mmap)?;

        // Detect variant from weight names
        let tensors = SafeTensors::deserialize(&mmap)
            .map_err(|e| crate::Error::Metal(format!("safetensors parse: {e}")))?;
        let variant = detect_variant(&tensors);
        drop(tensors); // done with the parsed header

        let config = BertConfig::from_json(&config_json, variant)?;
        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let head_dim = hidden_size / num_heads;
        let max_position_embeddings = config.max_position_embeddings;

        // Build model from weight refs
        let model = MetalBertModel::from_weight_refs(&refs, &config)?;

        // Determine intermediate dimension from first layer's weight shape
        let inter_dim = model.layers[0].ffn.intermediate_weight.shape[0] as i32;

        // Compile all MSL kernels into pipeline states
        let kernels = KernelPipelines::compile(&device)?;

        // Allocate workspace buffers
        let workspace = allocate_workspace(&device, &config, inter_dim)?;

        // Build RoPE cache for NomicBert
        let rope_cache = if variant == ModelVariant::NomicBert {
            Some(build_rope_cache(&device, &config)?)
        } else {
            None
        };

        tracing::info!(
            device = %device.name(),
            chip_family = ?chip_family,
            hidden_size,
            num_heads,
            head_dim,
            layers = config.num_hidden_layers,
            variant = ?variant,
            weights_bytes = mmap.len(),
            "Metal backend initialized with zero-copy weights + 20 MSL kernels"
        );

        Ok(Self {
            device,
            queue,
            chip_family,
            weight_buffer,
            model,
            kernels,
            workspace,
            rope_cache,
            hidden_size,
            num_heads,
            head_dim,
            variant,
            max_position_embeddings,
            _mmap: mmap,
        })
    }

    /// Encode the attention sub-layer for one transformer layer.
    ///
    /// Reads from `input` buffer, writes result to `output` buffer.
    /// Uses ping-pong workspace buffers for intermediate results.
    ///
    /// Dispatches: QKV projection -> bias -> QKV split/reshape -> `RoPE` ->
    /// per-head attention scores -> scale+mask+softmax -> per-head output ->
    /// reshape -> output projection -> bias+residual+layernorm.
    #[expect(
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        clippy::too_many_arguments,
        clippy::too_many_lines,
        clippy::cast_precision_loss,
        clippy::similar_names,
        reason = "monolithic GPU attention dispatch requires many args, lines, and integer casts"
    )]
    fn encode_attention(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        layer: &MetalBertLayer,
        input: &ProtocolObject<dyn MTLBuffer>,
        output: &ProtocolObject<dyn MTLBuffer>,
        mask: &ProtocolObject<dyn MTLBuffer>,
        batch: i32,
        max_seq: i32,
    ) {
        let hd = self.hidden_size;
        let nh = self.num_heads;
        let head_dim = self.head_dim;
        let batch_seq = batch * max_seq;
        let batch_heads = batch * nh;
        let attn = &layer.attention;

        // Padded dimensions for GEMM (must be multiple of 8 for simdgroup tiles)
        let pad_hd = (hd as usize).next_multiple_of(8) as u32;
        let pad_3hd = (3 * hd as usize).next_multiple_of(8) as u32;

        // --- QKV projection ---
        match self.variant {
            ModelVariant::NomicBert => {
                // Fused QKV: one GEMM [batch*seq, hidden] @ [3*hidden, hidden]^T
                dispatch_gemm(
                    encoder,
                    &self.kernels.gemm,
                    input,
                    0,
                    &self.weight_buffer,
                    attn.qkv_weight.offset,
                    &self.workspace.qkv,
                    0,
                    batch_seq as u32,
                    pad_3hd,
                    pad_hd,
                    true,
                );
            }
            ModelVariant::ClassicBert => {
                // Separate Q/K/V GEMMs into contiguous qkv buffer
                // Q -> qkv[..., 0..hidden]
                // K -> qkv[..., hidden..2*hidden]
                // V -> qkv[..., 2*hidden..3*hidden]
                // But GEMM writes at stride N, so we write to separate workspace
                // buffers and then use qkv_split with head_reshape.

                // Q: input @ Q_weight^T -> q workspace
                dispatch_gemm(
                    encoder,
                    &self.kernels.gemm,
                    input,
                    0,
                    &self.weight_buffer,
                    attn.qkv_weight.offset,
                    &self.workspace.q,
                    0,
                    batch_seq as u32,
                    pad_hd,
                    pad_hd,
                    true,
                );

                // K: input @ K_weight^T -> k workspace
                if let Some(ref k_weight) = attn.k_weight {
                    dispatch_gemm(
                        encoder,
                        &self.kernels.gemm,
                        input,
                        0,
                        &self.weight_buffer,
                        k_weight.offset,
                        &self.workspace.k,
                        0,
                        batch_seq as u32,
                        pad_hd,
                        pad_hd,
                        true,
                    );
                }

                // V: input @ V_weight^T -> v workspace
                if let Some(ref v_weight) = attn.v_weight {
                    dispatch_gemm(
                        encoder,
                        &self.kernels.gemm,
                        input,
                        0,
                        &self.weight_buffer,
                        v_weight.offset,
                        &self.workspace.v,
                        0,
                        batch_seq as u32,
                        pad_hd,
                        pad_hd,
                        true,
                    );
                }

                // Add Q bias
                if let Some(ref bias) = attn.qkv_bias {
                    let total = batch_seq * hd;
                    encoder.setComputePipelineState(&self.kernels.add_bias);
                    set_buffer(encoder, &self.workspace.q, 0, 0);
                    set_buffer(encoder, &self.weight_buffer, bias.offset, 1);
                    set_i32_param(encoder, batch_seq, 2);
                    set_i32_param(encoder, hd, 3);
                    dispatch_1d(encoder, &self.kernels.add_bias, total as usize);
                }

                // Add K bias
                if let Some(ref bias) = attn.k_bias {
                    let total = batch_seq * hd;
                    encoder.setComputePipelineState(&self.kernels.add_bias);
                    set_buffer(encoder, &self.workspace.k, 0, 0);
                    set_buffer(encoder, &self.weight_buffer, bias.offset, 1);
                    set_i32_param(encoder, batch_seq, 2);
                    set_i32_param(encoder, hd, 3);
                    dispatch_1d(encoder, &self.kernels.add_bias, total as usize);
                }

                // Add V bias
                if let Some(ref bias) = attn.v_bias {
                    let total = batch_seq * hd;
                    encoder.setComputePipelineState(&self.kernels.add_bias);
                    set_buffer(encoder, &self.workspace.v, 0, 0);
                    set_buffer(encoder, &self.weight_buffer, bias.offset, 1);
                    set_i32_param(encoder, batch_seq, 2);
                    set_i32_param(encoder, hd, 3);
                    dispatch_1d(encoder, &self.kernels.add_bias, total as usize);
                }

                // Head reshape Q: [batch*seq, hidden] -> [batch*nh, seq, head_dim]
                // Use scratch as temp, then copy back (q currently has the linear output)
                let total_head = batch_heads * max_seq * head_dim;
                encoder.setComputePipelineState(&self.kernels.head_reshape);
                set_buffer(encoder, &self.workspace.scratch, 0, 0);
                set_buffer(encoder, &self.workspace.q, 0, 1);
                set_i32_param(encoder, batch, 2);
                set_i32_param(encoder, max_seq, 3);
                set_i32_param(encoder, nh, 4);
                set_i32_param(encoder, head_dim, 5);
                dispatch_1d(encoder, &self.kernels.head_reshape, total_head as usize);

                // Copy scratch -> q (swap pointers not possible, so use attn_out as temp for k)
                // Actually we need all 3 reshaped. Use qkv buffer as scratch space.
                // scratch now has reshaped Q -> copy to scores (temp), reshape K to scratch, etc.
                // This is getting complex. Let me use a simpler approach: reshape in-place is
                // not possible, so use qkv buffer as a large scratch area.

                // Q is now in scratch, we need it in q. Use context as temp for K reshape.
                // Reshape Q: scratch -> qkv[0..total_head*4]  (reuse qkv buffer for Q)
                // Actually let's just swap: q has linear Q, scratch has reshaped Q.
                // We need reshaped Q in self.workspace.q. Copy scratch -> q.
                // But we can't do a simple copy with existing kernels.
                // Instead: reshape directly into workspace.attn_out (unused at this point),
                // then reshape K into workspace.scratch, reshape V into workspace.context.
                // Then do the attention GEMMs using attn_out(Q), scratch(K), context(V).
                // After attention is done, attn_out, scratch, context can be repurposed.

                // Reshape Q: q -> attn_out (as temp Q storage)
                encoder.setComputePipelineState(&self.kernels.head_reshape);
                set_buffer(encoder, &self.workspace.attn_out, 0, 0);
                set_buffer(encoder, &self.workspace.q, 0, 1);
                set_i32_param(encoder, batch, 2);
                set_i32_param(encoder, max_seq, 3);
                set_i32_param(encoder, nh, 4);
                set_i32_param(encoder, head_dim, 5);
                dispatch_1d(encoder, &self.kernels.head_reshape, total_head as usize);

                // Reshape K: k -> q (reuse q buffer for reshaped K)
                encoder.setComputePipelineState(&self.kernels.head_reshape);
                set_buffer(encoder, &self.workspace.q, 0, 0);
                set_buffer(encoder, &self.workspace.k, 0, 1);
                set_i32_param(encoder, batch, 2);
                set_i32_param(encoder, max_seq, 3);
                set_i32_param(encoder, nh, 4);
                set_i32_param(encoder, head_dim, 5);
                dispatch_1d(encoder, &self.kernels.head_reshape, total_head as usize);

                // Reshape V: v -> k (reuse k buffer for reshaped V)
                encoder.setComputePipelineState(&self.kernels.head_reshape);
                set_buffer(encoder, &self.workspace.k, 0, 0);
                set_buffer(encoder, &self.workspace.v, 0, 1);
                set_i32_param(encoder, batch, 2);
                set_i32_param(encoder, max_seq, 3);
                set_i32_param(encoder, nh, 4);
                set_i32_param(encoder, head_dim, 5);
                dispatch_1d(encoder, &self.kernels.head_reshape, total_head as usize);

                // Now: attn_out = reshaped Q, q = reshaped K, k = reshaped V
                // We need Q in q, K in k, V in v for the attention step.
                // Actually, let's just use these buffers directly in the attention GEMMs
                // with the right variable names. We'll reference attn_out as Q_buf,
                // q as K_buf, k as V_buf below.

                // Jump to the attention scores computation below -- but since we're
                // inside the ClassicBert match arm and need to unify with NomicBert,
                // we handle this after the match block by documenting the buffer mapping.
                // For ClassicBert after this block:
                //   Q is in attn_out, K is in q, V is in k
            }
        }

        // --- QKV split (NomicBert) or already done (ClassicBert) ---
        // For NomicBert: split qkv [batch*seq, 3*hidden] -> Q,K,V [batch*nh, seq, head_dim]
        if self.variant == ModelVariant::NomicBert {
            // Add QKV bias if present
            if let Some(ref bias) = attn.qkv_bias {
                let total_qkv = batch_seq * 3 * hd;
                encoder.setComputePipelineState(&self.kernels.add_bias);
                set_buffer(encoder, &self.workspace.qkv, 0, 0);
                set_buffer(encoder, &self.weight_buffer, bias.offset, 1);
                set_i32_param(encoder, batch_seq, 2);
                set_i32_param(encoder, 3 * hd, 3);
                dispatch_1d(encoder, &self.kernels.add_bias, total_qkv as usize);
            }

            let total_head = batch_heads * max_seq * head_dim;
            encoder.setComputePipelineState(&self.kernels.qkv_split);
            set_buffer(encoder, &self.workspace.q, 0, 0);
            set_buffer(encoder, &self.workspace.k, 0, 1);
            set_buffer(encoder, &self.workspace.v, 0, 2);
            set_buffer(encoder, &self.workspace.qkv, 0, 3);
            set_i32_param(encoder, batch, 4);
            set_i32_param(encoder, max_seq, 5);
            set_i32_param(encoder, hd, 6);
            set_i32_param(encoder, nh, 7);
            set_i32_param(encoder, head_dim, 8);
            dispatch_1d(encoder, &self.kernels.qkv_split, total_head as usize);
        }

        // Determine which buffers hold Q, K, V after projection + reshape
        let (q_buf, k_buf, v_buf) = match self.variant {
            ModelVariant::NomicBert => (&*self.workspace.q, &*self.workspace.k, &*self.workspace.v),
            ModelVariant::ClassicBert => (
                // After head_reshape: Q in attn_out, K in q, V in k
                &*self.workspace.attn_out,
                &*self.workspace.q,
                &*self.workspace.k,
            ),
        };

        // --- RoPE (NomicBert only) ---
        if let Some(ref rope) = self.rope_cache
            && attn.rotary_emb_base.is_some()
        {
            let half = head_dim / 2;
            let num_rows = batch_heads * max_seq;
            let total_rope = num_rows * half;

            // RoPE on Q
            encoder.setComputePipelineState(&self.kernels.rope_cached);
            set_buffer(encoder, q_buf, 0, 0);
            set_buffer(encoder, &rope.cos, 0, 1);
            set_buffer(encoder, &rope.sin, 0, 2);
            set_i32_param(encoder, num_rows, 3);
            set_i32_param(encoder, max_seq, 4);
            set_i32_param(encoder, head_dim, 5);
            set_i32_param(encoder, nh, 6);
            dispatch_1d(encoder, &self.kernels.rope_cached, total_rope as usize);

            // RoPE on K
            encoder.setComputePipelineState(&self.kernels.rope_cached);
            set_buffer(encoder, k_buf, 0, 0);
            set_buffer(encoder, &rope.cos, 0, 1);
            set_buffer(encoder, &rope.sin, 0, 2);
            set_i32_param(encoder, num_rows, 3);
            set_i32_param(encoder, max_seq, 4);
            set_i32_param(encoder, head_dim, 5);
            set_i32_param(encoder, nh, 6);
            dispatch_1d(encoder, &self.kernels.rope_cached, total_rope as usize);
        }

        // --- Attention scores: Q @ K^T (all heads in one batched dispatch) ---
        // Q, K are [batch*nh, seq, head_dim]. Scores = [batch*nh, seq, seq].
        let stride_qk = (max_seq * head_dim) as u32; // elements per head in Q/K
        let stride_scores = (max_seq * max_seq) as u32; // elements per head in scores

        dispatch_gemm_batched(
            encoder,
            &self.kernels.gemm_batched,
            q_buf,
            0,
            k_buf,
            0,
            &self.workspace.scores,
            0,
            max_seq as u32,  // M = seq
            max_seq as u32,  // N = seq
            head_dim as u32, // K = head_dim
            true,            // trans_b
            stride_qk,
            stride_qk,
            stride_scores,
            batch_heads as u32,
        );

        // --- Fused scale + mask + softmax ---
        {
            let scale = 1.0_f32 / (head_dim as f32).sqrt();
            let total_rows = batch_heads * max_seq;
            let threads = 256.min(max_seq as usize);
            encoder.setComputePipelineState(&self.kernels.fused_scale_mask_softmax);
            set_buffer(encoder, &self.workspace.scores, 0, 0);
            set_buffer(encoder, mask, 0, 1);
            set_i32_param(encoder, batch, 2);
            set_i32_param(encoder, nh, 3);
            set_i32_param(encoder, max_seq, 4);
            set_f32_param(encoder, scale, 5);
            dispatch_rows(
                encoder,
                &self.kernels.fused_scale_mask_softmax,
                total_rows as usize,
                threads,
            );
        }

        // --- Attention output: scores @ V per head ---
        // scores is [batch*nh, seq, seq], V is [batch*nh, seq, head_dim]
        // output is [batch*nh, seq, head_dim]
        //
        // For ClassicBert, V is in k_buf (workspace.k). We write output to
        // workspace.v (which is free since ClassicBert V was in workspace.k).
        // For NomicBert, V is in workspace.v. We write to workspace.attn_out.
        let attn_out_buf = match self.variant {
            ModelVariant::NomicBert => &*self.workspace.attn_out,
            // For ClassicBert: attn_out held Q, but Q is consumed. Use attn_out.
            // Wait -- attn_out has reshaped Q which was used for scores. Now scores
            // are computed, so attn_out is free to be overwritten.
            ModelVariant::ClassicBert => &*self.workspace.v,
        };

        // --- Attention output: scores @ V (all heads in one batched dispatch) ---
        dispatch_gemm_batched(
            encoder,
            &self.kernels.gemm_batched,
            &self.workspace.scores,
            0,
            v_buf,
            0,
            attn_out_buf,
            0,
            max_seq as u32,  // M = seq
            head_dim as u32, // N = head_dim
            max_seq as u32,  // K = seq
            false,           // no transpose
            stride_scores,
            stride_qk,
            stride_qk,
            batch_heads as u32,
        );

        // --- Reshape: [batch*nh, seq, head_dim] -> [batch*seq, hidden] ---
        {
            let total_ctx = batch_seq * hd;
            encoder.setComputePipelineState(&self.kernels.attn_reshape);
            set_buffer(encoder, &self.workspace.context, 0, 0);
            set_buffer(encoder, attn_out_buf, 0, 1);
            set_i32_param(encoder, batch, 2);
            set_i32_param(encoder, max_seq, 3);
            set_i32_param(encoder, nh, 4);
            set_i32_param(encoder, head_dim, 5);
            dispatch_1d(encoder, &self.kernels.attn_reshape, total_ctx as usize);
        }

        // --- Output projection: context @ output_weight^T -> projected ---
        dispatch_gemm(
            encoder,
            &self.kernels.gemm,
            &self.workspace.context,
            0,
            &self.weight_buffer,
            attn.output_weight.offset,
            &self.workspace.projected,
            0,
            batch_seq as u32,
            pad_hd,
            pad_hd,
            true,
        );

        // --- Bias + residual + LayerNorm ---
        if let Some(ref bias) = attn.output_bias {
            // Fused bias + residual: scratch = projected + bias + input
            let total_proj = batch_seq * hd;
            encoder.setComputePipelineState(&self.kernels.fused_bias_residual);
            set_buffer(encoder, &self.workspace.scratch, 0, 0);
            set_buffer(encoder, &self.workspace.projected, 0, 1);
            set_buffer(encoder, &self.weight_buffer, bias.offset, 2);
            set_buffer(encoder, input, 0, 3);
            set_i32_param(encoder, batch_seq, 4);
            set_i32_param(encoder, hd, 5);
            dispatch_1d(
                encoder,
                &self.kernels.fused_bias_residual,
                total_proj as usize,
            );

            // Layer norm: scratch -> output
            let eps = attn.layer_norm_eps;
            let threads = 256.min(hd as usize);
            encoder.setComputePipelineState(&self.kernels.layer_norm);
            set_buffer(encoder, output, 0, 0);
            set_buffer(encoder, &self.workspace.scratch, 0, 1);
            set_buffer(
                encoder,
                &self.weight_buffer,
                attn.output_ln_weight.offset,
                2,
            );
            set_buffer(encoder, &self.weight_buffer, attn.output_ln_bias.offset, 3);
            set_i32_param(encoder, batch_seq, 4);
            set_i32_param(encoder, hd, 5);
            set_f32_param(encoder, eps, 6);
            dispatch_rows(
                encoder,
                &self.kernels.layer_norm,
                batch_seq as usize,
                threads,
            );
        } else {
            // Fused residual + layernorm: output = layernorm(projected + input)
            let eps = attn.layer_norm_eps;
            let threads = 256.min(hd as usize);
            encoder.setComputePipelineState(&self.kernels.fused_residual_layernorm);
            set_buffer(encoder, output, 0, 0);
            set_buffer(encoder, &self.workspace.projected, 0, 1);
            set_buffer(encoder, input, 0, 2);
            set_buffer(
                encoder,
                &self.weight_buffer,
                attn.output_ln_weight.offset,
                3,
            );
            set_buffer(encoder, &self.weight_buffer, attn.output_ln_bias.offset, 4);
            set_i32_param(encoder, batch_seq, 5);
            set_i32_param(encoder, hd, 6);
            set_f32_param(encoder, eps, 7);
            dispatch_rows(
                encoder,
                &self.kernels.fused_residual_layernorm,
                batch_seq as usize,
                threads,
            );
        }
    }

    /// Encode the feed-forward sub-layer for one transformer layer.
    ///
    /// Reads from `input` buffer, writes result to `output` buffer.
    ///
    /// Dispatches: intermediate projection -> activation (GELU or `SwiGLU`) ->
    /// output projection -> bias+residual+layernorm.
    #[expect(
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::too_many_lines,
        reason = "monolithic GPU FFN dispatch requires many lines and integer casts"
    )]
    fn encode_ffn(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        ffn: &MetalBertFfn,
        input: &ProtocolObject<dyn MTLBuffer>,
        output: &ProtocolObject<dyn MTLBuffer>,
        batch_seq: i32,
    ) {
        let hd = self.hidden_size;
        let pad_hd = (hd as usize).next_multiple_of(8) as u32;
        let inter_dim = ffn.intermediate_weight.shape[0] as i32;

        match self.variant {
            ModelVariant::ClassicBert => {
                let pad_inter = (inter_dim as usize).next_multiple_of(8) as u32;

                // Intermediate projection: input @ intermediate_weight^T -> ffn_inter
                dispatch_gemm(
                    encoder,
                    &self.kernels.gemm,
                    input,
                    0,
                    &self.weight_buffer,
                    ffn.intermediate_weight.offset,
                    &self.workspace.ffn_inter,
                    0,
                    batch_seq as u32,
                    pad_inter,
                    pad_hd,
                    true,
                );

                // Fused bias + GELU or just GELU
                let total_act = batch_seq * inter_dim;
                if let Some(ref bias) = ffn.intermediate_bias {
                    encoder.setComputePipelineState(&self.kernels.fused_bias_gelu);
                    set_buffer(encoder, &self.workspace.ffn_inter, 0, 0);
                    set_buffer(encoder, &self.weight_buffer, bias.offset, 1);
                    set_i32_param(encoder, batch_seq, 2);
                    set_i32_param(encoder, inter_dim, 3);
                    dispatch_1d(encoder, &self.kernels.fused_bias_gelu, total_act as usize);
                } else {
                    encoder.setComputePipelineState(&self.kernels.gelu);
                    set_buffer(encoder, &self.workspace.ffn_inter, 0, 0);
                    set_i32_param(encoder, total_act, 1);
                    dispatch_1d(encoder, &self.kernels.gelu, total_act as usize);
                }

                // Output projection: ffn_inter @ output_weight^T -> scratch
                dispatch_gemm(
                    encoder,
                    &self.kernels.gemm,
                    &self.workspace.ffn_inter,
                    0,
                    &self.weight_buffer,
                    ffn.output_weight.offset,
                    &self.workspace.scratch,
                    0,
                    batch_seq as u32,
                    pad_hd,
                    pad_inter,
                    true,
                );
            }
            ModelVariant::NomicBert => {
                // SwiGLU with separate fc11 (value) and fc12 (gate) weights.
                // Zero-copy prevents concatenating into one [2*inter_dim, hidden]
                // weight, so we dispatch two GEMMs to separate buffers and use
                // `swiglu_two_input_kernel` which takes separate value/gate inputs.
                let pad_inter = (inter_dim as usize).next_multiple_of(8) as u32;

                // fc11 (value): input @ fc11_weight^T -> ffn_inter
                // fc12 (gate):  input @ fc12_weight^T -> activated
                // Then swiglu_two_input_kernel combines them element-wise.

                // fc11 (value): input @ fc11_weight^T -> ffn_inter
                dispatch_gemm(
                    encoder,
                    &self.kernels.gemm,
                    input,
                    0,
                    &self.weight_buffer,
                    ffn.intermediate_weight.offset,
                    &self.workspace.ffn_inter,
                    0,
                    batch_seq as u32,
                    pad_inter,
                    pad_hd,
                    true,
                );

                // fc12 (gate): input @ gate_weight^T -> activated
                if let Some(ref gate) = ffn.gate_weight {
                    dispatch_gemm(
                        encoder,
                        &self.kernels.gemm,
                        input,
                        0,
                        &self.weight_buffer,
                        gate.offset,
                        &self.workspace.activated,
                        0,
                        batch_seq as u32,
                        pad_inter,
                        pad_hd,
                        true,
                    );
                }

                // Two-input SwiGLU: activated = ffn_inter * silu(activated)
                let total_act = batch_seq * inter_dim;
                encoder.setComputePipelineState(&self.kernels.swiglu_two_input);
                set_buffer(encoder, &self.workspace.activated, 0, 0);
                set_buffer(encoder, &self.workspace.ffn_inter, 0, 1);
                set_buffer(encoder, &self.workspace.activated, 0, 2);
                set_i32_param(encoder, total_act, 3);
                dispatch_1d(encoder, &self.kernels.swiglu_two_input, total_act as usize);

                // Output projection: activated @ output_weight^T -> scratch
                dispatch_gemm(
                    encoder,
                    &self.kernels.gemm,
                    &self.workspace.activated,
                    0,
                    &self.weight_buffer,
                    ffn.output_weight.offset,
                    &self.workspace.scratch,
                    0,
                    batch_seq as u32,
                    pad_hd,
                    pad_inter,
                    true,
                );
            }
        }

        // --- Bias + residual + LayerNorm ---
        if let Some(ref bias) = ffn.output_bias {
            let total_out = batch_seq * hd;
            encoder.setComputePipelineState(&self.kernels.fused_bias_residual);
            set_buffer(encoder, &self.workspace.projected, 0, 0);
            set_buffer(encoder, &self.workspace.scratch, 0, 1);
            set_buffer(encoder, &self.weight_buffer, bias.offset, 2);
            set_buffer(encoder, input, 0, 3);
            set_i32_param(encoder, batch_seq, 4);
            set_i32_param(encoder, hd, 5);
            dispatch_1d(
                encoder,
                &self.kernels.fused_bias_residual,
                total_out as usize,
            );

            let eps = ffn.layer_norm_eps;
            let threads = 256.min(hd as usize);
            encoder.setComputePipelineState(&self.kernels.layer_norm);
            set_buffer(encoder, output, 0, 0);
            set_buffer(encoder, &self.workspace.projected, 0, 1);
            set_buffer(encoder, &self.weight_buffer, ffn.output_ln_weight.offset, 2);
            set_buffer(encoder, &self.weight_buffer, ffn.output_ln_bias.offset, 3);
            set_i32_param(encoder, batch_seq, 4);
            set_i32_param(encoder, hd, 5);
            set_f32_param(encoder, eps, 6);
            dispatch_rows(
                encoder,
                &self.kernels.layer_norm,
                batch_seq as usize,
                threads,
            );
        } else {
            let eps = ffn.layer_norm_eps;
            let threads = 256.min(hd as usize);
            encoder.setComputePipelineState(&self.kernels.fused_residual_layernorm);
            set_buffer(encoder, output, 0, 0);
            set_buffer(encoder, &self.workspace.scratch, 0, 1);
            set_buffer(encoder, input, 0, 2);
            set_buffer(encoder, &self.weight_buffer, ffn.output_ln_weight.offset, 3);
            set_buffer(encoder, &self.weight_buffer, ffn.output_ln_bias.offset, 4);
            set_i32_param(encoder, batch_seq, 5);
            set_i32_param(encoder, hd, 6);
            set_f32_param(encoder, eps, 7);
            dispatch_rows(
                encoder,
                &self.kernels.fused_residual_layernorm,
                batch_seq as usize,
                threads,
            );
        }
    }

    /// Full batched BERT forward pass on Metal.
    ///
    /// Pads all encodings to `max_seq_len` in the batch, transfers padded
    /// tensors to GPU, runs embeddings + transformer layers (ping-pong
    /// `hidden_a`/`hidden_b`) + CLS pooling + L2 normalize.
    ///
    /// Uses a ping-pong scheme:
    /// - Embeddings write to `hidden_a`.
    /// - Attention reads `hidden_a`, writes to `hidden_b`.
    /// - FFN reads `hidden_b`, writes back to `hidden_a`.
    /// - Next layer repeats.
    #[expect(
        unsafe_code,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        clippy::too_many_lines,
        reason = "Metal forward pass requires unsafe FFI, integer casts, and monolithic dispatch"
    )]
    fn forward_batch(&self, encodings: &[Encoding]) -> crate::Result<Vec<Vec<f32>>> {
        use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder};

        let batch = encodings.len() as i32;
        let hd = self.hidden_size;

        // Find max sequence length in this batch, rounded up to nearest
        // multiple of 8. The simdgroup GEMM kernel operates on 8×8 tiles;
        // padding avoids partial-tile edge effects in the per-head attention
        // GEMMs where the head stride puts adjacent heads' data at
        // non-tile-aligned boundaries. The attention mask zeros out padded
        // positions so they don't affect the result.
        let max_seq_raw = encodings
            .iter()
            .map(|e| e.input_ids.len())
            .max()
            .unwrap_or(0);
        let max_seq = max_seq_raw.next_multiple_of(8) as i32;
        let batch_seq = batch * max_seq;

        // Build padded input tensors on CPU: [batch, max_seq]
        let total = batch_seq as usize;
        let mut input_ids = vec![0_i32; total];
        let mut token_type_ids = vec![0_i32; total];
        let mut position_ids = vec![0_i32; total];
        let mut attn_mask_int = vec![0_i32; total];

        for (b, enc) in encodings.iter().enumerate() {
            let seq_len = enc.input_ids.len();
            let offset = b * max_seq as usize;
            for (i, &id) in enc.input_ids.iter().enumerate() {
                input_ids[offset + i] = id as i32;
            }
            for (i, &id) in enc.token_type_ids.iter().enumerate() {
                token_type_ids[offset + i] = id as i32;
            }
            for i in 0..seq_len {
                position_ids[offset + i] = i as i32;
            }
            for (i, &m) in enc.attention_mask.iter().enumerate() {
                attn_mask_int[offset + i] = m as i32;
            }
        }

        // Create GPU buffers for input tensors (shared memory -- no explicit copy)
        let make_i32_buf =
            |data: &[i32]| -> crate::Result<Retained<ProtocolObject<dyn MTLBuffer>>> {
                let size = core::mem::size_of_val(data) as NSUInteger;
                unsafe {
                    self.device.newBufferWithBytes_length_options(
                        std::ptr::NonNull::new(data.as_ptr() as *mut _)
                            .ok_or_else(|| crate::Error::Metal("null input data".into()))?,
                        size,
                        MTLResourceOptions::StorageModeShared,
                    )
                }
                .ok_or_else(|| crate::Error::Metal("input buffer alloc failed".into()))
            };

        let input_ids_buf = make_i32_buf(&input_ids)?;
        let token_type_ids_buf = make_i32_buf(&token_type_ids)?;
        let position_ids_buf = make_i32_buf(&position_ids)?;
        let attn_mask_int_buf = make_i32_buf(&attn_mask_int)?;

        // Create command buffer and encoder
        let cmd_buf = self
            .queue
            .commandBuffer()
            .ok_or_else(|| crate::Error::Metal("failed to create command buffer".into()))?;
        let enc = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| crate::Error::Metal("failed to create compute encoder".into()))?;

        // --- Build float attention mask ---
        let mask_total = batch * max_seq;
        enc.setComputePipelineState(&self.kernels.build_attn_mask);
        set_buffer(&enc, &self.workspace.mask, 0, 0);
        set_buffer(&enc, &attn_mask_int_buf, 0, 1);
        set_i32_param(&enc, mask_total, 2);
        dispatch_1d(&enc, &self.kernels.build_attn_mask, mask_total as usize);

        // --- Embeddings: [batch*max_seq, hidden] -> hidden_a ---
        let n = batch_seq * hd;

        // Word embedding lookup -> scratch
        enc.setComputePipelineState(&self.kernels.embedding_lookup);
        set_buffer(&enc, &self.workspace.scratch, 0, 0);
        set_buffer(
            &enc,
            &self.weight_buffer,
            self.model.embeddings.word_embeddings.offset,
            1,
        );
        set_buffer(&enc, &input_ids_buf, 0, 2);
        set_i32_param(&enc, batch_seq, 3);
        set_i32_param(&enc, hd, 4);
        dispatch_1d(&enc, &self.kernels.embedding_lookup, n as usize);

        // Add position embeddings (ClassicBert only)
        if let Some(ref pos_emb) = self.model.embeddings.position_embeddings {
            enc.setComputePipelineState(&self.kernels.add_embeddings);
            set_buffer(&enc, &self.workspace.scratch, 0, 0);
            set_buffer(&enc, &self.weight_buffer, pos_emb.offset, 1);
            set_buffer(&enc, &position_ids_buf, 0, 2);
            set_i32_param(&enc, batch_seq, 3);
            set_i32_param(&enc, hd, 4);
            dispatch_1d(&enc, &self.kernels.add_embeddings, n as usize);
        }

        // Add token type embeddings
        if let Some(ref tok_emb) = self.model.embeddings.token_type_embeddings {
            enc.setComputePipelineState(&self.kernels.add_embeddings);
            set_buffer(&enc, &self.workspace.scratch, 0, 0);
            set_buffer(&enc, &self.weight_buffer, tok_emb.offset, 1);
            set_buffer(&enc, &token_type_ids_buf, 0, 2);
            set_i32_param(&enc, batch_seq, 3);
            set_i32_param(&enc, hd, 4);
            dispatch_1d(&enc, &self.kernels.add_embeddings, n as usize);
        }

        // Embedding layer norm: scratch -> hidden_a
        {
            let eps = self.model.embeddings.layer_norm_eps;
            let threads = 256.min(hd as usize);
            enc.setComputePipelineState(&self.kernels.layer_norm);
            set_buffer(&enc, &self.workspace.hidden_a, 0, 0);
            set_buffer(&enc, &self.workspace.scratch, 0, 1);
            set_buffer(
                &enc,
                &self.weight_buffer,
                self.model.embeddings.layer_norm_weight.offset,
                2,
            );
            set_buffer(
                &enc,
                &self.weight_buffer,
                self.model.embeddings.layer_norm_bias.offset,
                3,
            );
            set_i32_param(&enc, batch_seq, 4);
            set_i32_param(&enc, hd, 5);
            set_f32_param(&enc, eps, 6);
            dispatch_rows(&enc, &self.kernels.layer_norm, batch_seq as usize, threads);
        }

        // --- Transformer layers: ping-pong hidden_a <-> hidden_b ---
        for layer in &self.model.layers {
            // Attention: hidden_a -> hidden_b
            self.encode_attention(
                &enc,
                layer,
                &self.workspace.hidden_a,
                &self.workspace.hidden_b,
                &self.workspace.mask,
                batch,
                max_seq,
            );

            // FFN: hidden_b -> hidden_a
            self.encode_ffn(
                &enc,
                &layer.ffn,
                &self.workspace.hidden_b,
                &self.workspace.hidden_a,
                batch_seq,
            );
        }

        // --- CLS pooling + L2 normalize ---
        let cls_total = batch * hd;
        enc.setComputePipelineState(&self.kernels.cls_pool);
        set_buffer(&enc, &self.workspace.cls, 0, 0);
        set_buffer(&enc, &self.workspace.hidden_a, 0, 1);
        set_i32_param(&enc, batch, 2);
        set_i32_param(&enc, max_seq, 3);
        set_i32_param(&enc, hd, 4);
        dispatch_1d(&enc, &self.kernels.cls_pool, cls_total as usize);

        {
            let threads = 256.min(hd as usize);
            enc.setComputePipelineState(&self.kernels.l2_normalize);
            set_buffer(&enc, &self.workspace.cls, 0, 0);
            set_i32_param(&enc, batch, 1);
            set_i32_param(&enc, hd, 2);
            dispatch_rows(&enc, &self.kernels.l2_normalize, batch as usize, threads);
        }

        // End encoding and execute
        enc.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        // Read back CLS embeddings
        let flat_result = unsafe {
            core::slice::from_raw_parts(
                self.workspace.cls.contents().as_ptr() as *const f32,
                (batch * hd) as usize,
            )
        };

        // Split into per-encoding vectors
        let hd_usize = hd as usize;
        let mut results = Vec::with_capacity(batch as usize);
        for b in 0..batch as usize {
            results.push(flat_result[b * hd_usize..(b + 1) * hd_usize].to_vec());
        }

        Ok(results)
    }
}

impl EmbedBackend for MetalBackend {
    /// Embed a batch of pre-tokenized inputs using the full BERT forward pass on Metal.
    ///
    /// Batches larger than `MAX_BATCH` are processed in sub-batches.
    fn embed_batch(&self, encodings: &[Encoding]) -> crate::Result<Vec<Vec<f32>>> {
        if encodings.is_empty() {
            return Ok(Vec::new());
        }

        let max_batch = MAX_BATCH as usize;
        let mut all_results = Vec::with_capacity(encodings.len());

        for chunk in encodings.chunks(max_batch) {
            let mut results = self.forward_batch(chunk)?;
            all_results.append(&mut results);
        }

        Ok(all_results)
    }

    fn supports_clone(&self) -> bool {
        false
    }

    fn clone_backend(&self) -> Box<dyn EmbedBackend> {
        panic!("MetalBackend does not support cloning");
    }

    fn is_gpu(&self) -> bool {
        true
    }

    #[expect(
        clippy::cast_sign_loss,
        reason = "max_position_embeddings is always positive from config"
    )]
    fn max_tokens(&self) -> usize {
        self.max_position_embeddings as usize
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[expect(
    unsafe_code,
    reason = "Metal buffer creation/readback requires unsafe FFI"
)]
mod tests {
    use core::ffi::c_void;
    use core::ptr::NonNull;

    use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLSize};

    use super::*;

    #[test]
    fn metal_device_creation() {
        let device = create_device().unwrap();
        let _queue = create_queue(&device).unwrap();
        let name = device.name().to_string();
        assert!(!name.is_empty(), "device should have a name");
    }

    #[test]
    fn metal_chip_family_detection() {
        let device = create_device().unwrap();
        let family = ChipFamily::detect(&device);
        // On any Apple Silicon Mac, should be one of the two variants.
        assert!(
            family == ChipFamily::Apple7_8 || family == ChipFamily::Apple9,
            "unexpected chip family: {family:?}"
        );
    }

    #[test]
    fn metal_kernel_compilation() {
        let device = create_device().unwrap();
        let library = compile_library(&device, super::super::metal_kernels::TEST_KERNEL).unwrap();
        let pipeline = create_pipeline(&device, &library, "add_one").unwrap();
        assert!(
            pipeline.maxTotalThreadsPerThreadgroup() > 0,
            "pipeline should support at least 1 thread per threadgroup"
        );
    }

    #[test]
    fn metal_compute_dispatch() {
        let device = create_device().unwrap();
        let queue = create_queue(&device).unwrap();
        let library = compile_library(&device, super::super::metal_kernels::TEST_KERNEL).unwrap();
        let pipeline = create_pipeline(&device, &library, "add_one").unwrap();

        // Create a buffer with [1.0, 2.0, 3.0, 4.0]
        let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let buf_size = (data.len() * core::mem::size_of::<f32>()) as NSUInteger;
        let buffer = unsafe {
            device.newBufferWithBytes_length_options(
                NonNull::new(data.as_ptr() as *mut c_void).unwrap(),
                buf_size,
                MTLResourceOptions::StorageModeShared,
            )
        }
        .expect("failed to create Metal buffer");

        // Encode and dispatch the add_one kernel
        let cmd_buf = queue
            .commandBuffer()
            .expect("failed to create command buffer");
        let encoder = cmd_buf
            .computeCommandEncoder()
            .expect("failed to create compute encoder");
        encoder.setComputePipelineState(&pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&buffer), 0, 0);
        }

        let grid = MTLSize {
            width: 4,
            height: 1,
            depth: 1,
        };
        let threadgroup = MTLSize {
            width: 4,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreads_threadsPerThreadgroup(grid, threadgroup);
        encoder.endEncoding();

        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        // Read back results -- should be [2.0, 3.0, 4.0, 5.0]
        let result =
            unsafe { core::slice::from_raw_parts(buffer.contents().as_ptr() as *const f32, 4) };
        assert_eq!(result, &[2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    #[ignore = "requires model download"]
    fn metal_zero_copy_weight_loading() {
        let backend = MetalBackend::load("BAAI/bge-small-en-v1.5", &DeviceHint::Auto).unwrap();
        assert_eq!(backend.hidden_size, 384);
        assert_eq!(backend.max_tokens(), 512);
        assert!(backend.is_gpu());
        assert!(!backend.supports_clone());
    }

    #[test]
    fn metal_all_kernels_compile() {
        let device = create_device().unwrap();
        let pipelines = KernelPipelines::compile(&device).unwrap();
        // Verify all 18 pipelines have non-zero max threadgroup size
        assert!(pipelines.embedding_lookup.maxTotalThreadsPerThreadgroup() > 0);
        assert!(pipelines.add_embeddings.maxTotalThreadsPerThreadgroup() > 0);
        assert!(pipelines.layer_norm.maxTotalThreadsPerThreadgroup() > 0);
        assert!(pipelines.gelu.maxTotalThreadsPerThreadgroup() > 0);
        assert!(pipelines.swiglu.maxTotalThreadsPerThreadgroup() > 0);
        assert!(pipelines.rope_cached.maxTotalThreadsPerThreadgroup() > 0);
        assert!(
            pipelines
                .fused_scale_mask_softmax
                .maxTotalThreadsPerThreadgroup()
                > 0
        );
        assert!(
            pipelines
                .fused_residual_layernorm
                .maxTotalThreadsPerThreadgroup()
                > 0
        );
        assert!(pipelines.fused_bias_gelu.maxTotalThreadsPerThreadgroup() > 0);
        assert!(
            pipelines
                .fused_bias_residual
                .maxTotalThreadsPerThreadgroup()
                > 0
        );
        assert!(pipelines.fused_swiglu.maxTotalThreadsPerThreadgroup() > 0);
        assert!(pipelines.qkv_split.maxTotalThreadsPerThreadgroup() > 0);
        assert!(pipelines.attn_reshape.maxTotalThreadsPerThreadgroup() > 0);
        assert!(pipelines.cls_pool.maxTotalThreadsPerThreadgroup() > 0);
        assert!(pipelines.l2_normalize.maxTotalThreadsPerThreadgroup() > 0);
        assert!(pipelines.build_attn_mask.maxTotalThreadsPerThreadgroup() > 0);
        assert!(pipelines.f32_to_f16.maxTotalThreadsPerThreadgroup() > 0);
        assert!(pipelines.gemm.maxTotalThreadsPerThreadgroup() > 0);
    }

    /// Helper: create a Metal buffer from a slice of f32 values.
    fn make_f32_buffer(
        device: &ProtocolObject<dyn MTLDevice>,
        data: &[f32],
    ) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        let size = core::mem::size_of_val(data) as NSUInteger;
        unsafe {
            device.newBufferWithBytes_length_options(
                NonNull::new(data.as_ptr() as *mut c_void).unwrap(),
                size,
                MTLResourceOptions::StorageModeShared,
            )
        }
        .expect("failed to create Metal buffer")
    }

    /// Helper: create a Metal buffer from a slice of i32 values.
    fn make_i32_buffer(
        device: &ProtocolObject<dyn MTLDevice>,
        data: &[i32],
    ) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        let size = core::mem::size_of_val(data) as NSUInteger;
        unsafe {
            device.newBufferWithBytes_length_options(
                NonNull::new(data.as_ptr() as *mut c_void).unwrap(),
                size,
                MTLResourceOptions::StorageModeShared,
            )
        }
        .expect("failed to create Metal buffer")
    }

    /// Helper: create a zero-initialized Metal buffer of `n` f32 elements.
    fn make_zero_buffer(
        device: &ProtocolObject<dyn MTLDevice>,
        n: usize,
    ) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        let size = (n * core::mem::size_of::<f32>()) as NSUInteger;
        device
            .newBufferWithLength_options(size, MTLResourceOptions::StorageModeShared)
            .expect("failed to create Metal buffer")
    }

    /// Helper: create a Metal buffer containing a single i32 constant.
    fn make_i32_const_buffer(
        device: &ProtocolObject<dyn MTLDevice>,
        val: i32,
    ) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        make_i32_buffer(device, &[val])
    }

    /// Helper: create a Metal buffer containing a single f32 constant.
    fn make_f32_const_buffer(
        device: &ProtocolObject<dyn MTLDevice>,
        val: f32,
    ) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        make_f32_buffer(device, &[val])
    }

    /// Helper: read back f32 values from a Metal buffer.
    fn read_f32_buffer(buffer: &ProtocolObject<dyn MTLBuffer>, n: usize) -> Vec<f32> {
        unsafe { core::slice::from_raw_parts(buffer.contents().as_ptr() as *const f32, n) }.to_vec()
    }

    #[test]
    fn metal_layer_norm_kernel() {
        let device = create_device().unwrap();
        let queue = create_queue(&device).unwrap();
        let pipelines = KernelPipelines::compile(&device).unwrap();

        // Input: [1.0, 2.0, 3.0, 4.0] (1 row, 4 cols)
        // Weight: [1.0, 1.0, 1.0, 1.0], Bias: [0.0, 0.0, 0.0, 0.0], eps: 1e-5
        // Expected: standard layernorm of [1,2,3,4]
        // mean = 2.5, var = 1.25, std = sqrt(1.25) = 1.11803...
        // normalized = [-1.3416, -0.4472, 0.4472, 1.3416]
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let weight = [1.0_f32, 1.0, 1.0, 1.0];
        let bias = [0.0_f32, 0.0, 0.0, 0.0];

        let output_buf = make_zero_buffer(&device, 4);
        let input_buf = make_f32_buffer(&device, &input);
        let weight_buf = make_f32_buffer(&device, &weight);
        let bias_buf = make_f32_buffer(&device, &bias);
        let rows_buf = make_i32_const_buffer(&device, 1);
        let cols_buf = make_i32_const_buffer(&device, 4);
        let eps_buf = make_f32_const_buffer(&device, 1e-5);

        let cmd_buf = queue.commandBuffer().unwrap();
        let encoder = cmd_buf.computeCommandEncoder().unwrap();
        encoder.setComputePipelineState(&pipelines.layer_norm);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&output_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&input_buf), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&weight_buf), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&bias_buf), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&rows_buf), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(&cols_buf), 0, 5);
            encoder.setBuffer_offset_atIndex(Some(&eps_buf), 0, 6);
        }

        // One threadgroup for one row, with 4 threads
        let grid = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: 4,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
        encoder.endEncoding();

        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        let result = read_f32_buffer(&output_buf, 4);
        let expected = [-1.3416, -0.4472, 0.4472, 1.3416];
        for (got, exp) in result.iter().zip(expected.iter()) {
            assert!(
                (got - exp).abs() < 1e-3,
                "layer_norm mismatch: got {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn metal_gelu_kernel() {
        let device = create_device().unwrap();
        let queue = create_queue(&device).unwrap();
        let pipelines = KernelPipelines::compile(&device).unwrap();

        // GELU(0) = 0, GELU(1) ≈ 0.8412, GELU(-1) ≈ -0.1588
        let data = [0.0_f32, 1.0, -1.0, 2.0];
        let buf = make_f32_buffer(&device, &data);
        let n_buf = make_i32_const_buffer(&device, 4);

        let cmd_buf = queue.commandBuffer().unwrap();
        let encoder = cmd_buf.computeCommandEncoder().unwrap();
        encoder.setComputePipelineState(&pipelines.gelu);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&n_buf), 0, 1);
        }
        dispatch_1d(&encoder, &pipelines.gelu, 4);
        encoder.endEncoding();

        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        let result = read_f32_buffer(&buf, 4);
        let expected = [0.0, 0.8412, -0.1588, 1.9545];
        for (got, exp) in result.iter().zip(expected.iter()) {
            assert!(
                (got - exp).abs() < 1e-3,
                "GELU mismatch: got {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn metal_l2_normalize_kernel() {
        let device = create_device().unwrap();
        let queue = create_queue(&device).unwrap();
        let pipelines = KernelPipelines::compile(&device).unwrap();

        // [3.0, 4.0] -> L2 norm = 5.0 -> [0.6, 0.8]
        let data = [3.0_f32, 4.0];
        let buf = make_f32_buffer(&device, &data);
        let rows_buf = make_i32_const_buffer(&device, 1);
        let cols_buf = make_i32_const_buffer(&device, 2);

        let cmd_buf = queue.commandBuffer().unwrap();
        let encoder = cmd_buf.computeCommandEncoder().unwrap();
        encoder.setComputePipelineState(&pipelines.l2_normalize);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&rows_buf), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&cols_buf), 0, 2);
        }
        // dispatch_rows: 1 row, 2 threads per row
        let grid = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: 2,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
        encoder.endEncoding();

        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        let result = read_f32_buffer(&buf, 2);
        assert!(
            (result[0] - 0.6).abs() < 1e-5,
            "expected 0.6, got {}",
            result[0]
        );
        assert!(
            (result[1] - 0.8).abs() < 1e-5,
            "expected 0.8, got {}",
            result[1]
        );
    }

    #[test]
    fn metal_embedding_lookup_kernel() {
        let device = create_device().unwrap();
        let queue = create_queue(&device).unwrap();
        let pipelines = KernelPipelines::compile(&device).unwrap();

        // Embedding table: 3 tokens x 2 dims
        // [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]]
        let table = [10.0_f32, 11.0, 20.0, 21.0, 30.0, 31.0];
        let indices = [2_i32, 0]; // look up token 2 and token 0
        let hidden_dim = 2;
        let batch_seq = 2;

        let output_buf = make_zero_buffer(&device, 4);
        let table_buf = make_f32_buffer(&device, &table);
        let indices_buf = make_i32_buffer(&device, &indices);
        let batch_seq_buf = make_i32_const_buffer(&device, batch_seq);
        let hidden_dim_buf = make_i32_const_buffer(&device, hidden_dim);

        let cmd_buf = queue.commandBuffer().unwrap();
        let encoder = cmd_buf.computeCommandEncoder().unwrap();
        encoder.setComputePipelineState(&pipelines.embedding_lookup);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&output_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&table_buf), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&indices_buf), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&batch_seq_buf), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&hidden_dim_buf), 0, 4);
        }
        dispatch_1d(&encoder, &pipelines.embedding_lookup, 4);
        encoder.endEncoding();

        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        let result = read_f32_buffer(&output_buf, 4);
        // Token 2: [30.0, 31.0], Token 0: [10.0, 11.0]
        assert_eq!(result, vec![30.0, 31.0, 10.0, 11.0]);
    }

    #[test]
    fn metal_build_attn_mask_kernel() {
        let device = create_device().unwrap();
        let queue = create_queue(&device).unwrap();
        let pipelines = KernelPipelines::compile(&device).unwrap();

        let mask = [1_i32, 1, 0, 1];
        let output_buf = make_zero_buffer(&device, 4);
        let mask_buf = make_i32_buffer(&device, &mask);
        let total_buf = make_i32_const_buffer(&device, 4);

        let cmd_buf = queue.commandBuffer().unwrap();
        let encoder = cmd_buf.computeCommandEncoder().unwrap();
        encoder.setComputePipelineState(&pipelines.build_attn_mask);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&output_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&mask_buf), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&total_buf), 0, 2);
        }
        dispatch_1d(&encoder, &pipelines.build_attn_mask, 4);
        encoder.endEncoding();

        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        let result = read_f32_buffer(&output_buf, 4);
        assert!(
            (result[0] - 0.0_f32).abs() < f32::EPSILON,
            "expected 0.0, got {}",
            result[0]
        );
        assert!(
            (result[1] - 0.0_f32).abs() < f32::EPSILON,
            "expected 0.0, got {}",
            result[1]
        );
        assert!(
            (result[2] - (-1e9_f32)).abs() < 1.0,
            "expected -1e9, got {}",
            result[2]
        );
        assert!(
            (result[3] - 0.0_f32).abs() < f32::EPSILON,
            "expected 0.0, got {}",
            result[3]
        );
    }

    /// Helper: reference CPU matmul C = A * B (no transpose).
    fn cpu_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut c = vec![0.0_f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0_f32;
                for p in 0..k {
                    sum += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = sum;
            }
        }
        c
    }

    /// Helper: reference CPU matmul C = A * B^T.
    fn cpu_matmul_transb(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        // B is [N, K] row-major, so B^T[k,n] = B[n,k]
        let mut c = vec![0.0_f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0_f32;
                for p in 0..k {
                    sum += a[i * k + p] * b[j * k + p];
                }
                c[i * n + j] = sum;
            }
        }
        c
    }

    /// Helper: zero-pad a matrix to tile-aligned dimensions.
    ///
    /// Simdgroup loads read full 8x8 tiles even at matrix edges, so
    /// input buffers must be padded to avoid out-of-bounds reads.
    fn pad_matrix(
        data: &[f32],
        rows: usize,
        cols: usize,
        pad_rows: usize,
        pad_cols: usize,
    ) -> Vec<f32> {
        let mut padded = vec![0.0_f32; pad_rows * pad_cols];
        for i in 0..rows {
            for j in 0..cols {
                padded[i * pad_cols + j] = data[i * cols + j];
            }
        }
        padded
    }

    #[test]
    fn metal_gemm_small_no_transpose() {
        // A = [[1, 2], [3, 4], [5, 6]]  (3x2)
        // B = [[7, 8], [9, 10]]          (2x2)
        // C = A * B = [[25, 28], [57, 64], [89, 100]]  (3x2)
        let device = create_device().unwrap();
        let queue = create_queue(&device).unwrap();
        let gemm_lib = compile_library(&device, super::super::metal_kernels::GEMM_KERNEL).unwrap();
        let gemm_pipeline = create_pipeline(&device, &gemm_lib, "gemm_kernel").unwrap();

        let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data: Vec<f32> = vec![7.0, 8.0, 9.0, 10.0];

        let m: u32 = 3;
        let n: u32 = 2;
        let k: u32 = 2;

        // Pad all dimensions to 8-element boundaries so simdgroup 8x8
        // tile loads don't read out-of-bounds from input buffers.
        let pad_k = (k as usize).next_multiple_of(8);
        let pad_m = (m as usize).next_multiple_of(8);
        let pad_n = (n as usize).next_multiple_of(8);
        let a_padded = pad_matrix(&a_data, m as usize, k as usize, pad_m, pad_k);
        let b_padded = pad_matrix(&b_data, k as usize, n as usize, pad_k, pad_n);

        // Output buffer: pad M to tile boundary, use pad_n as stride
        let out_m = (m as usize).next_multiple_of(32);
        let buf_elems = out_m * pad_n;

        let a_buf = make_f32_buffer(&device, &a_padded);
        let b_buf = make_f32_buffer(&device, &b_padded);
        let c_buf = make_zero_buffer(&device, buf_elems);

        let cmd_buf = queue.commandBuffer().unwrap();
        let encoder = cmd_buf.computeCommandEncoder().unwrap();

        // Pass padded K and N as the kernel's strides
        dispatch_gemm(
            &encoder,
            &gemm_pipeline,
            &a_buf,
            0,
            &b_buf,
            0,
            &c_buf,
            0,
            m,
            pad_n as u32,
            pad_k as u32,
            false,
        );

        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        let result = read_f32_buffer(&c_buf, buf_elems);
        let expected = cpu_matmul(&a_data, &b_data, m as usize, n as usize, k as usize);

        for i in 0..m as usize {
            for j in 0..n as usize {
                let got = result[i * pad_n + j];
                let exp = expected[i * n as usize + j];
                assert!(
                    (got - exp).abs() < 1e-3,
                    "C[{i},{j}]: expected {exp}, got {got}"
                );
            }
        }
    }

    #[test]
    fn metal_gemm_small_transpose_b() {
        // A = [[1, 2, 3], [4, 5, 6]]     (2x3)
        // B = [[7, 8, 9], [10, 11, 12]]   (2x3) -- row-major, used as B^T
        // C = A * B^T = [[50, 68], [122, 167]]  (2x2)
        let device = create_device().unwrap();
        let queue = create_queue(&device).unwrap();
        let gemm_lib = compile_library(&device, super::super::metal_kernels::GEMM_KERNEL).unwrap();
        let gemm_pipeline = create_pipeline(&device, &gemm_lib, "gemm_kernel").unwrap();

        let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data: Vec<f32> = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];

        let m: u32 = 2;
        let n: u32 = 2;
        let k: u32 = 3;

        // Pad all dimensions to 8-element boundaries
        let pad_k = (k as usize).next_multiple_of(8);
        let pad_m = (m as usize).next_multiple_of(8);
        let pad_n = (n as usize).next_multiple_of(8);
        let a_padded = pad_matrix(&a_data, m as usize, k as usize, pad_m, pad_k);
        // B is [N, K] for transB mode
        let b_padded = pad_matrix(&b_data, n as usize, k as usize, pad_n, pad_k);

        let out_m = (m as usize).next_multiple_of(32);
        let buf_elems = out_m * pad_n;

        let a_buf = make_f32_buffer(&device, &a_padded);
        let b_buf = make_f32_buffer(&device, &b_padded);
        let c_buf = make_zero_buffer(&device, buf_elems);

        let cmd_buf = queue.commandBuffer().unwrap();
        let encoder = cmd_buf.computeCommandEncoder().unwrap();

        dispatch_gemm(
            &encoder,
            &gemm_pipeline,
            &a_buf,
            0,
            &b_buf,
            0,
            &c_buf,
            0,
            m,
            pad_n as u32,
            pad_k as u32,
            true,
        );

        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        let result = read_f32_buffer(&c_buf, buf_elems);
        let expected = cpu_matmul_transb(&a_data, &b_data, m as usize, n as usize, k as usize);

        for i in 0..m as usize {
            for j in 0..n as usize {
                let got = result[i * pad_n + j];
                let exp = expected[i * n as usize + j];
                assert!(
                    (got - exp).abs() < 1e-3,
                    "C[{i},{j}]: expected {exp}, got {got}"
                );
            }
        }
    }

    #[test]
    fn metal_gemm_bge_small_dims() {
        // Test with BGE-small dimensions: M=64, N=384, K=384, transB=true
        // This simulates a linear layer: output = input * weight^T
        let device = create_device().unwrap();
        let queue = create_queue(&device).unwrap();
        let gemm_lib = compile_library(&device, super::super::metal_kernels::GEMM_KERNEL).unwrap();
        let gemm_pipeline = create_pipeline(&device, &gemm_lib, "gemm_kernel").unwrap();

        let m: u32 = 64;
        let n: u32 = 384;
        let k: u32 = 384;

        // Generate deterministic test data (small values to avoid fp precision issues)
        let a_data: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();
        let b_data: Vec<f32> = (0..n * k).map(|i| ((i % 13) as f32 - 6.0) * 0.01).collect();

        // M and N are both multiples of 32 here, no padding needed
        let buf_elems = m as usize * n as usize;

        let a_buf = make_f32_buffer(&device, &a_data);
        let b_buf = make_f32_buffer(&device, &b_data);
        let c_buf = make_zero_buffer(&device, buf_elems);

        let cmd_buf = queue.commandBuffer().unwrap();
        let encoder = cmd_buf.computeCommandEncoder().unwrap();

        dispatch_gemm(
            &encoder,
            &gemm_pipeline,
            &a_buf,
            0,
            &b_buf,
            0,
            &c_buf,
            0,
            m,
            n,
            k,
            true,
        );

        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        let result = read_f32_buffer(&c_buf, buf_elems);
        let expected = cpu_matmul_transb(&a_data, &b_data, m as usize, n as usize, k as usize);

        // Check a sample of elements (full check would be slow)
        let spots = [(0, 0), (0, 383), (31, 192), (63, 0), (63, 383), (32, 128)];
        for (i, j) in spots {
            let got = result[i * n as usize + j];
            let exp = expected[i * n as usize + j];
            assert!(
                (got - exp).abs() < 1e-2,
                "C[{i},{j}]: expected {exp:.6}, got {got:.6}, diff={:.6}",
                (got - exp).abs()
            );
        }
    }

    // -----------------------------------------------------------------------
    // Stage-by-stage diagnostic: bisect the forward pass
    // -----------------------------------------------------------------------

    /// Helper: read f32 buffer and compute stats (min, max, mean, nonzero count)
    fn buffer_stats(buf: &ProtocolObject<dyn MTLBuffer>, n: usize) -> (f32, f32, f32, usize) {
        let data = read_f32_buffer(buf, n);
        let min = data.iter().copied().fold(f32::INFINITY, f32::min);
        let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = data.iter().sum();
        let mean = sum / n as f32;
        let nonzero = data.iter().filter(|&&v| v.abs() > 1e-10).count();
        (min, max, mean, nonzero)
    }

    /// Helper: create a new command buffer, encoder pair. Commits and waits on completion.
    fn run_encoder<F>(queue: &ProtocolObject<dyn MTLCommandQueue>, f: F)
    where
        F: FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>),
    {
        let cmd_buf = queue.commandBuffer().unwrap();
        let enc = cmd_buf.computeCommandEncoder().unwrap();
        f(&enc);
        enc.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
    }

    #[test]
    #[ignore = "requires model download; run with --nocapture to see diagnostics"]
    fn metal_forward_pass_stage_diagnostics() {
        let backend = MetalBackend::load("BAAI/bge-small-en-v1.5", &DeviceHint::Auto).unwrap();
        eprintln!(
            "Model loaded: hidden={}, heads={}, head_dim={}, layers={}, variant={:?}",
            backend.hidden_size,
            backend.num_heads,
            backend.head_dim,
            backend.model.layers.len(),
            backend.variant
        );

        // Test input: "Hello world" tokens (CLS, Hello, world, SEP)
        let enc = Encoding {
            input_ids: vec![101, 7592, 2088, 102],
            attention_mask: vec![1, 1, 1, 1],
            token_type_ids: vec![0, 0, 0, 0],
        };

        let hd = backend.hidden_size;
        let batch: i32 = 1;
        let max_seq: i32 = 4;
        let batch_seq = batch * max_seq;
        let n = (batch_seq * hd) as usize;

        // Build padded input tensors
        let input_ids: Vec<i32> = enc.input_ids.iter().map(|&x| x as i32).collect();
        let token_type_ids: Vec<i32> = enc.token_type_ids.iter().map(|&x| x as i32).collect();
        let position_ids: Vec<i32> = (0..max_seq).collect();
        let attn_mask_int: Vec<i32> = enc.attention_mask.iter().map(|&x| x as i32).collect();

        let input_ids_buf = make_i32_buffer(&backend.device, &input_ids);
        let token_type_ids_buf = make_i32_buffer(&backend.device, &token_type_ids);
        let position_ids_buf = make_i32_buffer(&backend.device, &position_ids);
        let attn_mask_int_buf = make_i32_buffer(&backend.device, &attn_mask_int);

        // ===== STAGE 1: Embedding lookup =====
        run_encoder(&backend.queue, |enc| {
            enc.setComputePipelineState(&backend.kernels.embedding_lookup);
            set_buffer(enc, &backend.workspace.scratch, 0, 0);
            set_buffer(
                enc,
                &backend.weight_buffer,
                backend.model.embeddings.word_embeddings.offset,
                1,
            );
            set_buffer(enc, &input_ids_buf, 0, 2);
            set_i32_param(enc, batch_seq, 3);
            set_i32_param(enc, hd, 4);
            dispatch_1d(enc, &backend.kernels.embedding_lookup, n);
        });

        let (min, max, mean, nz) = buffer_stats(&backend.workspace.scratch, n);
        eprintln!(
            "STAGE 1 - word embeddings:  min={min:.4}, max={max:.4}, mean={mean:.4}, nonzero={nz}/{n}"
        );
        assert!(nz > 0, "STAGE 1 FAILED: word embeddings are all zero!");

        // ===== STAGE 2: Add position + token_type embeddings =====
        if let Some(ref pos_emb) = backend.model.embeddings.position_embeddings {
            run_encoder(&backend.queue, |enc| {
                enc.setComputePipelineState(&backend.kernels.add_embeddings);
                set_buffer(enc, &backend.workspace.scratch, 0, 0);
                set_buffer(enc, &backend.weight_buffer, pos_emb.offset, 1);
                set_buffer(enc, &position_ids_buf, 0, 2);
                set_i32_param(enc, batch_seq, 3);
                set_i32_param(enc, hd, 4);
                dispatch_1d(enc, &backend.kernels.add_embeddings, n);
            });
            let (min, max, mean, nz) = buffer_stats(&backend.workspace.scratch, n);
            eprintln!(
                "STAGE 2a - +position emb:  min={min:.4}, max={max:.4}, mean={mean:.4}, nonzero={nz}/{n}"
            );
        }

        if let Some(ref tok_emb) = backend.model.embeddings.token_type_embeddings {
            run_encoder(&backend.queue, |enc| {
                enc.setComputePipelineState(&backend.kernels.add_embeddings);
                set_buffer(enc, &backend.workspace.scratch, 0, 0);
                set_buffer(enc, &backend.weight_buffer, tok_emb.offset, 1);
                set_buffer(enc, &token_type_ids_buf, 0, 2);
                set_i32_param(enc, batch_seq, 3);
                set_i32_param(enc, hd, 4);
                dispatch_1d(enc, &backend.kernels.add_embeddings, n);
            });
            let (min, max, mean, nz) = buffer_stats(&backend.workspace.scratch, n);
            eprintln!(
                "STAGE 2b - +token_type:    min={min:.4}, max={max:.4}, mean={mean:.4}, nonzero={nz}/{n}"
            );
        }

        // ===== STAGE 3: Embedding LayerNorm =====
        {
            let eps = backend.model.embeddings.layer_norm_eps;
            let threads = 256.min(hd as usize);
            run_encoder(&backend.queue, |enc| {
                enc.setComputePipelineState(&backend.kernels.layer_norm);
                set_buffer(enc, &backend.workspace.hidden_a, 0, 0);
                set_buffer(enc, &backend.workspace.scratch, 0, 1);
                set_buffer(
                    enc,
                    &backend.weight_buffer,
                    backend.model.embeddings.layer_norm_weight.offset,
                    2,
                );
                set_buffer(
                    enc,
                    &backend.weight_buffer,
                    backend.model.embeddings.layer_norm_bias.offset,
                    3,
                );
                set_i32_param(enc, batch_seq, 4);
                set_i32_param(enc, hd, 5);
                set_f32_param(enc, eps, 6);
                dispatch_rows(
                    enc,
                    &backend.kernels.layer_norm,
                    batch_seq as usize,
                    threads,
                );
            });
            let (min, max, mean, nz) = buffer_stats(&backend.workspace.hidden_a, n);
            eprintln!(
                "STAGE 3 - embedding LN:    min={min:.4}, max={max:.4}, mean={mean:.4}, nonzero={nz}/{n}"
            );
            assert!(nz > 0, "STAGE 3 FAILED: post-LN embeddings are all zero!");
        }

        // Print embedding output for inspection
        {
            let metal_flat = read_f32_buffer(&backend.workspace.hidden_a, n);
            eprintln!("\n  Token 0 (CLS), first 8 dims: {:?}", &metal_flat[..8]);
            eprintln!(
                "  Token 1 (Hello), first 8 dims: {:?}",
                &metal_flat[hd as usize..hd as usize + 8]
            );
            let norm: f32 = metal_flat[..hd as usize]
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt();
            eprintln!("  Token 0 L2 norm: {norm:.4}");
        }

        // ===== STAGE 4: Layer 0 attention =====
        eprintln!("\n--- Layer 0 attention ---");
        {
            // Build float attention mask first
            run_encoder(&backend.queue, |enc| {
                enc.setComputePipelineState(&backend.kernels.build_attn_mask);
                set_buffer(enc, &backend.workspace.mask, 0, 0);
                set_buffer(enc, &attn_mask_int_buf, 0, 1);
                set_i32_param(enc, batch * max_seq, 2);
                dispatch_1d(
                    enc,
                    &backend.kernels.build_attn_mask,
                    (batch * max_seq) as usize,
                );
            });

            let attn = &backend.model.layers[0].attention;
            let nh = backend.num_heads;
            let head_dim_i = backend.head_dim;
            let pad_hd_i = (hd as usize).next_multiple_of(8) as u32;

            // 4a: Q GEMM
            run_encoder(&backend.queue, |enc| {
                dispatch_gemm(
                    enc,
                    &backend.kernels.gemm,
                    &backend.workspace.hidden_a,
                    0,
                    &backend.weight_buffer,
                    attn.qkv_weight.offset,
                    &backend.workspace.q,
                    0,
                    batch_seq as u32,
                    pad_hd_i,
                    pad_hd_i,
                    true,
                );
            });
            let (mn, mx, avg, nz_q) = buffer_stats(&backend.workspace.q, n);
            eprintln!(
                "  4a - Q GEMM:         min={mn:.4}, max={mx:.4}, mean={avg:.4}, nz={nz_q}/{n}"
            );

            // 4b: add Q bias
            if let Some(ref bias) = attn.qkv_bias {
                run_encoder(&backend.queue, |enc| {
                    enc.setComputePipelineState(&backend.kernels.add_bias);
                    set_buffer(enc, &backend.workspace.q, 0, 0);
                    set_buffer(enc, &backend.weight_buffer, bias.offset, 1);
                    set_i32_param(enc, batch_seq, 2);
                    set_i32_param(enc, hd, 3);
                    dispatch_1d(enc, &backend.kernels.add_bias, n);
                });
                let (mn, mx, avg, nz_q) = buffer_stats(&backend.workspace.q, n);
                eprintln!(
                    "  4b - Q+bias:         min={mn:.4}, max={mx:.4}, mean={avg:.4}, nz={nz_q}/{n}"
                );
            }

            // 4c: K GEMM + bias
            if let Some(ref k_weight) = attn.k_weight {
                run_encoder(&backend.queue, |enc| {
                    dispatch_gemm(
                        enc,
                        &backend.kernels.gemm,
                        &backend.workspace.hidden_a,
                        0,
                        &backend.weight_buffer,
                        k_weight.offset,
                        &backend.workspace.k,
                        0,
                        batch_seq as u32,
                        pad_hd_i,
                        pad_hd_i,
                        true,
                    );
                });
                if let Some(ref bias) = attn.k_bias {
                    run_encoder(&backend.queue, |enc| {
                        enc.setComputePipelineState(&backend.kernels.add_bias);
                        set_buffer(enc, &backend.workspace.k, 0, 0);
                        set_buffer(enc, &backend.weight_buffer, bias.offset, 1);
                        set_i32_param(enc, batch_seq, 2);
                        set_i32_param(enc, hd, 3);
                        dispatch_1d(enc, &backend.kernels.add_bias, n);
                    });
                }
            }
            let (mn, mx, avg, nz_k) = buffer_stats(&backend.workspace.k, n);
            eprintln!(
                "  4c - K+bias:         min={mn:.4}, max={mx:.4}, mean={avg:.4}, nz={nz_k}/{n}"
            );

            // 4d: Head reshape Q -> attn_out, K -> q, V -> k
            let total_head = (batch * nh * max_seq * head_dim_i) as usize;
            run_encoder(&backend.queue, |enc| {
                // Reshape Q: workspace.q -> workspace.attn_out
                enc.setComputePipelineState(&backend.kernels.head_reshape);
                set_buffer(enc, &backend.workspace.attn_out, 0, 0);
                set_buffer(enc, &backend.workspace.q, 0, 1);
                set_i32_param(enc, batch, 2);
                set_i32_param(enc, max_seq, 3);
                set_i32_param(enc, nh, 4);
                set_i32_param(enc, head_dim_i, 5);
                dispatch_1d(enc, &backend.kernels.head_reshape, total_head);
            });
            let (mn, mx, avg, nz_qr) = buffer_stats(&backend.workspace.attn_out, total_head);
            eprintln!(
                "  4d - Q reshaped:     min={mn:.4}, max={mx:.4}, mean={avg:.4}, nz={nz_qr}/{total_head}"
            );

            // 4e: attention scores Q@K^T (just head 0 for diagnostics)
            let head_stride_qk_i = (max_seq * head_dim_i) as usize * core::mem::size_of::<f32>();
            let head_stride_scores_i = (max_seq * max_seq) as usize * core::mem::size_of::<f32>();
            let scores_per_head = (max_seq * max_seq) as usize;
            // Reshape K into workspace.q
            run_encoder(&backend.queue, |enc| {
                enc.setComputePipelineState(&backend.kernels.head_reshape);
                set_buffer(enc, &backend.workspace.q, 0, 0);
                set_buffer(enc, &backend.workspace.k, 0, 1);
                set_i32_param(enc, batch, 2);
                set_i32_param(enc, max_seq, 3);
                set_i32_param(enc, nh, 4);
                set_i32_param(enc, head_dim_i, 5);
                dispatch_1d(enc, &backend.kernels.head_reshape, total_head);
            });

            // Scores for head 0
            run_encoder(&backend.queue, |enc| {
                dispatch_gemm(
                    enc,
                    &backend.kernels.gemm,
                    &backend.workspace.attn_out,
                    0, // Q head 0
                    &backend.workspace.q,
                    0, // K head 0
                    &backend.workspace.scores,
                    0,
                    max_seq as u32,
                    max_seq as u32,
                    head_dim_i as u32,
                    true,
                );
            });
            let scores0 = read_f32_buffer(&backend.workspace.scores, scores_per_head);
            eprintln!("  4e - scores[head0]: {:?}", &scores0);

            // 4f: Apply scale + mask + softmax to head 0
            run_encoder(&backend.queue, |enc| {
                let scale = 1.0_f32 / (head_dim_i as f32).sqrt();
                enc.setComputePipelineState(&backend.kernels.fused_scale_mask_softmax);
                set_buffer(enc, &backend.workspace.scores, 0, 0);
                set_buffer(enc, &backend.workspace.mask, 0, 1);
                set_i32_param(enc, batch, 2);
                set_i32_param(enc, nh, 3);
                set_i32_param(enc, max_seq, 4);
                set_f32_param(enc, scale, 5);
                let threads = 256.min(max_seq as usize);
                // Only 1 head's worth of rows (max_seq rows)
                dispatch_rows(
                    enc,
                    &backend.kernels.fused_scale_mask_softmax,
                    max_seq as usize,
                    threads,
                );
            });
            let softmax0 = read_f32_buffer(&backend.workspace.scores, scores_per_head);
            eprintln!("  4f - softmax[head0]: {:?}", &softmax0);

            // Run the full attention layer
            run_encoder(&backend.queue, |enc| {
                backend.encode_attention(
                    enc,
                    &backend.model.layers[0],
                    &backend.workspace.hidden_a,
                    &backend.workspace.hidden_b,
                    &backend.workspace.mask,
                    batch,
                    max_seq,
                );
            });
            let (min, max, mean, nz) = buffer_stats(&backend.workspace.hidden_b, n);
            eprintln!(
                "  STAGE 4 - layer0 attn:   min={min:.4}, max={max:.4}, mean={mean:.4}, nz={nz}/{n}"
            );

            // Print per-token stats for attention output
            for t in 0..max_seq as usize {
                let start = t * hd as usize;
                let end = start + hd as usize;
                let token_data = read_f32_buffer(&backend.workspace.hidden_b, n);
                let tok = &token_data[start..end];
                let tok_min = tok.iter().copied().fold(f32::INFINITY, f32::min);
                let tok_max = tok.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let tok_nan = tok.iter().filter(|v| v.is_nan()).count();
                eprintln!("    token {t}: min={tok_min:.4}, max={tok_max:.4}, nan={tok_nan}");
            }
            assert!(nz > 0, "STAGE 4 FAILED: attention output is all zero!");
        }

        // ===== STAGE 5: Layer 0 FFN (sub-stages) =====
        eprintln!("\n--- Layer 0 FFN sub-stages ---");
        {
            let ffn = &backend.model.layers[0].ffn;
            let inter_dim = ffn.intermediate_weight.shape[0] as i32;
            let pad_hd_f = (hd as usize).next_multiple_of(8) as u32;
            let pad_inter = (inter_dim as usize).next_multiple_of(8) as u32;
            let inter_n = (batch_seq * inter_dim) as usize;

            // Print weight offsets for debugging
            eprintln!(
                "  FFN weight info: inter_weight offset={}, shape={:?}",
                ffn.intermediate_weight.offset, ffn.intermediate_weight.shape
            );
            if let Some(ref bias) = ffn.intermediate_bias {
                eprintln!(
                    "  FFN inter_bias offset={}, shape={:?}",
                    bias.offset, bias.shape
                );
                // Read and print the bias values' range
                let bias_data = unsafe {
                    core::slice::from_raw_parts(
                        (backend.weight_buffer.contents().as_ptr() as *const f32)
                            .add(bias.offset / 4),
                        bias.numel,
                    )
                };
                let bias_min = bias_data.iter().copied().fold(f32::INFINITY, f32::min);
                let bias_max = bias_data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let bias_nan = bias_data.iter().filter(|v| v.is_nan()).count();
                eprintln!(
                    "  FFN inter_bias values: min={bias_min:.4}, max={bias_max:.4}, nan={bias_nan}"
                );
            }

            // 5a: Intermediate GEMM: hidden_b → ffn_inter
            run_encoder(&backend.queue, |enc| {
                dispatch_gemm(
                    enc,
                    &backend.kernels.gemm,
                    &backend.workspace.hidden_b,
                    0,
                    &backend.weight_buffer,
                    ffn.intermediate_weight.offset,
                    &backend.workspace.ffn_inter,
                    0,
                    batch_seq as u32,
                    pad_inter,
                    pad_hd_f,
                    true,
                );
            });
            let (mn, mx, avg, nz_i) = buffer_stats(&backend.workspace.ffn_inter, inter_n);
            let nan_i = read_f32_buffer(&backend.workspace.ffn_inter, inter_n)
                .iter()
                .filter(|v| v.is_nan())
                .count();
            eprintln!(
                "  5a - intermediate GEMM:  min={mn:.4}, max={mx:.4}, mean={avg:.6}, nz={nz_i}/{inter_n}, nan={nan_i}"
            );

            // 5b: Fused bias + GELU — save pre-GELU values to find the NaN source
            let pre_gelu = read_f32_buffer(&backend.workspace.ffn_inter, inter_n);
            if let Some(ref bias) = ffn.intermediate_bias {
                // Also read bias values
                let bias_vals = unsafe {
                    core::slice::from_raw_parts(
                        (backend.weight_buffer.contents().as_ptr() as *const f32)
                            .add(bias.offset / 4),
                        bias.numel,
                    )
                };
                run_encoder(&backend.queue, |enc| {
                    enc.setComputePipelineState(&backend.kernels.fused_bias_gelu);
                    set_buffer(enc, &backend.workspace.ffn_inter, 0, 0);
                    set_buffer(enc, &backend.weight_buffer, bias.offset, 1);
                    set_i32_param(enc, batch_seq, 2);
                    set_i32_param(enc, inter_dim, 3);
                    dispatch_1d(enc, &backend.kernels.fused_bias_gelu, inter_n);
                });
                let post_gelu = read_f32_buffer(&backend.workspace.ffn_inter, inter_n);
                // Find which element(s) became NaN
                for (i, (pre, post)) in pre_gelu.iter().zip(post_gelu.iter()).enumerate() {
                    if post.is_nan() {
                        let col = i % inter_dim as usize;
                        let row = i / inter_dim as usize;
                        let b = bias_vals[col];
                        let v = pre + b;
                        eprintln!(
                            "  NaN at idx={i} (row={row}, col={col}): pre={pre:.6}, bias={b:.6}, v=pre+bias={v:.6}"
                        );
                        eprintln!(
                            "    v^3={:.6}, 0.044715*v^3={:.6}",
                            v * v * v,
                            0.044715 * v * v * v
                        );
                    }
                }
            }
            let (mn, mx, avg, nz_g) = buffer_stats(&backend.workspace.ffn_inter, inter_n);
            let nan_g = read_f32_buffer(&backend.workspace.ffn_inter, inter_n)
                .iter()
                .filter(|v| v.is_nan())
                .count();
            eprintln!(
                "  5b - after GELU:         min={mn:.4}, max={mx:.4}, mean={avg:.6}, nz={nz_g}/{inter_n}, nan={nan_g}"
            );

            // 5c: Output GEMM: ffn_inter → scratch
            run_encoder(&backend.queue, |enc| {
                dispatch_gemm(
                    enc,
                    &backend.kernels.gemm,
                    &backend.workspace.ffn_inter,
                    0,
                    &backend.weight_buffer,
                    ffn.output_weight.offset,
                    &backend.workspace.scratch,
                    0,
                    batch_seq as u32,
                    pad_hd_f,
                    pad_inter,
                    true,
                );
            });
            let (mn, mx, avg, nz_o) = buffer_stats(&backend.workspace.scratch, n);
            let nan_o = read_f32_buffer(&backend.workspace.scratch, n)
                .iter()
                .filter(|v| v.is_nan())
                .count();
            eprintln!(
                "  5c - output GEMM:        min={mn:.4}, max={mx:.4}, mean={avg:.6}, nz={nz_o}/{n}, nan={nan_o}"
            );

            // 5d: Full FFN (re-run from scratch to get final output)
            run_encoder(&backend.queue, |enc| {
                backend.encode_ffn(
                    enc,
                    &backend.model.layers[0].ffn,
                    &backend.workspace.hidden_b,
                    &backend.workspace.hidden_a,
                    batch_seq,
                );
            });
            let (min, max, mean, nz) = buffer_stats(&backend.workspace.hidden_a, n);
            let nan_f = read_f32_buffer(&backend.workspace.hidden_a, n)
                .iter()
                .filter(|v| v.is_nan())
                .count();
            eprintln!(
                "  5d - full FFN output:    min={min:.4}, max={max:.4}, mean={mean:.6}, nz={nz}/{n}, nan={nan_f}"
            );

            // Per-token NaN breakdown
            let all = read_f32_buffer(&backend.workspace.hidden_a, n);
            for t in 0..max_seq as usize {
                let tok = &all[t * hd as usize..(t + 1) * hd as usize];
                let nan_count = tok.iter().filter(|v| v.is_nan()).count();
                if nan_count > 0 {
                    eprintln!(
                        "    token {t}: {nan_count} NaN values (first 4: {:?})",
                        &tok[..4]
                    );
                }
            }
        }

        // ===== STAGE 6: Full embed_batch (all layers + CLS + L2) =====
        eprintln!("\n--- Full embed_batch ---");
        let result = backend.embed_batch(&[enc.clone()]).unwrap();
        let emb = &result[0];
        let nz = emb.iter().filter(|&&v| v.abs() > 1e-10).count();
        let l2: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        eprintln!(
            "STAGE 6 - final embedding:  dim={}, nonzero={nz}, L2={l2:.6}",
            emb.len()
        );
        eprintln!("  first 8: {:?}", &emb[..8]);

        assert!(nz > 0, "STAGE 6 FAILED: final embedding is all zero!");
        assert!((l2 - 1.0).abs() < 0.1, "L2 norm should be ~1.0, got {l2}");

        // Compare against CPU (only if cpu feature enabled)
        #[cfg(feature = "cpu")]
        {
            let cpu_backend = super::super::cpu::CpuBackend::load(
                "BAAI/bge-small-en-v1.5",
                &super::DeviceHint::Cpu,
            )
            .unwrap();
            let cpu_result = cpu_backend.embed_batch(&[enc]).unwrap();
            let cpu_emb = &cpu_result[0];
            eprintln!("  CPU first 8: {:?}", &cpu_emb[..8]);
            let cosine: f32 = emb.iter().zip(cpu_emb).map(|(m, c)| m * c).sum();
            eprintln!("  Metal vs CPU cosine similarity: {cosine:.6}");
            if cosine > 0.95 {
                eprintln!("  PASS: Metal matches CPU (cosine > 0.95)");
            } else {
                eprintln!("  FAIL: Metal and CPU diverged! cosine={cosine}");
            }
        }
    }

    #[test]
    #[ignore = "requires model download + cpu feature; run with --nocapture"]
    #[cfg(feature = "cpu")]
    fn metal_vs_cpu_sequence_lengths() {
        let metal = MetalBackend::load("BAAI/bge-small-en-v1.5", &DeviceHint::Auto).unwrap();
        let cpu =
            super::super::cpu::CpuBackend::load("BAAI/bge-small-en-v1.5", &super::DeviceHint::Cpu)
                .unwrap();

        // Test multiple sequence lengths: 4 (not mult of 8), 8, 16, 32
        for seq_len in [4, 8, 16, 32] {
            let mut ids: Vec<i64> = vec![101]; // CLS
            for i in 1..seq_len - 1 {
                ids.push(2000 + i as i64); // arbitrary word tokens
            }
            ids.push(102); // SEP

            let enc = Encoding {
                input_ids: ids.clone(),
                attention_mask: vec![1; seq_len],
                token_type_ids: vec![0; seq_len],
            };

            let m = metal.embed_batch(&[enc.clone()]).unwrap();
            let c = cpu.embed_batch(&[enc]).unwrap();
            let cos: f32 = m[0].iter().zip(&c[0]).map(|(a, b)| a * b).sum();
            let m_l2: f32 = m[0].iter().map(|x| x * x).sum::<f32>().sqrt();
            let c_l2: f32 = c[0].iter().map(|x| x * x).sum::<f32>().sqrt();
            eprintln!("seq={seq_len:>3}: cosine={cos:.6}, metal_L2={m_l2:.4}, cpu_L2={c_l2:.4}");
        }
    }
}
