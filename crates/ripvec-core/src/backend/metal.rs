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
use objc2::AnyThread;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSString, NSUInteger};
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLGPUFamily, MTLLibrary,
    MTLResourceOptions, MTLSize,
};
use objc2_metal_performance_shaders::{
    MPSDataType, MPSMatrix, MPSMatrixDescriptor, MPSMatrixMultiplication,
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
    /// Convert FP32 to FP16 (used by [`MetalBackend::create_fp16_weights`]).
    f32_to_f16: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Add bias in-place: `x[idx] += bias[idx % cols]`.
    add_bias: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Head reshape: `[batch*seq, hidden]` to `[batch*num_heads, seq, head_dim]`.
    ///
    /// No longer used in attention (replaced by fused QKV + `qkv_split`), but
    /// kept compiled for potential future use in other reshape operations.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "no longer used after QKV fusion, kept for potential future use"
        )
    )]
    head_reshape: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Two-input `SwiGLU`: `output = value * silu(gate)` with separate buffers.
    swiglu_two_input: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// FP32 GEMM kernel using `simdgroup_matrix_multiply_accumulate`.
    ///
    /// Superseded by [`Self::gemm_fp16`] for weight GEMMs; retained for
    /// tests and any future activation-activation non-batched GEMMs.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "weight GEMMs now use gemm_fp16; kept for tests and future use"
        )
    )]
    gemm: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Batched GEMM: same kernel with z-dimension for batch/head index.
    gemm_batched: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// FP16 mixed-precision GEMM: FP32 activations x FP16 weights -> FP32 output.
    ///
    /// Uses `half x half -> float` hardware matmul with FP32 accumulator.
    /// Superseded by MPS `MPSMatrixMultiplication` for weight GEMMs in production.
    /// Pipeline is compiled so the kernel remains available for standalone tests.
    #[expect(
        dead_code,
        reason = "weight GEMMs now use MPS; compiled for standalone test availability"
    )]
    gemm_fp16: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Batched FP16 mixed-precision GEMM with z-dimension for batch/head index.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "attention GEMMs are activation×activation (FP32); reserved for future weight-batched paths"
        )
    )]
    gemm_batched_fp16: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Fused `FlashAttention`: Q@K^T -> scale -> mask -> softmax -> @V in one kernel.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "batched GEMM is faster for small models; FA reserved for 768-dim+ models"
        )
    )]
    flash_attention: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    // ----- FP16 element-wise kernel variants (bandwidth optimization) -----
    /// FP16 layer normalization with FP32 accumulator reductions.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "FP16 kernels reserved for future activation-precision pipeline"
        )
    )]
    layer_norm_f16: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// FP16 GELU activation with FP32 compute.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "FP16 kernels reserved for future activation-precision pipeline"
        )
    )]
    gelu_f16: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// FP16 fused bias + GELU with FP32 compute.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "FP16 kernels reserved for future activation-precision pipeline"
        )
    )]
    fused_bias_gelu_f16: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// FP16 fused bias + residual add.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "FP16 kernels reserved for future activation-precision pipeline"
        )
    )]
    fused_bias_residual_f16: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// FP16 fused residual add + layer norm with FP32 accumulator reductions.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "FP16 kernels reserved for future activation-precision pipeline"
        )
    )]
    fused_residual_layernorm_f16: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// FP16 bias add (in-place).
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "FP16 kernels reserved for future activation-precision pipeline"
        )
    )]
    add_bias_f16: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// FP16 embedding add.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "FP16 kernels reserved for future activation-precision pipeline"
        )
    )]
    add_embeddings_f16: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// FP16 embedding lookup.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "FP16 kernels reserved for future activation-precision pipeline"
        )
    )]
    embedding_lookup_f16: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
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
            gemm_fp16: create_pipeline(device, &gemm_library, "gemm_fp16_kernel")?,
            gemm_batched_fp16: create_pipeline(device, &gemm_library, "gemm_batched_fp16_kernel")?,
            flash_attention: create_pipeline(device, &gemm_library, "flash_attention_kernel")?,
            // FP16 element-wise kernel variants
            layer_norm_f16: p("layer_norm_f16_kernel")?,
            gelu_f16: p("gelu_f16_kernel")?,
            fused_bias_gelu_f16: p("fused_bias_gelu_f16_kernel")?,
            fused_bias_residual_f16: p("fused_bias_residual_f16_kernel")?,
            fused_residual_layernorm_f16: p("fused_residual_layernorm_f16_kernel")?,
            add_bias_f16: p("add_bias_f16_kernel")?,
            add_embeddings_f16: p("add_embeddings_f16_kernel")?,
            embedding_lookup_f16: p("embedding_lookup_f16_kernel")?,
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

/// Dispatch a FP32 GEMM kernel: C\[M,N\] = A\[M,K\] * B\[K,N\] (or B^T when `trans_b`).
///
/// Each threadgroup computes a 32x32 output tile using 16 SIMD groups (4x4
/// arrangement of 8x8 tiles). Grid is `ceil(N/32) x ceil(M/32)` threadgroups.
///
/// Superseded by [`dispatch_gemm_fp16`] for weight GEMMs in production; retained
/// for tests that verify FP32 GEMM correctness.
///
/// # Safety
///
/// Caller must ensure buffer offsets and sizes are valid.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "weight GEMMs now use dispatch_gemm_fp16; kept for tests"
    )
)]
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
        let trans_b_u32: u32 = u32::from(trans_b);
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
        let params: [u32; 7] = [m, n, k, u32::from(trans_b), stride_a, stride_b, stride_c];
        for (i, val) in params.iter().enumerate() {
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new(std::ptr::from_ref::<u32>(val) as *mut _).unwrap(),
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

/// Dispatch FP16 mixed-precision GEMM: C\[M,N\] = A\[M,K\] * B\[K,N\] (or B^T).
///
/// A is FP32 activations (converted to half inside the kernel), B is FP16
/// weights (pre-converted at load time), C is FP32 output. The `b_offset`
/// is in **bytes** into the FP16 buffer (caller should pass `f32_offset / 2`).
///
/// Same threadgroup/tile layout as [`dispatch_gemm`].
///
/// Superseded by [`dispatch_mps_gemm`] for weight GEMMs in production; retained
/// for tests and sub-stage diagnostics.
///
/// # Safety
///
/// Caller must ensure buffer offsets and sizes are valid.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "weight GEMMs now use MPS; kept for tests and sub-stage diagnostics"
    )
)]
#[expect(
    unsafe_code,
    clippy::too_many_arguments,
    clippy::borrow_as_ptr,
    reason = "Metal buffer binding requires unsafe setBuffer/setBytes calls with raw pointers"
)]
fn dispatch_gemm_fp16(
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
        let trans_b_u32: u32 = u32::from(trans_b);
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

/// Create a compute command encoder from a command buffer.
///
/// Convenience wrapper that returns a [`crate::Result`] instead of an `Option`.
fn new_encoder(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
) -> crate::Result<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>> {
    cmd_buf
        .computeCommandEncoder()
        .ok_or_else(|| crate::Error::Metal("failed to create compute encoder".into()))
}

/// Dispatch a GEMM using Apple's MPS `MPSMatrixMultiplication`.
///
/// Encodes C\[M,N\] = alpha * A\[M,K\] * B\[K,N\] (or B^T when `trans_b`) + beta * C
/// directly onto the command buffer. MPS is hand-tuned per chip family and
/// typically 2-5x faster than our custom simdgroup GEMM for large weight matrices.
///
/// **Important**: MPS encodes directly to the command buffer, not to a compute
/// encoder. The caller must `endEncoding()` any active compute encoder before
/// calling this, and create a new encoder afterwards.
///
/// For FP16 weights with FP32 activations: pass the FP16 buffer for B and
/// `MPSDataType::Float16` for `b_data_type`. MPS will produce FP32 output.
///
/// # Safety
///
/// Caller must ensure buffer offsets and sizes are valid.
#[expect(
    unsafe_code,
    clippy::too_many_arguments,
    reason = "MPS matrix creation requires unsafe objc2 calls with many dimension parameters"
)]
fn dispatch_mps_gemm(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    device: &ProtocolObject<dyn MTLDevice>,
    a_buffer: &ProtocolObject<dyn MTLBuffer>,
    a_offset: usize,
    b_buffer: &ProtocolObject<dyn MTLBuffer>,
    b_offset: usize,
    c_buffer: &ProtocolObject<dyn MTLBuffer>,
    c_offset: usize,
    m: usize,
    n: usize,
    k: usize,
    trans_b: bool,
    a_data_type: MPSDataType,
    b_data_type: MPSDataType,
) {
    let a_elem_size = if a_data_type == MPSDataType::Float16 {
        2
    } else {
        4
    };
    let b_elem_size = if b_data_type == MPSDataType::Float16 {
        2
    } else {
        4
    };

    // Row bytes = columns * element_size
    let a_row_bytes = k * a_elem_size;
    let c_row_bytes = n * 4; // output is always FP32

    // B layout depends on transposition:
    // - trans_b=true:  B is [N,K] stored row-major, row_bytes = K * elem_size
    // - trans_b=false: B is [K,N] stored row-major, row_bytes = N * elem_size
    let (b_rows, b_cols, b_row_bytes) = if trans_b {
        (n, k, k * b_elem_size)
    } else {
        (k, n, n * b_elem_size)
    };

    unsafe {
        // Create matrix descriptors
        let desc_a = MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
            m as NSUInteger,
            k as NSUInteger,
            a_row_bytes as NSUInteger,
            a_data_type,
        );
        let desc_b = MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
            b_rows as NSUInteger,
            b_cols as NSUInteger,
            b_row_bytes as NSUInteger,
            b_data_type,
        );
        let desc_c = MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
            m as NSUInteger,
            n as NSUInteger,
            c_row_bytes as NSUInteger,
            MPSDataType::Float32,
        );

        // Wrap Metal buffers as MPS matrices (with byte offsets)
        let mat_a = MPSMatrix::initWithBuffer_offset_descriptor(
            MPSMatrix::alloc(),
            a_buffer,
            a_offset as NSUInteger,
            &desc_a,
        );
        let mat_b = MPSMatrix::initWithBuffer_offset_descriptor(
            MPSMatrix::alloc(),
            b_buffer,
            b_offset as NSUInteger,
            &desc_b,
        );
        let mat_c = MPSMatrix::initWithBuffer_offset_descriptor(
            MPSMatrix::alloc(),
            c_buffer,
            c_offset as NSUInteger,
            &desc_c,
        );

        // Create and encode the multiplication
        let gemm = MPSMatrixMultiplication::initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta(
            MPSMatrixMultiplication::alloc(),
            device,
            false,      // transposeLeft
            trans_b,    // transposeRight
            m as NSUInteger,
            n as NSUInteger,
            k as NSUInteger,
            1.0,        // alpha
            0.0,        // beta
        );

        gemm.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(
            cmd_buf, &mat_a, &mat_b, &mat_c,
        );
    }
}

/// Maximum batch size per forward pass.
///
/// Smaller batches reduce padding waste: `embed_distributed` grabs 128+
/// sequences sorted by descending length, then `embed_batch` sub-batches
/// them here. With MAX_BATCH=32, a 128-sequence grab becomes 4 sub-batches
/// each padded to their own (shorter) max_seq — much less padding than one
/// 128-sequence batch padded to the global max.
const MAX_BATCH: i32 = 32;

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

/// Reference to a weight tensor within a Metal buffer.
///
/// Instead of copying tensor data, `WeightRef` stores the byte offset and
/// shape so kernels can index directly into the Metal buffer. For zero-copy
/// weights this points into the mmap'd safetensors buffer; for fused weights
/// (e.g. concatenated QKV) it points into an entry in `fused_weights`.
struct WeightRef {
    /// Byte offset from the start of the backing Metal buffer.
    offset: usize,
    /// Tensor shape (e.g. `[384, 384]`).
    shape: Vec<usize>,
    /// Number of f32 elements.
    numel: usize,
    /// Which Metal buffer this ref points into.
    /// `None` = `weight_buffer` (zero-copy mmap), `Some(idx)` = `fused_weights[idx]`.
    buffer_idx: Option<usize>,
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
                buffer_idx: None,
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
        buffer_idx: r.buffer_idx,
    })
}

/// Take an optional weight ref by cloning its data from the map.
fn try_take_weight_ref(refs: &HashMap<String, WeightRef>, name: &str) -> Option<WeightRef> {
    try_get_weight_ref(refs, name).map(|r| WeightRef {
        offset: r.offset,
        shape: r.shape.clone(),
        numel: r.numel,
        buffer_idx: r.buffer_idx,
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
    /// Fused weight buffers allocated at load time (`ClassicBert`: Q+K+V concatenated).
    /// Kept alive so the [`WeightRef`]s pointing into them remain valid.
    fused_weights: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// FP16 copy of `weight_buffer`, converted on GPU at load time.
    ///
    /// Every f32 weight becomes an f16 half at `offset / 2`. Used by FP16
    /// mixed-precision GEMM kernels to reduce register pressure and improve
    /// GPU occupancy (~1.5x fewer registers per SIMD group).
    fp16_weight_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// FP16 copies of `fused_weights` buffers, one per fused buffer.
    fp16_fused_weights: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
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
    /// Optional early exit: run only this many encoder layers.
    /// `None` runs all layers. Used for fast query embedding.
    max_layers: Option<usize>,
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

        let weights_bytes = mmap.len();
        let mut backend = Self {
            device,
            queue,
            chip_family,
            weight_buffer,
            model,
            kernels,
            workspace,
            rope_cache,
            fused_weights: Vec::new(),
            fp16_weight_buffer: None,
            fp16_fused_weights: Vec::new(),
            hidden_size,
            num_heads,
            head_dim,
            variant,
            max_position_embeddings,
            max_layers: None,
            _mmap: mmap,
        };

        // Fuse Q+K+V weights for ClassicBert so both variants use the same
        // fast 2-dispatch QKV path (1 fused GEMM + 1 qkv_split).
        if variant == ModelVariant::ClassicBert {
            backend.fuse_qkv_weights()?;
        }

        // Pre-convert all weights to FP16 on GPU for mixed-precision GEMM.
        // This runs the f32_to_f16 kernel on the entire weight buffer and each
        // fused buffer, storing half-precision copies alongside the originals.
        backend.create_fp16_weights()?;

        tracing::info!(
            device = %backend.device.name(),
            chip_family = ?backend.chip_family,
            hidden_size,
            num_heads,
            head_dim,
            layers = config.num_hidden_layers,
            variant = ?variant,
            weights_bytes,
            fp16_weights_bytes = weights_bytes / 2,
            fused_buffers = backend.fused_weights.len(),
            "Metal backend initialized with zero-copy weights + FP16 GEMM + 22 MSL kernels"
        );

        Ok(backend)
    }

    /// Fuse separate Q, K, V weight matrices into a single `[3*hidden, hidden]`
    /// tensor per layer, enabling the unified 2-dispatch QKV path.
    ///
    /// For `ClassicBert`, the safetensors file stores Q/K/V as three separate
    /// `[hidden, hidden]` matrices. This method concatenates them into one
    /// allocated Metal buffer per layer and updates the `WeightRef`s so
    /// `encode_attention` can use a single fused GEMM + `qkv_split` kernel.
    ///
    /// Also fuses Q/K/V biases (`[hidden]` each -> `[3*hidden]`) when present.
    #[expect(
        unsafe_code,
        clippy::cast_sign_loss,
        reason = "Metal buffer creation requires unsafe FFI; hidden_size is positive"
    )]
    fn fuse_qkv_weights(&mut self) -> crate::Result<()> {
        let hd = self.hidden_size as usize;

        for layer in &mut self.model.layers {
            let attn = &mut layer.attention;

            // --- Fuse QKV weights ---
            if attn.k_weight.is_some() && attn.v_weight.is_some() {
                let q_ref = &attn.qkv_weight;
                let k_ref = attn.k_weight.as_ref().expect("checked is_some");
                let v_ref = attn.v_weight.as_ref().expect("checked is_some");

                // Read weight data from the zero-copy mmap buffer.
                // WeightRef.offset is a BYTE offset; cast to f32 pointer.
                let base = self.weight_buffer.contents().as_ptr() as *const f32;
                let q_data =
                    unsafe { core::slice::from_raw_parts(base.add(q_ref.offset / 4), q_ref.numel) };
                let k_data =
                    unsafe { core::slice::from_raw_parts(base.add(k_ref.offset / 4), k_ref.numel) };
                let v_data =
                    unsafe { core::slice::from_raw_parts(base.add(v_ref.offset / 4), v_ref.numel) };

                // Concatenate Q, K, V into [3*hidden, hidden] (Q rows, then K, then V)
                let fused_numel = 3 * hd * hd;
                let mut fused = Vec::with_capacity(fused_numel);
                fused.extend_from_slice(q_data);
                fused.extend_from_slice(k_data);
                fused.extend_from_slice(v_data);

                // Allocate a new Metal buffer for the fused weight
                let fused_size = (fused_numel * core::mem::size_of::<f32>()) as NSUInteger;
                let fused_buf = unsafe {
                    self.device.newBufferWithBytes_length_options(
                        std::ptr::NonNull::new(fused.as_ptr() as *mut _)
                            .ok_or_else(|| crate::Error::Metal("null fused data ptr".into()))?,
                        fused_size,
                        MTLResourceOptions::StorageModeShared,
                    )
                }
                .ok_or_else(|| crate::Error::Metal("fused QKV weight alloc failed".into()))?;

                let buf_idx = self.fused_weights.len();
                self.fused_weights.push(fused_buf);

                // Update qkv_weight to point at the fused buffer
                attn.qkv_weight = WeightRef {
                    offset: 0,
                    shape: vec![3 * hd, hd],
                    numel: fused_numel,
                    buffer_idx: Some(buf_idx),
                };
                attn.k_weight = None;
                attn.v_weight = None;
            }

            // --- Fuse QKV biases ---
            if attn.qkv_bias.is_some() && attn.k_bias.is_some() && attn.v_bias.is_some() {
                let q_bias = attn.qkv_bias.as_ref().expect("checked is_some");
                let k_bias = attn.k_bias.as_ref().expect("checked is_some");
                let v_bias = attn.v_bias.as_ref().expect("checked is_some");

                let base = self.weight_buffer.contents().as_ptr() as *const f32;
                let q_data =
                    unsafe { core::slice::from_raw_parts(base.add(q_bias.offset / 4), hd) };
                let k_data =
                    unsafe { core::slice::from_raw_parts(base.add(k_bias.offset / 4), hd) };
                let v_data =
                    unsafe { core::slice::from_raw_parts(base.add(v_bias.offset / 4), hd) };

                let fused_numel = 3 * hd;
                let mut fused = Vec::with_capacity(fused_numel);
                fused.extend_from_slice(q_data);
                fused.extend_from_slice(k_data);
                fused.extend_from_slice(v_data);

                let fused_size = (fused_numel * core::mem::size_of::<f32>()) as NSUInteger;
                let fused_buf = unsafe {
                    self.device.newBufferWithBytes_length_options(
                        std::ptr::NonNull::new(fused.as_ptr() as *mut _)
                            .ok_or_else(|| crate::Error::Metal("null fused bias ptr".into()))?,
                        fused_size,
                        MTLResourceOptions::StorageModeShared,
                    )
                }
                .ok_or_else(|| crate::Error::Metal("fused QKV bias alloc failed".into()))?;

                let buf_idx = self.fused_weights.len();
                self.fused_weights.push(fused_buf);

                attn.qkv_bias = Some(WeightRef {
                    offset: 0,
                    shape: vec![3 * hd],
                    numel: fused_numel,
                    buffer_idx: Some(buf_idx),
                });
                attn.k_bias = None;
                attn.v_bias = None;
            }
        }

        Ok(())
    }

    /// Get the Metal buffer backing a [`WeightRef`].
    ///
    /// Returns `weight_buffer` for zero-copy mmap weights, or the appropriate
    /// fused buffer for weights concatenated at load time.
    fn weight_buf(&self, wr: &WeightRef) -> &ProtocolObject<dyn MTLBuffer> {
        match wr.buffer_idx {
            None => &self.weight_buffer,
            Some(idx) => &self.fused_weights[idx],
        }
    }

    /// Get the FP16 Metal buffer backing a [`WeightRef`].
    ///
    /// Returns the half-precision equivalent of [`Self::weight_buf`]. The
    /// byte offset into the returned buffer is `wr.offset / 2` (since each
    /// 4-byte float becomes a 2-byte half at the same element index).
    fn fp16_weight_buf(&self, wr: &WeightRef) -> &ProtocolObject<dyn MTLBuffer> {
        match wr.buffer_idx {
            None => self
                .fp16_weight_buffer
                .as_ref()
                .expect("fp16_weight_buffer created at load time"),
            Some(idx) => &self.fp16_fused_weights[idx],
        }
    }

    /// Convert a single FP32 Metal buffer to FP16 on GPU using the `f32_to_f16` kernel.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "weight buffer element count fits in i32 (max ~50M floats for a 200MB model)"
    )]
    fn convert_buffer_to_fp16(
        &self,
        src: &ProtocolObject<dyn MTLBuffer>,
    ) -> crate::Result<Retained<ProtocolObject<dyn MTLBuffer>>> {
        use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder};

        let num_floats = src.length() / 4;
        let fp16_size = num_floats * 2;

        let fp16_buf = self
            .device
            .newBufferWithLength_options(fp16_size, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| crate::Error::Metal("FP16 buffer allocation failed".into()))?;

        let cmd_buf = self
            .queue
            .commandBuffer()
            .ok_or_else(|| crate::Error::Metal("command buffer creation failed".into()))?;
        let enc = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| crate::Error::Metal("compute encoder creation failed".into()))?;

        enc.setComputePipelineState(&self.kernels.f32_to_f16);
        set_buffer(&enc, &fp16_buf, 0, 0);
        set_buffer(&enc, src, 0, 1);
        set_i32_param(&enc, num_floats as i32, 2);
        dispatch_1d(&enc, &self.kernels.f32_to_f16, num_floats);
        enc.endEncoding();

        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        Ok(fp16_buf)
    }

    /// Pre-convert all weight buffers to FP16 on GPU.
    ///
    /// Converts the main `weight_buffer` (zero-copy mmap) and all `fused_weights`
    /// buffers using the `f32_to_f16` compute kernel. The resulting half-precision
    /// buffers have the same element layout but half the byte size.
    fn create_fp16_weights(&mut self) -> crate::Result<()> {
        // Convert main weight buffer
        let fp16_main = self.convert_buffer_to_fp16(&self.weight_buffer)?;
        tracing::debug!(
            fp32_bytes = self.weight_buffer.length(),
            fp16_bytes = fp16_main.length(),
            "converted weight_buffer to FP16"
        );
        self.fp16_weight_buffer = Some(fp16_main);

        // Convert each fused weight buffer
        let mut fp16_fused = Vec::with_capacity(self.fused_weights.len());
        for (i, fused_buf) in self.fused_weights.iter().enumerate() {
            let fp16_buf = self.convert_buffer_to_fp16(fused_buf)?;
            tracing::debug!(
                idx = i,
                fp32_bytes = fused_buf.length(),
                fp16_bytes = fp16_buf.length(),
                "converted fused_weights[{i}] to FP16"
            );
            fp16_fused.push(fp16_buf);
        }
        self.fp16_fused_weights = fp16_fused;

        Ok(())
    }

    /// Encode the attention sub-layer for one transformer layer.
    ///
    /// Reads from `input` buffer, writes result to `output` buffer.
    /// Uses ping-pong workspace buffers for intermediate results.
    ///
    /// Both `NomicBert` and `ClassicBert` use the same unified path:
    /// 1. Fused QKV GEMM: `input @ qkv_weight^T -> workspace.qkv`
    /// 2. Optional QKV bias add
    /// 3. `qkv_split` kernel: split `[batch*seq, 3*hidden]` -> Q, K, V
    /// 4. Optional `RoPE` (`NomicBert` only)
    /// 5. Batched attention: Q@K^T -> scale+mask+softmax -> @V
    /// 6. Reshape + output projection + residual + layer norm
    ///
    /// `ClassicBert` Q+K+V weights are fused at load time by
    /// [`Self::fuse_qkv_weights`], so both variants dispatch only 2 kernels
    /// for the QKV projection (1 GEMM + 1 split) instead of 9 (3 GEMMs + 3
    /// bias adds + 3 head reshapes).
    #[expect(
        clippy::cast_sign_loss,
        clippy::too_many_arguments,
        clippy::too_many_lines,
        clippy::cast_precision_loss,
        reason = "GPU attention dispatch requires many args, lines, and integer casts"
    )]
    fn encode_attention(
        &self,
        cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
        layer: &MetalBertLayer,
        input: &ProtocolObject<dyn MTLBuffer>,
        output: &ProtocolObject<dyn MTLBuffer>,
        mask: &ProtocolObject<dyn MTLBuffer>,
        batch: i32,
        max_seq: i32,
    ) -> crate::Result<()> {
        let hd = self.hidden_size;
        let nh = self.num_heads;
        let head_dim = self.head_dim;
        let batch_seq = batch * max_seq;
        let batch_heads = batch * nh;
        let attn = &layer.attention;

        // --- QKV GEMM via MPS: input @ qkv_weight^T -> workspace.qkv ---
        // MPS encodes directly to the command buffer (no compute encoder).
        // FP16 weights: MPS handles mixed-precision internally.
        dispatch_mps_gemm(
            cmd_buf,
            &self.device,
            input,
            0,
            self.fp16_weight_buf(&attn.qkv_weight),
            attn.qkv_weight.offset / 2,
            &self.workspace.qkv,
            0,
            batch_seq as usize,
            (3 * hd) as usize,
            hd as usize,
            true,
            MPSDataType::Float32,
            MPSDataType::Float16,
        );

        // --- Compute encoder for QKV post-processing + attention ---
        {
            let enc = new_encoder(cmd_buf)?;

            // QKV bias add (if present)
            if let Some(ref bias) = attn.qkv_bias {
                let total_qkv = batch_seq * 3 * hd;
                enc.setComputePipelineState(&self.kernels.add_bias);
                set_buffer(&enc, &self.workspace.qkv, 0, 0);
                set_buffer(&enc, self.weight_buf(bias), bias.offset, 1);
                set_i32_param(&enc, batch_seq, 2);
                set_i32_param(&enc, 3 * hd, 3);
                dispatch_1d(&enc, &self.kernels.add_bias, total_qkv as usize);
            }

            // QKV split: [batch*seq, 3*hidden] -> Q,K,V [batch*nh, seq, head_dim]
            {
                let total_head = batch_heads * max_seq * head_dim;
                enc.setComputePipelineState(&self.kernels.qkv_split);
                set_buffer(&enc, &self.workspace.q, 0, 0);
                set_buffer(&enc, &self.workspace.k, 0, 1);
                set_buffer(&enc, &self.workspace.v, 0, 2);
                set_buffer(&enc, &self.workspace.qkv, 0, 3);
                set_i32_param(&enc, batch, 4);
                set_i32_param(&enc, max_seq, 5);
                set_i32_param(&enc, hd, 6);
                set_i32_param(&enc, nh, 7);
                set_i32_param(&enc, head_dim, 8);
                dispatch_1d(&enc, &self.kernels.qkv_split, total_head as usize);
            }

            let (q_buf, k_buf, v_buf) =
                (&*self.workspace.q, &*self.workspace.k, &*self.workspace.v);

            // RoPE (NomicBert only)
            if let Some(ref rope) = self.rope_cache
                && attn.rotary_emb_base.is_some()
            {
                let half = head_dim / 2;
                let num_rows = batch_heads * max_seq;
                let total_rope = num_rows * half;

                enc.setComputePipelineState(&self.kernels.rope_cached);
                set_buffer(&enc, q_buf, 0, 0);
                set_buffer(&enc, &rope.cos, 0, 1);
                set_buffer(&enc, &rope.sin, 0, 2);
                set_i32_param(&enc, num_rows, 3);
                set_i32_param(&enc, max_seq, 4);
                set_i32_param(&enc, head_dim, 5);
                set_i32_param(&enc, nh, 6);
                dispatch_1d(&enc, &self.kernels.rope_cached, total_rope as usize);

                enc.setComputePipelineState(&self.kernels.rope_cached);
                set_buffer(&enc, k_buf, 0, 0);
                set_buffer(&enc, &rope.cos, 0, 1);
                set_buffer(&enc, &rope.sin, 0, 2);
                set_i32_param(&enc, num_rows, 3);
                set_i32_param(&enc, max_seq, 4);
                set_i32_param(&enc, head_dim, 5);
                set_i32_param(&enc, nh, 6);
                dispatch_1d(&enc, &self.kernels.rope_cached, total_rope as usize);
            }

            // --- Attention via batched GEMM (custom kernel -- small per-head matmuls) ---
            // Keep as custom batched GEMM: MPS doesn't efficiently handle the
            // strided batched pattern for small per-head matrices.
            let stride_qk = (max_seq * head_dim) as u32;
            let stride_scores = (max_seq * max_seq) as u32;

            dispatch_gemm_batched(
                &enc,
                &self.kernels.gemm_batched,
                q_buf,
                0,
                k_buf,
                0,
                &self.workspace.scores,
                0,
                max_seq as u32,
                max_seq as u32,
                head_dim as u32,
                true,
                stride_qk,
                stride_qk,
                stride_scores,
                batch_heads as u32,
            );

            {
                let scale = 1.0_f32 / (head_dim as f32).sqrt();
                let total_rows = batch_heads * max_seq;
                let threads = 256.min(max_seq as usize);
                enc.setComputePipelineState(&self.kernels.fused_scale_mask_softmax);
                set_buffer(&enc, &self.workspace.scores, 0, 0);
                set_buffer(&enc, mask, 0, 1);
                set_i32_param(&enc, batch, 2);
                set_i32_param(&enc, nh, 3);
                set_i32_param(&enc, max_seq, 4);
                set_f32_param(&enc, scale, 5);
                dispatch_rows(
                    &enc,
                    &self.kernels.fused_scale_mask_softmax,
                    total_rows as usize,
                    threads,
                );
            }

            // scores @ V -> attn_out
            dispatch_gemm_batched(
                &enc,
                &self.kernels.gemm_batched,
                &self.workspace.scores,
                0,
                v_buf,
                0,
                &self.workspace.attn_out,
                0,
                max_seq as u32,
                head_dim as u32,
                max_seq as u32,
                false,
                stride_scores,
                stride_qk,
                stride_qk,
                batch_heads as u32,
            );

            // Reshape: [batch*nh, seq, head_dim] -> [batch*seq, hidden]
            {
                let total_ctx = batch_seq * hd;
                enc.setComputePipelineState(&self.kernels.attn_reshape);
                set_buffer(&enc, &self.workspace.context, 0, 0);
                set_buffer(&enc, &self.workspace.attn_out, 0, 1);
                set_i32_param(&enc, batch, 2);
                set_i32_param(&enc, max_seq, 3);
                set_i32_param(&enc, nh, 4);
                set_i32_param(&enc, head_dim, 5);
                dispatch_1d(&enc, &self.kernels.attn_reshape, total_ctx as usize);
            }

            enc.endEncoding();
        }

        // --- Output projection via MPS: context @ output_weight^T -> projected ---
        dispatch_mps_gemm(
            cmd_buf,
            &self.device,
            &self.workspace.context,
            0,
            self.fp16_weight_buf(&attn.output_weight),
            attn.output_weight.offset / 2,
            &self.workspace.projected,
            0,
            batch_seq as usize,
            hd as usize,
            hd as usize,
            true,
            MPSDataType::Float32,
            MPSDataType::Float16,
        );

        // --- Bias + residual + LayerNorm (compute encoder) ---
        {
            let enc = new_encoder(cmd_buf)?;

            if let Some(ref bias) = attn.output_bias {
                let total_proj = batch_seq * hd;
                enc.setComputePipelineState(&self.kernels.fused_bias_residual);
                set_buffer(&enc, &self.workspace.scratch, 0, 0);
                set_buffer(&enc, &self.workspace.projected, 0, 1);
                set_buffer(&enc, &self.weight_buffer, bias.offset, 2);
                set_buffer(&enc, input, 0, 3);
                set_i32_param(&enc, batch_seq, 4);
                set_i32_param(&enc, hd, 5);
                dispatch_1d(&enc, &self.kernels.fused_bias_residual, total_proj as usize);

                let eps = attn.layer_norm_eps;
                let threads = 256.min(hd as usize);
                enc.setComputePipelineState(&self.kernels.layer_norm);
                set_buffer(&enc, output, 0, 0);
                set_buffer(&enc, &self.workspace.scratch, 0, 1);
                set_buffer(&enc, &self.weight_buffer, attn.output_ln_weight.offset, 2);
                set_buffer(&enc, &self.weight_buffer, attn.output_ln_bias.offset, 3);
                set_i32_param(&enc, batch_seq, 4);
                set_i32_param(&enc, hd, 5);
                set_f32_param(&enc, eps, 6);
                dispatch_rows(&enc, &self.kernels.layer_norm, batch_seq as usize, threads);
            } else {
                let eps = attn.layer_norm_eps;
                let threads = 256.min(hd as usize);
                enc.setComputePipelineState(&self.kernels.fused_residual_layernorm);
                set_buffer(&enc, output, 0, 0);
                set_buffer(&enc, &self.workspace.projected, 0, 1);
                set_buffer(&enc, input, 0, 2);
                set_buffer(&enc, &self.weight_buffer, attn.output_ln_weight.offset, 3);
                set_buffer(&enc, &self.weight_buffer, attn.output_ln_bias.offset, 4);
                set_i32_param(&enc, batch_seq, 5);
                set_i32_param(&enc, hd, 6);
                set_f32_param(&enc, eps, 7);
                dispatch_rows(
                    &enc,
                    &self.kernels.fused_residual_layernorm,
                    batch_seq as usize,
                    threads,
                );
            }

            enc.endEncoding();
        }

        Ok(())
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
        reason = "GPU FFN dispatch with alternating MPS/compute requires many lines and casts"
    )]
    fn encode_ffn(
        &self,
        cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
        ffn: &MetalBertFfn,
        input: &ProtocolObject<dyn MTLBuffer>,
        output: &ProtocolObject<dyn MTLBuffer>,
        batch_seq: i32,
    ) -> crate::Result<()> {
        let hd = self.hidden_size;
        let inter_dim = ffn.intermediate_weight.shape[0] as i32;

        match self.variant {
            ModelVariant::ClassicBert => {
                // --- Intermediate projection via MPS ---
                dispatch_mps_gemm(
                    cmd_buf,
                    &self.device,
                    input,
                    0,
                    self.fp16_weight_buf(&ffn.intermediate_weight),
                    ffn.intermediate_weight.offset / 2,
                    &self.workspace.ffn_inter,
                    0,
                    batch_seq as usize,
                    inter_dim as usize,
                    hd as usize,
                    true,
                    MPSDataType::Float32,
                    MPSDataType::Float16,
                );

                // --- Bias + GELU (compute encoder) ---
                {
                    let enc = new_encoder(cmd_buf)?;
                    let total_act = batch_seq * inter_dim;
                    if let Some(ref bias) = ffn.intermediate_bias {
                        enc.setComputePipelineState(&self.kernels.fused_bias_gelu);
                        set_buffer(&enc, &self.workspace.ffn_inter, 0, 0);
                        set_buffer(&enc, &self.weight_buffer, bias.offset, 1);
                        set_i32_param(&enc, batch_seq, 2);
                        set_i32_param(&enc, inter_dim, 3);
                        dispatch_1d(&enc, &self.kernels.fused_bias_gelu, total_act as usize);
                    } else {
                        enc.setComputePipelineState(&self.kernels.gelu);
                        set_buffer(&enc, &self.workspace.ffn_inter, 0, 0);
                        set_i32_param(&enc, total_act, 1);
                        dispatch_1d(&enc, &self.kernels.gelu, total_act as usize);
                    }
                    enc.endEncoding();
                }

                // --- Output projection via MPS ---
                dispatch_mps_gemm(
                    cmd_buf,
                    &self.device,
                    &self.workspace.ffn_inter,
                    0,
                    self.fp16_weight_buf(&ffn.output_weight),
                    ffn.output_weight.offset / 2,
                    &self.workspace.scratch,
                    0,
                    batch_seq as usize,
                    hd as usize,
                    inter_dim as usize,
                    true,
                    MPSDataType::Float32,
                    MPSDataType::Float16,
                );
            }
            ModelVariant::NomicBert => {
                // SwiGLU with separate fc11 (value) and fc12 (gate) weights.
                // Two MPS GEMMs to separate buffers, then compute SwiGLU element-wise.

                // --- fc11 (value) via MPS: input @ fc11_weight^T -> ffn_inter ---
                dispatch_mps_gemm(
                    cmd_buf,
                    &self.device,
                    input,
                    0,
                    self.fp16_weight_buf(&ffn.intermediate_weight),
                    ffn.intermediate_weight.offset / 2,
                    &self.workspace.ffn_inter,
                    0,
                    batch_seq as usize,
                    inter_dim as usize,
                    hd as usize,
                    true,
                    MPSDataType::Float32,
                    MPSDataType::Float16,
                );

                // --- fc12 (gate) via MPS: input @ gate_weight^T -> activated ---
                if let Some(ref gate) = ffn.gate_weight {
                    dispatch_mps_gemm(
                        cmd_buf,
                        &self.device,
                        input,
                        0,
                        self.fp16_weight_buf(gate),
                        gate.offset / 2,
                        &self.workspace.activated,
                        0,
                        batch_seq as usize,
                        inter_dim as usize,
                        hd as usize,
                        true,
                        MPSDataType::Float32,
                        MPSDataType::Float16,
                    );
                }

                // --- SwiGLU (compute encoder) ---
                {
                    let enc = new_encoder(cmd_buf)?;
                    let total_act = batch_seq * inter_dim;
                    enc.setComputePipelineState(&self.kernels.swiglu_two_input);
                    set_buffer(&enc, &self.workspace.activated, 0, 0);
                    set_buffer(&enc, &self.workspace.ffn_inter, 0, 1);
                    set_buffer(&enc, &self.workspace.activated, 0, 2);
                    set_i32_param(&enc, total_act, 3);
                    dispatch_1d(&enc, &self.kernels.swiglu_two_input, total_act as usize);
                    enc.endEncoding();
                }

                // --- Output projection via MPS ---
                dispatch_mps_gemm(
                    cmd_buf,
                    &self.device,
                    &self.workspace.activated,
                    0,
                    self.fp16_weight_buf(&ffn.output_weight),
                    ffn.output_weight.offset / 2,
                    &self.workspace.scratch,
                    0,
                    batch_seq as usize,
                    hd as usize,
                    inter_dim as usize,
                    true,
                    MPSDataType::Float32,
                    MPSDataType::Float16,
                );
            }
        }

        // --- Bias + residual + LayerNorm (compute encoder) ---
        {
            let enc = new_encoder(cmd_buf)?;

            if let Some(ref bias) = ffn.output_bias {
                let total_out = batch_seq * hd;
                enc.setComputePipelineState(&self.kernels.fused_bias_residual);
                set_buffer(&enc, &self.workspace.projected, 0, 0);
                set_buffer(&enc, &self.workspace.scratch, 0, 1);
                set_buffer(&enc, &self.weight_buffer, bias.offset, 2);
                set_buffer(&enc, input, 0, 3);
                set_i32_param(&enc, batch_seq, 4);
                set_i32_param(&enc, hd, 5);
                dispatch_1d(&enc, &self.kernels.fused_bias_residual, total_out as usize);

                let eps = ffn.layer_norm_eps;
                let threads = 256.min(hd as usize);
                enc.setComputePipelineState(&self.kernels.layer_norm);
                set_buffer(&enc, output, 0, 0);
                set_buffer(&enc, &self.workspace.projected, 0, 1);
                set_buffer(&enc, &self.weight_buffer, ffn.output_ln_weight.offset, 2);
                set_buffer(&enc, &self.weight_buffer, ffn.output_ln_bias.offset, 3);
                set_i32_param(&enc, batch_seq, 4);
                set_i32_param(&enc, hd, 5);
                set_f32_param(&enc, eps, 6);
                dispatch_rows(&enc, &self.kernels.layer_norm, batch_seq as usize, threads);
            } else {
                let eps = ffn.layer_norm_eps;
                let threads = 256.min(hd as usize);
                enc.setComputePipelineState(&self.kernels.fused_residual_layernorm);
                set_buffer(&enc, output, 0, 0);
                set_buffer(&enc, &self.workspace.scratch, 0, 1);
                set_buffer(&enc, input, 0, 2);
                set_buffer(&enc, &self.weight_buffer, ffn.output_ln_weight.offset, 3);
                set_buffer(&enc, &self.weight_buffer, ffn.output_ln_bias.offset, 4);
                set_i32_param(&enc, batch_seq, 5);
                set_i32_param(&enc, hd, 6);
                set_f32_param(&enc, eps, 7);
                dispatch_rows(
                    &enc,
                    &self.kernels.fused_residual_layernorm,
                    batch_seq as usize,
                    threads,
                );
            }

            enc.endEncoding();
        }

        Ok(())
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
    ///
    /// Weight GEMMs use MPS `MPSMatrixMultiplication` which encodes directly to
    /// the command buffer. The forward pass alternates between compute encoders
    /// (for element-wise ops, attention) and MPS encoding (for weight GEMMs).
    #[expect(
        unsafe_code,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        clippy::too_many_lines,
        reason = "Metal forward pass requires unsafe FFI, integer casts, and multi-encoder dispatch"
    )]
    fn forward_batch(&self, encodings: &[Encoding]) -> crate::Result<Vec<Vec<f32>>> {
        let batch = encodings.len() as i32;
        let hd = self.hidden_size;

        // Find max sequence length in this batch, rounded up to nearest
        // multiple of 8. The simdgroup GEMM kernel operates on 8x8 tiles;
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

        // Create command buffer
        let cmd_buf = self
            .queue
            .commandBuffer()
            .ok_or_else(|| crate::Error::Metal("failed to create command buffer".into()))?;

        // --- Embeddings (compute encoder) ---
        {
            let enc = new_encoder(&cmd_buf)?;

            // Build float attention mask
            let mask_total = batch * max_seq;
            enc.setComputePipelineState(&self.kernels.build_attn_mask);
            set_buffer(&enc, &self.workspace.mask, 0, 0);
            set_buffer(&enc, &attn_mask_int_buf, 0, 1);
            set_i32_param(&enc, mask_total, 2);
            dispatch_1d(&enc, &self.kernels.build_attn_mask, mask_total as usize);

            // Embeddings: [batch*max_seq, hidden] -> hidden_a
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

            enc.endEncoding();
        }

        // --- Transformer layers: ping-pong hidden_a <-> hidden_b ---
        // Each encode_attention/encode_ffn manages its own compute encoders
        // and MPS GEMM dispatches internally.
        let num_layers = self
            .max_layers
            .unwrap_or(self.model.layers.len())
            .min(self.model.layers.len());
        for layer in &self.model.layers[..num_layers] {
            // Attention: hidden_a -> hidden_b
            self.encode_attention(
                &cmd_buf,
                layer,
                &self.workspace.hidden_a,
                &self.workspace.hidden_b,
                &self.workspace.mask,
                batch,
                max_seq,
            )?;

            // FFN: hidden_b -> hidden_a
            self.encode_ffn(
                &cmd_buf,
                &layer.ffn,
                &self.workspace.hidden_b,
                &self.workspace.hidden_a,
                batch_seq,
            )?;
        }

        // --- CLS pooling + L2 normalize (compute encoder) ---
        {
            let enc = new_encoder(&cmd_buf)?;

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

            enc.endEncoding();
        }

        // Execute and wait
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
        // FP16 element-wise variants
        assert!(pipelines.layer_norm_f16.maxTotalThreadsPerThreadgroup() > 0);
        assert!(pipelines.gelu_f16.maxTotalThreadsPerThreadgroup() > 0);
        assert!(
            pipelines
                .fused_bias_gelu_f16
                .maxTotalThreadsPerThreadgroup()
                > 0
        );
        assert!(
            pipelines
                .fused_bias_residual_f16
                .maxTotalThreadsPerThreadgroup()
                > 0
        );
        assert!(
            pipelines
                .fused_residual_layernorm_f16
                .maxTotalThreadsPerThreadgroup()
                > 0
        );
        assert!(pipelines.add_bias_f16.maxTotalThreadsPerThreadgroup() > 0);
        assert!(pipelines.add_embeddings_f16.maxTotalThreadsPerThreadgroup() > 0);
        assert!(
            pipelines
                .embedding_lookup_f16
                .maxTotalThreadsPerThreadgroup()
                > 0
        );
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

    /// Helper: create a Metal buffer from f32 data stored as FP16 (half).
    ///
    /// Converts each f32 to f16 using `half::f16` and uploads to GPU.
    fn make_f16_buffer(
        device: &ProtocolObject<dyn MTLDevice>,
        data: &[f32],
    ) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        let halfs: Vec<u16> = data
            .iter()
            .map(|&v| half::f16::from_f32(v).to_bits())
            .collect();
        let size = (halfs.len() * core::mem::size_of::<u16>()) as NSUInteger;
        unsafe {
            device.newBufferWithBytes_length_options(
                NonNull::new(halfs.as_ptr() as *mut c_void).unwrap(),
                size,
                MTLResourceOptions::StorageModeShared,
            )
        }
        .expect("failed to create Metal FP16 buffer")
    }

    /// Helper: create a zero-initialized Metal buffer of `n` FP16 elements.
    fn make_zero_f16_buffer(
        device: &ProtocolObject<dyn MTLDevice>,
        n: usize,
    ) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        let size = (n * core::mem::size_of::<u16>()) as NSUInteger;
        device
            .newBufferWithLength_options(size, MTLResourceOptions::StorageModeShared)
            .expect("failed to create Metal FP16 buffer")
    }

    /// Helper: read FP16 values from a Metal buffer, returning as f32.
    fn read_f16_buffer(buffer: &ProtocolObject<dyn MTLBuffer>, n: usize) -> Vec<f32> {
        let raw =
            unsafe { core::slice::from_raw_parts(buffer.contents().as_ptr() as *const u16, n) };
        raw.iter()
            .map(|&bits| half::f16::from_bits(bits).to_f32())
            .collect()
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

    #[test]
    fn metal_gemm_fp16_mixed_precision() {
        // Verify FP16 mixed-precision GEMM matches FP32 within half-precision tolerance.
        // Uses BGE-small dimensions: M=64, N=384, K=384, transB=true.
        let device = create_device().unwrap();
        let queue = create_queue(&device).unwrap();
        let lib = compile_library(&device, super::super::metal_kernels::KERNELS).unwrap();
        let f32_to_f16_pipeline = create_pipeline(&device, &lib, "f32_to_f16_kernel").unwrap();
        let gemm_lib = compile_library(&device, super::super::metal_kernels::GEMM_KERNEL).unwrap();
        let fp16_pipeline = create_pipeline(&device, &gemm_lib, "gemm_fp16_kernel").unwrap();

        let m: u32 = 64;
        let n: u32 = 384;
        let k: u32 = 384;

        let a_data: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();
        let b_data: Vec<f32> = (0..n * k).map(|i| ((i % 13) as f32 - 6.0) * 0.01).collect();

        let a_buf = make_f32_buffer(&device, &a_data);
        let b_f32_buf = make_f32_buffer(&device, &b_data);

        // Convert B to FP16 on GPU using the f32_to_f16 kernel
        let b_num_floats = (n * k) as usize;
        let b_fp16_buf = device
            .newBufferWithLength_options(b_num_floats * 2, MTLResourceOptions::StorageModeShared)
            .unwrap();
        {
            let cmd = queue.commandBuffer().unwrap();
            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(&f32_to_f16_pipeline);
            set_buffer(&enc, &b_fp16_buf, 0, 0);
            set_buffer(&enc, &b_f32_buf, 0, 1);
            set_i32_param(&enc, b_num_floats as i32, 2);
            dispatch_1d(&enc, &f32_to_f16_pipeline, b_num_floats);
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
        }

        let buf_elems = m as usize * n as usize;
        let c_buf = make_zero_buffer(&device, buf_elems);

        let cmd_buf = queue.commandBuffer().unwrap();
        let encoder = cmd_buf.computeCommandEncoder().unwrap();

        dispatch_gemm_fp16(
            &encoder,
            &fp16_pipeline,
            &a_buf,
            0,
            &b_fp16_buf,
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

        // FP16 mixed precision introduces rounding — allow ~1e-2 tolerance
        // (half precision has ~3 decimal digits of mantissa)
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

        // Also verify overall L2 error is small
        let mut sum_sq_err = 0.0_f64;
        let mut sum_sq_ref = 0.0_f64;
        for (g, e) in result.iter().zip(expected.iter()) {
            sum_sq_err += ((*g as f64) - (*e as f64)).powi(2);
            sum_sq_ref += (*e as f64).powi(2);
        }
        let rel_err = (sum_sq_err / sum_sq_ref.max(1e-30)).sqrt();
        assert!(rel_err < 1e-2, "relative L2 error too large: {rel_err:.6}");
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

    /// Helper: create a command buffer for MPS-aware operations. Commits and waits.
    fn run_cmd_buf<F>(queue: &ProtocolObject<dyn MTLCommandQueue>, f: F)
    where
        F: FnOnce(&ProtocolObject<dyn MTLCommandBuffer>),
    {
        let cmd_buf = queue.commandBuffer().unwrap();
        f(&cmd_buf);
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
            run_cmd_buf(&backend.queue, |cmd_buf| {
                backend
                    .encode_attention(
                        cmd_buf,
                        &backend.model.layers[0],
                        &backend.workspace.hidden_a,
                        &backend.workspace.hidden_b,
                        &backend.workspace.mask,
                        batch,
                        max_seq,
                    )
                    .unwrap();
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
            run_cmd_buf(&backend.queue, |cmd_buf| {
                backend
                    .encode_ffn(
                        cmd_buf,
                        &backend.model.layers[0].ffn,
                        &backend.workspace.hidden_b,
                        &backend.workspace.hidden_a,
                        batch_seq,
                    )
                    .unwrap();
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

    /// Sweep early exit layers and MRL truncation dims.
    /// Measures cosine similarity vs full N-layer embedding.
    fn run_early_exit_sweep(model_repo: &str) {
        let mut backend = MetalBackend::load(model_repo, &DeviceHint::Auto).unwrap();
        let hd = backend.hidden_size as usize;
        let num_layers = backend.model.layers.len();
        eprintln!(
            "Model: {model_repo}, hidden={hd}, layers={num_layers}, variant={:?}",
            backend.variant
        );
        // Use generic token IDs that work for both BGE and NomicBert tokenizers
        // (CLS=101/1, content tokens, SEP=102/2)
        let enc = Encoding {
            input_ids: vec![
                1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 2,
            ],
            attention_mask: vec![1; 14],
            token_type_ids: vec![0; 14],
        };

        backend.max_layers = None;
        let full = backend.embed_batch(&[enc.clone()]).unwrap();
        let full_emb = &full[0];
        eprintln!(
            "Full embedding: {hd} dims, L2={:.4}\n",
            full_emb.iter().map(|x| x * x).sum::<f32>().sqrt()
        );

        // --- Early exit sweep ---
        eprintln!("=== Early Exit (cosine vs full {num_layers}-layer) ===");
        let layer_steps: Vec<usize> = (1..=num_layers).collect();
        for &layers in &layer_steps {
            backend.max_layers = Some(layers);
            let result = backend.embed_batch(&[enc.clone()]).unwrap();
            let emb = &result[0];
            let cos: f32 = emb.iter().zip(full_emb).map(|(a, b)| a * b).sum();
            eprintln!("  layers={layers:>2}: cosine={cos:.6}");
        }

        // --- Combined: early exit + truncation ---
        let mrl_dims: Vec<usize> = [64, 128, 256, hd / 2, hd]
            .into_iter()
            .filter(|&d| d <= hd)
            .collect::<std::collections::BTreeSet<usize>>()
            .into_iter()
            .collect();
        eprintln!("\n=== Combined: Early Exit + MRL Truncation ===");
        eprintln!("  (cosine vs full-{num_layers}-layer-{hd}-dim)");
        for &layers in &[1, 2, 4, 6, num_layers / 2, num_layers] {
            if layers > num_layers {
                continue;
            }
            backend.max_layers = Some(layers);
            let result = backend.embed_batch(&[enc.clone()]).unwrap();
            let emb = &result[0];
            for &dims in &mrl_dims {
                let trunc: Vec<f32> = emb[..dims].to_vec();
                let t_norm: f32 = trunc.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
                let f_norm: f32 = full_emb[..dims]
                    .iter()
                    .map(|x| x * x)
                    .sum::<f32>()
                    .sqrt()
                    .max(1e-12);
                let cos: f32 = trunc
                    .iter()
                    .zip(&full_emb[..dims])
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
                    / (t_norm * f_norm);
                eprintln!("  layers={layers:>2} dims={dims:>3}: cosine={cos:.6}");
            }
        }
    }

    #[test]
    #[ignore = "requires model download; run with --nocapture"]
    fn early_exit_and_mrl_quality() {
        run_early_exit_sweep("BAAI/bge-small-en-v1.5");
    }

    #[test]
    #[ignore = "requires model download; run with --nocapture"]
    fn early_exit_coderank() {
        run_early_exit_sweep("nomic-ai/CodeRankEmbed");
    }
}
