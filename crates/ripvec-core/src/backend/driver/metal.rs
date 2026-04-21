//! Metal GPU compute driver for Apple Silicon.
//!
//! Implements the [`Driver`] trait using Metal compute pipelines (custom MSL
//! kernels) and MPS `MPSMatrixMultiplication` for weight GEMMs. Each driver
//! method creates its own command buffer, encodes operations, commits, and
//! waits for completion.
//!
//! # Design
//!
//! - **`MetalTensor`**: wraps a `Retained<ProtocolObject<dyn MTLBuffer>>` with
//!   a byte offset, enabling sub-buffer views into weight buffers.
//! - **Per-call command buffers**: each trait method is self-contained. The
//!   architecture layer can batch operations later via a command-buffer
//!   batching wrapper.
//! - **Reuses MSL sources**: compiles kernels from [`super::super::metal_kernels`].

use std::collections::HashMap;
use std::path::Path;

use objc2::AnyThread;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSString, NSUInteger, ns_string};
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLLibrary, MTLResourceOptions, MTLSize,
};
use objc2_metal_performance_shaders::{
    MPSDataType, MPSMatrix, MPSMatrixDescriptor, MPSMatrixMultiplication,
};
use safetensors::SafeTensors;

use super::{BatchInputs, Driver};
use crate::backend::Encoding;
use crate::backend::arch::classic_bert::{
    ClassicBertArch, ClassicBertLayerWeights, ClassicBertWeights,
};
use crate::backend::arch::modern_bert::{
    ModernBertArch, ModernBertLayerWeights, ModernBertWeights, RopeCache,
};

// ---------------------------------------------------------------------------
// CoreGraphics linkage (required for MTLCreateSystemDefaultDevice)
// ---------------------------------------------------------------------------

#[expect(unsafe_code, reason = "extern block required for CoreGraphics linkage")]
#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}

// ---------------------------------------------------------------------------
// MetalTensor
// ---------------------------------------------------------------------------

/// A tensor on Metal -- a buffer reference with a byte offset.
///
/// Multiple `MetalTensor` values may reference the same underlying
/// `MTLBuffer` at different offsets (e.g. weight sub-tensors within a
/// single memory-mapped safetensors file).
pub struct MetalTensor {
    /// The backing Metal buffer (FP32).
    pub buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Byte offset into `buffer` where this tensor's data starts.
    pub offset: usize,
    /// Optional FP16 version of this tensor's data (for MPS weight GEMMs).
    /// Created lazily by [`MetalDriver::ensure_fp16`]. When present, `gemm`
    /// uses this buffer with `MPSDataType::Float16` for the weight matrix,
    /// halving memory bandwidth.
    pub fp16: std::cell::RefCell<Option<Retained<ProtocolObject<dyn MTLBuffer>>>>,
    /// Optional block_q8_0 quantized weights (scales embedded in blocks).
    pub q8: std::cell::RefCell<Option<Retained<ProtocolObject<dyn MTLBuffer>>>>,
}

impl MetalTensor {
    /// Create a new tensor with no FP16 companion.
    fn new(buffer: Retained<ProtocolObject<dyn MTLBuffer>>, offset: usize) -> Self {
        Self {
            buffer,
            offset,
            fp16: std::cell::RefCell::new(None),
            q8: std::cell::RefCell::new(None),
        }
    }
}

/// GPU-resident TurboQuant corpus buffers. Uploaded once, reused for all queries.
pub struct TurboQuantGpuCorpus {
    radii: Retained<ProtocolObject<dyn MTLBuffer>>,
    indices: Retained<ProtocolObject<dyn MTLBuffer>>,
}

// SAFETY: Metal shared-mode buffers are safe to hold across threads.
#[expect(unsafe_code, reason = "Metal StorageModeShared is thread-safe")]
unsafe impl Send for TurboQuantGpuCorpus {}
#[expect(unsafe_code, reason = "Metal StorageModeShared is thread-safe")]
unsafe impl Sync for TurboQuantGpuCorpus {}

// SAFETY: Metal buffers use `StorageModeShared` on Apple Silicon (unified memory).
// The Metal framework guarantees thread-safe access to buffer contents once the
// command buffer that wrote the data has completed. Weight tensors are read-only
// after loading, so sharing across threads is safe.
#[expect(unsafe_code, reason = "Metal shared-mode buffers are thread-safe")]
unsafe impl Send for MetalTensor {}
#[expect(unsafe_code, reason = "Metal shared-mode buffers are thread-safe")]
unsafe impl Sync for MetalTensor {}

// ---------------------------------------------------------------------------
// KernelPipelines
// ---------------------------------------------------------------------------

/// Pre-compiled Metal compute pipeline states for all BERT inference kernels.
///
/// Created once at driver init time by compiling the MSL source and extracting
/// each named function.
struct KernelPipelines {
    /// Embedding table lookup.
    embedding_lookup: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Add embedding table values to existing output.
    add_embeddings: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Layer normalization with threadgroup reduction.
    layer_norm: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// GELU activation (tanh approximation, in-place).
    gelu: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Fused scale + attention-mask + softmax.
    fused_scale_mask_softmax: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Fused residual add + layer norm.
    fused_residual_layernorm: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Fused bias + GELU activation for `ClassicBert` FFN.
    fused_bias_gelu: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Fused bias + residual add for output projections.
    fused_bias_residual: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Split QKV `[batch*seq, 3*hidden]` into Q,K,V `[batch*num_heads, seq, head_dim]`.
    qkv_split: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Reshape attention output `[batch*num_heads, seq, head_dim]` to `[batch*seq, hidden]`.
    attn_reshape: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// CLS pooling (extract row 0 per batch element).
    cls_pool: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Mean pooling: attention-mask-weighted average of all tokens.
    mean_pool: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// L2 normalize each row.
    l2_normalize: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Build float attention mask from int mask.
    build_attn_mask: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Add bias in-place: `x[idx] += bias[idx % cols]`.
    add_bias: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Two-input `SwiGLU`: `output = value * silu(gate)` with separate buffers.
    swiglu_two_input: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Split `[rows, 2*cols]` into two `[rows, cols]` halves.
    split_gate_value: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Two-input `GeGLU`: `output = gelu(value) * gate` with separate buffers.
    geglu: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// `TurboQuant` compressed scan: one thread per vector, centroid table lookup.
    turboquant_scan: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Banded Q@K^T: sliding-window attention scores `[batch_heads, seq, window]`.
    banded_qk: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Banded softmax over window dimension.
    banded_softmax: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Banded scores@V: weighted sum from banded attention.
    banded_sv: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Residual add without bias: `output = hidden + residual`.
    residual_add: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Fused scale + padding mask + sliding window mask + softmax.
    fused_scale_mask_softmax_windowed: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// `RoPE` with pre-computed cos/sin tables.
    rope_cached: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Batched GEMM: same kernel with z-dimension for batch/head index.
    gemm_batched: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Convert FP32 to FP16.
    f32_to_f16: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Scatter flat `[total_tokens, dim]` into padded `[batch, max_seq, dim]`.
    pad_to_batch: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Gather real tokens from padded `[batch, max_seq, dim]` back to flat.
    unpad_from_batch: Retained<ProtocolObject<dyn MTLComputePipelineState>>,

    // --- FP16 attention-path pipeline states ---
    /// FP16 layer normalization.
    layer_norm_f16: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// FP16 fused residual add + layer norm.
    fused_residual_layernorm_f16: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Convert FP16 back to FP32.
    f16_to_f32: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// FP16 `GeGLU` gated activation.
    geglu_f16: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// FP16 split `[rows, 2*cols]` into two halves.
    split_gate_value_f16: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// FP16 residual add (no bias).
    residual_add_f16: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// FP16 fused scale + mask + softmax.
    fused_scale_mask_softmax_f16: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// FP16 fused scale + mask + windowed softmax.
    fused_scale_mask_softmax_windowed_f16: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// FP16 QKV split.
    qkv_split_f16: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// FP16 attention reshape.
    attn_reshape_f16: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// FP16 pad to batch.
    pad_to_batch_f16: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// FP16 unpad from batch.
    unpad_from_batch_f16: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// FP16 `RoPE`.
    rope_encode_f16: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Full FP16 batched GEMM (half A, half B, half C).
    gemm_batched_f16: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Mixed-precision GEMM: FP32 activations × FP16 weights → FP32 output.
    /// Native simdgroup ops, no MFA wrapper.
    gemm_f16w_f32a: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// INT8-weight GEMM: FP16 activations × INT8 weights + per-row scales → FP16 output.
    gemm_q8w: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl KernelPipelines {
    /// Compile all MSL kernels and create pipeline states.
    fn compile(device: &ProtocolObject<dyn MTLDevice>) -> crate::Result<Self> {
        let library = compile_library(device, crate::backend::metal_kernels::KERNELS)?;
        let p = |name: &str| create_pipeline(device, &library, name);

        let gemm_library = compile_library(device, crate::backend::metal_kernels::GEMM_KERNEL)?;
        let native_gemm_library =
            compile_library(device, crate::backend::metal_kernels::NATIVE_GEMM_KERNEL)?;

        Ok(Self {
            embedding_lookup: p("embedding_lookup_kernel")?,
            add_embeddings: p("add_embeddings_kernel")?,
            layer_norm: p("layer_norm_kernel")?,
            gelu: p("gelu_kernel")?,
            fused_scale_mask_softmax: p("fused_scale_mask_softmax_kernel")?,
            fused_residual_layernorm: p("fused_residual_layernorm_kernel")?,
            fused_bias_gelu: p("fused_bias_gelu_kernel")?,
            fused_bias_residual: p("fused_bias_residual_kernel")?,
            qkv_split: p("qkv_split_kernel")?,
            attn_reshape: p("attn_reshape_kernel")?,
            cls_pool: p("cls_pool_kernel")?,
            mean_pool: p("mean_pool_kernel")?,
            l2_normalize: p("l2_normalize_kernel")?,
            build_attn_mask: p("build_attn_mask_kernel")?,
            add_bias: p("add_bias_kernel")?,
            swiglu_two_input: p("swiglu_two_input_kernel")?,
            split_gate_value: p("split_gate_value_kernel")?,
            geglu: p("geglu_kernel")?,
            turboquant_scan: p("turboquant_scan_kernel")?,
            banded_qk: p("banded_qk_kernel")?,
            banded_softmax: p("banded_softmax_kernel")?,
            banded_sv: p("banded_sv_kernel")?,
            residual_add: p("residual_add_kernel")?,
            fused_scale_mask_softmax_windowed: p("fused_scale_mask_softmax_windowed_kernel")?,
            rope_cached: p("rope_cached_kernel")?,
            gemm_batched: create_pipeline(device, &gemm_library, "gemm_batched_kernel")?,
            f32_to_f16: p("f32_to_f16_kernel")?,
            pad_to_batch: p("pad_to_batch_kernel")?,
            unpad_from_batch: p("unpad_from_batch_kernel")?,
            // FP16 attention-path pipelines
            layer_norm_f16: p("layer_norm_f16_kernel")?,
            fused_residual_layernorm_f16: p("fused_residual_layernorm_f16_kernel")?,
            f16_to_f32: p("f16_to_f32_kernel")?,
            geglu_f16: p("geglu_f16_kernel")?,
            split_gate_value_f16: p("split_gate_value_f16_kernel")?,
            residual_add_f16: p("residual_add_f16_kernel")?,
            fused_scale_mask_softmax_f16: p("fused_scale_mask_softmax_f16_kernel")?,
            fused_scale_mask_softmax_windowed_f16: p(
                "fused_scale_mask_softmax_windowed_f16_kernel",
            )?,
            qkv_split_f16: p("qkv_split_f16_kernel")?,
            attn_reshape_f16: p("attn_reshape_f16_kernel")?,
            pad_to_batch_f16: p("pad_to_batch_f16_kernel")?,
            unpad_from_batch_f16: p("unpad_from_batch_f16_kernel")?,
            rope_encode_f16: p("rope_encode_f16_kernel")?,
            gemm_batched_f16: create_pipeline(device, &gemm_library, "gemm_batched_f16_kernel")?,
            gemm_f16w_f32a: create_pipeline(device, &native_gemm_library, "gemm_f16w_f32a_kernel")?,
            gemm_q8w: create_pipeline(device, &native_gemm_library, "gemm_q8w_f16a_kernel")?,
        })
    }
}

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
    let queue = device
        .newCommandQueue()
        .ok_or_else(|| crate::Error::Metal("failed to create command queue".into()))?;
    queue.setLabel(Some(ns_string!("ripvec-compute")));
    Ok(queue)
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

/// Create a compute command encoder from a command buffer.
fn new_encoder(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
) -> crate::Result<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>> {
    cmd_buf
        .computeCommandEncoder()
        .ok_or_else(|| crate::Error::Metal("failed to create compute encoder".into()))
}

/// Create a new command buffer from the queue.
fn new_command_buffer(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
) -> crate::Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
    queue
        .commandBuffer()
        .ok_or_else(|| crate::Error::Metal("failed to create command buffer".into()))
}

// ---------------------------------------------------------------------------
// Dispatch helpers
// ---------------------------------------------------------------------------

/// Dispatch a 1D compute kernel over `n` elements.
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

/// Dispatch a batched GEMM with `batch_count` slices at strided offsets.
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

/// Set a `constant int&` parameter at the given buffer index.
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

/// Set a `constant uint&` parameter at the given buffer index.
#[expect(
    unsafe_code,
    clippy::borrow_as_ptr,
    reason = "Metal setBytes requires unsafe FFI with raw ptr"
)]
fn set_u32_param(encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>, value: u32, index: usize) {
    unsafe {
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::new(&value as *const u32 as *mut _).expect("non-null const ptr"),
            core::mem::size_of::<u32>(),
            index,
        );
    }
}

/// Set a `constant float&` parameter at the given buffer index.
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

/// Dispatch a GEMM using Apple's MPS `MPSMatrixMultiplication`.
///
/// Encodes C[M,N] = alpha * A[M,K] * B[K,N] (or B^T when `trans_b`) + beta * C
/// directly onto the command buffer.
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
    // Output type matches A type (FP16 pipeline stays FP16, FP32 stays FP32).
    let c_data_type = a_data_type;

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
    let c_elem_size = if c_data_type == MPSDataType::Float16 {
        2
    } else {
        4
    };

    let a_row_bytes = k * a_elem_size;
    let c_row_bytes = n * c_elem_size;

    let (b_rows, b_cols, b_row_bytes) = if trans_b {
        (n, k, k * b_elem_size)
    } else {
        (k, n, n * b_elem_size)
    };

    unsafe {
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
            c_data_type,
        );

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

        let gemm = MPSMatrixMultiplication::initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta(
            MPSMatrixMultiplication::alloc(),
            device,
            false,
            trans_b,
            m as NSUInteger,
            n as NSUInteger,
            k as NSUInteger,
            1.0,
            0.0,
        );

        gemm.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(
            cmd_buf, &mat_a, &mat_b, &mat_c,
        );
    }
}

// ---------------------------------------------------------------------------
// Buffer allocation helper
// ---------------------------------------------------------------------------

/// Allocate a Metal buffer of `n` floats with shared storage.
fn alloc_f32_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    n: usize,
) -> crate::Result<Retained<ProtocolObject<dyn MTLBuffer>>> {
    let size = (n * core::mem::size_of::<f32>()) as NSUInteger;
    device
        .newBufferWithLength_options(size, MTLResourceOptions::StorageModeShared)
        .ok_or_else(|| crate::Error::Metal(format!("buffer alloc failed ({n} floats)")))
}

/// Allocate a Metal buffer of `n` half-precision (FP16) elements with shared storage.
fn alloc_f16_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    n: usize,
) -> crate::Result<Retained<ProtocolObject<dyn MTLBuffer>>> {
    let size = (n * 2) as NSUInteger; // 2 bytes per f16
    device
        .newBufferWithLength_options(size, MTLResourceOptions::StorageModeShared)
        .ok_or_else(|| crate::Error::Metal(format!("f16 buffer alloc failed ({n} halves)")))
}

/// Create a Metal buffer from i32 data.
#[expect(unsafe_code, reason = "newBufferWithBytes requires unsafe FFI")]
fn make_i32_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    data: &[i32],
) -> crate::Result<Retained<ProtocolObject<dyn MTLBuffer>>> {
    let size = core::mem::size_of_val(data) as NSUInteger;
    unsafe {
        device.newBufferWithBytes_length_options(
            std::ptr::NonNull::new(data.as_ptr() as *mut _)
                .ok_or_else(|| crate::Error::Metal("null input data".into()))?,
            size,
            MTLResourceOptions::StorageModeShared,
        )
    }
    .ok_or_else(|| crate::Error::Metal("input buffer alloc failed".into()))
}

/// Create a Metal buffer from u8 data.
#[expect(unsafe_code, reason = "newBufferWithBytes requires unsafe FFI")]
fn make_u8_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    data: &[u8],
) -> crate::Result<Retained<ProtocolObject<dyn MTLBuffer>>> {
    let size = data.len() as NSUInteger;
    unsafe {
        device.newBufferWithBytes_length_options(
            std::ptr::NonNull::new(data.as_ptr() as *mut _)
                .ok_or_else(|| crate::Error::Metal("null input data".into()))?,
            size,
            MTLResourceOptions::StorageModeShared,
        )
    }
    .ok_or_else(|| crate::Error::Metal("u8 buffer alloc failed".into()))
}

/// Create a Metal buffer from f32 data.
#[expect(unsafe_code, reason = "newBufferWithBytes requires unsafe FFI")]
fn make_f32_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    data: &[f32],
) -> crate::Result<Retained<ProtocolObject<dyn MTLBuffer>>> {
    let size = core::mem::size_of_val(data) as NSUInteger;
    unsafe {
        device.newBufferWithBytes_length_options(
            std::ptr::NonNull::new(data.as_ptr() as *mut _)
                .ok_or_else(|| crate::Error::Metal("null input data".into()))?,
            size,
            MTLResourceOptions::StorageModeShared,
        )
    }
    .ok_or_else(|| crate::Error::Metal("f32 buffer alloc failed".into()))
}

// ---------------------------------------------------------------------------
// MetalDriver
// ---------------------------------------------------------------------------

/// Metal GPU compute driver using MPS for weight GEMMs and custom MSL kernels.
///
/// Supports two modes:
/// - **Per-call** (default): each trait method creates its own command buffer.
/// - **Batched**: call [`begin_batch`] before a forward pass. All operations
///   encode into ONE command buffer. [`end_batch`] commits and waits once.
///   This eliminates per-call overhead (~220 commits → 1) and enables MPS GEMM.
///
/// # Thread safety
///
/// Metal device and command queue are thread-safe. The batch command buffer
/// uses `RefCell` — batched mode is single-threaded (architectures are `&self`).
pub struct MetalDriver {
    /// Metal GPU device handle.
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    /// Command queue for submitting GPU work.
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    /// Pre-compiled MSL pipeline states.
    kernels: KernelPipelines,
    /// Shared command buffer for batched mode. `None` = per-call mode.
    batch_cmd: std::cell::RefCell<Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>>,
    /// Persistent compute encoder for batched mode. Reused across run_compute
    /// calls to avoid encoder creation overhead.
    /// Closed before MPS/blit calls (which encode directly to command buffer).
    batch_enc: std::cell::RefCell<Option<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>>>,
    /// Whether the persistent encoder has been used since last creation.
    /// Avoids unnecessary flush when consecutive MPS calls have no compute between them.
    enc_dirty: std::cell::Cell<bool>,
    /// FP32 buffer pool: reuse Metal buffers across forward passes.
    pool: std::cell::RefCell<Vec<Retained<ProtocolObject<dyn MTLBuffer>>>>,
    pool_cursor: std::cell::Cell<usize>,
    /// FP16 buffer pool: separate from FP32 to prevent mixed-type reuse.
    pool_f16: std::cell::RefCell<Vec<Retained<ProtocolObject<dyn MTLBuffer>>>>,
    pool_f16_cursor: std::cell::Cell<usize>,
}

impl MetalDriver {
    /// Create a new Metal driver, initializing the device, queue, and kernels.
    ///
    /// # Errors
    ///
    /// Returns an error if no Metal device is available, the command queue
    /// cannot be created, or MSL kernel compilation fails.
    pub fn new() -> crate::Result<Self> {
        let device = create_device()?;
        let queue = create_queue(&device)?;
        let kernels = KernelPipelines::compile(&device)?;
        Ok(Self {
            device,
            queue,
            kernels,
            batch_cmd: std::cell::RefCell::new(None),
            batch_enc: std::cell::RefCell::new(None),
            enc_dirty: std::cell::Cell::new(false),
            pool: std::cell::RefCell::new(Vec::with_capacity(150)),
            pool_cursor: std::cell::Cell::new(0),
            pool_f16: std::cell::RefCell::new(Vec::with_capacity(350)),
            pool_f16_cursor: std::cell::Cell::new(0),
        })
    }

    /// Begin batched mode: all subsequent Driver operations encode into ONE
    /// command buffer instead of creating individual ones.
    ///
    /// Call [`end_batch`] to commit and wait. This eliminates ~220 per-call
    /// commits per forward pass → 1 commit.
    pub fn begin_batch(&self) -> crate::Result<()> {
        let cmd = new_command_buffer(&self.queue)?;
        cmd.setLabel(Some(ns_string!("forward-pass")));
        *self.batch_cmd.borrow_mut() = Some(cmd);
        self.pool_cursor.set(0);
        self.pool_f16_cursor.set(0);
        Ok(())
    }

    /// Flush the current command buffer and start a new one WITHOUT
    /// resetting pool cursors. Use mid-forward-pass to prevent GPU timeouts
    /// on models with many layers (e.g., ModernBERT 22 layers).
    ///
    /// All pool tensors remain valid — only the command buffer is replaced.
    pub fn flush_batch(&self) -> crate::Result<()> {
        if let Some(enc) = self.batch_enc.borrow_mut().take() {
            enc.endEncoding();
        }
        self.enc_dirty.set(false);
        if let Some(cmd) = self.batch_cmd.borrow_mut().take() {
            cmd.commit();
            cmd.waitUntilCompleted();
            let status = cmd.status();
            if status == objc2_metal::MTLCommandBufferStatus::Error {
                if let Some(err) = cmd.error() {
                    return Err(crate::Error::Metal(format!("GPU flush error: {err}")));
                }
                return Err(crate::Error::Metal("GPU flush error (unknown)".into()));
            }
        }
        // Start a new command buffer — pool cursors NOT reset
        let cmd = new_command_buffer(&self.queue)?;
        cmd.setLabel(Some(ns_string!("forward-pass")));
        *self.batch_cmd.borrow_mut() = Some(cmd);
        Ok(())
    }

    /// End batched mode: commit the shared command buffer and wait.
    ///
    /// Returns an error if no batch is active.
    pub fn end_batch(&self) -> crate::Result<()> {
        // Force-close any open compute encoder before committing
        if let Some(enc) = self.batch_enc.borrow_mut().take() {
            enc.endEncoding();
        }
        self.enc_dirty.set(false);
        let cmd =
            self.batch_cmd.borrow_mut().take().ok_or_else(|| {
                crate::Error::Metal("end_batch called without begin_batch".into())
            })?;
        cmd.commit();
        cmd.waitUntilCompleted();

        // Check for GPU errors
        let status = cmd.status();
        if status == objc2_metal::MTLCommandBufferStatus::Error {
            if let Some(err) = cmd.error() {
                return Err(crate::Error::Metal(format!("GPU batch error: {err}")));
            }
            return Err(crate::Error::Metal("GPU batch error (unknown)".into()));
        }
        Ok(())
    }

    /// Borrow the underlying Metal device handle.
    ///
    /// Needed by weight-loading code that creates zero-copy buffers from
    /// memory-mapped safetensors files.
    pub fn device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }

    /// Allocate a [`MetalTensor`] of `n` floats, reusing a pooled buffer if
    /// possible.
    ///
    /// In batched mode: checks the pool for a buffer with sufficient capacity.
    /// On the first forward pass, buffers are allocated fresh and added to the
    /// pool. On subsequent passes (after `begin_batch` resets the cursor),
    /// the same buffers are reused — zero allocation overhead.
    ///
    /// # Errors
    ///
    /// Returns an error if Metal buffer allocation fails.
    pub fn alloc_tensor(&self, n: usize) -> crate::Result<MetalTensor> {
        let needed = n * core::mem::size_of::<f32>();
        let cursor = self.pool_cursor.get();
        let mut pool = self.pool.borrow_mut();

        // Reuse any buffer that's large enough. MPS uses descriptor dimensions
        // (M, N, K) for dispatch — NOT buffer.length/rowBytes. Oversized buffers
        // are safe per Apple's documented API contract.
        if cursor < pool.len() && pool[cursor].length() >= needed {
            let buffer = pool[cursor].clone();
            self.pool_cursor.set(cursor + 1);
            return Ok(MetalTensor::new(buffer, 0));
        }

        if cursor < pool.len() {
            // Buffer too small — skip slot, allocate fresh WITHOUT replacing.
            // Replacing would destroy a large buffer that a different layer needs
            // (reset_layer_workspace resets cursor, so layers share pool slots).
            self.pool_cursor.set(cursor + 1);
            let buffer = alloc_f32_buffer(&self.device, n)?;
            return Ok(MetalTensor::new(buffer, 0));
        }

        // Pool exhausted — allocate fresh and extend pool.
        let buffer = alloc_f32_buffer(&self.device, n)?;
        pool.push(buffer.clone());
        self.pool_cursor.set(cursor + 1);
        Ok(MetalTensor::new(buffer, 0))
    }

    /// Allocate a [`MetalTensor`] of `n` half-precision (FP16) elements.
    ///
    /// Uses the same pool as `alloc_tensor` for buffer reuse. The returned
    /// tensor holds a buffer with `n * 2` bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if Metal buffer allocation fails.
    pub fn alloc_f16_tensor(&self, n: usize) -> crate::Result<MetalTensor> {
        // Separate FP16 pool — prevents mixed-type reuse with FP32 buffers.
        let needed = n * 2; // 2 bytes per half
        let cursor = self.pool_f16_cursor.get();
        let mut pool = self.pool_f16.borrow_mut();

        if cursor < pool.len() && pool[cursor].length() >= needed {
            let buffer = pool[cursor].clone();
            self.pool_f16_cursor.set(cursor + 1);
            return Ok(MetalTensor::new(buffer, 0));
        }

        let buffer = alloc_f16_buffer(&self.device, n)?;
        if cursor < pool.len() {
            pool[cursor] = buffer.clone();
        } else {
            pool.push(buffer.clone());
        }
        self.pool_f16_cursor.set(cursor + 1);
        Ok(MetalTensor::new(buffer, 0))
    }

    /// Run `TurboQuant` compressed scan on the GPU.
    ///
    /// Dispatches one thread per vector. The centroid table (24 KB for d=768)
    /// is passed as a constant buffer — Metal places it in argument buffer
    /// memory, accessible at full bandwidth.
    ///
    /// At 100K vectors on M2 Max: ~50µs (vs ~33ms CPU scalar).
    /// Upload corpus data to GPU once. Returns opaque handles for `turboquant_scan_gpu`.
    pub fn turboquant_upload_corpus(
        &self,
        radii: &[f32],
        indices: &[u8],
    ) -> crate::Result<TurboQuantGpuCorpus> {
        Ok(TurboQuantGpuCorpus {
            radii: make_f32_buffer(&self.device, radii)?,
            indices: make_u8_buffer(&self.device, indices)?,
        })
    }

    /// Fast GPU scan using pre-uploaded corpus buffers. Only the centroid table
    /// (24 KB) is uploaded per query. The corpus data stays on GPU.
    #[expect(unsafe_code, reason = "Metal buffer readback")]
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "n_vectors/n_pairs/n_levels are small ML dimensions that fit in i32"
    )]
    pub fn turboquant_scan_gpu(
        &self,
        gpu_corpus: &TurboQuantGpuCorpus,
        centroid_q: &[f32],
        n_vectors: usize,
        n_pairs: usize,
        n_levels: usize,
    ) -> crate::Result<Vec<f32>> {
        let centroid_buf = make_f32_buffer(&self.device, centroid_q)?;
        let scores_buf = alloc_f32_buffer(&self.device, n_vectors)?;

        self.run_compute("turboquant-scan", |enc| {
            enc.setComputePipelineState(&self.kernels.turboquant_scan);
            set_buffer(enc, &gpu_corpus.radii, 0, 0);
            set_buffer(enc, &gpu_corpus.indices, 0, 1);
            set_buffer(enc, &centroid_buf, 0, 2);
            set_buffer(enc, &scores_buf, 0, 3);
            set_i32_param(enc, n_vectors as i32, 4);
            set_i32_param(enc, n_pairs as i32, 5);
            set_i32_param(enc, n_levels as i32, 6);
            dispatch_1d(enc, &self.kernels.turboquant_scan, n_vectors);
            Ok(())
        })?;

        let scores = unsafe {
            let ptr = scores_buf.contents().as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, n_vectors).to_vec()
        };
        Ok(scores)
    }

    /// Upload + scan in one call (convenience, re-uploads corpus each time).
    pub fn turboquant_scan(
        &self,
        radii: &[f32],
        indices: &[u8],
        centroid_q: &[f32],
        n_vectors: usize,
        n_pairs: usize,
        n_levels: usize,
    ) -> crate::Result<Vec<f32>> {
        let gpu = self.turboquant_upload_corpus(radii, indices)?;
        self.turboquant_scan_gpu(&gpu, centroid_q, n_vectors, n_pairs, n_levels)
    }

    /// Pre-convert a weight tensor to FP16 on the GPU.
    ///
    /// The FP16 buffer is stored inside the tensor's `fp16` field and used
    /// automatically by [`MetalDriver::gemm`] for the weight (B) matrix. The FP16 offset
    /// is `original_offset / 2` (half the bytes per element).
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "num_elements is a small ML dimension that fits in i32"
    )]
    pub fn ensure_fp16(&self, tensor: &MetalTensor, num_elements: usize) -> crate::Result<()> {
        if tensor.fp16.borrow().is_some() {
            return Ok(());
        }
        let fp16_size = (num_elements * 2) as NSUInteger; // 2 bytes per f16
        let fp16_buf = self
            .device
            .newBufferWithLength_options(fp16_size, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| crate::Error::Metal("fp16 buffer alloc failed".into()))?;

        // Convert on GPU using f32_to_f16 kernel
        self.run_compute("ensure-fp16", |enc| {
            enc.setComputePipelineState(&self.kernels.f32_to_f16);
            set_buffer(enc, &fp16_buf, 0, 0);
            set_buffer(enc, &tensor.buffer, tensor.offset, 1);
            set_i32_param(enc, num_elements as i32, 2);
            dispatch_1d(enc, &self.kernels.f32_to_f16, num_elements);
            Ok(())
        })?;

        *tensor.fp16.borrow_mut() = Some(fp16_buf);
        Ok(())
    }

    /// Execute a compute operation with a debug label for Metal System Trace.
    ///
    /// In batched mode: reuses a persistent encoder (created on first call,
    /// closed before MPS). This eliminates per-operation encoder overhead
    /// (~221 encoders → ~12 per forward pass, one per MPS boundary).
    /// In per-call mode: creates a new command buffer, commits, and waits.
    ///
    /// The `label` appears as a debug group in Metal System Trace / Instruments,
    /// replacing generic "Compute Command N" with meaningful operation names.
    fn run_compute<F>(&self, label: &str, f: F) -> crate::Result<()>
    where
        F: FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>) -> crate::Result<()>,
    {
        let ns_label = NSString::from_str(label);
        if self.batch_cmd.borrow().is_some() {
            // Ensure persistent encoder exists
            if self.batch_enc.borrow().is_none() {
                let batch = self.batch_cmd.borrow();
                let cmd_buf = batch.as_ref().unwrap();
                let enc = new_encoder(cmd_buf)?;
                drop(batch);
                *self.batch_enc.borrow_mut() = Some(enc);
            }
            let enc_ref = self.batch_enc.borrow();
            let enc = enc_ref.as_ref().unwrap();
            enc.pushDebugGroup(&ns_label);
            let result = f(enc);
            enc.popDebugGroup();
            result?;
            self.enc_dirty.set(true);
            Ok(())
        } else {
            let cmd_buf = new_command_buffer(&self.queue)?;
            cmd_buf.setLabel(Some(&ns_label));
            let enc = new_encoder(&cmd_buf)?;
            enc.pushDebugGroup(&ns_label);
            f(&enc)?;
            enc.popDebugGroup();
            enc.endEncoding();
            cmd_buf.commit();
            cmd_buf.waitUntilCompleted();
            Ok(())
        }
    }

    /// Close the persistent compute encoder if it has pending work.
    /// No-op if clean (avoids unnecessary encoder tear-down between
    /// consecutive MPS calls with no intervening compute).
    fn flush_compute_encoder(&self) {
        if self.enc_dirty.get() {
            if let Some(enc) = self.batch_enc.borrow_mut().take() {
                enc.endEncoding();
            }
            self.enc_dirty.set(false);
        }
    }

    /// Execute an MPS operation (encodes directly to command buffer).
    ///
    /// In batched mode: closes any open compute encoder first, then encodes
    /// MPS directly to the shared command buffer. A new compute encoder
    /// will be created on the next `run_compute` call.
    ///
    /// The `label` is used for the command buffer label in non-batch mode.
    fn run_mps<F>(&self, label: &str, f: F) -> crate::Result<()>
    where
        F: FnOnce(&ProtocolObject<dyn MTLCommandBuffer>) -> crate::Result<()>,
    {
        if self.batch_cmd.borrow().is_some() {
            // Must close compute encoder before MPS encoding
            self.flush_compute_encoder();
            let batch = self.batch_cmd.borrow();
            f(batch.as_ref().unwrap())?;
            Ok(())
        } else {
            let cmd_buf = new_command_buffer(&self.queue)?;
            cmd_buf.setLabel(Some(&NSString::from_str(label)));
            f(&cmd_buf)?;
            cmd_buf.commit();
            cmd_buf.waitUntilCompleted();
            Ok(())
        }
    }

    /// Load `ModernBERT` weights from a safetensors file into [`MetalTensor`]s.
    ///
    /// Memory-maps the file and wraps it as a single zero-copy Metal buffer via
    /// `newBufferWithBytesNoCopy`. Individual weight tensors are sub-views into
    /// this buffer at the byte offsets recorded in the safetensors header.
    ///
    /// Also pre-computes the two `RoPE` cos/sin caches (global theta=160000,
    /// local theta=10000) and allocates a zero-bias buffer.
    ///
    /// Returns `(arch, mmap)` — the caller **must** keep `mmap` alive as long as
    /// the arch's tensors are in use (the Metal buffer references the mmap'd pages).
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened/mapped, safetensors parsing
    /// fails, or any expected weight tensor is missing.
    #[expect(
        unsafe_code,
        reason = "mmap + newBufferWithBytesNoCopy require unsafe FFI"
    )]
    #[expect(
        clippy::too_many_lines,
        reason = "weight loading is a single logical unit mapping tensor names to fields"
    )]
    pub fn load_modern_bert_weights(
        &self,
        weights_path: &Path,
        config: &ModernBertConfig,
    ) -> crate::Result<(ModernBertArch<MetalTensor>, memmap2::Mmap)> {
        // 1. mmap the safetensors file
        let file = std::fs::File::open(weights_path).map_err(|e| crate::Error::Io {
            path: weights_path.display().to_string(),
            source: e,
        })?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| crate::Error::Io {
            path: weights_path.display().to_string(),
            source: e,
        })?;

        // 2. Create zero-copy Metal buffer
        let page_size: usize = 16384; // Apple Silicon 16KB pages
        let aligned_len = mmap.len().next_multiple_of(page_size);
        let weight_buffer = unsafe {
            self.device
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    std::ptr::NonNull::new(mmap.as_ptr() as *mut _)
                        .ok_or_else(|| crate::Error::Metal("mmap returned null pointer".into()))?,
                    aligned_len as NSUInteger,
                    MTLResourceOptions::StorageModeShared,
                    None,
                )
        }
        .ok_or_else(|| {
            crate::Error::Metal(
                "zero-copy buffer creation failed (pointer not page-aligned?)".into(),
            )
        })?;

        // 3. Parse safetensors header to get tensor offsets
        let tensors = SafeTensors::deserialize(&mmap)
            .map_err(|e| crate::Error::Metal(format!("safetensors parse: {e}")))?;

        let mmap_base = mmap.as_ptr() as usize;
        let mut refs: HashMap<String, (usize, Vec<usize>)> = HashMap::new();
        for (name, view) in tensors.tensors() {
            let offset = view.data().as_ptr() as usize - mmap_base;
            let shape: Vec<usize> = view.shape().to_vec();
            refs.insert(name.clone(), (offset, shape));
        }
        drop(tensors);

        // Helper: look up a required weight by name.
        let get_ref = |name: &str| -> crate::Result<(usize, &[usize])> {
            let (offset, shape) = refs
                .get(name)
                .ok_or_else(|| crate::Error::Metal(format!("missing weight: {name}")))?;
            Ok((*offset, shape.as_slice()))
        };

        // Helper: create a MetalTensor pointing into the weight buffer at a byte offset.
        let tensor_at =
            |offset: usize| -> MetalTensor { MetalTensor::new(weight_buffer.clone(), offset) };

        let hidden = config.hidden_size;
        let num_layers = config.num_hidden_layers;
        let num_heads = config.num_attention_heads;
        let head_dim = hidden / num_heads;
        let intermediate = config.intermediate_size;
        let global_attn_every_n = config.global_attn_every_n_layers;

        // 4. Build per-layer weights
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let qkv_offset = get_ref(&format!("layers.{i}.attn.Wqkv.weight"))?.0;
            let wo_offset = get_ref(&format!("layers.{i}.attn.Wo.weight"))?.0;
            let attn_norm_offset = if i == 0 {
                None // Layer 0 has no attn_norm (identity)
            } else {
                Some(get_ref(&format!("layers.{i}.attn_norm.weight"))?.0)
            };
            let wi_offset = get_ref(&format!("layers.{i}.mlp.Wi.weight"))?.0;
            let mlp_wo_offset = get_ref(&format!("layers.{i}.mlp.Wo.weight"))?.0;
            let mlp_norm_offset = get_ref(&format!("layers.{i}.mlp_norm.weight"))?.0;

            let is_global = i % global_attn_every_n == 0;

            layers.push(ModernBertLayerWeights {
                qkv_weight: tensor_at(qkv_offset),
                output_weight: tensor_at(wo_offset),
                attn_norm_weight: attn_norm_offset.map(&tensor_at),
                mlp_wi_weight: tensor_at(wi_offset),
                mlp_wo_weight: tensor_at(mlp_wo_offset),
                mlp_norm_weight: tensor_at(mlp_norm_offset),
                is_global,
            });
        }

        // 5. Embedding + final norm weights
        let tok_emb_offset = get_ref("embeddings.tok_embeddings.weight")?.0;
        let emb_norm_offset = get_ref("embeddings.norm.weight")?.0;
        let final_norm_offset = get_ref("final_norm.weight")?.0;

        // Allocate a zero-filled buffer for the dummy LN bias.
        let zero_bias = self.alloc_tensor(hidden)?;

        let weights = ModernBertWeights {
            tok_embeddings: tensor_at(tok_emb_offset),
            emb_norm_weight: tensor_at(emb_norm_offset),
            final_norm_weight: tensor_at(final_norm_offset),
            zero_bias,
            layers,
            num_heads,
            head_dim,
            hidden_dim: hidden,
            intermediate_dim: intermediate,
            layer_norm_eps: config.norm_eps,
            local_window: config.local_attention,
        };

        // 6. Build RoPE caches
        let max_seq = config.max_position_embeddings;
        let global_rope = build_rope_cache(self, head_dim, max_seq, config.global_rope_theta)?;
        let local_rope = build_rope_cache(self, head_dim, max_seq, config.local_rope_theta)?;

        // Pre-convert all projection weights to FP16 for the full FP16 pipeline.
        // The gemm_f16 kernel reads from MetalTensor.fp16 for weight matrices.
        // Norm weights (small [hidden] vectors) stay FP32 — the FP16 LN kernel
        // accepts FP32 weight/bias directly.
        let use_q8 = std::env::var("RIPVEC_Q8").is_ok_and(|v| v == "1");

        for layer in &weights.layers {
            let qkv_elems = 3 * hidden * hidden;
            self.ensure_fp16(&layer.qkv_weight, qkv_elems)?;

            let out_elems = hidden * hidden;
            self.ensure_fp16(&layer.output_weight, out_elems)?;

            let wi_elems = 2 * intermediate * hidden;
            self.ensure_fp16(&layer.mlp_wi_weight, wi_elems)?;

            let wo_elems = hidden * intermediate;
            self.ensure_fp16(&layer.mlp_wo_weight, wo_elems)?;

            if use_q8 {
                let q8 = self.quantize_weights_q8(&layer.qkv_weight, 3 * hidden, hidden)?;
                *layer.qkv_weight.q8.borrow_mut() = Some(q8.buffer.clone());

                let q8 = self.quantize_weights_q8(&layer.output_weight, hidden, hidden)?;
                *layer.output_weight.q8.borrow_mut() = Some(q8.buffer.clone());

                let q8 =
                    self.quantize_weights_q8(&layer.mlp_wi_weight, 2 * intermediate, hidden)?;
                *layer.mlp_wi_weight.q8.borrow_mut() = Some(q8.buffer.clone());

                let q8 = self.quantize_weights_q8(&layer.mlp_wo_weight, intermediate, hidden)?;
                *layer.mlp_wo_weight.q8.borrow_mut() = Some(q8.buffer.clone());
            }
        }

        let arch = ModernBertArch {
            weights,
            global_rope,
            local_rope,
        };

        Ok((arch, mmap))
    }

    /// Load `ClassicBert` weights from a safetensors file into [`MetalTensor`]s.
    ///
    /// Memory-maps the file and wraps it as a single zero-copy Metal buffer.
    /// Fuses separate Q, K, V weight matrices into a single `[3*hidden, hidden]`
    /// tensor (and `[3*hidden]` bias) at load time for a single GEMM per layer.
    ///
    /// Returns `(arch, mmap)` — the caller **must** keep `mmap` alive as long as
    /// the arch's tensors are in use.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened/mapped, safetensors parsing
    /// fails, or any expected weight tensor is missing.
    #[expect(
        unsafe_code,
        reason = "mmap + newBufferWithBytesNoCopy require unsafe FFI"
    )]
    #[expect(
        clippy::too_many_lines,
        reason = "weight loading is inherently verbose"
    )]
    pub fn load_classic_bert_weights(
        &self,
        weights_path: &Path,
        config: &ClassicBertConfig,
    ) -> crate::Result<(ClassicBertArch<MetalTensor>, memmap2::Mmap)> {
        // 1. mmap the safetensors file
        let file = std::fs::File::open(weights_path).map_err(|e| crate::Error::Io {
            path: weights_path.display().to_string(),
            source: e,
        })?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| crate::Error::Io {
            path: weights_path.display().to_string(),
            source: e,
        })?;

        // 2. Create zero-copy Metal buffer
        let page_size: usize = 16384; // Apple Silicon 16KB pages
        let aligned_len = mmap.len().next_multiple_of(page_size);
        let weight_buffer = unsafe {
            self.device
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    std::ptr::NonNull::new(mmap.as_ptr() as *mut _)
                        .ok_or_else(|| crate::Error::Metal("mmap returned null pointer".into()))?,
                    aligned_len as NSUInteger,
                    MTLResourceOptions::StorageModeShared,
                    None,
                )
        }
        .ok_or_else(|| {
            crate::Error::Metal(
                "zero-copy buffer creation failed (pointer not page-aligned?)".into(),
            )
        })?;

        // 3. Parse safetensors header to get tensor offsets
        let tensors = SafeTensors::deserialize(&mmap)
            .map_err(|e| crate::Error::Metal(format!("safetensors parse: {e}")))?;

        let mmap_base = mmap.as_ptr() as usize;
        let mut refs: HashMap<String, (usize, Vec<usize>)> = HashMap::new();
        for (name, view) in tensors.tensors() {
            let offset = view.data().as_ptr() as usize - mmap_base;
            let shape: Vec<usize> = view.shape().to_vec();
            refs.insert(name.clone(), (offset, shape));
        }
        drop(tensors);

        // Helper: look up a required weight by name.
        let get_ref = |name: &str| -> crate::Result<(usize, &[usize])> {
            let (offset, shape) = refs
                .get(name)
                .ok_or_else(|| crate::Error::Metal(format!("missing weight: {name}")))?;
            Ok((*offset, shape.as_slice()))
        };

        // Helper: create a MetalTensor pointing into the weight buffer at a byte offset.
        let tensor_at =
            |offset: usize| -> MetalTensor { MetalTensor::new(weight_buffer.clone(), offset) };

        // Helper: read raw f32 data from the mmap at a given byte offset and element count.
        #[expect(
            clippy::cast_ptr_alignment,
            reason = "safetensors data is f32; caller guarantees alignment via safetensors layout"
        )]
        let read_f32 = |offset: usize, n: usize| -> &[f32] {
            let ptr = unsafe { mmap.as_ptr().add(offset) }.cast::<f32>();
            unsafe { std::slice::from_raw_parts(ptr, n) }
        };

        let hidden = config.hidden_size;
        let num_layers = config.num_hidden_layers;
        let num_heads = config.num_attention_heads;
        let head_dim = hidden / num_heads;
        let intermediate = config.intermediate_size;

        // 4. Build per-layer weights with fused QKV
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let prefix = format!("encoder.layer.{i}");

            // Fuse Q+K+V weights into [3*hidden, hidden] and bias into [3*hidden].
            let (q_w_off, _) = get_ref(&format!("{prefix}.attention.self.query.weight"))?;
            let (k_w_off, _) = get_ref(&format!("{prefix}.attention.self.key.weight"))?;
            let (v_w_off, _) = get_ref(&format!("{prefix}.attention.self.value.weight"))?;
            let (q_b_off, _) = get_ref(&format!("{prefix}.attention.self.query.bias"))?;
            let (k_b_off, _) = get_ref(&format!("{prefix}.attention.self.key.bias"))?;
            let (v_b_off, _) = get_ref(&format!("{prefix}.attention.self.value.bias"))?;

            let qkv_w_size = hidden * hidden; // per Q/K/V
            let q_w = read_f32(q_w_off, qkv_w_size);
            let k_w = read_f32(k_w_off, qkv_w_size);
            let v_w = read_f32(v_w_off, qkv_w_size);
            let mut fused_qkv_w = Vec::with_capacity(3 * qkv_w_size);
            fused_qkv_w.extend_from_slice(q_w);
            fused_qkv_w.extend_from_slice(k_w);
            fused_qkv_w.extend_from_slice(v_w);
            let qkv_weight = upload_f32_to_metal(self, &fused_qkv_w)?;

            let q_b = read_f32(q_b_off, hidden);
            let k_b = read_f32(k_b_off, hidden);
            let v_b = read_f32(v_b_off, hidden);
            let mut fused_qkv_b = Vec::with_capacity(3 * hidden);
            fused_qkv_b.extend_from_slice(q_b);
            fused_qkv_b.extend_from_slice(k_b);
            fused_qkv_b.extend_from_slice(v_b);
            let qkv_bias = upload_f32_to_metal(self, &fused_qkv_b)?;

            let out_w_off = get_ref(&format!("{prefix}.attention.output.dense.weight"))?.0;
            let out_b_off = get_ref(&format!("{prefix}.attention.output.dense.bias"))?.0;
            let out_ln_w_off = get_ref(&format!("{prefix}.attention.output.LayerNorm.weight"))?.0;
            let out_ln_b_off = get_ref(&format!("{prefix}.attention.output.LayerNorm.bias"))?.0;
            let inter_w_off = get_ref(&format!("{prefix}.intermediate.dense.weight"))?.0;
            let inter_b_off = get_ref(&format!("{prefix}.intermediate.dense.bias"))?.0;
            let ffn_out_w_off = get_ref(&format!("{prefix}.output.dense.weight"))?.0;
            let ffn_out_b_off = get_ref(&format!("{prefix}.output.dense.bias"))?.0;
            let ffn_ln_w_off = get_ref(&format!("{prefix}.output.LayerNorm.weight"))?.0;
            let ffn_ln_b_off = get_ref(&format!("{prefix}.output.LayerNorm.bias"))?.0;

            layers.push(ClassicBertLayerWeights {
                qkv_weight,
                qkv_bias,
                output_weight: tensor_at(out_w_off),
                output_bias: tensor_at(out_b_off),
                output_ln_weight: tensor_at(out_ln_w_off),
                output_ln_bias: tensor_at(out_ln_b_off),
                ffn_inter_weight: tensor_at(inter_w_off),
                ffn_inter_bias: tensor_at(inter_b_off),
                ffn_out_weight: tensor_at(ffn_out_w_off),
                ffn_out_bias: tensor_at(ffn_out_b_off),
                ffn_ln_weight: tensor_at(ffn_ln_w_off),
                ffn_ln_bias: tensor_at(ffn_ln_b_off),
            });
        }

        // 5. Embedding weights
        let word_emb_off = get_ref("embeddings.word_embeddings.weight")?.0;
        let pos_emb_off = get_ref("embeddings.position_embeddings.weight")?.0;
        let tok_type_emb_off = get_ref("embeddings.token_type_embeddings.weight")?.0;
        let emb_ln_w_off = get_ref("embeddings.LayerNorm.weight")?.0;
        let emb_ln_b_off = get_ref("embeddings.LayerNorm.bias")?.0;

        let weights = ClassicBertWeights {
            word_embeddings: tensor_at(word_emb_off),
            position_embeddings: tensor_at(pos_emb_off),
            token_type_embeddings: tensor_at(tok_type_emb_off),
            emb_ln_weight: tensor_at(emb_ln_w_off),
            emb_ln_bias: tensor_at(emb_ln_b_off),
            layers,
            num_heads,
            head_dim,
            hidden_dim: hidden,
            intermediate_dim: intermediate,
            layer_norm_eps: config.layer_norm_eps,
        };

        let arch = ClassicBertArch { weights };

        // Pre-convert all weight matrices to FP16 for faster MPS GEMMs.
        for layer in &arch.weights.layers {
            let h = config.hidden_size;
            let inter = config.intermediate_size;
            self.ensure_fp16(&layer.qkv_weight, 3 * h * h)?;
            self.ensure_fp16(&layer.qkv_bias, 3 * h)?;
            self.ensure_fp16(&layer.output_weight, h * h)?;
            self.ensure_fp16(&layer.ffn_inter_weight, inter * h)?;
            self.ensure_fp16(&layer.ffn_out_weight, h * inter)?;
        }

        Ok((arch, mmap))
    }
}

/// Build a `RoPE` cos/sin cache for the given theta and sequence length.
///
/// Computes `cos(pos * freq)` and `sin(pos * freq)` for each position in
/// `[0, max_seq)` and each frequency dimension in `[0, head_dim/2)`.
/// The result is uploaded to Metal buffers as `[max_seq, head_dim/2]` tables.
#[expect(
    clippy::cast_precision_loss,
    reason = "head_dim and position indices are small enough for exact f32"
)]
fn build_rope_cache(
    driver: &MetalDriver,
    head_dim: usize,
    max_seq: usize,
    theta: f32,
) -> crate::Result<RopeCache<MetalTensor>> {
    let half_dim = head_dim / 2;
    let n = max_seq * half_dim;
    let mut cos_data = Vec::with_capacity(n);
    let mut sin_data = Vec::with_capacity(n);

    for pos in 0..max_seq {
        for d in 0..half_dim {
            let freq = (pos as f32) / theta.powf(2.0 * d as f32 / head_dim as f32);
            cos_data.push(freq.cos());
            sin_data.push(freq.sin());
        }
    }

    let cos = upload_f32_to_metal(driver, &cos_data)?;
    let sin = upload_f32_to_metal(driver, &sin_data)?;
    Ok(RopeCache { cos, sin })
}

/// Upload a host `f32` slice to a new Metal buffer as a [`MetalTensor`].
#[expect(unsafe_code, reason = "newBufferWithBytes requires unsafe FFI")]
fn upload_f32_to_metal(driver: &MetalDriver, data: &[f32]) -> crate::Result<MetalTensor> {
    let size = std::mem::size_of_val(data) as NSUInteger;
    let buffer = unsafe {
        driver.device().newBufferWithBytes_length_options(
            std::ptr::NonNull::new(data.as_ptr() as *mut _)
                .ok_or_else(|| crate::Error::Metal("null data pointer".into()))?,
            size,
            MTLResourceOptions::StorageModeShared,
        )
    }
    .ok_or_else(|| crate::Error::Metal("RoPE buffer alloc failed".into()))?;
    Ok(MetalTensor::new(buffer, 0))
}

/// Parsed `ModernBERT` model configuration from `config.json`.
///
/// Contains all geometry and hyperparameters needed to build the model
/// architecture and load weights.
pub struct ModernBertConfig {
    /// Hidden dimension (768 for `modernbert-embed-base`).
    pub hidden_size: usize,
    /// MLP intermediate dimension (1152).
    pub intermediate_size: usize,
    /// Number of encoder layers (22).
    pub num_hidden_layers: usize,
    /// Number of attention heads (12).
    pub num_attention_heads: usize,
    /// Global attention applied every N layers (3).
    pub global_attn_every_n_layers: usize,
    /// Sliding window size for local attention (128).
    pub local_attention: usize,
    /// `RoPE` theta for global attention layers (160000.0).
    pub global_rope_theta: f32,
    /// `RoPE` theta for local attention layers (10000.0).
    pub local_rope_theta: f32,
    /// Layer normalization epsilon (1e-5).
    pub norm_eps: f32,
    /// Maximum position embeddings / sequence length (8192).
    pub max_position_embeddings: usize,
    /// Vocabulary size (50368).
    pub vocab_size: usize,
}

impl ModernBertConfig {
    /// Parse a `ModernBERT` config from a `config.json` value.
    ///
    /// # Errors
    ///
    /// Returns an error if any required field is missing or has an unexpected type.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "HuggingFace config ints always fit in usize; f64 rope/eps values fit in f32"
    )]
    pub fn from_json(json: &serde_json::Value) -> crate::Result<Self> {
        let get_usize = |key: &str| -> crate::Result<usize> {
            json.get(key)
                .and_then(serde_json::Value::as_u64)
                .map(|v| v as usize)
                .ok_or_else(|| {
                    crate::Error::Metal(format!("config.json missing or invalid: {key}"))
                })
        };
        let get_f64 = |key: &str| -> crate::Result<f64> {
            json.get(key)
                .and_then(serde_json::Value::as_f64)
                .ok_or_else(|| {
                    crate::Error::Metal(format!("config.json missing or invalid: {key}"))
                })
        };

        Ok(Self {
            hidden_size: get_usize("hidden_size")?,
            intermediate_size: get_usize("intermediate_size")?,
            num_hidden_layers: get_usize("num_hidden_layers")?,
            num_attention_heads: get_usize("num_attention_heads")?,
            global_attn_every_n_layers: get_usize("global_attn_every_n_layers")?,
            local_attention: get_usize("local_attention")?,
            global_rope_theta: get_f64("global_rope_theta")? as f32,
            local_rope_theta: get_f64("local_rope_theta")? as f32,
            norm_eps: get_f64("norm_eps").unwrap_or(1e-5) as f32,
            max_position_embeddings: get_usize("max_position_embeddings")?,
            vocab_size: get_usize("vocab_size")?,
        })
    }
}

/// Parsed `ClassicBert` model configuration from `config.json`.
///
/// Contains geometry and hyperparameters needed to build the `ClassicBert`
/// architecture and load weights (e.g. `BAAI/bge-small-en-v1.5`).
pub struct ClassicBertConfig {
    /// Hidden dimension (384 for BGE-small).
    pub hidden_size: usize,
    /// FFN intermediate dimension (1536 for BGE-small).
    pub intermediate_size: usize,
    /// Number of encoder layers (12 for BGE-small).
    pub num_hidden_layers: usize,
    /// Number of attention heads (12 for BGE-small).
    pub num_attention_heads: usize,
    /// Layer normalization epsilon (typically 1e-12).
    pub layer_norm_eps: f32,
    /// Maximum position embeddings / sequence length (512 for BGE-small).
    pub max_position_embeddings: usize,
    /// Vocabulary size (30522 for BGE-small).
    pub vocab_size: usize,
}

impl ClassicBertConfig {
    /// Parse a `ClassicBert` config from a `config.json` value.
    ///
    /// Expects standard BERT config keys (`hidden_size`, `intermediate_size`, etc.).
    ///
    /// # Errors
    ///
    /// Returns an error if any required field is missing or has an unexpected type.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "config ints are small positive values"
    )]
    pub fn from_json(json: &serde_json::Value) -> crate::Result<Self> {
        let get_usize = |key: &str| -> crate::Result<usize> {
            json.get(key)
                .and_then(serde_json::Value::as_u64)
                .map(|v| v as usize)
                .ok_or_else(|| {
                    crate::Error::Metal(format!("config.json missing or invalid: {key}"))
                })
        };
        let get_f64 = |key: &str| -> crate::Result<f64> {
            json.get(key)
                .and_then(serde_json::Value::as_f64)
                .ok_or_else(|| {
                    crate::Error::Metal(format!("config.json missing or invalid: {key}"))
                })
        };

        Ok(Self {
            hidden_size: get_usize("hidden_size")?,
            intermediate_size: get_usize("intermediate_size")?,
            num_hidden_layers: get_usize("num_hidden_layers")?,
            num_attention_heads: get_usize("num_attention_heads")?,
            layer_norm_eps: get_f64("layer_norm_eps")
                .or_else(|_| get_f64("layer_norm_epsilon"))
                .unwrap_or(1e-12) as f32,
            max_position_embeddings: get_usize("max_position_embeddings").unwrap_or(512),
            vocab_size: get_usize("vocab_size")?,
        })
    }
}

// SAFETY: Metal device and command queue are thread-safe per Apple docs.
#[expect(unsafe_code, reason = "Metal device and queue are thread-safe")]
unsafe impl Send for MetalDriver {}
#[expect(unsafe_code, reason = "Metal device and queue are thread-safe")]
unsafe impl Sync for MetalDriver {}

// ---------------------------------------------------------------------------
// Driver trait implementation
// ---------------------------------------------------------------------------

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "dimension values are small positive integers that fit in i32/u32"
)]
impl Driver for MetalDriver {
    type Tensor = MetalTensor;

    fn name(&self) -> &'static str {
        "Metal"
    }

    fn begin_batch(&self) -> crate::Result<()> {
        self.begin_batch()
    }

    fn end_batch(&self) -> crate::Result<()> {
        self.end_batch()
    }

    fn flush_batch(&self) -> crate::Result<()> {
        self.flush_batch()
    }

    fn segment_encoder(&self) {
        self.flush_compute_encoder();
    }

    fn save_pool_cursor(&self) -> usize {
        self.pool_cursor.get()
    }

    fn restore_pool_cursor(&self, saved: usize) {
        // Restore cursor to a saved position so the NEXT layer reuses the
        // pool slots that the CURRENT layer's dropped tensors occupied.
        //
        // Safe because the architecture drops all transient tensors (qkv,
        // scores, context, etc.) before calling this. Only hidden_states
        // survives — and it was allocated BEFORE the saved position (at the
        // embedding phase or previous layer's output slot).
        self.pool_cursor.set(saved);
        self.pool_f16_cursor
            .set(saved.min(self.pool_f16_cursor.get()));
    }

    fn alloc_zeros(&self, n: usize) -> crate::Result<MetalTensor> {
        self.alloc_tensor(n)
    }

    #[expect(
        unsafe_code,
        reason = "Metal buffer copy requires unsafe contents access"
    )]
    fn clone_tensor(&self, tensor: &MetalTensor, n: usize) -> crate::Result<MetalTensor> {
        let new_tensor = self.alloc_tensor(n)?;
        let byte_count = (n * core::mem::size_of::<f32>()) as NSUInteger;

        // Use GPU blit copy so it works in batched mode (CPU memcpy would
        // read stale data from uncommitted command buffers).
        let is_batched = self.batch_cmd.borrow().is_some();
        if is_batched {
            // Must close compute encoder before creating blit encoder
            self.flush_compute_encoder();
            let batch = self.batch_cmd.borrow();
            let cmd_buf = batch.as_ref().unwrap();
            let blit = cmd_buf
                .blitCommandEncoder()
                .ok_or_else(|| crate::Error::Metal("blit encoder failed".into()))?;
            unsafe {
                blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    &tensor.buffer,
                    tensor.offset as NSUInteger,
                    &new_tensor.buffer,
                    0,
                    byte_count,
                );
            }
            blit.endEncoding();
        } else {
            // Per-call mode: CPU copy is fine (data already committed)
            unsafe {
                let src = tensor
                    .buffer
                    .contents()
                    .as_ptr()
                    .cast::<u8>()
                    .add(tensor.offset);
                let dst = new_tensor.buffer.contents().as_ptr().cast::<u8>();
                std::ptr::copy_nonoverlapping(src, dst, byte_count as usize);
            }
        }
        Ok(new_tensor)
    }

    fn prepare_batch(
        &self,
        encodings: &[Encoding],
        max_seq: usize,
    ) -> crate::Result<BatchInputs<MetalTensor>> {
        let batch = encodings.len();
        let total = batch * max_seq;

        // Build padded input tensors on CPU
        let mut input_ids = vec![0_i32; total];
        let mut token_type_ids = vec![0_i32; total];
        let mut position_ids = vec![0_i32; total];
        let mut attn_mask_int = vec![0_i32; total];

        for (b, enc) in encodings.iter().enumerate() {
            let seq_len = enc.input_ids.len();
            let offset = b * max_seq;
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

        // Upload to GPU
        let input_ids_buf = make_i32_buffer(&self.device, &input_ids)?;
        let token_type_ids_buf = make_i32_buffer(&self.device, &token_type_ids)?;
        let position_ids_buf = make_i32_buffer(&self.device, &position_ids)?;
        let attn_mask_int_buf = make_i32_buffer(&self.device, &attn_mask_int)?;

        // Build float attention bias mask (0.0 for real, -1e9 for pad)
        let float_mask_buf = alloc_f32_buffer(&self.device, total)?;
        self.run_compute("build-attn-mask", |enc| {
            enc.setComputePipelineState(&self.kernels.build_attn_mask);
            set_buffer(enc, &float_mask_buf, 0, 0);
            set_buffer(enc, &attn_mask_int_buf, 0, 1);
            set_i32_param(enc, total as i32, 2);
            dispatch_1d(enc, &self.kernels.build_attn_mask, total);
            Ok(())
        })?;

        // Build pooling mask (1.0 for real, 0.0 for pad) — for mean pooling
        let pooling_mask: Vec<f32> = attn_mask_int
            .iter()
            .map(|&m| if m == 1 { 1.0 } else { 0.0 })
            .collect();
        let pooling_mask_buf = make_f32_buffer(&self.device, &pooling_mask)?;

        // Compute per-sequence lengths and total actual tokens
        let seq_lengths: Vec<usize> = encodings.iter().map(|e| e.input_ids.len()).collect();
        let total_tokens: usize = seq_lengths.iter().sum();

        Ok(BatchInputs {
            input_ids: MetalTensor::new(input_ids_buf, 0),
            attention_mask: MetalTensor::new(attn_mask_int_buf, 0),
            token_type_ids: MetalTensor::new(token_type_ids_buf, 0),
            position_ids: MetalTensor::new(position_ids_buf, 0),
            float_mask: MetalTensor::new(float_mask_buf, 0),
            pooling_mask: MetalTensor::new(pooling_mask_buf, 0),
            batch,
            max_seq,
            total_tokens,
            seq_lengths,
            cu_seqlens: None, // Padded mode for now
        })
    }

    fn prepare_batch_unpadded(
        &self,
        encodings: &[Encoding],
    ) -> crate::Result<BatchInputs<MetalTensor>> {
        let batch = encodings.len();
        let seq_lengths: Vec<usize> = encodings.iter().map(|e| e.input_ids.len()).collect();
        let total_tokens: usize = seq_lengths.iter().sum();
        let max_seq = seq_lengths
            .iter()
            .copied()
            .max()
            .unwrap_or(0)
            .next_multiple_of(8);

        // Build cu_seqlens: [0, len0, len0+len1, ...]
        let mut cu_seqlens = Vec::with_capacity(batch + 1);
        cu_seqlens.push(0);
        let mut cumsum = 0;
        for &len in &seq_lengths {
            cumsum += len;
            cu_seqlens.push(cumsum);
        }

        // Concatenate all tokens flat — NO padding
        let mut input_ids = Vec::with_capacity(total_tokens);
        let mut token_type_ids = Vec::with_capacity(total_tokens);
        let mut position_ids = Vec::with_capacity(total_tokens);

        for enc in encodings {
            for (i, &id) in enc.input_ids.iter().enumerate() {
                input_ids.push(id as i32);
                token_type_ids.push(enc.token_type_ids[i] as i32);
                position_ids.push(i as i32);
            }
        }

        // Upload to GPU
        let input_ids_buf = make_i32_buffer(&self.device, &input_ids)?;
        let token_type_ids_buf = make_i32_buffer(&self.device, &token_type_ids)?;
        let position_ids_buf = make_i32_buffer(&self.device, &position_ids)?;

        // Build padded attention mask from seq_lengths: [batch * max_seq]
        // 1 for real tokens, 0 for padding positions.
        let padded_total = batch * max_seq;
        let mut attn_mask_int = vec![0_i32; padded_total];
        for (b, &len) in seq_lengths.iter().enumerate() {
            let offset = b * max_seq;
            for i in 0..len {
                attn_mask_int[offset + i] = 1;
            }
        }
        let attn_mask_int_buf = make_i32_buffer(&self.device, &attn_mask_int)?;

        // Build float attention bias mask on GPU (0.0 for real, -1e9 for pad)
        let float_mask_buf = alloc_f32_buffer(&self.device, padded_total)?;
        self.run_compute("build-attn-mask", |enc| {
            enc.setComputePipelineState(&self.kernels.build_attn_mask);
            set_buffer(enc, &float_mask_buf, 0, 0);
            set_buffer(enc, &attn_mask_int_buf, 0, 1);
            set_i32_param(enc, padded_total as i32, 2);
            dispatch_1d(enc, &self.kernels.build_attn_mask, padded_total);
            Ok(())
        })?;

        // Build padded pooling mask (1.0 for real, 0.0 for pad)
        let pooling_mask_padded: Vec<f32> = attn_mask_int
            .iter()
            .map(|&m| if m == 1 { 1.0 } else { 0.0 })
            .collect();
        let pooling_mask_buf = make_f32_buffer(&self.device, &pooling_mask_padded)?;

        Ok(BatchInputs {
            input_ids: MetalTensor::new(input_ids_buf, 0),
            attention_mask: MetalTensor::new(attn_mask_int_buf, 0),
            token_type_ids: MetalTensor::new(token_type_ids_buf, 0),
            position_ids: MetalTensor::new(position_ids_buf, 0),
            float_mask: MetalTensor::new(float_mask_buf, 0),
            pooling_mask: MetalTensor::new(pooling_mask_buf, 0),
            batch,
            max_seq,
            total_tokens,
            seq_lengths,
            cu_seqlens: Some(cu_seqlens),
        })
    }

    fn pad_to_batch(
        &self,
        flat: &MetalTensor,
        padded: &mut MetalTensor,
        seq_lengths: &[usize],
        max_seq: usize,
        dim: usize,
    ) -> crate::Result<()> {
        let batch = seq_lengths.len();
        let total_out = batch * max_seq * dim;

        // Build cu_seqlens on CPU: [0, len0, len0+len1, ...]
        let mut cu: Vec<i32> = Vec::with_capacity(batch + 1);
        cu.push(0);
        let mut acc: i32 = 0;
        for &len in seq_lengths {
            acc += len as i32;
            cu.push(acc);
        }
        let cu_buf = make_i32_buffer(&self.device, &cu)?;

        // Use caller's pre-allocated buffer — do NOT re-allocate.
        let padded_buf = padded.buffer.clone();
        let padded_offset = padded.offset;
        let flat_buf = flat.buffer.clone();
        let flat_offset = flat.offset;

        self.run_compute("pad-to-batch", |enc| {
            enc.setComputePipelineState(&self.kernels.pad_to_batch);
            set_buffer(enc, &padded_buf, padded_offset, 0);
            set_buffer(enc, &flat_buf, flat_offset, 1);
            set_buffer(enc, &cu_buf, 0, 2);
            set_i32_param(enc, max_seq as i32, 3);
            set_i32_param(enc, dim as i32, 4);
            set_i32_param(enc, batch as i32, 5);
            dispatch_1d(enc, &self.kernels.pad_to_batch, total_out);
            Ok(())
        })
    }

    fn unpad_from_batch(
        &self,
        padded: &MetalTensor,
        flat: &mut MetalTensor,
        seq_lengths: &[usize],
        max_seq: usize,
        dim: usize,
    ) -> crate::Result<()> {
        let batch = seq_lengths.len();
        let total_tokens: usize = seq_lengths.iter().sum();

        // Build cu_seqlens on CPU: [0, len0, len0+len1, ...]
        let mut cu: Vec<i32> = Vec::with_capacity(batch + 1);
        cu.push(0);
        let mut acc: i32 = 0;
        for &len in seq_lengths {
            acc += len as i32;
            cu.push(acc);
        }
        let cu_buf = make_i32_buffer(&self.device, &cu)?;

        // Use caller's pre-allocated buffer — do NOT re-allocate.
        // The caller sizes flat to g.total_tokens * dim which may differ from
        // sum(seq_lengths) in the padded path. Re-allocating would cause a
        // buffer overflow in residual_add.
        let flat_buf = flat.buffer.clone();
        let flat_offset = flat.offset;
        let padded_buf = padded.buffer.clone();
        let padded_offset = padded.offset;

        self.run_compute("unpad-from-batch", |enc| {
            enc.setComputePipelineState(&self.kernels.unpad_from_batch);
            set_buffer(enc, &flat_buf, flat_offset, 0);
            set_buffer(enc, &padded_buf, padded_offset, 1);
            set_buffer(enc, &cu_buf, 0, 2);
            set_i32_param(enc, max_seq as i32, 3);
            set_i32_param(enc, dim as i32, 4);
            set_i32_param(enc, total_tokens as i32, 5);
            dispatch_1d(enc, &self.kernels.unpad_from_batch, total_tokens * dim);
            Ok(())
        })
    }

    fn embedding_lookup(
        &self,
        word_ids: &MetalTensor,
        embedding_table: &MetalTensor,
        seq_len: usize,
        hidden: usize,
    ) -> crate::Result<MetalTensor> {
        let n = seq_len * hidden;
        let output = self.alloc_tensor(n)?;

        self.run_compute("embedding-lookup", |enc| {
            enc.setComputePipelineState(&self.kernels.embedding_lookup);
            set_buffer(enc, &output.buffer, output.offset, 0);
            set_buffer(enc, &embedding_table.buffer, embedding_table.offset, 1);
            set_buffer(enc, &word_ids.buffer, word_ids.offset, 2);
            set_i32_param(enc, seq_len as i32, 3);
            set_i32_param(enc, hidden as i32, 4);
            dispatch_1d(enc, &self.kernels.embedding_lookup, n);
            Ok(())
        })?;

        Ok(output)
    }

    fn add_embeddings(
        &self,
        hidden: &mut MetalTensor,
        table: &MetalTensor,
        ids: &MetalTensor,
        seq_len: usize,
        hidden_dim: usize,
    ) -> crate::Result<()> {
        let n = seq_len * hidden_dim;
        self.run_compute("add-embeddings", |enc| {
            enc.setComputePipelineState(&self.kernels.add_embeddings);
            set_buffer(enc, &hidden.buffer, hidden.offset, 0);
            set_buffer(enc, &table.buffer, table.offset, 1);
            set_buffer(enc, &ids.buffer, ids.offset, 2);
            set_i32_param(enc, seq_len as i32, 3);
            set_i32_param(enc, hidden_dim as i32, 4);
            dispatch_1d(enc, &self.kernels.add_embeddings, n);
            Ok(())
        })
    }

    fn layer_norm(
        &self,
        output: &mut MetalTensor,
        input: &MetalTensor,
        weight: &MetalTensor,
        bias: &MetalTensor,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> crate::Result<()> {
        let threads = 256.min(cols);
        self.run_compute("layer-norm", |enc| {
            enc.setComputePipelineState(&self.kernels.layer_norm);
            set_buffer(enc, &output.buffer, output.offset, 0);
            set_buffer(enc, &input.buffer, input.offset, 1);
            set_buffer(enc, &weight.buffer, weight.offset, 2);
            set_buffer(enc, &bias.buffer, bias.offset, 3);
            set_i32_param(enc, rows as i32, 4);
            set_i32_param(enc, cols as i32, 5);
            set_f32_param(enc, eps, 6);
            dispatch_rows(enc, &self.kernels.layer_norm, rows, threads);
            Ok(())
        })
    }

    #[expect(
        clippy::many_single_char_names,
        reason = "a, b, m, n, k are standard GEMM parameter names from BLAS"
    )]
    fn gemm(
        &self,
        a: &MetalTensor,
        b: &MetalTensor,
        output: &mut MetalTensor,
        m: usize,
        n: usize,
        k: usize,
        transpose_b: bool,
    ) -> crate::Result<()> {
        // MPS GEMM (FP32 path always uses MPS — native kernel requires FP16 activations)
        self.run_mps("mps-gemm", |cmd_buf| {
            dispatch_mps_gemm(
                cmd_buf,
                &self.device,
                &a.buffer,
                a.offset,
                &b.buffer,
                b.offset,
                &output.buffer,
                output.offset,
                m,
                n,
                k,
                transpose_b,
                MPSDataType::Float32,
                MPSDataType::Float32,
            );
            Ok(())
        })
    }

    #[expect(
        clippy::many_single_char_names,
        reason = "a, b, m, n, k are standard GEMM parameter names from BLAS"
    )]
    fn gemm_batched(
        &self,
        a: &MetalTensor,
        b: &MetalTensor,
        output: &mut MetalTensor,
        m: usize,
        n: usize,
        k: usize,
        transpose_b: bool,
        stride_a: usize,
        stride_b: usize,
        stride_c: usize,
        batch_count: usize,
    ) -> crate::Result<()> {
        self.run_compute("gemm-batched", |enc| {
            dispatch_gemm_batched(
                enc,
                &self.kernels.gemm_batched,
                &a.buffer,
                a.offset,
                &b.buffer,
                b.offset,
                &output.buffer,
                output.offset,
                m as u32,
                n as u32,
                k as u32,
                transpose_b,
                stride_a as u32,
                stride_b as u32,
                stride_c as u32,
                batch_count as u32,
            );
            Ok(())
        })
    }

    fn fused_scale_mask_softmax(
        &self,
        scores: &mut MetalTensor,
        mask: &MetalTensor,
        batch: usize,
        num_heads: usize,
        seq_len: usize,
        scale: f32,
    ) -> crate::Result<()> {
        let total_rows = batch * num_heads * seq_len;
        let threads = 256.min(seq_len.next_power_of_two());
        self.run_compute("softmax", |enc| {
            enc.setComputePipelineState(&self.kernels.fused_scale_mask_softmax);
            set_buffer(enc, &scores.buffer, scores.offset, 0);
            set_buffer(enc, &mask.buffer, mask.offset, 1);
            set_i32_param(enc, batch as i32, 2);
            set_i32_param(enc, num_heads as i32, 3);
            set_i32_param(enc, seq_len as i32, 4);
            set_f32_param(enc, scale, 5);
            dispatch_rows(
                enc,
                &self.kernels.fused_scale_mask_softmax,
                total_rows,
                threads,
            );
            Ok(())
        })
    }

    fn fused_scale_mask_softmax_windowed(
        &self,
        scores: &mut MetalTensor,
        mask: &MetalTensor,
        batch: usize,
        num_heads: usize,
        seq_len: usize,
        scale: f32,
        window_size: usize,
    ) -> crate::Result<()> {
        let total_rows = batch * num_heads * seq_len;
        let threads = 256.min(seq_len.next_power_of_two());
        let half_window = window_size / 2;
        self.run_compute("softmax-windowed", |enc| {
            enc.setComputePipelineState(&self.kernels.fused_scale_mask_softmax_windowed);
            set_buffer(enc, &scores.buffer, scores.offset, 0);
            set_buffer(enc, &mask.buffer, mask.offset, 1);
            set_i32_param(enc, batch as i32, 2);
            set_i32_param(enc, num_heads as i32, 3);
            set_i32_param(enc, seq_len as i32, 4);
            set_f32_param(enc, scale, 5);
            set_i32_param(enc, half_window as i32, 6);
            dispatch_rows(
                enc,
                &self.kernels.fused_scale_mask_softmax_windowed,
                total_rows,
                threads,
            );
            Ok(())
        })
    }

    fn build_attn_mask(
        &self,
        output: &mut MetalTensor,
        int_mask: &MetalTensor,
        n: usize,
    ) -> crate::Result<()> {
        self.run_compute("build-attn-mask", |enc| {
            enc.setComputePipelineState(&self.kernels.build_attn_mask);
            set_buffer(enc, &output.buffer, output.offset, 0);
            set_buffer(enc, &int_mask.buffer, int_mask.offset, 1);
            set_i32_param(enc, n as i32, 2);
            dispatch_1d(enc, &self.kernels.build_attn_mask, n);
            Ok(())
        })
    }

    fn qkv_split(
        &self,
        q: &mut MetalTensor,
        k: &mut MetalTensor,
        v: &mut MetalTensor,
        qkv: &MetalTensor,
        batch: usize,
        seq: usize,
        hidden: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> crate::Result<()> {
        let total_head = batch * num_heads * seq * head_dim;
        self.run_compute("qkv-split", |enc| {
            enc.setComputePipelineState(&self.kernels.qkv_split);
            set_buffer(enc, &q.buffer, q.offset, 0);
            set_buffer(enc, &k.buffer, k.offset, 1);
            set_buffer(enc, &v.buffer, v.offset, 2);
            set_buffer(enc, &qkv.buffer, qkv.offset, 3);
            set_i32_param(enc, batch as i32, 4);
            set_i32_param(enc, seq as i32, 5);
            set_i32_param(enc, hidden as i32, 6);
            set_i32_param(enc, num_heads as i32, 7);
            set_i32_param(enc, head_dim as i32, 8);
            dispatch_1d(enc, &self.kernels.qkv_split, total_head);
            Ok(())
        })
    }

    fn attn_reshape(
        &self,
        output: &mut MetalTensor,
        input: &MetalTensor,
        batch: usize,
        seq: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> crate::Result<()> {
        let total = batch * seq * num_heads * head_dim;
        self.run_compute("attn-reshape", |enc| {
            enc.setComputePipelineState(&self.kernels.attn_reshape);
            set_buffer(enc, &output.buffer, output.offset, 0);
            set_buffer(enc, &input.buffer, input.offset, 1);
            set_i32_param(enc, batch as i32, 2);
            set_i32_param(enc, seq as i32, 3);
            set_i32_param(enc, num_heads as i32, 4);
            set_i32_param(enc, head_dim as i32, 5);
            dispatch_1d(enc, &self.kernels.attn_reshape, total);
            Ok(())
        })
    }

    fn apply_rope(
        &self,
        qk: &mut MetalTensor,
        cos: &MetalTensor,
        sin: &MetalTensor,
        num_rows: usize,
        seq_len: usize,
        head_dim: usize,
        num_heads: usize,
    ) -> crate::Result<()> {
        let half = head_dim / 2;
        let total = num_rows * half;
        self.run_compute("apply-rope", |enc| {
            enc.setComputePipelineState(&self.kernels.rope_cached);
            set_buffer(enc, &qk.buffer, qk.offset, 0);
            set_buffer(enc, &cos.buffer, cos.offset, 1);
            set_buffer(enc, &sin.buffer, sin.offset, 2);
            set_i32_param(enc, num_rows as i32, 3);
            set_i32_param(enc, seq_len as i32, 4);
            set_i32_param(enc, head_dim as i32, 5);
            set_i32_param(enc, num_heads as i32, 6);
            dispatch_1d(enc, &self.kernels.rope_cached, total);
            Ok(())
        })
    }

    fn gelu(&self, x: &mut MetalTensor, n: usize) -> crate::Result<()> {
        self.run_compute("gelu", |enc| {
            enc.setComputePipelineState(&self.kernels.gelu);
            set_buffer(enc, &x.buffer, x.offset, 0);
            set_i32_param(enc, n as i32, 1);
            dispatch_1d(enc, &self.kernels.gelu, n);
            Ok(())
        })
    }

    fn swiglu(
        &self,
        value: &MetalTensor,
        gate: &MetalTensor,
        output: &mut MetalTensor,
        n: usize,
    ) -> crate::Result<()> {
        self.run_compute("swiglu", |enc| {
            enc.setComputePipelineState(&self.kernels.swiglu_two_input);
            set_buffer(enc, &output.buffer, output.offset, 0);
            set_buffer(enc, &value.buffer, value.offset, 1);
            set_buffer(enc, &gate.buffer, gate.offset, 2);
            set_i32_param(enc, n as i32, 3);
            dispatch_1d(enc, &self.kernels.swiglu_two_input, n);
            Ok(())
        })
    }

    fn split_gate_value(
        &self,
        first: &mut MetalTensor,
        second: &mut MetalTensor,
        input: &MetalTensor,
        rows: usize,
        cols: usize,
    ) -> crate::Result<()> {
        let total = rows * cols;
        self.run_compute("split-gate-value", |enc| {
            enc.setComputePipelineState(&self.kernels.split_gate_value);
            set_buffer(enc, &first.buffer, first.offset, 0);
            set_buffer(enc, &second.buffer, second.offset, 1);
            set_buffer(enc, &input.buffer, input.offset, 2);
            set_i32_param(enc, rows as i32, 3);
            set_i32_param(enc, cols as i32, 4);
            dispatch_1d(enc, &self.kernels.split_gate_value, total);
            Ok(())
        })
    }

    fn geglu(
        &self,
        value: &MetalTensor,
        gate: &MetalTensor,
        output: &mut MetalTensor,
        n: usize,
    ) -> crate::Result<()> {
        self.run_compute("geglu", |enc| {
            enc.setComputePipelineState(&self.kernels.geglu);
            set_buffer(enc, &output.buffer, output.offset, 0);
            set_buffer(enc, &value.buffer, value.offset, 1);
            set_buffer(enc, &gate.buffer, gate.offset, 2);
            set_i32_param(enc, n as i32, 3);
            dispatch_1d(enc, &self.kernels.geglu, n);
            Ok(())
        })
    }

    fn fused_bias_gelu(
        &self,
        x: &mut MetalTensor,
        bias: &MetalTensor,
        rows: usize,
        cols: usize,
    ) -> crate::Result<()> {
        let total = rows * cols;
        self.run_compute("fused-bias-gelu", |enc| {
            enc.setComputePipelineState(&self.kernels.fused_bias_gelu);
            set_buffer(enc, &x.buffer, x.offset, 0);
            set_buffer(enc, &bias.buffer, bias.offset, 1);
            set_i32_param(enc, rows as i32, 2);
            set_i32_param(enc, cols as i32, 3);
            dispatch_1d(enc, &self.kernels.fused_bias_gelu, total);
            Ok(())
        })
    }

    fn fused_bias_residual(
        &self,
        output: &mut MetalTensor,
        input: &MetalTensor,
        bias: &MetalTensor,
        residual: &MetalTensor,
        n: usize,
        cols: usize,
    ) -> crate::Result<()> {
        let rows = n / cols;
        self.run_compute("fused-bias-residual", |enc| {
            enc.setComputePipelineState(&self.kernels.fused_bias_residual);
            set_buffer(enc, &output.buffer, output.offset, 0);
            set_buffer(enc, &input.buffer, input.offset, 1);
            set_buffer(enc, &bias.buffer, bias.offset, 2);
            set_buffer(enc, &residual.buffer, residual.offset, 3);
            set_i32_param(enc, rows as i32, 4);
            set_i32_param(enc, cols as i32, 5);
            dispatch_1d(enc, &self.kernels.fused_bias_residual, n);
            Ok(())
        })
    }

    fn fused_residual_layernorm(
        &self,
        output: &mut MetalTensor,
        hidden: &MetalTensor,
        residual: &MetalTensor,
        weight: &MetalTensor,
        bias: &MetalTensor,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> crate::Result<()> {
        let threads = 256.min(cols);
        self.run_compute("fused-residual-layernorm", |enc| {
            enc.setComputePipelineState(&self.kernels.fused_residual_layernorm);
            set_buffer(enc, &output.buffer, output.offset, 0);
            set_buffer(enc, &hidden.buffer, hidden.offset, 1);
            set_buffer(enc, &residual.buffer, residual.offset, 2);
            set_buffer(enc, &weight.buffer, weight.offset, 3);
            set_buffer(enc, &bias.buffer, bias.offset, 4);
            set_i32_param(enc, rows as i32, 5);
            set_i32_param(enc, cols as i32, 6);
            set_f32_param(enc, eps, 7);
            dispatch_rows(enc, &self.kernels.fused_residual_layernorm, rows, threads);
            Ok(())
        })
    }

    fn residual_add(
        &self,
        output: &mut MetalTensor,
        hidden: &MetalTensor,
        residual: &MetalTensor,
        n: usize,
    ) -> crate::Result<()> {
        self.run_compute("residual-add", |enc| {
            enc.setComputePipelineState(&self.kernels.residual_add);
            set_buffer(enc, &output.buffer, output.offset, 0);
            set_buffer(enc, &hidden.buffer, hidden.offset, 1);
            set_buffer(enc, &residual.buffer, residual.offset, 2);
            set_i32_param(enc, n as i32, 3);
            dispatch_1d(enc, &self.kernels.residual_add, n);
            Ok(())
        })
    }

    fn add_bias(
        &self,
        x: &mut MetalTensor,
        bias: &MetalTensor,
        rows: usize,
        cols: usize,
    ) -> crate::Result<()> {
        let total = rows * cols;
        self.run_compute("add-bias", |enc| {
            enc.setComputePipelineState(&self.kernels.add_bias);
            set_buffer(enc, &x.buffer, x.offset, 0);
            set_buffer(enc, &bias.buffer, bias.offset, 1);
            set_i32_param(enc, rows as i32, 2);
            set_i32_param(enc, cols as i32, 3);
            dispatch_1d(enc, &self.kernels.add_bias, total);
            Ok(())
        })
    }

    fn cls_pool(
        &self,
        output: &mut MetalTensor,
        hidden: &MetalTensor,
        batch: usize,
        seq: usize,
        hidden_dim: usize,
    ) -> crate::Result<()> {
        let total = batch * hidden_dim;
        self.run_compute("cls-pool", |enc| {
            enc.setComputePipelineState(&self.kernels.cls_pool);
            set_buffer(enc, &output.buffer, output.offset, 0);
            set_buffer(enc, &hidden.buffer, hidden.offset, 1);
            set_i32_param(enc, batch as i32, 2);
            set_i32_param(enc, seq as i32, 3);
            set_i32_param(enc, hidden_dim as i32, 4);
            dispatch_1d(enc, &self.kernels.cls_pool, total);
            Ok(())
        })
    }

    fn mean_pool(
        &self,
        output: &mut MetalTensor,
        hidden: &MetalTensor,
        mask: &MetalTensor,
        batch: usize,
        seq: usize,
        hidden_dim: usize,
    ) -> crate::Result<()> {
        let total = batch * hidden_dim;
        self.run_compute("mean-pool", |enc| {
            enc.setComputePipelineState(&self.kernels.mean_pool);
            set_buffer(enc, &output.buffer, output.offset, 0);
            set_buffer(enc, &hidden.buffer, hidden.offset, 1);
            set_buffer(enc, &mask.buffer, mask.offset, 2);
            set_i32_param(enc, batch as i32, 3);
            set_i32_param(enc, seq as i32, 4);
            set_i32_param(enc, hidden_dim as i32, 5);
            dispatch_1d(enc, &self.kernels.mean_pool, total);
            Ok(())
        })
    }

    fn l2_normalize(&self, data: &mut MetalTensor, rows: usize, cols: usize) -> crate::Result<()> {
        let threads = 256.min(cols);
        self.run_compute("l2-normalize", |enc| {
            enc.setComputePipelineState(&self.kernels.l2_normalize);
            set_buffer(enc, &data.buffer, data.offset, 0);
            set_i32_param(enc, rows as i32, 1);
            set_i32_param(enc, cols as i32, 2);
            dispatch_rows(enc, &self.kernels.l2_normalize, rows, threads);
            Ok(())
        })
    }

    fn banded_qk(
        &self,
        q: &MetalTensor,
        k: &MetalTensor,
        scores: &mut MetalTensor,
        batch_heads: usize,
        seq: usize,
        head_dim: usize,
        window: usize,
        stride_qk: usize,
        stride_scores: usize,
    ) -> crate::Result<()> {
        let total = batch_heads * seq * window;
        self.run_compute("banded-qk", |enc| {
            enc.setComputePipelineState(&self.kernels.banded_qk);
            set_buffer(enc, &scores.buffer, scores.offset, 0);
            set_buffer(enc, &q.buffer, q.offset, 1);
            set_buffer(enc, &k.buffer, k.offset, 2);
            set_i32_param(enc, batch_heads as i32, 3);
            set_i32_param(enc, seq as i32, 4);
            set_i32_param(enc, head_dim as i32, 5);
            set_i32_param(enc, window as i32, 6);
            set_i32_param(enc, stride_qk as i32, 7);
            set_i32_param(enc, stride_scores as i32, 8);
            dispatch_1d(enc, &self.kernels.banded_qk, total);
            Ok(())
        })
    }

    fn banded_sv(
        &self,
        scores: &MetalTensor,
        v: &MetalTensor,
        output: &mut MetalTensor,
        batch_heads: usize,
        seq: usize,
        head_dim: usize,
        window: usize,
        stride_scores: usize,
        stride_v: usize,
        stride_out: usize,
    ) -> crate::Result<()> {
        let total = batch_heads * seq * head_dim;
        self.run_compute("banded-sv", |enc| {
            enc.setComputePipelineState(&self.kernels.banded_sv);
            set_buffer(enc, &output.buffer, output.offset, 0);
            set_buffer(enc, &scores.buffer, scores.offset, 1);
            set_buffer(enc, &v.buffer, v.offset, 2);
            set_i32_param(enc, batch_heads as i32, 3);
            set_i32_param(enc, seq as i32, 4);
            set_i32_param(enc, head_dim as i32, 5);
            set_i32_param(enc, window as i32, 6);
            set_i32_param(enc, stride_scores as i32, 7);
            set_i32_param(enc, stride_v as i32, 8);
            set_i32_param(enc, stride_out as i32, 9);
            dispatch_1d(enc, &self.kernels.banded_sv, total);
            Ok(())
        })
    }

    fn banded_softmax(
        &self,
        scores: &mut MetalTensor,
        total_rows: usize,
        window: usize,
        scale: f32,
    ) -> crate::Result<()> {
        let threads = 256.min(window).max(1);
        self.run_compute("banded-softmax", |enc| {
            enc.setComputePipelineState(&self.kernels.banded_softmax);
            set_buffer(enc, &scores.buffer, scores.offset, 0);
            set_i32_param(enc, window as i32, 1);
            set_f32_param(enc, scale, 2);
            dispatch_rows(enc, &self.kernels.banded_softmax, total_rows, threads);
            Ok(())
        })
    }

    #[expect(unsafe_code, reason = "Metal buffer readback requires unsafe FFI")]
    fn to_host(
        &self,
        tensor: &MetalTensor,
        batch: usize,
        dim: usize,
    ) -> crate::Result<Vec<Vec<f32>>> {
        let flat = unsafe {
            let ptr = tensor.buffer.contents().as_ptr() as *const f32;
            let offset_elems = tensor.offset / core::mem::size_of::<f32>();
            core::slice::from_raw_parts(ptr.add(offset_elems), batch * dim)
        };

        let mut results = Vec::with_capacity(batch);
        for b in 0..batch {
            results.push(flat[b * dim..(b + 1) * dim].to_vec());
        }
        Ok(results)
    }

    // =======================================================================
    #[expect(
        clippy::cast_possible_truncation,
        reason = "m/n/k are small ML dimensions that fit in u32"
    )]
    fn gemm_mixed(
        &self,
        a_f16: &MetalTensor,
        b_f16: &MetalTensor,
        output_f32: &mut MetalTensor,
        m: usize,
        n: usize,
        k: usize,
        transpose_b: bool,
    ) -> crate::Result<()> {
        let b_fp16_ref = b_f16.fp16.borrow();
        let (b_buf, b_off) = if let Some(ref fp16_buf) = *b_fp16_ref {
            (fp16_buf as &ProtocolObject<dyn MTLBuffer>, 0)
        } else {
            (
                &*b_f16.buffer as &ProtocolObject<dyn MTLBuffer>,
                b_f16.offset,
            )
        };
        self.run_compute("gemm-mixed-f16-to-f32", |enc| {
            enc.setComputePipelineState(&self.kernels.gemm_f16w_f32a);
            set_buffer(enc, &a_f16.buffer, a_f16.offset, 0);
            set_buffer(enc, b_buf, b_off, 1);
            set_buffer(enc, &output_f32.buffer, output_f32.offset, 2);
            set_u32_param(enc, m as u32, 3);
            set_u32_param(enc, n as u32, 4);
            set_u32_param(enc, k as u32, 5);
            set_u32_param(enc, u32::from(transpose_b), 6);
            set_u32_param(enc, (m * k) as u32, 7);
            #[expect(
                clippy::if_same_then_else,
                reason = "n*k and k*n compute to the same value but preserve shape semantics"
            )]
            let b_stride = if transpose_b {
                (n * k) as u32
            } else {
                (k * n) as u32
            };
            set_u32_param(enc, b_stride, 8);
            set_u32_param(enc, (m * n) as u32, 9);
            let grid = MTLSize {
                width: n.div_ceil(64),
                height: m.div_ceil(64),
                depth: 1,
            };
            let threads = MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            };
            enc.dispatchThreadgroups_threadsPerThreadgroup(grid, threads);
            Ok(())
        })
    }

    // FP16 operations for full half-precision pipeline
    // =======================================================================

    fn alloc_zeros_f16(&self, n: usize) -> crate::Result<MetalTensor> {
        self.alloc_f16_tensor(n)
    }

    fn f32_to_f16(
        &self,
        output: &mut MetalTensor,
        input: &MetalTensor,
        n: usize,
    ) -> crate::Result<()> {
        self.run_compute("f32-to-f16", |enc| {
            enc.setComputePipelineState(&self.kernels.f32_to_f16);
            set_buffer(enc, &output.buffer, output.offset, 0);
            set_buffer(enc, &input.buffer, input.offset, 1);
            set_i32_param(enc, n as i32, 2);
            dispatch_1d(enc, &self.kernels.f32_to_f16, n);
            Ok(())
        })
    }

    fn f16_to_f32(
        &self,
        output: &mut MetalTensor,
        input: &MetalTensor,
        n: usize,
    ) -> crate::Result<()> {
        self.run_compute("f16-to-f32", |enc| {
            enc.setComputePipelineState(&self.kernels.f16_to_f32);
            set_buffer(enc, &output.buffer, output.offset, 0);
            set_buffer(enc, &input.buffer, input.offset, 1);
            set_i32_param(enc, n as i32, 2);
            dispatch_1d(enc, &self.kernels.f16_to_f32, n);
            Ok(())
        })
    }

    #[expect(
        clippy::many_single_char_names,
        reason = "a, b, m, n, k are standard GEMM parameter names from BLAS"
    )]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "m/n/k are small ML dimensions that fit in u32"
    )]
    fn gemm_f16(
        &self,
        a: &MetalTensor,
        b: &MetalTensor,
        output: &mut MetalTensor,
        m: usize,
        n: usize,
        k: usize,
        transpose_b: bool,
    ) -> crate::Result<()> {
        // RIPVEC_NO_MPS=1 uses the tiled compute GEMM to eliminate encoder
        // transitions (88 per ModernBERT forward pass → 0).
        static USE_COMPUTE_GEMM: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

        // B may be a weight tensor (FP32 main + FP16 in .fp16 field)
        // or an FP16 activation (half in main buffer, no .fp16 field).
        let b_fp16 = b.fp16.borrow();
        let (b_buf, b_off) = if let Some(ref fp16_buf) = *b_fp16 {
            (fp16_buf as &ProtocolObject<dyn MTLBuffer>, 0)
        } else {
            (&*b.buffer as &ProtocolObject<dyn MTLBuffer>, b.offset)
        };

        // INT8 block-quantized path: if B has Q8 data (block_q8_0 format),
        // dispatch the INT8 kernel. Eliminates MPS transitions + halves weight bandwidth.
        {
            let q8_ref = b.q8.borrow();
            if let Some(q8_buf) = &*q8_ref {
                let q8_tensor = MetalTensor::new(q8_buf.clone(), 0);
                return self.gemm_q8(a, &q8_tensor, output, m, n, k, transpose_b);
            }
        }

        let use_compute = *USE_COMPUTE_GEMM
            .get_or_init(|| std::env::var("RIPVEC_NO_MPS").is_ok_and(|v| v == "1"));

        if use_compute {
            // Fused: FP16 in → native simdgroup GEMM (FP32 accumulators) →
            // fused float→half store → FP16 out. No temp buffer.
            self.run_compute("gemm-fused-f16", |enc| {
                enc.setComputePipelineState(&self.kernels.gemm_f16w_f32a);
                set_buffer(enc, &a.buffer, a.offset, 0);
                set_buffer(enc, b_buf, b_off, 1);
                set_buffer(enc, &output.buffer, output.offset, 2);
                set_u32_param(enc, m as u32, 3);
                set_u32_param(enc, n as u32, 4);
                set_u32_param(enc, k as u32, 5);
                set_u32_param(enc, u32::from(transpose_b), 6);
                set_u32_param(enc, (m * k) as u32, 7);
                #[expect(
                    clippy::if_same_then_else,
                    reason = "n*k and k*n compute to the same value but preserve shape semantics"
                )]
                let b_stride = if transpose_b {
                    (n * k) as u32
                } else {
                    (k * n) as u32
                };
                set_u32_param(enc, b_stride, 8);
                set_u32_param(enc, (m * n) as u32, 9);
                let grid = MTLSize {
                    width: n.div_ceil(64),
                    height: m.div_ceil(64),
                    depth: 1,
                };
                let threads = MTLSize {
                    width: 128,
                    height: 1,
                    depth: 1,
                };
                enc.dispatchThreadgroups_threadsPerThreadgroup(grid, threads);
                Ok(())
            })
        } else {
            self.run_mps("mps-gemm-f16", |cmd_buf| {
                dispatch_mps_gemm(
                    cmd_buf,
                    &self.device,
                    &a.buffer,
                    a.offset,
                    b_buf,
                    b_off,
                    &output.buffer,
                    output.offset,
                    m,
                    n,
                    k,
                    transpose_b,
                    MPSDataType::Float16,
                    MPSDataType::Float16,
                );
                Ok(())
            })
        }
    }

    #[expect(
        clippy::many_single_char_names,
        reason = "a, b, m, n, k are standard GEMM parameter names from BLAS"
    )]
    fn gemm_batched_f16(
        &self,
        a: &MetalTensor,
        b: &MetalTensor,
        output: &mut MetalTensor,
        m: usize,
        n: usize,
        k: usize,
        transpose_b: bool,
        stride_a: usize,
        stride_b: usize,
        stride_c: usize,
        batch_count: usize,
    ) -> crate::Result<()> {
        self.run_compute("gemm-batched-f16", |enc| {
            dispatch_gemm_batched(
                enc,
                &self.kernels.gemm_batched_f16,
                &a.buffer,
                a.offset,
                &b.buffer,
                b.offset,
                &output.buffer,
                output.offset,
                m as u32,
                n as u32,
                k as u32,
                transpose_b,
                stride_a as u32,
                stride_b as u32,
                stride_c as u32,
                batch_count as u32,
            );
            Ok(())
        })
    }

    fn layer_norm_f16(
        &self,
        output: &mut MetalTensor,
        input: &MetalTensor,
        weight: &MetalTensor,
        bias: &MetalTensor,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> crate::Result<()> {
        let threads = 256.min(cols);
        self.run_compute("layer-norm-f16", |enc| {
            enc.setComputePipelineState(&self.kernels.layer_norm_f16);
            set_buffer(enc, &output.buffer, output.offset, 0);
            set_buffer(enc, &input.buffer, input.offset, 1);
            set_buffer(enc, &weight.buffer, weight.offset, 2);
            set_buffer(enc, &bias.buffer, bias.offset, 3);
            set_i32_param(enc, rows as i32, 4);
            set_i32_param(enc, cols as i32, 5);
            set_f32_param(enc, eps, 6);
            dispatch_rows(enc, &self.kernels.layer_norm_f16, rows, threads);
            Ok(())
        })
    }

    fn fused_scale_mask_softmax_f16(
        &self,
        scores: &mut MetalTensor,
        mask: &MetalTensor,
        batch: usize,
        num_heads: usize,
        seq_len: usize,
        scale: f32,
    ) -> crate::Result<()> {
        let total_rows = batch * num_heads * seq_len;
        let threads = 256.min(seq_len.next_power_of_two());
        self.run_compute("softmax-f16", |enc| {
            enc.setComputePipelineState(&self.kernels.fused_scale_mask_softmax_f16);
            set_buffer(enc, &scores.buffer, scores.offset, 0);
            set_buffer(enc, &mask.buffer, mask.offset, 1);
            set_i32_param(enc, batch as i32, 2);
            set_i32_param(enc, num_heads as i32, 3);
            set_i32_param(enc, seq_len as i32, 4);
            set_f32_param(enc, scale, 5);
            dispatch_rows(
                enc,
                &self.kernels.fused_scale_mask_softmax_f16,
                total_rows,
                threads,
            );
            Ok(())
        })
    }

    fn fused_scale_mask_softmax_windowed_f16(
        &self,
        scores: &mut MetalTensor,
        mask: &MetalTensor,
        batch: usize,
        num_heads: usize,
        seq_len: usize,
        scale: f32,
        window_size: usize,
    ) -> crate::Result<()> {
        let total_rows = batch * num_heads * seq_len;
        let threads = 256.min(seq_len.next_power_of_two());
        let half_window = window_size / 2;
        self.run_compute("softmax-windowed-f16", |enc| {
            enc.setComputePipelineState(&self.kernels.fused_scale_mask_softmax_windowed_f16);
            set_buffer(enc, &scores.buffer, scores.offset, 0);
            set_buffer(enc, &mask.buffer, mask.offset, 1);
            set_i32_param(enc, batch as i32, 2);
            set_i32_param(enc, num_heads as i32, 3);
            set_i32_param(enc, seq_len as i32, 4);
            set_f32_param(enc, scale, 5);
            set_i32_param(enc, half_window as i32, 6);
            dispatch_rows(
                enc,
                &self.kernels.fused_scale_mask_softmax_windowed_f16,
                total_rows,
                threads,
            );
            Ok(())
        })
    }

    fn qkv_split_f16(
        &self,
        q: &mut MetalTensor,
        k: &mut MetalTensor,
        v: &mut MetalTensor,
        qkv: &MetalTensor,
        batch: usize,
        seq: usize,
        hidden: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> crate::Result<()> {
        let total_head = batch * num_heads * seq * head_dim;
        self.run_compute("qkv-split-f16", |enc| {
            enc.setComputePipelineState(&self.kernels.qkv_split_f16);
            set_buffer(enc, &q.buffer, q.offset, 0);
            set_buffer(enc, &k.buffer, k.offset, 1);
            set_buffer(enc, &v.buffer, v.offset, 2);
            set_buffer(enc, &qkv.buffer, qkv.offset, 3);
            set_i32_param(enc, batch as i32, 4);
            set_i32_param(enc, seq as i32, 5);
            set_i32_param(enc, hidden as i32, 6);
            set_i32_param(enc, num_heads as i32, 7);
            set_i32_param(enc, head_dim as i32, 8);
            dispatch_1d(enc, &self.kernels.qkv_split_f16, total_head);
            Ok(())
        })
    }

    fn attn_reshape_f16(
        &self,
        output: &mut MetalTensor,
        input: &MetalTensor,
        batch: usize,
        seq: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> crate::Result<()> {
        let total = batch * seq * num_heads * head_dim;
        self.run_compute("attn-reshape-f16", |enc| {
            enc.setComputePipelineState(&self.kernels.attn_reshape_f16);
            set_buffer(enc, &output.buffer, output.offset, 0);
            set_buffer(enc, &input.buffer, input.offset, 1);
            set_i32_param(enc, batch as i32, 2);
            set_i32_param(enc, seq as i32, 3);
            set_i32_param(enc, num_heads as i32, 4);
            set_i32_param(enc, head_dim as i32, 5);
            dispatch_1d(enc, &self.kernels.attn_reshape_f16, total);
            Ok(())
        })
    }

    fn pad_to_batch_f16(
        &self,
        flat: &MetalTensor,
        padded: &mut MetalTensor,
        seq_lengths: &[usize],
        max_seq: usize,
        dim: usize,
    ) -> crate::Result<()> {
        let batch = seq_lengths.len();
        let total_out = batch * max_seq * dim;

        let mut cu: Vec<i32> = Vec::with_capacity(batch + 1);
        cu.push(0);
        let mut acc: i32 = 0;
        for &len in seq_lengths {
            acc += len as i32;
            cu.push(acc);
        }
        let cu_buf = make_i32_buffer(&self.device, &cu)?;

        // Use caller's pre-allocated buffer — do NOT re-allocate.
        let padded_buf = padded.buffer.clone();
        let padded_offset = padded.offset;
        let flat_buf = flat.buffer.clone();
        let flat_offset = flat.offset;

        self.run_compute("pad-to-batch-f16", |enc| {
            enc.setComputePipelineState(&self.kernels.pad_to_batch_f16);
            set_buffer(enc, &padded_buf, padded_offset, 0);
            set_buffer(enc, &flat_buf, flat_offset, 1);
            set_buffer(enc, &cu_buf, 0, 2);
            set_i32_param(enc, max_seq as i32, 3);
            set_i32_param(enc, dim as i32, 4);
            set_i32_param(enc, batch as i32, 5);
            dispatch_1d(enc, &self.kernels.pad_to_batch_f16, total_out);
            Ok(())
        })
    }

    fn unpad_from_batch_f16(
        &self,
        padded: &MetalTensor,
        flat: &mut MetalTensor,
        seq_lengths: &[usize],
        max_seq: usize,
        dim: usize,
    ) -> crate::Result<()> {
        let batch = seq_lengths.len();
        let total_tokens: usize = seq_lengths.iter().sum();

        let mut cu: Vec<i32> = Vec::with_capacity(batch + 1);
        cu.push(0);
        let mut acc: i32 = 0;
        for &len in seq_lengths {
            acc += len as i32;
            cu.push(acc);
        }
        let cu_buf = make_i32_buffer(&self.device, &cu)?;

        // Use caller's pre-allocated buffer — do NOT re-allocate (see unpad_from_batch).
        let flat_buf = flat.buffer.clone();
        let flat_offset = flat.offset;
        let padded_buf = padded.buffer.clone();
        let padded_offset = padded.offset;

        self.run_compute("unpad-from-batch-f16", |enc| {
            enc.setComputePipelineState(&self.kernels.unpad_from_batch_f16);
            set_buffer(enc, &flat_buf, flat_offset, 0);
            set_buffer(enc, &padded_buf, padded_offset, 1);
            set_buffer(enc, &cu_buf, 0, 2);
            set_i32_param(enc, max_seq as i32, 3);
            set_i32_param(enc, dim as i32, 4);
            set_i32_param(enc, total_tokens as i32, 5);
            dispatch_1d(enc, &self.kernels.unpad_from_batch_f16, total_tokens * dim);
            Ok(())
        })
    }

    fn rope_encode_f16(
        &self,
        qk: &mut MetalTensor,
        cos: &MetalTensor,
        sin: &MetalTensor,
        num_rows: usize,
        seq_len: usize,
        head_dim: usize,
        num_heads: usize,
    ) -> crate::Result<()> {
        let half_d = head_dim / 2;
        let total = num_rows * half_d;
        self.run_compute("rope-encode-f16", |enc| {
            enc.setComputePipelineState(&self.kernels.rope_encode_f16);
            set_buffer(enc, &qk.buffer, qk.offset, 0);
            set_buffer(enc, &cos.buffer, cos.offset, 1);
            set_buffer(enc, &sin.buffer, sin.offset, 2);
            set_i32_param(enc, num_rows as i32, 3);
            set_i32_param(enc, seq_len as i32, 4);
            set_i32_param(enc, head_dim as i32, 5);
            set_i32_param(enc, num_heads as i32, 6);
            dispatch_1d(enc, &self.kernels.rope_encode_f16, total);
            Ok(())
        })
    }

    fn geglu_f16(
        &self,
        value: &MetalTensor,
        gate: &MetalTensor,
        output: &mut MetalTensor,
        n: usize,
    ) -> crate::Result<()> {
        self.run_compute("geglu-f16", |enc| {
            enc.setComputePipelineState(&self.kernels.geglu_f16);
            set_buffer(enc, &output.buffer, output.offset, 0);
            set_buffer(enc, &value.buffer, value.offset, 1);
            set_buffer(enc, &gate.buffer, gate.offset, 2);
            set_i32_param(enc, n as i32, 3);
            dispatch_1d(enc, &self.kernels.geglu_f16, n);
            Ok(())
        })
    }

    fn fused_residual_layernorm_f16(
        &self,
        output: &mut MetalTensor,
        hidden: &MetalTensor,
        residual: &MetalTensor,
        weight: &MetalTensor,
        bias: &MetalTensor,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> crate::Result<()> {
        let threads = 256.min(cols);
        self.run_compute("fused-residual-layernorm-f16", |enc| {
            enc.setComputePipelineState(&self.kernels.fused_residual_layernorm_f16);
            set_buffer(enc, &output.buffer, output.offset, 0);
            set_buffer(enc, &hidden.buffer, hidden.offset, 1);
            set_buffer(enc, &residual.buffer, residual.offset, 2);
            set_buffer(enc, &weight.buffer, weight.offset, 3);
            set_buffer(enc, &bias.buffer, bias.offset, 4);
            set_i32_param(enc, rows as i32, 5);
            set_i32_param(enc, cols as i32, 6);
            set_f32_param(enc, eps, 7);
            dispatch_rows(
                enc,
                &self.kernels.fused_residual_layernorm_f16,
                rows,
                threads,
            );
            Ok(())
        })
    }

    fn residual_add_f16(
        &self,
        output: &mut MetalTensor,
        hidden: &MetalTensor,
        residual: &MetalTensor,
        n: usize,
    ) -> crate::Result<()> {
        self.run_compute("residual-add-f16", |enc| {
            enc.setComputePipelineState(&self.kernels.residual_add_f16);
            set_buffer(enc, &output.buffer, output.offset, 0);
            set_buffer(enc, &hidden.buffer, hidden.offset, 1);
            set_buffer(enc, &residual.buffer, residual.offset, 2);
            set_i32_param(enc, n as i32, 3);
            dispatch_1d(enc, &self.kernels.residual_add_f16, n);
            Ok(())
        })
    }

    fn split_gate_value_f16(
        &self,
        first: &mut MetalTensor,
        second: &mut MetalTensor,
        input: &MetalTensor,
        rows: usize,
        cols: usize,
    ) -> crate::Result<()> {
        let total = rows * cols;
        self.run_compute("split-gate-value-f16", |enc| {
            enc.setComputePipelineState(&self.kernels.split_gate_value_f16);
            set_buffer(enc, &first.buffer, first.offset, 0);
            set_buffer(enc, &second.buffer, second.offset, 1);
            set_buffer(enc, &input.buffer, input.offset, 2);
            set_i32_param(enc, rows as i32, 3);
            set_i32_param(enc, cols as i32, 4);
            dispatch_1d(enc, &self.kernels.split_gate_value_f16, total);
            Ok(())
        })
    }

    // =======================================================================
    // INT8 weight quantization + dispatch
    // =======================================================================
}

// INT8 weight quantization + dispatch (Metal-specific, not on Driver trait).
impl MetalDriver {
    /// Quantize an FP16 weight tensor to block_q8_0 format (llama.cpp compatible).
    ///
    /// Input: `weights` — a [`MetalTensor`] whose `.fp16` field (or main buffer)
    /// contains `half[N * K]` in row-major order (N rows, K elements each).
    /// K must be a multiple of 32.
    ///
    /// Output: `MetalTensor` with `block_q8_0[N * K/32]` — each block is 34 bytes
    /// (2 byte half scale + 32 int8 values). Per-block symmetric quantization.
    ///
    /// # Panics
    ///
    /// Panics if `k` is not a multiple of 32.
    #[expect(unsafe_code, reason = "Metal SharedMode buffer contents access")]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "IEEE 754 bit manipulation requires controlled narrowing; values are clamped/masked before cast"
    )]
    #[expect(
        clippy::cast_ptr_alignment,
        reason = "Metal SharedMode buffer is u8-typed; we know the underlying data is u16-aligned FP16"
    )]
    pub fn quantize_weights_q8(
        &self,
        weights: &MetalTensor,
        n: usize,
        k: usize,
    ) -> crate::Result<MetalTensor> {
        // FP16 → FP32 conversion (IEEE 754 half-precision).
        #[inline]
        fn f16_to_f32(bits: u16) -> f32 {
            let sign = u32::from((bits >> 15) & 1);
            let exp = u32::from((bits >> 10) & 0x1F);
            let mant = u32::from(bits & 0x3FF);
            if exp == 0 {
                // Subnormal or zero
                let f = (mant as f32) * (1.0 / (1 << 24) as f32);
                if sign == 1 { -f } else { f }
            } else if exp == 31 {
                // Inf or NaN
                if mant == 0 {
                    if sign == 1 {
                        f32::NEG_INFINITY
                    } else {
                        f32::INFINITY
                    }
                } else {
                    f32::NAN
                }
            } else {
                let f_bits = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
                f32::from_bits(f_bits)
            }
        }

        // FP32 → FP16 for the scale value
        #[inline]
        fn f32_to_f16(f: f32) -> u16 {
            let bits = f.to_bits();
            let sign = (bits >> 16) & 0x8000;
            let exp = ((bits >> 23) & 0xFF).cast_signed() - 127 + 15;
            let mant = (bits >> 13) & 0x3FF;
            if exp <= 0 {
                sign as u16 // flush subnormals to zero
            } else if exp >= 31 {
                (sign | 0x7C00) as u16 // inf
            } else {
                (sign | (exp.cast_unsigned() << 10) | mant) as u16
            }
        }

        // Prefer the pre-converted FP16 buffer when available.
        let fp16_borrow = weights.fp16.borrow();
        let (fp16_buf, fp16_off_bytes) = if let Some(ref buf) = *fp16_borrow {
            (buf as &ProtocolObject<dyn MTLBuffer>, 0usize)
        } else {
            (
                &*weights.buffer as &ProtocolObject<dyn MTLBuffer>,
                weights.offset,
            )
        };

        // Read FP16 data as raw u16 from the CPU-accessible shared buffer.
        let fp16_raw: &[u16] = unsafe {
            let base = fp16_buf.contents().as_ptr().cast::<u8>();
            let ptr = base.add(fp16_off_bytes).cast::<u16>();
            std::slice::from_raw_parts(ptr, n * k)
        };

        // Quantize in block_q8_0 format: 32-element blocks, each with own half scale.
        assert!(
            k.is_multiple_of(32),
            "K must be a multiple of 32 for block_q8_0"
        );
        let blocks_per_row = k / 32;
        let total_blocks = n * blocks_per_row;
        let block_size = 34_usize; // 2 (half d) + 32 (int8 qs)

        let mut buf: Vec<u8> = Vec::with_capacity(total_blocks * block_size);

        for row in 0..n {
            let row_data = &fp16_raw[row * k..(row + 1) * k];
            for blk in 0..blocks_per_row {
                let start = blk * 32;
                // Find max abs in this 32-element block
                let mut max_abs: f32 = 0.0;
                for i in 0..32 {
                    let v = f16_to_f32(row_data[start + i]).abs();
                    if v > max_abs {
                        max_abs = v;
                    }
                }
                let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
                let inv_scale = 1.0 / scale;

                // Write half d (2 bytes, little-endian)
                buf.extend_from_slice(&f32_to_f16(scale).to_le_bytes());

                // Write 32 quantized int8 values
                for i in 0..32 {
                    let v = f16_to_f32(row_data[start + i]);
                    let q = (v * inv_scale).round().clamp(-127.0, 127.0) as i8;
                    buf.push(q.cast_unsigned());
                }
            }
        }

        // Upload to Metal buffer
        let buf_size = buf.len() as NSUInteger;
        let q8_buf = unsafe {
            self.device.newBufferWithBytes_length_options(
                std::ptr::NonNull::new(buf.as_ptr() as *mut _)
                    .ok_or_else(|| crate::Error::Metal("q8 block data null".into()))?,
                buf_size,
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or_else(|| crate::Error::Metal(format!("q8 buffer failed ({buf_size} bytes)")))?;

        Ok(MetalTensor::new(q8_buf, 0))
    }

    /// Dispatch the INT8-weight GEMM kernel (block_q8_0 format).
    ///
    /// Computes `output_f16 = A_f16 @ B_q8.T` (when `transpose_b = true`)
    /// dequantizing on-the-fly using per-block scales embedded in block_q8_0.
    ///
    /// - `a_f16`: FP16 activations `[M, K]`
    /// - `b_q8`: block_q8_0 quantized weights `[N * K/32]` blocks (34 bytes each)
    /// - `output_f16`: FP16 output `[M, N]`
    #[expect(
        clippy::cast_possible_truncation,
        reason = "m/n/k are small ML dimensions that fit in u32"
    )]
    pub fn gemm_q8(
        &self,
        a_f16: &MetalTensor,
        b_q8: &MetalTensor, // block_q8_0 data (scales embedded in blocks)
        output_f16: &mut MetalTensor,
        m: usize,
        n: usize,
        k: usize,
        transpose_b: bool,
    ) -> crate::Result<()> {
        let blocks_per_row = k / 32;
        self.run_compute("gemm-q8w-f16a", |enc| {
            enc.setComputePipelineState(&self.kernels.gemm_q8w);
            set_buffer(enc, &a_f16.buffer, a_f16.offset, 0);
            set_buffer(enc, &b_q8.buffer, b_q8.offset, 1);
            set_buffer(enc, &output_f16.buffer, output_f16.offset, 2);
            set_u32_param(enc, m as u32, 3);
            set_u32_param(enc, n as u32, 4);
            set_u32_param(enc, k as u32, 5);
            set_u32_param(enc, u32::from(transpose_b), 6);
            set_u32_param(enc, (m * k) as u32, 7); // stride_A (half elements)
            set_u32_param(enc, blocks_per_row as u32, 8); // stride_B (blocks per row)
            set_u32_param(enc, (m * n) as u32, 9); // stride_C (half elements)
            let grid = MTLSize {
                width: n.div_ceil(64),
                height: m.div_ceil(64),
                depth: 1,
            };
            let threads = MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            };
            enc.dispatchThreadgroups_threadsPerThreadgroup(grid, threads);
            Ok(())
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// CPU scalar FP32→FP16 conversion for test verification.
    #[inline]
    fn f32_to_f16(f: f32) -> u16 {
        let bits = f.to_bits();
        let sign = (bits >> 31) & 1;
        let exp = ((bits >> 23) & 0xFF) as i32;
        let mant = bits & 0x7F_FFFF;
        if exp == 0 {
            return (sign << 15) as u16;
        }
        let new_exp = exp - 127 + 15;
        if new_exp <= 0 {
            return (sign << 15) as u16;
        }
        if new_exp >= 31 {
            return ((sign << 15) | (31 << 10)) as u16;
        }
        ((sign << 15) | (new_exp as u32) << 10 | (mant >> 13)) as u16
    }

    /// CPU scalar FP16→FP32 conversion for test verification.
    #[inline]
    fn f16_to_f32(bits: u16) -> f32 {
        let sign = ((bits >> 15) & 1) as u32;
        let exp = ((bits >> 10) & 0x1F) as u32;
        let mant = (bits & 0x3FF) as u32;
        if exp == 0 {
            return 0.0;
        }
        if exp == 31 {
            return if mant == 0 { f32::INFINITY } else { f32::NAN };
        }
        f32::from_bits((sign << 31) | ((exp + 112) << 23) | (mant << 13))
    }

    #[test]
    fn metal_driver_creates() {
        let driver = MetalDriver::new().unwrap();
        // Verify it initialized by allocating a small tensor.
        let _tensor = driver.alloc_tensor(16).unwrap();
    }

    /// Verify that `MetalDriver` satisfies `Send + Sync` bounds.
    #[test]
    fn driver_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MetalDriver>();
    }

    /// Basic GELU test: apply GELU to known values and verify output.
    #[test]
    #[expect(unsafe_code, reason = "Metal buffer readback requires unsafe")]
    fn gelu_smoke_test() {
        let driver = MetalDriver::new().unwrap();
        let n = 4;

        // Allocate and fill a tensor with known values
        let mut tensor = driver.alloc_tensor(n).unwrap();
        let data: [f32; 4] = [0.0, 1.0, -1.0, 2.0];
        unsafe {
            let ptr = tensor.buffer.contents().as_ptr().cast::<f32>();
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, n);
        }

        driver.gelu(&mut tensor, n).unwrap();

        // Read back results
        let result = unsafe {
            let ptr = tensor.buffer.contents().as_ptr().cast::<f32>();
            core::slice::from_raw_parts(ptr, n)
        };

        // GELU(0) = 0
        assert!(
            (result[0]).abs() < 1e-4,
            "GELU(0) should be ~0, got {}",
            result[0]
        );
        // GELU(1) ~= 0.8412
        assert!(
            (result[1] - 0.8412).abs() < 0.01,
            "GELU(1) should be ~0.8412, got {}",
            result[1]
        );
        // GELU(-1) ~= -0.1588
        assert!(
            (result[2] - (-0.1588)).abs() < 0.01,
            "GELU(-1) should be ~-0.1588, got {}",
            result[2]
        );
    }

    /// Basic L2 normalize test.
    #[test]
    #[expect(unsafe_code, reason = "Metal buffer readback requires unsafe")]
    fn l2_normalize_smoke_test() {
        let driver = MetalDriver::new().unwrap();
        let rows = 1;
        let cols = 4;

        let mut tensor = driver.alloc_tensor(rows * cols).unwrap();
        let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        unsafe {
            let ptr = tensor.buffer.contents().as_ptr().cast::<f32>();
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, cols);
        }

        driver.l2_normalize(&mut tensor, rows, cols).unwrap();

        let result = unsafe {
            let ptr = tensor.buffer.contents().as_ptr().cast::<f32>();
            core::slice::from_raw_parts(ptr, cols)
        };

        // Check that the result is L2-normalized (norm ~= 1.0)
        let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "L2 norm should be ~1.0, got {norm}",
        );
    }

    /// INT8 weight GEMM correctness test (block_q8_0 format).
    /// C_f16[M,N] = A_f16[M,K] × B_q8^T[N,K] with per-block dequant.
    #[test]
    #[expect(unsafe_code, reason = "Metal buffer readback requires unsafe")]
    fn gemm_q8_correctness() {
        let driver = MetalDriver::new().unwrap();
        let m: usize = 128;
        let n: usize = 768;
        let k: usize = 768;

        // --- Build A (FP16): varying values across K to catch axis transposition ---
        // Keep values small to avoid FP16 overflow at large K (max ~65504).
        let mut a_f32: Vec<f32> = vec![0.0; m * k];
        for i in 0..m {
            for j in 0..k {
                a_f32[i * k + j] = ((i % 8 + 1) as f32) * 0.001 + (j as f32) * 0.0001;
            }
        }

        // f32_to_f16 and f16_to_f32 are defined at the tests module level.

        let a_f16: Vec<u16> = a_f32.iter().map(|&f| f32_to_f16(f)).collect();

        // --- Build B in block_q8_0 format ---
        // Each block: 2 bytes (half scale) + 32 bytes (int8 values) = 34 bytes.
        assert!(k % 32 == 0);
        let blocks_per_row = k / 32;
        let block_size = 34_usize;

        // Build raw q8 values and per-block scales for CPU reference,
        // plus the packed block_q8_0 byte stream for the GPU.
        let mut b_q8_vals: Vec<i8> = vec![0i8; n * k];
        let mut block_scales_f16: Vec<u16> = vec![0u16; n * blocks_per_row];
        let mut b_bytes: Vec<u8> = Vec::with_capacity(n * blocks_per_row * block_size);

        for j in 0..n {
            for blk in 0..blocks_per_row {
                let scale = 0.005 + (j % 10) as f32 * 0.001 + (blk % 3) as f32 * 0.0005;
                let scale_f16 = f32_to_f16(scale);
                block_scales_f16[j * blocks_per_row + blk] = scale_f16;

                // Write half d (2 bytes, little-endian)
                b_bytes.extend_from_slice(&scale_f16.to_le_bytes());

                // Write 32 quantized int8 values
                for i in 0..32 {
                    let col = blk * 32 + i;
                    let q = ((col as i32 * 3 + j as i32 * 7) % 255 - 127) as i8;
                    b_q8_vals[j * k + col] = q;
                    b_bytes.push(q as u8);
                }
            }
        }

        // --- Expected output: CPU reference using per-block scales ---
        let mut expected = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f64;
                for kk in 0..k {
                    let a_val = f16_to_f32(a_f16[i * k + kk]) as f64;
                    let blk_idx = kk / 32;
                    let scale = f16_to_f32(block_scales_f16[j * blocks_per_row + blk_idx]) as f64;
                    let b_val = (b_q8_vals[j * k + kk] as f64) * scale;
                    sum += a_val * b_val;
                }
                expected[i * n + j] = sum as f32;
            }
        }

        // --- Upload to GPU ---
        let a_buf = unsafe {
            driver
                .device
                .newBufferWithBytes_length_options(
                    std::ptr::NonNull::new(a_f16.as_ptr() as *mut _).unwrap(),
                    (m * k * 2) as NSUInteger,
                    MTLResourceOptions::StorageModeShared,
                )
                .unwrap()
        };
        let a_tensor = MetalTensor::new(a_buf, 0);

        let b_buf = unsafe {
            driver
                .device
                .newBufferWithBytes_length_options(
                    std::ptr::NonNull::new(b_bytes.as_ptr() as *mut _).unwrap(),
                    b_bytes.len() as NSUInteger,
                    MTLResourceOptions::StorageModeShared,
                )
                .unwrap()
        };
        let b_tensor = MetalTensor::new(b_buf, 0);

        // Output buffer: FP16, M*N elements
        let mut output_tensor = driver.alloc_f16_tensor(m * n).unwrap();

        // --- Dispatch ---
        driver
            .gemm_q8(&a_tensor, &b_tensor, &mut output_tensor, m, n, k, true)
            .unwrap();

        // --- Readback FP16 output ---
        let result_f16 = unsafe {
            let ptr = output_tensor.buffer.contents().as_ptr() as *const u16;
            std::slice::from_raw_parts(ptr, m * n)
        };
        let result_f32: Vec<f32> = result_f16.iter().map(|&b| f16_to_f32(b)).collect();

        // --- Compare ---
        let mut max_err: f32 = 0.0;
        for i in 0..m {
            for j in 0..n {
                let got = result_f32[i * n + j];
                let exp = expected[i * n + j];
                let err = (got - exp).abs();
                if err > max_err {
                    max_err = err;
                }
                assert!(
                    err < 0.05,
                    "C[{i},{j}] mismatch: got={got}, expected={exp}, err={err}",
                );
            }
        }
        eprintln!("[gemm_q8_correctness] max error: {max_err:.6}, PASS");
    }

    /// FP16 weight GEMM correctness test.
    /// C_f16[M,N] = A_f16[M,K] × B_f16^T[N,K] using gemm_f16w_f32a_kernel.
    #[test]
    #[expect(unsafe_code, reason = "Metal buffer readback requires unsafe")]
    fn gemm_f16_correctness() {
        let driver = MetalDriver::new().unwrap();
        let m: usize = 128;
        let n: usize = 768;
        let k: usize = 768;

        // --- Build A (FP16): varying values across K ---
        let mut a_f32: Vec<f32> = vec![0.0; m * k];
        for i in 0..m {
            for j in 0..k {
                a_f32[i * k + j] = ((i % 8 + 1) as f32) * 0.001 + (j as f32) * 0.0001;
            }
        }

        // f32_to_f16 and f16_to_f32 are defined at the tests module level.

        let a_f16: Vec<u16> = a_f32.iter().map(|&f| f32_to_f16(f)).collect();

        // --- Build B (FP16): varying values ---
        let mut b_f32: Vec<f32> = vec![0.0; n * k];
        for j in 0..n {
            for kk in 0..k {
                b_f32[j * k + kk] = ((kk % 5 + 1) as f32) * 0.001 + (j % 10) as f32 * 0.0002;
            }
        }
        let b_f16: Vec<u16> = b_f32.iter().map(|&f| f32_to_f16(f)).collect();

        // --- Expected output: CPU reference using FP16 values ---
        let mut expected = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f64;
                for kk in 0..k {
                    let a_val = f16_to_f32(a_f16[i * k + kk]) as f64;
                    let b_val = f16_to_f32(b_f16[j * k + kk]) as f64;
                    sum += a_val * b_val;
                }
                expected[i * n + j] = sum as f32;
            }
        }

        // --- Upload to GPU ---
        let a_buf = unsafe {
            driver
                .device
                .newBufferWithBytes_length_options(
                    std::ptr::NonNull::new(a_f16.as_ptr() as *mut _).unwrap(),
                    (m * k * 2) as NSUInteger,
                    MTLResourceOptions::StorageModeShared,
                )
                .unwrap()
        };
        let a_tensor = MetalTensor::new(a_buf, 0);

        let b_buf = unsafe {
            driver
                .device
                .newBufferWithBytes_length_options(
                    std::ptr::NonNull::new(b_f16.as_ptr() as *mut _).unwrap(),
                    (n * k * 2) as NSUInteger,
                    MTLResourceOptions::StorageModeShared,
                )
                .unwrap()
        };
        let b_tensor = MetalTensor::new(b_buf, 0);

        // Output buffer: FP16, M*N elements
        let output_tensor = driver.alloc_f16_tensor(m * n).unwrap();

        // --- Dispatch gemm_f16w_f32a_kernel directly ---
        driver
            .run_compute("gemm-f16-test", |enc| {
                enc.setComputePipelineState(&driver.kernels.gemm_f16w_f32a);
                set_buffer(enc, &a_tensor.buffer, a_tensor.offset, 0);
                set_buffer(enc, &b_tensor.buffer, b_tensor.offset, 1);
                set_buffer(enc, &output_tensor.buffer, output_tensor.offset, 2);
                set_u32_param(enc, m as u32, 3);
                set_u32_param(enc, n as u32, 4);
                set_u32_param(enc, k as u32, 5);
                set_u32_param(enc, 1_u32, 6); // transB = true (B is [N,K])
                set_u32_param(enc, (m * k) as u32, 7); // stride_A
                set_u32_param(enc, (n * k) as u32, 8); // stride_B
                set_u32_param(enc, (m * n) as u32, 9); // stride_C
                let grid = MTLSize {
                    width: n.div_ceil(64),
                    height: m.div_ceil(64),
                    depth: 1,
                };
                let threads = MTLSize {
                    width: 128,
                    height: 1,
                    depth: 1,
                };
                enc.dispatchThreadgroups_threadsPerThreadgroup(grid, threads);
                Ok(())
            })
            .unwrap();

        // --- Readback FP16 output ---
        let result_f16 = unsafe {
            let ptr = output_tensor.buffer.contents().as_ptr() as *const u16;
            std::slice::from_raw_parts(ptr, m * n)
        };
        let result_f32: Vec<f32> = result_f16.iter().map(|&b| f16_to_f32(b)).collect();

        // --- Compare ---
        let mut max_err: f32 = 0.0;
        for i in 0..m {
            for j in 0..n {
                let got = result_f32[i * n + j];
                let exp = expected[i * n + j];
                let err = (got - exp).abs();
                if err > max_err {
                    max_err = err;
                }
                assert!(
                    err < 0.05,
                    "C[{i},{j}] mismatch: got={got}, expected={exp}, err={err}",
                );
            }
        }
        eprintln!("[gemm_f16_correctness] max error: {max_err:.6}, PASS");
    }

    /// Test `to_host` readback.
    #[test]
    #[expect(unsafe_code, reason = "Metal buffer fill requires unsafe")]
    fn to_host_readback() {
        let driver = MetalDriver::new().unwrap();
        let batch = 2;
        let dim = 3;

        let tensor = driver.alloc_tensor(batch * dim).unwrap();
        let data: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        unsafe {
            let ptr = tensor.buffer.contents().as_ptr().cast::<f32>();
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, batch * dim);
        }

        let result = driver.to_host(&tensor, batch, dim).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(result[1], vec![4.0, 5.0, 6.0]);
    }
}
