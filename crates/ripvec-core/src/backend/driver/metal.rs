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
use objc2_foundation::{NSString, NSUInteger};
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary,
    MTLResourceOptions, MTLSize,
};
use objc2_metal_performance_shaders::{
    MPSDataType, MPSMatrix, MPSMatrixDescriptor, MPSMatrixMultiplication,
};
use safetensors::SafeTensors;

use super::{BatchInputs, Driver};
use crate::backend::Encoding;
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
    /// The backing Metal buffer.
    pub buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Byte offset into `buffer` where this tensor's data starts.
    pub offset: usize,
}

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
    /// Residual add without bias: `output = hidden + residual`.
    residual_add: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Fused scale + padding mask + sliding window mask + softmax.
    fused_scale_mask_softmax_windowed: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// `RoPE` with pre-computed cos/sin tables.
    rope_cached: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Batched GEMM: same kernel with z-dimension for batch/head index.
    gemm_batched: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl KernelPipelines {
    /// Compile all MSL kernels and create pipeline states.
    fn compile(device: &ProtocolObject<dyn MTLDevice>) -> crate::Result<Self> {
        let library = compile_library(device, crate::backend::metal_kernels::KERNELS)?;
        let p = |name: &str| create_pipeline(device, &library, name);

        let gemm_library = compile_library(device, crate::backend::metal_kernels::GEMM_KERNEL)?;

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
            residual_add: p("residual_add_kernel")?,
            fused_scale_mask_softmax_windowed: p("fused_scale_mask_softmax_windowed_kernel")?,
            rope_cached: p("rope_cached_kernel")?,
            gemm_batched: create_pipeline(device, &gemm_library, "gemm_batched_kernel")?,
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

/// Dispatch a batched GEMM kernel with z-dimension for batch/head index.
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

    let a_row_bytes = k * a_elem_size;
    let c_row_bytes = n * 4; // output is always FP32

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
            MPSDataType::Float32,
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
/// Implements [`Driver`] for Apple Silicon GPUs. Each trait method creates its
/// own command buffer, encodes the operation, commits, and waits for completion.
///
/// # Thread safety
///
/// Metal device and command queue are thread-safe. `MetalDriver` is `Send + Sync`.
pub struct MetalDriver {
    /// Metal GPU device handle.
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    /// Command queue for submitting GPU work.
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    /// Pre-compiled MSL pipeline states.
    kernels: KernelPipelines,
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
        })
    }

    /// Borrow the underlying Metal device handle.
    ///
    /// Needed by weight-loading code that creates zero-copy buffers from
    /// memory-mapped safetensors files.
    pub fn device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }

    /// Allocate a new [`MetalTensor`] of `n` floats.
    ///
    /// # Errors
    ///
    /// Returns an error if Metal buffer allocation fails.
    pub fn alloc_tensor(&self, n: usize) -> crate::Result<MetalTensor> {
        let buffer = alloc_f32_buffer(&self.device, n)?;
        Ok(MetalTensor { buffer, offset: 0 })
    }

    /// Execute a compute operation: create command buffer + encoder, run `f`,
    /// end encoding, commit, and wait.
    fn run_compute<F>(&self, f: F) -> crate::Result<()>
    where
        F: FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>) -> crate::Result<()>,
    {
        let cmd_buf = new_command_buffer(&self.queue)?;
        let enc = new_encoder(&cmd_buf)?;
        f(&enc)?;
        enc.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        Ok(())
    }

    /// Execute an operation that uses MPS (encodes directly to command buffer).
    fn run_mps<F>(&self, f: F) -> crate::Result<()>
    where
        F: FnOnce(&ProtocolObject<dyn MTLCommandBuffer>) -> crate::Result<()>,
    {
        let cmd_buf = new_command_buffer(&self.queue)?;
        f(&cmd_buf)?;
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        Ok(())
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
        clippy::cast_possible_truncation,
        reason = "mmap + newBufferWithBytesNoCopy require unsafe FFI; config ints are small"
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
        let tensor_at = |offset: usize| -> MetalTensor {
            MetalTensor {
                buffer: weight_buffer.clone(),
                offset,
            }
        };

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

        let arch = ModernBertArch {
            weights,
            global_rope,
            local_rope,
            max_layers: None,
        };

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
    let size = (data.len() * core::mem::size_of::<f32>()) as NSUInteger;
    let buffer = unsafe {
        driver.device().newBufferWithBytes_length_options(
            std::ptr::NonNull::new(data.as_ptr() as *mut _)
                .ok_or_else(|| crate::Error::Metal("null data pointer".into()))?,
            size,
            MTLResourceOptions::StorageModeShared,
        )
    }
    .ok_or_else(|| crate::Error::Metal("RoPE buffer alloc failed".into()))?;
    Ok(MetalTensor { buffer, offset: 0 })
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

    fn alloc_zeros(&self, n: usize) -> crate::Result<MetalTensor> {
        self.alloc_tensor(n)
    }

    #[expect(
        unsafe_code,
        reason = "Metal buffer copy requires unsafe contents access"
    )]
    fn clone_tensor(&self, tensor: &MetalTensor, n: usize) -> crate::Result<MetalTensor> {
        let new_tensor = self.alloc_tensor(n)?;
        let byte_count = n * core::mem::size_of::<f32>();
        unsafe {
            let src = tensor
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(tensor.offset);
            let dst = new_tensor.buffer.contents().as_ptr().cast::<u8>();
            std::ptr::copy_nonoverlapping(src, dst, byte_count);
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
        self.run_compute(|enc| {
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

        Ok(BatchInputs {
            input_ids: MetalTensor {
                buffer: input_ids_buf,
                offset: 0,
            },
            attention_mask: MetalTensor {
                buffer: attn_mask_int_buf,
                offset: 0,
            },
            token_type_ids: MetalTensor {
                buffer: token_type_ids_buf,
                offset: 0,
            },
            position_ids: MetalTensor {
                buffer: position_ids_buf,
                offset: 0,
            },
            float_mask: MetalTensor {
                buffer: float_mask_buf,
                offset: 0,
            },
            pooling_mask: MetalTensor {
                buffer: pooling_mask_buf,
                offset: 0,
            },
            batch,
            max_seq,
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

        self.run_compute(|enc| {
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
        self.run_compute(|enc| {
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
        self.run_compute(|enc| {
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
        self.run_mps(|cmd_buf| {
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
        self.run_compute(|enc| {
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
        let threads = 256.min(seq_len);
        self.run_compute(|enc| {
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
        let threads = 256.min(seq_len);
        let half_window = window_size / 2;
        self.run_compute(|enc| {
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
        self.run_compute(|enc| {
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
        self.run_compute(|enc| {
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
        self.run_compute(|enc| {
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
        self.run_compute(|enc| {
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
        self.run_compute(|enc| {
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
        self.run_compute(|enc| {
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
        self.run_compute(|enc| {
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
        self.run_compute(|enc| {
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
        self.run_compute(|enc| {
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
        self.run_compute(|enc| {
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
        self.run_compute(|enc| {
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
        self.run_compute(|enc| {
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
        self.run_compute(|enc| {
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
        self.run_compute(|enc| {
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
        self.run_compute(|enc| {
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
        self.run_compute(|enc| {
            enc.setComputePipelineState(&self.kernels.l2_normalize);
            set_buffer(enc, &data.buffer, data.offset, 0);
            set_i32_param(enc, rows as i32, 1);
            set_i32_param(enc, cols as i32, 2);
            dispatch_rows(enc, &self.kernels.l2_normalize, rows, threads);
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
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

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
