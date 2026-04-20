//! CUDA compute driver for `ModernBERT` inference.
//!
//! Implements the [`Driver`] trait using cudarc for device management, cuBLAS
//! for matrix multiplications (FP32 TF32 and FP16 tensor core paths), and
//! NVRTC-compiled CUDA kernels for activations, layer normalization, softmax,
//! embedding lookup, RoPE, and attention reshaping.
//!
//! Supports full FP16 pipeline: embedding in FP32, all encoder layers in FP16
//! (tensor core GEMM + half-precision kernels), final norm + pooling in FP32.

use std::path::Path;
use std::sync::Arc;

use crate::backend::nvrtc_cubin::compile_cubin;
use cudarc::cublas::{sys, CudaBlas};
use cudarc::cublaslt::{self, CudaBlasLT, MatmulShared};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, DevicePtrMut,
    LaunchConfig, PushKernelArg,
};
use safetensors::SafeTensors;

use super::{BatchInputs, Driver};
use crate::backend::arch::modern_bert::{
    ModernBertArch, ModernBertLayerWeights, ModernBertWeights, RopeCache,
};
use crate::backend::Encoding;

// ---------------------------------------------------------------------------
// Error helper
// ---------------------------------------------------------------------------

/// Convert any cudarc/cublas error into our crate error.
fn cuda_err(e: impl std::fmt::Display) -> crate::Error {
    crate::Error::Cuda(e.to_string())
}

// ---------------------------------------------------------------------------
// Launch helpers
// ---------------------------------------------------------------------------

/// Standard 1D launch config for `n` elements with 256 threads per block.
#[expect(
    clippy::cast_possible_truncation,
    reason = "GPU launch grids are always <2^32; n is bounded by GPU memory"
)]
fn launch_1d(n: usize) -> LaunchConfig {
    let threads = 256_u32;
    let blocks = (n as u32).div_ceil(threads);
    LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Per-row launch with shared memory (one block per row).
#[expect(
    clippy::cast_possible_truncation,
    reason = "row count is bounded by GPU memory; always fits in u32"
)]
fn launch_per_row(rows: usize, threads: u32, shared_bytes: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: shared_bytes,
    }
}

/// Launch a kernel builder, discarding the optional event pair return value.
///
/// cudarc 0.19 `launch()` returns `Result<Option<(CudaEvent, CudaEvent)>>`.
/// We don't need the events for synchronisation (we use `stream.synchronize()`
/// in `end_batch`), so this helper discards them cleanly.
///
/// # Safety
///
/// The caller must ensure the kernel arguments and launch config are valid.
#[expect(unsafe_code, reason = "wraps the unsafe launch call")]
unsafe fn launch_kernel(
    mut builder: cudarc::driver::LaunchArgs<'_>,
    cfg: LaunchConfig,
) -> crate::Result<()> {
    let _ = unsafe { builder.launch(cfg) }.map_err(cuda_err)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// CudaTensor
// ---------------------------------------------------------------------------

/// Opaque GPU tensor handle: FP32 device buffer + optional FP16/INT8 companions.
///
/// The FP16 companion is created at weight-load time for GEMM weight matrices.
/// FP16 GEMM paths read from the companion; FP32 kernels read from the primary.
/// The INT8 companion enables cuBLASLt INT8 tensor core GEMMs with fused scaling.
pub struct CudaTensor {
    /// Primary FP32 device buffer.
    f32_buf: CudaSlice<f32>,
    /// Optional FP16 companion for weight matrices (used by FP16 GEMM path).
    fp16: Option<CudaSlice<u16>>,
    /// Optional INT8 quantized weights (per-channel symmetric).
    int8: Option<CudaSlice<i8>>,
    /// Per-column scale factors for INT8 weights: `fp_value = int8_value * scale[col]`.
    int8_col_scales: Option<CudaSlice<f32>>,
}

impl CudaTensor {
    /// Create a new tensor with no FP16 or INT8 companion.
    fn new(buf: CudaSlice<f32>) -> Self {
        Self {
            f32_buf: buf,
            fp16: None,
            int8: None,
            int8_col_scales: None,
        }
    }

    /// Create a new tensor wrapping an FP16 buffer stored as the primary.
    ///
    /// The FP32 buffer is a zero-length placeholder. Used for FP16-only
    /// intermediate tensors (QKV, scores, etc. in the FP16 forward path).
    fn new_f16_only(f16_buf: CudaSlice<u16>, dummy_f32: CudaSlice<f32>) -> Self {
        Self {
            f32_buf: dummy_f32,
            fp16: Some(f16_buf),
            int8: None,
            int8_col_scales: None,
        }
    }

    /// Get the FP16 buffer, falling back to the primary FP32 buffer reinterpreted.
    fn fp16_ref(&self) -> Option<&CudaSlice<u16>> {
        self.fp16.as_ref()
    }
}

// SAFETY: CudaSlice is internally an Arc around a device allocation.
// The CUDA context is thread-safe when using a single device.
#[expect(
    unsafe_code,
    reason = "CudaSlice is Arc-based; single-device CUDA is thread-safe"
)]
unsafe impl Send for CudaTensor {}
// SAFETY: All mutable access goes through `&mut CudaTensor` or interior
// mutability in CudaDriver (RefCell, not shared across threads).
#[expect(unsafe_code, reason = "no shared mutable state without &mut")]
unsafe impl Sync for CudaTensor {}

// ---------------------------------------------------------------------------
// CUDA kernel source (compiled at runtime via NVRTC)
// ---------------------------------------------------------------------------

/// All CUDA kernels for `ModernBERT` inference.
///
/// Compiled once at driver creation via NVRTC. Includes FP32 kernels, FP16
/// kernels (using PTX inline asm for half-precision conversion), and new
/// kernels for `ModernBERT`-specific operations (GeGLU, sliding window, etc.).
const MODERN_KERNELS: &str = r#"
// =========================================================================
// FP32 kernels
// =========================================================================

extern "C" __global__ void gelu_kernel(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        x[i] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
    }
}

extern "C" __global__ void layer_norm_kernel(
    float* output, const float* input, const float* weight, const float* bias,
    int rows, int cols, float eps
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float sdata[];
    float* s_sum = sdata;
    float* s_sq = sdata + blockDim.x;

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        local_sum += input[row * cols + i];
    }
    s_sum[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
        __syncthreads();
    }
    float mean = s_sum[0] / (float)cols;
    __syncthreads();

    float local_sq = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float diff = input[row * cols + i] - mean;
        local_sq += diff * diff;
    }
    s_sq[threadIdx.x] = local_sq;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) s_sq[threadIdx.x] += s_sq[threadIdx.x + stride];
        __syncthreads();
    }
    float inv_std = rsqrtf(s_sq[0] / (float)cols + eps);
    __syncthreads();

    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        int idx = row * cols + i;
        output[idx] = (input[idx] - mean) * inv_std * weight[i] + bias[i];
    }
}

extern "C" __global__ void fused_residual_layernorm_kernel(
    float* output, const float* hidden, const float* residual,
    const float* weight, const float* bias,
    int rows, int cols, float eps
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    extern __shared__ float sdata[];

    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float val = hidden[row * cols + i] + residual[row * cols + i];
        output[row * cols + i] = val;
        thread_sum += val;
    }
    sdata[threadIdx.x] = thread_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float mean = sdata[0] / (float)cols;

    float thread_var = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float diff = output[row * cols + i] - mean;
        thread_var += diff * diff;
    }
    sdata[threadIdx.x] = thread_var;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float inv_std = rsqrtf(sdata[0] / (float)cols + eps);

    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float val = (output[row * cols + i] - mean) * inv_std;
        output[row * cols + i] = val * weight[i] + bias[i];
    }
}

extern "C" __global__ void fused_scale_mask_softmax_kernel(
    float* scores, const float* mask,
    int batch, int num_heads, int seq, float scale
) {
    int row = blockIdx.x;
    int total_rows = batch * num_heads * seq;
    if (row >= total_rows) return;

    extern __shared__ float sdata[];
    float* row_data = scores + row * seq;
    int b = row / (num_heads * seq);

    float thread_max = -1e30f;
    for (int i = threadIdx.x; i < seq; i += blockDim.x) {
        float val = row_data[i] * scale + mask[b * seq + i];
        row_data[i] = val;
        thread_max = fmaxf(thread_max, val);
    }
    sdata[threadIdx.x] = thread_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    float row_max = sdata[0];

    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < seq; i += blockDim.x) {
        float val = __expf(row_data[i] - row_max);
        row_data[i] = val;
        thread_sum += val;
    }
    sdata[threadIdx.x] = thread_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float total = sdata[0];

    float inv_sum = 1.0f / fmaxf(total, 1e-12f);
    for (int i = threadIdx.x; i < seq; i += blockDim.x) {
        row_data[i] *= inv_sum;
    }
}

extern "C" __global__ void fused_scale_mask_softmax_windowed_kernel(
    float* scores, const float* mask,
    int batch, int num_heads, int seq, float scale, int window_size
) {
    int row = blockIdx.x;
    int total_rows = batch * num_heads * seq;
    if (row >= total_rows) return;

    extern __shared__ float sdata[];
    float* row_data = scores + row * seq;
    int b = row / (num_heads * seq);
    int q_pos = row % seq;
    int half_w = window_size / 2;

    float thread_max = -1e30f;
    for (int i = threadIdx.x; i < seq; i += blockDim.x) {
        int dist = (q_pos > i) ? (q_pos - i) : (i - q_pos);
        float window_mask = (dist > half_w) ? -1e9f : 0.0f;
        float val = row_data[i] * scale + mask[b * seq + i] + window_mask;
        row_data[i] = val;
        thread_max = fmaxf(thread_max, val);
    }
    sdata[threadIdx.x] = thread_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    float row_max = sdata[0];

    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < seq; i += blockDim.x) {
        float val = __expf(row_data[i] - row_max);
        row_data[i] = val;
        thread_sum += val;
    }
    sdata[threadIdx.x] = thread_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float total = sdata[0];

    float inv_sum = 1.0f / fmaxf(total, 1e-12f);
    for (int i = threadIdx.x; i < seq; i += blockDim.x) {
        row_data[i] *= inv_sum;
    }
}

extern "C" __global__ void embedding_lookup_kernel(
    float* output, const float* table, const int* indices,
    int batch_seq, int hidden_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_seq * hidden_dim) {
        int token = idx / hidden_dim;
        int dim = idx % hidden_dim;
        output[idx] = table[indices[token] * hidden_dim + dim];
    }
}

extern "C" __global__ void add_embeddings_kernel(
    float* output, const float* table, const int* indices,
    int batch_seq, int hidden_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_seq * hidden_dim) {
        int token = idx / hidden_dim;
        int dim = idx % hidden_dim;
        output[idx] += table[indices[token] * hidden_dim + dim];
    }
}

extern "C" __global__ void build_attn_mask_kernel(
    float* output, const int* mask, int total
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        output[i] = (mask[i] == 1) ? 0.0f : -1e9f;
    }
}

extern "C" __global__ void qkv_split_kernel(
    float* q, float* k, float* v,
    const float* qkv,
    int batch, int seq, int hidden, int num_heads, int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * num_heads * seq * head_dim;
    if (idx >= total) return;

    int per_batch = num_heads * seq * head_dim;
    int b = idx / per_batch;
    int rem0 = idx % per_batch;
    int h = rem0 / (seq * head_dim);
    int rem1 = rem0 % (seq * head_dim);
    int t = rem1 / head_dim;
    int d = rem1 % head_dim;

    int qkv_idx = b * seq * (3 * hidden) + t * (3 * hidden) + h * head_dim + d;
    q[idx] = qkv[qkv_idx];
    k[idx] = qkv[qkv_idx + hidden];
    v[idx] = qkv[qkv_idx + 2 * hidden];
}

extern "C" __global__ void attn_reshape_kernel(
    float* output, const float* heads,
    int batch, int seq, int num_heads, int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int hidden = num_heads * head_dim;
    int total = batch * seq * hidden;
    if (idx >= total) return;

    int b = idx / (seq * hidden);
    int rem = idx % (seq * hidden);
    int t = rem / hidden;
    int flat_hd = rem % hidden;
    int h = flat_hd / head_dim;
    int d = flat_hd % head_dim;

    int heads_idx = (b * num_heads + h) * (seq * head_dim) + t * head_dim + d;
    output[idx] = heads[heads_idx];
}

extern "C" __global__ void rope_cached_kernel(
    float* q_or_k,
    const float* cos_table,
    const float* sin_table,
    int num_rows, int seq, int head_dim, int num_heads
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half = head_dim / 2;
    int total = num_rows * half;
    if (idx >= total) return;

    int row = idx / half;
    int d = idx % half;
    int pos = row % seq;

    int first_idx = row * head_dim + d;
    int second_idx = first_idx + half;

    float cos_val = cos_table[pos * half + d];
    float sin_val = sin_table[pos * half + d];

    float first = q_or_k[first_idx];
    float second = q_or_k[second_idx];
    q_or_k[first_idx] = first * cos_val - second * sin_val;
    q_or_k[second_idx] = first * sin_val + second * cos_val;
}

extern "C" __global__ void cls_pool_kernel(
    float* output, const float* hidden,
    int batch, int seq, int hidden_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * hidden_dim;
    if (idx < total) {
        int b = idx / hidden_dim;
        int d = idx % hidden_dim;
        output[idx] = hidden[b * seq * hidden_dim + d];
    }
}

extern "C" __global__ void l2_normalize_kernel(float* x, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float sdata[];
    float local_sq = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = x[row * cols + i];
        local_sq += v * v;
    }
    sdata[threadIdx.x] = local_sq;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        __syncthreads();
    }
    float inv_norm = rsqrtf(fmaxf(sdata[0], 1e-12f));
    __syncthreads();
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        x[row * cols + i] *= inv_norm;
    }
}

extern "C" __global__ void f32_to_f16_kernel(
    unsigned short* output, const float* input, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        asm("cvt.rn.f16.f32 %0, %1;" : "=h"(output[i]) : "f"(input[i]));
    }
}

extern "C" __global__ void f16_to_f32_kernel(
    float* output, const unsigned short* input, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        asm("cvt.f32.f16 %0, %1;" : "=f"(output[i]) : "h"(input[i]));
    }
}

extern "C" __global__ void add_bias_kernel(float* x, const float* bias, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        x[idx] += bias[idx % cols];
    }
}

extern "C" __global__ void residual_add_kernel(
    float* output, const float* hidden, const float* residual, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = hidden[i] + residual[i];
}

extern "C" __global__ void fused_bias_gelu_kernel(
    float* x, const float* bias, int rows, int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    int col = idx % cols;
    float v = x[idx] + bias[col];
    x[idx] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
}

extern "C" __global__ void fused_bias_residual_kernel(
    float* output, const float* input, const float* bias,
    const float* residual, int rows, int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    output[idx] = input[idx] + bias[idx % cols] + residual[idx];
}

// GeGLU: output[i] = gelu(value[i]) * gate[i]
extern "C" __global__ void geglu_kernel(
    float* output, const float* value, const float* gate, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = value[i];
        float g = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
        output[i] = g * gate[i];
    }
}

// SwiGLU: output[i] = value[i] * silu(gate[i])
extern "C" __global__ void swiglu_kernel(
    float* output, const float* value, const float* gate, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        float silu = g / (1.0f + __expf(-g));
        output[i] = value[i] * silu;
    }
}

// Split [rows, 2*cols] into two [rows, cols] halves.
extern "C" __global__ void split_gate_value_kernel(
    float* first, float* second, const float* input,
    int rows, int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx >= total) return;
    int row = idx / cols;
    int col = idx % cols;
    int src = row * 2 * cols;
    first[idx] = input[src + col];
    second[idx] = input[src + cols + col];
}

// Mean pooling: mask-weighted mean over seq dim.
// hidden: [batch, seq, hidden_dim], mask: [batch, seq] (1.0/0.0)
// output: [batch, hidden_dim]
extern "C" __global__ void mean_pool_kernel(
    float* output, const float* hidden, const float* mask,
    int batch, int seq, int hidden_dim
) {
    // One block per (batch, dim) pair
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * hidden_dim;
    if (idx >= total) return;

    int b = idx / hidden_dim;
    int d = idx % hidden_dim;

    float sum = 0.0f;
    float mask_sum = 0.0f;
    for (int s = 0; s < seq; s++) {
        float m = mask[b * seq + s];
        sum += hidden[(b * seq + s) * hidden_dim + d] * m;
        mask_sum += m;
    }
    output[idx] = (mask_sum > 0.0f) ? (sum / mask_sum) : 0.0f;
}

// Scatter flat [total_tokens, dim] into padded [batch, max_seq, dim].
// cu_seqlens: [batch+1] cumulative sequence lengths.
extern "C" __global__ void pad_to_batch_kernel(
    float* padded, const float* flat, const int* cu_seqlens,
    int max_seq, int dim, int batch
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * max_seq * dim;
    if (idx >= total) return;

    int b = idx / (max_seq * dim);
    int rem = idx % (max_seq * dim);
    int t = rem / dim;
    int d = rem % dim;

    int seq_start = cu_seqlens[b];
    int seq_len = cu_seqlens[b + 1] - seq_start;

    if (t < seq_len) {
        padded[idx] = flat[(seq_start + t) * dim + d];
    } else {
        padded[idx] = 0.0f;
    }
}

// Gather padded [batch, max_seq, dim] back to flat [total_tokens, dim].
extern "C" __global__ void unpad_from_batch_kernel(
    float* flat, const float* padded, const int* cu_seqlens,
    int max_seq, int dim, int batch
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Total output elements = cu_seqlens[batch] * dim
    // We launch for batch * max_seq * dim and skip padding positions
    int total = batch * max_seq * dim;
    if (idx >= total) return;

    int b = idx / (max_seq * dim);
    int rem = idx % (max_seq * dim);
    int t = rem / dim;
    int d = rem % dim;

    int seq_start = cu_seqlens[b];
    int seq_len = cu_seqlens[b + 1] - seq_start;

    if (t < seq_len) {
        flat[(seq_start + t) * dim + d] = padded[idx];
    }
}

// Banded Q@K^T: sliding window attention scores.
// Q, K: [batch_heads, seq, head_dim]
// scores: [batch_heads, seq, window]
extern "C" __global__ void banded_qk_kernel(
    float* scores, const float* q, const float* k,
    int batch_heads, int seq, int head_dim, int window,
    int stride_qk, int stride_scores
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_heads * seq * window;
    if (idx >= total) return;

    int h = idx / (seq * window);
    int rem = idx % (seq * window);
    int i = rem / window;
    int w = rem % window;

    int half_w = window / 2;
    int k_pos = i - half_w + w;

    if (k_pos < 0 || k_pos >= seq) {
        scores[h * stride_scores + i * window + w] = -1e9f;
        return;
    }

    float dot = 0.0f;
    for (int d = 0; d < head_dim; d++) {
        dot += q[h * stride_qk + i * head_dim + d]
             * k[h * stride_qk + k_pos * head_dim + d];
    }
    scores[h * stride_scores + i * window + w] = dot;
}

// Banded scores@V: weighted sum using banded attention scores.
extern "C" __global__ void banded_sv_kernel(
    float* output, const float* scores, const float* v,
    int batch_heads, int seq, int head_dim, int window,
    int stride_scores, int stride_v, int stride_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_heads * seq * head_dim;
    if (idx >= total) return;

    int h = idx / (seq * head_dim);
    int rem = idx % (seq * head_dim);
    int i = rem / head_dim;
    int d = rem % head_dim;
    int half_w = window / 2;

    float sum = 0.0f;
    for (int w = 0; w < window; w++) {
        int v_pos = i - half_w + w;
        if (v_pos >= 0 && v_pos < seq) {
            sum += scores[h * stride_scores + i * window + w]
                 * v[h * stride_v + v_pos * head_dim + d];
        }
    }
    output[h * stride_out + i * head_dim + d] = sum;
}

// Banded softmax: scale + softmax over window dimension.
// scores: [total_rows, window] — one block per row.
extern "C" __global__ void banded_softmax_kernel(
    float* scores, int total_rows, int window, float scale
) {
    int row = blockIdx.x;
    if (row >= total_rows) return;

    extern __shared__ float sdata[];
    float* row_data = scores + row * window;

    float thread_max = -1e30f;
    for (int i = threadIdx.x; i < window; i += blockDim.x) {
        float val = row_data[i] * scale;
        row_data[i] = val;
        thread_max = fmaxf(thread_max, val);
    }
    sdata[threadIdx.x] = thread_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    float row_max = sdata[0];

    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < window; i += blockDim.x) {
        float val = __expf(row_data[i] - row_max);
        row_data[i] = val;
        thread_sum += val;
    }
    sdata[threadIdx.x] = thread_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float total = sdata[0];
    float inv_sum = 1.0f / fmaxf(total, 1e-12f);
    for (int i = threadIdx.x; i < window; i += blockDim.x) {
        row_data[i] *= inv_sum;
    }
}

// =========================================================================
// FP16 kernels — read/write unsigned short (half), FP32 reductions
// =========================================================================

// Helper: inline half→float and float→half via PTX
#define H2F(h, f) asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h))
#define F2H(f, h) asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f))

extern "C" __global__ void layer_norm_f16_kernel(
    unsigned short* output, const unsigned short* input,
    const float* weight, const float* bias,
    int rows, int cols, float eps
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    extern __shared__ float sdata[];
    float* s_sum = sdata;
    float* s_sq = sdata + blockDim.x;

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v; H2F(input[row * cols + i], v);
        local_sum += v;
    }
    s_sum[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
        __syncthreads();
    }
    float mean = s_sum[0] / (float)cols;
    __syncthreads();

    float local_sq = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v; H2F(input[row * cols + i], v);
        float diff = v - mean;
        local_sq += diff * diff;
    }
    s_sq[threadIdx.x] = local_sq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) s_sq[threadIdx.x] += s_sq[threadIdx.x + s];
        __syncthreads();
    }
    float inv_std = rsqrtf(s_sq[0] / (float)cols + eps);
    __syncthreads();

    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v; H2F(input[row * cols + i], v);
        float out = (v - mean) * inv_std * weight[i] + bias[i];
        unsigned short h; F2H(out, h);
        output[row * cols + i] = h;
    }
}

extern "C" __global__ void fused_residual_layernorm_f16_kernel(
    unsigned short* output, const unsigned short* hidden,
    const unsigned short* residual,
    const float* weight, const float* bias,
    int rows, int cols, float eps
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    extern __shared__ float sdata[];

    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float h; H2F(hidden[row * cols + i], h);
        float r; H2F(residual[row * cols + i], r);
        thread_sum += (h + r);
    }
    sdata[threadIdx.x] = thread_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float mean = sdata[0] / (float)cols;

    float thread_var = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float h; H2F(hidden[row * cols + i], h);
        float r; H2F(residual[row * cols + i], r);
        float diff = (h + r) - mean;
        thread_var += diff * diff;
    }
    sdata[threadIdx.x] = thread_var;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float inv_std = rsqrtf(sdata[0] / (float)cols + eps);

    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float h; H2F(hidden[row * cols + i], h);
        float r; H2F(residual[row * cols + i], r);
        float val = ((h + r) - mean) * inv_std;
        float out = val * weight[i] + bias[i];
        unsigned short oh; F2H(out, oh);
        output[row * cols + i] = oh;
    }
}

// Warp-shuffle reduction helpers for softmax.
// Replaces shared-memory tree reduction: 5 __shfl_down_sync steps
// (log2(32)) per warp, then one cross-warp shared-memory pass.
__device__ __forceinline__ float warp_reduce_max(float val) {
    val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, 16));
    val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, 8));
    val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, 4));
    val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, 2));
    val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, 1));
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_down_sync(0xFFFFFFFF, val, 16);
    val += __shfl_down_sync(0xFFFFFFFF, val, 8);
    val += __shfl_down_sync(0xFFFFFFFF, val, 4);
    val += __shfl_down_sync(0xFFFFFFFF, val, 2);
    val += __shfl_down_sync(0xFFFFFFFF, val, 1);
    return val;
}

// Block-level reduction via warp shuffles + shared memory for cross-warp.
// sdata must have room for (blockDim.x / 32) floats.
__device__ __forceinline__ float block_reduce_max(float val, float* sdata) {
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    val = warp_reduce_max(val);
    if (lane == 0) sdata[wid] = val;
    __syncthreads();
    // First warp reduces across warps.
    int num_warps = blockDim.x >> 5;
    val = (threadIdx.x < num_warps) ? sdata[threadIdx.x] : -1e30f;
    if (wid == 0) val = warp_reduce_max(val);
    val = __shfl_sync(0xFFFFFFFF, val, 0);
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val, float* sdata) {
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    val = warp_reduce_sum(val);
    if (lane == 0) sdata[wid] = val;
    __syncthreads();
    int num_warps = blockDim.x >> 5;
    val = (threadIdx.x < num_warps) ? sdata[threadIdx.x] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    val = __shfl_sync(0xFFFFFFFF, val, 0);
    return val;
}

extern "C" __global__ void fused_scale_mask_softmax_f16_kernel(
    unsigned short* scores, const float* mask,
    int batch, int num_heads, int seq, float scale
) {
    int row = blockIdx.x;
    int total_rows = batch * num_heads * seq;
    if (row >= total_rows) return;

    extern __shared__ float sdata[];
    unsigned short* row_data = scores + row * seq;
    int b = row / (num_heads * seq);

    // Pass 1: scale + mask + max (warp-shuffle reduction)
    float thread_max = -1e30f;
    for (int i = threadIdx.x; i < seq; i += blockDim.x) {
        float v; H2F(row_data[i], v);
        float val = v * scale + mask[b * seq + i];
        thread_max = fmaxf(thread_max, val);
    }
    float row_max = block_reduce_max(thread_max, sdata);

    // Pass 2: exp + sum + normalize (fused, two reads → one)
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < seq; i += blockDim.x) {
        float v; H2F(row_data[i], v);
        float val = __expf(v * scale + mask[b * seq + i] - row_max);
        thread_sum += val;
    }
    float inv_sum = 1.0f / fmaxf(block_reduce_sum(thread_sum, sdata), 1e-12f);

    for (int i = threadIdx.x; i < seq; i += blockDim.x) {
        float v; H2F(row_data[i], v);
        float val = __expf(v * scale + mask[b * seq + i] - row_max) * inv_sum;
        unsigned short h; F2H(val, h);
        row_data[i] = h;
    }
}

extern "C" __global__ void fused_scale_mask_softmax_windowed_f16_kernel(
    unsigned short* scores, const float* mask,
    int batch, int num_heads, int seq, float scale, int window_size
) {
    int row = blockIdx.x;
    int total_rows = batch * num_heads * seq;
    if (row >= total_rows) return;

    extern __shared__ float sdata[];
    unsigned short* row_data = scores + row * seq;
    int b = row / (num_heads * seq);
    int q_pos = row % seq;
    int half_w = window_size / 2;

    // Only iterate over the window range — skip positions outside.
    int lo = (q_pos - half_w > 0) ? (q_pos - half_w) : 0;
    int hi = (q_pos + half_w + 1 < seq) ? (q_pos + half_w + 1) : seq;
    int win_len = hi - lo;

    // Pass 1: scale + padding mask + max (within window only)
    float thread_max = -1e30f;
    for (int w = threadIdx.x; w < win_len; w += blockDim.x) {
        int i = lo + w;
        float v; H2F(row_data[i], v);
        float val = v * scale + mask[b * seq + i];
        thread_max = fmaxf(thread_max, val);
    }
    float row_max = block_reduce_max(thread_max, sdata);

    // Pass 2: exp + sum within window
    float thread_sum = 0.0f;
    for (int w = threadIdx.x; w < win_len; w += blockDim.x) {
        int i = lo + w;
        float v; H2F(row_data[i], v);
        float val = __expf(v * scale + mask[b * seq + i] - row_max);
        thread_sum += val;
    }
    float inv_sum = 1.0f / fmaxf(block_reduce_sum(thread_sum, sdata), 1e-12f);

    // Pass 3: normalize within window, zero outside
    // First, zero positions before window (if any threads reach there)
    for (int i = threadIdx.x; i < lo; i += blockDim.x) {
        unsigned short zero = 0;
        row_data[i] = zero;
    }
    // Normalize within window
    for (int w = threadIdx.x; w < win_len; w += blockDim.x) {
        int i = lo + w;
        float v; H2F(row_data[i], v);
        float val = __expf(v * scale + mask[b * seq + i] - row_max) * inv_sum;
        unsigned short h; F2H(val, h);
        row_data[i] = h;
    }
    // Zero positions after window
    for (int i = hi + threadIdx.x; i < seq; i += blockDim.x) {
        unsigned short zero = 0;
        row_data[i] = zero;
    }
}

extern "C" __global__ void qkv_split_f16_kernel(
    unsigned short* q, unsigned short* k, unsigned short* v,
    const unsigned short* qkv,
    int batch, int seq, int hidden, int num_heads, int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * num_heads * seq * head_dim;
    if (idx >= total) return;

    int per_batch = num_heads * seq * head_dim;
    int b = idx / per_batch;
    int rem0 = idx % per_batch;
    int h = rem0 / (seq * head_dim);
    int rem1 = rem0 % (seq * head_dim);
    int t = rem1 / head_dim;
    int d = rem1 % head_dim;

    int qkv_idx = b * seq * (3 * hidden) + t * (3 * hidden) + h * head_dim + d;
    q[idx] = qkv[qkv_idx];
    k[idx] = qkv[qkv_idx + hidden];
    v[idx] = qkv[qkv_idx + 2 * hidden];
}

extern "C" __global__ void attn_reshape_f16_kernel(
    unsigned short* output, const unsigned short* heads,
    int batch, int seq, int num_heads, int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int hidden = num_heads * head_dim;
    int total = batch * seq * hidden;
    if (idx >= total) return;

    int b = idx / (seq * hidden);
    int rem = idx % (seq * hidden);
    int t = rem / hidden;
    int flat_hd = rem % hidden;
    int h = flat_hd / head_dim;
    int d = flat_hd % head_dim;

    int heads_idx = (b * num_heads + h) * (seq * head_dim) + t * head_dim + d;
    output[idx] = heads[heads_idx];
}

// Fused pad + QKV split: read flat [total_tokens, 3*hidden] → write Q,K,V [batch*heads, max_seq, head_dim].
// Eliminates the padded intermediate buffer and its 2 memory round-trips.
extern "C" __global__ void fused_pad_qkv_split_f16_kernel(
    unsigned short* q, unsigned short* k, unsigned short* v,
    const unsigned short* qkv_flat,
    const int* cu_seqlens,
    int batch, int max_seq, int hidden, int num_heads, int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * num_heads * max_seq * head_dim;
    if (idx >= total) return;

    // Decompose output index: [batch*num_heads, max_seq, head_dim]
    int per_head = max_seq * head_dim;
    int bh = idx / per_head;
    int rem = idx % per_head;
    int t = rem / head_dim;
    int d = rem % head_dim;
    int b = bh / num_heads;
    int h = bh % num_heads;

    int seq_start = cu_seqlens[b];
    int seq_len = cu_seqlens[b + 1] - seq_start;

    if (t < seq_len) {
        // Read from flat: qkv_flat[(seq_start + t) * 3*hidden + h*head_dim + d] for Q
        int flat_base = (seq_start + t) * 3 * hidden;
        q[idx] = qkv_flat[flat_base + h * head_dim + d];
        k[idx] = qkv_flat[flat_base + hidden + h * head_dim + d];
        v[idx] = qkv_flat[flat_base + 2 * hidden + h * head_dim + d];
    } else {
        unsigned short zero; F2H(0.0f, zero);
        q[idx] = zero;
        k[idx] = zero;
        v[idx] = zero;
    }
}

// Fused attn_reshape + unpad: read [batch*heads, max_seq, head_dim] → write [total_tokens, hidden].
// Eliminates the padded context intermediate.
extern "C" __global__ void fused_reshape_unpad_f16_kernel(
    unsigned short* flat, const unsigned short* heads,
    const int* cu_seqlens,
    int batch, int max_seq, int num_heads, int head_dim
) {
    int hidden = num_heads * head_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Iterate over output: [total_tokens, hidden]
    // We need total_tokens but don't have it — use batch*max_seq and skip padding.
    int total = batch * max_seq * hidden;
    if (idx >= total) return;

    int b = idx / (max_seq * hidden);
    int rem = idx % (max_seq * hidden);
    int t = rem / hidden;
    int flat_hd = rem % hidden;

    int seq_start = cu_seqlens[b];
    int seq_len = cu_seqlens[b + 1] - seq_start;

    if (t < seq_len) {
        int h = flat_hd / head_dim;
        int d = flat_hd % head_dim;
        int heads_idx = (b * num_heads + h) * (max_seq * head_dim) + t * head_dim + d;
        flat[(seq_start + t) * hidden + flat_hd] = heads[heads_idx];
    }
}

extern "C" __global__ void rope_encode_f16_kernel(
    unsigned short* q_or_k,
    const float* cos_table, const float* sin_table,
    int num_rows, int seq, int head_dim, int num_heads
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half = head_dim / 2;
    int total = num_rows * half;
    if (idx >= total) return;

    int row = idx / half;
    int d = idx % half;
    int pos = row % seq;

    int first_idx = row * head_dim + d;
    int second_idx = first_idx + half;

    float cos_val = cos_table[pos * half + d];
    float sin_val = sin_table[pos * half + d];

    float first; H2F(q_or_k[first_idx], first);
    float second; H2F(q_or_k[second_idx], second);
    float out_first = first * cos_val - second * sin_val;
    float out_second = first * sin_val + second * cos_val;
    unsigned short h1, h2;
    F2H(out_first, h1);
    F2H(out_second, h2);
    q_or_k[first_idx] = h1;
    q_or_k[second_idx] = h2;
}

extern "C" __global__ void geglu_f16_kernel(
    unsigned short* output, const unsigned short* value,
    const unsigned short* gate, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v; H2F(value[i], v);
        float g; H2F(gate[i], g);
        float gelu_v = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
        float result = gelu_v * g;
        unsigned short h; F2H(result, h);
        output[i] = h;
    }
}

extern "C" __global__ void split_gate_value_f16_kernel(
    unsigned short* first, unsigned short* second,
    const unsigned short* input, int rows, int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx >= total) return;
    int row = idx / cols;
    int col = idx % cols;
    int src = row * 2 * cols;
    first[idx] = input[src + col];
    second[idx] = input[src + cols + col];
}

// Fused split + GeGLU: read [rows, 2*cols], write [rows, cols].
// Eliminates two intermediate buffers and halves HBM bandwidth.
extern "C" __global__ void fused_split_geglu_f16_kernel(
    unsigned short* output, const unsigned short* input,
    int rows, int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx >= total) return;
    int row = idx / cols;
    int col = idx % cols;
    int src = row * 2 * cols;
    float value; H2F(input[src + col], value);
    float gate; H2F(input[src + cols + col], gate);
    float gelu_v = 0.5f * value * (1.0f + tanhf(0.7978845608f * (value + 0.044715f * value * value * value)));
    float result = gelu_v * gate;
    unsigned short h; F2H(result, h);
    output[idx] = h;
}

extern "C" __global__ void residual_add_f16_kernel(
    unsigned short* output, const unsigned short* hidden,
    const unsigned short* residual, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float h; H2F(hidden[i], h);
        float r; H2F(residual[i], r);
        float out = h + r;
        unsigned short oh; F2H(out, oh);
        output[i] = oh;
    }
}

// =========================================================================
// INT8 quantization kernels
// =========================================================================

// Per-column max abs -> scale for weight quantization at load time.
extern "C" __global__ void quantize_col_scales_kernel(
    float* scales, const unsigned short* weights, int rows, int cols
) {
    int col = blockIdx.x;
    if (col >= cols) return;
    extern __shared__ float sdata[];
    float local_max = 0.0f;
    for (int r = threadIdx.x; r < rows; r += blockDim.x) {
        float v; H2F(weights[r * cols + col], v);
        local_max = fmaxf(local_max, fabsf(v));
    }
    float col_max = block_reduce_max(local_max, sdata);
    if (threadIdx.x == 0) scales[col] = fmaxf(col_max, 1e-12f) / 127.0f;
}

// Flat element-wise quantization: FP16 -> INT8 using pre-computed per-column scales.
extern "C" __global__ void quantize_weights_i8_kernel(
    signed char* output, const unsigned short* weights, const float* scales,
    int rows, int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    int col = idx % cols;
    float v; H2F(weights[idx], v);
    int q = __float2int_rn(v / scales[col]);
    q = max(-127, min(127, q));
    output[idx] = (signed char)q;
}

// Per-row fused quantize: find max|a| per row, compute scale, quantize in one pass.
extern "C" __global__ void quantize_activation_rowwise_kernel(
    signed char* output, float* scales_out,
    const unsigned short* activations, int rows, int cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    extern __shared__ float sdata[];
    const unsigned short* row_data = activations + row * cols;
    signed char* row_out = output + row * cols;
    float local_max = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v; H2F(row_data[i], v);
        local_max = fmaxf(local_max, fabsf(v));
    }
    float row_max = block_reduce_max(local_max, sdata);
    float scale = fmaxf(row_max, 1e-12f) / 127.0f;
    if (threadIdx.x == 0) scales_out[row] = scale;
    float inv_scale = 1.0f / scale;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v; H2F(row_data[i], v);
        int q = __float2int_rn(v * inv_scale);
        q = max(-127, min(127, q));
        row_out[i] = (signed char)q;
    }
}

// Dequantize INT32 GEMM output to FP16 with per-row/per-col scales.
// One block per row; vectorised int4 loads for bandwidth.
extern "C" __global__ void dequantize_i32_to_f16_kernel(
    unsigned short* output, const int* input,
    const float* act_row_scales, const float* weight_col_scales,
    int m, int n
) {
    int row = blockIdx.x;
    if (row >= m) return;
    float a_scale = act_row_scales[row];
    const int* row_in = input + row * n;
    unsigned short* row_out = output + row * n;
    // Vectorised: process 4 ints at a time (16 bytes per load).
    int n4 = n / 4;
    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
        int col = i * 4;
        int4 vals = reinterpret_cast<const int4*>(row_in)[i];
        float v0 = (float)vals.x * a_scale * weight_col_scales[col];
        float v1 = (float)vals.y * a_scale * weight_col_scales[col + 1];
        float v2 = (float)vals.z * a_scale * weight_col_scales[col + 2];
        float v3 = (float)vals.w * a_scale * weight_col_scales[col + 3];
        unsigned short h0, h1, h2, h3;
        F2H(v0, h0); F2H(v1, h1); F2H(v2, h2); F2H(v3, h3);
        row_out[col] = h0; row_out[col+1] = h1; row_out[col+2] = h2; row_out[col+3] = h3;
    }
    for (int col = n4 * 4 + threadIdx.x; col < n; col += blockDim.x) {
        float val = (float)row_in[col] * a_scale * weight_col_scales[col];
        unsigned short h; F2H(val, h);
        row_out[col] = h;
    }
}

extern "C" __global__ void pad_to_batch_f16_kernel(
    unsigned short* padded, const unsigned short* flat,
    const int* cu_seqlens, int max_seq, int dim, int batch
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * max_seq * dim;
    if (idx >= total) return;

    int b = idx / (max_seq * dim);
    int rem = idx % (max_seq * dim);
    int t = rem / dim;
    int d = rem % dim;

    int seq_start = cu_seqlens[b];
    int seq_len = cu_seqlens[b + 1] - seq_start;

    if (t < seq_len) {
        padded[idx] = flat[(seq_start + t) * dim + d];
    } else {
        unsigned short zero; F2H(0.0f, zero);
        padded[idx] = zero;
    }
}

extern "C" __global__ void unpad_from_batch_f16_kernel(
    unsigned short* flat, const unsigned short* padded,
    const int* cu_seqlens, int max_seq, int dim, int batch
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * max_seq * dim;
    if (idx >= total) return;

    int b = idx / (max_seq * dim);
    int rem = idx % (max_seq * dim);
    int t = rem / dim;
    int d = rem % dim;

    int seq_start = cu_seqlens[b];
    int seq_len = cu_seqlens[b + 1] - seq_start;

    if (t < seq_len) {
        flat[(seq_start + t) * dim + d] = padded[idx];
    }
}
"#;

// ---------------------------------------------------------------------------
// Compiled kernel handles
// ---------------------------------------------------------------------------

/// Pre-compiled CUDA kernel function handles for `ModernBERT`.
struct KernelPipelines {
    // FP32 kernels
    gelu: CudaFunction,
    layer_norm: CudaFunction,
    fused_residual_layernorm: CudaFunction,
    fused_scale_mask_softmax: CudaFunction,
    fused_scale_mask_softmax_windowed: CudaFunction,
    embedding_lookup: CudaFunction,
    add_embeddings: CudaFunction,
    build_attn_mask: CudaFunction,
    qkv_split: CudaFunction,
    attn_reshape: CudaFunction,
    rope_cached: CudaFunction,
    cls_pool: CudaFunction,
    l2_normalize: CudaFunction,
    f32_to_f16: CudaFunction,
    f16_to_f32: CudaFunction,
    add_bias: CudaFunction,
    residual_add: CudaFunction,
    fused_bias_gelu: CudaFunction,
    fused_bias_residual: CudaFunction,
    geglu: CudaFunction,
    swiglu: CudaFunction,
    split_gate_value: CudaFunction,
    mean_pool: CudaFunction,
    pad_to_batch: CudaFunction,
    unpad_from_batch: CudaFunction,
    banded_qk: CudaFunction,
    banded_sv: CudaFunction,
    banded_softmax: CudaFunction,
    // FP16 kernels
    layer_norm_f16: CudaFunction,
    fused_residual_layernorm_f16: CudaFunction,
    fused_scale_mask_softmax_f16: CudaFunction,
    fused_scale_mask_softmax_windowed_f16: CudaFunction,
    qkv_split_f16: CudaFunction,
    attn_reshape_f16: CudaFunction,
    rope_encode_f16: CudaFunction,
    geglu_f16: CudaFunction,
    split_gate_value_f16: CudaFunction,
    fused_split_geglu_f16: CudaFunction,
    residual_add_f16: CudaFunction,
    pad_to_batch_f16: CudaFunction,
    fused_pad_qkv_split_f16: CudaFunction,
    fused_reshape_unpad_f16: CudaFunction,
    unpad_from_batch_f16: CudaFunction,
    // INT8 quantization kernels
    quantize_col_scales: CudaFunction,
    quantize_weights_i8: CudaFunction,
    quantize_activation_rowwise: CudaFunction,
    dequantize_i32_to_f16: CudaFunction,
}

impl KernelPipelines {
    /// Compile all CUDA kernels and load function handles.
    fn compile(ctx: &Arc<CudaContext>) -> crate::Result<(Arc<CudaModule>, Self)> {
        // Detect GPU compute capability at runtime and emit SASS directly.
        // Targeting `sm_XX` (real arch) with CUBIN output via NVRTC bypasses
        // the PTX-version gate that caused "unsupported PTX version" failures
        // with newer NVRTC toolkits (e.g. CUDA 13.2) on older drivers (590 /
        // CUDA 13.1). See `backend::nvrtc_cubin` for the lifecycle.
        use cudarc::driver::sys::CUdevice_attribute;
        let major = ctx
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .map_err(cuda_err)?;
        let minor = ctx
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .map_err(cuda_err)?;
        let arch = format!("sm_{major}{minor}");
        let cubin = compile_cubin(MODERN_KERNELS, &arch).map_err(cuda_err)?;
        let module = ctx.load_module(cubin).map_err(cuda_err)?;
        let module = Arc::new(module);

        let load = |name: &str| -> crate::Result<CudaFunction> {
            module.load_function(name).map_err(cuda_err)
        };

        Ok((
            Arc::clone(&module),
            Self {
                gelu: load("gelu_kernel")?,
                layer_norm: load("layer_norm_kernel")?,
                fused_residual_layernorm: load("fused_residual_layernorm_kernel")?,
                fused_scale_mask_softmax: load("fused_scale_mask_softmax_kernel")?,
                fused_scale_mask_softmax_windowed: load(
                    "fused_scale_mask_softmax_windowed_kernel",
                )?,
                embedding_lookup: load("embedding_lookup_kernel")?,
                add_embeddings: load("add_embeddings_kernel")?,
                build_attn_mask: load("build_attn_mask_kernel")?,
                qkv_split: load("qkv_split_kernel")?,
                attn_reshape: load("attn_reshape_kernel")?,
                rope_cached: load("rope_cached_kernel")?,
                cls_pool: load("cls_pool_kernel")?,
                l2_normalize: load("l2_normalize_kernel")?,
                f32_to_f16: load("f32_to_f16_kernel")?,
                f16_to_f32: load("f16_to_f32_kernel")?,
                add_bias: load("add_bias_kernel")?,
                residual_add: load("residual_add_kernel")?,
                fused_bias_gelu: load("fused_bias_gelu_kernel")?,
                fused_bias_residual: load("fused_bias_residual_kernel")?,
                geglu: load("geglu_kernel")?,
                swiglu: load("swiglu_kernel")?,
                split_gate_value: load("split_gate_value_kernel")?,
                mean_pool: load("mean_pool_kernel")?,
                pad_to_batch: load("pad_to_batch_kernel")?,
                unpad_from_batch: load("unpad_from_batch_kernel")?,
                banded_qk: load("banded_qk_kernel")?,
                banded_sv: load("banded_sv_kernel")?,
                banded_softmax: load("banded_softmax_kernel")?,
                // FP16 kernels
                layer_norm_f16: load("layer_norm_f16_kernel")?,
                fused_residual_layernorm_f16: load("fused_residual_layernorm_f16_kernel")?,
                fused_scale_mask_softmax_f16: load("fused_scale_mask_softmax_f16_kernel")?,
                fused_scale_mask_softmax_windowed_f16: load(
                    "fused_scale_mask_softmax_windowed_f16_kernel",
                )?,
                qkv_split_f16: load("qkv_split_f16_kernel")?,
                attn_reshape_f16: load("attn_reshape_f16_kernel")?,
                rope_encode_f16: load("rope_encode_f16_kernel")?,
                geglu_f16: load("geglu_f16_kernel")?,
                split_gate_value_f16: load("split_gate_value_f16_kernel")?,
                fused_split_geglu_f16: load("fused_split_geglu_f16_kernel")?,
                residual_add_f16: load("residual_add_f16_kernel")?,
                pad_to_batch_f16: load("pad_to_batch_f16_kernel")?,
                fused_pad_qkv_split_f16: load("fused_pad_qkv_split_f16_kernel")?,
                fused_reshape_unpad_f16: load("fused_reshape_unpad_f16_kernel")?,
                unpad_from_batch_f16: load("unpad_from_batch_f16_kernel")?,
                // INT8 quantization kernels
                quantize_col_scales: load("quantize_col_scales_kernel")?,
                quantize_weights_i8: load("quantize_weights_i8_kernel")?,
                quantize_activation_rowwise: load("quantize_activation_rowwise_kernel")?,
                dequantize_i32_to_f16: load("dequantize_i32_to_f16_kernel")?,
            },
        ))
    }
}

// ---------------------------------------------------------------------------
// CudaDriver
// ---------------------------------------------------------------------------

/// CUDA compute driver for `ModernBERT` inference.
///
/// Uses cudarc for device management, cuBLAS for GEMM (FP32 TF32 and FP16
/// tensor core paths), and NVRTC-compiled kernels for all other operations.
/// Supports full FP16 pipeline with FP32 accumulation in reductions.
pub struct CudaDriver {
    /// CUDA device context (kept alive; stream references it).
    _ctx: Arc<CudaContext>,
    /// CUDA stream for all operations.
    stream: Arc<CudaStream>,
    /// cuBLAS handle for matrix multiplications.
    blas: CudaBlas,
    /// cuBLASLt handle for INT8 tensor core GEMMs with fused scale epilogue.
    blas_lt: CudaBlasLT,
    /// Workspace buffer for cuBLASLt matmul heuristics (4 MiB, or 32 MiB on Hopper).
    lt_workspace: CudaSlice<u8>,
    /// Size of the cuBLASLt workspace in bytes.
    lt_workspace_size: usize,
    /// Pre-compiled kernel function handles.
    kernels: KernelPipelines,
    /// Compiled NVRTC module (kept alive for kernel function references).
    _module: Arc<CudaModule>,
}

// SAFETY: CudaDriver holds Arc handles to CUDA resources. All mutable state
// (pool, cursor) is behind RefCell/Cell and only accessed from a single thread
// per forward pass. The CUDA context is thread-safe for single-device use.
#[expect(
    unsafe_code,
    reason = "Arc-based CUDA handles; single-thread access pattern"
)]
unsafe impl Send for CudaDriver {}
#[expect(
    unsafe_code,
    reason = "RefCell/Cell interior mutability is single-thread"
)]
unsafe impl Sync for CudaDriver {}

impl CudaDriver {
    /// Create a new CUDA driver on device 0.
    ///
    /// Initialises the CUDA context, stream, cuBLAS handle, and compiles all
    /// NVRTC kernels. This is the main entry point for CUDA backend setup.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA device initialisation or kernel compilation fails.
    pub fn new() -> crate::Result<Self> {
        let ctx = cudarc::driver::CudaContext::new(0).map_err(cuda_err)?;
        // Disable event tracking: we use a single stream so cross-stream
        // synchronisation events are pure overhead (~14k events at 7k kernel launches).
        // SAFETY: single-stream usage means no cross-stream hazards.
        #[expect(unsafe_code, reason = "single-stream → no cross-stream sync needed")]
        unsafe {
            ctx.disable_event_tracking();
        }
        let stream = ctx.default_stream();
        let blas = CudaBlas::new(stream.clone()).map_err(cuda_err)?;
        let blas_lt = CudaBlasLT::new(stream.clone()).map_err(cuda_err)?;

        // Allocate cuBLASLt workspace (4 MiB default, 32 MiB on Hopper SM90+).
        let major = ctx
            .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .map_err(cuda_err)?;
        let lt_workspace_size = if major >= 9 { 33_554_432 } else { 4_194_304 };
        // SAFETY: workspace is scratch memory; cuBLASLt manages its contents internally.
        #[expect(unsafe_code, reason = "workspace is scratch memory for cuBLASLt")]
        let lt_workspace = unsafe { stream.alloc::<u8>(lt_workspace_size) }.map_err(cuda_err)?;

        let (module, kernels) = KernelPipelines::compile(&ctx)?;

        Ok(Self {
            _ctx: ctx,
            stream,
            blas,
            blas_lt,
            lt_workspace,
            lt_workspace_size,
            kernels,
            _module: module,
        })
    }

    /// Allocate an FP32 tensor. Uses `cuMemAllocAsync` (no memset — caller
    /// must fully overwrite before reading).
    ///
    /// cudarc's `CudaSlice::clone()` does a full D2D memcpy, so a pool-based
    /// approach like Metal's (where `Retained<MTLBuffer>` is refcounted) doesn't
    /// work. Instead we allocate on demand via the async allocator, which amortises
    /// `cuMemAlloc` overhead across the stream.
    #[expect(
        unsafe_code,
        reason = "alloc returns uninitialised memory; caller overwrites"
    )]
    fn alloc_tensor(&self, n: usize) -> crate::Result<CudaSlice<f32>> {
        // SAFETY: caller guarantees the buffer is fully written before any read.
        unsafe { self.stream.alloc::<f32>(n) }.map_err(cuda_err)
    }

    /// Allocate an FP16 tensor (uninitialised — caller must overwrite).
    #[expect(
        unsafe_code,
        reason = "alloc returns uninitialised memory; caller overwrites"
    )]
    fn alloc_f16_tensor(&self, n: usize) -> crate::Result<CudaSlice<u16>> {
        // SAFETY: caller guarantees the buffer is fully written before any read.
        unsafe { self.stream.alloc::<u16>(n) }.map_err(cuda_err)
    }

    /// Build cumulative sequence lengths for pad/unpad kernels.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "sequence lengths are small (<8192); total tokens fit in i32"
    )]
    fn build_cu_seqlens(&self, seq_lengths: &[usize]) -> crate::Result<CudaSlice<i32>> {
        let mut cu: Vec<i32> = Vec::with_capacity(seq_lengths.len() + 1);
        cu.push(0);
        let mut acc: i32 = 0;
        for &len in seq_lengths {
            acc += len as i32;
            cu.push(acc);
        }
        self.stream.clone_htod(&cu).map_err(cuda_err)
    }

    /// Load `ModernBERT` weights from a safetensors file into GPU tensors.
    ///
    /// Memory-maps the file, parses the safetensors header, transfers each tensor
    /// to GPU via H2D copy, and pre-converts GEMM weights to FP16 for the tensor
    /// core path.
    ///
    /// Returns `(arch, mmap)` — the mmap must be kept alive by the caller.
    ///
    /// # Errors
    ///
    /// Returns an error if file I/O, safetensors parsing, or GPU allocation fails.
    #[expect(
        unsafe_code,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::too_many_lines,
        reason = "monolithic weight loader: unsafe for mmap + kernel; small-int casts"
    )]
    pub fn load_modern_bert_weights(
        &self,
        weights_path: &Path,
        config: &ModernBertConfig,
    ) -> crate::Result<(ModernBertArch<CudaTensor>, memmap2::Mmap)> {
        // 1. mmap the safetensors file
        let file = std::fs::File::open(weights_path).map_err(|e| crate::Error::Io {
            path: weights_path.display().to_string(),
            source: e,
        })?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| crate::Error::Io {
            path: weights_path.display().to_string(),
            source: e,
        })?;

        // 2. Parse safetensors header
        let tensors = SafeTensors::deserialize(&mmap)
            .map_err(|e| crate::Error::Cuda(format!("safetensors parse: {e}")))?;

        let hidden = config.hidden_size;
        let num_layers = config.num_hidden_layers;
        let num_heads = config.num_attention_heads;
        let head_dim = hidden / num_heads;
        let intermediate = config.intermediate_size;
        let global_attn_every_n = config.global_attn_every_n_layers;

        // Helper: load a tensor from safetensors as Vec<f32>, then H2D copy.
        let load_gpu = |name: &str| -> crate::Result<CudaSlice<f32>> {
            let tensor = tensors
                .tensor(name)
                .map_err(|_| crate::Error::Cuda(format!("missing weight: {name}")))?;
            let data: Vec<f32> = tensor
                .data()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            self.stream.clone_htod(&data).map_err(cuda_err)
        };

        // Helper: convert an FP32 device buffer to FP16.
        let convert_f16 = |input: &CudaSlice<f32>, n: usize| -> crate::Result<CudaSlice<u16>> {
            let mut output = self.stream.alloc_zeros::<u16>(n).map_err(cuda_err)?;
            let n_i32 = n as i32;
            let mut builder = self.stream.launch_builder(&self.kernels.f32_to_f16);
            builder.arg(&mut output);
            builder.arg(input);
            builder.arg(&n_i32);
            // SAFETY: kernel reads n f32 from input, writes n u16 to output.
            // Both buffers are pre-allocated with sufficient size.
            // SAFETY: kernel args set above; config matches element count.
            unsafe { launch_kernel(builder, launch_1d(n)) }?;
            Ok(output)
        };

        // Helper: quantize FP16 weights to INT8 with per-column symmetric scaling.
        let quantize_i8 = |fp16: &CudaSlice<u16>,
                           rows: usize,
                           cols: usize|
         -> crate::Result<(CudaSlice<i8>, CudaSlice<f32>)> {
            // Phase 1: compute per-column scales (one block per column).
            let mut scales = self.stream.alloc_zeros::<f32>(cols).map_err(cuda_err)?;
            let threads = 256_u32;
            let shared = (threads / 32) * 4;
            let rows_i = rows as i32;
            let cols_i = cols as i32;
            {
                let mut builder = self
                    .stream
                    .launch_builder(&self.kernels.quantize_col_scales);
                builder.arg(&mut scales);
                builder.arg(fp16);
                builder.arg(&rows_i);
                builder.arg(&cols_i);
                // SAFETY: one block per column; reads fp16[rows*cols], writes scales[cols].
                // Shared mem holds (threads/32) floats for warp-level reduction.
                unsafe {
                    launch_kernel(builder, launch_per_row(cols, threads, shared))?;
                }
            }
            // Phase 2: quantize each element to INT8.
            let total = rows * cols;
            let mut i8_buf = unsafe { self.stream.alloc::<i8>(total) }.map_err(cuda_err)?;
            {
                let mut builder = self
                    .stream
                    .launch_builder(&self.kernels.quantize_weights_i8);
                builder.arg(&mut i8_buf);
                builder.arg(fp16);
                builder.arg(&scales);
                builder.arg(&rows_i);
                builder.arg(&cols_i);
                // SAFETY: reads fp16[total] and scales[cols], writes i8_buf[total].
                unsafe {
                    launch_kernel(builder, launch_1d(total))?;
                }
            }
            Ok((i8_buf, scales))
        };

        // 3. Build per-layer weights
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let qkv_f32 = load_gpu(&format!("layers.{i}.attn.Wqkv.weight"))?;
            let wo_f32 = load_gpu(&format!("layers.{i}.attn.Wo.weight"))?;
            let attn_norm_f32 = if i == 0 {
                None
            } else {
                Some(load_gpu(&format!("layers.{i}.attn_norm.weight"))?)
            };
            let wi_f32 = load_gpu(&format!("layers.{i}.mlp.Wi.weight"))?;
            let mlp_wo_f32 = load_gpu(&format!("layers.{i}.mlp.Wo.weight"))?;
            let mlp_norm_f32 = load_gpu(&format!("layers.{i}.mlp_norm.weight"))?;

            let is_global = i % global_attn_every_n == 0;

            // Pre-convert GEMM weights to FP16 for the tensor core path.
            let qkv_fp16 = convert_f16(&qkv_f32, 3 * hidden * hidden)?;
            let wo_fp16 = convert_f16(&wo_f32, hidden * hidden)?;
            let wi_fp16 = convert_f16(&wi_f32, 2 * intermediate * hidden)?;
            let mlp_wo_fp16 = convert_f16(&mlp_wo_f32, hidden * intermediate)?;

            // INT8 quantize each GEMM weight matrix for cuBLASLt tensor core path.
            // Weights are [out_features, in_features] row-major.
            let (qkv_i8, qkv_scales) = quantize_i8(&qkv_fp16, 3 * hidden, hidden)?;
            let (wo_i8, wo_scales) = quantize_i8(&wo_fp16, hidden, hidden)?;
            let (wi_i8, wi_scales) = quantize_i8(&wi_fp16, 2 * intermediate, hidden)?;
            let (mlp_wo_i8, mlp_wo_scales) = quantize_i8(&mlp_wo_fp16, hidden, intermediate)?;

            layers.push(ModernBertLayerWeights {
                qkv_weight: CudaTensor {
                    f32_buf: qkv_f32,
                    fp16: Some(qkv_fp16),
                    int8: Some(qkv_i8),
                    int8_col_scales: Some(qkv_scales),
                },
                output_weight: CudaTensor {
                    f32_buf: wo_f32,
                    fp16: Some(wo_fp16),
                    int8: Some(wo_i8),
                    int8_col_scales: Some(wo_scales),
                },
                attn_norm_weight: attn_norm_f32.map(CudaTensor::new),
                mlp_wi_weight: CudaTensor {
                    f32_buf: wi_f32,
                    fp16: Some(wi_fp16),
                    int8: Some(wi_i8),
                    int8_col_scales: Some(wi_scales),
                },
                mlp_wo_weight: CudaTensor {
                    f32_buf: mlp_wo_f32,
                    fp16: Some(mlp_wo_fp16),
                    int8: Some(mlp_wo_i8),
                    int8_col_scales: Some(mlp_wo_scales),
                },
                mlp_norm_weight: CudaTensor::new(mlp_norm_f32),
                is_global,
            });
        }

        // 4. Embedding + final norm weights
        let tok_emb_f32 = load_gpu("embeddings.tok_embeddings.weight")?;
        let emb_norm_f32 = load_gpu("embeddings.norm.weight")?;
        let final_norm_f32 = load_gpu("final_norm.weight")?;
        let zero_bias_f32 = self.stream.alloc_zeros::<f32>(hidden).map_err(cuda_err)?;

        let weights = ModernBertWeights {
            tok_embeddings: CudaTensor::new(tok_emb_f32),
            emb_norm_weight: CudaTensor::new(emb_norm_f32),
            final_norm_weight: CudaTensor::new(final_norm_f32),
            zero_bias: CudaTensor::new(zero_bias_f32),
            layers,
            num_heads,
            head_dim,
            hidden_dim: hidden,
            intermediate_dim: intermediate,
            layer_norm_eps: config.norm_eps,
            local_window: config.local_attention,
        };

        // 5. Build RoPE caches
        let max_seq = config.max_position_embeddings;
        let global_rope =
            build_rope_cache(&self.stream, head_dim, max_seq, config.global_rope_theta)?;
        let local_rope =
            build_rope_cache(&self.stream, head_dim, max_seq, config.local_rope_theta)?;

        let arch = ModernBertArch {
            weights,
            global_rope,
            local_rope,
        };

        Ok((arch, mmap))
    }

    /// Perform INT8 GEMM via cuBLASLt with fused per-row/per-column scale epilogue.
    ///
    /// Computes: `output_f16 = (A_i8 @ B_i8^T) * a_row_scales * b_col_scales`
    /// in a single cuBLASLt call, eliminating the separate dequantize kernel.
    ///
    /// The activation matrix A is quantized on-the-fly from FP16 to INT8 with
    /// per-row scaling; weight matrix B is pre-quantized at load time with
    /// per-column scaling.
    ///
    /// # cuBLAS column-major convention
    ///
    /// For row-major A\[m,k\] x B\[n,k\]^T = C\[m,n\]:
    /// - cuBLAS sees: C(n,m) = B(n,k) x A(k,m) with `transa=N` on B, `transb=T` on A
    /// - Or equivalently: set transa/transb on the *cuBLAS* operands
    ///
    /// # Errors
    ///
    /// Returns an error if activation quantization or the cuBLASLt matmul fails.
    #[expect(
        unsafe_code,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::too_many_lines,
        reason = "cuBLASLt raw API requires unsafe; int casts match CUDA types"
    )]
    /// INT8 GEMM via `cublasGemmEx` with per-row dequantize.
    ///
    /// Flow: quantize A (FP16→INT8) → `cublasGemmEx(I8×I8→I32)` → per-row
    /// dequantize+convert (INT32→FP16 with row×col scales).
    ///
    /// The dequantize uses a per-row kernel (one block per row) to amortize
    /// the scale lookup and apply vectorised INT32→FP16 conversion.
    #[expect(
        unsafe_code,
        clippy::too_many_arguments,
        clippy::too_many_lines,
        reason = "cuBLAS INT8 GEMM requires unsafe FFI; args mirror cuBLAS API"
    )]
    fn gemm_i8_impl(
        &self,
        a_fp16_tensor: &CudaTensor,
        b_i8: &CudaSlice<i8>,
        b_col_scales: &CudaSlice<f32>,
        output: &mut CudaTensor,
        m: usize,
        n: usize,
        k: usize,
        transpose_b: bool,
    ) -> crate::Result<()> {
        let a_f16 = a_fp16_tensor
            .fp16_ref()
            .ok_or_else(|| crate::Error::Cuda("gemm_i8: A has no FP16 buffer".into()))?;
        let out_f16 = output
            .fp16
            .as_mut()
            .ok_or_else(|| crate::Error::Cuda("gemm_i8: output has no FP16 buffer".into()))?;

        // 1. Per-row quantize activations (FP16 -> INT8 + row scales).
        let mut a_row_scales =
            // SAFETY: kernel will fully overwrite all m elements before they are read.
            unsafe { self.stream.alloc::<f32>(m) }.map_err(cuda_err)?;
        let mut a_i8 =
            // SAFETY: kernel will fully overwrite all m*k elements before they are read.
            unsafe { self.stream.alloc::<i8>(m * k) }.map_err(cuda_err)?;
        {
            let threads = 256_u32;
            let shared = (threads / 32) * 4;
            let rows_i = m as i32;
            let cols_i = k as i32;
            let mut builder = self
                .stream
                .launch_builder(&self.kernels.quantize_activation_rowwise);
            builder.arg(&mut a_i8);
            builder.arg(&mut a_row_scales);
            builder.arg(a_f16);
            builder.arg(&rows_i);
            builder.arg(&cols_i);
            // SAFETY: one block per row; reads a_f16[m*k], writes a_i8[m*k] + a_row_scales[m].
            // Shared mem holds (threads/32) floats for warp-level block_reduce_max.
            unsafe { launch_kernel(builder, launch_per_row(m, threads, shared))? };
        }

        // 2. cublasGemmEx: INT8 × INT8 → INT32 (tensor cores).
        // CUDA 13.1 bug: FP32 output from INT8 GEMM returns NOT_SUPPORTED.
        // Use INT32 output and dequantize separately. Fixed in CUDA 13.2.
        let c_elements = m * n;
        let m_i = m as i32;
        let n_i = n as i32;
        let k_i = k as i32;
        let alpha_i32 = 1_i32;
        let beta_i32 = 0_i32;
        let handle = *self.blas.handle();

        let mut c_i32: CudaSlice<i32> =
            // SAFETY: GEMM will fully overwrite all m*n elements.
            unsafe { self.stream.alloc::<i32>(c_elements) }.map_err(cuda_err)?;

        {
            let (a_ptr, _a_sync) = a_i8.device_ptr(&self.stream);
            let (b_ptr, _b_sync) = b_i8.device_ptr(&self.stream);
            let (c_ptr, _c_sync) = c_i32.device_ptr_mut(&self.stream);

            if transpose_b {
                // cuBLAS col-major: C(n,m) = B_T(n,k) @ A(k,m)
                // SAFETY: INT8 buffers correctly sized; FP32 output with int→float.
                unsafe {
                    sys::cublasGemmEx(
                        handle,
                        sys::cublasOperation_t::CUBLAS_OP_T,
                        sys::cublasOperation_t::CUBLAS_OP_N,
                        n_i,
                        m_i,
                        k_i,
                        std::ptr::from_ref(&alpha_i32).cast(),
                        b_ptr as *const _,
                        sys::cudaDataType_t::CUDA_R_8I,
                        k_i,
                        a_ptr as *const _,
                        sys::cudaDataType_t::CUDA_R_8I,
                        k_i,
                        std::ptr::from_ref(&beta_i32).cast(),
                        c_ptr as *mut _,
                        sys::cudaDataType_t::CUDA_R_32I,
                        n_i,
                        sys::cublasComputeType_t::CUBLAS_COMPUTE_32I,
                        sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                    )
                    .result()
                    .map_err(cuda_err)?;
                }
            } else {
                // SAFETY: see above.
                unsafe {
                    sys::cublasGemmEx(
                        handle,
                        sys::cublasOperation_t::CUBLAS_OP_N,
                        sys::cublasOperation_t::CUBLAS_OP_N,
                        n_i,
                        m_i,
                        k_i,
                        std::ptr::from_ref(&alpha_i32).cast(),
                        b_ptr as *const _,
                        sys::cudaDataType_t::CUDA_R_8I,
                        n_i,
                        a_ptr as *const _,
                        sys::cudaDataType_t::CUDA_R_8I,
                        k_i,
                        std::ptr::from_ref(&beta_i32).cast(),
                        c_ptr as *mut _,
                        sys::cudaDataType_t::CUDA_R_32I,
                        n_i,
                        sys::cublasComputeType_t::CUBLAS_COMPUTE_32I,
                        sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                    )
                    .result()
                    .map_err(cuda_err)?;
                }
            }
        } // drop borrow guards

        // 3. Per-row scale + convert: FP16[i,j] = FP32[i,j] * a_row_scale[i] * b_col_scale[j]
        // FP32 output already has the dot product; just need to apply quant scales.
        // One block per row — a_scale is loaded once per row (amortized).
        {
            let threads = 256_u32;
            let mut builder = self
                .stream
                .launch_builder(&self.kernels.dequantize_i32_to_f16);
            builder.arg(out_f16);
            builder.arg(&c_i32);
            builder.arg(&a_row_scales);
            builder.arg(b_col_scales);
            builder.arg(&m_i);
            builder.arg(&n_i);
            // SAFETY: one block per row; reads m×n FP32, writes m×n FP16.
            unsafe { launch_kernel(builder, launch_per_row(m, threads, 0))? };
        }

        Ok(())
    }
}

/// Build RoPE cos/sin tables and upload to GPU.
fn build_rope_cache(
    stream: &Arc<CudaStream>,
    head_dim: usize,
    max_seq: usize,
    theta: f32,
) -> crate::Result<RopeCache<CudaTensor>> {
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

    let cos_gpu = stream.clone_htod(&cos_data).map_err(cuda_err)?;
    let sin_gpu = stream.clone_htod(&sin_data).map_err(cuda_err)?;

    Ok(RopeCache {
        cos: CudaTensor::new(cos_gpu),
        sin: CudaTensor::new(sin_gpu),
    })
}

// ---------------------------------------------------------------------------
// Driver trait implementation
// ---------------------------------------------------------------------------

#[expect(
    unsafe_code,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "CUDA kernel launches require unsafe; integer casts match GPU kernel i32 params"
)]
impl Driver for CudaDriver {
    type Tensor = CudaTensor;

    // --- Batching ---

    fn begin_batch(&self) -> crate::Result<()> {
        // No pool to reset — CUDA async allocator handles buffer lifecycle.
        Ok(())
    }

    fn end_batch(&self) -> crate::Result<()> {
        // Synchronise the stream: wait for all enqueued work to complete.
        self.stream.synchronize().map_err(cuda_err)
    }

    fn flush_batch(&self) -> crate::Result<()> {
        self.stream.synchronize().map_err(cuda_err)
    }

    // Pool cursor ops are no-ops — we don't use a pool. cudarc's CudaSlice::clone
    // does a full D2D memcpy (unlike Metal's refcounted buffers), so a pool-based
    // approach causes 4.5s of D2D copies. Instead we allocate via the CUDA async
    // allocator and let cuFreeAsync handle deallocation.

    // --- Allocation ---

    fn alloc_zeros(&self, n: usize) -> crate::Result<CudaTensor> {
        let buf = self.alloc_tensor(n)?;
        Ok(CudaTensor::new(buf))
    }

    fn clone_tensor(&self, tensor: &CudaTensor, _n: usize) -> crate::Result<CudaTensor> {
        let dst = self.stream.clone_dtod(&tensor.f32_buf).map_err(cuda_err)?;
        Ok(CudaTensor::new(dst))
    }

    // --- Batch preparation ---

    fn prepare_batch(
        &self,
        encodings: &[Encoding],
        max_seq: usize,
    ) -> crate::Result<BatchInputs<CudaTensor>> {
        let batch = encodings.len();
        let total = batch * max_seq;

        let mut input_ids = vec![0_i32; total];
        let mut token_type_ids = vec![0_i32; total];
        let mut position_ids = vec![0_i32; total];
        let mut attn_mask_int = vec![0_i32; total];

        for (b, enc) in encodings.iter().enumerate() {
            let offset = b * max_seq;
            for (i, &id) in enc.input_ids.iter().enumerate() {
                input_ids[offset + i] = id as i32;
            }
            for (i, &id) in enc.token_type_ids.iter().enumerate() {
                token_type_ids[offset + i] = id as i32;
            }
            for i in 0..enc.input_ids.len() {
                position_ids[offset + i] = i as i32;
            }
            for (i, &m) in enc.attention_mask.iter().enumerate() {
                attn_mask_int[offset + i] = m as i32;
            }
        }

        // Upload i32 tensors — we store them as CudaTensor with f32_buf holding
        // the i32 data bit-reinterpreted. The kernels read `int*` from these buffers.
        let ids_dev = self.stream.clone_htod(&input_ids).map_err(cuda_err)?;
        let ttype_dev = self.stream.clone_htod(&token_type_ids).map_err(cuda_err)?;
        let pos_dev = self.stream.clone_htod(&position_ids).map_err(cuda_err)?;
        let mask_int_dev = self.stream.clone_htod(&attn_mask_int).map_err(cuda_err)?;

        // Build float attention bias mask via kernel.
        let n = total as i32;
        let mut float_mask_dev = self.stream.alloc_zeros::<f32>(total).map_err(cuda_err)?;
        {
            let mut builder = self.stream.launch_builder(&self.kernels.build_attn_mask);
            builder.arg(&mut float_mask_dev);
            builder.arg(&mask_int_dev);
            builder.arg(&n);
            // SAFETY: kernel reads `total` i32 from mask_int_dev, writes `total` f32.
            // SAFETY: kernel args set above; config matches element count.
            unsafe { launch_kernel(builder, launch_1d(total)) }?;
        }

        // Build pooling mask on host, upload.
        let pooling_mask: Vec<f32> = attn_mask_int
            .iter()
            .map(|&m| if m == 1 { 1.0 } else { 0.0 })
            .collect();
        let pooling_dev = self.stream.clone_htod(&pooling_mask).map_err(cuda_err)?;

        let seq_lengths: Vec<usize> = encodings.iter().map(|e| e.input_ids.len()).collect();
        let total_tokens: usize = seq_lengths.iter().sum();

        // Wrap i32 device slices as CudaTensor. The f32_buf field is a
        // type-punned CudaSlice<f32> that kernels interpret as int*.
        // SAFETY: CudaSlice<i32> and CudaSlice<f32> have the same size (4 bytes).
        // Kernels cast the pointer to the appropriate type.
        let wrap_i32 = |s: CudaSlice<i32>| -> CudaTensor {
            CudaTensor::new(unsafe { std::mem::transmute::<CudaSlice<i32>, CudaSlice<f32>>(s) })
        };

        Ok(BatchInputs {
            input_ids: wrap_i32(ids_dev),
            attention_mask: wrap_i32(mask_int_dev),
            token_type_ids: wrap_i32(ttype_dev),
            position_ids: wrap_i32(pos_dev),
            float_mask: CudaTensor::new(float_mask_dev),
            pooling_mask: CudaTensor::new(pooling_dev),
            batch,
            max_seq,
            total_tokens,
            seq_lengths,
            cu_seqlens: None,
        })
    }

    fn prepare_batch_unpadded(
        &self,
        encodings: &[Encoding],
    ) -> crate::Result<BatchInputs<CudaTensor>> {
        let batch = encodings.len();
        let seq_lengths: Vec<usize> = encodings.iter().map(|e| e.input_ids.len()).collect();
        let total_tokens: usize = seq_lengths.iter().sum();
        let max_seq = seq_lengths
            .iter()
            .copied()
            .max()
            .unwrap_or(0)
            .next_multiple_of(8);

        // Build cu_seqlens
        let mut cu_seqlens = Vec::with_capacity(batch + 1);
        cu_seqlens.push(0);
        let mut cumsum = 0;
        for &len in &seq_lengths {
            cumsum += len;
            cu_seqlens.push(cumsum);
        }

        // Concatenate all tokens flat
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

        let ids_dev = self.stream.clone_htod(&input_ids).map_err(cuda_err)?;
        let ttype_dev = self.stream.clone_htod(&token_type_ids).map_err(cuda_err)?;
        let pos_dev = self.stream.clone_htod(&position_ids).map_err(cuda_err)?;

        // Build padded attention mask: [batch * max_seq]
        let padded_total = batch * max_seq;
        let mut attn_mask_int = vec![0_i32; padded_total];
        for (b, &len) in seq_lengths.iter().enumerate() {
            let offset = b * max_seq;
            for i in 0..len {
                attn_mask_int[offset + i] = 1;
            }
        }
        let mask_int_dev = self.stream.clone_htod(&attn_mask_int).map_err(cuda_err)?;

        // Float attention mask via kernel.
        let n = padded_total as i32;
        let mut float_mask_dev = self
            .stream
            .alloc_zeros::<f32>(padded_total)
            .map_err(cuda_err)?;
        {
            let mut builder = self.stream.launch_builder(&self.kernels.build_attn_mask);
            builder.arg(&mut float_mask_dev);
            builder.arg(&mask_int_dev);
            builder.arg(&n);
            // SAFETY: kernel reads padded_total i32, writes padded_total f32.
            // SAFETY: kernel args set above; config matches element count.
            unsafe { launch_kernel(builder, launch_1d(padded_total)) }?;
        }

        // Pooling mask
        let pooling_mask: Vec<f32> = attn_mask_int
            .iter()
            .map(|&m| if m == 1 { 1.0 } else { 0.0 })
            .collect();
        let pooling_dev = self.stream.clone_htod(&pooling_mask).map_err(cuda_err)?;

        let wrap_i32 = |s: CudaSlice<i32>| -> CudaTensor {
            // SAFETY: i32 and f32 are both 4 bytes; kernels cast to int* appropriately.
            CudaTensor::new(unsafe { std::mem::transmute::<CudaSlice<i32>, CudaSlice<f32>>(s) })
        };

        Ok(BatchInputs {
            input_ids: wrap_i32(ids_dev),
            attention_mask: wrap_i32(mask_int_dev),
            token_type_ids: wrap_i32(ttype_dev),
            position_ids: wrap_i32(pos_dev),
            float_mask: CudaTensor::new(float_mask_dev),
            pooling_mask: CudaTensor::new(pooling_dev),
            batch,
            max_seq,
            total_tokens,
            seq_lengths,
            cu_seqlens: Some(cu_seqlens),
        })
    }

    fn pad_to_batch(
        &self,
        flat: &CudaTensor,
        padded: &mut CudaTensor,
        seq_lengths: &[usize],
        max_seq: usize,
        dim: usize,
    ) -> crate::Result<()> {
        let batch = seq_lengths.len();
        let total = batch * max_seq * dim;
        let cu_dev = self.build_cu_seqlens(seq_lengths)?;

        let max_seq_i = max_seq as i32;
        let dim_i = dim as i32;
        let batch_i = batch as i32;

        let mut builder = self.stream.launch_builder(&self.kernels.pad_to_batch);
        builder.arg(&mut padded.f32_buf);
        builder.arg(&flat.f32_buf);
        builder.arg(&cu_dev);
        builder.arg(&max_seq_i);
        builder.arg(&dim_i);
        builder.arg(&batch_i);
        // SAFETY: kernel reads flat[total_tokens * dim], writes padded[batch * max_seq * dim].
        // SAFETY: kernel args set above; config matches element count.
        unsafe { launch_kernel(builder, launch_1d(total)) }
    }

    fn unpad_from_batch(
        &self,
        padded: &CudaTensor,
        flat: &mut CudaTensor,
        seq_lengths: &[usize],
        max_seq: usize,
        dim: usize,
    ) -> crate::Result<()> {
        let batch = seq_lengths.len();
        let total = batch * max_seq * dim;
        let cu_dev = self.build_cu_seqlens(seq_lengths)?;

        let max_seq_i = max_seq as i32;
        let dim_i = dim as i32;
        let batch_i = batch as i32;

        let mut builder = self.stream.launch_builder(&self.kernels.unpad_from_batch);
        builder.arg(&mut flat.f32_buf);
        builder.arg(&padded.f32_buf);
        builder.arg(&cu_dev);
        builder.arg(&max_seq_i);
        builder.arg(&dim_i);
        builder.arg(&batch_i);
        // SAFETY: kernel reads padded[batch*max_seq*dim], writes flat[total_tokens*dim].
        // SAFETY: kernel args set above; config matches element count.
        unsafe { launch_kernel(builder, launch_1d(total)) }
    }

    // --- Embedding operations ---

    fn embedding_lookup(
        &self,
        word_ids: &CudaTensor,
        embedding_table: &CudaTensor,
        seq_len: usize,
        hidden: usize,
    ) -> crate::Result<CudaTensor> {
        let n = seq_len * hidden;
        let mut output = self.alloc_tensor(n)?;
        let total_i = n as i32;
        let hidden_i = hidden as i32;

        let mut builder = self.stream.launch_builder(&self.kernels.embedding_lookup);
        builder.arg(&mut output);
        builder.arg(&embedding_table.f32_buf);
        builder.arg(&word_ids.f32_buf); // Actually i32* — kernel casts.
        builder.arg(&total_i);
        builder.arg(&hidden_i);
        // SAFETY: kernel reads seq_len indices, gathers from embedding_table,
        // writes seq_len * hidden floats to output.
        // SAFETY: kernel args set above; launch config matches element count.
        unsafe { launch_kernel(builder, launch_1d(n)) }?;
        Ok(CudaTensor::new(output))
    }

    fn add_embeddings(
        &self,
        hidden: &mut CudaTensor,
        table: &CudaTensor,
        ids: &CudaTensor,
        seq_len: usize,
        hidden_dim: usize,
    ) -> crate::Result<()> {
        let n = seq_len * hidden_dim;
        let total_i = n as i32;
        let hidden_i = hidden_dim as i32;

        let mut builder = self.stream.launch_builder(&self.kernels.add_embeddings);
        builder.arg(&mut hidden.f32_buf);
        builder.arg(&table.f32_buf);
        builder.arg(&ids.f32_buf); // Actually i32*.
        builder.arg(&total_i);
        builder.arg(&hidden_i);
        // SAFETY: kernel reads/writes n floats in hidden, reads from table.
        // SAFETY: kernel args set above; config matches element count.
        unsafe { launch_kernel(builder, launch_1d(n)) }
    }

    // --- Normalization ---

    fn layer_norm(
        &self,
        output: &mut CudaTensor,
        input: &CudaTensor,
        weight: &CudaTensor,
        bias: &CudaTensor,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> crate::Result<()> {
        let threads = 256_u32;
        let shared = 2 * threads * 4; // 2 arrays of threads f32

        let rows_i = rows as i32;
        let cols_i = cols as i32;

        let mut builder = self.stream.launch_builder(&self.kernels.layer_norm);
        builder.arg(&mut output.f32_buf);
        builder.arg(&input.f32_buf);
        builder.arg(&weight.f32_buf);
        builder.arg(&bias.f32_buf);
        builder.arg(&rows_i);
        builder.arg(&cols_i);
        builder.arg(&eps);
        // SAFETY: one block per row; shared mem holds 2 × threads floats for reductions.
        // SAFETY: kernel args set above; one block per row.
        unsafe { launch_kernel(builder, launch_per_row(rows, threads, shared)) }
    }

    // --- Linear algebra ---

    fn gemm(
        &self,
        a: &CudaTensor,
        b: &CudaTensor,
        output: &mut CudaTensor,
        m: usize,
        n: usize,
        k: usize,
        transpose_b: bool,
    ) -> crate::Result<()> {
        let alpha = 1.0_f32;
        let beta = 0.0_f32;
        let handle = *self.blas.handle();
        let m_i = m as i32;
        let n_i = n as i32;
        let k_i = k as i32;

        let (a_ptr, _a_sync) = a.f32_buf.device_ptr(&self.stream);
        let (b_ptr, _b_sync) = b.f32_buf.device_ptr(&self.stream);
        let (c_ptr, _c_sync) = output.f32_buf.device_ptr_mut(&self.stream);

        if transpose_b {
            // Row-major A[m,k] @ B[n,k]^T = C[m,n]
            // cuBLAS col-major: C(n,m) = B(n,k) @ A(k,m)
            // transa=T on B (k→n transpose), transb=N on A
            // SAFETY: all device pointers from valid CudaSlice allocations sized for GEMM dims.
            unsafe {
                sys::cublasGemmEx(
                    handle,
                    sys::cublasOperation_t::CUBLAS_OP_T,
                    sys::cublasOperation_t::CUBLAS_OP_N,
                    n_i,
                    m_i,
                    k_i,
                    std::ptr::from_ref(&alpha).cast(),
                    b_ptr as *const _,
                    sys::cudaDataType_t::CUDA_R_32F,
                    k_i,
                    a_ptr as *const _,
                    sys::cudaDataType_t::CUDA_R_32F,
                    k_i,
                    std::ptr::from_ref(&beta).cast(),
                    c_ptr as *mut _,
                    sys::cudaDataType_t::CUDA_R_32F,
                    n_i,
                    sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32,
                    sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                )
                .result()
                .map_err(cuda_err)?;
            }
        } else {
            // Row-major A[m,k] @ B[k,n] = C[m,n]
            // cuBLAS col-major: C(n,m) = B(n,k) @ A(k,m)
            // transa=N on B, transb=N on A
            // SAFETY: see above.
            unsafe {
                sys::cublasGemmEx(
                    handle,
                    sys::cublasOperation_t::CUBLAS_OP_N,
                    sys::cublasOperation_t::CUBLAS_OP_N,
                    n_i,
                    m_i,
                    k_i,
                    std::ptr::from_ref(&alpha).cast(),
                    b_ptr as *const _,
                    sys::cudaDataType_t::CUDA_R_32F,
                    n_i,
                    a_ptr as *const _,
                    sys::cudaDataType_t::CUDA_R_32F,
                    k_i,
                    std::ptr::from_ref(&beta).cast(),
                    c_ptr as *mut _,
                    sys::cudaDataType_t::CUDA_R_32F,
                    n_i,
                    sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32,
                    sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                )
                .result()
                .map_err(cuda_err)?;
            }
        }
        Ok(())
    }

    fn gemm_batched(
        &self,
        a: &CudaTensor,
        b: &CudaTensor,
        output: &mut CudaTensor,
        m: usize,
        n: usize,
        k: usize,
        transpose_b: bool,
        stride_a: usize,
        stride_b: usize,
        stride_c: usize,
        batch_count: usize,
    ) -> crate::Result<()> {
        let alpha = 1.0_f32;
        let beta = 0.0_f32;
        let handle = *self.blas.handle();
        let m_i = m as i32;
        let n_i = n as i32;
        let k_i = k as i32;

        let (a_ptr, _a_sync) = a.f32_buf.device_ptr(&self.stream);
        let (b_ptr, _b_sync) = b.f32_buf.device_ptr(&self.stream);
        let (c_ptr, _c_sync) = output.f32_buf.device_ptr_mut(&self.stream);

        let (op_b, ldb, stride_b_64) = if transpose_b {
            (
                sys::cublasOperation_t::CUBLAS_OP_T,
                k_i,
                i64::try_from(stride_b).unwrap_or(0),
            )
        } else {
            (
                sys::cublasOperation_t::CUBLAS_OP_N,
                n_i,
                i64::try_from(stride_b).unwrap_or(0),
            )
        };

        // SAFETY: all device pointers from valid allocations; strides match layout.
        unsafe {
            sys::cublasGemmStridedBatchedEx(
                handle,
                op_b,
                sys::cublasOperation_t::CUBLAS_OP_N,
                n_i,
                m_i,
                k_i,
                std::ptr::from_ref(&alpha).cast(),
                b_ptr as *const _,
                sys::cudaDataType_t::CUDA_R_32F,
                ldb,
                stride_b_64,
                a_ptr as *const _,
                sys::cudaDataType_t::CUDA_R_32F,
                k_i,
                i64::try_from(stride_a).unwrap_or(0),
                std::ptr::from_ref(&beta).cast(),
                c_ptr as *mut _,
                sys::cudaDataType_t::CUDA_R_32F,
                n_i,
                i64::try_from(stride_c).unwrap_or(0),
                batch_count as i32,
                sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32,
                sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            )
            .result()
            .map_err(cuda_err)?;
        }
        Ok(())
    }

    // --- Attention ---

    fn fused_scale_mask_softmax(
        &self,
        scores: &mut CudaTensor,
        mask: &CudaTensor,
        batch: usize,
        num_heads: usize,
        seq_len: usize,
        scale: f32,
    ) -> crate::Result<()> {
        let total_rows = batch * num_heads * seq_len;
        let threads = 256_u32;
        let shared = threads * 4; // One float per thread for reductions.

        let batch_i = batch as i32;
        let nh_i = num_heads as i32;
        let seq_i = seq_len as i32;

        let mut builder = self
            .stream
            .launch_builder(&self.kernels.fused_scale_mask_softmax);
        builder.arg(&mut scores.f32_buf);
        builder.arg(&mask.f32_buf);
        builder.arg(&batch_i);
        builder.arg(&nh_i);
        builder.arg(&seq_i);
        builder.arg(&scale);
        // SAFETY: one block per row, shared mem for thread-level reductions.
        // SAFETY: kernel args set above; one block per row.
        unsafe { launch_kernel(builder, launch_per_row(total_rows, threads, shared)) }
    }

    fn fused_scale_mask_softmax_windowed(
        &self,
        scores: &mut CudaTensor,
        mask: &CudaTensor,
        batch: usize,
        num_heads: usize,
        seq_len: usize,
        scale: f32,
        window_size: usize,
    ) -> crate::Result<()> {
        let total_rows = batch * num_heads * seq_len;
        let threads = 256_u32;
        let shared = threads * 4;

        let batch_i = batch as i32;
        let nh_i = num_heads as i32;
        let seq_i = seq_len as i32;
        let win_i = window_size as i32;

        let mut builder = self
            .stream
            .launch_builder(&self.kernels.fused_scale_mask_softmax_windowed);
        builder.arg(&mut scores.f32_buf);
        builder.arg(&mask.f32_buf);
        builder.arg(&batch_i);
        builder.arg(&nh_i);
        builder.arg(&seq_i);
        builder.arg(&scale);
        builder.arg(&win_i);
        // SAFETY: one block per row, shared mem for reductions.
        // SAFETY: kernel args set above; one block per row.
        unsafe { launch_kernel(builder, launch_per_row(total_rows, threads, shared)) }
    }

    fn build_attn_mask(
        &self,
        output: &mut CudaTensor,
        int_mask: &CudaTensor,
        n: usize,
    ) -> crate::Result<()> {
        let n_i = n as i32;
        let mut builder = self.stream.launch_builder(&self.kernels.build_attn_mask);
        builder.arg(&mut output.f32_buf);
        builder.arg(&int_mask.f32_buf); // Actually i32*.
        builder.arg(&n_i);
        // SAFETY: reads n i32, writes n f32.
        // SAFETY: kernel args set above; config matches element count.
        unsafe { launch_kernel(builder, launch_1d(n)) }
    }

    fn qkv_split(
        &self,
        q: &mut CudaTensor,
        k: &mut CudaTensor,
        v: &mut CudaTensor,
        qkv: &CudaTensor,
        batch: usize,
        seq: usize,
        hidden: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> crate::Result<()> {
        let total = batch * num_heads * seq * head_dim;
        let batch_i = batch as i32;
        let seq_i = seq as i32;
        let hidden_i = hidden as i32;
        let nh_i = num_heads as i32;
        let hd_i = head_dim as i32;

        let mut builder = self.stream.launch_builder(&self.kernels.qkv_split);
        builder.arg(&mut q.f32_buf);
        builder.arg(&mut k.f32_buf);
        builder.arg(&mut v.f32_buf);
        builder.arg(&qkv.f32_buf);
        builder.arg(&batch_i);
        builder.arg(&seq_i);
        builder.arg(&hidden_i);
        builder.arg(&nh_i);
        builder.arg(&hd_i);
        // SAFETY: reads batch*seq*3*hidden from qkv; writes total to each of q, k, v.
        // SAFETY: kernel args set above; config matches element count.
        unsafe { launch_kernel(builder, launch_1d(total)) }
    }

    fn banded_qk(
        &self,
        q: &CudaTensor,
        k: &CudaTensor,
        scores: &mut CudaTensor,
        batch_heads: usize,
        seq: usize,
        head_dim: usize,
        window: usize,
        stride_qk: usize,
        stride_scores: usize,
    ) -> crate::Result<()> {
        let total = batch_heads * seq * window;
        let bh_i = batch_heads as i32;
        let seq_i = seq as i32;
        let hd_i = head_dim as i32;
        let win_i = window as i32;
        let sqk_i = stride_qk as i32;
        let ss_i = stride_scores as i32;

        let mut builder = self.stream.launch_builder(&self.kernels.banded_qk);
        builder.arg(&mut scores.f32_buf);
        builder.arg(&q.f32_buf);
        builder.arg(&k.f32_buf);
        builder.arg(&bh_i);
        builder.arg(&seq_i);
        builder.arg(&hd_i);
        builder.arg(&win_i);
        builder.arg(&sqk_i);
        builder.arg(&ss_i);
        // SAFETY: reads q/k within strides, writes scores within stride_scores bounds.
        // SAFETY: kernel args set above; config matches element count.
        unsafe { launch_kernel(builder, launch_1d(total)) }
    }

    fn banded_sv(
        &self,
        scores: &CudaTensor,
        v: &CudaTensor,
        output: &mut CudaTensor,
        batch_heads: usize,
        seq: usize,
        head_dim: usize,
        window: usize,
        stride_scores: usize,
        stride_v: usize,
        stride_out: usize,
    ) -> crate::Result<()> {
        let total = batch_heads * seq * head_dim;
        let bh_i = batch_heads as i32;
        let seq_i = seq as i32;
        let hd_i = head_dim as i32;
        let win_i = window as i32;
        let ss_i = stride_scores as i32;
        let sv_i = stride_v as i32;
        let so_i = stride_out as i32;

        let mut builder = self.stream.launch_builder(&self.kernels.banded_sv);
        builder.arg(&mut output.f32_buf);
        builder.arg(&scores.f32_buf);
        builder.arg(&v.f32_buf);
        builder.arg(&bh_i);
        builder.arg(&seq_i);
        builder.arg(&hd_i);
        builder.arg(&win_i);
        builder.arg(&ss_i);
        builder.arg(&sv_i);
        builder.arg(&so_i);
        // SAFETY: reads scores/v within stride bounds, writes output within stride_out.
        // SAFETY: kernel args set above; config matches element count.
        unsafe { launch_kernel(builder, launch_1d(total)) }
    }

    fn banded_softmax(
        &self,
        scores: &mut CudaTensor,
        total_rows: usize,
        window: usize,
        scale: f32,
    ) -> crate::Result<()> {
        let threads = 256_u32.min(window as u32).max(32);
        let shared = threads * 4;
        let total_rows_i = total_rows as i32;
        let win_i = window as i32;

        let mut builder = self.stream.launch_builder(&self.kernels.banded_softmax);
        builder.arg(&mut scores.f32_buf);
        builder.arg(&total_rows_i);
        builder.arg(&win_i);
        builder.arg(&scale);
        // SAFETY: one block per row, shared mem for reductions.
        // SAFETY: kernel args set above; one block per row.
        unsafe { launch_kernel(builder, launch_per_row(total_rows, threads, shared)) }
    }

    fn attn_reshape(
        &self,
        output: &mut CudaTensor,
        input: &CudaTensor,
        batch: usize,
        seq: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> crate::Result<()> {
        let total = batch * seq * num_heads * head_dim;
        let batch_i = batch as i32;
        let seq_i = seq as i32;
        let nh_i = num_heads as i32;
        let hd_i = head_dim as i32;

        let mut builder = self.stream.launch_builder(&self.kernels.attn_reshape);
        builder.arg(&mut output.f32_buf);
        builder.arg(&input.f32_buf);
        builder.arg(&batch_i);
        builder.arg(&seq_i);
        builder.arg(&nh_i);
        builder.arg(&hd_i);
        // SAFETY: reads/writes total elements with head→linear layout transform.
        // SAFETY: kernel args set above; config matches element count.
        unsafe { launch_kernel(builder, launch_1d(total)) }
    }

    fn apply_rope(
        &self,
        qk: &mut CudaTensor,
        cos: &CudaTensor,
        sin: &CudaTensor,
        num_rows: usize,
        seq_len: usize,
        head_dim: usize,
        num_heads: usize,
    ) -> crate::Result<()> {
        let half = head_dim / 2;
        let total = num_rows * half;
        let nr_i = num_rows as i32;
        let seq_i = seq_len as i32;
        let hd_i = head_dim as i32;
        let nh_i = num_heads as i32;

        let mut builder = self.stream.launch_builder(&self.kernels.rope_cached);
        builder.arg(&mut qk.f32_buf);
        builder.arg(&cos.f32_buf);
        builder.arg(&sin.f32_buf);
        builder.arg(&nr_i);
        builder.arg(&seq_i);
        builder.arg(&hd_i);
        builder.arg(&nh_i);
        // SAFETY: reads/writes num_rows * head_dim elements; cos/sin are [max_seq, half].
        // SAFETY: kernel args set above; config matches element count.
        unsafe { launch_kernel(builder, launch_1d(total)) }
    }

    // --- Tensor manipulation ---

    fn split_gate_value(
        &self,
        first: &mut CudaTensor,
        second: &mut CudaTensor,
        input: &CudaTensor,
        rows: usize,
        cols: usize,
    ) -> crate::Result<()> {
        let total = rows * cols;
        let rows_i = rows as i32;
        let cols_i = cols as i32;

        let mut builder = self.stream.launch_builder(&self.kernels.split_gate_value);
        builder.arg(&mut first.f32_buf);
        builder.arg(&mut second.f32_buf);
        builder.arg(&input.f32_buf);
        builder.arg(&rows_i);
        builder.arg(&cols_i);
        // SAFETY: reads rows*2*cols from input, writes rows*cols to each half.
        // SAFETY: kernel args set above; config matches element count.
        unsafe { launch_kernel(builder, launch_1d(total)) }
    }

    // --- Activations ---

    fn gelu(&self, x: &mut CudaTensor, n: usize) -> crate::Result<()> {
        let n_i = n as i32;
        let mut builder = self.stream.launch_builder(&self.kernels.gelu);
        builder.arg(&mut x.f32_buf);
        builder.arg(&n_i);
        // SAFETY: in-place on n elements.
        // SAFETY: kernel args set above; config matches element count.
        unsafe { launch_kernel(builder, launch_1d(n)) }
    }

    fn swiglu(
        &self,
        value: &CudaTensor,
        gate: &CudaTensor,
        output: &mut CudaTensor,
        n: usize,
    ) -> crate::Result<()> {
        let n_i = n as i32;
        let mut builder = self.stream.launch_builder(&self.kernels.swiglu);
        builder.arg(&mut output.f32_buf);
        builder.arg(&value.f32_buf);
        builder.arg(&gate.f32_buf);
        builder.arg(&n_i);
        // SAFETY: reads n from value/gate, writes n to output.
        // SAFETY: kernel args set above; config matches element count.
        unsafe { launch_kernel(builder, launch_1d(n)) }
    }

    fn geglu(
        &self,
        value: &CudaTensor,
        gate: &CudaTensor,
        output: &mut CudaTensor,
        n: usize,
    ) -> crate::Result<()> {
        let n_i = n as i32;
        let mut builder = self.stream.launch_builder(&self.kernels.geglu);
        builder.arg(&mut output.f32_buf);
        builder.arg(&value.f32_buf);
        builder.arg(&gate.f32_buf);
        builder.arg(&n_i);
        // SAFETY: reads n from value/gate, writes n to output.
        // SAFETY: kernel args set above; config matches element count.
        unsafe { launch_kernel(builder, launch_1d(n)) }
    }

    fn fused_bias_gelu(
        &self,
        x: &mut CudaTensor,
        bias: &CudaTensor,
        rows: usize,
        cols: usize,
    ) -> crate::Result<()> {
        let total = rows * cols;
        let rows_i = rows as i32;
        let cols_i = cols as i32;

        let mut builder = self.stream.launch_builder(&self.kernels.fused_bias_gelu);
        builder.arg(&mut x.f32_buf);
        builder.arg(&bias.f32_buf);
        builder.arg(&rows_i);
        builder.arg(&cols_i);
        // SAFETY: in-place on rows*cols elements, bias broadcast per row.
        // SAFETY: kernel args set above; config matches element count.
        unsafe { launch_kernel(builder, launch_1d(total)) }
    }

    // --- Fused residual operations ---

    fn fused_bias_residual(
        &self,
        output: &mut CudaTensor,
        input: &CudaTensor,
        bias: &CudaTensor,
        residual: &CudaTensor,
        n: usize,
        cols: usize,
    ) -> crate::Result<()> {
        let rows = n / cols;
        let rows_i = rows as i32;
        let cols_i = cols as i32;

        let mut builder = self
            .stream
            .launch_builder(&self.kernels.fused_bias_residual);
        builder.arg(&mut output.f32_buf);
        builder.arg(&input.f32_buf);
        builder.arg(&bias.f32_buf);
        builder.arg(&residual.f32_buf);
        builder.arg(&rows_i);
        builder.arg(&cols_i);
        // SAFETY: reads input/bias/residual, writes output; all n elements.
        // SAFETY: kernel args set above; config matches element count.
        unsafe { launch_kernel(builder, launch_1d(n)) }
    }

    fn fused_residual_layernorm(
        &self,
        output: &mut CudaTensor,
        hidden: &CudaTensor,
        residual: &CudaTensor,
        weight: &CudaTensor,
        bias: &CudaTensor,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> crate::Result<()> {
        let threads = 256_u32;
        let shared = threads * 4; // One array for reductions.

        let rows_i = rows as i32;
        let cols_i = cols as i32;

        let mut builder = self
            .stream
            .launch_builder(&self.kernels.fused_residual_layernorm);
        builder.arg(&mut output.f32_buf);
        builder.arg(&hidden.f32_buf);
        builder.arg(&residual.f32_buf);
        builder.arg(&weight.f32_buf);
        builder.arg(&bias.f32_buf);
        builder.arg(&rows_i);
        builder.arg(&cols_i);
        builder.arg(&eps);
        // SAFETY: one block per row, shared mem for reductions.
        // SAFETY: kernel args set above; one block per row.
        unsafe { launch_kernel(builder, launch_per_row(rows, threads, shared)) }
    }

    fn residual_add(
        &self,
        output: &mut CudaTensor,
        hidden: &CudaTensor,
        residual: &CudaTensor,
        n: usize,
    ) -> crate::Result<()> {
        let n_i = n as i32;
        let mut builder = self.stream.launch_builder(&self.kernels.residual_add);
        builder.arg(&mut output.f32_buf);
        builder.arg(&hidden.f32_buf);
        builder.arg(&residual.f32_buf);
        builder.arg(&n_i);
        // SAFETY: reads n from hidden/residual, writes n to output.
        // SAFETY: kernel args set above; config matches element count.
        unsafe { launch_kernel(builder, launch_1d(n)) }
    }

    fn add_bias(
        &self,
        x: &mut CudaTensor,
        bias: &CudaTensor,
        rows: usize,
        cols: usize,
    ) -> crate::Result<()> {
        let total = rows * cols;
        let rows_i = rows as i32;
        let cols_i = cols as i32;

        let mut builder = self.stream.launch_builder(&self.kernels.add_bias);
        builder.arg(&mut x.f32_buf);
        builder.arg(&bias.f32_buf);
        builder.arg(&rows_i);
        builder.arg(&cols_i);
        // SAFETY: in-place on rows*cols elements, bias broadcast.
        // SAFETY: kernel args set above; config matches element count.
        unsafe { launch_kernel(builder, launch_1d(total)) }
    }

    // --- Pooling ---

    fn cls_pool(
        &self,
        output: &mut CudaTensor,
        hidden: &CudaTensor,
        batch: usize,
        seq: usize,
        hidden_dim: usize,
    ) -> crate::Result<()> {
        let total = batch * hidden_dim;
        let batch_i = batch as i32;
        let seq_i = seq as i32;
        let hd_i = hidden_dim as i32;

        let mut builder = self.stream.launch_builder(&self.kernels.cls_pool);
        builder.arg(&mut output.f32_buf);
        builder.arg(&hidden.f32_buf);
        builder.arg(&batch_i);
        builder.arg(&seq_i);
        builder.arg(&hd_i);
        // SAFETY: extracts row 0 per batch element from [batch, seq, hidden].
        // SAFETY: kernel args set above; config matches element count.
        unsafe { launch_kernel(builder, launch_1d(total)) }
    }

    fn mean_pool(
        &self,
        output: &mut CudaTensor,
        hidden: &CudaTensor,
        mask: &CudaTensor,
        batch: usize,
        seq: usize,
        hidden_dim: usize,
    ) -> crate::Result<()> {
        let total = batch * hidden_dim;
        let batch_i = batch as i32;
        let seq_i = seq as i32;
        let hd_i = hidden_dim as i32;

        let mut builder = self.stream.launch_builder(&self.kernels.mean_pool);
        builder.arg(&mut output.f32_buf);
        builder.arg(&hidden.f32_buf);
        builder.arg(&mask.f32_buf);
        builder.arg(&batch_i);
        builder.arg(&seq_i);
        builder.arg(&hd_i);
        // SAFETY: reads hidden[batch*seq*hidden_dim], mask[batch*seq], writes batch*hidden_dim.
        // SAFETY: kernel args set above; config matches element count.
        unsafe { launch_kernel(builder, launch_1d(total)) }
    }

    // --- Post-processing ---

    fn l2_normalize(&self, data: &mut CudaTensor, rows: usize, cols: usize) -> crate::Result<()> {
        let threads = 256_u32;
        let shared = threads * 4;
        let rows_i = rows as i32;
        let cols_i = cols as i32;

        let mut builder = self.stream.launch_builder(&self.kernels.l2_normalize);
        builder.arg(&mut data.f32_buf);
        builder.arg(&rows_i);
        builder.arg(&cols_i);
        // SAFETY: one block per row, shared mem for sum-of-squares reduction.
        // SAFETY: kernel args set above; one block per row.
        unsafe { launch_kernel(builder, launch_per_row(rows, threads, shared)) }
    }

    fn to_host(
        &self,
        tensor: &CudaTensor,
        batch: usize,
        dim: usize,
    ) -> crate::Result<Vec<Vec<f32>>> {
        let host = self.stream.clone_dtoh(&tensor.f32_buf).map_err(cuda_err)?;
        let mut results = Vec::with_capacity(batch);
        for b in 0..batch {
            results.push(host[b * dim..(b + 1) * dim].to_vec());
        }
        Ok(results)
    }

    // =======================================================================
    // FP16 operations
    // =======================================================================

    fn alloc_zeros_f16(&self, n: usize) -> crate::Result<CudaTensor> {
        let f16_buf = self.alloc_f16_tensor(n)?;
        // Create a zero-length f32 placeholder — the f16 is the primary buffer.
        let dummy = self.stream.alloc_zeros::<f32>(1).map_err(cuda_err)?;
        Ok(CudaTensor::new_f16_only(f16_buf, dummy))
    }

    fn f32_to_f16(
        &self,
        output: &mut CudaTensor,
        input: &CudaTensor,
        n: usize,
    ) -> crate::Result<()> {
        let n_i = n as i32;
        let fp16 = output
            .fp16
            .as_mut()
            .ok_or_else(|| crate::Error::Cuda("f32_to_f16: output has no FP16 buffer".into()))?;

        let mut builder = self.stream.launch_builder(&self.kernels.f32_to_f16);
        builder.arg(fp16);
        builder.arg(&input.f32_buf);
        builder.arg(&n_i);
        // SAFETY: reads n f32 from input, writes n u16 to output's fp16 buffer.
        // SAFETY: kernel args set above; config matches element count.
        unsafe { launch_kernel(builder, launch_1d(n)) }
    }

    fn f16_to_f32(
        &self,
        output: &mut CudaTensor,
        input: &CudaTensor,
        n: usize,
    ) -> crate::Result<()> {
        let n_i = n as i32;
        let fp16_in = input
            .fp16
            .as_ref()
            .ok_or_else(|| crate::Error::Cuda("f16_to_f32: input has no FP16 buffer".into()))?;

        let mut builder = self.stream.launch_builder(&self.kernels.f16_to_f32);
        builder.arg(&mut output.f32_buf);
        builder.arg(fp16_in);
        builder.arg(&n_i);
        // SAFETY: reads n u16 from input's fp16, writes n f32 to output.
        // SAFETY: kernel args set above; config matches element count.
        unsafe { launch_kernel(builder, launch_1d(n)) }
    }

    fn gemm_f16(
        &self,
        a: &CudaTensor,
        b: &CudaTensor,
        output: &mut CudaTensor,
        m: usize,
        n: usize,
        k: usize,
        transpose_b: bool,
    ) -> crate::Result<()> {
        // INT8 fast path: if B has pre-quantized INT8 weights, use cuBLASLt
        // with fused per-row/per-column scale epilogue (eliminates dequantize kernel).
        if let (Some(b_i8), Some(b_scales)) = (&b.int8, &b.int8_col_scales) {
            return self.gemm_i8_impl(a, b_i8, b_scales, output, m, n, k, transpose_b);
        }

        let alpha = 1.0_f32;
        let beta = 0.0_f32;
        let handle = *self.blas.handle();
        let m_i = m as i32;
        let n_i = n as i32;
        let k_i = k as i32;

        let a_f16 = a
            .fp16_ref()
            .ok_or_else(|| crate::Error::Cuda("gemm_f16: A has no FP16 buffer".into()))?;
        let b_f16 = b
            .fp16_ref()
            .ok_or_else(|| crate::Error::Cuda("gemm_f16: B has no FP16 buffer".into()))?;
        let out_f16 = output
            .fp16
            .as_mut()
            .ok_or_else(|| crate::Error::Cuda("gemm_f16: output has no FP16 buffer".into()))?;

        let (a_ptr, _a_sync) = a_f16.device_ptr(&self.stream);
        let (b_ptr, _b_sync) = b_f16.device_ptr(&self.stream);
        let (c_ptr, _c_sync) = out_f16.device_ptr_mut(&self.stream);

        if transpose_b {
            // SAFETY: FP16 buffers sized for GEMM dims; FP32 accumulation for precision.
            unsafe {
                sys::cublasGemmEx(
                    handle,
                    sys::cublasOperation_t::CUBLAS_OP_T,
                    sys::cublasOperation_t::CUBLAS_OP_N,
                    n_i,
                    m_i,
                    k_i,
                    std::ptr::from_ref(&alpha).cast(),
                    b_ptr as *const _,
                    sys::cudaDataType_t::CUDA_R_16F,
                    k_i,
                    a_ptr as *const _,
                    sys::cudaDataType_t::CUDA_R_16F,
                    k_i,
                    std::ptr::from_ref(&beta).cast(),
                    c_ptr as *mut _,
                    sys::cudaDataType_t::CUDA_R_16F,
                    n_i,
                    sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                )
                .result()
                .map_err(cuda_err)?;
            }
        } else {
            // SAFETY: see above.
            unsafe {
                sys::cublasGemmEx(
                    handle,
                    sys::cublasOperation_t::CUBLAS_OP_N,
                    sys::cublasOperation_t::CUBLAS_OP_N,
                    n_i,
                    m_i,
                    k_i,
                    std::ptr::from_ref(&alpha).cast(),
                    b_ptr as *const _,
                    sys::cudaDataType_t::CUDA_R_16F,
                    n_i,
                    a_ptr as *const _,
                    sys::cudaDataType_t::CUDA_R_16F,
                    k_i,
                    std::ptr::from_ref(&beta).cast(),
                    c_ptr as *mut _,
                    sys::cudaDataType_t::CUDA_R_16F,
                    n_i,
                    sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                )
                .result()
                .map_err(cuda_err)?;
            }
        }
        Ok(())
    }

    fn gemm_batched_f16(
        &self,
        a: &CudaTensor,
        b: &CudaTensor,
        output: &mut CudaTensor,
        m: usize,
        n: usize,
        k: usize,
        transpose_b: bool,
        stride_a: usize,
        stride_b: usize,
        stride_c: usize,
        batch_count: usize,
    ) -> crate::Result<()> {
        let alpha = 1.0_f32;
        let beta = 0.0_f32;
        let handle = *self.blas.handle();
        let m_i = m as i32;
        let n_i = n as i32;
        let k_i = k as i32;

        let a_f16 = a
            .fp16_ref()
            .ok_or_else(|| crate::Error::Cuda("gemm_batched_f16: A has no FP16 buffer".into()))?;
        let b_f16 = b
            .fp16_ref()
            .ok_or_else(|| crate::Error::Cuda("gemm_batched_f16: B has no FP16 buffer".into()))?;
        let out_f16 = output.fp16.as_mut().ok_or_else(|| {
            crate::Error::Cuda("gemm_batched_f16: output has no FP16 buffer".into())
        })?;

        let (a_ptr, _a_sync) = a_f16.device_ptr(&self.stream);
        let (b_ptr, _b_sync) = b_f16.device_ptr(&self.stream);
        let (c_ptr, _c_sync) = out_f16.device_ptr_mut(&self.stream);

        let (op_b, ldb, stride_b_64) = if transpose_b {
            (
                sys::cublasOperation_t::CUBLAS_OP_T,
                k_i,
                i64::try_from(stride_b).unwrap_or(0),
            )
        } else {
            (
                sys::cublasOperation_t::CUBLAS_OP_N,
                n_i,
                i64::try_from(stride_b).unwrap_or(0),
            )
        };

        // SAFETY: FP16 buffers with FP32 accumulation; strides match layout.
        unsafe {
            sys::cublasGemmStridedBatchedEx(
                handle,
                op_b,
                sys::cublasOperation_t::CUBLAS_OP_N,
                n_i,
                m_i,
                k_i,
                std::ptr::from_ref(&alpha).cast(),
                b_ptr as *const _,
                sys::cudaDataType_t::CUDA_R_16F,
                ldb,
                stride_b_64,
                a_ptr as *const _,
                sys::cudaDataType_t::CUDA_R_16F,
                k_i,
                i64::try_from(stride_a).unwrap_or(0),
                std::ptr::from_ref(&beta).cast(),
                c_ptr as *mut _,
                sys::cudaDataType_t::CUDA_R_16F,
                n_i,
                i64::try_from(stride_c).unwrap_or(0),
                batch_count as i32,
                sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            )
            .result()
            .map_err(cuda_err)?;
        }
        Ok(())
    }

    fn layer_norm_f16(
        &self,
        output: &mut CudaTensor,
        input: &CudaTensor,
        weight: &CudaTensor,
        bias: &CudaTensor,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> crate::Result<()> {
        let threads = 256_u32;
        let shared = 2 * threads * 4;
        let rows_i = rows as i32;
        let cols_i = cols as i32;

        let in_f16 = input
            .fp16_ref()
            .ok_or_else(|| crate::Error::Cuda("layer_norm_f16: input has no FP16 buffer".into()))?;
        let out_f16 = output.fp16.as_mut().ok_or_else(|| {
            crate::Error::Cuda("layer_norm_f16: output has no FP16 buffer".into())
        })?;

        let mut builder = self.stream.launch_builder(&self.kernels.layer_norm_f16);
        builder.arg(out_f16);
        builder.arg(in_f16);
        builder.arg(&weight.f32_buf); // Norm weights stay FP32.
        builder.arg(&bias.f32_buf); // Norm bias stays FP32.
        builder.arg(&rows_i);
        builder.arg(&cols_i);
        builder.arg(&eps);
        // SAFETY: FP16 in/out, FP32 weights/bias; shared mem for reductions.
        // SAFETY: kernel args set above; one block per row.
        unsafe { launch_kernel(builder, launch_per_row(rows, threads, shared)) }
    }

    fn fused_scale_mask_softmax_f16(
        &self,
        scores: &mut CudaTensor,
        mask: &CudaTensor,
        batch: usize,
        num_heads: usize,
        seq_len: usize,
        scale: f32,
    ) -> crate::Result<()> {
        let total_rows = batch * num_heads * seq_len;
        let threads = 256_u32;
        // Warp-shuffle reduction only needs (threads/32) floats for cross-warp.
        let shared = (threads / 32) * 4;

        let batch_i = batch as i32;
        let nh_i = num_heads as i32;
        let seq_i = seq_len as i32;

        let scores_f16 = scores
            .fp16
            .as_mut()
            .ok_or_else(|| crate::Error::Cuda("softmax_f16: scores has no FP16 buffer".into()))?;

        let mut builder = self
            .stream
            .launch_builder(&self.kernels.fused_scale_mask_softmax_f16);
        builder.arg(scores_f16);
        builder.arg(&mask.f32_buf); // Mask stays FP32.
        builder.arg(&batch_i);
        builder.arg(&nh_i);
        builder.arg(&seq_i);
        builder.arg(&scale);
        // SAFETY: FP16 scores with FP32 mask and accumulators.
        // SAFETY: kernel args set above; one block per row.
        unsafe { launch_kernel(builder, launch_per_row(total_rows, threads, shared)) }
    }

    fn fused_scale_mask_softmax_windowed_f16(
        &self,
        scores: &mut CudaTensor,
        mask: &CudaTensor,
        batch: usize,
        num_heads: usize,
        seq_len: usize,
        scale: f32,
        window_size: usize,
    ) -> crate::Result<()> {
        let total_rows = batch * num_heads * seq_len;
        let threads = 256_u32;
        let shared = (threads / 32) * 4;

        let batch_i = batch as i32;
        let nh_i = num_heads as i32;
        let seq_i = seq_len as i32;
        let win_i = window_size as i32;

        let scores_f16 = scores.fp16.as_mut().ok_or_else(|| {
            crate::Error::Cuda("softmax_windowed_f16: scores has no FP16 buffer".into())
        })?;

        let mut builder = self
            .stream
            .launch_builder(&self.kernels.fused_scale_mask_softmax_windowed_f16);
        builder.arg(scores_f16);
        builder.arg(&mask.f32_buf);
        builder.arg(&batch_i);
        builder.arg(&nh_i);
        builder.arg(&seq_i);
        builder.arg(&scale);
        builder.arg(&win_i);
        // SAFETY: FP16 scores with FP32 mask; windowed softmax.
        // SAFETY: kernel args set above; one block per row.
        unsafe { launch_kernel(builder, launch_per_row(total_rows, threads, shared)) }
    }

    fn qkv_split_f16(
        &self,
        q: &mut CudaTensor,
        k: &mut CudaTensor,
        v: &mut CudaTensor,
        qkv: &CudaTensor,
        batch: usize,
        seq: usize,
        hidden: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> crate::Result<()> {
        let total = batch * num_heads * seq * head_dim;
        let batch_i = batch as i32;
        let seq_i = seq as i32;
        let hidden_i = hidden as i32;
        let nh_i = num_heads as i32;
        let hd_i = head_dim as i32;

        let qkv_f16 = qkv
            .fp16_ref()
            .ok_or_else(|| crate::Error::Cuda("qkv_split_f16: qkv has no FP16 buffer".into()))?;
        let q_f16 = q
            .fp16
            .as_mut()
            .ok_or_else(|| crate::Error::Cuda("qkv_split_f16: q has no FP16 buffer".into()))?;
        let k_f16 = k
            .fp16
            .as_mut()
            .ok_or_else(|| crate::Error::Cuda("qkv_split_f16: k has no FP16 buffer".into()))?;
        let v_f16 = v
            .fp16
            .as_mut()
            .ok_or_else(|| crate::Error::Cuda("qkv_split_f16: v has no FP16 buffer".into()))?;

        let mut builder = self.stream.launch_builder(&self.kernels.qkv_split_f16);
        builder.arg(q_f16);
        builder.arg(k_f16);
        builder.arg(v_f16);
        builder.arg(qkv_f16);
        builder.arg(&batch_i);
        builder.arg(&seq_i);
        builder.arg(&hidden_i);
        builder.arg(&nh_i);
        builder.arg(&hd_i);
        // SAFETY: reads batch*seq*3*hidden FP16, writes total to each of q, k, v FP16.
        // SAFETY: kernel args set above; config matches element count.
        unsafe { launch_kernel(builder, launch_1d(total)) }
    }

    fn attn_reshape_f16(
        &self,
        output: &mut CudaTensor,
        input: &CudaTensor,
        batch: usize,
        seq: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> crate::Result<()> {
        let total = batch * seq * num_heads * head_dim;
        let batch_i = batch as i32;
        let seq_i = seq as i32;
        let nh_i = num_heads as i32;
        let hd_i = head_dim as i32;

        let in_f16 = input.fp16_ref().ok_or_else(|| {
            crate::Error::Cuda("attn_reshape_f16: input has no FP16 buffer".into())
        })?;
        let out_f16 = output.fp16.as_mut().ok_or_else(|| {
            crate::Error::Cuda("attn_reshape_f16: output has no FP16 buffer".into())
        })?;

        let mut builder = self.stream.launch_builder(&self.kernels.attn_reshape_f16);
        builder.arg(out_f16);
        builder.arg(in_f16);
        builder.arg(&batch_i);
        builder.arg(&seq_i);
        builder.arg(&nh_i);
        builder.arg(&hd_i);
        // SAFETY: FP16 head→linear layout transform.
        // SAFETY: kernel args set above; config matches element count.
        unsafe { launch_kernel(builder, launch_1d(total)) }
    }

    fn pad_to_batch_f16(
        &self,
        flat: &CudaTensor,
        padded: &mut CudaTensor,
        seq_lengths: &[usize],
        max_seq: usize,
        dim: usize,
    ) -> crate::Result<()> {
        let batch = seq_lengths.len();
        let total = batch * max_seq * dim;
        let cu_dev = self.build_cu_seqlens(seq_lengths)?;

        let max_seq_i = max_seq as i32;
        let dim_i = dim as i32;
        let batch_i = batch as i32;

        let flat_f16 = flat.fp16_ref().ok_or_else(|| {
            crate::Error::Cuda("pad_to_batch_f16: flat has no FP16 buffer".into())
        })?;
        let padded_f16 = padded.fp16.as_mut().ok_or_else(|| {
            crate::Error::Cuda("pad_to_batch_f16: padded has no FP16 buffer".into())
        })?;

        let mut builder = self.stream.launch_builder(&self.kernels.pad_to_batch_f16);
        builder.arg(padded_f16);
        builder.arg(flat_f16);
        builder.arg(&cu_dev);
        builder.arg(&max_seq_i);
        builder.arg(&dim_i);
        builder.arg(&batch_i);
        // SAFETY: FP16 scatter with cu_seqlens boundaries.
        // SAFETY: kernel args set above; config matches element count.
        unsafe { launch_kernel(builder, launch_1d(total)) }
    }

    fn unpad_from_batch_f16(
        &self,
        padded: &CudaTensor,
        flat: &mut CudaTensor,
        seq_lengths: &[usize],
        max_seq: usize,
        dim: usize,
    ) -> crate::Result<()> {
        let batch = seq_lengths.len();
        let total = batch * max_seq * dim;
        let cu_dev = self.build_cu_seqlens(seq_lengths)?;

        let max_seq_i = max_seq as i32;
        let dim_i = dim as i32;
        let batch_i = batch as i32;

        let padded_f16 = padded.fp16_ref().ok_or_else(|| {
            crate::Error::Cuda("unpad_from_batch_f16: padded has no FP16 buffer".into())
        })?;
        let flat_f16 = flat.fp16.as_mut().ok_or_else(|| {
            crate::Error::Cuda("unpad_from_batch_f16: flat has no FP16 buffer".into())
        })?;

        let mut builder = self
            .stream
            .launch_builder(&self.kernels.unpad_from_batch_f16);
        builder.arg(flat_f16);
        builder.arg(padded_f16);
        builder.arg(&cu_dev);
        builder.arg(&max_seq_i);
        builder.arg(&dim_i);
        builder.arg(&batch_i);
        // SAFETY: FP16 gather with cu_seqlens boundaries.
        // SAFETY: kernel args set above; config matches element count.
        unsafe { launch_kernel(builder, launch_1d(total)) }
    }

    #[expect(clippy::too_many_arguments, reason = "mirrors pad + qkv_split args")]
    fn fused_pad_qkv_split_f16(
        &self,
        q: &mut CudaTensor,
        k: &mut CudaTensor,
        v: &mut CudaTensor,
        qkv_flat: &CudaTensor,
        seq_lengths: &[usize],
        max_seq: usize,
        batch: usize,
        hidden: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> crate::Result<()> {
        let total = batch * num_heads * max_seq * head_dim;
        let cu_dev = self.build_cu_seqlens(seq_lengths)?;
        let batch_i = batch as i32;
        let max_seq_i = max_seq as i32;
        let hidden_i = hidden as i32;
        let nh_i = num_heads as i32;
        let hd_i = head_dim as i32;

        let qkv_f16 = qkv_flat
            .fp16_ref()
            .ok_or_else(|| crate::Error::Cuda("fused_pad_qkv_split_f16: qkv has no FP16".into()))?;
        let q_f16 = q
            .fp16
            .as_mut()
            .ok_or_else(|| crate::Error::Cuda("fused_pad_qkv_split_f16: q has no FP16".into()))?;
        let k_f16 = k
            .fp16
            .as_mut()
            .ok_or_else(|| crate::Error::Cuda("fused_pad_qkv_split_f16: k has no FP16".into()))?;
        let v_f16 = v
            .fp16
            .as_mut()
            .ok_or_else(|| crate::Error::Cuda("fused_pad_qkv_split_f16: v has no FP16".into()))?;

        let mut builder = self
            .stream
            .launch_builder(&self.kernels.fused_pad_qkv_split_f16);
        builder.arg(q_f16);
        builder.arg(k_f16);
        builder.arg(v_f16);
        builder.arg(qkv_f16);
        builder.arg(&cu_dev);
        builder.arg(&batch_i);
        builder.arg(&max_seq_i);
        builder.arg(&hidden_i);
        builder.arg(&nh_i);
        builder.arg(&hd_i);
        // SAFETY: reads flat QKV with cu_seqlens, writes to Q/K/V per-head layout.
        unsafe { launch_kernel(builder, launch_1d(total)) }
    }

    fn fused_reshape_unpad_f16(
        &self,
        flat: &mut CudaTensor,
        heads: &CudaTensor,
        seq_lengths: &[usize],
        max_seq: usize,
        batch: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> crate::Result<()> {
        let hidden = num_heads * head_dim;
        let total = batch * max_seq * hidden;
        let cu_dev = self.build_cu_seqlens(seq_lengths)?;
        let batch_i = batch as i32;
        let max_seq_i = max_seq as i32;
        let nh_i = num_heads as i32;
        let hd_i = head_dim as i32;

        let heads_f16 = heads.fp16_ref().ok_or_else(|| {
            crate::Error::Cuda("fused_reshape_unpad_f16: heads has no FP16".into())
        })?;
        let flat_f16 = flat.fp16.as_mut().ok_or_else(|| {
            crate::Error::Cuda("fused_reshape_unpad_f16: flat has no FP16".into())
        })?;

        let mut builder = self
            .stream
            .launch_builder(&self.kernels.fused_reshape_unpad_f16);
        builder.arg(flat_f16);
        builder.arg(heads_f16);
        builder.arg(&cu_dev);
        builder.arg(&batch_i);
        builder.arg(&max_seq_i);
        builder.arg(&nh_i);
        builder.arg(&hd_i);
        // SAFETY: reads per-head layout, writes to flat using cu_seqlens.
        unsafe { launch_kernel(builder, launch_1d(total)) }
    }

    fn rope_encode_f16(
        &self,
        qk: &mut CudaTensor,
        cos: &CudaTensor,
        sin: &CudaTensor,
        num_rows: usize,
        seq_len: usize,
        head_dim: usize,
        num_heads: usize,
    ) -> crate::Result<()> {
        let half = head_dim / 2;
        let total = num_rows * half;
        let nr_i = num_rows as i32;
        let seq_i = seq_len as i32;
        let hd_i = head_dim as i32;
        let nh_i = num_heads as i32;

        let qk_f16 = qk
            .fp16
            .as_mut()
            .ok_or_else(|| crate::Error::Cuda("rope_encode_f16: qk has no FP16 buffer".into()))?;

        let mut builder = self.stream.launch_builder(&self.kernels.rope_encode_f16);
        builder.arg(qk_f16);
        builder.arg(&cos.f32_buf); // cos/sin tables stay FP32.
        builder.arg(&sin.f32_buf);
        builder.arg(&nr_i);
        builder.arg(&seq_i);
        builder.arg(&hd_i);
        builder.arg(&nh_i);
        // SAFETY: FP16 Q/K with FP32 cos/sin tables.
        // SAFETY: kernel args set above; config matches element count.
        unsafe { launch_kernel(builder, launch_1d(total)) }
    }

    fn geglu_f16(
        &self,
        value: &CudaTensor,
        gate: &CudaTensor,
        output: &mut CudaTensor,
        n: usize,
    ) -> crate::Result<()> {
        let n_i = n as i32;

        let val_f16 = value
            .fp16_ref()
            .ok_or_else(|| crate::Error::Cuda("geglu_f16: value has no FP16 buffer".into()))?;
        let gate_f16 = gate
            .fp16_ref()
            .ok_or_else(|| crate::Error::Cuda("geglu_f16: gate has no FP16 buffer".into()))?;
        let out_f16 = output
            .fp16
            .as_mut()
            .ok_or_else(|| crate::Error::Cuda("geglu_f16: output has no FP16 buffer".into()))?;

        let mut builder = self.stream.launch_builder(&self.kernels.geglu_f16);
        builder.arg(out_f16);
        builder.arg(val_f16);
        builder.arg(gate_f16);
        builder.arg(&n_i);
        // SAFETY: reads n FP16 from value/gate, writes n FP16 to output.
        // SAFETY: kernel args set above; config matches element count.
        unsafe { launch_kernel(builder, launch_1d(n)) }
    }

    fn fused_split_geglu_f16(
        &self,
        output: &mut CudaTensor,
        input: &CudaTensor,
        rows: usize,
        cols: usize,
    ) -> crate::Result<()> {
        let n = rows * cols;
        let rows_i = rows as i32;
        let cols_i = cols as i32;

        let inp_f16 = input.fp16_ref().ok_or_else(|| {
            crate::Error::Cuda("fused_split_geglu_f16: input has no FP16 buffer".into())
        })?;
        let out_f16 = output.fp16.as_mut().ok_or_else(|| {
            crate::Error::Cuda("fused_split_geglu_f16: output has no FP16 buffer".into())
        })?;

        let mut builder = self
            .stream
            .launch_builder(&self.kernels.fused_split_geglu_f16);
        builder.arg(out_f16);
        builder.arg(inp_f16);
        builder.arg(&rows_i);
        builder.arg(&cols_i);
        // SAFETY: reads [rows, 2*cols] FP16 from input, writes [rows, cols] to output.
        unsafe { launch_kernel(builder, launch_1d(n)) }
    }

    fn fused_residual_layernorm_f16(
        &self,
        output: &mut CudaTensor,
        hidden: &CudaTensor,
        residual: &CudaTensor,
        weight: &CudaTensor,
        bias: &CudaTensor,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> crate::Result<()> {
        let threads = 256_u32;
        let shared = threads * 4;
        let rows_i = rows as i32;
        let cols_i = cols as i32;

        let hid_f16 = hidden.fp16_ref().ok_or_else(|| {
            crate::Error::Cuda("fused_residual_ln_f16: hidden has no FP16 buffer".into())
        })?;
        let res_f16 = residual.fp16_ref().ok_or_else(|| {
            crate::Error::Cuda("fused_residual_ln_f16: residual has no FP16 buffer".into())
        })?;
        let out_f16 = output.fp16.as_mut().ok_or_else(|| {
            crate::Error::Cuda("fused_residual_ln_f16: output has no FP16 buffer".into())
        })?;

        let mut builder = self
            .stream
            .launch_builder(&self.kernels.fused_residual_layernorm_f16);
        builder.arg(out_f16);
        builder.arg(hid_f16);
        builder.arg(res_f16);
        builder.arg(&weight.f32_buf); // Norm weights FP32.
        builder.arg(&bias.f32_buf); // Norm bias FP32.
        builder.arg(&rows_i);
        builder.arg(&cols_i);
        builder.arg(&eps);
        // SAFETY: FP16 hidden/residual, FP32 weight/bias; shared mem for reductions.
        // SAFETY: kernel args set above; one block per row.
        unsafe { launch_kernel(builder, launch_per_row(rows, threads, shared)) }
    }

    fn residual_add_f16(
        &self,
        output: &mut CudaTensor,
        hidden: &CudaTensor,
        residual: &CudaTensor,
        n: usize,
    ) -> crate::Result<()> {
        let n_i = n as i32;

        let hid_f16 = hidden.fp16_ref().ok_or_else(|| {
            crate::Error::Cuda("residual_add_f16: hidden has no FP16 buffer".into())
        })?;
        let res_f16 = residual.fp16_ref().ok_or_else(|| {
            crate::Error::Cuda("residual_add_f16: residual has no FP16 buffer".into())
        })?;
        let out_f16 = output.fp16.as_mut().ok_or_else(|| {
            crate::Error::Cuda("residual_add_f16: output has no FP16 buffer".into())
        })?;

        let mut builder = self.stream.launch_builder(&self.kernels.residual_add_f16);
        builder.arg(out_f16);
        builder.arg(hid_f16);
        builder.arg(res_f16);
        builder.arg(&n_i);
        // SAFETY: reads n FP16 from hidden/residual, writes n FP16 to output.
        // SAFETY: kernel args set above; config matches element count.
        unsafe { launch_kernel(builder, launch_1d(n)) }
    }

    fn split_gate_value_f16(
        &self,
        first: &mut CudaTensor,
        second: &mut CudaTensor,
        input: &CudaTensor,
        rows: usize,
        cols: usize,
    ) -> crate::Result<()> {
        let total = rows * cols;
        let rows_i = rows as i32;
        let cols_i = cols as i32;

        let in_f16 = input.fp16_ref().ok_or_else(|| {
            crate::Error::Cuda("split_gate_value_f16: input has no FP16 buffer".into())
        })?;
        let first_f16 = first.fp16.as_mut().ok_or_else(|| {
            crate::Error::Cuda("split_gate_value_f16: first has no FP16 buffer".into())
        })?;
        let second_f16 = second.fp16.as_mut().ok_or_else(|| {
            crate::Error::Cuda("split_gate_value_f16: second has no FP16 buffer".into())
        })?;

        let mut builder = self
            .stream
            .launch_builder(&self.kernels.split_gate_value_f16);
        builder.arg(first_f16);
        builder.arg(second_f16);
        builder.arg(in_f16);
        builder.arg(&rows_i);
        builder.arg(&cols_i);
        // SAFETY: reads rows*2*cols FP16, writes rows*cols to each half.
        // SAFETY: kernel args set above; config matches element count.
        unsafe { launch_kernel(builder, launch_1d(total)) }
    }
}

// ---------------------------------------------------------------------------
// ModernBertConfig
// ---------------------------------------------------------------------------

/// Parsed `ModernBERT` config from `config.json` for the CUDA backend.
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
        reason = "config floats (theta, eps) are small and fit in f32"
    )]
    pub fn from_json(json: &serde_json::Value) -> crate::Result<Self> {
        let get_usize = |key: &str| -> crate::Result<usize> {
            json.get(key)
                .and_then(serde_json::Value::as_u64)
                .map(|v| v as usize)
                .ok_or_else(|| crate::Error::Cuda(format!("config.json missing or invalid: {key}")))
        };
        let get_f64 = |key: &str| -> crate::Result<f64> {
            json.get(key)
                .and_then(serde_json::Value::as_f64)
                .ok_or_else(|| crate::Error::Cuda(format!("config.json missing or invalid: {key}")))
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
