//! CUDA embedding backend using cudarc (cuBLAS + custom NVRTC kernels).
//!
//! Implements BERT inference on NVIDIA GPUs using [`cudarc`] for device
//! management, cuBLAS for matrix multiplications, and runtime-compiled CUDA
//! kernels for activations, layer normalization, softmax, and embedding lookup.
//!
//! Supports the `ClassicBert` family (BGE models): learned position embeddings,
//! GELU activation, and QKV projections with bias.

use std::sync::Arc;

use crate::backend::nvrtc_cubin::compile_cubin;
use cudarc::cublas::{sys, CudaBlas};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, DevicePtrMut,
    LaunchConfig, PushKernelArg,
};
use hf_hub::api::sync::Api;
use safetensors::SafeTensors;

use super::{DeviceHint, EmbedBackend, Encoding};

/// Maximum batch size for pre-allocated workspace buffers.
///
/// Workspace is allocated once for `MAX_BATCH * max_seq_len`. Batches larger
/// than this are split into sub-batches automatically.
const MAX_BATCH: i32 = 128;

// ---------------------------------------------------------------------------
// Error helper
// ---------------------------------------------------------------------------

/// Convert any cudarc error into our crate error.
fn cuda_err(e: impl std::fmt::Display) -> crate::Error {
    crate::Error::Cuda(e.to_string())
}

// ---------------------------------------------------------------------------
// Architecture validation
// ---------------------------------------------------------------------------

/// Validate that the loaded weights are a recognized `ClassicBert` model.
///
/// `ClassicBert` has `embeddings.position_embeddings.weight`. Returns an
/// error if the architecture is not recognized.
fn detect_variant(tensors: &SafeTensors<'_>) -> crate::Result<()> {
    if tensors
        .tensor("embeddings.position_embeddings.weight")
        .is_ok()
    {
        Ok(())
    } else {
        Err(crate::Error::Other(anyhow::anyhow!(
            "unrecognized model architecture: no position_embeddings found"
        )))
    }
}

// ---------------------------------------------------------------------------
// BERT model configuration
// ---------------------------------------------------------------------------

/// Configuration for a BERT-style encoder model.
#[derive(Debug, Clone)]
struct BertConfig {
    /// Hidden dimension (e.g. 384 for bge-small).
    hidden_size: i32,
    /// Number of transformer layers.
    num_hidden_layers: i32,
    /// Number of attention heads.
    num_attention_heads: i32,
    /// Maximum sequence length (512 for ClassicBert).
    max_position_embeddings: i32,
    /// Layer norm epsilon.
    layer_norm_eps: f32,
}

impl BertConfig {
    /// Parse from a `config.json` value.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "config values are small ints/floats that fit in i32/f32"
    )]
    fn from_json(v: &serde_json::Value) -> crate::Result<Self> {
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

        Ok(Self {
            hidden_size: get_i32("hidden_size")?,
            num_hidden_layers: get_i32("num_hidden_layers")?,
            num_attention_heads: get_i32("num_attention_heads")?,
            max_position_embeddings: get_i32("max_position_embeddings").unwrap_or(512),
            layer_norm_eps,
        })
    }
}

// ---------------------------------------------------------------------------
// Safetensors -> GPU helpers
// ---------------------------------------------------------------------------

/// Load a named tensor from safetensors as `Vec<f32>` (host), returning shape.
fn load_tensor_host(
    tensors: &SafeTensors<'_>,
    name: &str,
) -> crate::Result<(Vec<f32>, Vec<usize>)> {
    let tensor = tensors
        .tensor(name)
        .map_err(|_| crate::Error::Other(anyhow::anyhow!("missing weight: {name}")))?;
    let shape = tensor.shape().to_vec();
    let data: Vec<f32> = tensor
        .data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    Ok((data, shape))
}

/// Load a tensor to GPU, returning the device slice and its shape.
fn load_to_gpu(
    stream: &Arc<CudaStream>,
    tensors: &SafeTensors<'_>,
    name: &str,
) -> crate::Result<(CudaSlice<f32>, Vec<usize>)> {
    let (data, shape) = load_tensor_host(tensors, name)?;
    let device_slice = stream.clone_htod(&data).map_err(cuda_err)?;
    Ok((device_slice, shape))
}

/// Optionally load a tensor to GPU -- returns `None` if missing.
fn try_load_to_gpu(
    stream: &Arc<CudaStream>,
    tensors: &SafeTensors<'_>,
    name: &str,
) -> crate::Result<Option<CudaSlice<f32>>> {
    if tensors.tensor(name).is_ok() {
        let (slice, _shape) = load_to_gpu(stream, tensors, name)?;
        Ok(Some(slice))
    } else {
        Ok(None)
    }
}

// ---------------------------------------------------------------------------
// CUDA kernel source (compiled at runtime via NVRTC)
// ---------------------------------------------------------------------------

/// CUDA kernels for BERT inference operations.
///
/// Compiled once at model load time via NVRTC, then reused for all forward
/// passes. Each kernel uses a simple 1D or per-row dispatch pattern.
const KERNELS: &str = r#"
extern "C" __global__ void gelu_kernel(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        x[i] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
    }
}

extern "C" __global__ void silu_kernel(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        x[i] = v / (1.0f + expf(-v));
    }
}

// SwiGLU: output[i] = value[i] * silu(gate[i])
// value = first half, gate = second half of input
extern "C" __global__ void swiglu_kernel(
    float* output, const float* input, int rows, int half_cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * half_cols;
    if (idx < total) {
        int row = idx / half_cols;
        int col = idx % half_cols;
        float value = input[row * (2 * half_cols) + col];
        float gate = input[row * (2 * half_cols) + half_cols + col];
        float silu_gate = gate / (1.0f + expf(-gate));
        output[idx] = value * silu_gate;
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

    // Phase 1: compute partial sums for mean
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        local_sum += input[row * cols + i];
    }
    s_sum[threadIdx.x] = local_sum;
    __syncthreads();

    // Reduce sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float mean = s_sum[0] / (float)cols;
    __syncthreads();

    // Phase 2: compute variance
    float local_sq = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float diff = input[row * cols + i] - mean;
        local_sq += diff * diff;
    }
    s_sq[threadIdx.x] = local_sq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sq[threadIdx.x] += s_sq[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float inv_std = rsqrtf(s_sq[0] / (float)cols + eps);
    __syncthreads();

    // Phase 3: normalize
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        int idx = row * cols + i;
        output[idx] = (input[idx] - mean) * inv_std * weight[i] + bias[i];
    }
}

extern "C" __global__ void softmax_kernel(float* x, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float sdata[];

    // Find max (for numerical stability)
    float local_max = -1e30f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = x[row * cols + i];
        if (v > local_max) local_max = v;
    }
    sdata[threadIdx.x] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    float max_val = sdata[0];
    __syncthreads();

    // Compute exp and sum
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = expf(x[row * cols + i] - max_val);
        x[row * cols + i] = v;
        local_sum += v;
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float sum = sdata[0];
    __syncthreads();

    // Normalize
    float inv_sum = 1.0f / sum;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        x[row * cols + i] *= inv_sum;
    }
}

extern "C" __global__ void add_bias_kernel(float* x, const float* bias, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        x[idx] += bias[idx % cols];
    }
}

extern "C" __global__ void residual_add_kernel(float* output, const float* residual, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] += residual[i];
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

// Build attention mask: 0.0 for real tokens (mask=1), -1e9 for padding (mask=0)
// Batched: mask is [batch * seq], output is [batch * seq]
extern "C" __global__ void build_attn_mask_kernel(
    float* output, const int* mask, int total
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        output[i] = (mask[i] == 1) ? 0.0f : -1e9f;
    }
}

// Add attention mask to scores: scores[b*nh*seq*seq] += mask[b*seq]
// Each batch element's heads share the same mask row.
extern "C" __global__ void add_attn_mask_kernel(
    float* scores, const float* mask,
    int batch, int num_heads, int seq
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * num_heads * seq * seq;
    if (idx < total) {
        int b = idx / (num_heads * seq * seq);
        int col = idx % seq;
        scores[idx] += mask[b * seq + col];
    }
}

// Scale all elements: x[i] *= scale
extern "C" __global__ void scale_kernel(float* x, float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= scale;
}

// CLS pooling: extract row 0 of each batch element from [batch, seq, hidden]
// output is [batch, hidden]
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

// L2 normalize each row of a [rows, cols] matrix in-place
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
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float inv_norm = rsqrtf(fmaxf(sdata[0], 1e-12f));
    __syncthreads();

    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        x[row * cols + i] *= inv_norm;
    }
}

// Apply RoPE to a [seq, head_dim] matrix (Q or K for a single head).
// mat is contiguous [seq * head_dim]. Output written in-place.
extern "C" __global__ void rope_kernel(
    float* mat, int seq, int head_dim, float base
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half = head_dim / 2;
    int total = seq * half;
    if (idx >= total) return;

    int pos = idx / half;
    int i = idx % half;

    double theta = pow((double)base, -2.0 * (double)i / (double)head_dim);
    double angle = (double)pos * theta;
    float cos_a = (float)cos(angle);
    float sin_a = (float)sin(angle);

    int first_idx = pos * head_dim + i;
    int second_idx = pos * head_dim + i + half;

    float first = mat[first_idx];
    float second = mat[second_idx];
    mat[first_idx] = first * cos_a - second * sin_a;
    mat[second_idx] = first * sin_a + second * cos_a;
}

// Split QKV [batch*seq, 3*hidden] into Q,K,V each [batch*num_heads, seq, head_dim] on GPU.
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

// Reshape attention output from [batch*num_heads, seq, head_dim] back to [batch*seq, hidden].
extern "C" __global__ void attn_reshape_kernel(
    float* output,
    const float* heads,
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

// Fused scale + attention-mask + softmax.
// Replaces three separate kernels (scale_kernel, add_attn_mask_kernel,
// softmax_kernel) with a single pass over the scores matrix.
// scores: [batch*num_heads*seq, seq] — modified in-place.
// mask:   [batch*seq] — 0.0 for real tokens, -1e9 for padding.
// One block per row; shared memory for reductions.
extern "C" __global__ void fused_scale_mask_softmax_kernel(
    float* scores, const float* mask,
    int batch, int num_heads, int seq, float scale
) {
    int row = blockIdx.x;
    int total_rows = batch * num_heads * seq;
    if (row >= total_rows) return;

    extern __shared__ float sdata[];
    float* row_data = scores + row * seq;

    // Decompose row → (b, head, row_in_seq) to index into mask
    int b = row / (num_heads * seq);

    // Pass 1: scale + mask + find row max (numerical stability)
    float thread_max = -1e30f;
    for (int i = threadIdx.x; i < seq; i += blockDim.x) {
        float val = row_data[i] * scale + mask[b * seq + i];
        row_data[i] = val;
        thread_max = fmaxf(thread_max, val);
    }

    // Reduce max
    sdata[threadIdx.x] = thread_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    float row_max = sdata[0];

    // Pass 2: exp(x - max) and sum
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < seq; i += blockDim.x) {
        float val = expf(row_data[i] - row_max);
        row_data[i] = val;
        thread_sum += val;
    }

    // Reduce sum
    sdata[threadIdx.x] = thread_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float total = sdata[0];

    // Pass 3: normalize
    float inv_sum = 1.0f / fmaxf(total, 1e-12f);
    for (int i = threadIdx.x; i < seq; i += blockDim.x) {
        row_data[i] *= inv_sum;
    }
}

// Fused bias + GELU activation (ClassicBert FFN).
// Replaces separate add_bias_kernel + gelu_kernel calls.
extern "C" __global__ void fused_bias_gelu_kernel(
    float* x, const float* bias, int rows, int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    int col = idx % cols;
    float v = x[idx] + bias[col];
    x[idx] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
}

// Fused bias + SwiGLU (ModernBERT FFN).
// input has shape [rows, 2*half_cols] with bias [2*half_cols].
// output[i] = value[i] * silu(gate[i]) where value/gate include bias.
extern "C" __global__ void fused_bias_swiglu_kernel(
    float* output, const float* input, const float* bias,
    int rows, int half_cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * half_cols;
    if (idx < total) {
        int row = idx / half_cols;
        int col = idx % half_cols;
        float value = input[row * (2 * half_cols) + col] + bias[col];
        float gate = input[row * (2 * half_cols) + half_cols + col] + bias[half_cols + col];
        float silu_gate = gate / (1.0f + expf(-gate));
        output[idx] = value * silu_gate;
    }
}

// RoPE with pre-computed cos/sin tables (ModernBERT).
// Replaces rope_kernel which recomputed theta/cos/sin on every call.
extern "C" __global__ void rope_cached_kernel(
    float* q_or_k,           // [num_rows, head_dim]
    const float* cos_table,  // [max_seq, half_dim]
    const float* sin_table,  // [max_seq, half_dim]
    int num_rows,            // total rows (num_heads * seq for the batch)
    int seq,                 // sequence length
    int head_dim,
    int num_heads
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half = head_dim / 2;
    int total = num_rows * half;
    if (idx >= total) return;

    int row = idx / half;
    int d = idx % half;
    int pos = (row % seq);

    int first_idx = row * head_dim + d;
    int second_idx = first_idx + half;

    float cos_val = cos_table[pos * half + d];
    float sin_val = sin_table[pos * half + d];

    float first = q_or_k[first_idx];
    float second = q_or_k[second_idx];
    q_or_k[first_idx] = first * cos_val - second * sin_val;
    q_or_k[second_idx] = first * sin_val + second * cos_val;
}

// Fused residual add + layer norm.
// Replaces sequential residual_add_kernel + layer_norm_kernel.
// output = layernorm(hidden + residual)
extern "C" __global__ void fused_residual_layernorm_kernel(
    float* output,
    const float* hidden,
    const float* residual,
    const float* weight,
    const float* bias,
    int rows, int cols, float eps
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    extern __shared__ float sdata[];

    // Pass 1: add residual + compute mean
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

    // Pass 2: compute variance
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

    // Pass 3: normalize + scale + shift
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float val = (output[row * cols + i] - mean) * inv_std;
        output[row * cols + i] = val * weight[i] + bias[i];
    }
}

// Unified SwiGLU kernel handling both bias and no-bias paths.
// Replaces both fused_bias_swiglu_kernel (when has_bias=1) and
// swiglu_kernel (when has_bias=0).
extern "C" __global__ void fused_swiglu_kernel(
    float* output,
    const float* input,
    const float* bias,       // may be NULL when has_bias=0
    int rows, int out_cols,
    int has_bias
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * out_cols) return;
    int row = idx / out_cols;
    int col = idx % out_cols;

    float value = input[row * 2 * out_cols + col];
    float gate = input[row * 2 * out_cols + out_cols + col];

    if (has_bias) {
        value += bias[col];
        gate += bias[out_cols + col];
    }

    gate = gate / (1.0f + expf(-gate));
    output[idx] = value * gate;
}

// Fused bias + residual add for output projections (ClassicBert).
// Replaces sequential add_bias_kernel + residual_add_kernel.
extern "C" __global__ void fused_bias_residual_kernel(
    float* output,
    const float* input,
    const float* bias,
    const float* residual,
    int rows, int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    output[idx] = input[idx] + bias[idx % cols] + residual[idx];
}

// Convert FP32 to FP16 using PTX inline asm (no cuda_fp16.h dependency).
// Output is unsigned short (u16) holding FP16 bit pattern.
extern "C" __global__ void f32_to_f16_kernel(
    unsigned short* output, const float* input, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        asm("cvt.rn.f16.f32 %0, %1;" : "=h"(output[i]) : "f"(input[i]));
    }
}
"#;

// ---------------------------------------------------------------------------
// Compiled kernel handles
// ---------------------------------------------------------------------------

/// Pre-compiled CUDA kernel function handles.
///
/// Created once at model load time, then used for all forward passes.
struct KernelHandles {
    /// GELU activation (in-place).
    gelu: CudaFunction,
    /// `SwiGLU` activation (value * silu(gate)). Superseded by `fused_swiglu`.
    #[expect(dead_code, reason = "kept for potential standalone use")]
    swiglu: CudaFunction,
    /// Layer normalization.
    layer_norm: CudaFunction,
    /// Softmax along last axis (per-row). Superseded by `fused_scale_mask_softmax`.
    #[expect(dead_code, reason = "kept for potential standalone use")]
    softmax: CudaFunction,
    /// Add bias vector to rows.
    add_bias: CudaFunction,
    /// Element-wise residual addition. Superseded by `fused_residual_layernorm` / `fused_bias_residual`.
    #[expect(dead_code, reason = "kept for potential standalone use")]
    residual_add: CudaFunction,
    /// Embedding table lookup.
    embedding_lookup: CudaFunction,
    /// Add embedding table values to existing output.
    add_embeddings: CudaFunction,
    /// Build float attention mask from int mask.
    build_attn_mask: CudaFunction,
    /// Add attention mask to score matrix. Superseded by `fused_scale_mask_softmax`.
    #[expect(dead_code, reason = "kept for potential standalone use")]
    add_attn_mask: CudaFunction,
    /// Scale all elements by a constant. Superseded by `fused_scale_mask_softmax`.
    #[expect(dead_code, reason = "kept for potential standalone use")]
    scale: CudaFunction,
    /// CLS pooling (extract row 0 per batch element).
    cls_pool: CudaFunction,
    /// L2 normalize each row.
    l2_normalize: CudaFunction,
    /// Rotary position embedding. Superseded by `rope_cached`.
    #[expect(dead_code, reason = "kept for potential standalone use")]
    rope: CudaFunction,
    /// Split QKV `[batch*seq, 3*hidden]` into Q,K,V `[batch*num_heads, seq, head_dim]` on GPU.
    qkv_split: CudaFunction,
    /// Reshape attention output `[batch*num_heads, seq, head_dim]` to `[batch*seq, hidden]`.
    attn_reshape: CudaFunction,
    /// Fused scale + attention-mask + softmax (replaces 3 separate kernels).
    fused_scale_mask_softmax: CudaFunction,
    /// Fused bias + GELU activation for `ClassicBert` FFN.
    fused_bias_gelu: CudaFunction,
    /// Fused bias + `SwiGLU` activation for `ModernBERT` FFN. Superseded by `fused_swiglu`.
    #[expect(dead_code, reason = "kept for potential standalone use")]
    fused_bias_swiglu: CudaFunction,
    /// `RoPE` with pre-computed cos/sin tables.
    rope_cached: CudaFunction,
    /// Fused residual add + layer norm (replaces separate `residual_add` + `layer_norm`).
    fused_residual_layernorm: CudaFunction,
    /// Unified `SwiGLU` kernel handling both bias and no-bias paths.
    fused_swiglu: CudaFunction,
    /// Fused bias + residual add for output projections.
    fused_bias_residual: CudaFunction,
    /// Convert FP32 to FP16 (for tensor core GEMM input conversion).
    f32_to_f16: CudaFunction,
}

impl KernelHandles {
    /// Compile CUDA kernels and load function handles.
    fn compile(ctx: &Arc<CudaContext>) -> crate::Result<(Arc<CudaModule>, Self)> {
        // Emit SASS directly for the live GPU's compute capability. See
        // `backend::nvrtc_cubin` for why we skip PTX. Runtime detection also
        // lets us run on CUDA 13+ NVRTCs that dropped the hardcoded sm_70
        // support this file used to rely on.
        use cudarc::driver::sys::CUdevice_attribute;
        let major = ctx
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .map_err(cuda_err)?;
        let minor = ctx
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .map_err(cuda_err)?;
        let arch = format!("sm_{major}{minor}");
        let cubin = compile_cubin(KERNELS, &arch).map_err(cuda_err)?;
        let module = ctx.load_module(cubin).map_err(cuda_err)?;
        let module = Arc::new(module);

        let load = |name: &str| -> crate::Result<CudaFunction> {
            module.load_function(name).map_err(cuda_err)
        };

        Ok((
            Arc::clone(&module),
            Self {
                gelu: load("gelu_kernel")?,
                swiglu: load("swiglu_kernel")?,
                layer_norm: load("layer_norm_kernel")?,
                softmax: load("softmax_kernel")?,
                add_bias: load("add_bias_kernel")?,
                residual_add: load("residual_add_kernel")?,
                embedding_lookup: load("embedding_lookup_kernel")?,
                add_embeddings: load("add_embeddings_kernel")?,
                build_attn_mask: load("build_attn_mask_kernel")?,
                add_attn_mask: load("add_attn_mask_kernel")?,
                scale: load("scale_kernel")?,
                cls_pool: load("cls_pool_kernel")?,
                l2_normalize: load("l2_normalize_kernel")?,
                rope: load("rope_kernel")?,
                qkv_split: load("qkv_split_kernel")?,
                attn_reshape: load("attn_reshape_kernel")?,
                fused_scale_mask_softmax: load("fused_scale_mask_softmax_kernel")?,
                fused_bias_gelu: load("fused_bias_gelu_kernel")?,
                fused_bias_swiglu: load("fused_bias_swiglu_kernel")?,
                rope_cached: load("rope_cached_kernel")?,
                fused_residual_layernorm: load("fused_residual_layernorm_kernel")?,
                fused_swiglu: load("fused_swiglu_kernel")?,
                fused_bias_residual: load("fused_bias_residual_kernel")?,
                f32_to_f16: load("f32_to_f16_kernel")?,
            },
        ))
    }
}

// ---------------------------------------------------------------------------
// GPU model structures
// ---------------------------------------------------------------------------

/// BERT embeddings layer on GPU.
struct CudaBertEmbeddings {
    /// Word embedding table `[vocab_size, hidden]`.
    word_embeddings: CudaSlice<f32>,
    /// Learned position embeddings (`ClassicBert` only) `[max_seq, hidden]`.
    position_embeddings: Option<CudaSlice<f32>>,
    /// Token type embeddings (`ClassicBert` only) `[2, hidden]`.
    token_type_embeddings: Option<CudaSlice<f32>>,
    /// Layer norm weight `[hidden]`.
    layer_norm_weight: CudaSlice<f32>,
    /// Layer norm bias `[hidden]`.
    layer_norm_bias: CudaSlice<f32>,
    /// Layer norm epsilon.
    layer_norm_eps: f32,
}

/// Self-attention sub-layer on GPU.
struct CudaBertSelfAttention {
    /// Fused Q/K/V weight matrix `[3*hidden, hidden]` stored as FP16.
    qkv_weight: CudaSlice<u16>,
    /// Fused Q/K/V bias `[3*hidden]` (`ClassicBert` only).
    qkv_bias: Option<CudaSlice<f32>>,
    /// Output projection weight `[hidden, hidden]` stored as FP16.
    output_weight: CudaSlice<u16>,
    /// Output projection bias `[hidden]` (`ClassicBert` only).
    output_bias: Option<CudaSlice<f32>>,
    /// Post-attention `LayerNorm` weight `[hidden]`.
    output_ln_weight: CudaSlice<f32>,
    /// Post-attention `LayerNorm` bias `[hidden]`.
    output_ln_bias: CudaSlice<f32>,
    /// Layer norm epsilon.
    layer_norm_eps: f32,
}

/// Feed-forward network sub-layer on GPU.
struct CudaBertFfn {
    /// Intermediate projection weight `[intermediate, hidden]` stored as FP16.
    intermediate_weight: CudaSlice<u16>,
    /// Intermediate projection bias `[intermediate]`.
    intermediate_bias: Option<CudaSlice<f32>>,
    /// Output projection weight `[hidden, intermediate]` stored as FP16.
    output_weight: CudaSlice<u16>,
    /// Output projection bias `[hidden]`.
    output_bias: Option<CudaSlice<f32>>,
    /// Post-FFN `LayerNorm` weight `[hidden]`.
    output_ln_weight: CudaSlice<f32>,
    /// Post-FFN `LayerNorm` bias `[hidden]`.
    output_ln_bias: CudaSlice<f32>,
    /// Layer norm epsilon.
    layer_norm_eps: f32,
    /// Intermediate dimension.
    intermediate_dim: i32,
}

/// A single BERT encoder layer on GPU.
struct CudaBertLayer {
    /// Self-attention sub-layer.
    attention: CudaBertSelfAttention,
    /// Feed-forward sub-layer.
    ffn: CudaBertFfn,
}

/// Pre-allocated GPU workspace buffers to eliminate per-forward `cudaMalloc` calls.
///
/// Sized for `MAX_BATCH * max_seq_len` at model load time. Reused across
/// all forward passes -- contents are overwritten, never freed until model drop.
struct CudaWorkspace {
    /// Fused QKV output `[batch*seq, 3*hidden]`.
    qkv: CudaSlice<f32>,
    /// Q heads `[batch*num_heads, seq, head_dim]`.
    q: CudaSlice<f32>,
    /// K heads `[batch*num_heads, seq, head_dim]`.
    k: CudaSlice<f32>,
    /// V heads `[batch*num_heads, seq, head_dim]`.
    v: CudaSlice<f32>,
    /// Attention scores `[batch*num_heads, seq, seq]`.
    scores: CudaSlice<f32>,
    /// Attention output `[batch*num_heads, seq, head_dim]`.
    attn_out: CudaSlice<f32>,
    /// Context after reshape `[batch*seq, hidden]`.
    context: CudaSlice<f32>,
    /// FFN intermediate `[batch*seq, max(intermediate, 2*intermediate)]`.
    ffn_inter: CudaSlice<f32>,
    /// Ping-pong hidden state A `[batch*seq, hidden]`.
    hidden_a: CudaSlice<f32>,
    /// Ping-pong hidden state B `[batch*seq, hidden]`.
    hidden_b: CudaSlice<f32>,
    /// Projected output after attention `[batch*seq, hidden]`.
    projected: CudaSlice<f32>,
    /// `SwiGLU` activated output `[batch*seq, intermediate]`.
    activated: CudaSlice<f32>,
    /// Scratch buffer `[batch*seq, hidden]`.
    scratch: CudaSlice<f32>,
    /// CLS pooled + L2 normalized output `[batch, hidden]`.
    cls: CudaSlice<f32>,
    /// FP16 buffer for GEMM input conversion `[batch*seq, max(3*hidden, 2*intermediate)]`.
    input_f16: CudaSlice<u16>,
    /// FP16 Q heads for batched attention GEMM `[batch*num_heads, seq, head_dim]`.
    q_f16: CudaSlice<u16>,
    /// FP16 K heads for batched attention GEMM `[batch*num_heads, seq, head_dim]`.
    k_f16: CudaSlice<u16>,
    /// FP16 V heads for batched attention GEMM `[batch*num_heads, seq, head_dim]`.
    v_f16: CudaSlice<u16>,
    /// FP16 attention scores for batched attention GEMM `[batch*num_heads, seq, seq]`.
    scores_f16: CudaSlice<u16>,
}

/// Complete BERT model with all weights on GPU.
struct CudaBertModel {
    /// CUDA stream for all operations.
    stream: Arc<CudaStream>,
    /// cuBLAS handle for matrix operations.
    blas: CudaBlas,
    /// Pre-compiled kernel handles.
    kernels: KernelHandles,
    /// Embeddings layer.
    embeddings: CudaBertEmbeddings,
    /// Transformer encoder layers.
    layers: Vec<CudaBertLayer>,
    /// Hidden dimension.
    hidden_size: i32,
    /// Number of attention heads.
    num_heads: i32,
    /// Dimension per head.
    head_dim: i32,
    /// Pre-allocated workspace buffers.
    workspace: CudaWorkspace,
}

// ---------------------------------------------------------------------------
// FP16 conversion helper
// ---------------------------------------------------------------------------

/// Convert FP32 device buffer to FP16 (u16) in a pre-allocated workspace buffer.
///
/// Launches the `f32_to_f16_kernel` to convert `n` elements from `input` (f32)
/// into `output_f16` (u16 holding FP16 bit patterns).
#[expect(unsafe_code, reason = "CUDA kernel launch requires unsafe")]
fn convert_to_f16_inplace(
    stream: &Arc<CudaStream>,
    kernels: &KernelHandles,
    input: &CudaSlice<f32>,
    output_f16: &mut CudaSlice<u16>,
    n: i32,
) -> crate::Result<()> {
    let mut builder = stream.launch_builder(&kernels.f32_to_f16);
    builder.arg(output_f16);
    builder.arg(input);
    builder.arg(&n);
    // SAFETY: kernel reads `n` f32 elements from `input` and writes `n` u16
    // elements to `output_f16`. Both buffers are pre-allocated with sufficient size.
    unsafe { builder.launch(launch_cfg_1d(n)) }.map_err(cuda_err)?;
    Ok(())
}

/// Convert FP32 device buffer to FP16 (u16), allocating a new buffer.
///
/// Used at model load time to convert weight matrices to FP16 storage.
#[expect(
    clippy::cast_sign_loss,
    reason = "n is a positive dimension from model config"
)]
fn convert_to_f16(
    stream: &Arc<CudaStream>,
    kernels: &KernelHandles,
    input: &CudaSlice<f32>,
    n: i32,
) -> crate::Result<CudaSlice<u16>> {
    let mut f16_data = stream.alloc_zeros::<u16>(n as usize).map_err(cuda_err)?;
    convert_to_f16_inplace(stream, kernels, input, &mut f16_data, n)?;
    Ok(f16_data)
}

// ---------------------------------------------------------------------------
// cuBLAS GEMM helpers (FP16 tensor core)
// ---------------------------------------------------------------------------

/// FP16 tensor core linear: C(f32) = A(f16) @ W(f16)^T
///
/// Input `a` is `[m, k]` row-major (FP32, converted to FP16 via workspace).
/// `weight_f16` is `[n, k]` row-major stored as FP16.
/// Output `[m, n]` is FP32 (for downstream layernorm/softmax precision).
///
/// Uses `cublasGemmEx` with FP16 inputs, FP32 compute, FP32 output to
/// engage tensor cores (330 TFLOPS on RTX 4090 vs 83 TFLOPS FP32).
#[expect(
    unsafe_code,
    clippy::too_many_arguments,
    reason = "cublasGemmEx requires unsafe raw pointer access; args mirror cuBLAS API"
)]
fn gpu_linear_f16(
    blas: &CudaBlas,
    stream: &Arc<CudaStream>,
    kernels: &KernelHandles,
    input_f32: &CudaSlice<f32>,
    weight_f16: &CudaSlice<u16>,
    output: &mut CudaSlice<f32>,
    input_f16_buf: &mut CudaSlice<u16>,
    m: i32,
    n: i32,
    k: i32,
) -> crate::Result<()> {
    // Convert FP32 activation to FP16 in pre-allocated workspace
    convert_to_f16_inplace(stream, kernels, input_f32, input_f16_buf, m * k)?;

    let alpha = 1.0_f32;
    let beta = 0.0_f32;
    let handle = *blas.handle();

    // Obtain raw device pointers via the DevicePtr/DevicePtrMut traits.
    // The _sync guards must live until after the GEMM is scheduled.
    let (w_ptr, _w_sync) = weight_f16.device_ptr(stream);
    let (a_ptr, _a_sync) = input_f16_buf.device_ptr(stream);
    let (c_ptr, _c_sync) = output.device_ptr_mut(stream);

    // SAFETY: All device pointers come from valid CudaSlice allocations sized
    // for the GEMM dimensions. cublasGemmEx reads FP16 A/B, writes FP32 C
    // with FP32 accumulation. The handle is valid for the lifetime of `blas`.
    unsafe {
        // cuBLAS column-major: C(n,m) = W(n,k) @ A(k,m)
        // For row-major A[m,k] @ W[n,k]^T = C[m,n]:
        //   transa=T on weight, transb=N on input
        sys::cublasGemmEx(
            handle,
            sys::cublasOperation_t::CUBLAS_OP_T,
            sys::cublasOperation_t::CUBLAS_OP_N,
            n,
            m,
            k,
            std::ptr::from_ref(&alpha).cast(),
            w_ptr as *const _,
            sys::cudaDataType_t::CUDA_R_16F,
            k,
            a_ptr as *const _,
            sys::cudaDataType_t::CUDA_R_16F,
            k,
            std::ptr::from_ref(&beta).cast(),
            c_ptr as *mut _,
            sys::cudaDataType_t::CUDA_R_32F,
            n,
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        )
        .result()
        .map_err(cuda_err)?;
    }

    Ok(())
}

/// Batched FP16 tensor core GEMM for attention scores: C(f32) = Q(f16) @ K(f16)^T.
///
/// `q_f16` has shape `[batch_heads, seq, head_dim]` as FP16.
/// `k_f16` has shape `[batch_heads, seq, head_dim]` as FP16.
/// `scores` (output) has shape `[batch_heads, seq, seq]` as FP32.
#[expect(
    unsafe_code,
    clippy::too_many_arguments,
    reason = "cublasGemmStridedBatchedEx requires unsafe; args mirror cuBLAS API"
)]
fn gpu_batched_attn_scores_f16(
    blas: &CudaBlas,
    stream: &Arc<CudaStream>,
    q_f16: &CudaSlice<u16>,
    k_f16: &CudaSlice<u16>,
    scores: &mut CudaSlice<f32>,
    batch_heads: i32,
    seq: i32,
    head_dim: i32,
) -> crate::Result<()> {
    let alpha = 1.0_f32;
    let beta = 0.0_f32;
    let handle = *blas.handle();

    let (k_ptr, _k_sync) = k_f16.device_ptr(stream);
    let (q_ptr, _q_sync) = q_f16.device_ptr(stream);
    let (c_ptr, _c_sync) = scores.device_ptr_mut(stream);

    // SAFETY: All device pointers are valid CudaSlice allocations. Strides and
    // dimensions match the [batch_heads, seq, head_dim] layout. FP16 inputs
    // with FP32 accumulation and FP32 output.
    unsafe {
        sys::cublasGemmStridedBatchedEx(
            handle,
            sys::cublasOperation_t::CUBLAS_OP_T,
            sys::cublasOperation_t::CUBLAS_OP_N,
            seq,
            seq,
            head_dim,
            std::ptr::from_ref(&alpha).cast(),
            k_ptr as *const _,
            sys::cudaDataType_t::CUDA_R_16F,
            head_dim,
            i64::from(seq * head_dim),
            q_ptr as *const _,
            sys::cudaDataType_t::CUDA_R_16F,
            head_dim,
            i64::from(seq * head_dim),
            std::ptr::from_ref(&beta).cast(),
            c_ptr as *mut _,
            sys::cudaDataType_t::CUDA_R_32F,
            seq,
            i64::from(seq * seq),
            batch_heads,
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        )
        .result()
        .map_err(cuda_err)?;
    }

    Ok(())
}

/// Batched FP16 tensor core GEMM for attention output: C(f32) = scores(f16) @ V(f16).
///
/// `scores_f16` has shape `[batch_heads, seq, seq]` as FP16.
/// `v_f16` has shape `[batch_heads, seq, head_dim]` as FP16.
/// `output` has shape `[batch_heads, seq, head_dim]` as FP32.
#[expect(
    unsafe_code,
    clippy::too_many_arguments,
    reason = "cublasGemmStridedBatchedEx requires unsafe; args mirror cuBLAS API"
)]
fn gpu_batched_attn_output_f16(
    blas: &CudaBlas,
    stream: &Arc<CudaStream>,
    scores_f16: &CudaSlice<u16>,
    v_f16: &CudaSlice<u16>,
    output: &mut CudaSlice<f32>,
    batch_heads: i32,
    seq: i32,
    head_dim: i32,
) -> crate::Result<()> {
    let alpha = 1.0_f32;
    let beta = 0.0_f32;
    let handle = *blas.handle();

    let (v_ptr, _v_sync) = v_f16.device_ptr(stream);
    let (s_ptr, _s_sync) = scores_f16.device_ptr(stream);
    let (c_ptr, _c_sync) = output.device_ptr_mut(stream);

    // SAFETY: All device pointers are valid CudaSlice allocations. Strides and
    // dimensions match the batched layout. FP16 inputs with FP32 accumulation.
    unsafe {
        sys::cublasGemmStridedBatchedEx(
            handle,
            sys::cublasOperation_t::CUBLAS_OP_N,
            sys::cublasOperation_t::CUBLAS_OP_N,
            head_dim,
            seq,
            seq,
            std::ptr::from_ref(&alpha).cast(),
            v_ptr as *const _,
            sys::cudaDataType_t::CUDA_R_16F,
            head_dim,
            i64::from(seq * head_dim),
            s_ptr as *const _,
            sys::cudaDataType_t::CUDA_R_16F,
            seq,
            i64::from(seq * seq),
            std::ptr::from_ref(&beta).cast(),
            c_ptr as *mut _,
            sys::cudaDataType_t::CUDA_R_32F,
            head_dim,
            i64::from(seq * head_dim),
            batch_heads,
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        )
        .result()
        .map_err(cuda_err)?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Kernel launch helpers
// ---------------------------------------------------------------------------

/// Standard 1D launch config for `n` elements with 256 threads per block.
fn launch_cfg_1d(n: i32) -> LaunchConfig {
    let threads = 256_u32;
    let blocks = n.cast_unsigned().div_ceil(threads);
    LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Per-row launch with shared memory.
fn launch_cfg_per_row_shared(rows: i32, threads: u32, shared_bytes: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (rows.cast_unsigned(), 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: shared_bytes,
    }
}

// ---------------------------------------------------------------------------
// Forward pass implementation
// ---------------------------------------------------------------------------

impl CudaBertModel {
    /// Batched forward pass: process all encodings in a single GPU pass.
    ///
    /// Pads all encodings to `max_seq_len` in the batch, transfers padded
    /// tensors to GPU in one `clone_htod` per tensor, runs the full BERT
    /// forward pass with batch dimension, then extracts CLS embeddings
    /// for each batch element.
    ///
    /// Uses a ping-pong scheme between `hidden_a` and `hidden_b`:
    /// - Embeddings write to `hidden_a`.
    /// - Attention reads `hidden_a`, writes to `hidden_b`.
    /// - FFN reads `hidden_b`, writes back to `hidden_a`.
    /// - Next layer repeats.
    #[expect(
        unsafe_code,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        clippy::too_many_lines,
        reason = "monolithic GPU forward pass requires unsafe kernel launches and integer casts"
    )]
    fn forward_batch(&mut self, encodings: &[Encoding]) -> crate::Result<Vec<Vec<f32>>> {
        let batch = encodings.len() as i32;
        let hd = self.hidden_size;
        let nh = self.num_heads;
        let head_dim = self.head_dim;

        // Find max sequence length in this batch
        let max_seq = encodings
            .iter()
            .map(|e| e.input_ids.len())
            .max()
            .unwrap_or(0) as i32;
        let batch_seq = batch * max_seq;
        let batch_heads = batch * nh;

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

        // ONE clone_htod per tensor
        let input_ids_dev = self.stream.clone_htod(&input_ids).map_err(cuda_err)?;
        let token_type_ids_dev = self.stream.clone_htod(&token_type_ids).map_err(cuda_err)?;
        let position_ids_dev = self.stream.clone_htod(&position_ids).map_err(cuda_err)?;
        let attn_mask_int_dev = self.stream.clone_htod(&attn_mask_int).map_err(cuda_err)?;

        // Build float attention mask [batch * max_seq]: 0.0 for real, -1e9 for padding
        let mask_total = batch * max_seq;
        let mut attn_mask_dev = self
            .stream
            .alloc_zeros::<f32>(mask_total as usize)
            .map_err(cuda_err)?;
        {
            let mut builder = self.stream.launch_builder(&self.kernels.build_attn_mask);
            builder.arg(&mut attn_mask_dev);
            builder.arg(&attn_mask_int_dev);
            builder.arg(&mask_total);
            unsafe { builder.launch(launch_cfg_1d(mask_total)) }.map_err(cuda_err)?;
        }

        // ---------------------------------------------------------------
        // Embeddings: [batch*max_seq, hidden] → hidden_a
        // ---------------------------------------------------------------
        let n = batch_seq * hd;
        // Word embedding lookup → scratch
        {
            let mut builder = self.stream.launch_builder(&self.kernels.embedding_lookup);
            builder.arg(&mut self.workspace.scratch);
            builder.arg(&self.embeddings.word_embeddings);
            builder.arg(&input_ids_dev);
            builder.arg(&batch_seq);
            builder.arg(&hd);
            unsafe { builder.launch(launch_cfg_1d(n)) }.map_err(cuda_err)?;
        }
        if let Some(ref pos_emb) = self.embeddings.position_embeddings {
            let mut builder = self.stream.launch_builder(&self.kernels.add_embeddings);
            builder.arg(&mut self.workspace.scratch);
            builder.arg(pos_emb);
            builder.arg(&position_ids_dev);
            builder.arg(&batch_seq);
            builder.arg(&hd);
            unsafe { builder.launch(launch_cfg_1d(n)) }.map_err(cuda_err)?;
        }
        if let Some(ref tok_emb) = self.embeddings.token_type_embeddings {
            let mut builder = self.stream.launch_builder(&self.kernels.add_embeddings);
            builder.arg(&mut self.workspace.scratch);
            builder.arg(tok_emb);
            builder.arg(&token_type_ids_dev);
            builder.arg(&batch_seq);
            builder.arg(&hd);
            unsafe { builder.launch(launch_cfg_1d(n)) }.map_err(cuda_err)?;
        }
        // Embedding layer norm: scratch → hidden_a
        {
            let eps = self.embeddings.layer_norm_eps;
            let threads = 256_u32.min(hd as u32).next_power_of_two();
            let shared = threads * 2 * 4;
            let mut builder = self.stream.launch_builder(&self.kernels.layer_norm);
            builder.arg(&mut self.workspace.hidden_a);
            builder.arg(&self.workspace.scratch);
            builder.arg(&self.embeddings.layer_norm_weight);
            builder.arg(&self.embeddings.layer_norm_bias);
            builder.arg(&batch_seq);
            builder.arg(&hd);
            builder.arg(&eps);
            unsafe { builder.launch(launch_cfg_per_row_shared(batch_seq, threads, shared)) }
                .map_err(cuda_err)?;
        }

        // ---------------------------------------------------------------
        // Transformer layers: ping-pong hidden_a ↔ hidden_b
        //
        // Attention: reads hidden_a → writes hidden_b
        // FFN:       reads hidden_b → writes hidden_a
        // ---------------------------------------------------------------
        let num_layers = self.layers.len();
        for layer_idx in 0..num_layers {
            // === ATTENTION: hidden_a → hidden_b ===

            // QKV projection: [batch*seq, hidden] @ [3*hidden, hidden]^T (FP16 tensor core)
            gpu_linear_f16(
                &self.blas,
                &self.stream,
                &self.kernels,
                &self.workspace.hidden_a,
                &self.layers[layer_idx].attention.qkv_weight,
                &mut self.workspace.qkv,
                &mut self.workspace.input_f16,
                batch_seq,
                3 * hd,
                hd,
            )?;

            if let Some(ref bias) = self.layers[layer_idx].attention.qkv_bias {
                let total_qkv = batch_seq * 3 * hd;
                let cols = 3 * hd;
                let mut builder = self.stream.launch_builder(&self.kernels.add_bias);
                builder.arg(&mut self.workspace.qkv);
                builder.arg(bias);
                builder.arg(&batch_seq);
                builder.arg(&cols);
                unsafe { builder.launch(launch_cfg_1d(total_qkv)) }.map_err(cuda_err)?;
            }

            // Split QKV → Q, K, V each [batch*nh, seq, head_dim]
            let total_head_elems = batch_heads * max_seq * head_dim;
            {
                let mut builder = self.stream.launch_builder(&self.kernels.qkv_split);
                builder.arg(&mut self.workspace.q);
                builder.arg(&mut self.workspace.k);
                builder.arg(&mut self.workspace.v);
                builder.arg(&self.workspace.qkv);
                builder.arg(&batch);
                builder.arg(&max_seq);
                builder.arg(&hd);
                builder.arg(&nh);
                builder.arg(&head_dim);
                unsafe { builder.launch(launch_cfg_1d(total_head_elems)) }.map_err(cuda_err)?;
            }

            // Convert Q, K, V to FP16 for tensor core batched GEMM
            {
                let head_elems = batch_heads * max_seq * head_dim;
                convert_to_f16_inplace(
                    &self.stream,
                    &self.kernels,
                    &self.workspace.q,
                    &mut self.workspace.q_f16,
                    head_elems,
                )?;
                convert_to_f16_inplace(
                    &self.stream,
                    &self.kernels,
                    &self.workspace.k,
                    &mut self.workspace.k_f16,
                    head_elems,
                )?;
                convert_to_f16_inplace(
                    &self.stream,
                    &self.kernels,
                    &self.workspace.v,
                    &mut self.workspace.v_f16,
                    head_elems,
                )?;
            }

            // Attention scores: [batch*nh, seq, seq] (FP16 tensor core)
            gpu_batched_attn_scores_f16(
                &self.blas,
                &self.stream,
                &self.workspace.q_f16,
                &self.workspace.k_f16,
                &mut self.workspace.scores,
                batch_heads,
                max_seq,
                head_dim,
            )?;

            // Fused scale + attention-mask + softmax (one kernel, one memory pass)
            {
                let scale = 1.0_f32 / (head_dim as f32).sqrt();
                let total_rows = batch_heads * max_seq;
                let threads = 256_u32.min(max_seq as u32).next_power_of_two();
                let shared = threads * 4; // one float per thread for reductions
                let mut builder = self
                    .stream
                    .launch_builder(&self.kernels.fused_scale_mask_softmax);
                builder.arg(&mut self.workspace.scores);
                builder.arg(&attn_mask_dev);
                builder.arg(&batch);
                builder.arg(&nh);
                builder.arg(&max_seq);
                builder.arg(&scale);
                unsafe { builder.launch(launch_cfg_per_row_shared(total_rows, threads, shared)) }
                    .map_err(cuda_err)?;
            }

            // Convert softmax scores to FP16 for tensor core GEMM
            convert_to_f16_inplace(
                &self.stream,
                &self.kernels,
                &self.workspace.scores,
                &mut self.workspace.scores_f16,
                batch_heads * max_seq * max_seq,
            )?;

            // Attention output: [batch*nh, seq, head_dim] (FP16 tensor core)
            gpu_batched_attn_output_f16(
                &self.blas,
                &self.stream,
                &self.workspace.scores_f16,
                &self.workspace.v_f16,
                &mut self.workspace.attn_out,
                batch_heads,
                max_seq,
                head_dim,
            )?;

            // Reshape → [batch*seq, hidden]
            {
                let total_ctx = batch_seq * hd;
                let mut builder = self.stream.launch_builder(&self.kernels.attn_reshape);
                builder.arg(&mut self.workspace.context);
                builder.arg(&self.workspace.attn_out);
                builder.arg(&batch);
                builder.arg(&max_seq);
                builder.arg(&nh);
                builder.arg(&head_dim);
                unsafe { builder.launch(launch_cfg_1d(total_ctx)) }.map_err(cuda_err)?;
            }

            // Output projection → projected (FP16 tensor core)
            gpu_linear_f16(
                &self.blas,
                &self.stream,
                &self.kernels,
                &self.workspace.context,
                &self.layers[layer_idx].attention.output_weight,
                &mut self.workspace.projected,
                &mut self.workspace.input_f16,
                batch_seq,
                hd,
                hd,
            )?;

            // Fused bias+residual or just residual, then fused residual+layernorm
            if let Some(ref bias) = self.layers[layer_idx].attention.output_bias {
                // Fused bias + residual: scratch = projected + bias + hidden_a
                let total_proj = batch_seq * hd;
                let mut builder = self
                    .stream
                    .launch_builder(&self.kernels.fused_bias_residual);
                builder.arg(&mut self.workspace.scratch);
                builder.arg(&self.workspace.projected);
                builder.arg(bias);
                builder.arg(&self.workspace.hidden_a);
                builder.arg(&batch_seq);
                builder.arg(&hd);
                unsafe { builder.launch(launch_cfg_1d(total_proj)) }.map_err(cuda_err)?;

                // Layer norm: scratch → hidden_b
                let eps = self.layers[layer_idx].attention.layer_norm_eps;
                let threads = 256_u32.min(hd as u32).next_power_of_two();
                let shared = threads * 2 * 4;
                let mut builder = self.stream.launch_builder(&self.kernels.layer_norm);
                builder.arg(&mut self.workspace.hidden_b);
                builder.arg(&self.workspace.scratch);
                builder.arg(&self.layers[layer_idx].attention.output_ln_weight);
                builder.arg(&self.layers[layer_idx].attention.output_ln_bias);
                builder.arg(&batch_seq);
                builder.arg(&hd);
                builder.arg(&eps);
                unsafe { builder.launch(launch_cfg_per_row_shared(batch_seq, threads, shared)) }
                    .map_err(cuda_err)?;
            } else {
                // No bias: fused residual + layernorm in one kernel
                // hidden_b = layernorm(projected + hidden_a)
                let eps = self.layers[layer_idx].attention.layer_norm_eps;
                let threads = 256_u32.min(hd as u32).next_power_of_two();
                let shared = threads * 4;
                let mut builder = self
                    .stream
                    .launch_builder(&self.kernels.fused_residual_layernorm);
                builder.arg(&mut self.workspace.hidden_b);
                builder.arg(&self.workspace.projected);
                builder.arg(&self.workspace.hidden_a);
                builder.arg(&self.layers[layer_idx].attention.output_ln_weight);
                builder.arg(&self.layers[layer_idx].attention.output_ln_bias);
                builder.arg(&batch_seq);
                builder.arg(&hd);
                builder.arg(&eps);
                unsafe { builder.launch(launch_cfg_per_row_shared(batch_seq, threads, shared)) }
                    .map_err(cuda_err)?;
            }

            // === FFN: hidden_b → hidden_a ===

            let inter_dim = self.layers[layer_idx].ffn.intermediate_dim;

            // Intermediate projection (FP16 tensor core)
            gpu_linear_f16(
                &self.blas,
                &self.stream,
                &self.kernels,
                &self.workspace.hidden_b,
                &self.layers[layer_idx].ffn.intermediate_weight,
                &mut self.workspace.ffn_inter,
                &mut self.workspace.input_f16,
                batch_seq,
                inter_dim,
                hd,
            )?;

            // Fused bias + GELU activation, then output projection → scratch
            {
                let total_act = batch_seq * inter_dim;
                if let Some(ref bias) = self.layers[layer_idx].ffn.intermediate_bias {
                    let mut builder = self.stream.launch_builder(&self.kernels.fused_bias_gelu);
                    builder.arg(&mut self.workspace.ffn_inter);
                    builder.arg(bias);
                    builder.arg(&batch_seq);
                    builder.arg(&inter_dim);
                    unsafe { builder.launch(launch_cfg_1d(total_act)) }.map_err(cuda_err)?;
                } else {
                    let mut builder = self.stream.launch_builder(&self.kernels.gelu);
                    builder.arg(&mut self.workspace.ffn_inter);
                    builder.arg(&total_act);
                    unsafe { builder.launch(launch_cfg_1d(total_act)) }.map_err(cuda_err)?;
                }
                gpu_linear_f16(
                    &self.blas,
                    &self.stream,
                    &self.kernels,
                    &self.workspace.ffn_inter,
                    &self.layers[layer_idx].ffn.output_weight,
                    &mut self.workspace.scratch,
                    &mut self.workspace.input_f16,
                    batch_seq,
                    hd,
                    inter_dim,
                )?;
            }

            // Fused FFN output: bias+residual or just residual, then fused residual+layernorm
            if let Some(ref bias) = self.layers[layer_idx].ffn.output_bias {
                // Fused bias + residual: projected = scratch + bias + hidden_b
                let total_out = batch_seq * hd;
                let mut builder = self
                    .stream
                    .launch_builder(&self.kernels.fused_bias_residual);
                builder.arg(&mut self.workspace.projected);
                builder.arg(&self.workspace.scratch);
                builder.arg(bias);
                builder.arg(&self.workspace.hidden_b);
                builder.arg(&batch_seq);
                builder.arg(&hd);
                unsafe { builder.launch(launch_cfg_1d(total_out)) }.map_err(cuda_err)?;

                // Layer norm: projected → hidden_a
                let eps = self.layers[layer_idx].ffn.layer_norm_eps;
                let threads = 256_u32.min(hd as u32).next_power_of_two();
                let shared = threads * 2 * 4;
                let mut builder = self.stream.launch_builder(&self.kernels.layer_norm);
                builder.arg(&mut self.workspace.hidden_a);
                builder.arg(&self.workspace.projected);
                builder.arg(&self.layers[layer_idx].ffn.output_ln_weight);
                builder.arg(&self.layers[layer_idx].ffn.output_ln_bias);
                builder.arg(&batch_seq);
                builder.arg(&hd);
                builder.arg(&eps);
                unsafe { builder.launch(launch_cfg_per_row_shared(batch_seq, threads, shared)) }
                    .map_err(cuda_err)?;
            } else {
                // No bias: fused residual + layernorm in one kernel
                // hidden_a = layernorm(scratch + hidden_b)
                let eps = self.layers[layer_idx].ffn.layer_norm_eps;
                let threads = 256_u32.min(hd as u32).next_power_of_two();
                let shared = threads * 4;
                let mut builder = self
                    .stream
                    .launch_builder(&self.kernels.fused_residual_layernorm);
                builder.arg(&mut self.workspace.hidden_a);
                builder.arg(&self.workspace.scratch);
                builder.arg(&self.workspace.hidden_b);
                builder.arg(&self.layers[layer_idx].ffn.output_ln_weight);
                builder.arg(&self.layers[layer_idx].ffn.output_ln_bias);
                builder.arg(&batch_seq);
                builder.arg(&hd);
                builder.arg(&eps);
                unsafe { builder.launch(launch_cfg_per_row_shared(batch_seq, threads, shared)) }
                    .map_err(cuda_err)?;
            }
        }

        // ---------------------------------------------------------------
        // CLS pooling + L2 normalize
        // ---------------------------------------------------------------
        let cls_total = batch * hd;
        {
            let mut builder = self.stream.launch_builder(&self.kernels.cls_pool);
            builder.arg(&mut self.workspace.cls);
            builder.arg(&self.workspace.hidden_a);
            builder.arg(&batch);
            builder.arg(&max_seq);
            builder.arg(&hd);
            unsafe { builder.launch(launch_cfg_1d(cls_total)) }.map_err(cuda_err)?;
        }
        {
            let threads = 256_u32.min(hd as u32).next_power_of_two();
            let shared = threads * 4;
            let mut builder = self.stream.launch_builder(&self.kernels.l2_normalize);
            builder.arg(&mut self.workspace.cls);
            builder.arg(&batch);
            builder.arg(&hd);
            unsafe { builder.launch(launch_cfg_per_row_shared(batch, threads, shared)) }
                .map_err(cuda_err)?;
        }

        // ONE clone_dtoh to bring back all embeddings [batch * hidden]
        let flat_result = self
            .stream
            .clone_dtoh(&self.workspace.cls)
            .map_err(cuda_err)?;

        // Split into per-encoding vectors
        let hd_usize = hd as usize;
        let mut results = Vec::with_capacity(batch as usize);
        for b in 0..batch as usize {
            results.push(flat_result[b * hd_usize..(b + 1) * hd_usize].to_vec());
        }

        Ok(results)
    }
}

// ---------------------------------------------------------------------------
// Weight loading
// ---------------------------------------------------------------------------

/// Load a weight tensor to GPU as FP32, then convert to FP16 storage.
///
/// Returns the FP16 `CudaSlice<u16>` and the original shape.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "tensor element count fits in i32 for embedding models"
)]
fn load_to_gpu_f16(
    stream: &Arc<CudaStream>,
    kernels: &KernelHandles,
    tensors: &SafeTensors<'_>,
    name: &str,
) -> crate::Result<(CudaSlice<u16>, Vec<usize>)> {
    let (f32_slice, shape) = load_to_gpu(stream, tensors, name)?;
    let n = shape.iter().product::<usize>() as i32;
    let f16_slice = convert_to_f16(stream, kernels, &f32_slice, n)?;
    Ok((f16_slice, shape))
}

/// Load `ClassicBert` encoder layer weights to GPU.
///
/// GEMM weights are stored as FP16 for tensor core acceleration.
/// Biases and layernorm weights remain FP32 for precision.
#[expect(
    clippy::too_many_lines,
    reason = "monolithic weight loading for one encoder layer"
)]
fn load_classic_layer_gpu(
    stream: &Arc<CudaStream>,
    kernels: &KernelHandles,
    tensors: &SafeTensors<'_>,
    i: i32,
    config: &BertConfig,
) -> crate::Result<CudaBertLayer> {
    let prefix = format!("encoder.layer.{i}");

    // Load separate Q/K/V weights then fuse via concatenation, then convert to FP16
    let (q_weight, _) =
        load_tensor_host(tensors, &format!("{prefix}.attention.self.query.weight"))?;
    let (k_weight, _) = load_tensor_host(tensors, &format!("{prefix}.attention.self.key.weight"))?;
    let (v_weight, _) =
        load_tensor_host(tensors, &format!("{prefix}.attention.self.value.weight"))?;

    let mut qkv_data = Vec::with_capacity(q_weight.len() + k_weight.len() + v_weight.len());
    qkv_data.extend_from_slice(&q_weight);
    qkv_data.extend_from_slice(&k_weight);
    qkv_data.extend_from_slice(&v_weight);
    let qkv_f32 = stream.clone_htod(&qkv_data).map_err(cuda_err)?;
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "QKV weight element count fits in i32"
    )]
    let qkv_weight = convert_to_f16(stream, kernels, &qkv_f32, qkv_data.len() as i32)?;

    // Fuse biases (stay FP32)
    let q_bias = load_tensor_host(tensors, &format!("{prefix}.attention.self.query.bias")).ok();
    let k_bias = load_tensor_host(tensors, &format!("{prefix}.attention.self.key.bias")).ok();
    let v_bias = load_tensor_host(tensors, &format!("{prefix}.attention.self.value.bias")).ok();
    let qkv_bias = match (&q_bias, &k_bias, &v_bias) {
        (Some((qb, _)), Some((kb, _)), Some((vb, _))) => {
            let mut fused = Vec::with_capacity(qb.len() + kb.len() + vb.len());
            fused.extend_from_slice(qb);
            fused.extend_from_slice(kb);
            fused.extend_from_slice(vb);
            Some(stream.clone_htod(&fused).map_err(cuda_err)?)
        }
        _ => None,
    };

    let (output_weight, _) = load_to_gpu_f16(
        stream,
        kernels,
        tensors,
        &format!("{prefix}.attention.output.dense.weight"),
    )?;
    let output_bias = try_load_to_gpu(
        stream,
        tensors,
        &format!("{prefix}.attention.output.dense.bias"),
    )?;
    let (output_ln_weight, _) = load_to_gpu(
        stream,
        tensors,
        &format!("{prefix}.attention.output.LayerNorm.weight"),
    )?;
    let (output_ln_bias, _) = load_to_gpu(
        stream,
        tensors,
        &format!("{prefix}.attention.output.LayerNorm.bias"),
    )?;

    let attention = CudaBertSelfAttention {
        qkv_weight,
        qkv_bias,
        output_weight,
        output_bias,
        output_ln_weight,
        output_ln_bias,
        layer_norm_eps: config.layer_norm_eps,
    };

    // FFN — GEMM weights as FP16, biases and LN as FP32
    let (inter_weight, inter_shape) = load_to_gpu_f16(
        stream,
        kernels,
        tensors,
        &format!("{prefix}.intermediate.dense.weight"),
    )?;
    let inter_bias = try_load_to_gpu(
        stream,
        tensors,
        &format!("{prefix}.intermediate.dense.bias"),
    )?;
    let (out_weight, _) = load_to_gpu_f16(
        stream,
        kernels,
        tensors,
        &format!("{prefix}.output.dense.weight"),
    )?;
    let out_bias = try_load_to_gpu(stream, tensors, &format!("{prefix}.output.dense.bias"))?;
    let (out_ln_weight, _) = load_to_gpu(
        stream,
        tensors,
        &format!("{prefix}.output.LayerNorm.weight"),
    )?;
    let (out_ln_bias, _) =
        load_to_gpu(stream, tensors, &format!("{prefix}.output.LayerNorm.bias"))?;

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "intermediate dim is a small positive int from model config"
    )]
    let intermediate_dim = inter_shape[0] as i32;

    let ffn = CudaBertFfn {
        intermediate_weight: inter_weight,
        intermediate_bias: inter_bias,
        output_weight: out_weight,
        output_bias: out_bias,
        output_ln_weight: out_ln_weight,
        output_ln_bias: out_ln_bias,
        layer_norm_eps: config.layer_norm_eps,
        intermediate_dim,
    };

    Ok(CudaBertLayer { attention, ffn })
}

// ---------------------------------------------------------------------------
// Workspace allocation
// ---------------------------------------------------------------------------

/// Allocate workspace buffers for the given model config.
#[expect(
    clippy::cast_sign_loss,
    reason = "all dimensions are small positive ints from model config"
)]
fn allocate_workspace(
    stream: &Arc<CudaStream>,
    config: &BertConfig,
    intermediate_dim: i32,
) -> crate::Result<CudaWorkspace> {
    let hd = config.hidden_size;
    let nh = config.num_attention_heads;
    let head_dim = hd / nh;
    let max_seq = config.max_position_embeddings.min(512);
    let max_batch = MAX_BATCH;
    let bs = max_batch * max_seq;
    let bh = max_batch * nh;

    let alloc = |size: usize| -> crate::Result<CudaSlice<f32>> {
        stream.alloc_zeros::<f32>(size).map_err(cuda_err)
    };
    let alloc_f16 = |size: usize| -> crate::Result<CudaSlice<u16>> {
        stream.alloc_zeros::<u16>(size).map_err(cuda_err)
    };

    // FP16 input buffer must be large enough for the biggest GEMM input:
    // max(batch*seq * 3*hidden, batch*seq * intermediate)
    let max_input_f16 = (bs * (3 * hd).max(intermediate_dim)) as usize;

    Ok(CudaWorkspace {
        qkv: alloc((bs * 3 * hd) as usize)?,
        q: alloc((bh * max_seq * head_dim) as usize)?,
        k: alloc((bh * max_seq * head_dim) as usize)?,
        v: alloc((bh * max_seq * head_dim) as usize)?,
        scores: alloc((bh * max_seq * max_seq) as usize)?,
        attn_out: alloc((bh * max_seq * head_dim) as usize)?,
        context: alloc((bs * hd) as usize)?,
        ffn_inter: alloc((bs * intermediate_dim) as usize)?,
        hidden_a: alloc((bs * hd) as usize)?,
        hidden_b: alloc((bs * hd) as usize)?,
        projected: alloc((bs * hd) as usize)?,
        activated: alloc((bs * intermediate_dim) as usize)?,
        scratch: alloc((bs * hd) as usize)?,
        cls: alloc((max_batch * hd) as usize)?,
        input_f16: alloc_f16(max_input_f16)?,
        q_f16: alloc_f16((bh * max_seq * head_dim) as usize)?,
        k_f16: alloc_f16((bh * max_seq * head_dim) as usize)?,
        v_f16: alloc_f16((bh * max_seq * head_dim) as usize)?,
        scores_f16: alloc_f16((bh * max_seq * max_seq) as usize)?,
    })
}

// ---------------------------------------------------------------------------
// Public backend
// ---------------------------------------------------------------------------

/// CUDA-based BERT embedding backend using cudarc.
///
/// Uses cuBLAS for matrix multiplications and custom CUDA kernels (compiled
/// at runtime via NVRTC) for activations, layer normalization, softmax,
/// embedding lookup, and other element-wise operations.
///
/// All model weights reside on GPU memory. The forward pass runs entirely
/// on the GPU, with only the final L2-normalized embedding vector copied
/// back to the host.
///
/// Supports the `ClassicBert` family (BGE models), detected automatically
/// from weight names.
pub struct CudaBackend {
    /// The BERT model with all weights on GPU (interior mutability for workspace).
    model: std::cell::UnsafeCell<CudaBertModel>,
    /// Hidden dimension for output vector size validation.
    hidden_size: i32,
    /// Maximum sequence length supported by the model.
    max_position_embeddings: i32,
    /// Keep the CUDA module alive (holds the compiled PTX).
    _module: Arc<CudaModule>,
}

// Safety: CudaBackend is Send because all GPU resources (CudaSlice, CudaBlas,
// CudaStream) are internally reference-counted. The CUDA driver API is
// thread-safe for distinct handles.
#[expect(
    unsafe_code,
    reason = "GPU resources are refcounted and CUDA driver is thread-safe"
)]
unsafe impl Send for CudaBackend {}
// Safety: embed_batch serializes through the CUDA stream. The UnsafeCell is
// needed because embed_batch takes &self but forward_batch needs &mut self
// (for workspace buffers). Concurrent calls serialize through the CUDA stream.
#[expect(unsafe_code, reason = "CUDA stream serializes all GPU operations")]
unsafe impl Sync for CudaBackend {}

impl std::fmt::Debug for CudaBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaBackend")
            .field("hidden_size", &self.hidden_size)
            .field("max_position_embeddings", &self.max_position_embeddings)
            .finish_non_exhaustive()
    }
}

impl CudaBackend {
    /// Load a `ClassicBert` (BGE) embedding model onto an NVIDIA GPU.
    ///
    /// Downloads `model.safetensors` and `config.json` on first call;
    /// subsequent calls use the `hf-hub` cache. Compiles CUDA kernels via
    /// NVRTC and uploads all model weights to GPU memory. Returns an error
    /// if the model architecture is not recognized.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No CUDA device is available
    /// - The model cannot be downloaded
    /// - NVRTC kernel compilation fails
    /// - Weight loading or GPU memory allocation fails
    #[expect(
        clippy::cast_sign_loss,
        clippy::too_many_lines,
        reason = "monolithic load function; num_layers is a small positive int from config"
    )]
    pub fn load(model_repo: &str, _device_hint: &DeviceHint) -> crate::Result<Self> {
        // Initialize CUDA
        let ctx = CudaContext::new(0).map_err(cuda_err)?;
        let stream = ctx.default_stream();

        // Compile kernels
        let (module, kernels) = KernelHandles::compile(&ctx)?;

        // Create cuBLAS handle
        let blas = CudaBlas::new(stream.clone()).map_err(cuda_err)?;

        // Download model
        let api = Api::new().map_err(|e| crate::Error::Download(e.to_string()))?;
        let repo = api.model(model_repo.to_string());

        let config_path = repo
            .get("config.json")
            .map_err(|e| crate::Error::Download(e.to_string()))?;
        let weights_path = repo
            .get("model.safetensors")
            .map_err(|e| crate::Error::Download(e.to_string()))?;

        let model_bytes = std::fs::read(&weights_path).map_err(|e| crate::Error::Io {
            path: weights_path.display().to_string(),
            source: e,
        })?;

        let tensors = SafeTensors::deserialize(&model_bytes)
            .map_err(|e| crate::Error::Other(anyhow::anyhow!("safetensors parse error: {e}")))?;
        detect_variant(&tensors)?;

        let config_str = std::fs::read_to_string(&config_path).map_err(|e| crate::Error::Io {
            path: config_path.display().to_string(),
            source: e,
        })?;
        let config_json: serde_json::Value = serde_json::from_str(&config_str)
            .map_err(|e| crate::Error::Other(anyhow::anyhow!("config parse error: {e}")))?;
        let config = BertConfig::from_json(&config_json)?;

        let hidden_size = config.hidden_size;
        let max_position_embeddings = config.max_position_embeddings;
        let num_heads = config.num_attention_heads;
        let head_dim = hidden_size / num_heads;

        // Load embeddings to GPU
        let (word_emb, _) = load_to_gpu(&stream, &tensors, "embeddings.word_embeddings.weight")?;
        let (pos_emb, _) = load_to_gpu(&stream, &tensors, "embeddings.position_embeddings.weight")?;
        let tok_emb =
            try_load_to_gpu(&stream, &tensors, "embeddings.token_type_embeddings.weight")?;
        let (ln_w, _) = load_to_gpu(&stream, &tensors, "embeddings.LayerNorm.weight")?;
        let (ln_b, _) = load_to_gpu(&stream, &tensors, "embeddings.LayerNorm.bias")?;
        let embeddings = CudaBertEmbeddings {
            word_embeddings: word_emb,
            position_embeddings: Some(pos_emb),
            token_type_embeddings: tok_emb,
            layer_norm_weight: ln_w,
            layer_norm_bias: ln_b,
            layer_norm_eps: config.layer_norm_eps,
        };

        // Load encoder layers
        let mut layers = Vec::with_capacity(config.num_hidden_layers as usize);
        let mut intermediate_dim = 0_i32;
        for i in 0..config.num_hidden_layers {
            let layer = load_classic_layer_gpu(&stream, &kernels, &tensors, i, &config)?;
            intermediate_dim = layer.ffn.intermediate_dim;
            layers.push(layer);
        }

        // Allocate workspace buffers
        let workspace = allocate_workspace(&stream, &config, intermediate_dim)?;

        let model = CudaBertModel {
            stream,
            blas,
            kernels,
            embeddings,
            layers,
            hidden_size,
            num_heads,
            head_dim,
            workspace,
        };

        Ok(Self {
            model: std::cell::UnsafeCell::new(model),
            hidden_size,
            max_position_embeddings,
            _module: module,
        })
    }
}

impl EmbedBackend for CudaBackend {
    /// Embed a batch of pre-tokenized inputs using the full BERT forward pass on GPU.
    ///
    /// All encodings in the batch are padded to the same sequence length and
    /// processed in a single GPU pass. This eliminates per-encoding kernel
    /// launch overhead and enables batched cuBLAS GEMMs.
    #[expect(
        unsafe_code,
        reason = "UnsafeCell access is safe because CUDA stream serializes operations"
    )]
    fn embed_batch(&self, encodings: &[Encoding]) -> crate::Result<Vec<Vec<f32>>> {
        if encodings.is_empty() {
            return Ok(vec![]);
        }

        // Safety: we need &mut for workspace buffers, but the trait requires &self.
        // This is safe because the CUDA stream serializes all GPU operations, so
        // concurrent calls are ordered. The workspace is only modified during
        // forward_batch and not aliased.
        let model = unsafe { &mut *self.model.get() };

        // Split into sub-batches if needed (workspace is sized for MAX_BATCH)
        let max_batch = MAX_BATCH as usize;
        let mut all_results = Vec::with_capacity(encodings.len());

        for chunk in encodings.chunks(max_batch) {
            let mut results = model.forward_batch(chunk)?;
            all_results.append(&mut results);
        }

        Ok(all_results)
    }

    /// CUDA backend does not support cheap cloning (GPU resources are not trivially clonable).
    fn supports_clone(&self) -> bool {
        false
    }

    /// Not supported for CUDA backend.
    ///
    /// # Panics
    ///
    /// Always panics. Callers must check `supports_clone()` first.
    fn clone_backend(&self) -> Box<dyn EmbedBackend> {
        unimplemented!("CUDA backend does not support cloning; use ring-buffer pipeline instead")
    }

    /// CUDA backend runs on GPU.
    fn is_gpu(&self) -> bool {
        true
    }

    /// Maximum tokens from model config (512 for `ClassicBert`).
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
mod tests {
    use super::*;

    const BGE_SMALL: &str = "BAAI/bge-small-en-v1.5";

    #[test]
    #[ignore = "requires NVIDIA GPU; run with: cargo test -p ripvec-core --features cuda -- cuda --ignored"]
    fn cuda_backend_loads_and_embeds() {
        let backend = CudaBackend::load(BGE_SMALL, &DeviceHint::Gpu).unwrap();
        assert_eq!(backend.hidden_size, 384);
        assert_eq!(backend.max_position_embeddings, 512);
        assert!(backend.is_gpu());
        assert!(!backend.supports_clone());
        assert_eq!(backend.max_tokens(), 512);

        let enc = Encoding {
            input_ids: vec![101, 7592, 102],
            attention_mask: vec![1, 1, 1],
            token_type_ids: vec![0, 0, 0],
        };
        let results = backend.embed_batch(&[enc]).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].len(), 384);

        // Verify L2 normalization
        let norm: f32 = results[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "L2 norm should be ~1.0, got {norm}"
        );
    }

    #[test]
    #[ignore = "requires NVIDIA GPU; run with: cargo test -p ripvec-core --features cuda -- cuda --ignored"]
    fn cuda_backend_empty_batch() {
        let backend = CudaBackend::load(BGE_SMALL, &DeviceHint::Gpu).unwrap();
        let result = backend.embed_batch(&[]).unwrap();
        assert!(result.is_empty(), "empty batch should return empty vec");
    }

    #[test]
    #[ignore = "requires NVIDIA GPU; run with: cargo test -p ripvec-core --features cuda -- cuda --ignored"]
    fn cuda_backend_different_inputs_differ() {
        let backend = CudaBackend::load(BGE_SMALL, &DeviceHint::Gpu).unwrap();
        let enc1 = Encoding {
            input_ids: vec![101, 7592, 2088, 102],
            attention_mask: vec![1, 1, 1, 1],
            token_type_ids: vec![0, 0, 0, 0],
        };
        let enc2 = Encoding {
            input_ids: vec![101, 19387, 8840, 4313, 102],
            attention_mask: vec![1, 1, 1, 1, 1],
            token_type_ids: vec![0, 0, 0, 0, 0],
        };
        let results = backend.embed_batch(&[enc1, enc2]).unwrap();
        assert_eq!(results.len(), 2);

        let dot: f32 = results[0]
            .iter()
            .zip(results[1].iter())
            .map(|(a, b)| a * b)
            .sum();
        assert!(
            dot < 0.99,
            "different inputs should produce different embeddings, cosine sim = {dot}"
        );
    }

    #[test]
    #[ignore = "requires NVIDIA GPU; run with: cargo test -p ripvec-core --features cuda -- cuda --ignored"]
    fn cuda_backend_output_dim() {
        let backend = CudaBackend::load(BGE_SMALL, &DeviceHint::Gpu).unwrap();
        let enc = Encoding {
            input_ids: vec![101, 7592, 2088, 102],
            attention_mask: vec![1, 1, 1, 1],
            token_type_ids: vec![0, 0, 0, 0],
        };
        let result = backend.embed_batch(&[enc]).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(
            result[0].len(),
            384,
            "BGE-small should produce 384-dim embeddings"
        );
    }
}
