//! CUDA embedding backend using cudarc (cuBLAS + custom NVRTC kernels).
//!
//! Implements BERT inference on NVIDIA GPUs using [`cudarc`] for device
//! management, cuBLAS for matrix multiplications, and runtime-compiled CUDA
//! kernels for activations, layer normalization, softmax, and embedding lookup.
//!
//! Supports two model families:
//! - **`ClassicBert`** (BGE models): learned position embeddings, GELU, QKV with bias.
//! - **`NomicBert`** (`CodeRankEmbed`, nomic-embed-text): `RoPE`, `SwiGLU`, no bias.

use std::sync::Arc;

use cudarc::cublas::{sys, CudaBlas, Gemm, GemmConfig, StridedBatchedConfig};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;
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
// Model variant detection (shared logic with CPU backend)
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

// ---------------------------------------------------------------------------
// BERT model configuration
// ---------------------------------------------------------------------------

/// Configuration for a BERT-style encoder model.
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
    /// `SwiGLU` activation (value * silu(gate)).
    swiglu: CudaFunction,
    /// Layer normalization.
    layer_norm: CudaFunction,
    /// Softmax along last axis (per-row).
    softmax: CudaFunction,
    /// Add bias vector to rows.
    add_bias: CudaFunction,
    /// Element-wise residual addition.
    residual_add: CudaFunction,
    /// Embedding table lookup.
    embedding_lookup: CudaFunction,
    /// Add embedding table values to existing output.
    add_embeddings: CudaFunction,
    /// Build float attention mask from int mask.
    build_attn_mask: CudaFunction,
    /// Add attention mask to score matrix.
    add_attn_mask: CudaFunction,
    /// Scale all elements by a constant.
    scale: CudaFunction,
    /// CLS pooling (extract row 0 per batch element).
    cls_pool: CudaFunction,
    /// L2 normalize each row.
    l2_normalize: CudaFunction,
    /// Rotary position embedding.
    rope: CudaFunction,
    /// Split QKV `[batch*seq, 3*hidden]` into Q,K,V `[batch*num_heads, seq, head_dim]` on GPU.
    qkv_split: CudaFunction,
    /// Reshape attention output `[batch*num_heads, seq, head_dim]` to `[batch*seq, hidden]`.
    attn_reshape: CudaFunction,
}

impl KernelHandles {
    /// Compile CUDA kernels and load function handles.
    fn compile(ctx: &Arc<CudaContext>) -> crate::Result<(Arc<CudaModule>, Self)> {
        let ptx = compile_ptx(KERNELS).map_err(cuda_err)?;
        let module = ctx.load_module(ptx).map_err(cuda_err)?;
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
    /// Fused Q/K/V weight matrix `[3*hidden, hidden]`.
    qkv_weight: CudaSlice<f32>,
    /// Fused Q/K/V bias `[3*hidden]` (`ClassicBert` only).
    qkv_bias: Option<CudaSlice<f32>>,
    /// Output projection weight `[hidden, hidden]`.
    output_weight: CudaSlice<f32>,
    /// Output projection bias `[hidden]` (`ClassicBert` only).
    output_bias: Option<CudaSlice<f32>>,
    /// Post-attention `LayerNorm` weight `[hidden]`.
    output_ln_weight: CudaSlice<f32>,
    /// Post-attention `LayerNorm` bias `[hidden]`.
    output_ln_bias: CudaSlice<f32>,
    /// Layer norm epsilon.
    layer_norm_eps: f32,
    /// Rotary embedding base (`NomicBert` only).
    rotary_emb_base: Option<f32>,
}

/// Feed-forward network sub-layer on GPU.
struct CudaBertFfn {
    /// Intermediate projection weight.
    ///
    /// `ClassicBert`: `[intermediate, hidden]`.
    /// `NomicBert`: `[2*intermediate, hidden]` (gate+value fused).
    intermediate_weight: CudaSlice<f32>,
    /// Intermediate projection bias (`ClassicBert` only).
    intermediate_bias: Option<CudaSlice<f32>>,
    /// Output projection weight `[hidden, intermediate]`.
    output_weight: CudaSlice<f32>,
    /// Output projection bias (`ClassicBert` only).
    output_bias: Option<CudaSlice<f32>>,
    /// Post-FFN `LayerNorm` weight `[hidden]`.
    output_ln_weight: CudaSlice<f32>,
    /// Post-FFN `LayerNorm` bias `[hidden]`.
    output_ln_bias: CudaSlice<f32>,
    /// Layer norm epsilon.
    layer_norm_eps: f32,
    /// Intermediate dimension (for `ClassicBert`) or half-dim (for `NomicBert` `SwiGLU`).
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
    /// Model variant.
    variant: ModelVariant,
    /// Pre-allocated workspace buffers.
    workspace: CudaWorkspace,
}

// ---------------------------------------------------------------------------
// cuBLAS GEMM helpers
// ---------------------------------------------------------------------------

/// GPU matrix multiply: C = alpha * A @ B^T + beta * C
///
/// Input `a` is `[m, k]` row-major, `weight` is `[n, k]` row-major (stored
/// as-is from safetensors). This computes `C = A * W^T` producing `[m, n]`.
/// Output written to pre-allocated `output` buffer.
#[expect(unsafe_code, reason = "cuBLAS GEMM requires unsafe")]
fn gpu_linear(
    blas: &CudaBlas,
    input: &CudaSlice<f32>,
    weight: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    m: i32,
    n: i32,
    k: i32,
) -> crate::Result<()> {
    let cfg = GemmConfig {
        transa: sys::cublasOperation_t::CUBLAS_OP_T,
        transb: sys::cublasOperation_t::CUBLAS_OP_N,
        m: n,
        n: m,
        k,
        alpha: 1.0_f32,
        lda: k,
        ldb: k,
        beta: 0.0_f32,
        ldc: n,
    };

    unsafe {
        blas.gemm(cfg, weight, input, output).map_err(cuda_err)?;
    }

    Ok(())
}

/// Batched GEMM for multi-head attention: C = Q @ K^T with per-head strides.
///
/// `q` has shape `[batch_heads, seq, head_dim]`.
/// `k` has shape `[batch_heads, seq, head_dim]`.
/// `scores` (output) has shape `[batch_heads, seq, seq]`.
#[expect(unsafe_code, reason = "cuBLAS batched GEMM requires unsafe")]
fn gpu_batched_attn_scores(
    blas: &CudaBlas,
    q: &CudaSlice<f32>,
    k: &CudaSlice<f32>,
    scores: &mut CudaSlice<f32>,
    batch_heads: i32,
    seq: i32,
    head_dim: i32,
) -> crate::Result<()> {
    let cfg = StridedBatchedConfig {
        gemm: GemmConfig {
            transa: sys::cublasOperation_t::CUBLAS_OP_T,
            transb: sys::cublasOperation_t::CUBLAS_OP_N,
            m: seq,
            n: seq,
            k: head_dim,
            alpha: 1.0_f32,
            lda: head_dim,
            ldb: head_dim,
            beta: 0.0_f32,
            ldc: seq,
        },
        batch_size: batch_heads,
        stride_a: i64::from(seq * head_dim),
        stride_b: i64::from(seq * head_dim),
        stride_c: i64::from(seq * seq),
    };

    unsafe {
        blas.gemm_strided_batched(cfg, k, q, scores)
            .map_err(cuda_err)?;
    }

    Ok(())
}

/// Batched GEMM for attention output: C = scores @ V.
///
/// `scores` has shape `[batch_heads, seq, seq]`.
/// `v` has shape `[batch_heads, seq, head_dim]`.
/// `output` has shape `[batch_heads, seq, head_dim]`.
#[expect(unsafe_code, reason = "cuBLAS batched GEMM requires unsafe")]
fn gpu_batched_attn_output(
    blas: &CudaBlas,
    scores: &CudaSlice<f32>,
    v: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    batch_heads: i32,
    seq: i32,
    head_dim: i32,
) -> crate::Result<()> {
    let cfg = StridedBatchedConfig {
        gemm: GemmConfig {
            transa: sys::cublasOperation_t::CUBLAS_OP_N,
            transb: sys::cublasOperation_t::CUBLAS_OP_N,
            m: head_dim,
            n: seq,
            k: seq,
            alpha: 1.0_f32,
            lda: head_dim,
            ldb: seq,
            beta: 0.0_f32,
            ldc: head_dim,
        },
        batch_size: batch_heads,
        stride_a: i64::from(seq * head_dim),
        stride_b: i64::from(seq * seq),
        stride_c: i64::from(seq * head_dim),
    };

    unsafe {
        blas.gemm_strided_batched(cfg, v, scores, output)
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

            // QKV projection: [batch*seq, hidden] @ [3*hidden, hidden]^T
            gpu_linear(
                &self.blas,
                &self.workspace.hidden_a,
                &self.layers[layer_idx].attention.qkv_weight,
                &mut self.workspace.qkv,
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

            // RoPE for NomicBert
            if let Some(base) = self.layers[layer_idx].attention.rotary_emb_base {
                let half = head_dim / 2;
                let total_rope = batch_heads * max_seq * half;
                let rope_seq = batch_heads * max_seq;
                {
                    let mut builder = self.stream.launch_builder(&self.kernels.rope);
                    builder.arg(&mut self.workspace.q);
                    builder.arg(&rope_seq);
                    builder.arg(&head_dim);
                    builder.arg(&base);
                    unsafe { builder.launch(launch_cfg_1d(total_rope)) }.map_err(cuda_err)?;
                }
                {
                    let mut builder = self.stream.launch_builder(&self.kernels.rope);
                    builder.arg(&mut self.workspace.k);
                    builder.arg(&rope_seq);
                    builder.arg(&head_dim);
                    builder.arg(&base);
                    unsafe { builder.launch(launch_cfg_1d(total_rope)) }.map_err(cuda_err)?;
                }
            }

            // Attention scores: [batch*nh, seq, seq]
            gpu_batched_attn_scores(
                &self.blas,
                &self.workspace.q,
                &self.workspace.k,
                &mut self.workspace.scores,
                batch_heads,
                max_seq,
                head_dim,
            )?;

            // Scale
            let scale = 1.0_f32 / (head_dim as f32).sqrt();
            let total_scores = batch_heads * max_seq * max_seq;
            {
                let mut builder = self.stream.launch_builder(&self.kernels.scale);
                builder.arg(&mut self.workspace.scores);
                builder.arg(&scale);
                builder.arg(&total_scores);
                unsafe { builder.launch(launch_cfg_1d(total_scores)) }.map_err(cuda_err)?;
            }

            // Add attention mask (batched)
            {
                let mut builder = self.stream.launch_builder(&self.kernels.add_attn_mask);
                builder.arg(&mut self.workspace.scores);
                builder.arg(&attn_mask_dev);
                builder.arg(&batch);
                builder.arg(&nh);
                builder.arg(&max_seq);
                unsafe { builder.launch(launch_cfg_1d(total_scores)) }.map_err(cuda_err)?;
            }

            // Softmax
            {
                let total_rows = batch_heads * max_seq;
                let threads = 256_u32.min(max_seq as u32).next_power_of_two();
                let shared = threads * 4;
                let mut builder = self.stream.launch_builder(&self.kernels.softmax);
                builder.arg(&mut self.workspace.scores);
                builder.arg(&total_rows);
                builder.arg(&max_seq);
                unsafe { builder.launch(launch_cfg_per_row_shared(total_rows, threads, shared)) }
                    .map_err(cuda_err)?;
            }

            // Attention output: [batch*nh, seq, head_dim]
            gpu_batched_attn_output(
                &self.blas,
                &self.workspace.scores,
                &self.workspace.v,
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

            // Output projection → projected
            gpu_linear(
                &self.blas,
                &self.workspace.context,
                &self.layers[layer_idx].attention.output_weight,
                &mut self.workspace.projected,
                batch_seq,
                hd,
                hd,
            )?;

            if let Some(ref bias) = self.layers[layer_idx].attention.output_bias {
                let total_proj = batch_seq * hd;
                let mut builder = self.stream.launch_builder(&self.kernels.add_bias);
                builder.arg(&mut self.workspace.projected);
                builder.arg(bias);
                builder.arg(&batch_seq);
                builder.arg(&hd);
                unsafe { builder.launch(launch_cfg_1d(total_proj)) }.map_err(cuda_err)?;
            }

            // Residual: projected += hidden_a
            {
                let total_res = batch_seq * hd;
                let mut builder = self.stream.launch_builder(&self.kernels.residual_add);
                builder.arg(&mut self.workspace.projected);
                builder.arg(&self.workspace.hidden_a);
                builder.arg(&total_res);
                unsafe { builder.launch(launch_cfg_1d(total_res)) }.map_err(cuda_err)?;
            }

            // Attention layer norm: projected → hidden_b
            {
                let eps = self.layers[layer_idx].attention.layer_norm_eps;
                let threads = 256_u32.min(hd as u32).next_power_of_two();
                let shared = threads * 2 * 4;
                let mut builder = self.stream.launch_builder(&self.kernels.layer_norm);
                builder.arg(&mut self.workspace.hidden_b);
                builder.arg(&self.workspace.projected);
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
            let inter_out_dim = match self.variant {
                ModelVariant::ClassicBert => inter_dim,
                ModelVariant::NomicBert => 2 * inter_dim,
            };

            // Intermediate projection
            gpu_linear(
                &self.blas,
                &self.workspace.hidden_b,
                &self.layers[layer_idx].ffn.intermediate_weight,
                &mut self.workspace.ffn_inter,
                batch_seq,
                inter_out_dim,
                hd,
            )?;

            if let Some(ref bias) = self.layers[layer_idx].ffn.intermediate_bias {
                let total_inter = batch_seq * inter_out_dim;
                let mut builder = self.stream.launch_builder(&self.kernels.add_bias);
                builder.arg(&mut self.workspace.ffn_inter);
                builder.arg(bias);
                builder.arg(&batch_seq);
                builder.arg(&inter_out_dim);
                unsafe { builder.launch(launch_cfg_1d(total_inter)) }.map_err(cuda_err)?;
            }

            // Activation + output projection → scratch
            match self.variant {
                ModelVariant::ClassicBert => {
                    let total_act = batch_seq * inter_dim;
                    {
                        let mut builder = self.stream.launch_builder(&self.kernels.gelu);
                        builder.arg(&mut self.workspace.ffn_inter);
                        builder.arg(&total_act);
                        unsafe { builder.launch(launch_cfg_1d(total_act)) }.map_err(cuda_err)?;
                    }
                    gpu_linear(
                        &self.blas,
                        &self.workspace.ffn_inter,
                        &self.layers[layer_idx].ffn.output_weight,
                        &mut self.workspace.scratch,
                        batch_seq,
                        hd,
                        inter_dim,
                    )?;
                }
                ModelVariant::NomicBert => {
                    let half_cols = inter_dim;
                    let total_act = batch_seq * half_cols;
                    {
                        let mut builder = self.stream.launch_builder(&self.kernels.swiglu);
                        builder.arg(&mut self.workspace.activated);
                        builder.arg(&self.workspace.ffn_inter);
                        builder.arg(&batch_seq);
                        builder.arg(&half_cols);
                        unsafe { builder.launch(launch_cfg_1d(total_act)) }.map_err(cuda_err)?;
                    }
                    gpu_linear(
                        &self.blas,
                        &self.workspace.activated,
                        &self.layers[layer_idx].ffn.output_weight,
                        &mut self.workspace.scratch,
                        batch_seq,
                        hd,
                        inter_dim,
                    )?;
                }
            }

            // Add FFN output bias
            if let Some(ref bias) = self.layers[layer_idx].ffn.output_bias {
                let total_out = batch_seq * hd;
                let mut builder = self.stream.launch_builder(&self.kernels.add_bias);
                builder.arg(&mut self.workspace.scratch);
                builder.arg(bias);
                builder.arg(&batch_seq);
                builder.arg(&hd);
                unsafe { builder.launch(launch_cfg_1d(total_out)) }.map_err(cuda_err)?;
            }

            // Residual: scratch += hidden_b
            {
                let total_res = batch_seq * hd;
                let mut builder = self.stream.launch_builder(&self.kernels.residual_add);
                builder.arg(&mut self.workspace.scratch);
                builder.arg(&self.workspace.hidden_b);
                builder.arg(&total_res);
                unsafe { builder.launch(launch_cfg_1d(total_res)) }.map_err(cuda_err)?;
            }

            // FFN layer norm: scratch → hidden_a (ready for next layer)
            {
                let eps = self.layers[layer_idx].ffn.layer_norm_eps;
                let threads = 256_u32.min(hd as u32).next_power_of_two();
                let shared = threads * 2 * 4;
                let mut builder = self.stream.launch_builder(&self.kernels.layer_norm);
                builder.arg(&mut self.workspace.hidden_a);
                builder.arg(&self.workspace.scratch);
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

/// Load `ClassicBert` encoder layer weights to GPU.
fn load_classic_layer_gpu(
    stream: &Arc<CudaStream>,
    tensors: &SafeTensors<'_>,
    i: i32,
    config: &BertConfig,
) -> crate::Result<CudaBertLayer> {
    let prefix = format!("encoder.layer.{i}");

    // Load separate Q/K/V weights then fuse via concatenation
    let (q_weight, _) =
        load_tensor_host(tensors, &format!("{prefix}.attention.self.query.weight"))?;
    let (k_weight, _) = load_tensor_host(tensors, &format!("{prefix}.attention.self.key.weight"))?;
    let (v_weight, _) =
        load_tensor_host(tensors, &format!("{prefix}.attention.self.value.weight"))?;

    let mut qkv_data = Vec::with_capacity(q_weight.len() + k_weight.len() + v_weight.len());
    qkv_data.extend_from_slice(&q_weight);
    qkv_data.extend_from_slice(&k_weight);
    qkv_data.extend_from_slice(&v_weight);
    let qkv_weight = stream.clone_htod(&qkv_data).map_err(cuda_err)?;

    // Fuse biases
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

    let (output_weight, _) = load_to_gpu(
        stream,
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
        rotary_emb_base: None,
    };

    // FFN
    let (inter_weight, inter_shape) = load_to_gpu(
        stream,
        tensors,
        &format!("{prefix}.intermediate.dense.weight"),
    )?;
    let inter_bias = try_load_to_gpu(
        stream,
        tensors,
        &format!("{prefix}.intermediate.dense.bias"),
    )?;
    let (out_weight, _) = load_to_gpu(stream, tensors, &format!("{prefix}.output.dense.weight"))?;
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

/// Load `NomicBert` encoder layer weights to GPU.
fn load_nomic_layer_gpu(
    stream: &Arc<CudaStream>,
    tensors: &SafeTensors<'_>,
    i: i32,
    config: &BertConfig,
) -> crate::Result<CudaBertLayer> {
    let prefix = format!("encoder.layers.{i}");

    let (qkv_weight, _) = load_to_gpu(stream, tensors, &format!("{prefix}.attn.Wqkv.weight"))?;

    let (output_weight, _) =
        load_to_gpu(stream, tensors, &format!("{prefix}.attn.out_proj.weight"))?;
    let (output_ln_weight, _) = load_to_gpu(stream, tensors, &format!("{prefix}.norm1.weight"))?;
    let (output_ln_bias, _) = load_to_gpu(stream, tensors, &format!("{prefix}.norm1.bias"))?;

    let attention = CudaBertSelfAttention {
        qkv_weight,
        qkv_bias: None,
        output_weight,
        output_bias: None,
        output_ln_weight,
        output_ln_bias,
        layer_norm_eps: config.layer_norm_eps,
        rotary_emb_base: Some(config.rotary_emb_base),
    };

    // SwiGLU: fc11 = value/up, fc12 = gate, fc2 = down
    let (fc11, fc11_shape) = load_tensor_host(tensors, &format!("{prefix}.mlp.fc11.weight"))?;
    let (fc12, _) = load_tensor_host(tensors, &format!("{prefix}.mlp.fc12.weight"))?;

    let mut gate_up = Vec::with_capacity(fc11.len() + fc12.len());
    gate_up.extend_from_slice(&fc11);
    gate_up.extend_from_slice(&fc12);
    let intermediate_weight = stream.clone_htod(&gate_up).map_err(cuda_err)?;

    let (output_weight_ffn, _) = load_to_gpu(stream, tensors, &format!("{prefix}.mlp.fc2.weight"))?;
    let (out_ln_weight, _) = load_to_gpu(stream, tensors, &format!("{prefix}.norm2.weight"))?;
    let (out_ln_bias, _) = load_to_gpu(stream, tensors, &format!("{prefix}.norm2.bias"))?;

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "intermediate dim is a small positive int from model config"
    )]
    let intermediate_dim = fc11_shape[0] as i32;

    let ffn = CudaBertFfn {
        intermediate_weight,
        intermediate_bias: None,
        output_weight: output_weight_ffn,
        output_bias: None,
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
    // Cap workspace seq length at 512 — real chunks rarely exceed this.
    // NomicBert's 8192 max_position_embeddings would require 400+ GB for scores.
    // If a batch exceeds this, forward_batch will allocate dynamically.
    let max_seq = config.max_position_embeddings.min(512);
    let max_batch = MAX_BATCH;
    let bs = max_batch * max_seq;
    let bh = max_batch * nh;

    let inter_out = match config.variant {
        ModelVariant::ClassicBert => intermediate_dim,
        ModelVariant::NomicBert => 2 * intermediate_dim,
    };

    let alloc = |size: usize| -> crate::Result<CudaSlice<f32>> {
        stream.alloc_zeros::<f32>(size).map_err(cuda_err)
    };

    Ok(CudaWorkspace {
        qkv: alloc((bs * 3 * hd) as usize)?,
        q: alloc((bh * max_seq * head_dim) as usize)?,
        k: alloc((bh * max_seq * head_dim) as usize)?,
        v: alloc((bh * max_seq * head_dim) as usize)?,
        scores: alloc((bh * max_seq * max_seq) as usize)?,
        attn_out: alloc((bh * max_seq * head_dim) as usize)?,
        context: alloc((bs * hd) as usize)?,
        ffn_inter: alloc((bs * inter_out) as usize)?,
        hidden_a: alloc((bs * hd) as usize)?,
        hidden_b: alloc((bs * hd) as usize)?,
        projected: alloc((bs * hd) as usize)?,
        activated: alloc((bs * intermediate_dim) as usize)?,
        scratch: alloc((bs * hd) as usize)?,
        cls: alloc((max_batch * hd) as usize)?,
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
/// Supports both `ClassicBert` (BGE) and `NomicBert` (`CodeRankEmbed`) model
/// families, detected automatically from weight names.
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
    /// Load a BERT/BGE/`NomicBert` embedding model onto an NVIDIA GPU.
    ///
    /// Downloads `model.safetensors` and `config.json` on first call;
    /// subsequent calls use the `hf-hub` cache. Compiles CUDA kernels via
    /// NVRTC and uploads all model weights to GPU memory.
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
        reason = "num_layers is a small positive int from config"
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
        let variant = detect_variant(&tensors);

        let config_str = std::fs::read_to_string(&config_path).map_err(|e| crate::Error::Io {
            path: config_path.display().to_string(),
            source: e,
        })?;
        let config_json: serde_json::Value = serde_json::from_str(&config_str)
            .map_err(|e| crate::Error::Other(anyhow::anyhow!("config parse error: {e}")))?;
        let config = BertConfig::from_json(&config_json, variant)?;

        let hidden_size = config.hidden_size;
        let max_position_embeddings = config.max_position_embeddings;
        let num_heads = config.num_attention_heads;
        let head_dim = hidden_size / num_heads;

        // Load embeddings to GPU
        let embeddings = match config.variant {
            ModelVariant::ClassicBert => {
                let (word_emb, _) =
                    load_to_gpu(&stream, &tensors, "embeddings.word_embeddings.weight")?;
                let (pos_emb, _) =
                    load_to_gpu(&stream, &tensors, "embeddings.position_embeddings.weight")?;
                let tok_emb =
                    try_load_to_gpu(&stream, &tensors, "embeddings.token_type_embeddings.weight")?;
                let (ln_w, _) = load_to_gpu(&stream, &tensors, "embeddings.LayerNorm.weight")?;
                let (ln_b, _) = load_to_gpu(&stream, &tensors, "embeddings.LayerNorm.bias")?;

                CudaBertEmbeddings {
                    word_embeddings: word_emb,
                    position_embeddings: Some(pos_emb),
                    token_type_embeddings: tok_emb,
                    layer_norm_weight: ln_w,
                    layer_norm_bias: ln_b,
                    layer_norm_eps: config.layer_norm_eps,
                }
            }
            ModelVariant::NomicBert => {
                let (word_emb, _) =
                    load_to_gpu(&stream, &tensors, "embeddings.word_embeddings.weight")?;
                let tok_emb =
                    try_load_to_gpu(&stream, &tensors, "embeddings.token_type_embeddings.weight")?;
                let (ln_w, _) = load_to_gpu(&stream, &tensors, "emb_ln.weight")?;
                let (ln_b, _) = load_to_gpu(&stream, &tensors, "emb_ln.bias")?;

                CudaBertEmbeddings {
                    word_embeddings: word_emb,
                    position_embeddings: None,
                    token_type_embeddings: tok_emb,
                    layer_norm_weight: ln_w,
                    layer_norm_bias: ln_b,
                    layer_norm_eps: config.layer_norm_eps,
                }
            }
        };

        // Load encoder layers
        let mut layers = Vec::with_capacity(config.num_hidden_layers as usize);
        let mut intermediate_dim = 0_i32;
        for i in 0..config.num_hidden_layers {
            let layer = match config.variant {
                ModelVariant::ClassicBert => load_classic_layer_gpu(&stream, &tensors, i, &config)?,
                ModelVariant::NomicBert => load_nomic_layer_gpu(&stream, &tensors, i, &config)?,
            };
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
            variant,
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

    /// Maximum tokens from model config (512 for BERT, 8192 for `NomicBert`).
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
