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
extern "C" __global__ void build_attn_mask_kernel(
    float* output, const int* mask, int seq_len
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < seq_len) {
        output[i] = (mask[i] == 1) ? 0.0f : -1e9f;
    }
}

// Add attention mask to scores: scores[row, col] += mask[col]
extern "C" __global__ void add_attn_mask_kernel(
    float* scores, const float* mask, int rows, int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        scores[idx] += mask[idx % cols];
    }
}

// Scale all elements: x[i] *= scale
extern "C" __global__ void scale_kernel(float* x, float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= scale;
}

// CLS pooling: copy first row to output
extern "C" __global__ void cls_pool_kernel(
    float* output, const float* hidden, int hidden_dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hidden_dim) {
        output[i] = hidden[i];
    }
}

// L2 normalize a single vector
extern "C" __global__ void l2_normalize_kernel(float* x, int cols) {
    extern __shared__ float sdata[];

    float local_sq = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        local_sq += x[i] * x[i];
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
        x[i] *= inv_norm;
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
    /// CLS pooling (copy first row).
    cls_pool: CudaFunction,
    /// L2 normalize a vector.
    l2_normalize: CudaFunction,
    /// Rotary position embedding.
    rope: CudaFunction,
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
    /// Number of attention heads.
    num_heads: i32,
    /// Dimension per head (`hidden / num_heads`).
    head_dim: i32,
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
    /// Model variant (determines GELU vs `SwiGLU` activation).
    variant: ModelVariant,
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
}

// ---------------------------------------------------------------------------
// cuBLAS GEMM helpers
// ---------------------------------------------------------------------------

/// GPU matrix multiply: C = alpha * A @ B^T + beta * C
///
/// Input `a` is `[m, k]` row-major, `weight` is `[n, k]` row-major (stored
/// as-is from safetensors). This computes `C = A * W^T` producing `[m, n]`.
///
/// cuBLAS uses column-major ordering. For row-major data:
/// `C_rm = A_rm @ B_rm^T` is equivalent to `C_cm^T = B_cm @ A_cm^T`.
/// So we call sgemm with `(B, A)` in cuBLAS terms with `CUBLAS_OP_T` on A.
#[expect(
    unsafe_code,
    clippy::cast_sign_loss,
    reason = "cuBLAS GEMM requires unsafe; dimensions are small positive ints"
)]
fn gpu_linear(
    blas: &CudaBlas,
    stream: &Arc<CudaStream>,
    input: &CudaSlice<f32>,
    weight: &CudaSlice<f32>,
    m: i32,
    n: i32,
    k: i32,
) -> crate::Result<CudaSlice<f32>> {
    let mut output = stream
        .alloc_zeros::<f32>((m * n) as usize)
        .map_err(cuda_err)?;

    // cuBLAS column-major: C(n,m) = B(n,k) @ A(k,m)
    // where B = weight [n,k] and A = input [m,k]^T
    // transa = CUBLAS_OP_T (transpose A from [m,k] to [k,m])
    // transb = CUBLAS_OP_N (B is already [n,k] which in col-major is fine)
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
        blas.gemm(cfg, weight, input, &mut output)
            .map_err(cuda_err)?;
    }

    Ok(output)
}

/// Batched GEMM for multi-head attention: C = A @ B^T with per-head strides.
///
/// `q` has shape `[num_heads, seq, head_dim]` (contiguous, stride = `seq*head_dim`).
/// `k` has shape `[num_heads, seq, head_dim]` (same layout).
/// Output has shape `[num_heads, seq, seq]` (stride = `seq*seq`).
#[expect(
    unsafe_code,
    clippy::cast_sign_loss,
    reason = "cuBLAS batched GEMM requires unsafe; dimensions are small positive ints"
)]
fn gpu_batched_attn_scores(
    blas: &CudaBlas,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<f32>,
    k: &CudaSlice<f32>,
    num_heads: i32,
    seq: i32,
    head_dim: i32,
) -> crate::Result<CudaSlice<f32>> {
    let mut scores = stream
        .alloc_zeros::<f32>((num_heads * seq * seq) as usize)
        .map_err(cuda_err)?;

    // Row-major C[seq,seq] = Q[seq,hd] @ K[seq,hd]^T
    // cuBLAS is col-major. For row-major A[m,k] @ B[n,k]^T = C[m,n]:
    //   call sgemm with: transa=T, transb=N, m=seq, n=seq, k=hd,
    //   A_cublas=K (col-major view: [hd,seq]), B_cublas=Q (col-major view: [hd,seq])
    //   lda=seq (stride between cols in row-major K = seq),
    //   ldb=seq (stride between cols in row-major Q = seq),
    //   ldc=seq
    // Actually simpler: swap A and B, use N and T:
    //   C_cm[seq,seq] = K_rm_as_cm[hd,seq] @ Q_rm_as_cm^T[seq,hd]
    // But cuBLAS output C is col-major [seq,seq] which is same as row-major [seq,seq] for square.
    //
    // Correct approach for row-major: C = A @ B^T → cuBLAS: C^T = B @ A^T
    // Since C is symmetric-shaped (seq×seq), C^T has same layout.
    // A=Q[seq,hd] stored row-major → col-major view [hd,seq], ld=hd
    // B=K[seq,hd] stored row-major → col-major view [hd,seq], ld=hd
    // C^T[seq,seq] = B_cm[hd,seq].T @ A_cm[hd,seq]
    // → transa=T on K: K^T_cm = [seq,hd], transb=N on Q: Q_cm = [hd,seq]
    // → m=seq (rows of op(A)), n=seq (cols of op(B)), k=hd
    // → lda=hd (leading dim of A before op = rows of K_cm = hd)
    // → ldb=hd (leading dim of B = rows of Q_cm = hd)
    // → ldc=seq
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
        batch_size: num_heads,
        stride_a: i64::from(seq * head_dim),
        stride_b: i64::from(seq * head_dim),
        stride_c: i64::from(seq * seq),
    };

    unsafe {
        blas.gemm_strided_batched(cfg, k, q, &mut scores)
            .map_err(cuda_err)?;
    }

    Ok(scores)
}

/// Batched GEMM for attention output: C = scores @ V.
///
/// `scores` has shape `[num_heads, seq, seq]`.
/// `v` has shape `[num_heads, seq, head_dim]`.
/// Output has shape `[num_heads, seq, head_dim]`.
#[expect(
    unsafe_code,
    clippy::cast_sign_loss,
    reason = "cuBLAS batched GEMM requires unsafe; dimensions are small positive ints"
)]
fn gpu_batched_attn_output(
    blas: &CudaBlas,
    stream: &Arc<CudaStream>,
    scores: &CudaSlice<f32>,
    v: &CudaSlice<f32>,
    num_heads: i32,
    seq: i32,
    head_dim: i32,
) -> crate::Result<CudaSlice<f32>> {
    let mut output = stream
        .alloc_zeros::<f32>((num_heads * seq * head_dim) as usize)
        .map_err(cuda_err)?;

    // scores[h] @ V[h]: [seq, seq] @ [seq, head_dim] = [seq, head_dim]
    // In col-major: C_cm[hd, seq] = V_cm[hd, seq] @ scores_cm[seq, seq]
    // A = V, transa = N, B = scores, transb = N
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
        batch_size: num_heads,
        stride_a: i64::from(seq * head_dim),
        stride_b: i64::from(seq * seq),
        stride_c: i64::from(seq * head_dim),
    };

    unsafe {
        blas.gemm_strided_batched(cfg, v, scores, &mut output)
            .map_err(cuda_err)?;
    }

    Ok(output)
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
    /// Run embedding lookup + optional position / token-type + layer norm.
    #[expect(
        unsafe_code,
        clippy::cast_sign_loss,
        reason = "kernel launches require unsafe; dimensions are small non-negative values"
    )]
    fn forward_embeddings(
        &self,
        input_ids: &CudaSlice<i32>,
        token_type_ids: &CudaSlice<i32>,
        position_ids: &CudaSlice<i32>,
        seq_len: i32,
    ) -> crate::Result<CudaSlice<f32>> {
        let hd = self.hidden_size;
        let n = seq_len * hd;

        // Word embedding lookup
        let mut output = self
            .stream
            .alloc_zeros::<f32>(n as usize)
            .map_err(cuda_err)?;
        {
            let mut builder = self.stream.launch_builder(&self.kernels.embedding_lookup);
            builder.arg(&mut output);
            builder.arg(&self.embeddings.word_embeddings);
            builder.arg(input_ids);
            builder.arg(&seq_len);
            builder.arg(&hd);
            unsafe { builder.launch(launch_cfg_1d(n)) }.map_err(cuda_err)?;
        }

        // Add position embeddings (ClassicBert)
        if let Some(ref pos_emb) = self.embeddings.position_embeddings {
            let mut builder = self.stream.launch_builder(&self.kernels.add_embeddings);
            builder.arg(&mut output);
            builder.arg(pos_emb);
            builder.arg(position_ids);
            builder.arg(&seq_len);
            builder.arg(&hd);
            unsafe { builder.launch(launch_cfg_1d(n)) }.map_err(cuda_err)?;
        }

        // Add token type embeddings (ClassicBert)
        if let Some(ref tok_emb) = self.embeddings.token_type_embeddings {
            let mut builder = self.stream.launch_builder(&self.kernels.add_embeddings);
            builder.arg(&mut output);
            builder.arg(tok_emb);
            builder.arg(token_type_ids);
            builder.arg(&seq_len);
            builder.arg(&hd);
            unsafe { builder.launch(launch_cfg_1d(n)) }.map_err(cuda_err)?;
        }

        // Layer norm
        let mut normed = self
            .stream
            .alloc_zeros::<f32>(n as usize)
            .map_err(cuda_err)?;
        let threads = 256_u32.min(hd as u32).next_power_of_two();
        let shared = threads * 2 * 4; // 2 shared arrays of floats
        {
            let eps = self.embeddings.layer_norm_eps;
            let mut builder = self.stream.launch_builder(&self.kernels.layer_norm);
            builder.arg(&mut normed);
            builder.arg(&output);
            builder.arg(&self.embeddings.layer_norm_weight);
            builder.arg(&self.embeddings.layer_norm_bias);
            builder.arg(&seq_len);
            builder.arg(&hd);
            builder.arg(&eps);
            unsafe { builder.launch(launch_cfg_per_row_shared(seq_len, threads, shared)) }
                .map_err(cuda_err)?;
        }

        Ok(normed)
    }

    /// Self-attention forward pass for one layer.
    #[expect(
        unsafe_code,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        clippy::too_many_lines,
        reason = "kernel launches require unsafe; dimensions from config are small positive ints"
    )]
    fn forward_attention(
        &self,
        layer: &CudaBertSelfAttention,
        hidden: &CudaSlice<f32>,
        mask: &CudaSlice<f32>,
        seq: i32,
    ) -> crate::Result<CudaSlice<f32>> {
        let hd = self.hidden_size;
        let nh = layer.num_heads;
        let head_dim = layer.head_dim;

        // Fused QKV projection: [seq, hidden] @ [hidden, 3*hidden]^T = [seq, 3*hidden]
        let mut qkv = gpu_linear(
            &self.blas,
            &self.stream,
            hidden,
            &layer.qkv_weight,
            seq,
            3 * hd,
            hd,
        )?;

        // Add QKV bias if present
        if let Some(ref bias) = layer.qkv_bias {
            let total = seq * 3 * hd;
            let cols = 3 * hd;
            let mut builder = self.stream.launch_builder(&self.kernels.add_bias);
            builder.arg(&mut qkv);
            builder.arg(bias);
            builder.arg(&seq);
            builder.arg(&cols);
            unsafe { builder.launch(launch_cfg_1d(total)) }.map_err(cuda_err)?;
        }

        // QKV is [seq, 3*hidden] row-major. Split into Q, K, V each [seq, hidden],
        // then reshape to [num_heads, seq, head_dim] for batched attention.
        // We rearrange via host for simplicity. This is a small tensor.
        let qkv_host = self.stream.clone_dtoh(&qkv).map_err(cuda_err)?;

        let head_stride = (nh * seq * head_dim) as usize;
        let mut q_heads = vec![0.0_f32; head_stride];
        let mut k_heads = vec![0.0_f32; head_stride];
        let mut v_heads = vec![0.0_f32; head_stride];

        for t in 0..seq as usize {
            for h in 0..nh as usize {
                for d in 0..head_dim as usize {
                    let qkv_base = t * (3 * hd as usize);
                    let head_offset = h * head_dim as usize + d;
                    let heads_idx =
                        h * (seq as usize * head_dim as usize) + t * head_dim as usize + d;

                    q_heads[heads_idx] = qkv_host[qkv_base + head_offset];
                    k_heads[heads_idx] = qkv_host[qkv_base + hd as usize + head_offset];
                    v_heads[heads_idx] = qkv_host[qkv_base + 2 * hd as usize + head_offset];
                }
            }
        }

        let mut q_dev = self.stream.clone_htod(&q_heads).map_err(cuda_err)?;
        let mut k_dev = self.stream.clone_htod(&k_heads).map_err(cuda_err)?;
        let v_dev = self.stream.clone_htod(&v_heads).map_err(cuda_err)?;

        // Apply RoPE for NomicBert
        if let Some(base) = layer.rotary_emb_base {
            let half = head_dim / 2;
            let total_rope = nh * seq * half;
            let rope_seq = nh * seq;

            {
                let mut builder = self.stream.launch_builder(&self.kernels.rope);
                builder.arg(&mut q_dev);
                builder.arg(&rope_seq);
                builder.arg(&head_dim);
                builder.arg(&base);
                unsafe { builder.launch(launch_cfg_1d(total_rope)) }.map_err(cuda_err)?;
            }
            {
                let mut builder = self.stream.launch_builder(&self.kernels.rope);
                builder.arg(&mut k_dev);
                builder.arg(&rope_seq);
                builder.arg(&head_dim);
                builder.arg(&base);
                unsafe { builder.launch(launch_cfg_1d(total_rope)) }.map_err(cuda_err)?;
            }
        }

        // Batched attention scores: [nh, seq, seq] = Q @ K^T
        let mut scores =
            gpu_batched_attn_scores(&self.blas, &self.stream, &q_dev, &k_dev, nh, seq, head_dim)?;

        // Scale by 1/sqrt(head_dim)
        let scale = 1.0_f32 / (head_dim as f32).sqrt();
        let total_scores = nh * seq * seq;
        {
            let mut builder = self.stream.launch_builder(&self.kernels.scale);
            builder.arg(&mut scores);
            builder.arg(&scale);
            builder.arg(&total_scores);
            unsafe { builder.launch(launch_cfg_1d(total_scores)) }.map_err(cuda_err)?;
        }

        // Add attention mask to each head's score matrix
        {
            let total_mask = nh * seq * seq;
            let mask_rows = nh * seq;
            let mut builder = self.stream.launch_builder(&self.kernels.add_attn_mask);
            builder.arg(&mut scores);
            builder.arg(mask);
            builder.arg(&mask_rows);
            builder.arg(&seq);
            unsafe { builder.launch(launch_cfg_1d(total_mask)) }.map_err(cuda_err)?;
        }

        // Softmax per row
        {
            let total_rows = nh * seq;
            let threads = 256_u32.min(seq as u32).next_power_of_two();
            let shared = threads * 4;
            let mut builder = self.stream.launch_builder(&self.kernels.softmax);
            builder.arg(&mut scores);
            builder.arg(&total_rows);
            builder.arg(&seq);
            unsafe { builder.launch(launch_cfg_per_row_shared(total_rows, threads, shared)) }
                .map_err(cuda_err)?;
        }

        // Attention output: [nh, seq, head_dim] = scores @ V
        let attn_out =
            gpu_batched_attn_output(&self.blas, &self.stream, &scores, &v_dev, nh, seq, head_dim)?;

        // Rearrange back from [nh, seq, head_dim] to [seq, hidden]
        let attn_host = self.stream.clone_dtoh(&attn_out).map_err(cuda_err)?;
        let mut context = vec![0.0_f32; (seq * hd) as usize];
        for h in 0..nh as usize {
            for t in 0..seq as usize {
                for d in 0..head_dim as usize {
                    let src = h * (seq as usize * head_dim as usize) + t * head_dim as usize + d;
                    let dst = t * hd as usize + h * head_dim as usize + d;
                    context[dst] = attn_host[src];
                }
            }
        }
        let context_dev = self.stream.clone_htod(&context).map_err(cuda_err)?;

        // Output projection: [seq, hidden] @ [hidden, hidden]^T = [seq, hidden]
        let mut projected = gpu_linear(
            &self.blas,
            &self.stream,
            &context_dev,
            &layer.output_weight,
            seq,
            hd,
            hd,
        )?;

        // Add output bias if present
        if let Some(ref bias) = layer.output_bias {
            let total = seq * hd;
            let mut builder = self.stream.launch_builder(&self.kernels.add_bias);
            builder.arg(&mut projected);
            builder.arg(bias);
            builder.arg(&seq);
            builder.arg(&hd);
            unsafe { builder.launch(launch_cfg_1d(total)) }.map_err(cuda_err)?;
        }

        // Residual connection: projected += hidden
        {
            let total = seq * hd;
            let mut builder = self.stream.launch_builder(&self.kernels.residual_add);
            builder.arg(&mut projected);
            builder.arg(hidden);
            builder.arg(&total);
            unsafe { builder.launch(launch_cfg_1d(total)) }.map_err(cuda_err)?;
        }

        // Layer norm
        let mut normed = self
            .stream
            .alloc_zeros::<f32>((seq * hd) as usize)
            .map_err(cuda_err)?;
        {
            let eps = layer.layer_norm_eps;
            let threads = 256_u32.min(hd as u32).next_power_of_two();
            let shared = threads * 2 * 4;
            let mut builder = self.stream.launch_builder(&self.kernels.layer_norm);
            builder.arg(&mut normed);
            builder.arg(&projected);
            builder.arg(&layer.output_ln_weight);
            builder.arg(&layer.output_ln_bias);
            builder.arg(&seq);
            builder.arg(&hd);
            builder.arg(&eps);
            unsafe { builder.launch(launch_cfg_per_row_shared(seq, threads, shared)) }
                .map_err(cuda_err)?;
        }

        Ok(normed)
    }

    /// FFN forward pass for one layer.
    #[expect(
        unsafe_code,
        clippy::cast_sign_loss,
        reason = "kernel launches require unsafe; dimensions from config are small positive ints"
    )]
    fn forward_ffn(
        &self,
        layer: &CudaBertFfn,
        hidden: &CudaSlice<f32>,
        seq: i32,
    ) -> crate::Result<CudaSlice<f32>> {
        let hd = self.hidden_size;

        // Intermediate projection: [seq, hidden] @ [inter_out, hidden]^T = [seq, inter_out]
        let inter_out_dim = match layer.variant {
            ModelVariant::ClassicBert => layer.intermediate_dim,
            ModelVariant::NomicBert => 2 * layer.intermediate_dim,
        };

        let mut intermediate = gpu_linear(
            &self.blas,
            &self.stream,
            hidden,
            &layer.intermediate_weight,
            seq,
            inter_out_dim,
            hd,
        )?;

        // Add intermediate bias (ClassicBert only)
        if let Some(ref bias) = layer.intermediate_bias {
            let total = seq * inter_out_dim;
            let mut builder = self.stream.launch_builder(&self.kernels.add_bias);
            builder.arg(&mut intermediate);
            builder.arg(bias);
            builder.arg(&seq);
            builder.arg(&inter_out_dim);
            unsafe { builder.launch(launch_cfg_1d(total)) }.map_err(cuda_err)?;
        }

        // Activation
        let activated = match layer.variant {
            ModelVariant::ClassicBert => {
                let total = seq * layer.intermediate_dim;
                let mut builder = self.stream.launch_builder(&self.kernels.gelu);
                builder.arg(&mut intermediate);
                builder.arg(&total);
                unsafe { builder.launch(launch_cfg_1d(total)) }.map_err(cuda_err)?;
                intermediate
            }
            ModelVariant::NomicBert => {
                let half_cols = layer.intermediate_dim;
                let total = seq * half_cols;
                let mut activated = self
                    .stream
                    .alloc_zeros::<f32>(total as usize)
                    .map_err(cuda_err)?;
                let mut builder = self.stream.launch_builder(&self.kernels.swiglu);
                builder.arg(&mut activated);
                builder.arg(&intermediate);
                builder.arg(&seq);
                builder.arg(&half_cols);
                unsafe { builder.launch(launch_cfg_1d(total)) }.map_err(cuda_err)?;
                activated
            }
        };

        // Output projection: [seq, inter] @ [hidden, inter]^T = [seq, hidden]
        let mut output = gpu_linear(
            &self.blas,
            &self.stream,
            &activated,
            &layer.output_weight,
            seq,
            hd,
            layer.intermediate_dim,
        )?;

        // Add output bias (ClassicBert only)
        if let Some(ref bias) = layer.output_bias {
            let total = seq * hd;
            let mut builder = self.stream.launch_builder(&self.kernels.add_bias);
            builder.arg(&mut output);
            builder.arg(bias);
            builder.arg(&seq);
            builder.arg(&hd);
            unsafe { builder.launch(launch_cfg_1d(total)) }.map_err(cuda_err)?;
        }

        // Residual connection: output += hidden
        {
            let total = seq * hd;
            let mut builder = self.stream.launch_builder(&self.kernels.residual_add);
            builder.arg(&mut output);
            builder.arg(hidden);
            builder.arg(&total);
            unsafe { builder.launch(launch_cfg_1d(total)) }.map_err(cuda_err)?;
        }

        // Layer norm
        let mut normed = self
            .stream
            .alloc_zeros::<f32>((seq * hd) as usize)
            .map_err(cuda_err)?;
        {
            let eps = layer.layer_norm_eps;
            let threads = 256_u32.min(hd as u32).next_power_of_two();
            let shared = threads * 2 * 4;
            let mut builder = self.stream.launch_builder(&self.kernels.layer_norm);
            builder.arg(&mut normed);
            builder.arg(&output);
            builder.arg(&layer.output_ln_weight);
            builder.arg(&layer.output_ln_bias);
            builder.arg(&seq);
            builder.arg(&hd);
            builder.arg(&eps);
            unsafe { builder.launch(launch_cfg_per_row_shared(seq, threads, shared)) }
                .map_err(cuda_err)?;
        }

        Ok(normed)
    }

    /// Full forward pass for a single sequence.
    #[expect(
        unsafe_code,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        reason = "kernel launches require unsafe; token IDs and seq lengths are small positive values"
    )]
    fn forward(&self, encoding: &Encoding) -> crate::Result<Vec<f32>> {
        let seq = encoding.input_ids.len() as i32;
        let hd = self.hidden_size;

        // Build integer input arrays for GPU
        let input_ids: Vec<i32> = encoding.input_ids.iter().map(|&id| id as i32).collect();
        let token_type_ids: Vec<i32> = encoding
            .token_type_ids
            .iter()
            .map(|&id| id as i32)
            .collect();
        let position_ids: Vec<i32> = (0..seq).collect();
        let attn_mask_int: Vec<i32> = encoding.attention_mask.iter().map(|&m| m as i32).collect();

        // Upload to GPU
        let input_ids_dev = self.stream.clone_htod(&input_ids).map_err(cuda_err)?;
        let token_type_ids_dev = self.stream.clone_htod(&token_type_ids).map_err(cuda_err)?;
        let position_ids_dev = self.stream.clone_htod(&position_ids).map_err(cuda_err)?;
        let attn_mask_int_dev = self.stream.clone_htod(&attn_mask_int).map_err(cuda_err)?;

        // Build float attention mask: 0.0 for real tokens, -1e9 for padding
        let mut attn_mask_dev = self
            .stream
            .alloc_zeros::<f32>(seq as usize)
            .map_err(cuda_err)?;
        {
            let mut builder = self.stream.launch_builder(&self.kernels.build_attn_mask);
            builder.arg(&mut attn_mask_dev);
            builder.arg(&attn_mask_int_dev);
            builder.arg(&seq);
            unsafe { builder.launch(launch_cfg_1d(seq)) }.map_err(cuda_err)?;
        }

        // Embeddings
        let mut hidden =
            self.forward_embeddings(&input_ids_dev, &token_type_ids_dev, &position_ids_dev, seq)?;

        // Transformer layers
        for layer in &self.layers {
            let after_attn =
                self.forward_attention(&layer.attention, &hidden, &attn_mask_dev, seq)?;
            hidden = self.forward_ffn(&layer.ffn, &after_attn, seq)?;
        }

        // CLS pooling (first row) + L2 normalize
        let mut cls = self
            .stream
            .alloc_zeros::<f32>(hd as usize)
            .map_err(cuda_err)?;
        {
            let mut builder = self.stream.launch_builder(&self.kernels.cls_pool);
            builder.arg(&mut cls);
            builder.arg(&hidden);
            builder.arg(&hd);
            unsafe { builder.launch(launch_cfg_1d(hd)) }.map_err(cuda_err)?;
        }

        // L2 normalize
        {
            let threads = 256_u32.min(hd as u32).next_power_of_two();
            let shared = threads * 4;
            let mut builder = self.stream.launch_builder(&self.kernels.l2_normalize);
            builder.arg(&mut cls);
            builder.arg(&hd);
            unsafe { builder.launch(launch_cfg_per_row_shared(1, threads, shared)) }
                .map_err(cuda_err)?;
        }

        // Copy result back to host
        let result = self.stream.clone_dtoh(&cls).map_err(cuda_err)?;
        Ok(result)
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
        num_heads: config.num_attention_heads,
        head_dim: config.hidden_size / config.num_attention_heads,
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
        variant: config.variant,
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
        num_heads: config.num_attention_heads,
        head_dim: config.hidden_size / config.num_attention_heads,
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
        variant: config.variant,
        intermediate_dim,
    };

    Ok(CudaBertLayer { attention, ffn })
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
    /// The BERT model with all weights on GPU.
    model: CudaBertModel,
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
// Safety: embed_batch takes &self and all GPU ops serialize through the CUDA
// stream. Concurrent calls from multiple threads are safe because the stream
// guarantees ordered execution.
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
        for i in 0..config.num_hidden_layers {
            let layer = match config.variant {
                ModelVariant::ClassicBert => load_classic_layer_gpu(&stream, &tensors, i, &config)?,
                ModelVariant::NomicBert => load_nomic_layer_gpu(&stream, &tensors, i, &config)?,
            };
            layers.push(layer);
        }

        let model = CudaBertModel {
            stream,
            blas,
            kernels,
            embeddings,
            layers,
            hidden_size,
        };

        Ok(Self {
            model,
            hidden_size,
            max_position_embeddings,
            _module: module,
        })
    }
}

impl EmbedBackend for CudaBackend {
    /// Embed a batch of pre-tokenized inputs using the full BERT forward pass on GPU.
    ///
    /// Runs: embeddings -> N attention+FFN layers -> CLS pooling -> L2 normalize.
    /// Each batch item is processed independently through the GPU pipeline.
    fn embed_batch(&self, encodings: &[Encoding]) -> crate::Result<Vec<Vec<f32>>> {
        if encodings.is_empty() {
            return Ok(vec![]);
        }

        let mut results = Vec::with_capacity(encodings.len());
        for enc in encodings {
            let embedding = self.model.forward(enc)?;
            results.push(embedding);
        }

        Ok(results)
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
