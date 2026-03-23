//! Metal Shading Language (MSL) kernel sources.
//!
//! Contains MSL compute kernels used by the Metal backend. Kernels are compiled
//! at runtime via `MTLDevice::newLibraryWithSource_options_error`.
//!
//! These kernels are ported from the CUDA backend (`cuda.rs`) and cover the
//! full set of element-wise, reduction, and reshape operations needed for BERT
//! inference on Apple Silicon GPUs.

/// Simple test kernel that adds 1.0 to every element of a float buffer.
///
/// Used by integration tests to verify the full Metal pipeline: device creation,
/// MSL compilation, buffer allocation, compute dispatch, and result readback.
pub const TEST_KERNEL: &str = r"
#include <metal_stdlib>
using namespace metal;

kernel void add_one(
    device float *data [[buffer(0)]],
    uint tid [[thread_position_in_grid]]
) {
    data[tid] = data[tid] + 1.0;
}
";

/// MSL kernels for BERT inference operations.
///
/// Compiled once at model load time via `MTLDevice::newLibraryWithSource`, then
/// pipeline states are created for each function. Each kernel mirrors a
/// corresponding CUDA kernel from the CUDA backend.
///
/// # Kernel list
///
/// - `embedding_lookup_kernel` -- index into word embedding table
/// - `add_embeddings_kernel` -- sum word + position + `token_type` embeddings
/// - `layer_norm_kernel` -- threadgroup reduction for mean/variance, normalize
/// - `gelu_kernel` -- tanh approximation GELU activation
/// - `swiglu_kernel` -- split intermediate into value/gate, compute `SwiGLU`
/// - `rope_cached_kernel` -- rotate Q/K using pre-computed cos/sin tables
/// - `fused_scale_mask_softmax_kernel` -- scale + mask + three-pass softmax
/// - `fused_residual_layernorm_kernel` -- residual add + layernorm
/// - `fused_bias_gelu_kernel` -- bias + GELU activation
/// - `fused_bias_residual_kernel` -- bias + residual add
/// - `fused_swiglu_kernel` -- unified bias/no-bias `SwiGLU`
/// - `qkv_split_kernel` -- split QKV into separate Q, K, V tensors
/// - `attn_reshape_kernel` -- reshape multi-head attention output
/// - `cls_pool_kernel` -- extract first token per batch element
/// - `l2_normalize_kernel` -- per-vector L2 normalization
/// - `build_attn_mask_kernel` -- build float attention mask from int mask
/// - `f32_to_f16_kernel` -- convert f32 to f16
pub const KERNELS: &str = r"
#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Embedding lookup: output[idx] = table[indices[token] * hidden_dim + dim]
// ---------------------------------------------------------------------------
kernel void embedding_lookup_kernel(
    device float* output           [[buffer(0)]],
    const device float* table      [[buffer(1)]],
    const device int* indices      [[buffer(2)]],
    constant int& batch_seq        [[buffer(3)]],
    constant int& hidden_dim       [[buffer(4)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= uint(batch_seq * hidden_dim)) return;
    int token = int(idx) / hidden_dim;
    int dim = int(idx) % hidden_dim;
    output[idx] = table[indices[token] * hidden_dim + dim];
}

// ---------------------------------------------------------------------------
// Add embeddings: output[idx] += table[indices[token] * hidden_dim + dim]
// ---------------------------------------------------------------------------
kernel void add_embeddings_kernel(
    device float* output           [[buffer(0)]],
    const device float* table      [[buffer(1)]],
    const device int* indices      [[buffer(2)]],
    constant int& batch_seq        [[buffer(3)]],
    constant int& hidden_dim       [[buffer(4)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= uint(batch_seq * hidden_dim)) return;
    int token = int(idx) / hidden_dim;
    int dim = int(idx) % hidden_dim;
    output[idx] += table[indices[token] * hidden_dim + dim];
}

// ---------------------------------------------------------------------------
// Layer normalization with threadgroup reductions.
// One threadgroup per row. Uses threadgroup memory for sum/variance reduction.
// ---------------------------------------------------------------------------
kernel void layer_norm_kernel(
    device float* output           [[buffer(0)]],
    const device float* input      [[buffer(1)]],
    const device float* weight     [[buffer(2)]],
    const device float* bias       [[buffer(3)]],
    constant int& rows             [[buffer(4)]],
    constant int& cols             [[buffer(5)]],
    constant float& eps            [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]]
) {
    if (row >= uint(rows)) return;

    // Threadgroup shared memory for reductions (2 * tpg floats)
    threadgroup float s_sum[256];
    threadgroup float s_sq[256];

    // Phase 1: partial sums for mean
    float local_sum = 0.0;
    for (int i = int(tid); i < cols; i += int(tpg)) {
        local_sum += input[row * uint(cols) + uint(i)];
    }
    s_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce sum
    for (uint stride = tpg / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = s_sum[0] / float(cols);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: variance
    float local_sq = 0.0;
    for (int i = int(tid); i < cols; i += int(tpg)) {
        float diff = input[row * uint(cols) + uint(i)] - mean;
        local_sq += diff * diff;
    }
    s_sq[tid] = local_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tpg / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sq[tid] += s_sq[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_std = rsqrt(s_sq[0] / float(cols) + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: normalize
    for (int i = int(tid); i < cols; i += int(tpg)) {
        uint idx = row * uint(cols) + uint(i);
        output[idx] = (input[idx] - mean) * inv_std * weight[i] + bias[i];
    }
}

// ---------------------------------------------------------------------------
// GELU activation (tanh approximation, in-place)
// 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// ---------------------------------------------------------------------------
kernel void gelu_kernel(
    device float* x                [[buffer(0)]],
    constant int& n                [[buffer(1)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= uint(n)) return;
    float v = x[i];
    x[i] = 0.5 * v * (1.0 + tanh(0.7978845608 * (v + 0.044715 * v * v * v)));
}

// ---------------------------------------------------------------------------
// SwiGLU: output[i] = value[i] * silu(gate[i])
// value = first half, gate = second half of input
// ---------------------------------------------------------------------------
kernel void swiglu_kernel(
    device float* output           [[buffer(0)]],
    const device float* input      [[buffer(1)]],
    constant int& rows_val         [[buffer(2)]],
    constant int& half_cols        [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    int total = rows_val * half_cols;
    if (idx >= uint(total)) return;
    int row = int(idx) / half_cols;
    int col = int(idx) % half_cols;
    float value = input[row * (2 * half_cols) + col];
    float gate = input[row * (2 * half_cols) + half_cols + col];
    float silu_gate = gate / (1.0 + exp(-gate));
    output[idx] = value * silu_gate;
}

// ---------------------------------------------------------------------------
// RoPE with pre-computed cos/sin tables (NomicBert).
// Rotates q_or_k in-place using cached cos/sin values.
// ---------------------------------------------------------------------------
kernel void rope_cached_kernel(
    device float* q_or_k           [[buffer(0)]],
    const device float* cos_table  [[buffer(1)]],
    const device float* sin_table  [[buffer(2)]],
    constant int& num_rows         [[buffer(3)]],
    constant int& seq              [[buffer(4)]],
    constant int& head_dim         [[buffer(5)]],
    constant int& num_heads        [[buffer(6)]],
    uint idx [[thread_position_in_grid]]
) {
    int half_dim = head_dim / 2;
    int total = num_rows * half_dim;
    if (idx >= uint(total)) return;

    int row = int(idx) / half_dim;
    int d = int(idx) % half_dim;
    int pos = row % seq;

    int first_idx = row * head_dim + d;
    int second_idx = first_idx + half_dim;

    float cos_val = cos_table[pos * half_dim + d];
    float sin_val = sin_table[pos * half_dim + d];

    float first = q_or_k[first_idx];
    float second = q_or_k[second_idx];
    q_or_k[first_idx] = first * cos_val - second * sin_val;
    q_or_k[second_idx] = first * sin_val + second * cos_val;
}

// ---------------------------------------------------------------------------
// Fused scale + attention-mask + softmax.
// Three-pass: (1) scale + mask + find max, (2) exp + sum, (3) normalize.
// One threadgroup per row; threadgroup memory for reductions.
// ---------------------------------------------------------------------------
kernel void fused_scale_mask_softmax_kernel(
    device float* scores           [[buffer(0)]],
    const device float* mask       [[buffer(1)]],
    constant int& batch            [[buffer(2)]],
    constant int& num_heads        [[buffer(3)]],
    constant int& seq              [[buffer(4)]],
    constant float& scale          [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]]
) {
    int total_rows = batch * num_heads * seq;
    if (row >= uint(total_rows)) return;

    threadgroup float sdata[256];
    device float* row_data = scores + row * uint(seq);

    // Decompose row -> batch index for mask lookup
    int b = int(row) / (num_heads * seq);

    // Pass 1: scale + mask + find row max
    float thread_max = -1e30;
    for (int i = int(tid); i < seq; i += int(tpg)) {
        float val = row_data[i] * scale + mask[b * seq + i];
        row_data[i] = val;
        thread_max = max(thread_max, val);
    }

    // Reduce max
    sdata[tid] = thread_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tpg / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = sdata[0];

    // Pass 2: exp(x - max) and sum
    float thread_sum = 0.0;
    for (int i = int(tid); i < seq; i += int(tpg)) {
        float val = exp(row_data[i] - row_max);
        row_data[i] = val;
        thread_sum += val;
    }

    // Reduce sum
    sdata[tid] = thread_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tpg / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float total = sdata[0];

    // Pass 3: normalize
    float inv_sum = 1.0 / max(total, 1e-12);
    for (int i = int(tid); i < seq; i += int(tpg)) {
        row_data[i] *= inv_sum;
    }
}

// ---------------------------------------------------------------------------
// Fused residual add + layer norm.
// output = layernorm(hidden + residual)
// One threadgroup per row.
// ---------------------------------------------------------------------------
kernel void fused_residual_layernorm_kernel(
    device float* output           [[buffer(0)]],
    const device float* hidden     [[buffer(1)]],
    const device float* residual   [[buffer(2)]],
    const device float* weight     [[buffer(3)]],
    const device float* bias       [[buffer(4)]],
    constant int& rows             [[buffer(5)]],
    constant int& cols             [[buffer(6)]],
    constant float& eps            [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]]
) {
    if (row >= uint(rows)) return;

    threadgroup float sdata[256];

    // Pass 1: add residual + compute mean
    float thread_sum = 0.0;
    for (int i = int(tid); i < cols; i += int(tpg)) {
        float val = hidden[row * uint(cols) + uint(i)] + residual[row * uint(cols) + uint(i)];
        output[row * uint(cols) + uint(i)] = val;
        thread_sum += val;
    }
    sdata[tid] = thread_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tpg / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = sdata[0] / float(cols);

    // Pass 2: variance
    float thread_var = 0.0;
    for (int i = int(tid); i < cols; i += int(tpg)) {
        float diff = output[row * uint(cols) + uint(i)] - mean;
        thread_var += diff * diff;
    }
    sdata[tid] = thread_var;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tpg / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_std = rsqrt(sdata[0] / float(cols) + eps);

    // Pass 3: normalize + scale + shift
    for (int i = int(tid); i < cols; i += int(tpg)) {
        uint idx = row * uint(cols) + uint(i);
        float val = (output[idx] - mean) * inv_std;
        output[idx] = val * weight[i] + bias[i];
    }
}

// ---------------------------------------------------------------------------
// Fused bias + GELU activation (ClassicBert FFN).
// ---------------------------------------------------------------------------
kernel void fused_bias_gelu_kernel(
    device float* x                [[buffer(0)]],
    const device float* bias       [[buffer(1)]],
    constant int& rows             [[buffer(2)]],
    constant int& cols             [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= uint(rows * cols)) return;
    int col = int(idx) % cols;
    float v = x[idx] + bias[col];
    x[idx] = 0.5 * v * (1.0 + tanh(0.7978845608 * (v + 0.044715 * v * v * v)));
}

// ---------------------------------------------------------------------------
// Fused bias + residual add for output projections (ClassicBert).
// output = input + bias + residual
// ---------------------------------------------------------------------------
kernel void fused_bias_residual_kernel(
    device float* output           [[buffer(0)]],
    const device float* input      [[buffer(1)]],
    const device float* bias       [[buffer(2)]],
    const device float* residual   [[buffer(3)]],
    constant int& rows             [[buffer(4)]],
    constant int& cols             [[buffer(5)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= uint(rows * cols)) return;
    output[idx] = input[idx] + bias[int(idx) % cols] + residual[idx];
}

// ---------------------------------------------------------------------------
// Unified SwiGLU kernel handling both bias and no-bias paths.
// When has_bias=1, adds bias to value and gate before SwiGLU.
// ---------------------------------------------------------------------------
kernel void fused_swiglu_kernel(
    device float* output           [[buffer(0)]],
    const device float* input      [[buffer(1)]],
    const device float* bias       [[buffer(2)]],
    constant int& rows_val         [[buffer(3)]],
    constant int& out_cols         [[buffer(4)]],
    constant int& has_bias         [[buffer(5)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= uint(rows_val * out_cols)) return;
    int row = int(idx) / out_cols;
    int col = int(idx) % out_cols;

    float value = input[row * 2 * out_cols + col];
    float gate = input[row * 2 * out_cols + out_cols + col];

    if (has_bias) {
        value += bias[col];
        gate += bias[out_cols + col];
    }

    gate = gate / (1.0 + exp(-gate));
    output[idx] = value * gate;
}

// ---------------------------------------------------------------------------
// Split QKV [batch*seq, 3*hidden] into Q,K,V each [batch*num_heads, seq, head_dim].
// ---------------------------------------------------------------------------
kernel void qkv_split_kernel(
    device float* q                [[buffer(0)]],
    device float* k                [[buffer(1)]],
    device float* v                [[buffer(2)]],
    const device float* qkv        [[buffer(3)]],
    constant int& batch            [[buffer(4)]],
    constant int& seq              [[buffer(5)]],
    constant int& hidden           [[buffer(6)]],
    constant int& num_heads        [[buffer(7)]],
    constant int& head_dim         [[buffer(8)]],
    uint idx [[thread_position_in_grid]]
) {
    int total = batch * num_heads * seq * head_dim;
    if (idx >= uint(total)) return;

    int per_batch = num_heads * seq * head_dim;
    int b = int(idx) / per_batch;
    int rem0 = int(idx) % per_batch;
    int h = rem0 / (seq * head_dim);
    int rem1 = rem0 % (seq * head_dim);
    int t = rem1 / head_dim;
    int d = rem1 % head_dim;

    int qkv_idx = b * seq * (3 * hidden) + t * (3 * hidden) + h * head_dim + d;
    q[idx] = qkv[qkv_idx];
    k[idx] = qkv[qkv_idx + hidden];
    v[idx] = qkv[qkv_idx + 2 * hidden];
}

// ---------------------------------------------------------------------------
// Reshape attention output from [batch*num_heads, seq, head_dim] back to
// [batch*seq, hidden].
// ---------------------------------------------------------------------------
kernel void attn_reshape_kernel(
    device float* output           [[buffer(0)]],
    const device float* heads      [[buffer(1)]],
    constant int& batch            [[buffer(2)]],
    constant int& seq              [[buffer(3)]],
    constant int& num_heads        [[buffer(4)]],
    constant int& head_dim         [[buffer(5)]],
    uint idx [[thread_position_in_grid]]
) {
    int hidden = num_heads * head_dim;
    int total = batch * seq * hidden;
    if (idx >= uint(total)) return;

    int b = int(idx) / (seq * hidden);
    int rem = int(idx) % (seq * hidden);
    int t = rem / hidden;
    int flat_hd = rem % hidden;
    int h = flat_hd / head_dim;
    int d = flat_hd % head_dim;

    int heads_idx = (b * num_heads + h) * (seq * head_dim) + t * head_dim + d;
    output[idx] = heads[heads_idx];
}

// ---------------------------------------------------------------------------
// CLS pooling: extract row 0 of each batch element from [batch, seq, hidden].
// output is [batch, hidden].
// ---------------------------------------------------------------------------
kernel void cls_pool_kernel(
    device float* output           [[buffer(0)]],
    const device float* hidden     [[buffer(1)]],
    constant int& batch            [[buffer(2)]],
    constant int& seq              [[buffer(3)]],
    constant int& hidden_dim       [[buffer(4)]],
    uint idx [[thread_position_in_grid]]
) {
    int total = batch * hidden_dim;
    if (idx >= uint(total)) return;
    int b = int(idx) / hidden_dim;
    int d = int(idx) % hidden_dim;
    output[idx] = hidden[b * seq * hidden_dim + d];
}

// ---------------------------------------------------------------------------
// L2 normalize each row of a [rows, cols] matrix in-place.
// One threadgroup per row; threadgroup memory for sum-of-squares reduction.
// ---------------------------------------------------------------------------
kernel void l2_normalize_kernel(
    device float* x                [[buffer(0)]],
    constant int& rows             [[buffer(1)]],
    constant int& cols             [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]]
) {
    if (row >= uint(rows)) return;

    threadgroup float sdata[256];

    float local_sq = 0.0;
    for (int i = int(tid); i < cols; i += int(tpg)) {
        float v = x[row * uint(cols) + uint(i)];
        local_sq += v * v;
    }
    sdata[tid] = local_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tpg / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_norm = rsqrt(max(sdata[0], 1e-12f));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int i = int(tid); i < cols; i += int(tpg)) {
        x[row * uint(cols) + uint(i)] *= inv_norm;
    }
}

// ---------------------------------------------------------------------------
// Build attention mask: 0.0 for real tokens (mask=1), -1e9 for padding (mask=0).
// ---------------------------------------------------------------------------
kernel void build_attn_mask_kernel(
    device float* output           [[buffer(0)]],
    const device int* mask         [[buffer(1)]],
    constant int& total            [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= uint(total)) return;
    output[i] = (mask[i] == 1) ? 0.0 : -1e9;
}

// ---------------------------------------------------------------------------
// Convert f32 to f16 (for future FP16 GEMM support).
// MSL has native half type so no inline asm needed.
// ---------------------------------------------------------------------------
kernel void f32_to_f16_kernel(
    device half* output            [[buffer(0)]],
    const device float* input      [[buffer(1)]],
    constant int& n                [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= uint(n)) return;
    output[i] = half(input[i]);
}
";
