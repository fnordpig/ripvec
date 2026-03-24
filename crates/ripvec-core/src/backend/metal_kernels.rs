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
    float inner = 0.7978845608 * (v + 0.044715 * v * v * v);
    inner = clamp(inner, -10.0f, 10.0f);
    x[i] = 0.5 * v * (1.0 + tanh(inner));
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
    // Clamp tanh argument to [-10, 10] to avoid NaN from GPU tanh
    // approximation on large inputs. For |v| > ~5, GELU ≈ v (positive)
    // or 0 (negative), so clamping tanh to ±1 is mathematically exact.
    float inner = 0.7978845608 * (v + 0.044715 * v * v * v);
    inner = clamp(inner, -10.0f, 10.0f);
    x[idx] = 0.5 * v * (1.0 + tanh(inner));
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

// ---------------------------------------------------------------------------
// Two-input SwiGLU: output[i] = value[i] * silu(gate[i])
// Takes separate value and gate buffers (for NomicBert with separate fc11/fc12 weights).
// Output is written to the output buffer (may alias value or gate).
// ---------------------------------------------------------------------------
kernel void swiglu_two_input_kernel(
    device float* output           [[buffer(0)]],
    const device float* value      [[buffer(1)]],
    const device float* gate       [[buffer(2)]],
    constant int& n                [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= uint(n)) return;
    float g = gate[i];
    float silu_g = g / (1.0 + exp(-g));
    output[i] = value[i] * silu_g;
}

// ---------------------------------------------------------------------------
// Add bias: x[idx] += bias[idx % cols]  (in-place)
// ---------------------------------------------------------------------------
kernel void add_bias_kernel(
    device float* x                [[buffer(0)]],
    const device float* bias       [[buffer(1)]],
    constant int& rows             [[buffer(2)]],
    constant int& cols             [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= uint(rows * cols)) return;
    x[idx] += bias[int(idx) % cols];
}

// ---------------------------------------------------------------------------
// Head reshape: [batch*seq, hidden] -> [batch*num_heads, seq, head_dim]
// Used for ClassicBert where Q/K/V are produced by separate GEMMs.
// ---------------------------------------------------------------------------
kernel void head_reshape_kernel(
    device float* output           [[buffer(0)]],
    const device float* input      [[buffer(1)]],
    constant int& batch            [[buffer(2)]],
    constant int& seq              [[buffer(3)]],
    constant int& num_heads        [[buffer(4)]],
    constant int& head_dim         [[buffer(5)]],
    uint idx [[thread_position_in_grid]]
) {
    int hidden = num_heads * head_dim;
    int total = batch * num_heads * seq * head_dim;
    if (idx >= uint(total)) return;

    int per_batch = num_heads * seq * head_dim;
    int b = int(idx) / per_batch;
    int rem0 = int(idx) % per_batch;
    int h = rem0 / (seq * head_dim);
    int rem1 = rem0 % (seq * head_dim);
    int t = rem1 / head_dim;
    int d = rem1 % head_dim;

    // Source: [batch*seq, hidden] = [b*seq + t, h*head_dim + d]
    int src_idx = (b * seq + t) * hidden + h * head_dim + d;
    output[idx] = input[src_idx];
}
";

/// MSL GEMM kernel using `simdgroup_matrix_multiply_accumulate`.
///
/// Computes C\[M,N\] = A\[M,K\] * B\[K,N\] (with optional B transpose) using
/// Apple Silicon's hardware 8x8 matrix multiply units. Each SIMD group
/// (32 threads) computes one 8x8 output tile, with 16 SIMD groups per
/// threadgroup producing a 32x32 output tile.
///
/// Includes the `metal_simdgroup_matrix_storage` header from the
/// [metal-flash-attention](https://github.com/philipturner/metal-flash-attention)
/// project (MIT licensed, Copyright 2024 Philip Turner).
pub const GEMM_KERNEL: &str = r"
#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Begin MFA simdgroup_matrix_storage header (MIT licensed)
// Copyright (c) 2024 Philip Turner
// ---------------------------------------------------------------------------

// -*- Metal -*-
//===-- metal_simdgroup_matrix_storage ------------------------------------===//
// Copyright (c) 2024 Philip Turner. See MIT LICENSE
//===----------------------------------------------------------------------===//

#ifndef __METAL_SIMDGROUP_MATRIX_STORAGE
#define __METAL_SIMDGROUP_MATRIX_STORAGE

// The layout of threads within a SIMD matrix.
//
//  0  0  1  1  8  8  9  9
//  2  2  3  3 10 10 11 11
//  4  4  5  5 12 12 13 13
//  6  6  7  7 14 14 15 15
// 16 16 17 17 24 24 25 25
// 18 18 19 19 26 26 27 27
// 20 20 21 21 28 28 29 29
// 22 22 23 23 30 30 31 31
//
// This is Morton order, a method for coalescing data accesses.
//
// Source: https://patents.google.com/patent/US11256518B2
METAL_FUNC static ushort2 morton_order(ushort thread_index_in_simdgroup) {
  ushort lane_id = thread_index_in_simdgroup;
  ushort quad_id = lane_id / 4;

  constexpr ushort QUADRANT_SPAN_M = 4;
  constexpr ushort THREADS_PER_QUADRANT = 8;
  ushort M_floor_of_quadrant = (quad_id / 4) * QUADRANT_SPAN_M;
  ushort M_in_quadrant = (lane_id / 2) % (THREADS_PER_QUADRANT / 2);
  ushort M_in_simd = M_floor_of_quadrant + M_in_quadrant;

  ushort N_floor_of_quadrant = (quad_id & 2) * 2; // 0 or 4
  ushort N_in_quadrant = (lane_id % 2) * 2; // 0 or 2
  ushort N_in_simd = N_floor_of_quadrant + N_in_quadrant;

  return ushort2(N_in_simd, M_in_simd);
}

#pragma METAL internals : enable
namespace metal
{
  template <typename T>
  struct simdgroup_matrix_storage {
    typedef vec<T, 64> storage_type;
    storage_type t;

    METAL_FUNC thread vec<T, 2>* thread_elements() thread {
      return reinterpret_cast<thread vec<T, 2>*>(&t);
    }

    METAL_FUNC simdgroup_matrix_storage() thread = default;

    METAL_FUNC simdgroup_matrix_storage(vec<T, 2> thread_elements) thread {
      *(this->thread_elements()) = thread_elements;
    }

    METAL_FUNC static device T* apply_offset(device T *src, uint elements_per_row, uint2 matrix_origin, bool transpose_matrix = false) {
      if (transpose_matrix) {
        return src + ulong(matrix_origin.x * elements_per_row) + matrix_origin.y;
      } else {
        return src + ulong(matrix_origin.y * elements_per_row) + matrix_origin.x;
      }
    }

    METAL_FUNC static threadgroup T* apply_offset(threadgroup T *src, ushort elements_per_row, ushort2 matrix_origin, bool transpose_matrix = false) {
      if (transpose_matrix) {
        return src + matrix_origin.x * elements_per_row + matrix_origin.y;
      } else {
        return src + matrix_origin.y * elements_per_row + matrix_origin.x;
      }
    }

    // load (device, non-bf16)
    template <typename U>
    METAL_FUNC void load(const device U *src, uint elements_per_row, ushort2 matrix_origin, bool transpose_matrix = false) {
      if (transpose_matrix) {
        uint address0 = uint(matrix_origin.x + 0) * elements_per_row + uint(matrix_origin.y);
        uint address1 = uint(matrix_origin.x + 1) * elements_per_row + uint(matrix_origin.y);
        U memoryForm0 = src[address0];
        U memoryForm1 = src[address1];
        ((thread T*)thread_elements())[0] = T(memoryForm0);
        ((thread T*)thread_elements())[1] = T(memoryForm1);
      } else if (elements_per_row % 2 != 0) {
        uint address0 = uint(matrix_origin.y) * elements_per_row + uint(matrix_origin.x + 0);
        uint address1 = uint(matrix_origin.y) * elements_per_row + uint(matrix_origin.x + 1);
        U memoryForm0 = src[address0];
        U memoryForm1 = src[address1];
        ((thread T*)thread_elements())[0] = T(memoryForm0);
        ((thread T*)thread_elements())[1] = T(memoryForm1);
      } else {
        auto combinedAddress = uint(matrix_origin.y) * elements_per_row + uint(matrix_origin.x + 0);
        vec<U, 2> memoryForm = *(const device vec<U, 2>*)(src + combinedAddress);
        *(thread_elements()) = vec<T, 2>(memoryForm);
      }
    }

    // load (device, bf16)
    // WARNING: 'T' must be 'float'.
    METAL_FUNC void load_bfloat(const device bfloat *src, uint elements_per_row, ushort2 matrix_origin, bool transpose_matrix = false) {
      if (transpose_matrix) {
        uint address0 = uint(matrix_origin.x + 0) * elements_per_row + uint(matrix_origin.y);
        uint address1 = uint(matrix_origin.x + 1) * elements_per_row + uint(matrix_origin.y);
        bfloat memoryForm0 = src[address0];
        bfloat memoryForm1 = src[address1];

        bfloat4 registerForm = *(thread bfloat4*)(thread_elements());
        registerForm[1] = memoryForm0;
        registerForm[3] = memoryForm1;
        ((thread bfloat4*)thread_elements())[0] = registerForm;
      } else {
        auto combinedAddress = uint(matrix_origin.y) * elements_per_row + uint(matrix_origin.x + 0);
        bfloat2 memoryForm = *(const device packed_bfloat2*)(src + combinedAddress);

        bfloat4 registerForm = *(thread bfloat4*)(thread_elements());
        ((thread float*)&registerForm)[1] = *(thread float*)(&memoryForm);
        ((thread bfloat*)&registerForm)[1] = memoryForm[0];
        ((thread bfloat4*)thread_elements())[0] = registerForm;
      }
    }

    // load (threadgroup, non-bf16)
    template <typename U>
    METAL_FUNC void load(const threadgroup U *src, ushort elements_per_row, ushort2 matrix_origin, bool transpose_matrix = false) {
      if (transpose_matrix) {
        ushort address0 = ushort(matrix_origin.x + 0) * elements_per_row + ushort(matrix_origin.y);
        ushort address1 = ushort(matrix_origin.x + 1) * elements_per_row + ushort(matrix_origin.y);
        U memoryForm0 = src[address0];
        U memoryForm1 = src[address1];
        ((thread T*)thread_elements())[0] = T(memoryForm0);
        ((thread T*)thread_elements())[1] = T(memoryForm1);
      } else if (elements_per_row % 2 != 0) {
        ushort address0 = ushort(matrix_origin.y) * elements_per_row + ushort(matrix_origin.x + 0);
        ushort address1 = ushort(matrix_origin.y) * elements_per_row + ushort(matrix_origin.x + 1);
        U memoryForm0 = src[address0];
        U memoryForm1 = src[address1];
        ((thread T*)thread_elements())[0] = T(memoryForm0);
        ((thread T*)thread_elements())[1] = T(memoryForm1);
      } else {
        auto combinedAddress = ushort(matrix_origin.y) * elements_per_row + ushort(matrix_origin.x + 0);
        vec<U, 2> memoryForm = *(const threadgroup vec<U, 2>*)(src + combinedAddress);
        *(thread_elements()) = vec<T, 2>(memoryForm);
      }
    }

    // load (threadgroup, bf16)
    // WARNING: 'T' must be 'float'.
    METAL_FUNC void load_bfloat(const threadgroup bfloat *src, ushort elements_per_row, ushort2 matrix_origin, bool transpose_matrix = false) {
      if (transpose_matrix) {
        ushort address0 = ushort(matrix_origin.x + 0) * elements_per_row + ushort(matrix_origin.y);
        ushort address1 = ushort(matrix_origin.x + 1) * elements_per_row + ushort(matrix_origin.y);
        bfloat memoryForm0 = src[address0];
        bfloat memoryForm1 = src[address1];

        bfloat4 registerForm = *(thread bfloat4*)(thread_elements());
        registerForm[1] = memoryForm0;
        registerForm[3] = memoryForm1;
        ((thread bfloat4*)thread_elements())[0] = registerForm;
      } else {
        auto combinedAddress = ushort(matrix_origin.y) * elements_per_row + ushort(matrix_origin.x + 0);
        bfloat2 memoryForm = *(const threadgroup packed_bfloat2*)(src + combinedAddress);

        bfloat4 registerForm = *(thread bfloat4*)(thread_elements());
        ((thread float*)&registerForm)[1] = *(thread float*)(&memoryForm);
        ((thread bfloat*)&registerForm)[1] = memoryForm[0];
        ((thread bfloat4*)thread_elements())[0] = registerForm;
      }
    }

    // store (device, non-bf16)
    template <typename U>
    METAL_FUNC void store(device U *dst, uint elements_per_row, ushort2 matrix_origin, bool transpose_matrix = false) {
      if (transpose_matrix) {
        uint address0 = uint(matrix_origin.x + 0) * elements_per_row + uint(matrix_origin.y);
        uint address1 = uint(matrix_origin.x + 1) * elements_per_row + uint(matrix_origin.y);
        T registerForm0 = ((thread T*)thread_elements())[0];
        T registerForm1 = ((thread T*)thread_elements())[1];
        dst[address0] = U(registerForm0);
        dst[address1] = U(registerForm1);
      } else if (elements_per_row % 2 != 0) {
        uint address0 = uint(matrix_origin.y) * elements_per_row + uint(matrix_origin.x + 0);
        uint address1 = uint(matrix_origin.y) * elements_per_row + uint(matrix_origin.x + 1);
        T registerForm0 = ((thread T*)thread_elements())[0];
        T registerForm1 = ((thread T*)thread_elements())[1];
        dst[address0] = U(registerForm0);
        dst[address1] = U(registerForm1);
      } else {
        auto combinedAddress = uint(matrix_origin.y) * elements_per_row + uint(matrix_origin.x + 0);
        vec<T, 2> registerForm = *(thread_elements());
        *(device vec<U, 2>*)(dst + combinedAddress) = vec<U, 2>(registerForm);
      }
    }

    // store (device, bf16)
    // WARNING: 'T' must be 'float'.
    METAL_FUNC void store_bfloat(device bfloat *dst, uint elements_per_row, ushort2 matrix_origin, bool transpose_matrix = false) {
      if (transpose_matrix) {
        uint address0 = uint(matrix_origin.x + 0) * elements_per_row + uint(matrix_origin.y);
        uint address1 = uint(matrix_origin.x + 1) * elements_per_row + uint(matrix_origin.y);
        bfloat4 registerForm = *(thread bfloat4*)(thread_elements());
        registerForm[2] = registerForm[1];
        dst[address0] = registerForm[2];
        dst[address1] = registerForm[3];
      } else {
        uint address0 = uint(matrix_origin.y) * elements_per_row + uint(matrix_origin.x + 0);
        uint address1 = uint(matrix_origin.y) * elements_per_row + uint(matrix_origin.x + 1);
        bfloat4 registerForm = *(thread bfloat4*)(thread_elements());
        registerForm[2] = registerForm[1];
        dst[address0] = registerForm[2];
        dst[address1] = registerForm[3];
      }
    }

    // store (threadgroup, non-bf16)
    template <typename U>
    METAL_FUNC void store(threadgroup U *dst, ushort elements_per_row, ushort2 matrix_origin, bool transpose_matrix = false) {
      if (transpose_matrix) {
        ushort address0 = ushort(matrix_origin.x + 0) * elements_per_row + ushort(matrix_origin.y);
        ushort address1 = ushort(matrix_origin.x + 1) * elements_per_row + ushort(matrix_origin.y);
        T registerForm0 = ((thread T*)thread_elements())[0];
        T registerForm1 = ((thread T*)thread_elements())[1];
        dst[address0] = U(registerForm0);
        dst[address1] = U(registerForm1);
      } else if (elements_per_row % 2 != 0) {
        ushort address0 = ushort(matrix_origin.y) * elements_per_row + ushort(matrix_origin.x + 0);
        ushort address1 = ushort(matrix_origin.y) * elements_per_row + ushort(matrix_origin.x + 1);
        T registerForm0 = ((thread T*)thread_elements())[0];
        T registerForm1 = ((thread T*)thread_elements())[1];
        dst[address0] = U(registerForm0);
        dst[address1] = U(registerForm1);
      } else {
        auto combinedAddress = ushort(matrix_origin.y) * elements_per_row + ushort(matrix_origin.x + 0);
        vec<T, 2> registerForm = *(thread_elements());
        *(threadgroup vec<U, 2>*)(dst + combinedAddress) = vec<U, 2>(registerForm);
      }
    }

    // store (threadgroup, bf16)
    // WARNING: 'T' must be 'float'.
    METAL_FUNC void store_bfloat(threadgroup bfloat *dst, ushort elements_per_row, ushort2 matrix_origin, bool transpose_matrix = false) {
      if (transpose_matrix) {
        ushort address0 = ushort(matrix_origin.x + 0) * elements_per_row + ushort(matrix_origin.y);
        ushort address1 = ushort(matrix_origin.x + 1) * elements_per_row + ushort(matrix_origin.y);
        bfloat4 registerForm = *(thread bfloat4*)(thread_elements());
        registerForm[2] = registerForm[1];
        dst[address0] = registerForm[2];
        dst[address1] = registerForm[3];
      } else {
        ushort address0 = ushort(matrix_origin.y) * elements_per_row + ushort(matrix_origin.x + 0);
        ushort address1 = ushort(matrix_origin.y) * elements_per_row + ushort(matrix_origin.x + 1);
        bfloat4 registerForm = *(thread bfloat4*)(thread_elements());
        registerForm[2] = registerForm[1];
        dst[address0] = registerForm[2];
        dst[address1] = registerForm[3];
      }
    }

    template <typename U, typename V>
    METAL_FUNC void multiply(simdgroup_matrix_storage<U> a, simdgroup_matrix_storage<V> b, bool accumulate = true) {
      if (!accumulate) {
        *(thread_elements()) = vec<T, 2>(0);
      }
      t = __metal_simdgroup_matrix_8x8_multiply_accumulate(a.t, b.t, t, typename simdgroup_matrix_storage<T>::storage_type());
    }
  };
} // namespace metal
#pragma METAL internals : disable

#endif // __METAL_SIMDGROUP_MATRIX_STORAGE

// ---------------------------------------------------------------------------
// End MFA simdgroup_matrix_storage header
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// GEMM kernel: C[M,N] = A[M,K] * B[K,N] (or A * B^T when transB=true)
//
// Tile sizes: 32x32 output per threadgroup, 8x8 per SIMD group.
// Each threadgroup has 16 SIMD groups (4 rows x 4 cols of 8x8 tiles).
// K dimension is tiled in steps of 8.
//
// Key design: each thread uses morton_order() to compute its unique
// position within the 8x8 tile. apply_offset() is called with the
// Morton offset so each thread gets a different base pointer. Then
// load() with origin (k, 0) reads the correct 2 elements for this
// thread's position in the tile.
// ---------------------------------------------------------------------------

constant uint TILE_M = 32;
constant uint TILE_N = 32;

kernel void gemm_kernel(
    device float* A          [[buffer(0)]],
    device float* B          [[buffer(1)]],
    device float* C          [[buffer(2)]],
    constant uint& M         [[buffer(3)]],
    constant uint& N         [[buffer(4)]],
    constant uint& K         [[buffer(5)]],
    constant uint& transB    [[buffer(6)]],
    uint2 tg_pos             [[threadgroup_position_in_grid]],
    ushort simd_id           [[simdgroup_index_in_threadgroup]],
    ushort lane_id           [[thread_index_in_simdgroup]]
) {
    // Per-thread Morton-order position within the 8x8 tile
    ushort2 morton_offset = morton_order(lane_id);

    // SIMD group layout within the 32x32 threadgroup tile:
    // 4 rows x 4 cols of 8x8 sub-tiles -> 16 SIMD groups
    ushort2 sid(simd_id % (TILE_N / 8), simd_id / (TILE_N / 8));

    uint M_offset = tg_pos.y * TILE_M;
    uint N_offset = tg_pos.x * TILE_N;

    // Bounds check: skip SIMD groups entirely outside the matrix
    if (M_offset + sid.y * 8 >= M || N_offset + sid.x * 8 >= N) return;

    // Per-thread offset within the threadgroup tile (includes Morton order)
    ushort2 offset_in_group(sid.x * 8 + morton_offset.x,
                            sid.y * 8 + morton_offset.y);

    // Per-thread global position for bounds checking
    uint my_row = M_offset + offset_in_group.y;
    uint my_col = N_offset + offset_in_group.x;

    // Initialize accumulator to zero
    simdgroup_matrix_storage<float> C_acc;
    *(C_acc.thread_elements()) = float2(0.0);

    // Tile over K dimension in steps of 8
    for (uint k = 0; k < K; k += 8) {
        simdgroup_matrix_storage<float> A_tile, B_tile;

        // --- Load A tile ---
        // A is [M, K] row-major. We want the 8x8 block at (global_row, k).
        if (my_row < M) {
            uint2 A_offset(k + morton_offset.x, my_row);
            device float* A_src = simdgroup_matrix_storage<float>::apply_offset(
                A, K, A_offset);
            A_tile.load(A_src, K, ushort2(0, 0));
        } else {
            *(A_tile.thread_elements()) = float2(0.0);
        }

        // --- Load B tile ---
        if (transB) {
            if (my_col < N) {
                uint2 B_offset(my_col, k + morton_offset.y);
                device float* B_src = simdgroup_matrix_storage<float>::apply_offset(
                    B, K, B_offset, true);
                B_tile.load(B_src, K, ushort2(0, 0), true);
            } else {
                *(B_tile.thread_elements()) = float2(0.0);
            }
        } else {
            if (my_col < N) {
                uint2 B_offset(my_col, k + morton_offset.y);
                device float* B_src = simdgroup_matrix_storage<float>::apply_offset(
                    B, N, B_offset);
                B_tile.load(B_src, N, ushort2(0, 0));
            } else {
                *(B_tile.thread_elements()) = float2(0.0);
            }
        }

        // 8x8 multiply-accumulate: C_acc += A_tile * B_tile
        C_acc.multiply(A_tile, B_tile, true);
    }

    // Store the result tile — skip out-of-bounds threads
    if (my_row < M && my_col < N) {
        uint2 C_offset(my_col, my_row);
        device float* C_dst = simdgroup_matrix_storage<float>::apply_offset(
            C, N, C_offset);
        C_acc.store(C_dst, N, ushort2(0, 0));
    }
}

// ---------------------------------------------------------------------------
// Batched GEMM: same as gemm_kernel but with a batch dimension in grid.z.
// Each batch slice uses strided offsets: A + batch*stride_A, etc.
// Eliminates per-head dispatch loop in attention (12 dispatches → 1).
// ---------------------------------------------------------------------------

kernel void gemm_batched_kernel(
    device float* A          [[buffer(0)]],
    device float* B          [[buffer(1)]],
    device float* C          [[buffer(2)]],
    constant uint& M         [[buffer(3)]],
    constant uint& N         [[buffer(4)]],
    constant uint& K         [[buffer(5)]],
    constant uint& transB    [[buffer(6)]],
    constant uint& stride_A  [[buffer(7)]],   // elements between batch slices of A
    constant uint& stride_B  [[buffer(8)]],   // elements between batch slices of B
    constant uint& stride_C  [[buffer(9)]],   // elements between batch slices of C
    uint3 tg_pos             [[threadgroup_position_in_grid]],
    ushort simd_id           [[simdgroup_index_in_threadgroup]],
    ushort lane_id           [[thread_index_in_simdgroup]]
) {
    uint batch_idx = tg_pos.z;
    device float* A_batch = A + batch_idx * stride_A;
    device float* B_batch = B + batch_idx * stride_B;
    device float* C_batch = C + batch_idx * stride_C;

    ushort2 morton_offset = morton_order(lane_id);
    ushort2 sid(simd_id % (TILE_N / 8), simd_id / (TILE_N / 8));

    uint M_offset = tg_pos.y * TILE_M;
    uint N_offset = tg_pos.x * TILE_N;

    if (M_offset + sid.y * 8 >= M || N_offset + sid.x * 8 >= N) return;

    ushort2 offset_in_group(sid.x * 8 + morton_offset.x,
                            sid.y * 8 + morton_offset.y);
    uint my_row = M_offset + offset_in_group.y;
    uint my_col = N_offset + offset_in_group.x;

    simdgroup_matrix_storage<float> C_acc;
    *(C_acc.thread_elements()) = float2(0.0);

    for (uint k = 0; k < K; k += 8) {
        simdgroup_matrix_storage<float> A_tile, B_tile;

        if (my_row < M) {
            uint2 A_off(k + morton_offset.x, my_row);
            device float* A_src = simdgroup_matrix_storage<float>::apply_offset(
                A_batch, K, A_off);
            A_tile.load(A_src, K, ushort2(0, 0));
        } else {
            *(A_tile.thread_elements()) = float2(0.0);
        }

        if (transB) {
            if (my_col < N) {
                uint2 B_off(my_col, k + morton_offset.y);
                device float* B_src = simdgroup_matrix_storage<float>::apply_offset(
                    B_batch, K, B_off, true);
                B_tile.load(B_src, K, ushort2(0, 0), true);
            } else {
                *(B_tile.thread_elements()) = float2(0.0);
            }
        } else {
            if (my_col < N) {
                uint2 B_off(my_col, k + morton_offset.y);
                device float* B_src = simdgroup_matrix_storage<float>::apply_offset(
                    B_batch, N, B_off);
                B_tile.load(B_src, N, ushort2(0, 0));
            } else {
                *(B_tile.thread_elements()) = float2(0.0);
            }
        }

        C_acc.multiply(A_tile, B_tile, true);
    }

    if (my_row < M && my_col < N) {
        uint2 C_off(my_col, my_row);
        device float* C_dst = simdgroup_matrix_storage<float>::apply_offset(
            C_batch, N, C_off);
        C_acc.store(C_dst, N, ushort2(0, 0));
    }
}

// ---------------------------------------------------------------------------
// FP16 mixed-precision GEMM: A (FP32 activations → half) × B (half weights) → C (FP32)
//
// Same tile/dispatch structure as gemm_kernel but:
//   - B buffer is device half* (pre-converted weights)
//   - A is loaded as float then narrowed to half for simdgroup multiply
//   - Accumulator C stays float for numerical stability
//   - half×half→float matmul uses hardware FP16 units
//
// Register budget per thread: A_tile(half,128B) + B_tile(half,128B) +
// C_acc(float,256B) = 512B vs 768B for all-FP32 → ~1.5× occupancy.
// ---------------------------------------------------------------------------

kernel void gemm_fp16_kernel(
    device float* A          [[buffer(0)]],   // activations [M, K] in FP32
    device half* B           [[buffer(1)]],    // weights [N, K] or [K, N] in FP16
    device float* C          [[buffer(2)]],   // output [M, N] in FP32
    constant uint& M         [[buffer(3)]],
    constant uint& N         [[buffer(4)]],
    constant uint& K         [[buffer(5)]],
    constant uint& transB    [[buffer(6)]],
    uint2 tg_pos             [[threadgroup_position_in_grid]],
    ushort simd_id           [[simdgroup_index_in_threadgroup]],
    ushort lane_id           [[thread_index_in_simdgroup]]
) {
    ushort2 morton_offset = morton_order(lane_id);
    ushort2 sid(simd_id % (TILE_N / 8), simd_id / (TILE_N / 8));

    uint M_offset = tg_pos.y * TILE_M;
    uint N_offset = tg_pos.x * TILE_N;

    if (M_offset + sid.y * 8 >= M || N_offset + sid.x * 8 >= N) return;

    ushort2 offset_in_group(sid.x * 8 + morton_offset.x,
                            sid.y * 8 + morton_offset.y);
    uint my_row = M_offset + offset_in_group.y;
    uint my_col = N_offset + offset_in_group.x;

    simdgroup_matrix_storage<float> C_acc;
    *(C_acc.thread_elements()) = float2(0.0);

    for (uint k = 0; k < K; k += 8) {
        simdgroup_matrix_storage<half> A_tile, B_tile;

        // Load A (FP32 activations) directly into half tile — the templated
        // load<float> reads 2 floats and converts to half in-register,
        // avoiding an intermediate simdgroup_matrix_storage<float>.
        if (my_row < M) {
            uint2 A_off(k + morton_offset.x, my_row);
            device float* A_src = simdgroup_matrix_storage<float>::apply_offset(
                A, K, A_off);
            A_tile.load(A_src, K, ushort2(0, 0));
        } else {
            *(A_tile.thread_elements()) = half2(0.0h);
        }

        // Load B (FP16 weights) directly
        if (transB) {
            if (my_col < N) {
                uint2 B_off(my_col, k + morton_offset.y);
                device half* B_src = simdgroup_matrix_storage<half>::apply_offset(
                    B, K, B_off, true);
                B_tile.load(B_src, K, ushort2(0, 0), true);
            } else {
                *(B_tile.thread_elements()) = half2(0.0h);
            }
        } else {
            if (my_col < N) {
                uint2 B_off(my_col, k + morton_offset.y);
                device half* B_src = simdgroup_matrix_storage<half>::apply_offset(
                    B, N, B_off);
                B_tile.load(B_src, N, ushort2(0, 0));
            } else {
                *(B_tile.thread_elements()) = half2(0.0h);
            }
        }

        // half × half → float accumulation
        C_acc.multiply(A_tile, B_tile, true);
    }

    if (my_row < M && my_col < N) {
        uint2 C_off(my_col, my_row);
        device float* C_dst = simdgroup_matrix_storage<float>::apply_offset(
            C, N, C_off);
        C_acc.store(C_dst, N, ushort2(0, 0));
    }
}

// ---------------------------------------------------------------------------
// Batched FP16 mixed-precision GEMM: same as gemm_fp16_kernel but with batch
// dimension in grid.z. A strides are in float elements, B strides are in half
// elements, C strides are in float elements.
// ---------------------------------------------------------------------------

kernel void gemm_batched_fp16_kernel(
    device float* A          [[buffer(0)]],   // activations in FP32
    device half* B           [[buffer(1)]],    // weights in FP16
    device float* C          [[buffer(2)]],   // output in FP32
    constant uint& M         [[buffer(3)]],
    constant uint& N         [[buffer(4)]],
    constant uint& K         [[buffer(5)]],
    constant uint& transB    [[buffer(6)]],
    constant uint& stride_A  [[buffer(7)]],   // float elements between batch slices of A
    constant uint& stride_B  [[buffer(8)]],   // half elements between batch slices of B
    constant uint& stride_C  [[buffer(9)]],   // float elements between batch slices of C
    uint3 tg_pos             [[threadgroup_position_in_grid]],
    ushort simd_id           [[simdgroup_index_in_threadgroup]],
    ushort lane_id           [[thread_index_in_simdgroup]]
) {
    uint batch_idx = tg_pos.z;
    device float* A_batch = A + batch_idx * stride_A;
    device half* B_batch = B + batch_idx * stride_B;
    device float* C_batch = C + batch_idx * stride_C;

    ushort2 morton_offset = morton_order(lane_id);
    ushort2 sid(simd_id % (TILE_N / 8), simd_id / (TILE_N / 8));

    uint M_offset = tg_pos.y * TILE_M;
    uint N_offset = tg_pos.x * TILE_N;

    if (M_offset + sid.y * 8 >= M || N_offset + sid.x * 8 >= N) return;

    ushort2 offset_in_group(sid.x * 8 + morton_offset.x,
                            sid.y * 8 + morton_offset.y);
    uint my_row = M_offset + offset_in_group.y;
    uint my_col = N_offset + offset_in_group.x;

    simdgroup_matrix_storage<float> C_acc;
    *(C_acc.thread_elements()) = float2(0.0);

    for (uint k = 0; k < K; k += 8) {
        simdgroup_matrix_storage<half> A_tile, B_tile;

        if (my_row < M) {
            uint2 A_off(k + morton_offset.x, my_row);
            device float* A_src = simdgroup_matrix_storage<float>::apply_offset(
                A_batch, K, A_off);
            A_tile.load(A_src, K, ushort2(0, 0));
        } else {
            *(A_tile.thread_elements()) = half2(0.0h);
        }

        if (transB) {
            if (my_col < N) {
                uint2 B_off(my_col, k + morton_offset.y);
                device half* B_src = simdgroup_matrix_storage<half>::apply_offset(
                    B_batch, K, B_off, true);
                B_tile.load(B_src, K, ushort2(0, 0), true);
            } else {
                *(B_tile.thread_elements()) = half2(0.0h);
            }
        } else {
            if (my_col < N) {
                uint2 B_off(my_col, k + morton_offset.y);
                device half* B_src = simdgroup_matrix_storage<half>::apply_offset(
                    B_batch, N, B_off);
                B_tile.load(B_src, N, ushort2(0, 0));
            } else {
                *(B_tile.thread_elements()) = half2(0.0h);
            }
        }

        C_acc.multiply(A_tile, B_tile, true);
    }

    if (my_row < M && my_col < N) {
        uint2 C_off(my_col, my_row);
        device float* C_dst = simdgroup_matrix_storage<float>::apply_offset(
            C_batch, N, C_off);
        C_acc.store(C_dst, N, ushort2(0, 0));
    }
}

// ---------------------------------------------------------------------------
// Fused FlashAttention with simdgroup hardware matrix multiply.
//
// Q@K^T and P@V use simdgroup_matrix_multiply_accumulate (8×8 tile FMA).
// Scores stay in threadgroup memory. Online softmax streams over K/V tiles.
//
// Grid: (batch*num_heads, ceil(seq/32)) — one threadgroup per (head, Q-tile).
// 16 SIMD groups per threadgroup (4×4 of 8×8 tiles), 512 threads total.
//
// Threadgroup memory budget (head_dim=32): ~17KB of 32KB limit.
// ---------------------------------------------------------------------------

constant uint FA_TQ = 32;   // Q rows per threadgroup
constant uint FA_TK = 32;   // K/V rows per tile
constant uint FA_HD = 32;   // head_dim (BGE-small); extend to 64 for CodeRankEmbed

kernel void flash_attention_kernel(
    device float* Q              [[buffer(0)]],
    device float* K              [[buffer(1)]],
    device float* V              [[buffer(2)]],
    device float* O              [[buffer(3)]],
    const device float* mask     [[buffer(4)]],
    constant uint& batch         [[buffer(5)]],
    constant uint& num_heads     [[buffer(6)]],
    constant uint& seq_len       [[buffer(7)]],
    constant uint& head_dim      [[buffer(8)]],
    constant float& scale        [[buffer(9)]],
    uint2 tg_pos                 [[threadgroup_position_in_grid]],
    ushort simd_id               [[simdgroup_index_in_threadgroup]],
    ushort lane_id               [[thread_index_in_simdgroup]]
) {
    uint head_idx = tg_pos.x;
    uint q_tile_idx = tg_pos.y;
    uint batch_idx = head_idx / num_heads;
    uint qr_start = q_tile_idx * FA_TQ;
    if (qr_start >= seq_len) return;

    // Flat thread ID for cooperative loads (0..511)
    uint tid = uint(simd_id) * 32 + uint(lane_id);
    constexpr uint TPG = 512;

    uint head_stride = seq_len * head_dim;
    device float* Q_head = Q + head_idx * head_stride;
    device float* K_head = K + head_idx * head_stride;
    device float* V_head = V + head_idx * head_stride;
    device float* O_head = O + head_idx * head_stride;
    const device float* mask_batch = mask + batch_idx * seq_len;

    // SIMD group layout: 4×4 of 8×8 sub-tiles within 32×32 output
    ushort2 morton_off = morton_order(lane_id);
    ushort2 sid(simd_id % 4, simd_id / 4);
    // Per-thread position within the 32×32 tile
    ushort2 tile_pos(sid.x * 8 + morton_off.x, sid.y * 8 + morton_off.y);

    // Threadgroup memory (~17KB for head_dim=32)
    threadgroup float q_tg[FA_TQ * FA_HD];       // Q tile [32, 32] = 4KB
    threadgroup float kv_tg[FA_TK * FA_HD];      // K then V [32, 32] = 4KB (shared)
    threadgroup float s_tg[FA_TQ * FA_TK];       // scores [32, 32] = 4KB
    threadgroup float o_tg[FA_TQ * FA_HD];       // output [32, 32] = 4KB
    threadgroup float m_tg[FA_TQ];               // row max [32] = 128B
    threadgroup float l_tg[FA_TQ];               // row sum [32] = 128B
    threadgroup float rescale_tg[FA_TQ];         // rescale [32] = 128B
    threadgroup float inv_l_tg[FA_TQ];           // 1/l [32] = 128B

    // --- Init ---
    for (uint i = tid; i < FA_TQ; i += TPG) { m_tg[i] = -1e30; l_tg[i] = 0.0; }
    for (uint i = tid; i < FA_TQ * FA_HD; i += TPG) { o_tg[i] = 0.0; }

    // --- Load Q tile (once) ---
    for (uint i = tid; i < FA_TQ * head_dim; i += TPG) {
        uint row = i / head_dim, col = i % head_dim;
        uint gr = qr_start + row;
        q_tg[i] = (gr < seq_len) ? Q_head[gr * head_dim + col] : 0.0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // === K/V tile loop ===
    for (uint kv_start = 0; kv_start < seq_len; kv_start += FA_TK) {

        // --- Load K tile --- (cooperative, all 512 threads)
        for (uint i = tid; i < FA_TK * head_dim; i += TPG) {
            uint row = i / head_dim, col = i % head_dim;
            uint gr = kv_start + row;
            kv_tg[i] = (gr < seq_len) ? K_head[gr * head_dim + col] : 0.0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Phase 1: S = Q @ K^T via simdgroup matmul → s_tg ---
        {
            simdgroup_matrix_storage<float> S_acc;
            *(S_acc.thread_elements()) = float2(0.0);

            for (uint k = 0; k < head_dim; k += 8) {
                simdgroup_matrix_storage<float> Q_tile, K_tile;

                // Load Q[tile_pos.y, k+morton.x] from threadgroup
                threadgroup float* q_src = simdgroup_matrix_storage<float>::apply_offset(
                    q_tg, (ushort)FA_HD, ushort2(ushort(k) + morton_off.x, tile_pos.y));
                Q_tile.load(q_src, (ushort)FA_HD, ushort2(0, 0));

                // Load K^T[k+morton.y, tile_pos.x] — transposed from kv_tg
                threadgroup float* k_src = simdgroup_matrix_storage<float>::apply_offset(
                    kv_tg, (ushort)FA_HD, ushort2(tile_pos.x, ushort(k) + morton_off.y), true);
                K_tile.load(k_src, (ushort)FA_HD, ushort2(0, 0), true);

                S_acc.multiply(Q_tile, K_tile, true);
            }

            // Store S to threadgroup s_tg
            threadgroup float* s_dst = simdgroup_matrix_storage<float>::apply_offset(
                s_tg, (ushort)FA_TK, tile_pos);
            S_acc.store(s_dst, (ushort)FA_TK, ushort2(0, 0));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Phase 1b: apply scale + mask --- (all 512 threads, element-wise)
        for (uint idx = tid; idx < FA_TQ * FA_TK; idx += TPG) {
            uint qi = idx / FA_TK, kj = idx % FA_TK;
            float mask_val = (qr_start + qi < seq_len && kv_start + kj < seq_len)
                           ? mask_batch[kv_start + kj] : -1e30;
            s_tg[idx] = s_tg[idx] * scale + mask_val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Phase 2: Online softmax + rescale factors --- (32 threads, per-row)
        if (tid < FA_TQ && (qr_start + tid) < seq_len) {
            uint qi = tid;
            float old_m = m_tg[qi], old_l = l_tg[qi];

            float row_max = -1e30;
            for (uint kj = 0; kj < FA_TK && kv_start + kj < seq_len; kj++)
                row_max = max(row_max, s_tg[qi * FA_TK + kj]);
            float m_new = max(old_m, row_max);

            float row_sum = 0.0;
            for (uint kj = 0; kj < FA_TK; kj++) {
                float p = (kv_start + kj < seq_len) ? exp(s_tg[qi * FA_TK + kj] - m_new) : 0.0;
                s_tg[qi * FA_TK + kj] = p;
                row_sum += p;
            }

            float correction = exp(old_m - m_new);
            float l_new = correction * old_l + row_sum;
            rescale_tg[qi] = (old_l > 0.0) ? (correction * old_l / max(l_new, 1e-12)) : 0.0;
            inv_l_tg[qi] = 1.0 / max(l_new, 1e-12);
            m_tg[qi] = m_new;
            l_tg[qi] = l_new;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Load V tile --- (overwrites K in kv_tg)
        for (uint i = tid; i < FA_TK * head_dim; i += TPG) {
            uint row = i / head_dim, col = i % head_dim;
            uint gr = kv_start + row;
            kv_tg[i] = (gr < seq_len) ? V_head[gr * head_dim + col] : 0.0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Phase 3: O = rescale*O + inv_l*(P@V) via simdgroup matmul ---
        {
            // P@V: P[32,32] @ V[32,32] → O_contrib[32,32]
            simdgroup_matrix_storage<float> PV_acc;
            *(PV_acc.thread_elements()) = float2(0.0);

            for (uint k = 0; k < FA_TK; k += 8) {
                simdgroup_matrix_storage<float> P_tile, V_tile;

                // Load P from s_tg (softmax output)
                threadgroup float* p_src = simdgroup_matrix_storage<float>::apply_offset(
                    s_tg, (ushort)FA_TK, ushort2(ushort(k) + morton_off.x, tile_pos.y));
                P_tile.load(p_src, (ushort)FA_TK, ushort2(0, 0));

                // Load V from kv_tg (not transposed)
                threadgroup float* v_src = simdgroup_matrix_storage<float>::apply_offset(
                    kv_tg, (ushort)FA_HD, ushort2(tile_pos.x, ushort(k) + morton_off.y));
                V_tile.load(v_src, (ushort)FA_HD, ushort2(0, 0));

                PV_acc.multiply(P_tile, V_tile, true);
            }

            // Load current O, apply: O_new = rescale * O_prev + inv_l * PV_acc
            threadgroup float* o_ptr = simdgroup_matrix_storage<float>::apply_offset(
                o_tg, (ushort)FA_HD, tile_pos);
            simdgroup_matrix_storage<float> O_prev;
            O_prev.load(o_ptr, (ushort)FA_HD, ushort2(0, 0));

            float rs = rescale_tg[tile_pos.y];
            float il = inv_l_tg[tile_pos.y];
            float2 o_old = *(O_prev.thread_elements());
            float2 pv_new = *(PV_acc.thread_elements());
            *(O_prev.thread_elements()) = rs * o_old + il * pv_new;

            O_prev.store(o_ptr, (ushort)FA_HD, ushort2(0, 0));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // --- Write O from threadgroup to global memory --- (cooperative)
    for (uint i = tid; i < FA_TQ * head_dim; i += TPG) {
        uint row = i / head_dim, col = i % head_dim;
        uint gr = qr_start + row;
        if (gr < seq_len)
            O_head[gr * head_dim + col] = o_tg[i];
    }
}
";
