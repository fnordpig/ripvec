# CUDA Architecture Guide

> Internal reference for developers working on ripvec's NVIDIA GPU backend.
> Covers the CudaDriver implementation, GEMM dispatch, INT8 quantization,
> kernel optimization history, and known platform issues.

---

## Overview

The CUDA backend implements the `Driver` trait (`crates/ripvec-core/src/backend/driver/cuda.rs`,
~3,500 lines) enabling `ModernBertArch<CudaTensor>` to run the full 22-layer encoder on
NVIDIA GPUs via cudarc 0.19.4.

**Throughput**: 435 chunks/s on RTX 4090 (React corpus, 20K chunks) — 5.9× faster than
Metal MPS on M2 Max (73.8/s).

## CudaTensor

```rust
pub struct CudaTensor {
    f32_buf: CudaSlice<f32>,           // Primary FP32 buffer
    fp16: Option<CudaSlice<u16>>,      // FP16 companion (GEMM weights)
    int8: Option<CudaSlice<i8>>,       // INT8 quantized weights
    int8_col_scales: Option<CudaSlice<f32>>, // Per-column scale factors
}
```

Weight tensors have all three representations (loaded at startup). Intermediate
activations have only `fp16` (FP16 forward path). The FP32 buffer is a placeholder
for FP16-only tensors.

## GEMM Dispatch

`gemm_f16()` has a two-tier dispatch:

1. **INT8 path** (if `b.int8` is Some): `cublasGemmEx(CUDA_R_8I × CUDA_R_8I → CUDA_R_32I)`
   with per-row activation quantization + per-row/per-col dequantize kernel.
   Currently ~5% slower than FP16 due to dequantize overhead (57K kernel launches).

2. **FP16 path** (default): `cublasGemmEx(CUDA_R_16F × CUDA_R_16F → CUDA_R_16F)`
   with `CUBLAS_COMPUTE_32F` (FP32 accumulation on tensor cores).

Batched attention GEMMs (Q@K^T, scores@V) always use FP16 via `gemm_batched_f16()`.

## Custom CUDA Kernels

~40 kernels compiled via NVRTC at startup. Key optimizations:

| Kernel | Optimization | Impact |
|--------|-------------|--------|
| `fused_scale_mask_softmax_windowed_f16` | Warp shuffle reduction + window-bounds loop | -49% softmax time |
| `fused_split_geglu_f16` | Single-pass split+GeGLU (eliminates 2 intermediate buffers) | -4.3% total |
| `fused_pad_qkv_split_f16` | Direct flat→per-head (eliminates padded intermediate) | -2.3s |
| `fused_reshape_unpad_f16` | Direct per-head→flat (eliminates padded context) | -0.5s |
| `quantize_activation_rowwise` | Fused per-row max-reduce + quantize (was 102s single-block) | Fixed INT8 |
| `dequantize_i32_to_f16` | Per-row kernel with vectorized int4 loads | 7s → 7s (launch-bound) |

All softmax kernels use `__expf()` (fast math) and `block_reduce_max`/`block_reduce_sum`
via warp shuffles (`__shfl_down_sync`).

## Memory Management

**No pool** — cudarc's `CudaSlice::clone()` does full D2D memcpy (unlike Metal's refcounted
buffers). All tensors allocated via `unsafe { stream.alloc() }` (uninitialised, caller overwrites).
The CUDA async memory allocator (`cuMemAllocAsync`) handles pooling internally.

Event tracking disabled (`ctx.disable_event_tracking()`) for single-stream usage.

## Weight Loading

`load_modern_bert_weights()` follows the Metal pattern:
1. mmap safetensors file
2. Parse header → HashMap of tensor offsets
3. H2D copy each tensor via `stream.clone_htod()`
4. Pre-convert GEMM weights to FP16 (`f32_to_f16_kernel`)
5. Pre-quantize GEMM weights to INT8 per-channel (`quantize_col_scales` + `quantize_weights_i8`)
6. Build RoPE cos/sin caches (global θ=160000, local θ=10000)

## Cache Compression

On-disk index objects use zstd level 1 compression (~8× smaller). Transparent
decompression on read (detects zstd magic, falls back to legacy uncompressed rkyv).

## Known Issues

### CUDA 13.1 INT8 Regression
`cublasLtMatmul` with INT8 inputs and INT32 outputs may return `CUBLAS_STATUS_NOT_SUPPORTED`
when batch_count > 1 or N > 65536. Our projection GEMMs are unbatched with N ≤ 4608,
so we use `cublasGemmEx` instead. The cuBLASLt `A_SCALE_POINTER`/`B_SCALE_POINTER` path
(which would eliminate the dequantize kernel entirely) is blocked by this bug.
Fix expected in CUDA 13.2.

### NVRTC Arch Flag
Use `compute_XX` (virtual arch, forward-compatible PTX) not `sm_XX` (real arch).
CUDA 13+ dropped support for old SM targets. Compute capability detected at runtime
via `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR/MINOR`.

## Optimization History

| Phase | Change | Throughput | Delta |
|-------|--------|-----------|-------|
| Phase 1 | CudaDriver functional | 59.0/s | baseline |
| Phase 2A | Remove pool D2D clone | 382.2/s | +548% |
| Phase 2C | Warp shuffle softmax | 414.1/s | +8.3% |
| Phase 2D | Fused split+geglu | 431.8/s | +4.3% |
| Phase 2E | Fused pad+qkv, reshape+unpad | 435.2/s | +0.8% |
| INT8 (investigated) | cublasGemmEx I8→I32 + dequant | 410/s | -5% (overhead) |

## Future Work

1. **CUDA 13.2 upgrade** → cuBLASLt `A_SCALE_POINTER`/`B_SCALE_POINTER` eliminates
   dequantize kernel, projected ~530/s
2. **cuDNN fused attention** → eliminates softmax + 2 batched GEMMs (~25% of GPU time)
3. **CUDA Graphs** → eliminates kernel launch overhead (~4.3s / 8.8% of wall time)
4. **Parallel tokenization** in streaming pipeline → single-threaded tokenizer is the
   bottleneck on very large corpora (1.28M chunks)
