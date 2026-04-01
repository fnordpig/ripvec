---
name: cuda-development
description: This skill should be used when working on ripvec's CUDA backend, optimizing CUDA kernels, debugging cudarc/cuBLAS issues, or investigating INT8/cuDNN features. Also use when the user mentions "CUDA", "cudarc", "PTX", "warp shuffle", "tensor core", "NVIDIA", "CudaDriver", or "nsys".
---

# CUDA Backend Development

The CUDA backend is fully shipped at **435 chunks/s on RTX 4090** — the fastest backend by 6x over Metal MPS (73.8/s). It supports ModernBERT via the Driver/Arch abstraction with FP16 tensor cores and fused CUDA kernels.

## Architecture

Two CUDA files exist in the codebase:

- `crates/ripvec-core/src/backend/driver/cuda.rs` (~4000 lines) — **ModernBERT CudaDriver** implementing ~40 Driver trait methods. This is the production CUDA path.
- `crates/ripvec-core/src/backend/cuda.rs` — Legacy monolithic ClassicBert backend (BGE-small). Used only with `--fast` flag.

The CudaDriver follows the same Driver/Arch split as Metal and CPU: `ModernBertArch<CudaTensor>` runs the generic forward pass, CudaDriver provides hardware primitives.

### CudaTensor

```rust
pub struct CudaTensor {
    pub(crate) slice: CudaSlice<f32>,
    pub(crate) offset: usize,
}
```

### GEMM dispatch

FP16 tensor cores via `cublasGemmEx` with FP16 inputs, FP32 compute, FP32 output. INT8 path (behind `RIPVEC_Q8=1`) uses `CUDA_R_8I` inputs, `CUDA_R_32I` output, `CUBLAS_COMPUTE_32I`.

### Custom kernels (~47)

All compiled at runtime via NVRTC. Key fused kernels that boosted throughput:
- `fused_split_geglu_f16` — eliminates separate split + geglu dispatch
- `fused_pad_qkv_split_f16` — pad + QKV split in one kernel
- `fused_reshape_unpad_f16` — attention output reshape + unpad

## Critical CUDA-specific lessons

1. **No pool pattern**: `CudaSlice::clone()` does full D2D memcpy (unlike Metal's refcounted buffers). Allocate fresh each time — the async allocator handles reuse.
2. **Disable event tracking**: `ctx.disable_event_tracking()` eliminates per-buffer event overhead (cudarc 0.19 records 130K+ events automatically).
3. **Runtime SM arch detection**: `cuDeviceGetAttribute(COMPUTE_CAPABILITY_MAJOR/MINOR)` → `compute_XX` (not `sm_XX` to avoid PTX version mismatches with newer NVRTC).
4. **INT8 per-tensor scale is disastrous**: Single-block reduction scanning ALL elements before every GEMM. Use per-row quantization.
5. **cuBLASLt INT8 has CUDA 13.1 bug**: `A_SCALE_POINTER`/`B_SCALE_POINTER` rejected as NOT_SUPPORTED. Needs CUDA 13.2.
6. **perf on Linux**: Use `--call-graph fp` NOT `dwarf` — CUDA driver debug symbols break dwarf unwinding.

## INT8 tensor cores

Investigated but not production-ready:
- GEMMs are 3.7x faster with INT8 tensor cores
- But dequantize overhead (7s for 57K kernel launches) eats the savings
- Net result: -5% vs FP16 at batch=32
- **Fix**: cuBLASLt epilogue fusion (A_SCALE_POINTER/B_SCALE_POINTER) eliminates the dequantize entirely — needs CUDA 13.2 driver

## Profiling workflow

```bash
# Capture with nsys
nsys profile --trace=cuda,nvtx,cublas -o /tmp/profile ./target/release/ripvec "query" corpus/

# Export to sqlite for tracemeld
nsys export --type sqlite -o /tmp/profile.sqlite /tmp/profile.nsys-rep

# Import into tracemeld
import_profile(source: "/tmp/profile.sqlite", format: "nsight_sqlite")
hotspots(dimension: "wall_ms")
focus_function("softmax")
```

For CPU-side profiling on Linux:
```bash
perf record -g --call-graph fp ./target/release/ripvec "query" corpus/
inferno-collapse-perf < perf.data > collapsed.txt
# Import collapsed stacks into tracemeld
```

## Future work

- **cuDNN fused multi-head attention**: Single biggest remaining GPU optimization. Fuses Q@K^T + softmax + scores@V into one kernel, eliminating softmax bottleneck (16% of GPU time).
- **cuBLASLt INT8 epilogue fusion**: When CUDA 13.2 driver ships. Projected ~530/s.
- **CUDA Graphs**: Blocked by CudaSlice Drop + async allocator ownership in Rust. Would save ~3.9s CPU overhead.
- **cuBLASLt algorithm heuristic tuning**: `cublasLtMatmulAlgGetHeuristic` searches for optimal CUTLASS kernel per GEMM shape.
- **Warp shuffle reductions**: 6 kernels (layer_norm, softmax, L2_normalize, mean_pool, banded_softmax, fused_residual_layernorm) currently use shared memory reduction. `__shfl_down_sync` eliminates shared memory for within-warp reductions.
