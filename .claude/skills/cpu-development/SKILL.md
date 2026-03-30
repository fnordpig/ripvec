---
name: cpu-development
description: This skill should be used when working on ripvec's CPU backend, optimizing BLAS performance, debugging Accelerate or OpenBLAS issues, understanding BLAS thread contention, or porting models to CPU. Also use when the user mentions "CPU backend", "Accelerate", "OpenBLAS", "BLAS threading", "CpuDriver", "--device cpu", or "AOCL".
---

# CPU Backend Development

The CPU backend uses system BLAS (Accelerate on macOS, OpenBLAS on Linux) for GEMMs and scalar Rust loops for element-wise ops.

## Architecture

```
CpuDriver (driver/cpu.rs)
  Tensors: Vec<f32> (heap-allocated)
  GEMMs: ndarray general_mat_mul → system BLAS
  Element-wise: scalar Rust loops (compiler auto-vectorizes)
  Pool: none (Vec allocation is cheap)
```

## BLAS Threading Model

System BLAS libraries use internal multi-threading for large GEMMs. This conflicts with rayon's parallelism:

**Single-backend mode** (ModernBERT): One worker thread runs the forward pass. BLAS uses all cores for intra-GEMM parallelism. This is optimal — 73.5/s on M2 Max.

**Multi-backend mode** (BGE-small with multiple workers): Each rayon worker calls BLAS, causing mutex contention. Fix: call `force_single_threaded_blas()` per worker thread to force single-threaded BLAS, letting rayon handle inter-batch parallelism.

```rust
// Per-thread: force single-threaded BLAS (macOS 15+)
#[cfg(target_os = "macos")]
fn force_single_threaded_blas() {
    // BLASSetThreading via dlsym (weak-link, macOS 15+)
    // Falls back to VECLIB_MAXIMUM_THREADS env var on older macOS
}

#[cfg(target_os = "linux")]
fn force_single_threaded_blas() {
    // openblas_set_num_threads(1)
}
```

## Platform-Specific BLAS

| Platform | BLAS | Feature flag | Notes |
|----------|------|--------------|-------|
| macOS (Apple Silicon) | Accelerate (AMX) | `cpu-accelerate` | Uses AMX coprocessor, 73.5/s ModernBERT |
| macOS (Intel) | Accelerate (vecLib) | `cpu-accelerate` | SSE/AVX, slower than Apple Silicon |
| Linux (x86_64) | OpenBLAS | `cpu` | Compile with `RUSTFLAGS="-C target-cpu=native"` for AVX2/AVX-512 |
| Linux (AMD) | AOCL | `cpu` | Link against `libblis` for zen3/zen4 optimized GEMM |

## Adding ModernBERT to CPU

The CPU backend was added in ~200 lines because CpuDriver already implemented every FP32 Driver method. Only `load_modern_bert_weights` was needed:

1. Parse `config.json` for model dimensions
2. Read safetensors: fuse Q+K+V weights into `[3*hidden, hidden]`
3. Build RoPE cache (cos/sin tables as `Vec<f32>`)
4. Construct `ModernBertArch<Vec<f32>>` with weight vectors
5. Wire into `load_modernbert_cpu()` in `mod.rs`

**Key difference from Metal**: No `ensure_fp16` or pool system. All tensors are `Vec<f32>`. No FP16 path. The FP32 forward pass runs with `use_f16=false`.

## Performance Characteristics

CPU ModernBERT at 22 layers on M2 Max:
- **73.5/s** — competitive with Metal MPS (73.8/s)
- BLAS uses AMX coprocessor for GEMMs (same hardware MPS uses internally)
- Zero dispatch overhead (synchronous BLAS calls vs Metal encoder transitions)
- Memory bandwidth is shared (unified memory on Apple Silicon)

## Debugging CPU Issues

**BLAS contention**: If multi-worker CPU embedding is slow, check if BLAS threads are fighting rayon threads. Symptom: <20/s with 12 workers. Fix: `force_single_threaded_blas()`.

**NaN from CPU**: Rare. Check for division by zero in LayerNorm (eps too small) or softmax (all -inf inputs).

**Memory**: CPU tensors are heap-allocated `Vec<f32>`. No pool reuse. At batch=32 with ModernBERT: ~100MB peak per forward pass. Garbage collected per batch.
