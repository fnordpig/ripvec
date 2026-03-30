---
name: cuda-development
description: This skill should be used when working on ripvec's CUDA backend, implementing CUDA kernels, porting ModernBERT to NVIDIA GPUs, or debugging cudarc/CUDA issues. Also use when the user mentions "CUDA", "cudarc", "PTX", "warp shuffle", "tensor core", "NVIDIA", or "CudaDriver".
---

# CUDA Backend Development

The CUDA backend is the least developed — it supports ClassicBert (BGE-small) only. ModernBERT support requires implementing the remaining Driver trait methods.

## Current State

`crates/ripvec-core/src/backend/cuda.rs` — monolithic ClassicBert implementation using cudarc.

**What exists:**
- Device initialization, PTX compilation
- FP32 GEMMs via cuBLAS
- Element-wise kernels (embedding lookup, layer norm, GELU, softmax)
- Attention (Q@K^T, scores@V, masking)
- CLS pooling, L2 normalize

**What's missing for ModernBERT:**
- alternating local/global attention (windowed attention for local layers)
- GeGLU activation (currently only GELU)
- `split_gate_value` kernel (splits FFN intermediate into gate + value)
- RoPE encoding
- Mean pooling (only CLS pooling exists)
- `fused_residual_layernorm` (combined residual add + layer norm)

## Approach: Driver Extraction (Recommended)

The research agent assessed two approaches:

**Option A (recommended)**: Extract a CudaDriver from the monolithic code. Implement missing Driver trait methods. Then `ModernBertArch<CudaTensor>` runs the generic forward pass.

**Option B**: Add ModernBERT as a variant in the monolithic backend (how NomicBert was done before removal). Rejected — creates code duplication and doesn't benefit from the Driver/Arch abstraction.

## Implementation Plan

1. **Define `CudaTensor`**: Wrapper around cudarc `CudaSlice<f32>` with offset (similar to `MetalTensor`)
2. **Implement FP32 Driver methods**: Start with what cuBLAS provides (gemm, gemm_batched). Implement element-wise ops as CUDA kernels.
3. **Missing kernels** (~5 new kernels):
   - `rope_encode_kernel`: Apply rotary position embeddings
   - `geglu_kernel`: GeGLU activation (gelu(value) * gate)
   - `split_gate_value_kernel`: Split [total, 2*inter] → [total, inter] × 2
   - `fused_residual_layernorm_kernel`: Residual add + layer norm in one pass
   - `mean_pool_kernel`: Mean pooling with mask
4. **Weight loading**: `load_modern_bert_weights` reading safetensors
5. **Wire into mod.rs**: `load_modernbert_cuda()` and routing

## cudarc Patterns

```rust
// Kernel launch
let func = module.get_func("kernel_name")?;
let cfg = LaunchConfig::for_num_elems(n as u32);
unsafe { func.launch(cfg, (&output, &input, n as i32)) }?;

// cuBLAS GEMM
let blas = CudaBlas::new(device.clone())?;
unsafe {
    blas.gemm(
        cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
        cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
        n as i32, m as i32, k as i32,
        &alpha, b_ptr, k as i32,
        a_ptr, k as i32,
        &beta, c_ptr, n as i32,
    )?;
}
```

## Performance Expectations

Based on the CPU experience: implementing the Driver trait methods is the main work. The GEMM performance comes from cuBLAS (NVIDIA's hand-tuned library, comparable to MPS on Apple Silicon). Custom CUDA kernels for element-wise ops are straightforward — much simpler than Metal because CUDA has mature tooling and no MPS encoder transition overhead.

**Expected**: Competitive with CPU on consumer GPUs (RTX 3060+). Faster on datacenter GPUs (A100/H100) due to tensor cores and higher memory bandwidth.

## Warp Shuffle Optimization (Phase 5.2)

Six kernels could benefit from warp shuffles for parallel reduction:
- Layer norm (sum + sum-of-squares reduction)
- Softmax (max + sum reduction)
- L2 normalize (sum-of-squares reduction)
- Mean pool (sum reduction)

Each currently uses shared memory reduction. Warp shuffles (`__shfl_down_sync`) eliminate shared memory for within-warp reductions.
