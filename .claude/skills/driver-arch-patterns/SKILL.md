---
name: driver-arch-patterns
description: This skill should be used when working with ripvec's Driver/Architecture abstraction, adding new backends, porting ModernBERT to CPU/CUDA, understanding the pool buffer system, modifying the forward pass, or debugging tensor lifecycle issues. Also use when the user mentions "Driver trait", "ModelArch", "pool cursor", "begin_batch", "end_batch", "alloc_zeros", "prepare_batch", or "forward pass".
---

# ripvec Driver/Architecture Pattern Guide

The Driver/Arch split separates hardware-specific compute primitives (Driver) from model-specific forward pass logic (ModelArch). Understanding this split is essential for adding backends, fixing bugs, and optimizing performance.

## The Split

```
ModelArch (modern_bert.rs)     Driver (metal.rs, cpu.rs)
  forward()                      gemm(), gemm_f16(), gemm_q8()
    → calls Driver methods         layer_norm(), gelu(), softmax()
    → orchestrates layers           alloc_zeros(), alloc_zeros_f16()
    → manages pool cursors          begin_batch(), end_batch()
```

**ModelArch is generic over `D: Driver`** — the same forward pass runs on Metal, CPU, or CUDA.

## Pool Buffer System

Metal and CPU drivers reuse GPU/CPU buffers across forward passes to eliminate allocation overhead:

- `begin_batch()`: Reset pool cursor to 0. All buffers from previous forward pass are reusable.
- `alloc_zeros(n)` / `alloc_zeros_f16(n)`: Return buffer at current cursor, advance cursor. Reuse if buffer at that slot is large enough; allocate fresh if not.
- `save_pool_cursor()` / `restore_pool_cursor(saved)`: Save cursor before a layer, restore after. This recycles transient tensors (QKV, scores, etc.) while preserving `hidden_states`.
- `end_batch()`: Commit command buffer, wait for completion, check GPU errors.

**Critical invariant**: `hidden_states` must be allocated BEFORE `save_pool_cursor()`. Otherwise `restore_pool_cursor` recycles it.

**Both FP32 and FP16 cursors are managed**: `restore_pool_cursor` resets both. When the FP16 forward path allocates FP32 temp buffers (for `gemm_mixed` → `f32_to_f16`), those FP32 slots are reclaimed per layer.

## Forward Pass: FP16 vs FP32

The ModernBERT forward pass has two paths controlled by `use_f16`:

**FP16 path** (default on Metal):
- `f32_to_f16` once after embedding LayerNorm
- All layers in FP16: `gemm_f16`, `layer_norm_f16`, `geglu_f16`, etc.
- `f16_to_f32` once before final LayerNorm + pooling
- Weight GEMMs route through `gemm_f16()` which dispatches MPS, native compute, or INT8

**FP32 path** (CPU, fallback):
- All layers in FP32: `gemm`, `layer_norm`, `gelu`, etc.
- `gemm()` always uses MPS on Metal — the native compute kernel requires FP16 activations

## Adding a New Backend

To add ModernBERT support to a new backend (e.g., CUDA):

1. **Implement the Driver trait** — start with the FP32 methods (gemm, layer_norm, gelu, etc.)
2. **Weight loading** — add `load_modern_bert_weights()` that reads safetensors and creates tensors
3. **Wire into mod.rs** — add `load_modernbert_xxx()` function and route in `detect_backends()`
4. **Test with FP32 path** — set `use_f16 = false` initially

The CPU backend was added in ~200 lines because CpuDriver already implemented every FP32 method. Only weight loading was new.

## GEMM Dispatch Hierarchy

`gemm_f16()` on MetalDriver checks in order:
1. **INT8 block_q8_0**: If `b.q8` has data → `gemm_q8()` (native compute kernel)
2. **Native FP16 compute**: If `RIPVEC_NO_MPS=1` → fused FP16 compute kernel
3. **MPS FP16**: Default → `MPSMatrixMultiplication` (fastest, uses AMX)

`gemm()` on MetalDriver:
- Always uses MPS (FP32 path). The native compute kernel requires FP16 activations.

## Tensor Types

`MetalTensor` wraps an `MTLBuffer` + offset with optional companion buffers:
- `.fp16`: Pre-converted FP16 weights (created by `ensure_fp16`)
- `.q8`: Block-quantized INT8 weights (created by `quantize_weights_q8`)

These are `RefCell<Option<Retained<MTLBuffer>>>` — lazily populated at model load time.

## Common Debugging Patterns

**Pool aliasing**: If two operations read the same pool buffer simultaneously, the second corrupts the first. Symptom: cosine similarity drops to ~0.57. Fix: ensure `save_pool_cursor` / `restore_pool_cursor` bracket each layer correctly.

**Buffer size mismatch**: `pad_to_batch` and `unpad_from_batch` must NOT re-allocate the caller's buffer. The caller controls sizing via `alloc_zeros`.

**FP16/FP32 path confusion**: The `use_f16` flag controls which path runs. Element-wise kernels have separate FP16 variants (`_f16` suffix). Mixing FP32 activations with FP16 kernels produces garbage.
