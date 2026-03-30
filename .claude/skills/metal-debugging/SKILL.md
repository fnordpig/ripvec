---
name: metal-gpu-debugging
description: This skill should be used when debugging Metal GPU hangs, NaN embeddings, incorrect GEMM output, command buffer timeouts, encoder overflow, pipeline state regressions, or any unexpected Metal behavior. Also use when the user mentions "GPU hang", "NaN", "watchdog", "encoder overflow", "pipeline regression", or "Metal driver bug".
---

# Metal GPU Debugging on Apple Silicon

Systematic approaches to diagnosing and fixing Metal compute shader issues in ripvec's embedding pipeline.

## Triage: Identify the Failure Mode

Metal failures fall into five categories. Identify which one before investigating:

| Symptom | Category | First Action |
|---------|----------|--------------|
| `waitUntilCompleted` never returns | GPU hang | Test with fewer layers / smaller batch |
| Embeddings are NaN | Numerical overflow | Force FP32, check softmax/layernorm |
| Results differ from reference | Correctness bug | Unit test with small known matrix |
| All kernels slow (including MPS) | Pipeline state regression | Revert kernel source, bisect |
| Specific kernel slow | Performance bug | Profile with tracemeld |

## GPU Hangs

**Never assume encoder dispatch limits or watchdog timers.** Test empirically:

1. **Vary batch size**: batch=1 vs batch=32. If batch=1 works, the hang is from large matrix sizes overwhelming the GPU, not dispatch count.
2. **Vary sequence length**: Same batch size, short vs long sequences. Isolates whether it's time-based or dispatch-based.
3. **No-op kernel test**: Replace the kernel body with `return;` — if the hang persists, the issue is dispatch overhead, not kernel compute.
4. **Single K-tile test**: Add `if (loop_k >= 32) return;` — if the single tile is slow, the bottleneck is cooperative tgmem loads, not the compute loop.

**Common causes found in ripvec:**
- `segment_encoder()` costs ~118ms per call — what looked like "GPU timeout" was actually self-inflicted overhead from too-frequent encoder segmentation
- `pad_to_batch` / `unpad_from_batch` re-allocating output buffers caused buffer overflow into adjacent GPU memory
- The `flush_batch()` workaround (commit + wait + new CB) was never needed after the buffer re-allocation fix was applied

## NaN Embeddings

**Always check the NaN boundary.** Add a debug check after `embed_all()`:
```rust
let nan_count = embeddings.iter().filter(|e| e.iter().any(|x| x.is_nan())).count();
eprintln!("[DEBUG] {nan_count}/{} NaN", embeddings.len());
```

Common NaN sources:
- **Softmax with non-power-of-2 threadgroups**: Halving reduction (`s = tpg/2; s >>= 1`) drops orphaned `sdata[]` slots. Fix: `256.min(seq_len.next_power_of_two())` at dispatch.
- **Attention mask overflow**: `-1e9` mask value + large attention logits can overflow FP32. Use `-65504` (max finite FP16) instead.
- **Mean pool divide-by-zero**: If `pooling_mask` sum is 0 for a sequence, mean pool produces NaN. Clamp denominator to epsilon.

## Pipeline State Regression (Critical)

**Adding new Metal kernels can slow ALL kernels by 20x.** This is a Metal driver bug:
- `device half*` buffer arguments in pipeline state signatures cause global GPU scheduling degradation
- **Workaround**: Declare `device float*` in the kernel signature, cast to `half*` inside the kernel body
- **Also**: Compile new kernels in a separate Metal library (`NATIVE_GEMM_KERNEL`) to isolate MSL compiler effects

Test for regressions after any kernel source change:
```bash
# Before: record baseline MPS throughput (use bench.py, not ad-hoc commands)
uv run scripts/bench/bench.py --configs mps --no-build
# After: must match within 5%
```

## Correctness Debugging Strategy

For GEMM kernel correctness, always use **unit tests at model-scale dimensions**:

```rust
#[test]
fn gemm_correctness() {
    // Use M=128, N=768, K=768 (not tiny 4×4 — small matrices hide bugs)
    // Create known A, B on CPU
    // Dispatch kernel
    // Read back, compare against CPU reference
    // Assert max error < threshold
}
```

Small matrices (4×4) pass but full-scale (35762×2304) fails when:
- Per-row quantization scale is too coarse (use per-32-element blocks like llama.cpp)
- NL1 only loads 32 of 64 N-rows (simdgroups 2+3 read uninitialized tgmem)
- Pointer advances (`x += 32; y += 32`) are missing in the K-loop

## Additional Resources

Consult `references/metal-pitfalls.md` for the complete catalog of Metal driver bugs and workarounds discovered in ripvec development.
