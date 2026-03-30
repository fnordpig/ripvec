# Metal Driver Pitfalls Catalog

Bugs and workarounds discovered during ripvec Metal backend development on Apple Silicon (M2 Max).

## Pipeline State `device half*` Regression

**Symptom**: Adding a kernel with `device half*` buffer arguments causes 20× slowdown on ALL GPU workloads, including MPS-backed GEMMs that don't use the new kernel.

**Root cause**: Metal driver bug. Pipeline states with half-precision buffer argument descriptors trigger a different (slower) GPU scheduling strategy globally.

**Workaround**: Declare all buffer arguments as `device float*` in the kernel signature, then cast inside the kernel:
```metal
kernel void my_kernel(device float* A_raw [[buffer(0)]]) {
    device half* A = (device half*)A_raw;
    // use A as half* from here
}
```

**Verification**: Always benchmark MPS throughput before and after adding any new kernel. Even kernels compiled in a separate Metal library can trigger this regression if their pipeline state is created.

## MSL Compiler Cross-Kernel Effects

**Symptom**: Adding complex simdgroup code to a Metal library slows other kernels in the same library.

**Fix**: Compile performance-critical native simdgroup kernels in a separate Metal library:
```rust
let native_gemm_library = compile_library(device, NATIVE_GEMM_KERNEL)?;
```

The main `KERNELS` and `GEMM_KERNEL` libraries stay unaffected.

## Encoder Segmentation Overhead

**Symptom**: `segment_encoder()` (endEncoding + lazy re-create) costs ~118ms per call on M2 Max.

**Context**: This was added as a workaround for GPU "hangs" at ≥15 layers. Testing proved the hangs were actually slow execution from the segmentation overhead itself, not encoder limits.

**Resolution**: Metal compute encoders handle 400+ dispatches in a single encoder. No segmentation needed. Remove all `segment_encoder()` calls.

## MPS Encoder Transition Cost

**Fact**: Each MPS dispatch requires closing the compute encoder, dispatching MPS, then creating a new encoder. This costs ~0.8ms per transition on M2 Max.

**Impact**: ModernBERT 22 layers × 4 MPS GEMMs = 88 transitions × 0.8ms = ~70ms overhead per forward pass (~7% at 75/s).

**This is inherent to MPS architecture** — cannot be eliminated while using MPS. Custom compute GEMMs eliminate transitions but are slower per-FLOP.

## FP16 Activation Pipeline Is a False Optimization

**Fact**: Apple Silicon `simdgroup_multiply_accumulate` runs at the **same clock speed** for both `half` and `float` inputs. FP16 activations provide zero compute throughput advantage.

**What FP16 helps**: Memory bandwidth for element-wise ops (2× less data to read/write). At 768-dim ModernBERT, this is marginal.

**What FP16 hurts**: Forces the MFA `simdgroup_matrix_storage` wrapper for type conversion at GEMM store boundaries, adding 26% overhead vs native simdgroup ops.

## `simdgroup_matrix_storage` (MFA Wrapper) vs Native API

**MFA wrapper** (`simdgroup_matrix_storage<half>::load/store`): Decomposes cooperative 8×8 tile operations into per-thread scalar arithmetic with morton indexing. ~26% overhead vs native.

**Native API** (`simdgroup_load/simdgroup_store`): Single hardware instruction, cooperative across all 32 threads. Cannot store `float8x8` to `device half*` — requires scratch buffer + conversion.

**Mixing types**: `simdgroup_half8x8` (native) is incompatible with `simdgroup_matrix_storage<half>` (MFA). Cannot call MFA `.multiply()` on native types.

## `simdgroup_store(float8x8, device half*)` Does Not Exist

The native `simdgroup_store` can write `float8x8` to `device float*` or `threadgroup float*`, and `half8x8` to `device half*` or `threadgroup half*`. No implicit type conversion.

**Workaround for FP32 accumulator → device half* output**:
1. `simdgroup_store(float8x8)` to threadgroup float scratch (64 floats per tile)
2. Same 32 threads convert float→half and write to device half*
3. Per-tile, no cross-simdgroup barriers needed

## Device Memory Bandwidth at Large M

At M=35762 (batch=32 with ModernBERT), the activation matrix A is 110MB at FP32 (55MB at FP16). The cooperative load reads scattered 4-8KB tiles across this buffer. L2 cache cannot hold it all.

**Impact**: FP32 device reads are 2× slower than FP16, making the A cooperative load bandwidth-bound, not compute-bound.

**This is why MPS (FP16) beats our compute kernel (FP32 activations)** — MPS reads FP16 activations natively.

## block_q8_0 Quantization Format

Per-32-element block with embedded scale: `{ half d; int8_t qs[32]; }` = 34 bytes.

**Per-row quantization (one scale per 768 elements) is too coarse** — outliers crush dynamic range. Per-block (32 elements) matches llama.cpp and produces <0.02 score difference vs FP16.

**Dequantize during cooperative load**: Read block header once, then 32 values. The dequantize cost is negligible vs memory read latency.

## xctrace Environment Variable Propagation

`xcrun xctrace record --launch` does NOT propagate parent shell environment variables to the launched process. Must use `--env VAR=VALUE` flag:
```bash
# WRONG: env var not seen by child
RIPVEC_NO_MPS=1 xcrun xctrace record --launch -- ./ripvec ...

# CORRECT: env var passed via --env flag
xcrun xctrace record --env RIPVEC_NO_MPS=1 --launch -- ./ripvec ...
```

## Metal System Trace Template Requires GPU Events

The "Metal System Trace" xctrace template cannot export data when there are zero MPS dispatches AND the trace is large (>2GB). Short captures (<15s) of compute-only paths may export; long ones fail with "Document Missing Template Error".

**Workaround**: Profile the MPS path for GPU timeline analysis. Use samply for CPU-side stall analysis of the compute path.
