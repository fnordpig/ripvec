# Learnings from the Metal GEMM Optimization Session

> Mistakes made, corrections found, and procedural lessons for future agents.
> Written by walking forward through the session's context, noting each wrong
> turn and what fixed it.

---

## Procedural Mistakes

### 1. Ad-hoc Bash Benchmarking Instead of bench.py

**What happened**: Repeatedly ran `cargo build --release && ./target/release/ripvec ... | rg 'done in'` in Bash, getting unreliable numbers from thermal variance, stale builds, and no cooldown between runs.

**Correction**: The user called this out explicitly — "Why aren't you using bench.py?" bench.py handles thermals, cooldown, result tracking, and reproducibility. It was created specifically to replace ad-hoc Bash.

**Rule**: ALL benchmarks through `uv run scripts/bench/bench.py`. If bench.py doesn't support a needed parameter, extend it.

### 2. Switching Directions Without Consulting

**What happened**: Flip-flopped between FP16 tiled kernel → FP32 compute → back to MPS FP16 based on shifting benchmark numbers. Reverted committed work without asking. The user had committed to the FP16 tiled kernel approach.

**Correction**: "Stop, don't switch directions without consulting me from now on."

**Rule**: When benchmark results conflict with expectations, investigate WHY (thermals? stale build? different corpus?) before concluding the approach is wrong. Present evidence and ASK before changing direction.

### 3. Not Using ripvec MCP and LSP

**What happened**: Fell back to Grep/Read for code navigation instead of using ripvec's own `search_code`/`get_repo_map` tools and rust-analyzer LSP. The user built these tools specifically for this purpose.

**Correction**: "I notice you're not using MCPs or LSPs, including ripvec!"

**Rule**: For code navigation: ripvec `search_code` first (meaning), LSP `definition`/`references` (symbols), Grep only for exact string patterns.

### 4. Running MPS Benchmarks When Not Needed

**What happened**: Repeatedly benchmarked the MPS path alongside the compute path, doubling test time. When debugging the compute kernel, MPS results add no information.

**Correction**: "Wait, why are you running MPS again?" and "Why are you looking at MPS at all? What is llama.cpp doing?"

**Rule**: Focus benchmarks on the path being optimized. Only benchmark MPS when checking for regressions after kernel source changes.

### 5. Not Looking at the Reference Implementation

**What happened**: Spent hours tweaking the tiled GEMM kernel's tile sizes, barrier placement, and store strategy without reading llama.cpp's actual kernel code. Made assumptions about llama.cpp's architecture that were wrong.

**Correction**: "Why aren't you looking at the llama.cpp implementation? Clone it and ripvec MCP it and -read it-"

**Rule**: When optimizing against a reference implementation, READ the reference code line-by-line before writing. Don't guess.

### 6. Dispatching Too Many Concurrent GPU Workloads

**What happened**: Background agents ran full Flask corpus ripvec processes simultaneously while the foreground also ran GPU benchmarks. Multiple Metal command buffers competing for GPU caused system near-halt.

**Correction**: Had to kill the parent PID to recover the system.

**Rule**: Only one GPU-heavy ripvec process at a time. Background agents should use `--device cpu` or small corpora.

---

## Technical Mistakes

### 7. Assuming FP16 Activations Are a Compute Win

**What happened**: Built an entire FP16 activation pipeline (f32_to_f16 conversions, FP16 element-wise kernels, FP16 inter-layer buffers) based on the assumption that Apple Silicon has 2× FP16 throughput.

**Correction**: Apple Silicon's `simdgroup_multiply_accumulate` runs at the SAME clock speed for half and float. FP16 only saves memory bandwidth, not compute. The user identified this: "The whole reason you were storing to device half* was the FP16 pipeline — which you've already established gives zero compute throughput advantage."

**Lesson**: Verify hardware capabilities before building optimization strategies. "2× FP16 throughput" is true for NVIDIA tensor cores but not for Apple Silicon simdgroup MACs.

### 8. Attributing GPU Hangs to Encoder Dispatch Limits

**What happened**: When the GPU hung at ≥15 layers, assumed Metal had an encoder dispatch count limit. Added `segment_encoder()` (endEncoding + new encoder) every 8 layers as a workaround.

**Correction**: The "hang" was actually just extreme slowness from `segment_encoder()` overhead (118ms × 22 = 2.6s per forward pass). Test timeouts expired before completion. Proved empirically: 22 layers in one encoder completes at 31/s with zero issues.

**Lesson**: "GPU hang" might be "GPU slow." Always test with longer timeouts before concluding something hangs. Use bench.py which handles this.

### 9. The Pipeline State `device half*` Regression

**What happened**: Added `gemm_f16w_f32a_kernel` with `device half*` arguments. MPS throughput dropped from 435/s to 21/s — even though MPS doesn't use this kernel.

**First wrong hypothesis**: "MSL compiler regression from complex simdgroup code in the same compilation unit."
**Second wrong hypothesis**: "Moving to a separate Metal library will fix it."

**Actual cause**: The pipeline state object with `device half*` buffer argument descriptors triggers a Metal driver bug that degrades ALL GPU scheduling.

**Fix**: Declare `device float*` in the signature, cast to `half*` inside the kernel body.

**Lesson**: Any kernel source or pipeline state change can affect the ENTIRE GPU. Always benchmark MPS after adding new kernels. Bisect: kernel source → pipeline state → dispatch code.

### 10. NL1=4 Loading Only 32 of 64 N-Rows

**What happened**: The cooperative B load used NL1=4 (128/4 = 32 threads for B loading), covering only rows 0-31. Simdgroups 2 and 3 read uninitialized threadgroup memory. This produced correct-looking results for simdgroups 0+1 but garbage for 2+3.

**Why it wasn't caught earlier**: The FP16 path routes through MPS (not the compute kernel), so the bug was latent. Only manifested when INT8 quantization forced the compute path.

**Fix**: NL1=2, restructured B load to mirror A load pattern (16 elements per thread, 64 rows).

**Lesson**: Unit tests at model-scale dimensions (128×768×768) catch bugs that 4×4 tests miss. Test ALL simdgroups, not just the first two.

### 11. Per-Row INT8 Quantization Is Too Coarse

**What happened**: Implemented INT8 with one scale per row (768 elements). The unit test (4×4 matrix) passed with max error 0.03%. Full-scale search returned wrong rankings.

**Correction**: llama.cpp uses per-32-element blocks (`block_q8_0`). A single outlier in 768 values sets the scale for the entire row, crushing dynamic range.

**Fix**: block_q8_0 format with per-block scales. Score difference dropped from "completely wrong" to <0.02 vs FP16.

**Lesson**: Quantization granularity matters enormously. Always match the reference implementation's block size.

### 12. Missing Pointer Advance in K-Loop

**What happened**: Copied the kernel from the FP16 version but forgot `x += 32; y_q8 += 32;` at the end of the K-loop. Every iteration re-read the first 32 elements of each row.

**Why it wasn't caught**: The 4×4 unit test with K=4 only had one K-iteration (4 < 32), so the pointer advance never fired.

**Lesson**: Copy-paste between kernels is error-prone. Diff the two kernels line-by-line after copying. Test with K > BK (K=768, BK=32 → 24 iterations).

### 13. FP32 Temp Buffer Bandwidth Overhead

**What happened**: Option 1 (GEMM → FP32 temp → f32_to_f16 → FP16 output) added 43GB of extra memory bandwidth per forward pass. 88 GEMMs × 82M elements × 6 bytes read+write.

**Result**: 34.7/s vs MPS at 72/s — the conversion pass was more expensive than the MFA wrapper overhead it replaced.

**Fix**: Fused store (simdgroup_store to scratch → float→half conversion in same kernel). Eliminated the separate f32_to_f16 pass.

**Lesson**: Memory bandwidth is the dominant cost at large M. Any extra pass over the output buffer is extremely expensive. Fuse operations to minimize passes.

### 14. Thermal Monitoring Reporting `cpu=0`

**What happened**: bench.py's thermal monitoring used `pmset -g therm` which reports CPU_Speed_Limit as percentage (100 = no throttling). Inverted to `100 - limit` giving 0 = "no throttling." User: "What does thermal=0 mean? I hear the fan running loud."

**Fix**: Added GPU utilization via ioreg, CPU speed %, and memory usage. Now shows `speed=100% gpu=99% mem=12.5GB`.

**Lesson**: Thermal monitoring must show actual sensor values, not derived "everything is fine" metrics.

---

## Strategic Lessons

### 15. The Correct Architecture Was Identified by the User

The path that finally worked (FP16 device activations → FP16 device weights → FP32 accumulators → fused half store) was proposed by the user, not discovered through experimentation. The user's insight: "Store to device float*. Your inter-layer buffers should be FP32."

Then refined further: "The whole reason you were storing to device half* was the FP16 pipeline — which you've already established gives zero compute throughput advantage."

**Lesson**: Present data honestly and ask for direction. The user often has architectural insights that experimentation alone won't find.

### 16. Small Experiments Before Big Implementations

The most productive debugging moments were quick isolation tests:
- No-op kernel: isolates dispatch vs compute overhead
- Single K-tile: isolates load vs compute within the kernel
- batch=1: isolates per-chunk vs batching issues
- layers=1 vs layers=22: isolates scaling vs per-layer issues

**Lesson**: Design experiments that isolate one variable. Measure before building.

### 17. tracemeld Diff Is the Ground Truth

The tracemeld `diff_profile` showed that removing `segment_encoder` saved 405ms of driver processing — this was invisible from throughput numbers alone. It identified the encoder transition as the #1 bottleneck when raw throughput numbers were confusing.

**Lesson**: Always save tracemeld baselines before making changes. The diff tells you exactly what improved and what regressed.

### 18. The Reference Implementation (llama.cpp) Is the Answer

Every major optimization that worked came from studying llama.cpp:
- 8×8-block tgmem layout
- NL=2 cooperative load pattern
- block_q8_0 quantization format
- `simdgroup_barrier(mem_flags::mem_none)` placement
- Separate `simdgroup_load` for A and B tiles

**Lesson**: Clone the reference, read the code, copy the architecture. Don't reinvent.

---

## CUDA Backend Lessons (Session 2)

### 19. cudarc CudaSlice::clone() Is a Full D2D Memcpy

**What happened**: Implemented a Metal-style buffer pool where `alloc_tensor` returns `pool[cursor].clone()`. On Metal, `Retained<MTLBuffer>::clone()` is a refcount bump (free). On CUDA, `CudaSlice::clone()` calls `cuMemcpyDtoDAsync` — a full device-to-device copy. Result: 4.5s of D2D copies, 59/s instead of 541/s.

**Fix**: Remove pool entirely. Use cudarc's async allocator (`unsafe { stream.alloc() }`) which amortizes allocation overhead across the stream. Pool-based reuse requires either `Arc<CudaSlice>` (ownership complexity) or a bump allocator (different abstraction).

**Lesson**: Backend abstractions that work on one platform (Metal refcounted buffers) may have completely different performance characteristics on another (CUDA owned buffers). Profile before assuming a pattern transfers.

### 20. cudarc Event Tracking Overhead

**What happened**: cudarc 0.19 automatically records events on every buffer access for multi-stream synchronization. With single-stream usage (all our CUDA work), this is pure overhead — 130K+ events per forward pass.

**Fix**: `ctx.disable_event_tracking()` at driver initialization. Safe because we use a single stream.

### 21. INT8 Tensor Core Overhead vs Savings

**What happened**: INT8 GEMMs were 3.7× faster than FP16 (7.1s vs 26s). But per-row activation quantization (1.7s) + INT32→FP16 dequantization (7.0s) ate all the savings. Net: 410/s INT8 vs 432/s FP16.

**Root cause**: 57K separate dequantize kernel launches. Each is tiny (M~50, N~768) but carries 5μs CPU dispatch overhead. The dequantize reads INT32 (4B/element) + writes FP16 (2B) = 6B/element bandwidth.

**What would fix it**: cuBLASLt `A_SCALE_POINTER`/`B_SCALE_POINTER` fuses dequantization into the GEMM epilogue (zero separate kernel). Available in CUDA 12.0+ but the cuBLASLt `NOT_SUPPORTED` error on CUDA 13.1 blocked this path. CUDA 13.2 reportedly fixes INT8 matmul issues.

**Lesson**: Quantization only pays off when the quant/dequant overhead is amortized. For small-M GEMMs at batch=32, the overhead dominates. Either fuse into the GEMM epilogue or increase batch size.

### 22. NVRTC Arch Flag and CUDA Version Compatibility

**What happened**: Hardcoded `sm_70` in NVRTC compile options. Worked on CUDA 12.0. CUDA 13.x dropped support for old SM targets, causing "invalid value for --gpu-architecture" at runtime.

**Fix**: Detect compute capability at runtime via `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR/MINOR`, use `compute_XX` (virtual arch, forward-compatible PTX) instead of `sm_XX` (real arch, tied to specific GPU).

### 23. Streaming Pipeline Deadlock

**What happened**: The streaming pipeline (chunk→tokenize→embed) was already implemented but disabled with `false &&` due to a deadlock. Stage 2's closure captured `batch_tx` by reference (not value), so it never dropped when the thread exited → channel never closed → Stage 3 waited forever.

**Fix**: Add `move` to Stage 2's closure so `batch_tx` is captured by value and drops when the thread exits.

**Lesson**: With `crossbeam_channel`, the channel closes when ALL senders drop. If a sender is borrowed (not moved) into a thread, it outlives the thread and the channel stays open.

### 24. Progress Bar in Streaming Mode

**What happened**: `embed_distributed` has an internal `done_counter` that resets to 0 on each call. In streaming mode, it's called per-batch, so the profiler sees done counts of 32, 32, 32... instead of 32, 64, 96... The total also jumped because it tracked chunks produced (upstream) rather than chunks embedded.

**Fix**: Pass `Profiler::noop()` to `embed_distributed` in streaming mode. Drive progress from the streaming loop with a cumulative counter. Show byte-based progress (total known from walk phase) instead of chunk-based (unknown until chunking finishes).

### 25. MCP Server Missing CUDA Feature

**What happened**: `ripvec-mcp` on Linux was built with `features = ["cpu"]` only — no CUDA. Query embedding used 4 CPU cores (OpenBLAS) instead of the RTX 4090. 50-60s warmup per query.

**Fix**: Add `cuda` to the non-macOS platform dependencies. Also cache on-demand indices by canonical path so `incremental_index` (which walks the entire source tree) only runs once per root per session.
