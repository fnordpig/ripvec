# Hybrid CPU+GPU Distributed Embedding Scheduler — Design Spec

## Goal

Run all available embedding backends (MLX GPU, CUDA GPU×N, Candle CPU) concurrently with work-stealing, using idle CPU cores to assist the GPU. Auto-detect the best backends at runtime.

## Problem

tracemeld profiling shows 85% of wall time is rayon threads sleeping during the GPU embed phase. The CPU has nothing to do while Metal processes batches. Running both backends concurrently would use all hardware: ~169/s (GPU) + ~30/s (CPU) = ~200/s.

## Architecture

### Work-Stealing Scheduler

A new `embed_distributed` function replaces the `embed_cpu_parallel` / `embed_gpu_pipelined` split.

**Input:** Pre-tokenized chunks (from rayon `par_iter`), N backends.

**Mechanism:** Each backend gets a dedicated worker thread. All workers race on a shared `AtomicUsize` cursor over the pre-tokenized chunk array.

```
Pre-tokenized chunks: [0, 1, 2, 3, 4, 5, ...]
                           ^
                      AtomicUsize cursor

Worker 0 (MLX GPU):     grabs chunks[0..128], embeds, grabs [128..256], ...
Worker 1 (Candle CPU):  grabs chunks[128..160], embeds, grabs [256..288], ...
Worker 2 (CUDA:0):      grabs chunks[160..288], embeds, ...
```

**Grab size:** Each worker's grab size starts at its natural batch size:
- GPU backends (`is_gpu() == true`): `batch_size * 4` (128)
- CPU backends: `batch_size` (32)

**Adaptive grab size:** After each round, workers adjust their grab size based on cursor progress:
- If cursor barely moved while I was working → I'm the fast worker → double grab size
- If cursor jumped far → others are fast → halve grab size
- Clamped to `[batch_size, batch_size * 8]`
- Batch size to `embed_batch` stays fixed (tensor shape optimization); grab size controls how many chunks are claimed per cursor increment

**Results:** Each worker collects `(original_index, Vec<f32>)` pairs. After all workers finish, scatter into a `Vec<Vec<f32>>` by index. Failed tokenizations produce empty embeddings at their index.

**Errors:** First error wins via `AtomicBool` flag + `Mutex<Option<Error>>`. Workers check the flag before each grab and exit early. The error is returned after all workers join.

**Profiler:** Shared `AtomicUsize` done counter. Workers call `profiler.embed_tick(done)` after each batch. Already handles concurrent access.

### Backend Auto-Detection

```rust
pub fn detect_backends(model_repo: &str) -> Result<Vec<Box<dyn EmbedBackend>>>
```

Detection order:
1. **MLX** — if compiled with `--features mlx`, attempt load. Skip on failure.
2. **CUDA** — if compiled with `--features cuda`, enumerate devices, one backend per device.
3. **CPU (Candle)** — always load as baseline helper.

Each backend loads its own copy of the model weights. On Apple Silicon, the OS page cache and unified memory architecture make this effectively free (mmap'd safetensors + MLX unified memory share the same physical pages).

### CLI Changes

`--backend` flag semantics change:
- **No flag (default):** `detect_backends()` — auto-detect all available, run hybrid
- **`--backend mlx`:** MLX only (no CPU helper)
- **`--backend candle`:** Candle CPU only
- **`--backend ort`:** ORT only

### API Changes

```rust
// embed_all: was single backend, now accepts multiple
pub fn embed_all(
    root: &Path,
    backends: &[&dyn EmbedBackend],
    tokenizer: &tokenizers::Tokenizer,
    cfg: &SearchConfig,
    profiler: &Profiler,
) -> Result<(Vec<CodeChunk>, Vec<Vec<f32>>)>

// search: same change
pub fn search(
    root: &Path,
    query: &str,
    backends: &[&dyn EmbedBackend],
    tokenizer: &tokenizers::Tokenizer,
    top_k: usize,
    cfg: &SearchConfig,
    profiler: &Profiler,
) -> Result<Vec<SearchResult>>
```

Query embedding uses `backends[0]` (the primary/fastest backend).

When `backends.len() == 1`, `embed_distributed` degenerates to a simple loop — no threads spawned.

### TUI Changes

`App.backends: Vec<Box<dyn EmbedBackend>>` replaces `App.backend: Box<dyn EmbedBackend>`. Query re-embedding in the interactive loop uses `backends[0]`.

## Memory Budget

For MLX + Candle CPU hybrid on a 137M-param model (bge-small f32):
- MLX: ~500MB in unified memory
- Candle: ~500MB mmap'd (OS page cache, same physical file)
- Pre-tokenized chunks (237K × 512 max tokens): ~2.9GB worst case
- Total: ~4GB peak — acceptable for Apple Silicon with 16-64GB RAM

## Expected Performance

Based on profiling (flask corpus, 2.3K chunks):
- MLX alone: 169/s
- Candle CPU alone: 27/s
- Hybrid (theoretical): ~196/s (+16%)
- On larger corpora where CPU contributes proportionally more (longer tail batches): potentially +20-25%

## Scope

**In scope:**
- `embed_distributed` with AtomicUsize work-stealing
- `detect_backends` auto-detection
- Adaptive grab sizing
- API changes to accept `&[&dyn EmbedBackend]`
- CLI default to auto-detect

**Out of scope:**
- Multi-node distributed embedding
- Dynamic backend hot-plugging
- Per-chunk backend routing (e.g., "send long chunks to GPU, short to CPU")
