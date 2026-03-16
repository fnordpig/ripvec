# ripvec Profiling Instrumentation Design

## Overview

Add `--profile` flag to the ripvec CLI that prints real-time pipeline timing to stderr. Output streams as each phase completes, with periodic throughput updates during long-running embedding. Reports thread utilization and mutex contention to diagnose parallelism bottlenecks.

## Activation

- `--profile` boolean flag on the CLI. When absent, zero overhead.
- `--profile-interval <seconds>` configurable reporting interval (default 10.0s).
- All profiling output goes to stderr (doesn't interfere with stdout results or piping).

## Output Format

```
[profile] ripvec 0.1.0 | 10-core | rayon: 8 threads | model: BAAI/bge-small-en-v1.5
[0.045s] model_load: 45ms
[0.057s] walk: 142 files in 12ms
[0.146s] chunk: 347 chunks from 142 files in 89ms (8 threads, 6 active, skew: 12-58 chunks/thread)
[0.154s] embed_query: 8ms
[10.2s]  embed: 285/347 (last 10s: 28.1/s, overall: 28.4/s) lock_wait: 3% inference: 97%
[12.4s]  embed: 347/347 done in 12.2s (28.4/s) lock_wait: 3% inference: 97%
[12.4s]  rank: top 10 from 347 in 0.1ms
[12.4s]  total: 12.4s
```

Edge cases:
- **0 files/chunks**: walk and chunk phases print `0 files` / `0 chunks` normally. Embed phase prints `embed: skipped (0 chunks)` and `embed_begin`/`embed_tick`/`embed_done` handle this without division-by-zero.
- **Fast runs (<1 interval)**: The periodic `embed_tick` update never fires; only `embed_done` prints. This is intentional — no special-case logic needed.

## Architecture

### New module: `crates/ripvec-core/src/profile.rs`

A `Profiler` enum with two variants: `Active` (collects timings, writes to stderr) and `Noop` (all methods are no-ops, zero cost at `--release` optimization levels).

#### Thread safety model

The `Profiler` must be `Sync` so `&Profiler` can cross rayon thread boundaries. Interior mutability is split by access pattern:

- **Sequential-only fields** (embed phase — only one thread calls these): use `Cell<T>` for `Copy` types (`Instant`, `usize`, `Duration`). `Cell` is `!Sync` on its own, but these fields are never accessed from rayon threads — only from the main thread during the sequential embed loop.
- **Parallel-accessed fields** (chunk phase — rayon threads write): use `Mutex<Vec<usize>>` for per-thread chunk counts.

To make the overall enum `Sync`, the sequential-only fields are wrapped in an `UnsafeCell`-based newtype with a manual `Sync` impl, or more practically: use `std::sync::Mutex<EmbedState>` for the embed fields too (acceptable since it's only locked from one thread — no contention). The chunk_counts field is a separate `Mutex<Vec<usize>>`.

```rust
pub enum Profiler {
    Active {
        start: Instant,
        interval: Duration,
        embed: Mutex<EmbedState>,
        chunk_counts: Mutex<Vec<usize>>,
    },
    Noop,
}

struct EmbedState {
    phase_start: Option<Instant>,
    last_report: Instant,
    chunks_at_last_report: usize,
    total_lock_wait: Duration,
    total_inference: Duration,
    total_chunks: usize,
}
```

#### Public API

Phase timing uses a guard-based API to prevent start/end name mismatches:

| Method | Purpose |
|--------|---------|
| `Profiler::new(enabled, interval)` | Construct Active or Noop |
| `header(model_repo, threads, cores)` | Print system info line. Core count via `std::thread::available_parallelism()`. |
| `phase(name) -> PhaseGuard` | Returns a guard that records start time; prints elapsed + name on drop |
| `chunk_thread_report(chunks)` | Called per rayon thread, records chunk count for that thread |
| `chunk_summary(total_chunks, total_files)` | Print chunk phase summary with thread utilization |
| `embed_begin(total)` | Start embed phase, record total chunk count |
| `embed_tick(done)` | Called after each chunk embed; prints periodic update if interval elapsed |
| `embed_lock_wait(duration)` | Accumulate mutex wait time |
| `embed_inference(duration)` | Accumulate ONNX inference time |
| `embed_done()` | Print final embed summary |
| `finish()` | Print total wall-clock time |

`PhaseGuard` holds a reference to the profiler and the phase name. On drop, it computes elapsed time and prints to stderr. For `Noop`, the guard is a no-op.

```rust
// Usage:
{
    let _guard = profiler.phase("walk");
    let files = collect_files(root);
} // prints "[0.057s] walk: 12ms" on drop

// With detail:
{
    let guard = profiler.phase("walk");
    let files = collect_files(root);
    guard.set_detail(format!("{} files", files.len()));
} // prints "[0.057s] walk: 142 files in 12ms"
```

### Changes to existing modules

**`embed.rs`** — `search()` gains a `profiler: &Profiler` parameter.
- Phase 1 (walk): `let _g = profiler.phase("walk");`
- Phase 2 (chunk): `par_iter` calls `profiler.chunk_thread_report(n)` per iteration using `rayon::current_thread_index()`. Ends with `profiler.chunk_summary(chunks.len(), files.len())`.
- Phase 3 (embed query): `let _g = profiler.phase("embed_query");`
- Phase 4 (embed chunks): calls `embed_begin`, then `embed_tick` per chunk.
- Phase 5 (rank): `let _g = profiler.phase("rank");`

**`embed_text()`** — Gains a `profiler: &Profiler` parameter. Times lock acquisition and inference separately:
```rust
let lock_start = Instant::now();
let mut guard = model.lock()...;
profiler.embed_lock_wait(lock_start.elapsed());

let infer_start = Instant::now();
let result = guard.embed(&ids, &mask, &type_ids);
profiler.embed_inference(infer_start.elapsed());
```

Note: since embedding is currently sequential (single Mutex), `lock_wait` will be near-zero. This metric becomes valuable when/if we move to multiple ONNX sessions or batch inference. It's forward-looking instrumentation.

**`cli.rs`** — Add two fields:
- `--profile` (`bool`, default false)
- `--profile-interval` (`f64`, default 10.0, help: "profiling report interval in seconds")

**`main.rs`** — Create profiler, call `header()` before model load, wrap model load in `profiler.phase("model_load")`, pass profiler to `search()`, call `finish()` after results.

### Rayon thread utilization tracking

For the chunk phase, we track per-thread work using a thread-safe shared counter. Inside the `par_iter` flat_map, each iteration calls `chunk_thread_report(n)` with the number of chunks produced from that file. We use `rayon::current_thread_index()` to attribute work to threads.

The summary prints:
- Total rayon threads (pool size)
- Active threads (threads that processed at least one file)
- Skew: min-max chunks per active thread

High skew (e.g., `skew: 2-180`) indicates a few very large files dominating one thread — a signal to consider splitting large files into sub-chunks.

### Zero-cost when disabled

When `--profile` is not passed, `Profiler::Noop` is constructed. All methods on `Noop` are empty and should be inlined away by the compiler at `--release` optimization levels. No `Instant::now()` calls, no string formatting, no stderr writes. In debug builds, the match on the enum variant is still present but negligible.

## Deliberate Omissions

- **No flamegraph integration** — that's a separate tool (`cargo flamegraph`), not in-app
- **No file-level timing** — too noisy; per-phase aggregates are more actionable
- **No JSON profiling output** — text is sufficient for Claude Code; add later if needed
- **No tracing crate dependency** — custom profiler is simpler for this use case
- **No walk phase parallelism tracking** — `ignore` uses its own crossbeam threads internally, not observable from our code
