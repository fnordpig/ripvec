# Profiling Instrumentation Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--profile` flag that prints real-time pipeline phase timing, embedding throughput, rayon thread utilization, and mutex contention to stderr.

**Architecture:** A `Profiler` enum (`Active`/`Noop`) in ripvec-core with guard-based phase timing and embed-specific progress methods. Passed as `&Profiler` through the `search()` pipeline. CLI creates it from `--profile` / `--profile-interval` flags.

**Tech Stack:** No new dependencies. Uses `std::time::Instant`, `std::sync::Mutex`, `std::io::Write` (stderr).

**Spec:** `docs/superpowers/specs/2026-03-16-profiling-design.md`

---

## File Structure

```
crates/ripvec-core/src/
├── profile.rs          # NEW: Profiler enum, PhaseGuard, EmbedState
├── embed.rs            # MODIFY: add profiler param to search() and embed_text()
├── lib.rs              # MODIFY: add pub mod profile
crates/ripvec/src/
├── cli.rs              # MODIFY: add --profile and --profile-interval flags
├── main.rs             # MODIFY: create profiler, pass to search(), call header/finish
```

---

## Chunk 1: Profiler Module + Integration

### Task 1: Create profile.rs — Profiler Enum and PhaseGuard

**Files:**
- Create: `crates/ripvec-core/src/profile.rs`
- Modify: `crates/ripvec-core/src/lib.rs`

- [ ] **Step 1: Write tests for Noop profiler**

Add to the bottom of `crates/ripvec-core/src/profile.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn profiler_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<Profiler>();
    }

    #[test]
    fn noop_profiler_all_methods_are_safe() {
        let p = Profiler::noop();
        p.header("0.1.0", "test/repo", 4, 8);
        {
            let g = p.phase("test_phase");
            g.set_detail("some detail".to_string());
        }
        p.chunk_thread_report(10);
        p.chunk_summary(100, 50, Duration::from_millis(89));
        p.embed_begin(100);
        p.embed_tick(1);
        p.embed_lock_wait(Duration::from_millis(1));
        p.embed_inference(Duration::from_millis(10));
        p.embed_done();
        p.finish();
    }

    #[test]
    fn active_profiler_phase_guard_formats_correctly() {
        let p = Profiler::new(true, Duration::from_secs(10));
        {
            let g = p.phase("test_phase");
            std::thread::sleep(Duration::from_millis(5));
            g.set_detail("42 items".to_string());
        }
        // Just verify it doesn't panic — stderr output is visual
    }

    #[test]
    fn embed_tick_respects_interval() {
        let p = Profiler::new(true, Duration::from_secs(100)); // very long interval
        p.embed_begin(10);
        for i in 1..=10 {
            p.embed_tick(i);
            // With a 100s interval, no periodic output should fire
        }
        p.embed_done();
    }

    #[test]
    fn chunk_summary_with_zero_files() {
        let p = Profiler::new(true, Duration::from_secs(10));
        p.chunk_summary(0, 0, Duration::from_millis(0));
        // Should not panic or divide by zero
    }

    #[test]
    fn embed_done_with_zero_chunks() {
        let p = Profiler::new(true, Duration::from_secs(10));
        p.embed_begin(0);
        p.embed_done();
        // Should print "skipped" line, not divide by zero
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p ripvec-core -- profile::tests`
Expected: FAIL — module doesn't exist yet

- [ ] **Step 3: Implement profile.rs**

```rust
//! Pipeline profiling instrumentation.
//!
//! The [`Profiler`] enum provides real-time timing output for each pipeline
//! phase. When disabled ([`Profiler::Noop`]), all methods are no-ops with
//! zero overhead at release optimization levels.

use std::cell::Cell;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Pipeline profiler that prints phase timing to stderr.
///
/// Use [`Profiler::new`] to create an active or noop profiler based on
/// whether `--profile` was passed.
pub enum Profiler {
    /// Actively collects timing and prints to stderr.
    Active {
        /// Wall-clock start of the entire run.
        start: Instant,
        /// Reporting interval for embed progress.
        interval: Duration,
        /// Mutable embed-phase state (sequential access only, Mutex for Sync).
        embed: Mutex<EmbedState>,
        /// Per-rayon-thread chunk counts (parallel access during chunk phase).
        chunk_counts: Mutex<Vec<usize>>,
    },
    /// No-op profiler. All methods are empty.
    Noop,
}

struct EmbedState {
    phase_start: Instant,
    last_report: Instant,
    chunks_at_last_report: usize,
    total_lock_wait: Duration,
    total_inference: Duration,
    total_chunks: usize,
}

impl EmbedState {
    fn new() -> Self {
        let now = Instant::now();
        Self {
            phase_start: now,
            last_report: now,
            chunks_at_last_report: 0,
            total_lock_wait: Duration::ZERO,
            total_inference: Duration::ZERO,
            total_chunks: 0,
        }
    }
}

impl Profiler {
    /// Create a new profiler. If `enabled` is false, returns `Noop`.
    #[must_use]
    pub fn new(enabled: bool, interval: Duration) -> Self {
        if enabled {
            Self::Active {
                start: Instant::now(),
                interval,
                embed: Mutex::new(EmbedState::new()),
                chunk_counts: Mutex::new(Vec::new()),
            }
        } else {
            Self::Noop
        }
    }

    /// Create a no-op profiler.
    #[must_use]
    pub fn noop() -> Self {
        Self::Noop
    }

    /// Print the system info header line.
    pub fn header(&self, version: &str, model_repo: &str, threads: usize, cores: usize) {
        if let Self::Active { .. } = self {
            eprintln!(
                "[profile] ripvec {} | {}-core | rayon: {} threads | model: {}",
                version,
                cores,
                threads,
                model_repo,
            );
        }
    }

    /// Start timing a named phase. Returns a guard that prints on drop.
    #[must_use]
    pub fn phase(&self, name: &'static str) -> PhaseGuard<'_> {
        PhaseGuard {
            profiler: self,
            name,
            start: Instant::now(),
            detail: Cell::new(None),
        }
    }

    /// Record that a rayon thread produced `n` chunks during the chunk phase.
    pub fn chunk_thread_report(&self, n: usize) {
        if let Self::Active { chunk_counts, .. } = self {
            if let Ok(mut counts) = chunk_counts.lock() {
                let idx = rayon::current_thread_index().unwrap_or(0);
                if counts.len() <= idx {
                    counts.resize(idx + 1, 0);
                }
                counts[idx] += n;
            }
        }
    }

    /// Print the chunk phase summary with thread utilization stats.
    pub fn chunk_summary(&self, total_chunks: usize, total_files: usize, elapsed: Duration) {
        if let Self::Active { start, chunk_counts, .. } = self {
            let wall = start.elapsed();
            if let Ok(counts) = chunk_counts.lock() {
                let active = counts.iter().filter(|&&c| c > 0).count();
                let pool_size = rayon::current_num_threads();
                if active > 0 {
                    let min = counts.iter().filter(|&&c| c > 0).min().copied().unwrap_or(0);
                    let max = counts.iter().max().copied().unwrap_or(0);
                    eprintln!(
                        "[{:.1}s]  chunk: {} chunks from {} files in {:.0?} ({} threads, {} active, skew: {}-{} chunks/thread)",
                        wall.as_secs_f64(), total_chunks, total_files, elapsed, pool_size, active, min, max,
                    );
                } else {
                    eprintln!(
                        "[{:.1}s]  chunk: {} chunks from {} files in {:.0?} ({} threads)",
                        wall.as_secs_f64(), total_chunks, total_files, elapsed, pool_size,
                    );
                }
            }
        }
    }

    /// Begin the embed phase. Call before the embedding loop.
    pub fn embed_begin(&self, total: usize) {
        if let Self::Active { embed, .. } = self {
            if let Ok(mut state) = embed.lock() {
                let now = Instant::now();
                state.phase_start = now;
                state.last_report = now;
                state.chunks_at_last_report = 0;
                state.total_lock_wait = Duration::ZERO;
                state.total_inference = Duration::ZERO;
                state.total_chunks = total;
            }
        }
    }

    /// Called after each chunk is embedded. Prints periodic progress.
    pub fn embed_tick(&self, done: usize) {
        if let Self::Active { start, interval, embed, .. } = self {
            if let Ok(mut state) = embed.lock() {
                let now = Instant::now();
                if now.duration_since(state.last_report) >= *interval {
                    let wall = start.elapsed();
                    let overall_elapsed = now.duration_since(state.phase_start).as_secs_f64();
                    let window_elapsed = now.duration_since(state.last_report).as_secs_f64();
                    let window_chunks = done - state.chunks_at_last_report;
                    let overall_rate = if overall_elapsed > 0.0 { done as f64 / overall_elapsed } else { 0.0 };
                    let window_rate = if window_elapsed > 0.0 { window_chunks as f64 / window_elapsed } else { 0.0 };
                    let total_timing = state.total_lock_wait + state.total_inference;
                    let lock_pct = if total_timing.as_nanos() > 0 {
                        state.total_lock_wait.as_nanos() as f64 / total_timing.as_nanos() as f64 * 100.0
                    } else { 0.0 };
                    eprintln!(
                        "[{:.1}s]  embed: {}/{} (last {:.0}s: {:.1}/s, overall: {:.1}/s) lock_wait: {:.0}% inference: {:.0}%",
                        wall.as_secs_f64(), done, state.total_chunks,
                        window_elapsed, window_rate, overall_rate,
                        lock_pct, 100.0 - lock_pct,
                    );
                    state.last_report = now;
                    state.chunks_at_last_report = done;
                }
            }
        }
    }

    /// Accumulate time spent waiting for the model mutex lock.
    pub fn embed_lock_wait(&self, duration: Duration) {
        if let Self::Active { embed, .. } = self {
            if let Ok(mut state) = embed.lock() {
                state.total_lock_wait += duration;
            }
        }
    }

    /// Accumulate time spent in ONNX inference.
    pub fn embed_inference(&self, duration: Duration) {
        if let Self::Active { embed, .. } = self {
            if let Ok(mut state) = embed.lock() {
                state.total_inference += duration;
            }
        }
    }

    /// Print the final embed phase summary.
    pub fn embed_done(&self) {
        if let Self::Active { start, embed, .. } = self {
            let wall = start.elapsed();
            if let Ok(state) = embed.lock() {
                if state.total_chunks == 0 {
                    eprintln!("[{:.1}s]  embed: skipped (0 chunks)", wall.as_secs_f64());
                    return;
                }
                let elapsed = Instant::now().duration_since(state.phase_start);
                let rate = if elapsed.as_secs_f64() > 0.0 {
                    state.total_chunks as f64 / elapsed.as_secs_f64()
                } else { 0.0 };
                let total_timing = state.total_lock_wait + state.total_inference;
                let lock_pct = if total_timing.as_nanos() > 0 {
                    state.total_lock_wait.as_nanos() as f64 / total_timing.as_nanos() as f64 * 100.0
                } else { 0.0 };
                eprintln!(
                    "[{:.1}s]  embed: {}/{} done in {:.1}s ({:.1}/s) lock_wait: {:.0}% inference: {:.0}%",
                    wall.as_secs_f64(), state.total_chunks, state.total_chunks,
                    elapsed.as_secs_f64(), rate, lock_pct, 100.0 - lock_pct,
                );
            }
        }
    }

    /// Print the total wall-clock time.
    pub fn finish(&self) {
        if let Self::Active { start, .. } = self {
            eprintln!("[{:.1}s]  total: {:.1}s", start.elapsed().as_secs_f64(), start.elapsed().as_secs_f64());
        }
    }
}

/// Guard returned by [`Profiler::phase`]. Prints elapsed time on drop.
pub struct PhaseGuard<'a> {
    profiler: &'a Profiler,
    name: &'static str,
    start: Instant,
    detail: Cell<Option<String>>,
}

impl PhaseGuard<'_> {
    /// Attach a detail string printed alongside the phase timing.
    pub fn set_detail(&self, detail: String) {
        self.detail.set(Some(detail));
    }
}

impl Drop for PhaseGuard<'_> {
    fn drop(&mut self) {
        if let Profiler::Active { start, .. } = self.profiler {
            let elapsed = self.start.elapsed();
            let wall = start.elapsed();
            if let Some(detail) = self.detail.take() {
                eprintln!(
                    "[{:.3}s] {}: {} in {:.1?}",
                    wall.as_secs_f64(), self.name, detail, elapsed,
                );
            } else {
                eprintln!(
                    "[{:.3}s] {}: {:.1?}",
                    wall.as_secs_f64(), self.name, elapsed,
                );
            }
        }
    }
}
```

- [ ] **Step 4: Add `pub mod profile;` to lib.rs**

- [ ] **Step 5: Run tests**

Run: `cargo test -p ripvec-core -- profile::tests`
Expected: 5 tests PASS

- [ ] **Step 6: Verify compiles**

Run: `cargo check --workspace`
Expected: SUCCESS

- [ ] **Step 7: Commit**

```bash
git add crates/ripvec-core/src/profile.rs crates/ripvec-core/src/lib.rs
git commit -m "feat(core): add Profiler with phase timing and embed progress"
```

---

### Task 2: Integrate Profiler into embed.rs

**Files:**
- Modify: `crates/ripvec-core/src/embed.rs`

- [ ] **Step 1: Update `search()` signature**

Add `profiler: &crate::profile::Profiler` as the last parameter. Update the doc comment to mention it.

- [ ] **Step 2: Instrument Phase 1 (walk)**

Replace:
```rust
let files = crate::walk::collect_files(root);
```
With:
```rust
let files = {
    let guard = profiler.phase("walk");
    let files = crate::walk::collect_files(root);
    guard.set_detail(format!("{} files", files.len()));
    files
};
```

- [ ] **Step 3: Instrument Phase 2 (chunk) with rayon thread tracking**

Use manual timing (no PhaseGuard) so chunk_summary can print everything on one line matching the spec format:

```rust
let chunk_start = std::time::Instant::now();
let chunks: Vec<CodeChunk> = files
    .par_iter()
    .flat_map(|path| {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        let Some(config) = crate::languages::config_for_extension(ext) else {
            return vec![];
        };
        let Ok(source) = std::fs::read_to_string(path) else {
            return vec![];
        };
        let chunks = crate::chunk::chunk_file(path, &source, &config);
        profiler.chunk_thread_report(chunks.len());
        chunks
    })
    .collect();
profiler.chunk_summary(chunks.len(), files.len(), chunk_start.elapsed());
```

- [ ] **Step 4: Update `embed_text()` signature and add timing (must be done before Steps 5-6)**

Change signature to add `profiler: &crate::profile::Profiler` as the last parameter.
Add timing around lock acquisition and inference:

```rust
fn embed_text(
    text: &str,
    model: &Mutex<EmbeddingModel>,
    tokenizer: &tokenizers::Tokenizer,
    profiler: &crate::profile::Profiler,
) -> crate::Result<Vec<f32>> {
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| crate::Error::Tokenization(e.to_string()))?;
    let ids: Vec<i64> = encoding.get_ids().iter().map(|&x| i64::from(x)).collect();
    let mask: Vec<i64> = encoding
        .get_attention_mask()
        .iter()
        .map(|&x| i64::from(x))
        .collect();
    let type_ids: Vec<i64> = encoding
        .get_type_ids()
        .iter()
        .map(|&x| i64::from(x))
        .collect();

    let lock_start = std::time::Instant::now();
    let mut guard = model
        .lock()
        .map_err(|e| crate::Error::Other(anyhow::anyhow!("model mutex poisoned: {e}")))?;
    profiler.embed_lock_wait(lock_start.elapsed());

    let infer_start = std::time::Instant::now();
    let result = guard.embed(&ids, &mask, &type_ids);
    profiler.embed_inference(infer_start.elapsed());

    result
}
```

- [ ] **Step 5: Instrument Phase 3 (embed query)**

Replace:
```rust
let query_embedding = embed_text(query, model, tokenizer)?;
```
With:
```rust
let query_embedding = {
    let _guard = profiler.phase("embed_query");
    embed_text(query, model, tokenizer, profiler)?
};
```

- [ ] **Step 6: Instrument Phase 4 (embed chunks)**

Replace the chunk embedding loop with:

```rust
profiler.embed_begin(chunks.len());
let mut results: Vec<SearchResult> = chunks
    .iter()
    .enumerate()
    .filter_map(|(i, chunk)| {
        let emb = embed_text(&chunk.content, model, tokenizer, profiler).ok()?;
        profiler.embed_tick(i + 1);
        let sim = similarity::dot_product(&query_embedding, &emb);
        Some(SearchResult {
            chunk: chunk.clone(),
            similarity: sim,
        })
    })
    .collect();
profiler.embed_done();
```

- [ ] **Step 7: Instrument Phase 5 (rank)**

Wrap the sort + truncate:

```rust
{
    let guard = profiler.phase("rank");
    results.sort_unstable_by(|a, b| {
        b.similarity
            .partial_cmp(&a.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(top_k);
    guard.set_detail(format!("top {} from {}", top_k.min(results.len()), chunks.len()));
}
```

- [ ] **Step 8: Verify compiles**

Run: `cargo check --workspace`
Expected: SUCCESS (the CLI won't pass a profiler yet — but ripvec-core compiles independently)

Wait — `search()` signature changed, so CLI `main.rs` and MCP `main.rs` won't compile. We need to update callers too. This is handled in Task 3, but `cargo check --workspace` will fail until then. Run `cargo check -p ripvec-core` instead.

Run: `cargo check -p ripvec-core`
Expected: SUCCESS

- [ ] **Step 9: Run tests**

Run: `cargo test -p ripvec-core`
Expected: All tests PASS (profile tests + existing tests)

- [ ] **Step 10: Commit**

```bash
git add crates/ripvec-core/src/embed.rs
git commit -m "feat(core): instrument search pipeline with profiler"
```

---

### Task 3: CLI Flags + main.rs Wiring

**Files:**
- Modify: `crates/ripvec/src/cli.rs`
- Modify: `crates/ripvec/src/main.rs`
- Modify: `crates/ripvec-mcp/src/main.rs`

- [ ] **Step 1: Add CLI flags to cli.rs**

Add these fields to the `Args` struct:

```rust
    /// Enable pipeline profiling output to stderr.
    #[arg(long)]
    pub profile: bool,

    /// Profiling report interval in seconds.
    #[arg(long, default_value_t = 10.0)]
    pub profile_interval: f64,
```

- [ ] **Step 2: Update main.rs**

Replace the entire main function body with:

```rust
fn main() -> Result<()> {
    let args = cli::Args::parse();

    // Create profiler
    let profiler = ripvec_core::profile::Profiler::new(
        args.profile,
        std::time::Duration::from_secs_f64(args.profile_interval),
    );

    // Configure thread pool
    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .context("failed to configure thread pool")?;
    }

    // Print profiler header
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    profiler.header(env!("CARGO_PKG_VERSION"), &args.model_repo, rayon::current_num_threads(), cores);

    // Load model and tokenizer
    let model = {
        let _guard = profiler.phase("model_load");
        ripvec_core::model::EmbeddingModel::load(&args.model_repo, &args.model_file)
            .context("failed to load embedding model")?
    };
    let model = Mutex::new(model);
    let tokenizer = ripvec_core::tokenize::load_tokenizer(&args.model_repo)
        .context("failed to load tokenizer")?;

    // Run search
    let results = ripvec_core::embed::search(
        std::path::Path::new(&args.path),
        &args.query,
        &model,
        &tokenizer,
        args.top_k,
        &profiler,
    )
    .context("search failed")?;

    profiler.finish();

    // Filter by threshold and print
    let filtered: Vec<_> = results
        .into_iter()
        .filter(|r| r.similarity >= args.threshold)
        .collect();
    output::print_results(&filtered, &args.format);

    Ok(())
}
```

- [ ] **Step 3: Update ripvec-mcp/src/main.rs**

The MCP server's `semantic_search` tool also calls `ripvec_core::embed::search`. Pass a `Profiler::noop()` since MCP doesn't support profiling output.

Find the `search()` call in the `spawn_blocking` closure and add `&ripvec_core::profile::Profiler::noop()` as the last argument.

- [ ] **Step 4: Verify workspace compiles**

Run: `cargo check --workspace`
Expected: SUCCESS

- [ ] **Step 5: Run all tests**

Run: `cargo test --workspace`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add crates/ripvec/src/cli.rs crates/ripvec/src/main.rs crates/ripvec-mcp/src/main.rs
git commit -m "feat(cli): add --profile flag with pipeline timing output"
```

---

### Task 4: Verification + Clippy

**Files:** None new

- [ ] **Step 1: Run fmt**

Run: `cargo fmt --check`
Expected: PASS

- [ ] **Step 2: Run clippy**

Run: `cargo clippy --all-targets -- -D warnings`
Expected: PASS (fix any warnings)

- [ ] **Step 3: Run all tests**

Run: `cargo test --workspace`
Expected: All tests PASS

- [ ] **Step 4: Manual smoke test**

Run: `cargo run --release --bin ripvec -- "embedding model" ./crates -n 3 --profile`
Expected: Profiling output on stderr, search results on stdout. Verify:
- Header line with core count and thread count
- `model_load` phase timing
- `walk` with file count
- `chunk` with thread utilization
- `embed_query` timing
- `embed` progress (may be too fast for periodic output on small codebase)
- `rank` timing
- `total` wall clock

- [ ] **Step 5: Commit if any fixes were needed**

```bash
git add -A && git commit -m "fix: resolve clippy warnings in profiling code"
```

- [ ] **Step 6: Reinstall**

Run: `cargo install --path crates/ripvec`
