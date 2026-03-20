# Hybrid Distributed Scheduler — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run all available embedding backends concurrently with work-stealing so idle CPU cores assist the GPU, auto-detecting the best backends at runtime.

**Architecture:** A shared `AtomicUsize` cursor over pre-tokenized chunks. Each backend gets a worker thread that grabs chunks, embeds them, and writes results by original index. Adaptive grab sizing lets fast backends claim more work. `detect_backends()` probes hardware and returns all available backends.

**Tech Stack:** Rust 2024, std::thread::scope, AtomicUsize, existing EmbedBackend trait, tracemeld for profiling

---

## File Structure

```
crates/ripvec-core/src/
├── embed.rs              # Add embed_distributed(), change embed_all/search signatures
├── backend/
│   └── mod.rs            # Add detect_backends() factory function
crates/ripvec/src/
├── main.rs               # load_pipeline returns Vec<Box<dyn EmbedBackend>>
├── tui/
│   └── mod.rs            # App.backends: Vec<Box<dyn EmbedBackend>>
```

---

### Task 1: Add `detect_backends()` to backend/mod.rs

**Files:**
- Modify: `crates/ripvec-core/src/backend/mod.rs`

- [ ] **Step 1: Write tests**

```rust
#[test]
fn detect_backends_returns_at_least_one() {
    let backends = detect_backends("BAAI/bge-small-en-v1.5").unwrap();
    assert!(!backends.is_empty(), "must detect at least CPU backend");
}

#[test]
fn detect_backends_cpu_always_present() {
    let backends = detect_backends("BAAI/bge-small-en-v1.5").unwrap();
    // Last backend should be CPU (it's the fallback)
    let last = backends.last().unwrap();
    assert!(!last.is_gpu(), "last backend should be CPU fallback");
}
```

- [ ] **Step 2: Implement `detect_backends`**

```rust
/// Detect all available backends and load them.
///
/// Probes for GPU backends (MLX, CUDA) first, then always adds CPU
/// as the baseline. Returns backends in priority order — the first
/// entry is the primary (used for query embedding).
///
/// # Errors
///
/// Returns an error if no backends can be loaded (not even CPU).
pub fn detect_backends(model_repo: &str) -> crate::Result<Vec<Box<dyn EmbedBackend>>> {
    let mut backends: Vec<Box<dyn EmbedBackend>> = Vec::new();

    // Try MLX (Apple Silicon GPU)
    #[cfg(feature = "mlx")]
    if let Ok(b) = mlx::MlxBackend::load(model_repo, &DeviceHint::Auto) {
        backends.push(Box::new(b));
    }

    // Try CUDA (each device — future: enumerate with candle/cuda)
    // #[cfg(feature = "cuda")]
    // for device_id in 0..cuda_device_count() { ... }

    // Always add CPU as helper/fallback
    if let Ok(b) = candle::CandleBackend::load(model_repo, &DeviceHint::Cpu) {
        backends.push(Box::new(b));
    }

    if backends.is_empty() {
        return Err(crate::Error::Other(anyhow::anyhow!(
            "no embedding backends available"
        )));
    }

    Ok(backends)
}
```

- [ ] **Step 3: Verify**

Use MCP: `cargo-check --workspace`, `cargo-clippy --all-targets -D warnings`, `cargo-test -p ripvec-core backend`

- [ ] **Step 4: Commit**

`git commit -m "feat: add detect_backends for auto-detection of available hardware"`

---

### Task 2: Implement `embed_distributed()`

**Files:**
- Modify: `crates/ripvec-core/src/embed.rs`

- [ ] **Step 1: Write tests**

```rust
#[test]
fn embed_distributed_single_backend_matches_gpu_pipelined() {
    // Load one backend, embed same chunks with both paths, compare
    let backend = crate::backend::load_backend(
        crate::backend::BackendKind::Candle,
        "BAAI/bge-small-en-v1.5",
        crate::backend::DeviceHint::Cpu,
    ).unwrap();
    let tokenizer = crate::tokenize::load_tokenizer("BAAI/bge-small-en-v1.5").unwrap();
    let profiler = crate::profile::Profiler::noop();

    // Small fixture
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let cfg = SearchConfig::default();
    let files = crate::walk::collect_files(&dir);
    // ... chunk, tokenize, embed with both paths, compare results
}
```

- [ ] **Step 2: Implement `embed_distributed`**

Key structure:

```rust
fn embed_distributed(
    tokenized: &[Option<Encoding>],
    backends: &[&dyn EmbedBackend],
    batch_size: usize,
    profiler: &crate::profile::Profiler,
) -> crate::Result<Vec<Vec<f32>>> {
    let n = tokenized.len();
    let cursor = std::sync::atomic::AtomicUsize::new(0);
    let done_counter = std::sync::atomic::AtomicUsize::new(0);
    let error_flag = std::sync::atomic::AtomicBool::new(false);
    let first_error: std::sync::Mutex<Option<crate::Error>> = std::sync::Mutex::new(None);

    // Pre-allocate results indexed by original position
    // Each slot written exactly once by one worker
    let results: Vec<std::sync::Mutex<Vec<f32>>> =
        (0..n).map(|_| std::sync::Mutex::new(vec![])).collect();

    std::thread::scope(|s| {
        for backend in backends {
            let grab_size = if backend.is_gpu() {
                batch_size * 4  // GPU: large batches
            } else {
                batch_size      // CPU: smaller batches
            };

            s.spawn(|| {
                // CPU backends clone for thread-local use
                let local_backend: Box<dyn EmbedBackend>;
                let backend_ref: &dyn EmbedBackend = if backend.supports_clone() {
                    local_backend = backend.clone_backend();
                    &*local_backend
                } else {
                    *backend
                };

                loop {
                    if error_flag.load(std::sync::atomic::Ordering::Relaxed) {
                        break;
                    }
                    let start = cursor.fetch_add(grab_size, std::sync::atomic::Ordering::Relaxed);
                    if start >= n {
                        break;
                    }
                    let end = n.min(start + grab_size);
                    let batch_slice = &tokenized[start..end];

                    // Extract valid encodings + track mask
                    let mut valid = Vec::new();
                    let mut positions = Vec::new();
                    for (offset, enc) in batch_slice.iter().enumerate() {
                        if let Some(e) = enc {
                            valid.push(Encoding {
                                input_ids: e.input_ids.clone(),
                                attention_mask: e.attention_mask.clone(),
                                token_type_ids: e.token_type_ids.clone(),
                            });
                            positions.push(start + offset);
                        }
                    }

                    if valid.is_empty() {
                        let done = done_counter.fetch_add(end - start, Relaxed) + (end - start);
                        profiler.embed_tick(done);
                        continue;
                    }

                    match backend_ref.embed_batch(&valid) {
                        Ok(embeddings) => {
                            for (emb, &pos) in embeddings.into_iter().zip(positions.iter()) {
                                if let Ok(mut slot) = results[pos].lock() {
                                    *slot = emb;
                                }
                            }
                            let done = done_counter.fetch_add(end - start, Relaxed) + (end - start);
                            profiler.embed_tick(done);
                        }
                        Err(e) => {
                            error_flag.store(true, Relaxed);
                            if let Ok(mut guard) = first_error.lock() {
                                if guard.is_none() { *guard = Some(e); }
                            }
                            break;
                        }
                    }
                }
            });
        }
    });

    if let Some(err) = first_error.into_inner().ok().flatten() {
        return Err(err);
    }

    // Collect results, unwrapping mutexes
    Ok(results.into_iter()
        .map(|m| m.into_inner().unwrap_or_default())
        .collect())
}
```

- [ ] **Step 3: Verify tests pass**

- [ ] **Step 4: Commit**

`git commit -m "feat: embed_distributed with work-stealing across N backends"`

---

### Task 3: Change `embed_all` and `search` to accept multiple backends

**Files:**
- Modify: `crates/ripvec-core/src/embed.rs`

- [ ] **Step 1: Change `embed_all` signature**

```rust
pub fn embed_all(
    root: &Path,
    backends: &[&dyn EmbedBackend],
    tokenizer: &tokenizers::Tokenizer,
    cfg: &SearchConfig,
    profiler: &crate::profile::Profiler,
) -> crate::Result<(Vec<CodeChunk>, Vec<Vec<f32>>)>
```

Internally: pre-tokenize all chunks, then call `embed_distributed(tokenized, backends, ...)`. Remove the old `is_gpu` routing and `embed_cpu_parallel` / `embed_gpu_pipelined` calls. Keep those functions for now (used by tests) but the main path goes through `embed_distributed`.

- [ ] **Step 2: Change `search` signature**

Same: `backends: &[&dyn EmbedBackend]`. Query embedding uses `backends[0].embed_batch(&[enc])`.

- [ ] **Step 3: Update all callers** — compile errors guide you:
  - `crates/ripvec/src/main.rs`: `run_oneshot`, `run_interactive`
  - `crates/ripvec-mcp/src/main.rs`: MCP server
  - `crates/ripvec-core/src/embed.rs`: test `search_with_backend_trait`

- [ ] **Step 4: Verify**

MCP: `cargo-check --workspace --all-targets`, `cargo-test --workspace`

- [ ] **Step 5: Commit**

`git commit -m "refactor: embed_all and search accept multiple backends"`

---

### Task 4: Update CLI with auto-detect default

**Files:**
- Modify: `crates/ripvec/src/cli.rs`
- Modify: `crates/ripvec/src/main.rs`

- [ ] **Step 1: Add `Auto` variant to `BackendArg`**

```rust
#[derive(clap::ValueEnum, Clone, Debug, Default)]
pub enum BackendArg {
    /// Auto-detect best available backends (default).
    #[default]
    Auto,
    /// Candle (pure-Rust, CPU + Metal + CUDA).
    Candle,
    /// MLX (Apple Silicon, macOS only).
    Mlx,
    /// ONNX Runtime (cross-platform, CPU + GPU).
    Ort,
}
```

Change default from `"candle"` to `"auto"`.

- [ ] **Step 2: Update `load_pipeline` in main.rs**

Replace single-backend loading with:

```rust
let backends: Vec<Box<dyn ripvec_core::backend::EmbedBackend>> = match args.backend {
    cli::BackendArg::Auto => {
        ripvec_core::backend::detect_backends(&args.model_repo)
            .context("failed to detect backends")?
    }
    other => {
        let kind = match other {
            cli::BackendArg::Candle => BackendKind::Candle,
            cli::BackendArg::Mlx => BackendKind::Mlx,
            cli::BackendArg::Ort => BackendKind::Ort,
            cli::BackendArg::Auto => unreachable!(),
        };
        vec![load_backend(kind, &args.model_repo, device_hint)?]
    }
};
```

Return `Vec<Box<dyn EmbedBackend>>` instead of single `Box`.

- [ ] **Step 3: Update `run_oneshot` and `run_interactive`** to pass `&backends.iter().map(AsRef::as_ref).collect::<Vec<_>>()` to `embed_all`/`search`.

- [ ] **Step 4: Verify**

MCP: `cargo-check --workspace`, `cargo-test --workspace`

- [ ] **Step 5: Commit**

`git commit -m "feat: auto-detect backends by default, --backend overrides"`

---

### Task 5: Update TUI for multiple backends

**Files:**
- Modify: `crates/ripvec/src/tui/mod.rs`
- Modify: `crates/ripvec/src/main.rs` (run_interactive)

- [ ] **Step 1: Change `App.backend` to `App.backends`**

```rust
pub backends: Vec<Box<dyn EmbedBackend>>,
```

- [ ] **Step 2: Update `rerank()` in tui/mod.rs**

Use `self.backends[0]` for query embedding (fastest/primary backend).

- [ ] **Step 3: Update `run_interactive` in main.rs**

Pass `backends` Vec instead of single `backend`.

- [ ] **Step 4: Verify TUI compiles**

MCP: `cargo-check --workspace --all-targets`

- [ ] **Step 5: Commit**

`git commit -m "refactor: TUI accepts multiple backends for hybrid mode"`

---

### Task 6: Profile and validate with tracemeld

**Files:** None (profiling only)

- [ ] **Step 1: Baseline profile (single MLX backend)**

```bash
./target/release/ripvec "error handling" tests/corpus/code/flask --backend mlx --profile -n 3 --format plain
```

Record rate and wall time.

- [ ] **Step 2: Profile with samply**

```bash
samply record --save-only --unstable-presymbolicate -o /tmp/ripvec-hybrid.json.gz \
  ./target/release/ripvec "error handling" tests/corpus/code/flask --profile -n 3 --format plain
```

- [ ] **Step 3: Import into tracemeld**

MCP: `import_profile` source="/tmp/ripvec-hybrid.json.gz" format="gecko"

- [ ] **Step 4: Analyze**

MCP: `profile_summary` group_by="lane", `bottleneck` dimension="wall_ms" top_n=10

Verify:
- CPU threads are no longer 85% sleeping
- Both GPU and CPU lanes show embedding work
- Wall time decreased vs baseline

- [ ] **Step 5: Commit results as memory update**

`git commit -m "perf: validate hybrid scheduler via tracemeld profiling"`

---

### Task 7: Remove dead code and cleanup

**Files:**
- Modify: `crates/ripvec-core/src/embed.rs`

- [ ] **Step 1: Remove `embed_cpu_parallel` and `embed_gpu_pipelined`** — they are superseded by `embed_distributed`. Keep only if tests reference them directly (update tests to use `embed_distributed` instead).

- [ ] **Step 2: Remove `SortOrder` enum and `sort_order` field** from `SearchConfig` — it was dead code (identified in code review, never applied).

- [ ] **Step 3: Run full verification**

MCP: `cargo-fmt`, `cargo-clippy --all-targets -D warnings`, `cargo-test --workspace`

- [ ] **Step 4: Commit**

`git commit -m "refactor: remove dead scheduling code, clean up embed.rs"`

---

## Risk Assessment

| Task | Risk | Mitigation |
|---|---|---|
| Task 1 (detect) | Low — simple factory | Fallback to CPU always |
| Task 2 (distributed) | Medium — concurrency | AtomicUsize is well-understood; thread::scope prevents leaks |
| Task 3 (API change) | Low — compiler-guided | Follow compile errors |
| Task 4 (CLI default) | Low — additive | Auto falls back to single backend |
| Task 5 (TUI) | Low — mechanical | Just change field type |
| Task 6 (profiling) | Low — measurement only | No code changes |
| Task 7 (cleanup) | Low — removal | Tests catch regressions |
