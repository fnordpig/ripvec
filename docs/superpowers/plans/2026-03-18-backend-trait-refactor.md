# Backend Trait Refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the embedding pipeline into a trait-based backend architecture so candle, MLX, and ORT can be swapped at runtime via `--backend candle|mlx|ort`.

**Architecture:** Define an `EmbedBackend` trait in `ripvec-core` that encapsulates model loading, batch embedding, and cloneability for parallel scheduling. The scheduling layer (`embed_cpu_parallel` / `embed_gpu_pipelined`) becomes generic over the backend. Each backend lives behind a cargo feature flag so unused backends don't bloat the binary.

**Tech Stack:** Rust 2024 edition, candle 0.9, mlx-rs 0.25, ort 2.x, rayon, tokenizers

---

## File Structure

```
crates/ripvec-core/src/
├── backend/
│   ├── mod.rs          # EmbedBackend trait + Encoding type + factory fn
│   ├── candle.rs       # CandleBackend (existing model.rs code, refactored)
│   ├── mlx.rs          # MlxBackend (new, Apple Silicon via mlx-rs)
│   └── ort.rs          # OrtBackend (restored from ed0c9f1, adapted to trait)
├── embed.rs            # Scheduling layer, generic over EmbedBackend
├── error.rs            # Add backend-specific error variants
├── lib.rs              # Add `pub mod backend;`, remove `pub mod model;`
└── ... (unchanged)

crates/ripvec/src/
├── cli.rs              # Add --backend arg (candle|mlx|ort)
└── main.rs             # Route backend selection through factory
```

Key decisions:
- `Encoding` moves from `model.rs` to `backend/mod.rs` (shared across all backends)
- `model.rs` is replaced by `backend/candle.rs` — no breaking change to the trait surface
- `embed.rs` scheduling functions become generic: `fn embed_parallel<B: EmbedBackend>(...)`
- Feature flags: `candle` (default), `mlx` (macOS only), `ort`
- `tokenize()` stays in `embed.rs` — tokenization is backend-independent

---

### Task 1: Define the `EmbedBackend` trait

**Files:**
- Create: `crates/ripvec-core/src/backend/mod.rs`
- Modify: `crates/ripvec-core/src/lib.rs`

- [ ] **Step 1: Write the failing test**

```rust
// crates/ripvec-core/src/backend/mod.rs
#[cfg(test)]
mod tests {
    use super::*;

    // Verify the trait is object-safe (can be used as dyn)
    fn _assert_object_safe(_: &dyn EmbedBackend) {}

    // Verify Encoding can be constructed
    #[test]
    fn encoding_round_trip() {
        let enc = Encoding {
            input_ids: vec![101, 2023, 102],
            attention_mask: vec![1, 1, 1],
            token_type_ids: vec![0, 0, 0],
        };
        assert_eq!(enc.input_ids.len(), 3);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p ripvec-core backend::tests`
Expected: FAIL — module doesn't exist

- [ ] **Step 3: Write the trait and Encoding**

```rust
// crates/ripvec-core/src/backend/mod.rs
//! Embedding backend trait and shared types.
//!
//! Each backend (candle, MLX, ORT) implements [`EmbedBackend`] to provide
//! model loading and batch inference. The scheduling layer in [`crate::embed`]
//! is generic over this trait.

#[cfg(feature = "candle")]
pub mod candle;
#[cfg(feature = "mlx")]
pub mod mlx;
#[cfg(feature = "ort")]
pub mod ort;

/// Pre-tokenized encoding ready for inference.
///
/// Backend-independent representation of tokenized text.
/// All backends consume this same type.
pub struct Encoding {
    /// Token IDs.
    pub input_ids: Vec<i64>,
    /// Attention mask (1 for real tokens, 0 for padding).
    pub attention_mask: Vec<i64>,
    /// Token type IDs (0 for single-sequence models).
    pub token_type_ids: Vec<i64>,
}

/// Trait for embedding backends.
///
/// Backends handle model loading and batch inference. The scheduling
/// layer clones the backend per rayon thread for CPU parallelism,
/// or uses it single-threaded for GPU pipelining.
pub trait EmbedBackend: Send + Sync {
    /// Embed a batch of tokenized inputs, returning L2-normalized vectors.
    ///
    /// Each output vector has `hidden_dim` elements (e.g., 384 for bge-small).
    /// Empty input returns an empty vec.
    fn embed_batch(&self, encodings: &[Encoding]) -> crate::Result<Vec<Vec<f32>>>;

    /// Whether this backend supports cheap cloning for per-thread instances.
    ///
    /// CPU backends (candle-cpu, ORT) return `true` — each rayon thread
    /// gets a clone. GPU backends return `false` — use pipelined scheduling.
    fn supports_clone(&self) -> bool;

    /// Clone this backend for use on another thread.
    ///
    /// Only called when `supports_clone()` returns `true`.
    /// Panics if called on a non-cloneable backend.
    fn clone_backend(&self) -> Box<dyn EmbedBackend>;

    /// Whether this backend runs on a GPU/accelerator.
    ///
    /// GPU backends use the pipelined (ring buffer) scheduler.
    /// CPU backends use rayon-parallel scheduling.
    fn is_gpu(&self) -> bool;
}

/// Backend selection for the CLI.
#[derive(Debug, Clone, Copy, Default)]
pub enum BackendKind {
    /// Candle (pure Rust) — CPU with Accelerate/MKL, or Metal/CUDA GPU.
    #[default]
    Candle,
    /// MLX — Apple Silicon native (macOS only).
    Mlx,
    /// ONNX Runtime — cross-platform, multiple execution providers.
    Ort,
}

/// Load a backend by kind.
///
/// # Errors
///
/// Returns an error if the requested backend is not compiled in
/// (missing feature flag) or model loading fails.
pub fn load_backend(
    kind: BackendKind,
    model_repo: &str,
    device_hint: &str,
) -> crate::Result<Box<dyn EmbedBackend>> {
    match kind {
        #[cfg(feature = "candle")]
        BackendKind::Candle => {
            let backend = candle::CandleBackend::load(model_repo, device_hint)?;
            Ok(Box::new(backend))
        }
        #[cfg(not(feature = "candle"))]
        BackendKind::Candle => Err(crate::Error::Other(anyhow::anyhow!(
            "candle backend requires: --features candle"
        ))),
        #[cfg(feature = "mlx")]
        BackendKind::Mlx => {
            let backend = mlx::MlxBackend::load(model_repo)?;
            Ok(Box::new(backend))
        }
        #[cfg(not(feature = "mlx"))]
        BackendKind::Mlx => Err(crate::Error::Other(anyhow::anyhow!(
            "MLX backend requires: --features mlx (macOS only)"
        ))),
        #[cfg(feature = "ort")]
        BackendKind::Ort => {
            let backend = ort::OrtBackend::load(model_repo, device_hint)?;
            Ok(Box::new(backend))
        }
        #[cfg(not(feature = "ort"))]
        BackendKind::Ort => Err(crate::Error::Other(anyhow::anyhow!(
            "ORT backend requires: --features ort"
        ))),
    }
}
```

Update `lib.rs`:
```rust
pub mod backend;
// keep `pub mod model;` as deprecated re-export until callers migrate
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p ripvec-core backend::tests`
Expected: PASS

- [ ] **Step 5: Commit**

```
git add crates/ripvec-core/src/backend/ crates/ripvec-core/src/lib.rs
git commit -m "refactor: define EmbedBackend trait and Encoding type"
```

---

### Task 2: Implement `CandleBackend` (extract from `model.rs`)

**Files:**
- Create: `crates/ripvec-core/src/backend/candle.rs`
- Modify: `crates/ripvec-core/src/backend/mod.rs` (enable import)
- Reference: `crates/ripvec-core/src/model.rs` (source of truth, will be deprecated)

- [ ] **Step 1: Write the failing test**

```rust
// in candle.rs
#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Encoding;

    #[test]
    fn candle_backend_loads_and_embeds() {
        let backend = CandleBackend::load("BAAI/bge-small-en-v1.5", "cpu").unwrap();
        let enc = Encoding {
            input_ids: vec![101, 7592, 102],  // [CLS] hello [SEP]
            attention_mask: vec![1, 1, 1],
            token_type_ids: vec![0, 0, 0],
        };
        let results = backend.embed_batch(&[enc]).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].len(), 384); // bge-small hidden dim
        // L2 normalized — norm should be ~1.0
        let norm: f32 = results[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4);
    }

    #[test]
    fn candle_backend_supports_clone_on_cpu() {
        let backend = CandleBackend::load("BAAI/bge-small-en-v1.5", "cpu").unwrap();
        assert!(backend.supports_clone());
        assert!(!backend.is_gpu());
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p ripvec-core backend::candle::tests`
Expected: FAIL — `CandleBackend` doesn't exist

- [ ] **Step 3: Implement CandleBackend**

Move logic from `model.rs` into `backend/candle.rs`. The struct wraps `Arc<BertModel>` + `Device`. `embed_batch` is the same code as `model::embed_batch`. `clone_backend` returns a new `Box<CandleBackend>` sharing the same `Arc`.

Key changes from `model.rs`:
- `load(model_repo, device_hint)` parses `device_hint` string ("cpu", "metal", "cuda") instead of taking a `DeviceKind` enum
- Implements `EmbedBackend` trait
- `supports_clone()` returns `true` for CPU, `false` for GPU
- `is_gpu()` checks `!matches!(device, Device::Cpu)`

- [ ] **Step 4: Run tests**

Run: `cargo test -p ripvec-core backend::candle::tests`
Expected: PASS

- [ ] **Step 5: Commit**

```
git commit -m "refactor: extract CandleBackend from model.rs"
```

---

### Task 3: Make scheduling layer generic over `EmbedBackend`

**Files:**
- Modify: `crates/ripvec-core/src/embed.rs`

- [ ] **Step 1: Write failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_uses_backend_trait() {
        // This test verifies the public API compiles with the new signature
        let backend = crate::backend::load_backend(
            crate::backend::BackendKind::Candle,
            "BAAI/bge-small-en-v1.5",
            "cpu",
        ).unwrap();
        let tokenizer = crate::tokenize::load_tokenizer("BAAI/bge-small-en-v1.5").unwrap();
        let cfg = SearchConfig::default();
        let profiler = crate::profile::Profiler::noop();
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");

        let results = search(&dir, "test query", backend.as_ref(), &tokenizer, 1, &cfg, &profiler);
        assert!(results.is_ok());
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p ripvec-core embed::tests::search_uses_backend_trait`
Expected: FAIL — `search` still takes `&EmbeddingModel`

- [ ] **Step 3: Refactor `search()` and scheduling functions**

Change signatures:
- `search(... model: &EmbeddingModel ...) → search(... backend: &dyn EmbedBackend ...)`
- `embed_cpu_parallel(... model: &EmbeddingModel ...) → embed_cpu_parallel(... backend: &dyn EmbedBackend ...)`
- `embed_gpu_pipelined(... model: &EmbeddingModel ...) → embed_gpu_pipelined(... backend: &dyn EmbedBackend ...)`

In `search()`, replace:
```rust
let is_gpu = !matches!(model.device(), candle_core::Device::Cpu);
```
with:
```rust
let is_gpu = backend.is_gpu();
```

In `embed_cpu_parallel`, replace:
```rust
.map_init(|| model.clone(), |thread_model, batch| {
    crate::model::embed_batch(thread_model, &encodings)
})
```
with:
```rust
.map_init(|| backend.clone_backend(), |thread_backend, batch| {
    thread_backend.embed_batch(&encodings)
})
```

In `embed_gpu_pipelined`, replace:
```rust
crate::model::embed_batch(model, &valid)
```
with:
```rust
backend.embed_batch(&valid)
```

Remove `use crate::model::{EmbeddingModel, Encoding}` — import from `crate::backend` instead.

- [ ] **Step 4: Run tests**

Run: `cargo test -p ripvec-core`
Expected: PASS

- [ ] **Step 5: Update CLI to use backend factory**

Modify `crates/ripvec/src/cli.rs` — add `--backend candle|mlx|ort` arg.
Modify `crates/ripvec/src/main.rs` — call `load_backend()` and pass to `search()`.

- [ ] **Step 6: Commit**

```
git commit -m "refactor: make scheduling layer generic over EmbedBackend"
```

---

### Task 4: Implement `OrtBackend` (restore from `ed0c9f1`)

**Files:**
- Create: `crates/ripvec-core/src/backend/ort.rs`
- Modify: `crates/ripvec-core/Cargo.toml` (add `ort` feature + deps)
- Reference: `git show ed0c9f1^:crates/ripvec-core/src/model.rs` (ORT code)

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Encoding;

    #[test]
    fn ort_backend_loads_and_embeds() {
        let backend = OrtBackend::load("BAAI/bge-small-en-v1.5", "cpu").unwrap();
        let enc = Encoding {
            input_ids: vec![101, 7592, 102],
            attention_mask: vec![1, 1, 1],
            token_type_ids: vec![0, 0, 0],
        };
        let results = backend.embed_batch(&[enc]).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].len(), 384);
        let norm: f32 = results[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4);
    }
}
```

- [ ] **Step 2: Implement OrtBackend**

Key differences from the old `model.rs`:
- Wraps `Mmap` + `Device` (same as old code)
- `embed_batch` uses `InMemorySession` created per-call or cached
- `supports_clone()` returns `true` (sessions are per-thread)
- `clone_backend()` shares the `Mmap` via `Arc` and creates a new session
- `is_gpu()` returns `true` for CoreML/CUDA

Add to `Cargo.toml`:
```toml
[dependencies]
ort = { version = "2", optional = true }
ndarray = { version = "0.16", optional = true }

[features]
ort = ["dep:ort", "dep:ndarray"]
```

The ONNX model file (`model.onnx`) is different from safetensors — HuggingFace repos typically have both. The old code downloaded `model.onnx` via `hf-hub`.

- [ ] **Step 3: Run tests**

Run: `cargo test -p ripvec-core --features ort backend::ort::tests`
Expected: PASS

- [ ] **Step 4: Commit**

```
git commit -m "feat: restore ORT backend behind --features ort"
```

---

### Task 5: Implement `MlxBackend` (new, Apple Silicon)

**Files:**
- Create: `crates/ripvec-core/src/backend/mlx.rs`
- Modify: `crates/ripvec-core/Cargo.toml` (add `mlx` feature + deps)

- [ ] **Step 1: Add mlx-rs dependency**

```toml
[dependencies]
mlx-rs = { version = "0.25", optional = true }

[features]
mlx = ["dep:mlx-rs"]
```

- [ ] **Step 2: Write the failing test**

Same pattern as candle/ort tests — load model, embed 3 tokens, check output shape and L2 norm.

- [ ] **Step 3: Implement MlxBackend**

This is the most complex task. MLX doesn't have a pre-built BertModel like candle. Options:
1. **Use `mlx-rs` low-level ops** to implement BERT forward pass manually (Linear, LayerNorm, attention, GELU — ~200 lines)
2. **Use `mlx-nn` crate** if it provides Linear/LayerNorm/Embedding modules
3. **Shell out to Python mlx** as a prototype (not recommended for production)

Recommended approach: option 1 with `mlx-rs` ops directly. BERT-small is only 6 layers — the forward pass is straightforward:
```
Embeddings(token + position + type) → LayerNorm →
  6× [SelfAttention → Add+LayerNorm → FFN → Add+LayerNorm] →
CLS pooling → L2 normalize
```

Load weights from safetensors (mlx-rs has optional safetensors support).

Key property: `supports_clone()` returns `false`, `is_gpu()` returns `true` — uses the pipelined scheduler. MLX handles CPU/GPU dispatch internally via its unified memory model.

- [ ] **Step 4: Run tests**

Run: `cargo test -p ripvec-core --features mlx backend::mlx::tests`
Expected: PASS

- [ ] **Step 5: Benchmark against candle CPU**

Run the criterion `bench_embed_batch` benchmark with both backends to compare.

- [ ] **Step 6: Commit**

```
git commit -m "feat: add MLX backend for Apple Silicon"
```

---

### Task 6: Deprecate `model.rs` and clean up

**Files:**
- Modify: `crates/ripvec-core/src/model.rs` (deprecation notice + re-exports)
- Modify: `crates/ripvec-core/src/lib.rs`
- Modify: `crates/ripvec-core/benches/pipeline.rs` (use new backend API)
- Modify: `crates/ripvec-mcp/src/main.rs` (if it uses model.rs directly)

- [ ] **Step 1: Add deprecation re-exports to model.rs**

```rust
//! Deprecated — use [`crate::backend`] instead.
#[deprecated(note = "use crate::backend::Encoding")]
pub use crate::backend::Encoding;
// ... etc
```

- [ ] **Step 2: Update benchmarks to use backend trait**

- [ ] **Step 3: Update MCP server if needed**

- [ ] **Step 4: Run full test suite**

Run: `cargo test --workspace && cargo clippy --all-targets -- -D warnings`
Expected: PASS (possibly with deprecation warnings)

- [ ] **Step 5: Commit**

```
git commit -m "refactor: deprecate model.rs, migrate all callers to backend trait"
```

---

## Feature flag matrix

| Feature | Backends enabled | Default? |
|---|---|---|
| `candle` (includes `accelerate`) | CandleBackend (CPU) | Yes |
| `candle` + `metal` | CandleBackend (CPU + Metal) | No |
| `mlx` | MlxBackend | No |
| `ort` | OrtBackend (CPU) | No |

Multiple backends can be compiled in simultaneously. `--backend` selects at runtime.

## Risk assessment

| Task | Risk | Mitigation |
|---|---|---|
| Task 1-3 (trait + candle + scheduling) | Low — pure refactor, no new deps | Existing tests cover behavior |
| Task 4 (ORT) | Medium — ort 2.x API may differ from old code | Reference ed0c9f1 closely, test with same model |
| Task 5 (MLX) | High — mlx-rs is immature, manual BERT impl | Start with tests, validate against candle output |
| Task 6 (cleanup) | Low — mechanical changes | Run full suite |
