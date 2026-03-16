# ripvec Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build ripvec — a semantic search CLI that searches code, structured text, and plain text by meaning using ONNX embeddings, tree-sitter chunking, and cosine similarity.

**Architecture:** Single-pass pipeline: model bootstrap → directory walk → code chunking → parallel embedding → ranking. Cargo workspace with 3 crates (ripvec-core library, ripvec CLI, ripvec-mcp server). Stateless — no daemon, no database, no persistent index.

**Tech Stack:** ort (ONNX Runtime), tree-sitter (parsing), hf-hub + tokenizers (HuggingFace), rayon (parallelism), ignore (file walking), clap (CLI), rmcp (MCP server)

**Reference:** `design.md` at project root contains the full architectural spec. This plan corrects API calls based on current crate versions (verified via context7 + crates.io on 2026-03-15).

**Key version corrections from design.md:**
- `rmcp`: 0.16 → 1.2.0 (major API change: `#[tool_router]` + `ToolRouter<Self>` pattern)
- `hf-hub`: 0.4.3 → 0.5.0
- `schemars`: 1.0 → 1.2.1
- `ort` output API: `extract_tensor` → `try_extract_array`
- `ort` input API: raw ndarray → `TensorRef::from_array_view`
- Model loading: `commit_from_memory_directly` → `commit_from_file` (simpler, OS page cache still provides fast subsequent runs; mmap zero-copy optimization deferred to future task)
- `tokenizers`: use `Tokenizer::from_file` + hf-hub download (no `http` feature needed)

---

## File Structure

```
ripvec/
├── Cargo.toml                    # Workspace root
├── crates/
│   ├── ripvec-core/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs            # Public API re-exports
│   │       ├── error.rs          # thiserror error types
│   │       ├── model.rs          # ONNX session + embedding
│   │       ├── tokenize.rs       # HuggingFace tokenizer wrapper
│   │       ├── chunk.rs          # Tree-sitter code chunking
│   │       ├── walk.rs           # ignore-based directory traversal
│   │       ├── embed.rs          # Parallel embedding pipeline
│   │       ├── similarity.rs     # Cosine similarity + ranking
│   │       └── languages.rs      # Language registry (extension → grammar)
│   ├── ripvec/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── main.rs           # Entry point
│   │       ├── cli.rs            # clap argument definitions
│   │       └── output.rs         # Result formatting (plain, JSON, colored)
│   └── ripvec-mcp/
│       ├── Cargo.toml
│       └── src/
│           └── main.rs           # MCP server with semantic_search tool
└── tests/
    ├── fixtures/                 # Sample source files for integration tests
    │   ├── sample.rs
    │   ├── sample.py
    │   └── sample.js
    └── integration.rs
```

---

## Chunk 1: Workspace Scaffolding + Error Types

### Task 1: Restructure into Cargo Workspace

**Files:**
- Modify: `Cargo.toml` (root — convert to workspace)
- Create: `crates/ripvec-core/Cargo.toml`
- Create: `crates/ripvec-core/src/lib.rs`
- Create: `crates/ripvec/Cargo.toml`
- Create: `crates/ripvec/src/main.rs`
- Create: `crates/ripvec-mcp/Cargo.toml`
- Create: `crates/ripvec-mcp/src/main.rs`
- Delete: `src/main.rs`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p crates/ripvec-core/src crates/ripvec/src crates/ripvec-mcp/src tests/fixtures
```

- [ ] **Step 2: Write root Cargo.toml**

Replace the entire root `Cargo.toml` with workspace configuration.
All dependency versions are centralized here via `[workspace.dependencies]`.

```toml
[workspace]
members = ["crates/ripvec-core", "crates/ripvec", "crates/ripvec-mcp"]
resolver = "2"

[workspace.package]
edition = "2024"
rust-version = "1.85.0"
license = "MIT OR Apache-2.0"
repository = "https://github.com/fnordpig/repvec"

[workspace.dependencies]
# Core inference
ort = "2.0.0-rc.12"
hf-hub = { version = "0.5.0", default-features = false, features = ["ureq", "rustls-tls"] }
tokenizers = { version = "0.22.2", default-features = false }
ndarray = "0.16"

# Code parsing
tree-sitter = "0.24"
tree-sitter-rust = "0.23"
tree-sitter-python = "0.23"
tree-sitter-javascript = "0.23"
tree-sitter-typescript = "0.23"
tree-sitter-go = "0.23"
tree-sitter-java = "0.23"
tree-sitter-c = "0.23"
tree-sitter-cpp = "0.23"

# Parallelism and file traversal
rayon = "1.10"
ignore = "0.4"

# CLI
clap = { version = "4.5", features = ["derive"] }

# Error handling
thiserror = "2.0"
anyhow = "1.0"

# MCP server
rmcp = { version = "1.2", features = ["server", "transport-io", "macros"] }
tokio = { version = "1", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
schemars = "1.0"

# Testing
assert_cmd = "2.1"
predicates = "3"
tempfile = "3"

[workspace.lints.rust]
unsafe_code = "warn"

[workspace.lints.clippy]
all = { level = "warn", priority = -1 }
pedantic = { level = "warn", priority = -1 }
```

- [ ] **Step 3: Write crate Cargo.toml files**

`crates/ripvec-core/Cargo.toml`:
```toml
[package]
name = "ripvec-core"
version = "0.1.0"
edition.workspace = true
license.workspace = true
description = "Core library for ripvec semantic code search"

[dependencies]
ort.workspace = true
hf-hub.workspace = true
tokenizers.workspace = true
ndarray.workspace = true
tree-sitter.workspace = true
tree-sitter-rust.workspace = true
tree-sitter-python.workspace = true
tree-sitter-javascript.workspace = true
tree-sitter-typescript.workspace = true
tree-sitter-go.workspace = true
tree-sitter-java.workspace = true
tree-sitter-c.workspace = true
tree-sitter-cpp.workspace = true
rayon.workspace = true
ignore.workspace = true
thiserror.workspace = true

[lints]
workspace = true
```

`crates/ripvec/Cargo.toml`:
```toml
[package]
name = "ripvec"
version = "0.1.0"
edition.workspace = true
license.workspace = true
description = "Semantic code search CLI — like ripgrep but for meaning"

[[bin]]
name = "ripvec"
path = "src/main.rs"

[dependencies]
ripvec-core = { path = "../ripvec-core" }
clap.workspace = true
anyhow.workspace = true
rayon.workspace = true

[dev-dependencies]
assert_cmd.workspace = true
predicates.workspace = true
tempfile.workspace = true

[lints]
workspace = true
```

`crates/ripvec-mcp/Cargo.toml`:
```toml
[package]
name = "ripvec-mcp"
version = "0.1.0"
edition.workspace = true
license.workspace = true
description = "MCP server wrapping ripvec for Claude Code / Cursor integration"
publish = false

[[bin]]
name = "ripvec-mcp"
path = "src/main.rs"

[dependencies]
ripvec-core = { path = "../ripvec-core" }
rmcp.workspace = true
tokio.workspace = true
serde.workspace = true
serde_json.workspace = true
schemars.workspace = true
anyhow.workspace = true
tokenizers.workspace = true

[lints]
workspace = true
```

- [ ] **Step 4: Write placeholder source files**

`crates/ripvec-core/src/lib.rs`:
```rust
//! Core library for ripvec semantic code search.
```

`crates/ripvec/src/main.rs`:
```rust
fn main() {
    println!("ripvec: semantic code search");
}
```

`crates/ripvec-mcp/src/main.rs`:
```rust
fn main() {
    println!("ripvec-mcp: semantic search MCP server");
}
```

- [ ] **Step 5: Remove old src/main.rs**

```bash
rm src/main.rs && rmdir src
```

- [ ] **Step 6: Verify workspace compiles**

Run: `cargo check --workspace`
Expected: SUCCESS with no errors

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "refactor: convert to cargo workspace with 3 crates"
```

---

### Task 2: Error Types + lib.rs Exports

**Files:**
- Create: `crates/ripvec-core/src/error.rs`
- Modify: `crates/ripvec-core/src/lib.rs`

- [ ] **Step 1: Write error types**

`crates/ripvec-core/src/error.rs` — see `design.md` "Error handling" section.
Use `thiserror` for structured errors. Include variants: `Download`, `Inference`, `Tokenization`, `Io`, `UnsupportedLanguage`, `Other`.

Add additional variant needed:
- `ShapeError` from ndarray (for array shape mismatches)

```rust
//! Error types for ripvec-core.

use thiserror::Error;

/// Errors that can occur in ripvec-core operations.
#[derive(Error, Debug)]
pub enum Error {
    /// Model download or cache retrieval failed.
    #[error("model download failed: {0}")]
    Download(String),

    /// ONNX Runtime inference failed.
    #[error("ONNX inference failed")]
    Inference(#[from] ort::Error),

    /// Tokenization of input text failed.
    #[error("tokenization failed: {0}")]
    Tokenization(String),

    /// File I/O error with path context.
    #[error("I/O error: {path}")]
    Io {
        /// Path that caused the error.
        path: String,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// Unsupported source file language.
    #[error("unsupported language: {0}")]
    UnsupportedLanguage(String),

    /// ndarray shape mismatch.
    #[error("array shape error")]
    Shape(#[from] ndarray::ShapeError),

    /// Catch-all for other errors.
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}
```

- [ ] **Step 2: Set up lib.rs with module declarations**

```rust
//! Core library for ripvec semantic code search.
//!
//! Provides ONNX embedding model loading, tree-sitter code chunking,
//! parallel embedding, and cosine similarity ranking.

pub mod error;

pub use error::Error;

/// Convenience Result type for ripvec-core.
pub type Result<T> = std::result::Result<T, Error>;
```

- [ ] **Step 3: Verify compiles**

Run: `cargo check -p ripvec-core`
Expected: SUCCESS

- [ ] **Step 4: Commit**

```bash
git add crates/ripvec-core/src/error.rs crates/ripvec-core/src/lib.rs
git commit -m "feat(core): add error types and lib.rs exports"
```

---

## Chunk 2: Model Loading + Tokenization

### Task 3: Tokenizer Wrapper

**Files:**
- Create: `crates/ripvec-core/src/tokenize.rs`
- Modify: `crates/ripvec-core/src/lib.rs` (add module)

- [ ] **Step 1: Write tokenize.rs**

Downloads tokenizer.json from HuggingFace via hf-hub, then loads it with `Tokenizer::from_file`.
No `http` feature needed on the tokenizers crate.

```rust
//! HuggingFace tokenizer wrapper.
//!
//! Downloads and caches the tokenizer.json from a HuggingFace model
//! repository using hf-hub, then loads it for fast encoding.

use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;

/// Load a tokenizer from a HuggingFace model repository.
///
/// Downloads `tokenizer.json` on first call; subsequent calls use the cache.
pub fn load_tokenizer(model_repo: &str) -> crate::Result<Tokenizer> {
    let api = Api::new().map_err(|e| crate::Error::Download(e.to_string()))?;
    let repo = api.model(model_repo.to_string());
    let tokenizer_path = repo
        .get("tokenizer.json")
        .map_err(|e| crate::Error::Download(e.to_string()))?;
    Tokenizer::from_file(tokenizer_path).map_err(|e| crate::Error::Tokenization(e.to_string()))
}
```

- [ ] **Step 2: Add module to lib.rs**

Add `pub mod tokenize;` to lib.rs.

- [ ] **Step 3: Verify compiles**

Run: `cargo check -p ripvec-core`
Expected: SUCCESS

- [ ] **Step 4: Commit**

```bash
git add crates/ripvec-core/src/tokenize.rs crates/ripvec-core/src/lib.rs
git commit -m "feat(core): add tokenizer wrapper using hf-hub"
```

---

### Task 4: ONNX Model Loading + Embedding

**Files:**
- Create: `crates/ripvec-core/src/model.rs`
- Modify: `crates/ripvec-core/src/lib.rs` (add module)

- [ ] **Step 1: Write model.rs**

Key corrections from design.md:
- Use `commit_from_file` instead of `commit_from_memory_directly` (simpler, OS page cache handles warmth)
- Use `try_extract_array` instead of `extract_tensor`
- Handle output shape: BGE models output `[batch, seq_len, hidden]`, CLS pooling takes token 0

```rust
//! ONNX embedding model loading and inference.
//!
//! Downloads model weights from HuggingFace, creates an ONNX Runtime
//! session, and provides embedding inference with CLS pooling and
//! L2 normalization.

use hf_hub::api::sync::Api;
use ndarray::Axis;
use ort::session::{builder::GraphOptimizationLevel, Session};

/// An embedding model backed by ONNX Runtime.
pub struct EmbeddingModel {
    session: Session,
}

impl EmbeddingModel {
    /// Load an ONNX embedding model from a HuggingFace repository.
    ///
    /// Downloads the model file on first call; subsequent calls use the cache.
    /// The OS page cache keeps weights hot between invocations.
    pub fn load(model_repo: &str, model_file: &str) -> crate::Result<Self> {
        let api = Api::new().map_err(|e| crate::Error::Download(e.to_string()))?;
        let repo = api.model(model_repo.to_string());
        let model_path = repo
            .get(model_file)
            .map_err(|e| crate::Error::Download(e.to_string()))?;

        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)? // parallelism via rayon, not ORT threads
            .commit_from_file(&model_path)?;

        Ok(Self { session })
    }

    /// Produce an L2-normalized embedding vector from tokenized input.
    ///
    /// Uses CLS pooling (first token) suitable for BGE models.
    /// Input arrays must all have the same length (token count).
    pub fn embed(
        &self,
        input_ids: &[i64],
        attention_mask: &[i64],
        token_type_ids: &[i64],
    ) -> crate::Result<Vec<f32>> {
        let len = input_ids.len();
        let ids = ndarray::Array2::from_shape_vec((1, len), input_ids.to_vec())?;
        let mask = ndarray::Array2::from_shape_vec((1, len), attention_mask.to_vec())?;
        let types = ndarray::Array2::from_shape_vec((1, len), token_type_ids.to_vec())?;

        let outputs = self.session.run(ort::inputs![
            "input_ids" => ids,
            "attention_mask" => mask,
            "token_type_ids" => types,
        ]?)?;

        // Model output shape: [1, seq_len, hidden_dim]
        // CLS pooling: take first token embedding
        let output = &outputs[0];
        let array = output.try_extract_array::<f32>()?;
        let cls = array.index_axis(Axis(1), 0);
        let embedding: Vec<f32> = cls.iter().copied().collect();

        // L2 normalize — required for cosine similarity = dot product
        Ok(l2_normalize(&embedding))
    }
}

/// L2-normalize a vector. Returns zero vector if norm is zero.
fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_normalize_unit_vector() {
        let v = vec![1.0, 0.0, 0.0];
        let n = l2_normalize(&v);
        assert!((n[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn l2_normalize_arbitrary_vector() {
        let v = vec![3.0, 4.0];
        let n = l2_normalize(&v);
        let norm: f32 = n.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn l2_normalize_zero_vector() {
        let v = vec![0.0, 0.0, 0.0];
        let n = l2_normalize(&v);
        assert_eq!(n, vec![0.0, 0.0, 0.0]);
    }
}
```

**NOTE:** The `ort::inputs!` macro API may need adjustment based on ort 2.0.0-rc.12's exact requirements. If `inputs!` doesn't accept raw ndarray arrays, wrap with `ort::value::TensorRef::from_array_view(&array.view())?`. Fix at compile time.

- [ ] **Step 2: Add module to lib.rs**

Add `pub mod model;` to lib.rs.

- [ ] **Step 3: Verify compiles**

Run: `cargo check -p ripvec-core`
Expected: SUCCESS (may need to adjust ort input API — see note above)

- [ ] **Step 4: Run unit tests**

Run: `cargo test -p ripvec-core -- model::tests`
Expected: 3 tests PASS (l2_normalize tests are pure math, no model needed)

- [ ] **Step 5: Commit**

```bash
git add crates/ripvec-core/src/model.rs crates/ripvec-core/src/lib.rs
git commit -m "feat(core): add ONNX model loading and embedding inference"
```

---

## Chunk 3: Code Chunking

### Task 5: Language Registry

**Files:**
- Create: `crates/ripvec-core/src/languages.rs`
- Modify: `crates/ripvec-core/src/lib.rs` (add module)

- [ ] **Step 1: Write failing test**

In `crates/ripvec-core/src/languages.rs`, write tests first:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rust_extension_resolves() {
        assert!(config_for_extension("rs").is_some());
    }

    #[test]
    fn python_extension_resolves() {
        assert!(config_for_extension("py").is_some());
    }

    #[test]
    fn unknown_extension_returns_none() {
        assert!(config_for_extension("xyz").is_none());
    }

    #[test]
    fn all_supported_extensions() {
        let exts = ["rs", "py", "js", "jsx", "ts", "tsx", "go", "java", "c", "h", "cpp", "cc", "cxx", "hpp"];
        for ext in &exts {
            assert!(config_for_extension(ext).is_some(), "failed for {ext}");
        }
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p ripvec-core -- languages::tests`
Expected: FAIL — `config_for_extension` not defined

- [ ] **Step 3: Implement languages.rs**

Follow `design.md` "Tree-sitter code chunking" section for the full implementation.
Key pattern: match file extension → (Language, Query) pair.

**Important:** `tree-sitter-typescript` exports `LANGUAGE_TYPESCRIPT` and `LANGUAGE_TSX`, not `LANGUAGE`. Handle "ts" → `LANGUAGE_TYPESCRIPT` and "tsx" → `LANGUAGE_TSX`.

```rust
//! Language registry mapping file extensions to tree-sitter grammars.
//!
//! Each supported language has a grammar and a tree-sitter query that
//! extracts function, class, and method definitions.

use tree_sitter::{Language, Query};

/// Configuration for a supported source language.
pub struct LangConfig {
    /// The tree-sitter Language grammar.
    pub language: Language,
    /// Query that extracts semantic chunks (@def captures with @name).
    pub query: Query,
}

/// Look up the language configuration for a file extension.
///
/// Returns `None` for unsupported extensions.
pub fn config_for_extension(ext: &str) -> Option<LangConfig> {
    let (lang, query_str) = match ext {
        "rs" => (
            tree_sitter_rust::LANGUAGE.into(),
            concat!(
                "(function_item name: (identifier) @name) @def\n",
                "(struct_item name: (type_identifier) @name) @def\n",
                "(impl_item) @def\n",
                "(trait_item name: (type_identifier) @name) @def",
            ),
        ),
        "py" => (
            tree_sitter_python::LANGUAGE.into(),
            concat!(
                "(function_definition name: (identifier) @name) @def\n",
                "(class_definition name: (identifier) @name) @def",
            ),
        ),
        "js" | "jsx" => (
            tree_sitter_javascript::LANGUAGE.into(),
            concat!(
                "(function_declaration name: (identifier) @name) @def\n",
                "(class_declaration name: (identifier) @name) @def\n",
                "(method_definition name: (property_identifier) @name) @def",
            ),
        ),
        "ts" | "tsx" => {
            // tree-sitter-typescript exports separate languages for TS and TSX.
            // Both use the same query patterns; TSX adds JSX support.
            let lang: Language = if ext == "tsx" {
                tree_sitter_typescript::LANGUAGE_TSX.into()
            } else {
                tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()
            };
            (
                lang,
                concat!(
                    "(function_declaration name: (identifier) @name) @def\n",
                    "(class_declaration name: (identifier) @name) @def\n",
                    "(method_definition name: (property_identifier) @name) @def\n",
                    "(interface_declaration name: (type_identifier) @name) @def",
                ),
            )
        }
        "go" => (
            tree_sitter_go::LANGUAGE.into(),
            concat!(
                "(function_declaration name: (identifier) @name) @def\n",
                "(method_declaration name: (field_identifier) @name) @def",
            ),
        ),
        "java" => (
            tree_sitter_java::LANGUAGE.into(),
            concat!(
                "(method_declaration name: (identifier) @name) @def\n",
                "(class_declaration name: (identifier) @name) @def\n",
                "(interface_declaration name: (identifier) @name) @def",
            ),
        ),
        "c" | "h" => (
            tree_sitter_c::LANGUAGE.into(),
            "(function_definition declarator: (function_declarator declarator: (identifier) @name)) @def",
        ),
        "cpp" | "cc" | "cxx" | "hpp" => (
            tree_sitter_cpp::LANGUAGE.into(),
            concat!(
                "(function_definition declarator: (function_declarator declarator: (identifier) @name)) @def\n",
                "(class_specifier name: (type_identifier) @name) @def",
            ),
        ),
        _ => return None,
    };
    let query = Query::new(&lang, query_str).ok()?;
    Some(LangConfig { language: lang, query })
}
```

**NOTE:** The `LANGUAGE_TYPESCRIPT`/`LANGUAGE_TSX` constants may have different names. If compilation fails, check the `tree_sitter_typescript` crate exports and adjust. Possible alternatives: `language_typescript()` function or `LANGUAGE` with a feature flag.

- [ ] **Step 4: Run tests**

Run: `cargo test -p ripvec-core -- languages::tests`
Expected: 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add crates/ripvec-core/src/languages.rs crates/ripvec-core/src/lib.rs
git commit -m "feat(core): add language registry for 8 languages"
```

---

### Task 6: Tree-sitter Code Chunking

**Files:**
- Create: `crates/ripvec-core/src/chunk.rs`
- Create: `tests/fixtures/sample.rs`
- Create: `tests/fixtures/sample.py`
- Create: `tests/fixtures/sample.js`
- Modify: `crates/ripvec-core/src/lib.rs` (add module)

- [ ] **Step 1: Create test fixtures**

`tests/fixtures/sample.rs`:
```rust
fn hello() {
    println!("hi");
}

fn world() {
    println!("world");
}

struct Config {
    name: String,
}
```

`tests/fixtures/sample.py`:
```python
def greet(name):
    print(f"Hello, {name}")

class Greeter:
    def __init__(self, name):
        self.name = name

    def say_hello(self):
        print(f"Hello, {self.name}")
```

`tests/fixtures/sample.js`:
```javascript
function fetchData(url) {
    return fetch(url).then(r => r.json());
}

class DataService {
    constructor(baseUrl) {
        this.baseUrl = baseUrl;
    }

    getData(path) {
        return fetchData(this.baseUrl + path);
    }
}
```

- [ ] **Step 2: Write failing tests in chunk.rs**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn chunks_rust_functions_and_structs() {
        let source = "fn hello() { println!(\"hi\"); }\nfn world() {}\nstruct Foo { x: i32 }";
        let config = crate::languages::config_for_extension("rs").unwrap();
        let chunks = chunk_file(Path::new("test.rs"), source, &config);
        assert!(chunks.len() >= 2, "expected at least 2 chunks, got {}", chunks.len());
        assert!(chunks.iter().any(|c| c.name == "hello"));
        assert!(chunks.iter().any(|c| c.name == "world"));
    }

    #[test]
    fn chunks_python_functions_and_classes() {
        let source = "def greet(name):\n    pass\n\nclass Foo:\n    pass\n";
        let config = crate::languages::config_for_extension("py").unwrap();
        let chunks = chunk_file(Path::new("test.py"), source, &config);
        assert!(chunks.len() >= 2);
        assert!(chunks.iter().any(|c| c.name == "greet"));
        assert!(chunks.iter().any(|c| c.name == "Foo"));
    }

    #[test]
    fn fallback_for_empty_query_matches() {
        // A file with no function/class definitions should fall back to whole-file chunk
        let source = "let x = 42;\nconsole.log(x);\n";
        let config = crate::languages::config_for_extension("js").unwrap();
        let chunks = chunk_file(Path::new("script.js"), source, &config);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].kind, "file");
    }

    #[test]
    fn empty_file_produces_no_chunks() {
        let config = crate::languages::config_for_extension("rs").unwrap();
        let chunks = chunk_file(Path::new("empty.rs"), "", &config);
        assert!(chunks.is_empty());
    }
}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cargo test -p ripvec-core -- chunk::tests`
Expected: FAIL — `chunk_file` not defined

- [ ] **Step 4: Implement chunk.rs**

Follow `design.md` "Code chunking" section. The `CodeChunk` struct and `chunk_file` function are well-specified there.

```rust
//! Tree-sitter based code chunking.
//!
//! Parses source files into ASTs and extracts semantic chunks at
//! function, class, and method boundaries. Falls back to whole-file
//! chunks when no semantic boundaries are found.

use std::path::Path;
use tree_sitter::{Parser, QueryCursor};

/// A semantic chunk extracted from a source file.
#[derive(Debug, Clone)]
pub struct CodeChunk {
    /// Path to the source file.
    pub file_path: String,
    /// Name of the definition (function name, class name, etc.).
    pub name: String,
    /// Kind of syntax node (e.g., "function_item", "class_definition").
    pub kind: String,
    /// 1-based start line number.
    pub start_line: usize,
    /// 1-based end line number.
    pub end_line: usize,
    /// Source text of the chunk.
    pub content: String,
}

/// Extract semantic chunks from a source file.
///
/// Uses tree-sitter to parse the file and extract definitions matching
/// the language's query patterns. Falls back to a single whole-file
/// chunk if no semantic boundaries are found.
pub fn chunk_file(path: &Path, source: &str, config: &crate::languages::LangConfig) -> Vec<CodeChunk> {
    let mut parser = Parser::new();
    if parser.set_language(&config.language).is_err() {
        return vec![];
    }

    let tree = match parser.parse(source, None) {
        Some(t) => t,
        None => return vec![],
    };

    let mut cursor = QueryCursor::new();
    let mut chunks = Vec::new();

    for m in cursor.matches(&config.query, tree.root_node(), source.as_bytes()) {
        let mut name = String::new();
        let mut def_node = None;
        for cap in m.captures {
            let cap_name = &config.query.capture_names()[cap.index as usize];
            if cap_name == "name" {
                name = source[cap.node.start_byte()..cap.node.end_byte()].to_string();
            } else if cap_name == "def" {
                def_node = Some(cap.node);
            }
        }
        if let Some(node) = def_node {
            chunks.push(CodeChunk {
                file_path: path.display().to_string(),
                name,
                kind: node.kind().to_string(),
                start_line: node.start_position().row + 1,
                end_line: node.end_position().row + 1,
                content: source[node.start_byte()..node.end_byte()].to_string(),
            });
        }
    }

    // Fallback: whole file as one chunk if no semantic matches
    if chunks.is_empty() && !source.trim().is_empty() {
        chunks.push(CodeChunk {
            file_path: path.display().to_string(),
            name: path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string(),
            kind: "file".to_string(),
            start_line: 1,
            end_line: source.lines().count(),
            content: source.to_string(),
        });
    }

    chunks
}
```

- [ ] **Step 5: Add module to lib.rs**

Add `pub mod chunk;` to lib.rs.

- [ ] **Step 6: Run tests**

Run: `cargo test -p ripvec-core -- chunk::tests`
Expected: 4 tests PASS

- [ ] **Step 7: Commit**

```bash
git add crates/ripvec-core/src/chunk.rs crates/ripvec-core/src/lib.rs tests/fixtures/
git commit -m "feat(core): add tree-sitter code chunking with 8-language support"
```

---

## Chunk 4: Search Pipeline

### Task 7: Directory Walk + Embedding Pipeline + Similarity

**Files:**
- Create: `crates/ripvec-core/src/walk.rs`
- Create: `crates/ripvec-core/src/similarity.rs`
- Create: `crates/ripvec-core/src/embed.rs`
- Modify: `crates/ripvec-core/src/lib.rs` (add modules)

- [ ] **Step 1: Write similarity.rs with tests first**

```rust
//! Cosine similarity computation and ranking.
//!
//! Since all embeddings are L2-normalized, cosine similarity equals
//! the dot product — no square roots needed at query time.

/// Cosine similarity between two L2-normalized vectors (= dot product).
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_normalized_vectors() {
        let v = vec![0.5773, 0.5773, 0.5773];
        let sim = dot_product(&v, &v);
        assert!((sim - 1.0).abs() < 0.01);
    }

    #[test]
    fn orthogonal_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = dot_product(&a, &b);
        assert!((sim).abs() < 1e-6);
    }

    #[test]
    fn opposite_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = dot_product(&a, &b);
        assert!((sim + 1.0).abs() < 1e-6);
    }
}
```

- [ ] **Step 2: Run similarity tests**

Run: `cargo test -p ripvec-core -- similarity::tests`
Expected: 3 tests PASS

- [ ] **Step 3: Write walk.rs**

```rust
//! Directory traversal using the `ignore` crate.
//!
//! Respects `.gitignore` rules, skips hidden files, and filters
//! to files with supported source extensions.

use ignore::WalkBuilder;
use std::path::{Path, PathBuf};

/// Walk a directory tree and collect paths to supported source files.
///
/// Respects `.gitignore` rules and skips hidden files and directories.
pub fn collect_files(root: &Path) -> Vec<PathBuf> {
    WalkBuilder::new(root)
        .hidden(true)
        .git_ignore(true)
        .git_global(true)
        .build()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_some_and(|ft| ft.is_file()))
        .filter(|e| {
            e.path()
                .extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| crate::languages::config_for_extension(ext).is_some())
        })
        .map(|e| e.into_path())
        .collect()
}
```

- [ ] **Step 4: Write embed.rs**

This is the main search pipeline. Follow `design.md` "parallel embedding pipeline" section with corrected ort API.

```rust
//! Parallel embedding pipeline.
//!
//! Two-phase architecture: discover files (I/O-bound via `ignore`),
//! then chunk and embed in parallel (CPU-bound via `rayon`).

use rayon::prelude::*;
use std::path::Path;

use crate::chunk::CodeChunk;
use crate::model::EmbeddingModel;
use crate::similarity;

/// A search result pairing a code chunk with its similarity score.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// The matched code chunk.
    pub chunk: CodeChunk,
    /// Cosine similarity to the query (0.0 to 1.0).
    pub similarity: f32,
}

/// Search a directory for code chunks semantically similar to a query.
///
/// Walks the directory, chunks all supported files, embeds everything
/// in parallel, and returns the top-k results ranked by similarity.
pub fn search(
    root: &Path,
    query: &str,
    model: &EmbeddingModel,
    tokenizer: &tokenizers::Tokenizer,
    top_k: usize,
) -> crate::Result<Vec<SearchResult>> {
    // Phase 1: Collect files
    let files = crate::walk::collect_files(root);

    // Phase 2: Chunk in parallel
    let chunks: Vec<CodeChunk> = files
        .par_iter()
        .flat_map(|path| {
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            let config = match crate::languages::config_for_extension(ext) {
                Some(c) => c,
                None => return vec![],
            };
            let source = match std::fs::read_to_string(path) {
                Ok(s) => s,
                Err(_) => return vec![],
            };
            crate::chunk::chunk_file(path, &source, &config)
        })
        .collect();

    // Phase 3: Embed query
    let query_embedding = embed_text(query, model, tokenizer)?;

    // Phase 4: Embed chunks and compute similarity
    let mut results: Vec<SearchResult> = chunks
        .par_iter()
        .filter_map(|chunk| {
            let emb = embed_text(&chunk.content, model, tokenizer).ok()?;
            let sim = similarity::dot_product(&query_embedding, &emb);
            Some(SearchResult {
                chunk: chunk.clone(),
                similarity: sim,
            })
        })
        .collect();

    // Phase 5: Rank and truncate
    results.sort_unstable_by(|a, b| {
        b.similarity
            .partial_cmp(&a.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(top_k);
    Ok(results)
}

/// Embed a text string using the model and tokenizer.
fn embed_text(
    text: &str,
    model: &EmbeddingModel,
    tokenizer: &tokenizers::Tokenizer,
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
    model.embed(&ids, &mask, &type_ids)
}
```

- [ ] **Step 5: Add modules to lib.rs**

Add `pub mod walk;`, `pub mod similarity;`, `pub mod embed;` to lib.rs.

- [ ] **Step 6: Verify compiles**

Run: `cargo check -p ripvec-core`
Expected: SUCCESS

- [ ] **Step 7: Run all tests so far**

Run: `cargo test -p ripvec-core`
Expected: All pure-logic tests PASS (similarity, l2_normalize, chunking, languages)

- [ ] **Step 8: Commit**

```bash
git add crates/ripvec-core/src/walk.rs crates/ripvec-core/src/similarity.rs \
        crates/ripvec-core/src/embed.rs crates/ripvec-core/src/lib.rs
git commit -m "feat(core): add search pipeline with walk, embed, and similarity"
```

---

## Chunk 5: CLI Binary

### Task 8: CLI Arguments + Output Formatting

**Files:**
- Create: `crates/ripvec/src/cli.rs`
- Create: `crates/ripvec/src/output.rs`
- Modify: `crates/ripvec/src/main.rs`

- [ ] **Step 1: Write cli.rs**

```rust
//! Command-line argument definitions using clap derive.

use clap::Parser;

/// Semantic code search — like ripgrep but for meaning.
#[derive(Parser, Debug)]
#[command(name = "ripvec", version, about)]
pub struct Args {
    /// Natural language query to search for.
    pub query: String,

    /// Root directory to search (defaults to current directory).
    #[arg(default_value = ".")]
    pub path: String,

    /// Number of results to show.
    #[arg(short = 'n', long, default_value_t = 10)]
    pub top_k: usize,

    /// HuggingFace model repository.
    #[arg(long, default_value = "BAAI/bge-small-en-v1.5")]
    pub model_repo: String,

    /// ONNX model filename within the repository.
    #[arg(long, default_value = "onnx/model.onnx")]
    pub model_file: String,

    /// Output format.
    #[arg(short, long, default_value = "color")]
    pub format: OutputFormat,

    /// Minimum similarity threshold (0.0 to 1.0).
    #[arg(short = 't', long, default_value_t = 0.0)]
    pub threshold: f32,

    /// Number of threads for parallel processing (0 = auto).
    #[arg(short = 'j', long, default_value_t = 0)]
    pub threads: usize,
}

/// Output format for search results.
#[derive(clap::ValueEnum, Clone, Debug)]
pub enum OutputFormat {
    /// Plain text without color.
    Plain,
    /// JSON output.
    Json,
    /// Colored terminal output (default).
    Color,
}
```

- [ ] **Step 2: Write output.rs**

```rust
//! Result formatting for different output modes.

use ripvec_core::embed::SearchResult;
use crate::cli::OutputFormat;

/// Format and print search results according to the chosen output format.
pub fn print_results(results: &[SearchResult], format: &OutputFormat) {
    match format {
        OutputFormat::Plain => print_plain(results),
        OutputFormat::Json => print_json(results),
        OutputFormat::Color => print_color(results),
    }
}

fn print_plain(results: &[SearchResult]) {
    for (i, r) in results.iter().enumerate() {
        println!(
            "{}. {} ({}:{}-{}) [{:.3}]",
            i + 1,
            r.chunk.name,
            r.chunk.file_path,
            r.chunk.start_line,
            r.chunk.end_line,
            r.similarity,
        );
        println!("{}", r.chunk.content);
        println!();
    }
}

fn print_json(results: &[SearchResult]) {
    let items: Vec<serde_json::Value> = results
        .iter()
        .map(|r| {
            serde_json::json!({
                "name": r.chunk.name,
                "file": r.chunk.file_path,
                "start_line": r.chunk.start_line,
                "end_line": r.chunk.end_line,
                "similarity": r.similarity,
                "content": r.chunk.content,
            })
        })
        .collect();
    println!("{}", serde_json::to_string_pretty(&items).unwrap_or_default());
}

fn print_color(results: &[SearchResult]) {
    for (i, r) in results.iter().enumerate() {
        // Bold green for rank, cyan for file info
        println!(
            "\x1b[1;32m{}.\x1b[0m \x1b[1m{}\x1b[0m \x1b[36m{}:{}-{}\x1b[0m \x1b[33m[{:.3}]\x1b[0m",
            i + 1,
            r.chunk.name,
            r.chunk.file_path,
            r.chunk.start_line,
            r.chunk.end_line,
            r.similarity,
        );
        println!("{}", r.chunk.content);
        println!();
    }
}
```

- [ ] **Step 3: Wire up main.rs**

```rust
mod cli;
mod output;

use anyhow::{Context, Result};
use clap::Parser;

fn main() -> Result<()> {
    let args = cli::Args::parse();

    // Configure thread pool
    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .context("failed to configure thread pool")?;
    }

    // Load model and tokenizer
    let model = ripvec_core::model::EmbeddingModel::load(&args.model_repo, &args.model_file)
        .context("failed to load embedding model")?;
    let tokenizer = ripvec_core::tokenize::load_tokenizer(&args.model_repo)
        .context("failed to load tokenizer")?;

    // Run search
    let results = ripvec_core::embed::search(
        std::path::Path::new(&args.path),
        &args.query,
        &model,
        &tokenizer,
        args.top_k,
    )
    .context("search failed")?;

    // Filter by threshold and print
    let filtered: Vec<_> = results
        .into_iter()
        .filter(|r| r.similarity >= args.threshold)
        .collect();
    output::print_results(&filtered, &args.format);

    Ok(())
}
```

- [ ] **Step 4: Add serde_json to ripvec CLI dependencies**

Add `serde_json.workspace = true` to `crates/ripvec/Cargo.toml` `[dependencies]` for JSON output.

- [ ] **Step 5: Verify compiles**

Run: `cargo check -p ripvec`
Expected: SUCCESS

- [ ] **Step 6: Commit**

```bash
git add crates/ripvec/src/
git commit -m "feat(cli): add clap CLI with plain, JSON, and color output"
```

---

### Task 9: CLI Integration Tests

**Files:**
- Create: `tests/integration.rs`

- [ ] **Step 1: Write integration tests**

These test the compiled binary via `assert_cmd`. Model-dependent tests are marked `#[ignore]`.

```rust
use assert_cmd::Command;
use predicates::prelude::*;

#[test]
fn prints_version() {
    Command::cargo_bin("ripvec")
        .unwrap()
        .arg("--version")
        .assert()
        .success()
        .stdout(predicate::str::starts_with("ripvec"));
}

#[test]
fn prints_help() {
    Command::cargo_bin("ripvec")
        .unwrap()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("Semantic code search"));
}

/// Requires model download — run with `cargo test -- --ignored`
#[test]
#[ignore]
fn searches_fixture_directory() {
    Command::cargo_bin("ripvec")
        .unwrap()
        .args(["find the main entry point", "tests/fixtures/", "-n", "3"])
        .assert()
        .success();
}

#[test]
fn fails_on_missing_query() {
    Command::cargo_bin("ripvec")
        .unwrap()
        .assert()
        .failure();
}
```

- [ ] **Step 2: Run non-ignored integration tests**

Run: `cargo test --test integration`
Expected: `prints_version`, `prints_help`, `fails_on_missing_query` PASS; `searches_fixture_directory` SKIPPED

- [ ] **Step 3: Commit**

```bash
git add tests/integration.rs
git commit -m "test: add CLI integration tests"
```

---

## Chunk 6: MCP Server

### Task 10: ripvec-mcp Server

**Files:**
- Modify: `crates/ripvec-mcp/src/main.rs`

- [ ] **Step 1: Implement MCP server**

Uses rmcp 1.2.0 API with `#[tool_router]` pattern (corrected from design.md's 0.16 API).

```rust
//! MCP server exposing ripvec semantic search as a tool.
//!
//! Communicates over stdin/stdout using the Model Context Protocol.
//! Designed for integration with Claude Code, Cursor, and other MCP clients.

use std::sync::Arc;
use rmcp::handler::server::tool::ToolRouter;
use rmcp::model::{Content, ServerCapabilities, ServerInfo};
use rmcp::{tool, tool_router, ErrorData as McpError, ServerHandler};
use rmcp::handler::server::wrapper::Parameters;
use schemars::JsonSchema;
use serde::Deserialize;

/// The ripvec MCP server. Model and tokenizer are loaded once at startup.
#[derive(Clone)]
pub struct RipvecServer {
    model: Arc<ripvec_core::model::EmbeddingModel>,
    tokenizer: Arc<tokenizers::Tokenizer>,
    tool_router: ToolRouter<Self>,
}

/// Parameters for the semantic_search tool.
#[derive(Deserialize, JsonSchema)]
pub struct SearchRequest {
    /// Natural language query describing the code you're looking for.
    pub query: String,
    /// Root directory to search (defaults to current directory).
    #[serde(default = "default_path")]
    pub path: Option<String>,
    /// Maximum number of results to return.
    #[serde(default = "default_top_k")]
    pub top_k: Option<usize>,
}

fn default_path() -> Option<String> {
    Some(".".to_string())
}

fn default_top_k() -> Option<usize> {
    Some(10)
}

#[tool_router]
impl RipvecServer {
    fn new(
        model: Arc<ripvec_core::model::EmbeddingModel>,
        tokenizer: Arc<tokenizers::Tokenizer>,
    ) -> Self {
        Self {
            model,
            tokenizer,
            tool_router: Self::tool_router(),
        }
    }

    /// Search code semantically by meaning using vector embeddings.
    /// Returns the most relevant functions, classes, and methods matching
    /// a natural language query.
    #[tool(
        name = "semantic_search",
        description = "Search code semantically by meaning using vector embeddings. Returns the most relevant functions, classes, and methods matching a natural language query."
    )]
    async fn semantic_search(
        &self,
        params: Parameters<SearchRequest>,
    ) -> Result<rmcp::model::CallToolResult, McpError> {
        let req = params.0;
        let path = req.path.unwrap_or_else(|| ".".to_string());
        let top_k = req.top_k.unwrap_or(10);
        let model = Arc::clone(&self.model);
        let tokenizer = Arc::clone(&self.tokenizer);

        let result = tokio::task::spawn_blocking(move || {
            ripvec_core::embed::search(
                std::path::Path::new(&path),
                &req.query,
                &model,
                &tokenizer,
                top_k,
            )
        })
        .await
        .map_err(|e| McpError::internal_error(e.to_string(), None))?
        .map_err(|e| McpError::internal_error(e.to_string(), None))?;

        let text = result
            .iter()
            .enumerate()
            .map(|(i, r)| {
                format!(
                    "{}. {} ({}:{}-{}, similarity: {:.3})\n```\n{}\n```",
                    i + 1,
                    r.chunk.name,
                    r.chunk.file_path,
                    r.chunk.start_line,
                    r.chunk.end_line,
                    r.similarity,
                    r.chunk.content,
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n");

        Ok(rmcp::model::CallToolResult::success(vec![Content::text(
            text,
        )]))
    }
}

impl ServerHandler for RipvecServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some("Semantic code search tool. Search codebases by meaning, not text patterns.".into()),
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            ..Default::default()
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load model and tokenizer once at startup
    let model = Arc::new(
        ripvec_core::model::EmbeddingModel::load("BAAI/bge-small-en-v1.5", "onnx/model.onnx")
            .expect("failed to load embedding model"),
    );
    let tokenizer = Arc::new(
        ripvec_core::tokenize::load_tokenizer("BAAI/bge-small-en-v1.5")
            .expect("failed to load tokenizer"),
    );

    let server = RipvecServer::new(model, tokenizer);
    let service = server
        .serve(rmcp::transport::io::stdio())
        .await?;
    service.waiting().await?;
    Ok(())
}
```

**NOTE:** The rmcp transport API may differ. If `rmcp::transport::io::stdio()` doesn't compile, check rmcp 1.2 docs for the correct stdio transport constructor. Alternatives:
- `rmcp::transport::stdio()`
- `(tokio::io::stdin(), tokio::io::stdout())`

- [ ] **Step 2: Verify compiles**

Run: `cargo check -p ripvec-mcp`
Expected: SUCCESS (may need transport API adjustments — see note)

- [ ] **Step 3: Commit**

```bash
git add crates/ripvec-mcp/src/main.rs
git commit -m "feat(mcp): add MCP server with semantic_search tool"
```

---

### Task 11: Final Verification + .gitignore

**Files:**
- Create: `.gitignore` (if not exists)

- [ ] **Step 1: Update .gitignore**

Ensure ONNX models and build artifacts are excluded:
```
/target
*.onnx
```

- [ ] **Step 2: Run full check suite**

```bash
cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test --workspace
```

Expected: All checks PASS, all non-ignored tests PASS.

- [ ] **Step 3: Fix any clippy warnings or test failures**

Address pedantic clippy lints (missing docs, etc.) iteratively until clean.

- [ ] **Step 4: Commit**

```bash
git add .gitignore
git commit -m "chore: add .gitignore and verify full check suite"
```

---

## Implementation Notes

### Model download on first run
The first invocation of `ripvec` will download ~50MB of model weights from HuggingFace. Subsequent runs use the cached files. Set `HF_HOME` to control cache location.

### Thread safety constraints
- `tree_sitter::Parser` is NOT Send/Sync — create per rayon thread in `par_iter`
- `tree_sitter::Query` is Send + Sync — but we create fresh per extension anyway
- `ort::Session` is Send + Sync — share freely via `&self`
- `tokenizers::Tokenizer` is Send + Sync — share freely via `&`

### Tests requiring network/model
Tests marked `#[ignore]` require downloading the ONNX model. Run them with:
```bash
cargo test --workspace -- --ignored
```

### Future optimizations (not in this plan)
- Memory-mapped model loading via `memmap2` + `commit_from_memory_directly` for zero-copy
- `simsimd` crate for AVX-512/NEON SIMD similarity computation
- Persistent embedding cache to skip re-embedding unchanged files
- SQL and Jinja2 tree-sitter grammars for structured text support
- Plain text chunking (paragraph/sentence splitting) for non-code files
