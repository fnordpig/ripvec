# ripvec: complete development plan for semantic code search in Rust

**ripvec is a stateless CLI that performs semantic (vector) code search using memory-mapped ONNX embeddings, tree-sitter code chunking, and cosine similarity ranking — like ripgrep but for meaning, not text.** This plan covers every detail needed to build it end-to-end: project structure, exact crate versions, code patterns, MCP server integration, Claude Code development environment, testing, CI/CD, and distribution. The architecture is deliberately simple — no daemon, no vector database, no persistent index. The OS page cache keeps model weights hot between invocations, making repeat searches fast without architectural complexity.

---

## Architecture overview and data flow

ripvec operates in a single-pass pipeline with five stages, all within one process invocation:

1. **Model bootstrap** — On first run, `hf-hub` downloads ONNX model weights and tokenizer from HuggingFace (~50MB). On subsequent runs, the cached files are memory-mapped directly via `memmap2`, and the OS page cache keeps them hot.
2. **Directory walk** — The `ignore` crate traverses the target directory respecting `.gitignore` rules, filtering to supported source file extensions.
3. **Code chunking** — `tree-sitter` parses each file and extracts semantic chunks at function/class/method boundaries using per-language query patterns.
4. **Parallel embedding** — `rayon` distributes chunks across CPU cores. Each thread tokenizes its chunks via the `tokenizers` crate and runs ONNX inference via `ort` to produce **384-dimensional embedding vectors**.
5. **Ranking** — The query string is embedded identically, then cosine similarity is computed against all chunk embeddings. Top-k results are printed with file paths, line ranges, and similarity scores.

The optional `ripvec-mcp` binary wraps the core library as an MCP (Model Context Protocol) server, enabling Claude Code and Cursor to invoke semantic search as a tool.

### Workspace layout

```
ripvec/
├── Cargo.toml                    # Workspace root
├── Cargo.lock
├── CLAUDE.md                     # Claude Code project configuration
├── .github/
│   └── workflows/
│       ├── ci.yml                # CI: check, test, lint, audit
│       └── release.yml           # Release: build, package, publish
├── crates/
│   ├── ripvec-core/              # Shared library crate
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs            # Public API re-exports
│   │       ├── error.rs          # thiserror error types
│   │       ├── model.rs          # ONNX session + mmap + embedding
│   │       ├── tokenize.rs       # HuggingFace tokenizer wrapper
│   │       ├── chunk.rs          # tree-sitter code chunking
│   │       ├── walk.rs           # ignore-based directory traversal
│   │       ├── embed.rs          # Parallel embedding pipeline
│   │       ├── similarity.rs     # Cosine similarity + ranking
│   │       └── languages.rs      # Language registry (extension → grammar)
│   ├── ripvec/                   # CLI binary crate
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── main.rs           # Entry point
│   │       ├── cli.rs            # clap argument definitions
│   │       └── output.rs         # Result formatting (plain, JSON, colored)
│   └── ripvec-mcp/              # MCP server binary crate
│       ├── Cargo.toml
│       └── src/
│           └── main.rs           # MCP server with semantic_search tool
└── tests/
    ├── fixtures/                 # Sample source files for integration tests
    └── integration.rs
```

---

## Dependency manifest with pinned versions

All dependency versions are specified once in the workspace root using `workspace.dependencies`, then referenced with `.workspace = true` in member crates.

### Root `Cargo.toml`

```toml
[workspace]
members = ["crates/ripvec-core", "crates/ripvec", "crates/ripvec-mcp"]
resolver = "2"

[workspace.package]
edition = "2021"
rust-version = "1.80.0"
license = "MIT OR Apache-2.0"
repository = "https://github.com/youruser/ripvec"
authors = ["Your Name <you@example.com>"]

[workspace.dependencies]
# Core inference
ort = "2.0.0-rc.12"
hf-hub = { version = "0.4.3", default-features = false, features = ["ureq", "rustls-tls"] }
tokenizers = { version = "0.22.2", default-features = false }
memmap2 = "0.9.9"
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
ignore = "0.4.25"
num_cpus = "1.16"

# CLI
clap = { version = "4.5", features = ["derive"] }

# Error handling
thiserror = "2.0"
anyhow = "1.0"

# MCP server
rmcp = { version = "0.16", features = ["server", "transport-io", "macros"] }
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
all = "warn"
pedantic = "warn"
```

### `crates/ripvec-core/Cargo.toml`

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
memmap2.workspace = true
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

### `crates/ripvec/Cargo.toml`

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

### `crates/ripvec-mcp/Cargo.toml`

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

[lints]
workspace = true
```

---

## Model loading with memory-mapped ONNX weights

The central design decision is pairing `memmap2` with `ort`'s `commit_from_memory_directly` to achieve zero-copy model loading. ONNX Runtime reads weights directly from the memory-mapped region. The OS page cache ensures that after the first invocation, model weights are already resident in RAM — subsequent invocations skip disk I/O entirely.

```rust
// crates/ripvec-core/src/model.rs
use std::fs::File;
use std::path::PathBuf;
use memmap2::Mmap;
use ort::session::{builder::GraphOptimizationLevel, Session};
use hf_hub::api::sync::Api;

pub struct EmbeddingModel {
    session: Session,
    _mmap: Mmap, // must outlive session — the underscore prefix is intentional
}

impl EmbeddingModel {
    pub fn load(model_repo: &str, model_file: &str) -> Result<Self, crate::Error> {
        // Download or retrieve from cache
        let api = Api::new()?;
        let repo = api.model(model_repo.to_string());
        let model_path: PathBuf = repo.get(model_file)?;

        // Memory-map the model file
        let file = File::open(&model_path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        // Create ONNX session from mmap — zero copy
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?  // single thread per inference; parallelism via rayon
            .commit_from_memory_directly(&mmap)?;

        Ok(Self { session, _mmap: mmap })
    }

    pub fn embed(&self, input_ids: &[i64], attention_mask: &[i64],
                 token_type_ids: &[i64]) -> Result<Vec<f32>, crate::Error> {
        let len = input_ids.len();
        let outputs = self.session.run(ort::inputs![
            "input_ids" => ndarray::Array2::from_shape_vec((1, len),
                input_ids.to_vec())?,
            "attention_mask" => ndarray::Array2::from_shape_vec((1, len),
                attention_mask.to_vec())?,
            "token_type_ids" => ndarray::Array2::from_shape_vec((1, len),
                token_type_ids.to_vec())?
        ])?;

        // CLS pooling for BGE models
        let tensor = outputs[0].extract_tensor::<f32>()?;
        let view = tensor.view();
        let cls = view.index_axis(ndarray::Axis(1), 0);
        let embedding: Vec<f32> = cls.iter().copied().collect();

        // L2 normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        Ok(embedding.iter().map(|x| x / norm).collect())
    }
}
```

**The recommended default model is `BAAI/bge-small-en-v1.5`** — 33M parameters, 384 embedding dimensions, ~127MB in fp32 or ~32MB quantized. It ranks highly on MTEB retrieval benchmarks and has readily available ONNX exports. For code-specific workloads, `jinaai/jina-embeddings-v2-base-code` (137M params, 768-dim, 8192-token context, trained on 30 programming languages) offers superior code understanding at the cost of 3× model size. The quantized BGE variant at `Qdrant/bge-small-en-v1.5-onnx-Q` brings size down to ~32MB with minimal quality loss.

---

## Tree-sitter code chunking across 8 languages

Each source file is parsed into an AST, then tree-sitter queries extract function, class, and method boundaries. This produces semantically meaningful chunks rather than arbitrary line-based splits. The key insight: **`Query` objects are `Send + Sync` and can be shared across threads, but `Parser` is not thread-safe** — create one per rayon thread.

```rust
// crates/ripvec-core/src/languages.rs
use tree_sitter::{Language, Query};

pub struct LangConfig {
    pub language: Language,
    pub query: Query,
}

pub fn config_for_extension(ext: &str) -> Option<LangConfig> {
    let (lang, query_str) = match ext {
        "rs" => (tree_sitter_rust::LANGUAGE.into(), concat!(
            "(function_item name: (identifier) @name) @def\n",
            "(struct_item name: (type_identifier) @name) @def\n",
            "(impl_item) @def\n",
            "(trait_item name: (type_identifier) @name) @def"
        )),
        "py" => (tree_sitter_python::LANGUAGE.into(), concat!(
            "(function_definition name: (identifier) @name) @def\n",
            "(class_definition name: (identifier) @name) @def"
        )),
        "js" | "jsx" => (tree_sitter_javascript::LANGUAGE.into(), concat!(
            "(function_declaration name: (identifier) @name) @def\n",
            "(class_declaration name: (identifier) @name) @def\n",
            "(method_definition name: (property_identifier) @name) @def"
        )),
        "ts" | "tsx" => (tree_sitter_typescript::LANGUAGE.into(), concat!(
            "(function_declaration name: (identifier) @name) @def\n",
            "(class_declaration name: (identifier) @name) @def\n",
            "(method_definition name: (property_identifier) @name) @def\n",
            "(interface_declaration name: (type_identifier) @name) @def"
        )),
        "go" => (tree_sitter_go::LANGUAGE.into(), concat!(
            "(function_declaration name: (identifier) @name) @def\n",
            "(method_declaration name: (field_identifier) @name) @def"
        )),
        "java" => (tree_sitter_java::LANGUAGE.into(), concat!(
            "(method_declaration name: (identifier) @name) @def\n",
            "(class_declaration name: (identifier) @name) @def\n",
            "(interface_declaration name: (identifier) @name) @def"
        )),
        "c" | "h" => (tree_sitter_c::LANGUAGE.into(), concat!(
            "(function_definition declarator: (function_declarator",
            " declarator: (identifier) @name)) @def"
        )),
        "cpp" | "cc" | "cxx" | "hpp" => (tree_sitter_cpp::LANGUAGE.into(), concat!(
            "(function_definition declarator: (function_declarator",
            " declarator: (identifier) @name)) @def\n",
            "(class_specifier name: (type_identifier) @name) @def"
        )),
        _ => return None,
    };
    let query = Query::new(&lang, query_str).ok()?;
    Some(LangConfig { language: lang, query })
}
```

The chunking function extracts matched nodes with their byte ranges, producing `CodeChunk` structs that carry the source text, file path, line range, and semantic name:

```rust
// crates/ripvec-core/src/chunk.rs
use tree_sitter::{Parser, QueryCursor};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct CodeChunk {
    pub file_path: String,
    pub name: String,
    pub kind: String,
    pub start_line: usize,
    pub end_line: usize,
    pub content: String,
}

pub fn chunk_file(path: &Path, source: &str,
                  config: &crate::languages::LangConfig) -> Vec<CodeChunk> {
    let mut parser = Parser::new();
    parser.set_language(&config.language).expect("grammar load");
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
            if *cap_name == "name" {
                name = source[cap.node.start_byte()..cap.node.end_byte()].to_string();
            } else if *cap_name == "def" {
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

    // Fallback: if no semantic chunks found, treat entire file as one chunk
    if chunks.is_empty() && !source.trim().is_empty() {
        chunks.push(CodeChunk {
            file_path: path.display().to_string(),
            name: path.file_name().unwrap_or_default().to_string_lossy().to_string(),
            kind: "file".to_string(),
            start_line: 1,
            end_line: source.lines().count(),
            content: source.to_string(),
        });
    }
    chunks
}
```

---

## The parallel embedding pipeline

The pipeline uses a **two-phase architecture**: the `ignore` crate discovers files (I/O-bound), then `rayon` processes them in parallel (CPU-bound). This mirrors ripgrep's own design. The `ignore` crate's `WalkBuilder` handles `.gitignore` rules, hidden files, and symlinks. Its `build_parallel()` method uses crossbeam work-stealing internally, while rayon handles the compute-heavy chunking and embedding.

```rust
// crates/ripvec-core/src/embed.rs
use rayon::prelude::*;
use ignore::WalkBuilder;
use std::path::Path;

pub struct SearchResult {
    pub chunk: crate::chunk::CodeChunk,
    pub similarity: f32,
}

pub fn search(
    root: &Path,
    query: &str,
    model: &crate::model::EmbeddingModel,
    tokenizer: &tokenizers::Tokenizer,
    top_k: usize,
) -> Result<Vec<SearchResult>, crate::Error> {
    // Phase 1: Collect files respecting .gitignore
    let files: Vec<_> = WalkBuilder::new(root)
        .hidden(true)
        .git_ignore(true)
        .git_global(true)
        .build()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map_or(false, |ft| ft.is_file()))
        .filter(|e| {
            e.path().extension()
                .and_then(|ext| ext.to_str())
                .and_then(crate::languages::config_for_extension)
                .is_some()
        })
        .map(|e| e.into_path())
        .collect();

    // Phase 2: Parse + chunk in parallel
    let chunks: Vec<_> = files.par_iter()
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

    // Phase 4: Embed all chunks in parallel and compute similarity
    let mut results: Vec<SearchResult> = chunks.par_iter()
        .filter_map(|chunk| {
            let emb = embed_text(&chunk.content, model, tokenizer).ok()?;
            let sim = dot_product(&query_embedding, &emb);
            Some(SearchResult { chunk: chunk.clone(), similarity: sim })
        })
        .collect();

    // Phase 5: Sort by similarity descending, take top-k
    results.sort_unstable_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
    results.truncate(top_k);
    Ok(results)
}

fn embed_text(text: &str, model: &crate::model::EmbeddingModel,
              tokenizer: &tokenizers::Tokenizer) -> Result<Vec<f32>, crate::Error> {
    let encoding = tokenizer.encode(text, true)
        .map_err(|e| crate::Error::Tokenization(e.to_string()))?;
    let ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
    let mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&x| x as i64).collect();
    let type_ids: Vec<i64> = encoding.get_type_ids().iter().map(|&x| x as i64).collect();
    model.embed(&ids, &mask, &type_ids)
}

/// Pre-normalized vectors: cosine similarity = dot product
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}
```

Since all embeddings are L2-normalized at creation time, **cosine similarity reduces to a simple dot product** — no square roots needed. For 384-dimensional vectors, LLVM's auto-vectorizer with `-C target-cpu=native` generates SIMD instructions automatically. This is fast enough for tens of thousands of chunks; the `simsimd` crate offers further acceleration for larger codebases via explicit AVX-512/NEON intrinsics.

---

## Error handling follows the thiserror/anyhow split

The library crate uses `thiserror` for structured, matchable error types. The binary crates use `anyhow` for convenient error propagation with contextual messages. This is the standard Rust pattern: **`thiserror` when callers need to differentiate failure modes, `anyhow` when they just need to report errors**.

```rust
// crates/ripvec-core/src/error.rs
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("model download failed: {0}")]
    Download(String),

    #[error("ONNX inference failed")]
    Inference(#[from] ort::Error),

    #[error("tokenization failed: {0}")]
    Tokenization(String),

    #[error("I/O error: {path}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("unsupported language: {0}")]
    UnsupportedLanguage(String),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}
```

The binary uses `anyhow::Result` throughout and adds `.context()` for human-readable error chains:

```rust
// crates/ripvec/src/main.rs
use anyhow::{Context, Result};

fn main() -> Result<()> {
    let args = cli::Args::parse();
    let model = ripvec_core::model::EmbeddingModel::load(
        &args.model_repo, &args.model_file
    ).context("failed to load embedding model")?;
    // ...
}
```

---

## CLI design with clap derive

```rust
// crates/ripvec/src/cli.rs
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "ripvec", version, about = "Semantic code search — like ripgrep but for meaning")]
pub struct Args {
    /// The natural language query to search for
    pub query: String,

    /// Root directory to search (defaults to current directory)
    #[arg(default_value = ".")]
    pub path: String,

    /// Number of results to show
    #[arg(short = 'n', long, default_value_t = 10)]
    pub top_k: usize,

    /// HuggingFace model repository
    #[arg(long, default_value = "BAAI/bge-small-en-v1.5")]
    pub model_repo: String,

    /// ONNX model filename within the repository
    #[arg(long, default_value = "onnx/model.onnx")]
    pub model_file: String,

    /// Output format: plain, json, or color
    #[arg(short, long, default_value = "color")]
    pub format: OutputFormat,

    /// Minimum similarity threshold (0.0 to 1.0)
    #[arg(short = 't', long, default_value_t = 0.0)]
    pub threshold: f32,

    /// Number of threads for parallel processing (0 = auto)
    #[arg(short = 'j', long, default_value_t = 0)]
    pub threads: usize,
}

#[derive(clap::ValueEnum, Clone, Debug)]
pub enum OutputFormat { Plain, Json, Color }
```

---

## MCP server for Claude Code and Cursor integration

The `ripvec-mcp` binary exposes a `semantic_search` tool over stdio transport using the `rmcp` crate. When configured in Claude Code or Cursor, it lets the AI invoke semantic code search as a native tool.

```rust
// crates/ripvec-mcp/src/main.rs
use rmcp::{ServerHandler, tool};
use rmcp::model::{ServerInfo, ServerCapabilities};
use serde::Deserialize;
use schemars::JsonSchema;

#[derive(Debug, Clone)]
pub struct RipvecServer;

#[derive(Deserialize, JsonSchema)]
pub struct SearchRequest {
    /// Natural language query describing the code you're looking for
    pub query: String,
    /// Root directory to search (defaults to current directory)
    #[serde(default = "default_path")]
    pub path: String,
    /// Maximum number of results to return
    #[serde(default = "default_top_k")]
    pub top_k: usize,
}

fn default_path() -> String { ".".to_string() }
fn default_top_k() -> usize { 10 }

#[tool(tool_box)]
impl RipvecServer {
    #[tool(description = "Search code semantically by meaning using vector embeddings. \
        Returns the most relevant functions, classes, and methods matching a natural \
        language query.")]
    async fn semantic_search(&self, #[tool(aggr)] req: SearchRequest) -> String {
        let model = ripvec_core::model::EmbeddingModel::load(
            "BAAI/bge-small-en-v1.5", "onnx/model.onnx"
        ).expect("model load");
        let tokenizer = ripvec_core::tokenize::load_tokenizer(
            "BAAI/bge-small-en-v1.5"
        ).expect("tokenizer load");

        let results = ripvec_core::embed::search(
            std::path::Path::new(&req.path),
            &req.query,
            &model,
            &tokenizer,
            req.top_k,
        ).expect("search");

        // Format as structured text for the LLM
        results.iter().enumerate().map(|(i, r)| {
            format!("{}. {} ({}:{}-{}, similarity: {:.3})\n```\n{}\n```",
                i + 1, r.chunk.name, r.chunk.file_path,
                r.chunk.start_line, r.chunk.end_line,
                r.similarity, r.chunk.content)
        }).collect::<Vec<_>>().join("\n\n")
    }
}

#[tool_handler]
impl ServerHandler for RipvecServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some("Semantic code search tool. Search codebases by meaning.".into()),
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            ..Default::default()
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let server = RipvecServer;
    let service = server.serve(tokio::io::stdin(), tokio::io::stdout()).await?;
    service.waiting().await?;
    Ok(())
}
```

Register the MCP server in Claude Code's configuration (`.claude/mcp.json`):

```json
{
  "mcpServers": {
    "ripvec": {
      "command": "cargo",
      "args": ["run", "--release", "--bin", "ripvec-mcp"],
      "cwd": "/path/to/ripvec"
    }
  }
}
```

Or after installing the binary: `{ "command": "ripvec-mcp" }`.

---

## Claude Code development environment setup

### CLAUDE.md for the ripvec project

```markdown
# CLAUDE.md

This is ripvec — a semantic code search CLI written in Rust. It uses ONNX
embedding models, tree-sitter parsing, and cosine similarity for meaning-based
code search.

## Commands
- `cargo build` — Build all workspace crates
- `cargo test --workspace` — Run all tests
- `cargo test -p ripvec-core` — Run core library tests only
- `cargo clippy --all-targets -- -D warnings` — Lint (must pass with zero warnings)
- `cargo fmt --check` — Format check
- `cargo nextest run` — Run tests with nextest (faster output)

## Architecture
Cargo workspace with three crates:
- `ripvec-core` — shared library (model loading, chunking, embedding, search)
- `ripvec` — CLI binary using clap
- `ripvec-mcp` — MCP server binary using rmcp

## Conventions
- Error handling: `thiserror` in ripvec-core, `anyhow` in binaries
- Use `?` operator everywhere — never `.unwrap()` except in tests
- Use `#[expect()]` over `#[allow()]` for lint suppression
- All public items must have doc comments
- Prefer `impl Trait` parameters over generic type parameters where possible
- tree-sitter `Parser` is not thread-safe — create per-thread with rayon's `for_each_init`
- tree-sitter `Query` is `Send + Sync` — share freely via `Arc`
- Keep `_mmap` field alive as long as the ONNX `Session` lives

## Testing
- Unit tests inline in modules (`#[cfg(test)]`)
- Integration tests in `tests/` using `assert_cmd`
- Test fixtures in `tests/fixtures/` with sample source files
- Run `cargo nextest run` for faster test execution

## Important
- The ONNX model is ~50MB — do not commit it to git
- `commit_from_memory_directly` requires the mmap to outlive the Session
- BGE models use CLS pooling; all-MiniLM uses mean pooling — don't mix them
```

### Recommended Claude Code plugins and tools

Install these for an optimal Rust development experience:

```bash
# Rust LSP integration — provides go-to-definition, diagnostics, completions
# Option 1: Official rust-analyzer LSP plugin
/plugin install rust-analyzer-lsp

# Option 2: Comprehensive Rust plugin with hooks (clippy, fmt, audit)
/plugin marketplace add zircote/rust-lsp
/plugin install rust-lsp@zircote

# Astral plugin for any Python tooling needs (useful if working on Python bindings)
/plugin marketplace add astral-sh/claude-code-plugins
/plugin install astral@astral-sh
# Provides /astral:uv, /astral:ruff, /astral:ty skills
```

Additionally, consider installing these MCP servers for enhanced AI-assisted Rust development:

- **`rust-mcp-server`** (`cargo install rust-mcp-server`) — Bridges LLMs with cargo commands: `cargo-check`, `cargo-build`, `cargo-test`, `cargo-clippy`, `cargo-fmt`, `cargo-machete`
- **`rust-analyzer-mcp`** (`cargo install rust-analyzer-mcp`) — Wraps rust-analyzer for symbol lookup, go-to-definition, find-references via MCP
- **`rust-docs-mcp-server`** — Fetches live crate documentation and provides a `query_rust_docs` tool

### Essential local development tools

```bash
rustup component add rust-analyzer clippy rustfmt
cargo install cargo-nextest       # Faster test runner
cargo install bacon               # Background compiler/watcher
cargo install cargo-insta         # Snapshot testing
cargo install cargo-audit         # Security vulnerability checker
cargo install cargo-machete       # Find unused dependencies
cargo install cargo-deny          # Dependency policy enforcement
cargo install cargo-semver-checks # Semver compatibility checker
```

Run `bacon` in a separate terminal during development — it continuously compiles and shows errors as you edit. Press `c` for clippy, `t` for tests, `n` for nextest.

---

## Testing strategy

### Unit tests for core logic

Every module in `ripvec-core` includes inline tests. The pattern is to test functions with known inputs and verify embedding dimensions, chunk boundaries, and similarity ordering:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_rust_function() {
        let source = "fn hello() { println!(\"hi\"); }\nfn world() {}";
        let config = crate::languages::config_for_extension("rs").unwrap();
        let chunks = chunk_file(Path::new("test.rs"), source, &config);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].name, "hello");
        assert_eq!(chunks[1].name, "world");
    }

    #[test]
    fn language_extension_mapping() {
        assert!(config_for_extension("rs").is_some());
        assert!(config_for_extension("py").is_some());
        assert!(config_for_extension("xyz").is_none());
    }

    #[test]
    fn dot_product_of_identical_normalized_vectors_is_one() {
        let v = vec![0.5773, 0.5773, 0.5773]; // approximately normalized
        let sim = dot_product(&v, &v);
        assert!((sim - 1.0).abs() < 0.01);
    }
}
```

### Integration tests with assert_cmd

```rust
// tests/integration.rs
use assert_cmd::Command;
use predicates::prelude::*;
use tempfile::TempDir;

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
fn searches_fixture_directory() {
    Command::cargo_bin("ripvec")
        .unwrap()
        .args(&["find the main entry point", "tests/fixtures/", "-n", "3"])
        .assert()
        .success()
        .stdout(predicate::str::contains("main"));
}

#[test]
fn fails_on_nonexistent_directory() {
    Command::cargo_bin("ripvec")
        .unwrap()
        .args(&["query", "/nonexistent/path"])
        .assert()
        .failure();
}
```

---

## CI/CD with GitHub Actions

### Continuous integration workflow

```yaml
# .github/workflows/ci.yml
name: CI
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
env:
  CARGO_TERM_COLOR: always
  RUSTFLAGS: -Dwarnings

jobs:
  check:
    name: Check & Lint
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: rustfmt, clippy
      - uses: Swatinem/rust-cache@v2
      - run: cargo fmt --all -- --check
      - run: cargo clippy --all-targets --all-features -- -D warnings
      - run: cargo doc --no-deps --all-features

  test:
    name: Test (${{ matrix.os }})
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - uses: taiki-e/install-action@cargo-nextest
      - run: cargo nextest run --all-features
      - run: cargo test --doc

  msrv:
    name: MSRV (1.80.0)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@master
        with:
          toolchain: "1.80.0"
      - uses: Swatinem/rust-cache@v2
      - run: cargo check --all-features

  security:
    name: Security Audit
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: rustsec/audit-check@v2
        with:
          token: ${{ secrets.GITHUB_TOKEN }}
```

### Release workflow with cargo-dist

```yaml
# .github/workflows/release.yml
name: Release
on:
  push:
    tags: ['v*']
permissions:
  contents: write

jobs:
  build:
    name: Build ${{ matrix.target }}
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        include:
          - { os: ubuntu-latest, target: x86_64-unknown-linux-gnu }
          - { os: ubuntu-latest, target: aarch64-unknown-linux-gnu, cross: true }
          - { os: macos-latest, target: x86_64-apple-darwin }
          - { os: macos-latest, target: aarch64-apple-darwin }
          - { os: windows-latest, target: x86_64-pc-windows-msvc }
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          targets: ${{ matrix.target }}
      - uses: Swatinem/rust-cache@v2
      - name: Install cross
        if: matrix.cross
        run: cargo install cross --git https://github.com/cross-rs/cross
      - name: Build
        shell: bash
        run: |
          CMD=${{ matrix.cross && 'cross' || 'cargo' }}
          $CMD build --release --target ${{ matrix.target }} -p ripvec -p ripvec-mcp
      - name: Package
        shell: bash
        run: |
          mkdir -p dist
          if [[ "${{ matrix.os }}" == "windows-latest" ]]; then
            7z a "dist/ripvec-${{ matrix.target }}.zip" \
              "target/${{ matrix.target }}/release/ripvec.exe" \
              "target/${{ matrix.target }}/release/ripvec-mcp.exe"
          else
            tar -czf "dist/ripvec-${{ matrix.target }}.tar.gz" \
              -C "target/${{ matrix.target }}/release" ripvec ripvec-mcp
          fi
      - uses: actions/upload-artifact@v4
        with:
          name: ${{ matrix.target }}
          path: dist/*

  release:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/download-artifact@v4
        with: { path: artifacts }
      - uses: softprops/action-gh-release@v1
        with:
          files: artifacts/**/*
          generate_release_notes: true

  publish:
    needs: release
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo publish -p ripvec-core
        env:
          CARGO_REGISTRY_TOKEN: ${{ secrets.CARGO_REGISTRY_TOKEN }}
      - run: sleep 30 && cargo publish -p ripvec
        env:
          CARGO_REGISTRY_TOKEN: ${{ secrets.CARGO_REGISTRY_TOKEN }}
```

For a more automated approach, initialize `cargo-dist` with `dist init --yes`, which generates the release workflow automatically and supports shell/PowerShell installers, Homebrew taps, and cargo-binstall metadata out of the box.

### Making the tool installable via cargo-binstall

Add this to the ripvec binary's `Cargo.toml`:

```toml
[package.metadata.binstall]
pkg-url = "{ repo }/releases/download/v{ version }/ripvec-{ target }.tar.gz"
bin-dir = "{ bin }{ binary-ext }"
pkg-fmt = "tgz"

[package.metadata.binstall.overrides.x86_64-pc-windows-msvc]
pkg-url = "{ repo }/releases/download/v{ version }/ripvec-{ target }.zip"
pkg-fmt = "zip"
```

Users can then install with `cargo binstall ripvec` (downloads pre-built binary) or `cargo install ripvec` (compiles from source).

---

## Phased implementation roadmap

This roadmap orders tasks by dependency and risk. Each phase produces a working, testable artifact.

**Phase 1 — Skeleton and model pipeline (days 1–3).** Set up the Cargo workspace, CLAUDE.md, and CI workflow. Implement model download via `hf-hub`, memory-mapped loading via `memmap2` + `ort`, tokenization via the `tokenizers` crate, and a basic embedding function. Validate by embedding a hardcoded string and printing the 384-dim vector.

**Phase 2 — Code chunking (days 4–5).** Implement the `languages.rs` registry mapping file extensions to tree-sitter grammars and query patterns. Build the `chunk.rs` module. Test against fixture files in all 8 supported languages. Validate that chunks correspond to real function/class boundaries.

**Phase 3 — Search pipeline (days 6–8).** Wire up `ignore`-based directory traversal, parallel chunking and embedding via rayon, cosine similarity ranking, and top-k selection. Build the clap CLI. At this point, `ripvec "find the main entry point" ./src` should return ranked results.

**Phase 4 — Polish and output (days 9–10).** Add colored terminal output with file paths, line numbers, similarity scores, and syntax-highlighted code snippets. Add JSON output mode. Handle edge cases: binary files, empty files, files exceeding the token limit (truncation), unsupported languages (skip gracefully).

**Phase 5 — MCP server (days 11–12).** Build `ripvec-mcp` using the `rmcp` crate. Test with Claude Code by configuring it in `.claude/mcp.json`. Verify that Claude can invoke `semantic_search` and receive structured results.

**Phase 6 — Distribution (days 13–14).** Configure `cargo-dist` or the manual release workflow. Set up the Homebrew tap. Add `cargo-binstall` metadata. Tag `v0.1.0`, push, and verify the full release pipeline produces binaries for all 5 target platforms.

---

## Choosing between embedding models

Three models deserve serious consideration, each with a distinct tradeoff profile:

| Model | Parameters | ONNX size | Dimensions | Max tokens | Code-trained | Best for |
|---|---|---|---|---|---|---|
| **BAAI/bge-small-en-v1.5** | 33M | ~127MB (fp32), ~32MB (int8) | 384 | 512 | No | General-purpose, good default |
| **all-MiniLM-L6-v2** | 22M | ~80MB | 384 | 256 | No | Fastest inference, smallest footprint |
| **jina-embeddings-v2-base-code** | 137M | ~300MB | 768 | 8192 | Yes (30 languages) | Best code search quality |

**The recommended default is `BAAI/bge-small-en-v1.5`** with CLS pooling. It offers the best balance of quality, size, and inference speed for a CLI tool. The quantized variant (`Qdrant/bge-small-en-v1.5-onnx-Q`) cuts size to ~32MB with negligible quality loss. For users working primarily with code and needing longer context windows, expose `jina-embeddings-v2-base-code` as a `--model-repo` flag option. Note that BGE uses CLS token pooling while MiniLM and Jina use mean pooling — the pooling strategy must match the model.

An alternative to building the embedding pipeline from scratch is the **`fastembed`** crate (v5+), which wraps `ort` + `tokenizers` into a single API: `TextEmbedding::try_new(InitOptions::new(EmbeddingModel::BGESmallENV15))`. This sacrifices control over mmap and inference threading but dramatically simplifies the code. Consider it for a rapid prototype before building the custom pipeline.

## Conclusion

The ripvec architecture is deliberately constrained — no server, no database, no persistent state — which makes the codebase small and the development path linear. The key technical risks are **ort's pre-release API stability** (the 2.x series is production-ready but not API-frozen, so pin the exact version) and **tree-sitter grammar version alignment** (grammar crates must match the core tree-sitter version's API). Start with Phase 1 and validate the mmap + ONNX pipeline first, since that's the most novel component. Everything else — file walking, parallel processing, CLI parsing — uses battle-tested crates from the ripgrep ecosystem. The MCP server integration transforms ripvec from a standalone tool into an AI-native development primitive, which is where the compound value lies: Claude Code can search by meaning, not just by text pattern, in any codebase it's working on.
