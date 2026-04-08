# ripvec

[![CI](https://github.com/fnordpig/ripvec/actions/workflows/ci.yml/badge.svg)](https://github.com/fnordpig/ripvec/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/ripvec.svg)](https://crates.io/crates/ripvec)
[![License: MIT/Apache-2.0](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE-MIT)

**Semantic code search + multi-language LSP. One binary, 19 grammars, zero setup.**

ripvec finds code by meaning, provides structural code intelligence across
every language it knows, and ranks results by how important each function is
in your codebase. It runs locally, bundles its own embedding model, and
uses whatever GPU you have.

```sh
$ ripvec "retry logic with exponential backoff" ~/src/my-project

 1. retry_handler.rs:42-78                                        [0.91]
    pub async fn with_retry<F, T>(f: F, max_attempts: u32) -> Result<T>
    where F: Fn() -> Future<Output = Result<T>> {
        let mut delay = Duration::from_millis(100);
        for attempt in 0..max_attempts {
            match f().await {
                Ok(v) => return Ok(v),
                Err(e) if attempt < max_attempts - 1 => {
                    sleep(delay).await;
                    delay *= 2;  // exponential backoff
    ...

 2. http_client.rs:156-189                                        [0.84]
    impl HttpClient {
        async fn request_with_backoff(&self, req: Request) -> Response {
    ...
```

The function is called `with_retry`, the variable is `delay` — "exponential
backoff" appears nowhere in the source. grep can't find this. ripvec can,
because it embeds both your query and the code into the same vector space
and measures similarity.

## When to use what

ripvec has three interfaces. Here's when each one matters:

| Interface | When to use it | Who uses it |
|-----------|---------------|-------------|
| **CLI** (`ripvec "query" .`) | Terminal search, interactive TUI, one-shot queries | You, directly |
| **MCP server** (`ripvec-mcp`) | AI agent needs to search or understand your codebase | Claude Code, Cursor, any MCP client |
| **LSP server** (`ripvec-mcp --lsp`) | Editor/agent needs symbols, definitions, diagnostics | Claude Code's LSP tool, editors |

The MCP server gives AI agents 7 tools (semantic search, repo maps, etc.).
The LSP server gives editors structural intelligence (outlines, go-to-definition,
syntax diagnostics). The CLI is for humans. Same binary for all three.

If you're using **Claude Code**, install the plugin — it sets up both MCP and LSP
automatically. Claude will use `search_code` when you ask conceptual questions
and the LSP for symbol navigation.

## Workflow: orient, search, navigate

ripvec is most useful when you combine its three capabilities:

**1. Orient** — `get_repo_map` returns a structural overview ranked by function-level
importance. One tool call replaces 10+ sequential file reads. Start here when
working on unfamiliar code.

**2. Search** — `search_code "authentication middleware"` finds implementations by
meaning across all 19 languages simultaneously. Results are ranked by relevance
and structural importance.

**3. Navigate** — LSP `documentSymbol` shows the file outline. `goToDefinition`
jumps to the likely definition (name-matched + ranked by importance).
`findReferences` shows usage sites. `incomingCalls`/`outgoingCalls` traces
the call graph.

## Semantic search

You describe behavior, ripvec finds the implementation:

| What you want | grep / ripgrep | ripvec |
|---------------|----------------|--------|
| "retry with backoff" | Nothing (code says `delay *= 2`) | Finds the retry handler |
| "database connection pool" | Comments mentioning "pool" | The pool implementation |
| "authentication middleware" | `// TODO: add auth` | The auth guard |
| "WebSocket lifecycle" | String "WebSocket" | Connect/disconnect handlers |

Search modes: `--mode hybrid` (default, semantic + BM25 fusion), `--mode semantic`
(pure vector similarity), `--mode keyword` (pure BM25). Hybrid is usually best.

## Multi-language LSP

ripvec serves LSP from a single binary for all 19 grammars. No per-language
server installs. It provides:

- **`documentSymbol`** — file outline: functions, fields, enum variants, constants, types, headings
- **`workspaceSymbol`** — cross-language symbol search with PageRank boost
- **`goToDefinition`** — best-effort name-based resolution, ranked by structural importance (not type-aware — use dedicated LSPs for precise resolution)
- **`findReferences`** — usage sites via hybrid search + content filtering
- **`hover`** — scope chain, signature, enriched context
- **`publishDiagnostics`** — tree-sitter syntax error detection after every edit
- **`incomingCalls` / `outgoingCalls`** — function-level call graph

For languages with dedicated LSPs (Rust, Python, Go, TypeScript), ripvec runs
alongside them — the dedicated server handles types, ripvec handles semantic
search and cross-language features. For languages without dedicated LSPs
(bash, HCL, Ruby, Kotlin, Swift, Scala), ripvec is the primary code intelligence.

JSON, YAML, TOML, and Markdown get structural outlines (keys, mappings, headings)
and syntax diagnostics — useful for navigating large config files, not comparable
to language-aware intelligence.

## Function-level PageRank

ripvec extracts call expressions from every function body using tree-sitter,
resolves callee names to definitions (same-file first, then imported files),
and computes PageRank on the resulting call graph.

This is name-based resolution, not type-aware — a call to `render()` resolves
to the first `render` definition found in scope. Precision is limited by
dynamic dispatch and polymorphism. But even approximate call graphs produce
meaningful structural importance signals, and the ranking improves search quality
in practice.

The boost is multiplicative and log-saturated so high-PageRank functions get
amplified without dominating results. Zero-relevance stays at zero regardless
of importance.

## Install

### Pre-built binaries (fastest)

```sh
cargo binstall ripvec ripvec-mcp
```

Requires [cargo-binstall](https://github.com/cargo-bins/cargo-binstall).
Downloads a pre-built binary for your platform — no compilation.

### From source

```sh
cargo install ripvec ripvec-mcp
```

For CUDA (Linux with NVIDIA GPU):

```sh
cargo install ripvec ripvec-mcp --features cuda
```

### Claude Code plugin

```sh
claude plugin install ripvec@fnordpig-my-claude-plugins
```

The plugin auto-downloads the binary for your platform on first use and
configures both MCP and LSP servers. It includes 3 skills (codebase orientation,
semantic discovery, change impact analysis), 3 commands (`/map`, `/find`,
`/repo-index`), and a code exploration agent. CUDA is auto-detected via `nvidia-smi`.

### Platforms

| Platform | Backends | GPU |
|----------|----------|-----|
| macOS Apple Silicon | Metal + MLX + CPU (Accelerate) | Metal auto-enabled |
| Linux x86_64 | CPU (OpenBLAS) | CUDA with `--features cuda` |
| Linux ARM64 (Graviton) | CPU (OpenBLAS) | CUDA with `--features cuda` |

Model weights (~100MB) download automatically on first run.

## Usage

### CLI

```sh
ripvec "error handling" .                    # Search current directory
ripvec "form validation hooks" -n 5          # Top 5 results
ripvec "database migration" --mode keyword   # BM25 only
ripvec "auth flow" --fast                    # Lighter model (BGE-small, 4x faster)
ripvec -i --index .                          # Interactive TUI with persistent index
```

### MCP server

```json
{ "mcpServers": { "ripvec": { "command": "ripvec-mcp" } } }
```

Tools: `search_code`, `search_text`, `find_similar`, `get_repo_map`,
`reindex`, `index_status`, `up_to_date`.

### LSP server

```sh
ripvec-mcp --lsp   # serves LSP over stdio
```

Same binary, `--lsp` flag selects protocol.

## Indexing

### No index required

ripvec works out of the box — point it at a directory and search. No pre-indexing,
no database, no config.

### Persistent cache

```sh
ripvec "query" --index    # First run embeds; subsequent runs are instant
```

Cache uses a Merkle-tree diff system: content-addressed chunks, per-directory
hash trees, zstd compression (~8x smaller). Only changed files get re-embedded.
The MCP server adds a file watcher for live re-indexing (2-second debounce).

### Team sharing

```sh
ripvec --index --repo-level "query"
git add .ripvec/ && git commit -m "add search index"
```

Teammates who clone get instant search — zero embedding time. Uses portable
bitcode serialization (architecture-independent). File timestamps self-heal
after clone.

## Supported languages

19 tree-sitter grammars, 30 file extensions:

| Language | Extensions | Extracted elements |
|----------|-----------|-------------------|
| Rust | `.rs` | functions, structs, enums, variants, fields, impls, traits, consts, mods |
| Python | `.py` | functions, classes, assignments |
| JavaScript | `.js` `.jsx` | functions, classes, methods, variables |
| TypeScript | `.ts` `.tsx` | functions, classes, interfaces, type aliases, enums |
| Go | `.go` | functions, methods, types, constants |
| Java | `.java` | methods, classes, interfaces, enums, fields, constructors |
| C | `.c` `.h` | functions, structs, enums, typedefs |
| C++ | `.cpp` `.cc` `.cxx` `.hpp` | functions, classes, namespaces, enums, fields |
| Bash | `.sh` `.bash` `.bats` | functions, variables |
| Ruby | `.rb` | methods, classes, modules, constants |
| HCL / Terraform | `.tf` `.tfvars` `.hcl` | blocks (resources, data, variables) |
| Kotlin | `.kt` `.kts` | functions, classes, objects, properties |
| Swift | `.swift` | functions, classes, protocols, properties |
| Scala | `.scala` | functions, classes, traits, objects, vals, types |
| TOML | `.toml` | tables, key-value pairs |
| JSON | `.json` | object keys |
| YAML | `.yaml` `.yml` | mapping keys |
| Markdown | `.md` | headings |

Unsupported file types get sliding-window plain-text chunking. The embedding
model handles any language — tree-sitter just provides better chunk boundaries.

## Performance

**Without an index** (first run on a codebase):

| Hardware | Throughput | Time (Flask corpus, 2383 chunks) |
|----------|-----------|----------------------------------|
| RTX 4090 (CUDA) | 435 chunks/s | ~5s |
| M2 Max (Metal) | 73.8 chunks/s | ~32s |
| M2 Max (CPU/Accelerate) | 73.5 chunks/s | ~32s |

Metal and CPU show similar throughput on M2 Max because macOS Accelerate
routes BLAS operations through the AMX coprocessor regardless of backend.
The Metal backend has headroom on larger batches and non-BLAS operations.

**With an index**: milliseconds. Merkle diff skips unchanged files entirely.

**Memory**: ~500MB during embedding (model weights + batch buffers). Index
queries use ~100MB (loaded embeddings + BM25 inverted index).

## How it compares

| Tool | Type | Key difference from ripvec |
|------|------|--------------------------|
| ripgrep | Text search | No semantic understanding |
| Sourcegraph | Cloud AI platform | $49-59/user/month, code leaves your machine |
| grepai | Local semantic search | Requires Ollama for embeddings |
| mgrep | Semantic search | Uses cloud embeddings (Mixedbread AI) |
| Serena | MCP symbol navigation | Requires per-language LSP servers installed |
| Bloop | Was semantic + navigation | Archived Jan 2025 |
| VS Code anycode | Tree-sitter outlines | Editor-only, no cross-file search |
| Cursor @Codebase | IDE semantic search | Cursor-only, sends embeddings to cloud |

ripvec is self-contained (no Ollama, no cloud, no per-language setup), runs
on your GPU, and combines search + LSP + structural ranking in one binary.

## Scoring pipeline

```
query → ModernBERT embedding (768-dim)
     → cosine similarity ranking
     → BM25 keyword ranking
     → Reciprocal Rank Fusion (k=60)
     → min-max normalization
     → × function-level PageRank boost (log-saturated)
     → threshold + top-k
```

The min-max normalization maps the best result to 1.0 within each query — this
means the threshold is relative, not absolute. A weak result set can still pass
threshold if it's the best available. This is a known tradeoff; future versions
may switch to z-score normalization for better calibration.

## Limitations

- **goToDefinition is best-effort**: name-based, not type-aware. Use dedicated
  LSPs (rust-analyzer, pyright, gopls) when you need precise type resolution.
- **Call graph is approximate**: name-based resolution has false positives with
  common function names (`new`, `run`, `render`). Cross-crate resolution is
  limited to workspace members.
- **Cold start**: first search without an index embeds everything — 5s on CUDA,
  32s on Apple Silicon for a medium codebase. Use `--index` for repeated searches.
- **English-centric**: ModernBERT was trained primarily on English text. Queries
  and code comments in other languages will have lower recall.

## Architecture

Cargo workspace with three crates:

| Crate | Role |
|-------|------|
| [`ripvec-core`](crates/ripvec-core) | Backends, chunking, embedding, search, repo map, cache, call graph |
| [`ripvec`](crates/ripvec) | CLI binary (clap + ratatui TUI) |
| [`ripvec-mcp`](crates/ripvec-mcp) | MCP + LSP server binary (rmcp + tower-lsp-server) |

### GPU backends

| Backend | Platform | Approach |
|---------|----------|----------|
| Metal | macOS (default) | Custom MSL kernels + MPS GEMMs |
| MLX | macOS (fallback) | mlx-rs, lazy eval graph fusion |
| CUDA | Linux | cudarc + cuBLAS FP16 tensor cores + fused NVRTC kernels |
| CPU | Everywhere | ndarray + system BLAS (Accelerate / OpenBLAS) |

### Embedding models

- **ModernBERT** (default) — 768-dim, mean pooling, 22 layers
- **BGE-small** (`--fast`) — 384-dim, CLS pooling, 12 layers

## Development

```sh
cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test --workspace
```

See [CLAUDE.md](CLAUDE.md) for detailed development conventions, architecture
notes, and MCP tool namespace resolution.

### Docs

- [Metal/MPS Architecture](docs/METAL_MPS_ARCHITECTURE.md)
- [CUDA Architecture](docs/CUDA_ARCHITECTURE.md)
- [Development Learnings](docs/LEARNINGS.md)

## License

Licensed under either of [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT) at your option.
