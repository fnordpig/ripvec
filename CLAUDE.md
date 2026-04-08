# ripvec

**CRITICAL** ALWAYS PREFER YOUR LSP / MCP / Rust SKILLS!

Semantic search CLI — like ripgrep but for meaning, not text. Four GPU/CPU
backends (Metal, MLX, CUDA, CPU), tree-sitter chunking, PageRank repo map,
and cosine similarity ranking. Rust 2024 edition.

## Commands
- `cargo check --workspace`              # Fast compilation check (prefer over build)
- `cargo test --workspace`               # All tests
- `cargo test -p ripvec-core`            # Core library tests only
- `cargo nextest run`                    # Faster test runner (requires cargo-nextest)
- `cargo clippy --all-targets -- -D warnings`  # Lint (zero warnings)
- `cargo fmt` / `cargo fmt --check`      # Format

## Architecture
Cargo workspace with three crates:
- `ripvec-core` — shared library (backends, chunking, embedding, search, repo map, cache)
- `ripvec` — CLI binary (clap, TUI with ratatui)
- `ripvec-mcp` — MCP server binary (rmcp, 7 tools + 1 resource) + LSP server (`--lsp` flag, tower-lsp-server)

### Backends (in detection priority order)
- **Metal** (default on macOS) — custom MSL kernels + MPS GEMMs, 73.8/s on M2 Max
- **MLX** (fallback on macOS) — mlx-rs, lazy eval graph fusion
- **CUDA** (Linux) — cudarc 0.19.4, cuBLAS tensor core FP16/INT8 GEMMs + custom NVRTC kernels, 435/s on RTX 4090
- **CPU** (everywhere) — ndarray + system BLAS (Accelerate/OpenBLAS/AOCL), 73.5/s on M2 Max

### Models
- **BGE-small-en-v1.5** (--fast) — ClassicBert, 384-dim, CLS pooling
- **ModernBERT** (default) — 768-dim, mean pooling

## MCP tool resolution

Two ripvec MCP servers run simultaneously in this project:

| Source | Command | Tool namespace |
|--------|---------|---------------|
| Project `.mcp.json` | `./target/release/ripvec-mcp` (local build) | `mcp__ripvec__*` |
| Plugin `.mcp.json` | downloaded binary | `mcp__plugin_ripvec_ripvec__*` |

**For development, prefer `mcp__ripvec__*`** — that's the local build with your latest changes.
The plugin binary is the last released version.

If tools don't appear in ToolSearch, try both namespaces:
```
ToolSearch("select:mcp__ripvec__search_code")
ToolSearch("select:mcp__plugin_ripvec_ripvec__search_code")
```

After `cargo build --release`, the project MCP picks up changes immediately
(next tool call). No restart needed — the binary is replaced in place.

Both servers running simultaneously is useful for A/B testing: call the same
query on `mcp__ripvec__*` (local build) and `mcp__plugin_ripvec_ripvec__*`
(released) to compare behavior before/after a change.

**A/B testing**: Search the same query on both to compare local vs released behavior:
```
mcp__ripvec__search_code("query")                      # local build
mcp__plugin_ripvec_ripvec__search_code("query")        # released version
```

## LSP server

`ripvec-mcp --lsp` serves Language Server Protocol over stdio, providing
code intelligence for all 21 supported languages (26 file extensions):

- `workspaceSymbol` — cross-language semantic search with PageRank boost
- `documentSymbol` — file outline via tree-sitter for all languages
- `goToDefinition` / `goToImplementation` — BM25 identifier match + PageRank
- `findReferences` — keyword search with content filtering
- `hover` — enriched content with scope chain
- `publishDiagnostics` — tree-sitter syntax error detection

Especially valuable for languages without dedicated LSPs (bash, HCL, TOML,
Ruby, Kotlin, Swift, Scala). For languages with dedicated LSPs (Rust, Go,
TypeScript), ripvec complements with cross-language semantic features.

## When to use ripvec vs LSP vs grep

| Need | Tool | Why |
|---|---|---|
| "How is the code organized?" | `get_repo_map` | PageRank-weighted structural overview |
| "Find error handling code" | `search_code` | Semantic meaning, not text match |
| "Go to definition of X" | LSP `goToDefinition` | Precise symbol resolution |
| "Who calls this function?" | LSP `findReferences` | Exact call sites |
| "Find files matching *.rs" | `Glob` | File pattern matching |
| "Find exact string 'TODO'" | `Grep` | Literal text search |

### Recommended workflow
1. `get_repo_map` — orient (which files matter, how they connect)
2. `search_code` — find implementations by meaning
3. LSP — navigate precisely (definitions, references, hover)

## Conventions
- Error handling: `thiserror` in ripvec-core, `anyhow` in binaries
- Never `.unwrap()` except in tests; use `?` everywhere
- `#[expect()]` over `#[allow()]` for lint suppression
- All public items must have doc comments (`///`)
- Prefer `impl Trait` parameters over generic type params where possible

## Chunking architecture
- Tree-sitter grammars are pluggable via a trait/registry — not hardcoded
- Support code (Rust, Python, JS/TS, Go, Java, C/C++) and plain text
  (paragraph/sentence splitting via sliding windows)
- Chunks enriched with scope chains + signatures for better embedding quality
- `Parser` is not thread-safe — create per-thread via rayon
- `Query` is Send + Sync — share via Arc

## Search modes
- `--mode hybrid` (default) — semantic + BM25 fusion via RRF (k=60)
- `--mode semantic` — pure vector similarity
- `--mode keyword` — pure BM25 keyword matching
- `--index` enables persistent cache with incremental re-embedding

## Cache resolution
1. `--cache-dir` override (highest priority)
2. `.ripvec/config.toml` in directory tree → `.ripvec/cache/` (repo-local)
3. `RIPVEC_CACHE` environment variable
4. `~/.cache/ripvec/` (default)

Use `--repo-level --index` to create a repo-local index. Subsequent runs
auto-detect `.ripvec/` — no flag needed.

## Key invariants
- Model weights (~33-100MB) must NOT be committed to git
- Metal: `_mmap` field must outlive `weight_buffer` (drop order matters)
- BGE-small uses CLS pooling; ModernBERT uses mean pooling — don't mix
- All embeddings are L2-normalized; cosine similarity = dot product
- Metal MAX_BATCH=32 for optimal padding; MLX MAX_BATCH=64; CUDA MAX_BATCH=32
- CUDA: `CudaSlice::clone()` does full D2D memcpy (not refcount) — never pool via clone
- CUDA: disable cudarc event tracking for single-stream usage (`ctx.disable_event_tracking()`)
- CUDA: use `compute_XX` (not `sm_XX`) for NVRTC arch to avoid PTX version mismatches
- Cache objects are zstd-compressed (level 1, ~8x smaller)

## Streaming pipeline
For corpora >= 1000 files, `embed_all` uses a three-stage streaming pipeline:
1. **Chunk** (rayon par_iter) → bounded channel → 2. **Tokenize** (single thread) → bounded channel → 3. **GPU embed** (main thread)
- Backpressure via `crossbeam_channel::bounded`
- GPU starts after first batch (~50ms), not after all files chunked
- Progress shown in bytes (total known from walk), not chunks (unknown until done)
- Small corpora use batch path (global sort-by-length for better padding)

## Before committing
Run: `cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test --workspace`
