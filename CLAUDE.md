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
- `ripvec-mcp` — MCP server binary (rmcp, 6 tools + 1 resource)

### Backends (in detection priority order)
- **Metal** (default on macOS) — custom MSL kernels + MPS GEMMs, 312/s on M2 Max
- **MLX** (fallback on macOS) — mlx-rs, lazy eval graph fusion
- **CUDA** (Linux) — cudarc, custom CUDA kernels, FP16 tensor core GEMM
- **CPU** (everywhere) — ndarray + system BLAS (Accelerate/OpenBLAS/AOCL)

### Models
- **BGE-small-en-v1.5** (default) — ClassicBert, 384-dim, CLS pooling
- **CodeRankEmbed** (--code) — NomicBert, 768-dim, mean pooling, MRL-trained

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
- Support code (Rust, Python, JS/TS, Go, Java, C/C++), structured text
  (SQL, Jinja2), and plain text (paragraph/sentence splitting)
- Chunks enriched with scope chains + signatures for better embedding quality
- `Parser` is not thread-safe — create per-thread via rayon
- `Query` is Send + Sync — share via Arc

## Key invariants
- Model weights (~33-100MB) must NOT be committed to git
- Metal: `_mmap` field must outlive `weight_buffer` (drop order matters)
- BGE models use CLS pooling; CodeRankEmbed uses mean pooling — don't mix
- All embeddings are L2-normalized; cosine similarity = dot product
- Metal MAX_BATCH=32 for optimal padding; MLX MAX_BATCH=64

## Before committing
Run: `cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test --workspace`
