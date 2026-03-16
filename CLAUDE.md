# ripvec

Semantic search CLI — like ripgrep but for meaning, not text. Searches code,
structured text (SQL, Jinja2, etc.), and plain text using ONNX embeddings,
tree-sitter chunking, and cosine similarity. Rust 2024 edition.

## Commands
- `cargo check --workspace`              # Fast compilation check (prefer over build)
- `cargo test --workspace`               # All tests
- `cargo test -p ripvec-core`            # Core library tests only
- `cargo nextest run`                    # Faster test runner (requires cargo-nextest)
- `cargo clippy --all-targets -- -D warnings`  # Lint (zero warnings)
- `cargo fmt` / `cargo fmt --check`      # Format

## Architecture
Cargo workspace with three crates (target structure):
- `ripvec-core` — shared library (model, chunking, embedding, search)
- `ripvec` — CLI binary (clap)
- `ripvec-mcp` — MCP server binary (rmcp)

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
- `Parser` is not thread-safe — create per-thread via rayon
- `Query` is Send + Sync — share via Arc

## Key invariants
- ONNX model (~50MB) must NOT be committed to git
- `_mmap` field must outlive the ONNX `Session`
- BGE models use CLS pooling; all-MiniLM uses mean pooling — don't mix
- All embeddings are L2-normalized; cosine similarity = dot product

## Before committing
Run: `cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test --workspace`
