# ripvec

**Semantic code search — like ripgrep, but for meaning.**

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

grep can't do this. ripvec understands what code *does*, not just what it says.
The function is called `with_retry` and the variable is `delay` — no mention of
"exponential backoff" anywhere — but ripvec finds it because it understands the
*meaning*.

## Why ripvec?

**You describe the behavior, ripvec finds the code.**

| What you're looking for | grep / ripgrep | ripvec |
|------------------------|----------------|--------|
| "retry with backoff" | Finds nothing (code says `delay *= 2`) | Finds the retry handler |
| "database connection pooling" | Matches comments mentioning "pool" | Finds the actual pool implementation |
| "authentication middleware" | Matches `// TODO: add auth` | Finds the auth guard/middleware |
| "WebSocket lifecycle" | Matches the string "WebSocket" | Finds connect/disconnect/reconnect handlers |

ripvec embeds your codebase into a vector space using ModernBERT, then ranks
results by cosine similarity. It also fuses BM25 keyword matching for hybrid
search that catches both meaning *and* exact terms.

## Install

```sh
cargo install ripvec ripvec-mcp
```

That's it. Model weights download automatically on first run (~100MB).

macOS gets Metal GPU acceleration by default. Linux gets CPU, or add CUDA:

```sh
cargo install ripvec ripvec-mcp --features cuda
```

## Usage

### Search from the command line

```sh
ripvec "error handling" .                    # Search current directory
ripvec "form validation hooks" -n 5          # Top 5 results
ripvec "database migration" --mode keyword   # BM25 only (fast, exact)
ripvec "auth flow" --fast                    # Lighter model, 4x faster
```

### Interactive TUI — search as you type

```sh
ripvec -i --index .
```

Embeds your codebase once, then gives you instant search-as-you-type with
syntax-highlighted previews. Press Enter to open in your editor.

### MCP server — give your AI editor semantic search

```json
{
  "mcpServers": {
    "ripvec": { "command": "ripvec-mcp" }
  }
}
```

Drop that in `.mcp.json` and Claude Code / Cursor gets 7 tools:
`search_code`, `search_text`, `find_similar`, `get_repo_map`, `reindex`,
`index_status`, `up_to_date`. Your AI can now search by meaning instead of
grepping blindly.

### No setup required — and no index required either

ripvec works out of the box with **zero configuration**. Just point it at a
directory and search. No pre-indexing step, no database, no config files:

```sh
ripvec "error handling" ~/src/some-project   # Just works. No setup.
```

### Persistent index with live updates

For repeated searches, add `--index` to cache embeddings:

```sh
ripvec "query" --index           # First run embeds, subsequent runs are instant
ripvec "query" --index --reindex # Force rebuild
```

The index uses a **Merkle-tree diffing system** modeled on git's object store:
content-addressed chunks with per-directory hash trees detect exactly which
files changed since the last run. Only modified files are re-embedded — everything
else loads from zstd-compressed cache (~8x smaller than raw). The MCP server
uses this with a file watcher for **live re-indexing** as you edit code (2-second
debounce).

## How fast?

**Without an index** (one-shot search):

| Setup | Embedding speed | Wall clock (Flask, 2383 chunks) |
|-------|----------------|-------------------------------|
| RTX 4090 (CUDA) | **435 chunks/s** | ~5s |
| M2 Max (Metal) | **73.8 chunks/s** | ~32s |
| M2 Max (CPU) | **73.5 chunks/s** | ~32s |

**With an index** (subsequent searches): **instant** (milliseconds).
On a 15MB Go codebase (~15K chunks), CUDA indexes in ~35s on first run.

## Supported languages

Tree-sitter semantic chunking (functions, classes, methods with scope context):
**Rust, Python, JavaScript/TypeScript, Go, Java, C/C++**

Every other file type gets sliding-window plain-text chunking. The embedding
model understands code semantics regardless of language — you can search
YAML, SQL, Markdown, config files, anything.

## How ripvec compares

**vs grep / ripgrep** — ripvec finds code by meaning. grep finds code by text.
Use both — ripvec for "find the retry logic", grep for "find `TODO`".

**vs Sourcegraph / GitHub search** — ripvec runs locally on your machine. Your
code never leaves your laptop. No servers, no subscriptions, no cloud.

**vs Serena / LSP tools** — ripvec finds *what* code to look at. LSP tells you
the *details* (definitions, references, types). They're complementary — ripvec
answers "where is authentication handled?" and LSP answers "who calls
`authenticate()` and what does it return?"

**vs grepai / mgrep / cloud tools** — ripvec is self-contained. No Ollama, no
API keys, no Docker, no external embedding service. One binary, bundled model
weights, GPU acceleration on hardware you already own.

**vs Bloop** — Bloop was archived in January 2025. ripvec fills the same niche
(Rust, semantic, local, open source) with better technology: ModernBERT
embeddings, hybrid BM25+vector ranking, PageRank repo maps, and Metal/CUDA
GPU acceleration.

## How it works

1. **Walk** your codebase, respecting `.gitignore`
2. **Chunk** files into semantic units via tree-sitter (or sliding windows)
3. **Embed** each chunk using ModernBERT (768-dim vectors, GPU-accelerated)
4. **Rank** by cosine similarity to your query + BM25 keyword fusion
5. **Cache** embeddings for instant subsequent searches

The search index also includes a PageRank-weighted repo map — a structural
overview showing which files are architecturally central based on their
import graph. Use `get_repo_map` in the MCP server or `ripvec --repo-map`.

---

## For contributors

### Architecture

Cargo workspace with three crates:

| Crate | Role |
|-------|------|
| [`ripvec-core`](crates/ripvec-core) | Backends, chunking, embedding, search, repo map, cache |
| [`ripvec`](crates/ripvec) | CLI binary (clap + ratatui TUI) |
| [`ripvec-mcp`](crates/ripvec-mcp) | MCP server binary (rmcp) |

### GPU backends

| Backend | Platform | How |
|---------|----------|-----|
| Metal | macOS (default) | Custom MSL kernels + MPS GEMMs via AMX |
| MLX | macOS (fallback) | mlx-rs, lazy eval graph fusion |
| CUDA | Linux | cudarc + cuBLAS FP16 tensor cores + fused NVRTC kernels |
| CPU | Everywhere | ndarray + system BLAS (Accelerate / OpenBLAS) |

### Embedding models

- **ModernBERT** (default) — `nomic-ai/modernbert-embed-base`, 768-dim, mean pooling, 22 layers
- **BGE-small** (`--fast`) — `BAAI/bge-small-en-v1.5`, 384-dim, CLS pooling, 12 layers

### Docs

- [Metal/MPS Architecture](docs/METAL_MPS_ARCHITECTURE.md)
- [CUDA Architecture](docs/CUDA_ARCHITECTURE.md)
- [Development Learnings](docs/LEARNINGS.md)

## License

Licensed under either of [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT) at your option.
