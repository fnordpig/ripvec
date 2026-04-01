# ripvec

Semantic code search — like ripgrep, but for meaning.

ripvec finds code by what it *does*, not what it says. Ask "retry logic with
exponential backoff" and get the actual implementation, even if the code never
uses those words. Built on ModernBERT embeddings, tree-sitter chunking,
PageRank-weighted repo maps, and cosine similarity ranking.

## Features

- **Semantic search**: natural-language queries over codebases of any size
- **Interactive TUI**: embed once, then search as you type with live results
- **MCP server**: plug into Claude Code, Cursor, or any MCP-compatible editor
- **Hybrid ranking**: fuses semantic similarity with BM25 keyword matching
- **Persistent index**: incremental re-embedding — only changed files are re-processed
- **PageRank repo map**: structural overview ranking files by import-graph centrality
- **GPU-accelerated**: Metal, CUDA, and MLX backends with automatic detection

## Installation

```sh
cargo install ripvec ripvec-mcp
```

On macOS, Metal and MLX backends are enabled automatically. On Linux with an
NVIDIA GPU, build with CUDA support:

```sh
cargo install ripvec ripvec-mcp --features cuda
```

### Requirements

- Rust 1.88+ (2024 edition)
- macOS: Xcode Command Line Tools (provides Metal framework)
- Linux + CUDA: NVIDIA GPU with CUDA toolkit (optional, CPU fallback always available)

## Quick Start

### One-shot search

```sh
# Search the current directory
ripvec "error handling with retries"

# Search a specific project, show top 5 results
ripvec "database connection pooling" ~/src/my-project -n 5

# Use the fast model for quicker results
ripvec "authentication middleware" --fast
```

### Interactive TUI

```sh
# Launch TUI mode with persistent index
ripvec -i --index .

# Type queries interactively, results update as you type
# Press Enter to open a result in your editor
```

### MCP Server (for AI editors)

```sh
# Start the MCP server (reads RIPVEC_ROOT or uses cwd)
RIPVEC_ROOT=/path/to/project ripvec-mcp
```

Add to your editor's MCP config (`.mcp.json`):

```json
{
  "mcpServers": {
    "ripvec": {
      "command": "ripvec-mcp",
      "env": { "RIPVEC_ROOT": "/path/to/project" }
    }
  }
}
```

The MCP server provides 7 tools: `search_code`, `search_text`, `find_similar`,
`get_repo_map`, `reindex`, `index_status`, and `up_to_date`.

## Performance

Throughput on ModernBERT (768-dim, 22 layers):

| Backend | Hardware | Throughput | Notes |
|---------|----------|-----------|-------|
| CUDA FP16 | RTX 4090 | **435 chunks/s** | cuBLAS tensor cores + fused kernels |
| Metal MPS FP16 | M2 Max | **73.8 chunks/s** | Production macOS default |
| CPU (Accelerate) | M2 Max | **73.5 chunks/s** | Apple AMX coprocessor via BLAS |
| BGE-small MPS | M2 Max | **349 chunks/s** | `--fast` flag, 384-dim model |

## Supported Languages

Tree-sitter semantic chunking for:
Rust, Python, JavaScript/TypeScript, Go, Java, C/C++

All other file types fall back to sliding-window plain-text chunking.
Semantic search works on any text file regardless of language.

## Architecture

Cargo workspace with three crates:

| Crate | Description |
|-------|-------------|
| [`ripvec-core`](crates/ripvec-core) | Shared library: backends, chunking, embedding, search, repo map, cache |
| [`ripvec`](crates/ripvec) | CLI binary with clap args and ratatui TUI |
| [`ripvec-mcp`](crates/ripvec-mcp) | MCP server binary for AI editor integration |

### Embedding Models

- **ModernBERT** (default) — `nomic-ai/modernbert-embed-base`, 768-dim, mean pooling
- **BGE-small-en-v1.5** (`--fast`) — `BAAI/bge-small-en-v1.5`, 384-dim, CLS pooling

Model weights are downloaded automatically from Hugging Face on first run.

### GPU Backends

Detected automatically at startup:

| Backend | Platform | Implementation |
|---------|----------|---------------|
| Metal | macOS (default) | Custom MSL kernels + MPS GEMMs |
| MLX | macOS (fallback) | mlx-rs with lazy eval graph fusion |
| CUDA | Linux | cudarc + cuBLAS FP16 tensor cores + fused NVRTC kernels |
| CPU | Everywhere | ndarray + system BLAS (Accelerate / OpenBLAS) |

Override with `--backend metal|mlx|cuda|cpu`.

## Search Modes

```sh
ripvec "query" --mode hybrid    # Default: semantic + BM25 fusion
ripvec "query" --mode semantic  # Pure vector similarity
ripvec "query" --mode keyword   # Pure BM25 keyword matching
```

## Persistent Index

```sh
# Build index on first run, reuse on subsequent runs
ripvec "query" --index

# Force full rebuild
ripvec "query" --index --reindex

# Clear cached index
ripvec --clear-cache
```

The index uses zstd-compressed object storage with incremental re-embedding.
Only files that changed since the last run are re-processed.

## Documentation

- [Metal/MPS Architecture](docs/METAL_MPS_ARCHITECTURE.md)
- [CUDA Architecture](docs/CUDA_ARCHITECTURE.md)
- [Development Learnings](docs/LEARNINGS.md)

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.
