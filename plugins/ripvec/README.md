# ripvec

Semantic code search for Claude Code — find code by meaning, understand architecture through dependency graphs.

## Installation

```shell
# 1. Install the MCP server (requires Rust toolchain)
cargo install --git https://github.com/fnordpig/ripvec ripvec-mcp

# 2. Install the plugin
/plugin marketplace add fnordpig/ripvec
/plugin install ripvec@ripvec
```

The plugin checks for `ripvec-mcp` at session start and shows install instructions if missing.

### Prerequisites

- [Rust toolchain](https://rustup.rs/) (for building ripvec-mcp)
- macOS: Metal GPU acceleration (default), or MLX fallback
- Linux: CPU backend (CUDA optional with `--features cuda`)

## What You Get

### MCP Server (6 tools)

| Tool | What it does |
|------|-------------|
| `get_repo_map` | PageRank-weighted structural overview — shows which files matter most |
| `search_code` | Find code by meaning, not text. "retry with backoff" finds the implementation |
| `search_text` | Same but for docs/comments |
| `find_similar` | Given a file+line, find similar patterns elsewhere |
| `reindex` | Force re-embedding (auto-updates on file change) |
| `index_status` | Check readiness |

### Skills (3)

Skills activate automatically when Claude Code encounters matching tasks:

- **codebase-orientation** — Triggers on "how does this project work", "explain the architecture". Uses `get_repo_map` to orient before reading files. Saves 10+ sequential file reads.
- **semantic-discovery** — Triggers on "find the code that handles X". Guides Claude to use `search_code` instead of Grep for conceptual queries.
- **change-impact** — Triggers on "what breaks if I change this". Combines `get_repo_map(focus_file)` + LSP `findReferences` + `find_similar` for full blast radius.

### Commands (2)

- `/ripvec:map [file]` — Quick structural overview (optional focus file)
- `/ripvec:find "query"` — Semantic code search

### Agent (1)

- **code-explorer** — Deep codebase exploration combining repo map, semantic search, and LSP navigation

## How Results Connect to LSP

ripvec search results include `lsp_location` fields (file, line, character). These connect directly to LSP tools for precise navigation:

```
1. search_code("trait all backends implement")
   → finds EmbedBackend in backend/mod.rs:42

2. LSP goToDefinition on EmbedBackend
   → jumps to the trait definition

3. LSP findReferences on embed_batch method
   → shows every implementation and call site

4. LSP incomingCalls on a specific impl
   → traces who calls this backend
```

The pattern: **ripvec finds the concept → LSP navigates the specifics**. ripvec answers "where is the authentication logic?" and LSP answers "who calls `authenticate()` and what does it return?"

## When to Use What

| You need | Use | Why |
|----------|-----|-----|
| Project overview | `get_repo_map` | PageRank shows architectural spine |
| "Find error handling" | `search_code` | Semantic meaning, not text match |
| "Find `TODO`" | `Grep` | Exact text |
| Definition of `Foo` | LSP `goToDefinition` | Precise symbol resolution |
| All callers of `bar()` | LSP `findReferences` | Exact reference list |
| "Code similar to this function" | `find_similar` | Structural similarity |

## Performance

On M2 Max with BGE-small (384-dim):
- Corpus embedding: 312/s (Metal GPU, MPS GEMMs)
- Query embedding: 4ms
- Ranking: 0.3ms for 670 chunks
- Repo map build: 228ms (one-time), render: 0.15ms

## Supported Languages

Tree-sitter parsing for definitions + imports:
Rust, Python, JavaScript/TypeScript, Go, Java, C/C++

Semantic search works on any text file — the embedding model understands code semantics regardless of language.
