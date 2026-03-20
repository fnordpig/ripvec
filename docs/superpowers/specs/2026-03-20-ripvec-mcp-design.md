# ripvec-mcp Server Design

**Date:** 2026-03-20
**Status:** Approved

## Goal

Make ripvec available as an MCP server for Claude Code, providing semantic
search over code and text with persistent in-memory indexing, lazy model
loading, and LSP-compatible result format.

## Server Lifecycle

1. Claude Code launches `ripvec-mcp` via stdio transport
2. Server reads configuration from environment:
   - `RIPVEC_ROOT` — project root to index (defaults to cwd)
   - `RIPVEC_MODEL` — default model repo (defaults to `BAAI/bge-small-en-v1.5`)
3. Spawns a background tokio task to index `RIPVEC_ROOT` immediately
4. Begins accepting tool calls while indexing is in progress
5. Search tools block until the index is ready; `index_status` is always available

## Shared State

```
RipvecServer {
    index:          Arc<RwLock<Option<SearchIndex>>>   // None until indexing completes
    chunks:         Arc<RwLock<Vec<CodeChunk>>>         // paired with index
    code_backend:   OnceCell<Arc<dyn EmbedBackend>>    // lazy, first search_code call
    text_backend:   OnceCell<Arc<dyn EmbedBackend>>    // lazy, first search_text call
    code_tokenizer: OnceCell<Arc<Tokenizer>>
    text_tokenizer: OnceCell<Arc<Tokenizer>>
    project_root:   PathBuf
    indexing:       Arc<AtomicBool>                    // true while background index runs
}
```

Embedding backends are loaded lazily via `OnceCell` — whichever model is needed
first gets loaded on that call. This saves ~50MB of memory when only one model
is used in a session.

## Tools

### `search_code`

Semantic search optimized for code. Uses `nomic-ai/CodeRankEmbed` with the
required query prefix applied automatically.

**Parameters:**
- `query: string` — natural language description of what you're looking for
- `top_k: int` (default 10) — maximum results
- `threshold: float` (default 0.3) — minimum similarity

### `search_text`

Semantic search for prose, documentation, comments. Uses `BAAI/bge-small-en-v1.5`.

**Parameters:** same as `search_code`.

### `find_similar`

Find code semantically similar to a given location. Locates the chunk
containing the specified line, uses its embedding as the query vector, and
returns ranked similar chunks (excluding the input chunk).

**Parameters:**
- `file_path: string` — absolute path to the file
- `line: int` — 0-based line number (LSP-compatible)
- `top_k: int` (default 10)

### `reindex`

Rebuild the in-memory index from the project root. Use after significant
file changes.

**Parameters:** none.

**Returns:** `{ chunks: int, files: int, duration_ms: int }`

### `index_status`

Report index health. Always available, even during indexing.

**Parameters:** none.

**Returns:**
```json
{
  "ready": true,
  "indexing": false,
  "chunks": 2383,
  "files": 19,
  "extensions": { "rs": 15, "py": 3, "toml": 1 },
  "project_root": "/Users/rwaugh/src/mine/ripvec"
}
```

## Result Format

All search tools (`search_code`, `search_text`, `find_similar`) return the
same structure. Results are designed to chain directly into LSP tools — Claude
can pass `lsp_location` fields straight to `rust_analyzer_hover`,
`rust_analyzer_references`, etc.

```json
{
  "results": [
    {
      "lsp_location": {
        "file_path": "/abs/path/file.rs",
        "start_line": 42,
        "start_character": 0,
        "end_line": 58,
        "end_character": 0
      },
      "symbol_name": "embed_batch",
      "similarity": 0.823,
      "preview": "fn embed_batch(&self, encodings: &[Encoding]) -> ..."
    }
  ]
}
```

- **Lines are 0-based** in MCP output (converted from 1-based internal `CodeChunk`)
- **`start_character` / `end_character`** default to 0 (chunk-level granularity)
- **`preview`** is truncated to ~200 characters
- **`symbol_name`** is the function/struct/definition name from tree-sitter

## Configuration (`.mcp.json`)

```json
{
  "mcpServers": {
    "ripvec": {
      "command": "ripvec-mcp",
      "env": {
        "RIPVEC_ROOT": "${workspaceFolder}",
        "RIPVEC_MODEL": "BAAI/bge-small-en-v1.5"
      }
    }
  }
}
```

## Implementation Notes

- The existing `SearchIndex` (ndarray-based BLAS dot products) is reused for
  `find_similar` — the query is just the chunk's own embedding vector.
- `reindex` cancels any in-progress indexing, then spawns a fresh background task.
- The default backend for indexing is the text backend (BGE-small). `search_code`
  loads CodeRankEmbed lazily and re-embeds the query only — the index embeddings
  are from whichever backend was used for indexing. This means `find_similar`
  works regardless of which search tool is used, since it compares embeddings
  within the same vector space.
- For `search_code`: the index is built with the text backend but the query
  is embedded with CodeRankEmbed. This is a known cross-model query and works
  because both models produce L2-normalized vectors in comparable semantic
  spaces. If quality is insufficient, a future iteration can maintain dual
  indexes.

## Out of Scope (Future)

- Dual-index (separate code and text embeddings) for higher recall
- Incremental re-indexing (only changed files)
- Disk-persisted index across sessions
- MCP resources (exposing chunks as browsable resources)
