# ripvec-mcp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the ripvec-mcp server with persistent indexing, lazy dual-model loading, 5 tools (search_code, search_text, find_similar, reindex, index_status), and LSP-compatible result format.

**Architecture:** The server indexes the project root at startup on a background tokio task. Two embedding models (CodeRankEmbed, BGE-small) load lazily on first use via `tokio::sync::OnceCell`. All search tools query an in-memory `SearchIndex` (BLAS dot products) and return results with `lsp_location` fields that chain directly into LSP tools.

**Tech Stack:** Rust 2024, rmcp (MCP protocol), tokio (async runtime), ripvec-core (embedding + chunking), ndarray (BLAS index), serde_json (structured results).

**Spec:** `docs/superpowers/specs/2026-03-20-ripvec-mcp-design.md`

---

## File Structure

### Move to ripvec-core (shared by CLI TUI + MCP)

- **Move:** `crates/ripvec/src/tui/index.rs` → `crates/ripvec-core/src/index.rs`
  - `SearchIndex` struct and `rank()` method
  - Re-export from `ripvec-core/src/lib.rs`
  - Update `crates/ripvec/src/tui/mod.rs` to use `ripvec_core::index::SearchIndex`

- **Move:** `tokenize_query` from `crates/ripvec/src/tui/mod.rs` → `crates/ripvec-core/src/tokenize.rs`
  - Re-export as `ripvec_core::tokenize::tokenize_query`
  - Update TUI to use the shared version

### Rewrite ripvec-mcp

- **Rewrite:** `crates/ripvec-mcp/src/main.rs` — startup, env config, background indexing, server launch
- **Create:** `crates/ripvec-mcp/src/server.rs` — `RipvecServer` struct, shared state, `ServerHandler` impl
- **Create:** `crates/ripvec-mcp/src/tools.rs` — all 5 tool handlers + request/response types
- **Create:** `crates/ripvec-mcp/src/result.rs` — `LspLocation`, `SearchResultItem`, JSON serialization

### Deps

- **Modify:** `crates/ripvec-mcp/Cargo.toml` — add `serde_json`, `ndarray` (via ripvec-core)
- **Modify:** `crates/ripvec-core/Cargo.toml` — add `ndarray` if not already present

---

## Task 1: Move SearchIndex to ripvec-core

**Files:**
- Move: `crates/ripvec/src/tui/index.rs` → `crates/ripvec-core/src/index.rs`
- Modify: `crates/ripvec-core/src/lib.rs`
- Modify: `crates/ripvec/src/tui/mod.rs`
- Modify: `crates/ripvec-core/Cargo.toml` (ndarray dep if needed)

- [ ] **Step 1: Check ndarray is in ripvec-core's deps**

Run: `grep ndarray crates/ripvec-core/Cargo.toml`
If missing, add it.

- [ ] **Step 2: Copy index.rs to ripvec-core**

Copy `crates/ripvec/src/tui/index.rs` to `crates/ripvec-core/src/index.rs`. Update the module path for `ripvec_core::similarity::rank_all` and `ripvec_core::chunk::CodeChunk` (they should already be correct since the file was using `ripvec_core::` paths).

- [ ] **Step 3: Add `pub mod index;` to ripvec-core's lib.rs**

- [ ] **Step 4: Update TUI to use `ripvec_core::index::SearchIndex`**

In `crates/ripvec/src/tui/mod.rs`, replace `mod index;` with `use ripvec_core::index;` and delete the old `crates/ripvec/src/tui/index.rs`.

- [ ] **Step 5: Verify**

Run: `cargo check --workspace --all-targets`
Run: `cargo test --workspace`

- [ ] **Step 6: Commit**

```
refactor: move SearchIndex to ripvec-core for sharing with MCP
```

---

## Task 2: Move tokenize_query to ripvec-core

**Files:**
- Modify: `crates/ripvec-core/src/tokenize.rs`
- Modify: `crates/ripvec/src/tui/mod.rs`

- [ ] **Step 1: Add `tokenize_query` to ripvec-core's tokenize.rs**

```rust
/// Tokenize a query string for embedding, truncating to `model_max_tokens`.
pub fn tokenize_query(
    text: &str,
    tokenizer: &tokenizers::Tokenizer,
    model_max_tokens: usize,
) -> crate::Result<crate::backend::Encoding> {
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| crate::Error::Other(anyhow::anyhow!("tokenization failed: {e}")))?;

    let len = encoding.get_ids().len().min(model_max_tokens);
    Ok(crate::backend::Encoding {
        input_ids: encoding.get_ids()[..len].iter().map(|&x| i64::from(x)).collect(),
        attention_mask: encoding.get_attention_mask()[..len].iter().map(|&x| i64::from(x)).collect(),
        token_type_ids: encoding.get_type_ids()[..len].iter().map(|&x| i64::from(x)).collect(),
    })
}
```

- [ ] **Step 2: Update TUI to use `ripvec_core::tokenize::tokenize_query`**

Remove the local `tokenize_query` from `crates/ripvec/src/tui/mod.rs`. Import the shared one.

- [ ] **Step 3: Verify**

Run: `cargo check --workspace --all-targets`
Run: `cargo test --workspace`

- [ ] **Step 4: Commit**

```
refactor: move tokenize_query to ripvec-core for sharing with MCP
```

---

## Task 3: Result types (result.rs)

**Files:**
- Create: `crates/ripvec-mcp/src/result.rs`

- [ ] **Step 1: Create result.rs with LSP-compatible types**

```rust
use serde::Serialize;
use ripvec_core::chunk::CodeChunk;

/// LSP-compatible source location. Lines and characters are 0-based.
#[derive(Serialize)]
pub struct LspLocation {
    pub file_path: String,
    pub start_line: usize,
    pub start_character: usize,
    pub end_line: usize,
    pub end_character: usize,
}

/// A single search result with LSP location, symbol name, similarity, and preview.
#[derive(Serialize)]
pub struct SearchResultItem {
    pub lsp_location: LspLocation,
    pub symbol_name: String,
    pub similarity: f32,
    pub preview: String,
}

/// Top-level search response.
#[derive(Serialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResultItem>,
}

const MAX_PREVIEW_LEN: usize = 200;

impl SearchResultItem {
    /// Build from a `CodeChunk` and similarity score.
    /// Converts 1-based chunk lines to 0-based LSP lines.
    pub fn from_chunk(chunk: &CodeChunk, similarity: f32) -> Self {
        let preview = if chunk.content.len() > MAX_PREVIEW_LEN {
            format!("{}...", &chunk.content[..MAX_PREVIEW_LEN])
        } else {
            chunk.content.clone()
        };
        Self {
            lsp_location: LspLocation {
                file_path: chunk.file_path.clone(),
                start_line: chunk.start_line.saturating_sub(1),
                start_character: 0,
                end_line: chunk.end_line.saturating_sub(1),
                end_character: 0,
            },
            symbol_name: chunk.name.clone(),
            similarity,
            preview,
        }
    }
}
```

- [ ] **Step 2: Verify**

Run: `cargo check -p ripvec-mcp`

- [ ] **Step 3: Commit**

```
feat(mcp): add LSP-compatible result types
```

---

## Task 4: Server state and startup (server.rs + main.rs)

**Files:**
- Create: `crates/ripvec-mcp/src/server.rs`
- Rewrite: `crates/ripvec-mcp/src/main.rs`
- Modify: `crates/ripvec-mcp/Cargo.toml`

- [ ] **Step 1: Add serde_json and ndarray deps to Cargo.toml**

Add `serde_json.workspace = true` and ensure `ndarray` is available via `ripvec-core`.

- [ ] **Step 2: Create server.rs with RipvecServer struct**

```rust
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use tokio::sync::{OnceCell, RwLock};
use ripvec_core::index::SearchIndex;
use ripvec_core::chunk::CodeChunk;
use ripvec_core::backend::EmbedBackend;

pub struct RipvecServer {
    pub index: Arc<RwLock<Option<SearchIndex>>>,
    pub chunks: Arc<RwLock<Vec<CodeChunk>>>,
    pub code_backend: OnceCell<Arc<dyn EmbedBackend>>,
    pub text_backend: OnceCell<Arc<dyn EmbedBackend>>,
    pub code_tokenizer: OnceCell<Arc<tokenizers::Tokenizer>>,
    pub text_tokenizer: OnceCell<Arc<tokenizers::Tokenizer>>,
    pub project_root: PathBuf,
    pub indexing: Arc<AtomicBool>,
    pub tool_router: rmcp::handler::server::tool::ToolRouter<Self>,
}
```

Include `Clone` impl (clone the Arcs), `ServerHandler` impl (delegate to tool_router), `get_info` with updated instructions listing all 5 tools.

- [ ] **Step 3: Create background indexing function**

```rust
pub async fn spawn_index(server: &RipvecServer) {
    // Set indexing flag
    // spawn_blocking: load text backend + tokenizer, call embed_all
    // Build SearchIndex from chunks + embeddings
    // Store in server.index and server.chunks under write lock
    // Clear indexing flag
}
```

- [ ] **Step 4: Rewrite main.rs**

Read `RIPVEC_ROOT` and `RIPVEC_MODEL` from env. Create `RipvecServer`. Spawn background index task. Start stdio MCP transport.

- [ ] **Step 5: Verify**

Run: `cargo check -p ripvec-mcp`

- [ ] **Step 6: Commit**

```
feat(mcp): server state with background indexing and env config
```

---

## Task 5: Tool handlers (tools.rs)

**Files:**
- Create: `crates/ripvec-mcp/src/tools.rs`
- Modify: `crates/ripvec-mcp/src/server.rs` (wire tool_router)

- [ ] **Step 1: Create request types**

```rust
#[derive(Deserialize, JsonSchema)]
pub struct SearchRequest {
    /// Natural language query.
    pub query: String,
    /// Max results (default 10).
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    /// Min similarity 0.0-1.0 (default 0.3).
    #[serde(default = "default_threshold")]
    pub threshold: f32,
}

#[derive(Deserialize, JsonSchema)]
pub struct FindSimilarRequest {
    /// Absolute path to the file.
    pub file_path: String,
    /// 0-based line number (LSP-compatible).
    pub line: usize,
    /// Max results (default 10).
    #[serde(default = "default_top_k")]
    pub top_k: usize,
}
```

- [ ] **Step 2: Implement helper — wait for index**

```rust
async fn wait_for_index(server: &RipvecServer) -> Result<(), rmcp::ErrorData> {
    // Spin-wait (with tokio::time::sleep) until server.indexing is false
    // Return error if index is still None after indexing completes
}
```

- [ ] **Step 3: Implement helper — search with backend**

```rust
async fn do_search(
    server: &RipvecServer,
    query: &str,
    top_k: usize,
    threshold: f32,
    use_code_model: bool,
) -> Result<SearchResponse, rmcp::ErrorData> {
    // wait_for_index
    // Lazy-load the appropriate backend + tokenizer via OnceCell
    // If code model, prepend "Represent this query for searching relevant code: "
    // tokenize_query, embed_batch, then rank against index
    // Convert to SearchResultItems
}
```

- [ ] **Step 4: Implement search_code tool**

Calls `do_search` with `use_code_model: true`.

- [ ] **Step 5: Implement search_text tool**

Calls `do_search` with `use_code_model: false`.

- [ ] **Step 6: Implement find_similar tool**

```rust
// wait_for_index
// Read chunks + index under read lock
// Find chunk whose file_path matches and start_line <= line <= end_line
// Get that chunk's embedding row from the index
// Rank all other chunks against it
// Return top_k as SearchResultItems
```

- [ ] **Step 7: Implement reindex tool**

```rust
// Set indexing flag, cancel any in-progress index
// Clear current index + chunks
// Spawn background task (same as startup)
// Return immediately with "reindexing started" or block until done
```

- [ ] **Step 8: Implement index_status tool**

```rust
// Read index + chunks under read lock
// Return JSON: ready, indexing, chunks count, file count, extensions breakdown, project_root
```

- [ ] **Step 9: Wire tool_router in server.rs**

Add `#[tool_router]` impl block with all 5 tools.

- [ ] **Step 10: Verify**

Run: `cargo check -p ripvec-mcp`
Run: `cargo clippy --all-targets -- -D warnings`

- [ ] **Step 11: Commit**

```
feat(mcp): implement all 5 tools — search_code, search_text, find_similar, reindex, index_status
```

---

## Task 6: Integration test

**Files:**
- Create: `crates/ripvec-mcp/tests/integration.rs`

- [ ] **Step 1: Write a smoke test**

Build the server with a small fixture directory, call `index_status` to verify indexing completes, call `search_text` with a query, verify the result has `lsp_location` with 0-based lines.

- [ ] **Step 2: Verify**

Run: `cargo test -p ripvec-mcp`

- [ ] **Step 3: Commit**

```
test(mcp): integration smoke test for search and index_status
```

---

## Task 7: Final verification and cleanup

- [ ] **Step 1: Full workspace check**

Run: `cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test --workspace`

- [ ] **Step 2: Manual test with Claude Code**

Add to `.mcp.json`:
```json
{
  "mcpServers": {
    "ripvec": {
      "command": "cargo",
      "args": ["run", "-p", "ripvec-mcp"],
      "env": { "RIPVEC_ROOT": "." }
    }
  }
}
```

Verify tools appear in Claude Code, run a search, check `index_status`.

- [ ] **Step 3: Commit and push**

```
feat(mcp): ripvec MCP server with 5 tools and LSP-compatible results
```
