# ripvec LSP Server

**Date**: 2026-04-07
**Status**: Approved

## Problem

Claude Code's LSP tool provides code intelligence (go to definition, find references, diagnostics, symbols) but requires per-language LSP plugins. Many languages ripvec supports (bash, HCL, TOML, Ruby, Kotlin, Swift, Scala) either have no LSP plugin in the Claude Code marketplace or require installing separate language server binaries. Additionally, no existing LSP provides cross-language semantic search — `workspace/symbol` on a Rust LSP can't find a similar Python implementation.

## Solution

Add `--lsp` mode to `ripvec-mcp` so it serves the Language Server Protocol over stdio alongside its existing MCP mode. One binary, one index, selected by a command-line flag. The ripvec plugin's `.lsp.json` configures Claude Code to use it for all 21 supported languages.

## Architecture

`ripvec-mcp --lsp` starts in LSP mode (tower-lsp over stdio). It shares the same `HybridIndex`, `RepoGraph`, and tree-sitter infrastructure from `ripvec-core` as the MCP path. No new crate — the LSP adapter lives in the existing `ripvec-mcp` crate behind the `--lsp` flag.

On startup:
1. Initialize tower-lsp server over stdio
2. Build/load `HybridIndex` from cache (same as MCP startup)
3. Build `RepoGraph` for PageRank
4. Start file watcher for incremental updates
5. Advertise capabilities for the 7 implemented operations

## v1 Operations

### workspaceSymbol

**Backend**: Hybrid search (semantic + BM25 + PageRank boost)

The query string from `workspace/symbol` is run through the full search pipeline. Results are returned as `SymbolInformation` with file path, line range, symbol kind (function, class, etc.), and container name from the scope chain.

This is the killer feature — cross-language semantic symbol search. "retry logic" finds implementations in Rust, Python, and Go simultaneously.

### documentSymbol

**Backend**: Tree-sitter chunk extraction

Parse the file with tree-sitter, extract all definition captures (`@def` + `@name`), return as `DocumentSymbol[]` with name, kind, range, and selection range. Works for all 21 languages. No embedding needed — pure structural extraction.

### goToDefinition

**Backend**: BM25 identifier match + PageRank boost

Extract the identifier at the cursor position (tree-sitter node text). Search the index for chunks whose `name` field matches (BM25 keyword mode for precision), boosted by PageRank. Return the top match's location.

For import statements, use the import resolution from `repo_map.rs` if available for that language.

### goToImplementation

**Backend**: Same as goToDefinition

The LSP spec distinguishes interface declarations from concrete implementations, but ripvec's similarity-based approach naturally finds implementations. Wire to the same backend.

### findReferences

**Backend**: Semantic similarity + PageRank callers

Embed the chunk at the cursor position, search for semantically similar chunks across the index. Also include files from the PageRank caller graph for the file containing the cursor. Return as `Location[]`.

### hover

**Backend**: Tree-sitter scope chain + enriched content

Return the `enriched_content` of the chunk under the cursor — includes scope chain, signature, and file path context. Format as Markdown in the hover response.

### textDocument/publishDiagnostics

**Backend**: Tree-sitter error node detection

After `textDocument/didChange` or `textDocument/didOpen`, re-parse the file with tree-sitter and walk the AST for `ERROR` and `MISSING` nodes. Report each as a diagnostic with severity Warning. This provides syntax checking for all 21 languages — particularly valuable for bash, HCL, TOML, and other languages without dedicated LSPs.

## Plugin Configuration

### .lsp.json

```json
{
  "ripvec": {
    "command": "${CLAUDE_PLUGIN_ROOT}/bin/ripvec-mcp",
    "args": ["--lsp"],
    "extensionToLanguage": {
      ".rs": "rust",
      ".py": "python",
      ".js": "javascript",
      ".jsx": "javascript",
      ".ts": "typescript",
      ".tsx": "typescript",
      ".go": "go",
      ".java": "java",
      ".c": "c",
      ".h": "c",
      ".cpp": "cpp",
      ".cc": "cpp",
      ".cxx": "cpp",
      ".hpp": "cpp",
      ".sh": "bash",
      ".bash": "bash",
      ".bats": "bash",
      ".rb": "ruby",
      ".tf": "hcl",
      ".tfvars": "hcl",
      ".hcl": "hcl",
      ".kt": "kotlin",
      ".kts": "kotlin",
      ".swift": "swift",
      ".scala": "scala",
      ".toml": "toml"
    }
  }
}
```

The `ensure-ripvec-mcp.sh` wrapper handles binary auto-download for both MCP and LSP modes (same binary).

### Coexistence with dedicated LSPs

For languages that have dedicated LSP plugins (Rust via rust-analyzer, Go via gopls, Python via pyright, TypeScript via typescript-language-server), Claude Code may have multiple LSP servers registered for the same file extension. The dedicated LSP provides type-aware features; ripvec provides semantic search features. Claude Code's LSP infrastructure handles routing.

For languages WITHOUT dedicated LSPs (bash, HCL, TOML, Ruby, Kotlin, Swift, Scala), ripvec is the only LSP — providing symbol outline, basic navigation, and syntax diagnostics.

## Implementation

### Dependencies

- `tower-lsp` — Rust LSP server framework (async, tower-based)
- `lsp-types` — LSP protocol type definitions (comes with tower-lsp)

### Code organization

All LSP code lives in `crates/ripvec-mcp/src/lsp/`:
- `mod.rs` — LSP server struct, initialization, capability advertisement
- `symbols.rs` — workspaceSymbol + documentSymbol handlers
- `navigation.rs` — goToDefinition, goToImplementation, findReferences
- `hover.rs` — hover handler
- `diagnostics.rs` — tree-sitter error node detection + publishDiagnostics

### Shared state

The LSP server holds `Arc` references to the same types the MCP server uses:
- `Arc<RwLock<Option<HybridIndex>>>` — search index
- `Arc<RwLock<Option<RepoGraph>>>` — PageRank graph
- `Arc<IndexProgress>` — progress state

The `--lsp` flag in `main.rs` selects which protocol adapter to run. Both paths call the same `run_background_index()` and `run_file_watcher()`.

### File synchronization

The LSP protocol sends `textDocument/didOpen`, `didChange`, `didClose` notifications. For diagnostics, we re-parse the file content from the notification (not from disk) to catch unsaved changes. For search operations, we use the persisted index (disk-synced via the file watcher).

## v2: Function-Level PageRank + Call Hierarchy

The current `RepoGraph` computes PageRank per file. v2 extends this to per-function (tree-sitter definition node) resolution:

- **Function-level PageRank**: Build edges between individual function definitions (not just files) based on call site resolution. Weight by incoming/outgoing call count.
- **Improved embedding ranking**: Use function-level PageRank as the boost signal instead of file-level. A highly-called utility function ranks higher than an unused function in the same file.
- **Call hierarchy operations**: `prepareCallHierarchy`, `incomingCalls`, `outgoingCalls` — backed by the function-level graph. These return the actual caller/callee functions, not just files.

This is a deeper architectural change to `repo_map.rs` and benefits both LSP and MCP search quality.

## Scope

### v1 (this spec)
1. Add `tower-lsp` dependency to ripvec-mcp
2. Add `--lsp` flag to ripvec-mcp binary
3. Implement 7 LSP operations (workspaceSymbol, documentSymbol, goToDefinition, goToImplementation, findReferences, hover, publishDiagnostics)
4. Add `.lsp.json` to the ripvec plugin
5. Shared index between MCP and LSP modes

### v2 (future)
- Function-level PageRank in repo_map.rs
- Call hierarchy operations (prepareCallHierarchy, incomingCalls, outgoingCalls)
- Function-level boost for embedding ranking (improves MCP search too)
