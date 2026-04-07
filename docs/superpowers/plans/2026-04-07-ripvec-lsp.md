# ripvec LSP Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--lsp` mode to `ripvec-mcp` so it serves Language Server Protocol, giving Claude Code semantic code intelligence across all 21 supported languages.

**Architecture:** `--lsp` flag selects LSP mode (tower-lsp-server over stdio) instead of MCP mode (rmcp). Both share the same `HybridIndex`, `RepoGraph`, and file watcher from `ripvec-core`. LSP code lives in `crates/ripvec-mcp/src/lsp/` module.

**Tech Stack:** tower-lsp-server 0.23 (maintained fork of tower-lsp), lsp-types, tree-sitter (for diagnostics)

---

### Task 1: Add tower-lsp-server dependency and LSP module skeleton

**Files:**
- Modify: `Cargo.toml` (workspace) — add tower-lsp-server, lsp-types
- Modify: `crates/ripvec-mcp/Cargo.toml` — add deps
- Create: `crates/ripvec-mcp/src/lsp/mod.rs` — LSP server struct + LanguageServer trait stub
- Modify: `crates/ripvec-mcp/src/main.rs` — add `--lsp` flag, dispatch to LSP or MCP

- [ ] **Step 1: Add dependencies to workspace Cargo.toml**

Add to `[workspace.dependencies]`:

```toml
tower-lsp-server = "0.23"
```

- [ ] **Step 2: Add to ripvec-mcp Cargo.toml**

Add under `[dependencies]`:

```toml
tower-lsp-server.workspace = true
```

- [ ] **Step 3: Create LSP module skeleton**

Create `crates/ripvec-mcp/src/lsp/mod.rs`:

```rust
//! Language Server Protocol adapter for ripvec.
//!
//! Provides semantic code intelligence (symbol search, go-to-definition,
//! diagnostics) across all 21 tree-sitter-supported languages.

pub mod diagnostics;
pub mod hover;
pub mod navigation;
pub mod symbols;

use std::path::PathBuf;
use std::sync::Arc;

use tower_lsp_server::jsonrpc::Result;
use tower_lsp_server::lsp_types::*;
use tower_lsp_server::{Client, LanguageServer, LspService, Server};

use crate::server::{IndexProgress, RipvecServer};

/// The ripvec LSP server, backed by the same index as the MCP server.
pub struct RipvecLsp {
    /// LSP client for sending notifications (diagnostics, etc.).
    client: Client,
    /// Shared search index.
    index: Arc<tokio::sync::RwLock<Option<ripvec_core::hybrid::HybridIndex>>>,
    /// Shared PageRank graph.
    repo_graph: Arc<std::sync::RwLock<Option<ripvec_core::repo_map::RepoGraph>>>,
    /// Project root.
    root: PathBuf,
    /// Indexing progress.
    progress: Arc<IndexProgress>,
}

impl RipvecLsp {
    /// Create a new LSP server sharing state with the given MCP server.
    pub fn new(
        client: Client,
        mcp: &RipvecServer,
    ) -> Self {
        Self {
            client,
            index: Arc::clone(&mcp.index),
            repo_graph: Arc::clone(&mcp.repo_graph),
            root: mcp.project_root.clone(),
            progress: Arc::clone(&mcp.progress),
        }
    }
}

#[tower_lsp_server::async_trait]
impl LanguageServer for RipvecLsp {
    async fn initialize(&self, _: InitializeParams) -> Result<InitializeResult> {
        Ok(InitializeResult {
            capabilities: ServerCapabilities {
                text_document_sync: Some(TextDocumentSyncCapability::Kind(
                    TextDocumentSyncKind::FULL,
                )),
                document_symbol_provider: Some(OneOf::Left(true)),
                workspace_symbol_provider: Some(OneOf::Left(true)),
                definition_provider: Some(OneOf::Left(true)),
                implementation_provider: Some(ImplementationProviderCapability::Simple(true)),
                references_provider: Some(OneOf::Left(true)),
                hover_provider: Some(HoverProviderCapability::Simple(true)),
                ..Default::default()
            },
            ..Default::default()
        })
    }

    async fn initialized(&self, _: InitializedParams) {
        self.client
            .log_message(MessageType::INFO, format!(
                "ripvec-lsp {} initialized (21 languages, semantic search + PageRank)",
                env!("CARGO_PKG_VERSION")
            ))
            .await;
    }

    async fn shutdown(&self) -> Result<()> {
        Ok(())
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        diagnostics::publish(&self.client, &params.text_document.uri, &params.text_document.text).await;
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        if let Some(change) = params.content_changes.into_iter().last() {
            diagnostics::publish(&self.client, &params.text_document.uri, &change.text).await;
        }
    }

    async fn document_symbol(
        &self,
        params: DocumentSymbolParams,
    ) -> Result<Option<DocumentSymbolResponse>> {
        symbols::document_symbol(&self.root, &params.text_document.uri).await
    }

    async fn symbol(
        &self,
        params: WorkspaceSymbolParams,
    ) -> Result<Option<Vec<SymbolInformation>>> {
        symbols::workspace_symbol(&params.query, &self.index, &self.repo_graph, &self.root).await
    }

    async fn goto_definition(
        &self,
        params: GotoDefinitionParams,
    ) -> Result<Option<GotoDefinitionResponse>> {
        navigation::goto_definition(params, &self.index, &self.repo_graph, &self.root).await
    }

    async fn goto_implementation(
        &self,
        params: GotoImplementationParams,
    ) -> Result<Option<GotoImplementationResponse>> {
        // Same backend as goto_definition
        let def_params = GotoDefinitionParams {
            text_document_position_params: params.text_document_position_params,
            work_done_progress_params: params.work_done_progress_params,
            partial_result_params: params.partial_result_params,
        };
        navigation::goto_definition(def_params, &self.index, &self.repo_graph, &self.root).await
    }

    async fn references(
        &self,
        params: ReferenceParams,
    ) -> Result<Option<Vec<Location>>> {
        navigation::find_references(params, &self.index, &self.repo_graph, &self.root).await
    }

    async fn hover(
        &self,
        params: HoverParams,
    ) -> Result<Option<Hover>> {
        hover::hover(params, &self.index, &self.root).await
    }
}

/// Start the LSP server over stdio.
pub async fn run(mcp_server: RipvecServer) -> anyhow::Result<()> {
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();

    let (service, socket) = LspService::new(|client| RipvecLsp::new(client, &mcp_server));
    Server::new(stdin, stdout, socket).serve(service).await;
    Ok(())
}
```

- [ ] **Step 4: Create stub submodules**

Create four stub files that compile but return empty results:

`crates/ripvec-mcp/src/lsp/symbols.rs`:
```rust
//! Symbol providers: documentSymbol and workspaceSymbol.

use tower_lsp_server::jsonrpc::Result;
use tower_lsp_server::lsp_types::*;

pub async fn document_symbol(
    _root: &std::path::Path,
    _uri: &Url,
) -> Result<Option<DocumentSymbolResponse>> {
    Ok(None) // TODO: implement in Task 3
}

pub async fn workspace_symbol(
    _query: &str,
    _index: &tokio::sync::RwLock<Option<ripvec_core::hybrid::HybridIndex>>,
    _repo_graph: &std::sync::RwLock<Option<ripvec_core::repo_map::RepoGraph>>,
    _root: &std::path::Path,
) -> Result<Option<Vec<SymbolInformation>>> {
    Ok(None) // TODO: implement in Task 4
}
```

`crates/ripvec-mcp/src/lsp/navigation.rs`:
```rust
//! Navigation providers: goToDefinition, findReferences.

use tower_lsp_server::jsonrpc::Result;
use tower_lsp_server::lsp_types::*;

pub async fn goto_definition(
    _params: GotoDefinitionParams,
    _index: &tokio::sync::RwLock<Option<ripvec_core::hybrid::HybridIndex>>,
    _repo_graph: &std::sync::RwLock<Option<ripvec_core::repo_map::RepoGraph>>,
    _root: &std::path::Path,
) -> Result<Option<GotoDefinitionResponse>> {
    Ok(None) // TODO: implement in Task 5
}

pub async fn find_references(
    _params: ReferenceParams,
    _index: &tokio::sync::RwLock<Option<ripvec_core::hybrid::HybridIndex>>,
    _repo_graph: &std::sync::RwLock<Option<ripvec_core::repo_map::RepoGraph>>,
    _root: &std::path::Path,
) -> Result<Option<Vec<Location>>> {
    Ok(None) // TODO: implement in Task 6
}
```

`crates/ripvec-mcp/src/lsp/hover.rs`:
```rust
//! Hover provider.

use tower_lsp_server::jsonrpc::Result;
use tower_lsp_server::lsp_types::*;

pub async fn hover(
    _params: HoverParams,
    _index: &tokio::sync::RwLock<Option<ripvec_core::hybrid::HybridIndex>>,
    _root: &std::path::Path,
) -> Result<Option<Hover>> {
    Ok(None) // TODO: implement in Task 7
}
```

`crates/ripvec-mcp/src/lsp/diagnostics.rs`:
```rust
//! Tree-sitter syntax diagnostics.

use tower_lsp_server::Client;
use tower_lsp_server::lsp_types::*;

pub async fn publish(_client: &Client, _uri: &Url, _text: &str) {
    // TODO: implement in Task 8
}
```

- [ ] **Step 5: Add `--lsp` flag to main.rs**

Replace the current `main.rs` with:

```rust
//! MCP and LSP server binary for ripvec semantic search.
//!
//! By default serves MCP over stdio. With `--lsp`, serves LSP instead.
//! Both modes share the same search index and PageRank graph.

mod lsp;
mod result;
mod server;
mod tools;

use rmcp::ServiceExt;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    // Handle --version before anything else
    if args.iter().any(|a| a == "--version" || a == "-V") {
        println!("ripvec-mcp {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }

    let lsp_mode = args.iter().any(|a| a == "--lsp");

    // Initialize tracing to stderr (both protocols use stdin/stdout for transport)
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();

    let root = std::env::var("RIPVEC_ROOT").map_or_else(
        |_| std::env::current_dir().expect("cannot determine current directory"),
        std::path::PathBuf::from,
    );

    let mode_str = if lsp_mode { "LSP" } else { "MCP" };
    eprintln!("[ripvec-mcp] {mode_str} mode, project root: {}", root.display());

    let server = server::RipvecServer::new(root);

    // Spawn background indexing (non-blocking)
    let bg_server = server.clone();
    tokio::spawn(async move {
        server::run_background_index(&bg_server, false).await;
    });

    // Spawn debounced file watcher
    let watcher_server = server.clone();
    tokio::spawn(async move {
        server::run_file_watcher(&watcher_server).await;
    });

    if lsp_mode {
        lsp::run(server).await
    } else {
        let service = server
            .serve(rmcp::transport::stdio())
            .await
            .map_err(|e| anyhow::anyhow!("MCP serve error: {e}"))?;
        service.waiting().await?;
        Ok(())
    }
}
```

- [ ] **Step 6: Verify it compiles**

Run: `cargo check -p ripvec-mcp`

NOTE: The `tower-lsp-server` crate might use different type names than expected (e.g., `ls_types` vs `lsp_types`). Check the actual exports after adding the dependency and adjust imports. Version 0.23.0 uses `ls-types` instead of `lsp-types` — you may need to adjust all `lsp_types::*` imports to `ls_types::*` or whatever the re-export is named.

- [ ] **Step 7: Verify --lsp flag is recognized**

Run: `cargo run -p ripvec-mcp -- --version`
Expected: prints version

Run: `echo '{}' | timeout 2 cargo run -p ripvec-mcp -- --lsp 2>/dev/null || true`
Expected: exits without panic (the LSP server will fail to initialize from `{}` but shouldn't crash)

- [ ] **Step 8: Commit**

```bash
git add Cargo.toml Cargo.lock crates/ripvec-mcp/Cargo.toml crates/ripvec-mcp/src/main.rs crates/ripvec-mcp/src/lsp/
git commit -m "feat: add --lsp mode skeleton with tower-lsp-server"
```

---

### Task 2: Make RipvecServer fields accessible to LSP

**Files:**
- Modify: `crates/ripvec-mcp/src/server.rs` — ensure fields needed by LSP are accessible

- [ ] **Step 1: Verify RipvecServer fields are pub**

Read `crates/ripvec-mcp/src/server.rs` and check that these fields are `pub`:
- `index: Arc<RwLock<Option<HybridIndex>>>`
- `repo_graph: Arc<RwLock<Option<RepoGraph>>>`
- `project_root: PathBuf`
- `progress: Arc<IndexProgress>`

They should already be pub (used by tools.rs). If any aren't, make them pub.

- [ ] **Step 2: Verify the LSP module can construct RipvecLsp from RipvecServer**

Run: `cargo check -p ripvec-mcp`
Expected: compiles (the `RipvecLsp::new` in Task 1 accesses these fields)

- [ ] **Step 3: Commit (if changes needed)**

```bash
git add crates/ripvec-mcp/src/server.rs
git commit -m "refactor: ensure server fields are accessible to LSP module"
```

---

### Task 3: Implement documentSymbol

**Files:**
- Modify: `crates/ripvec-mcp/src/lsp/symbols.rs` — implement `document_symbol`

- [ ] **Step 1: Implement document_symbol**

Replace the stub in `crates/ripvec-mcp/src/lsp/symbols.rs` `document_symbol` function:

```rust
pub async fn document_symbol(
    root: &std::path::Path,
    uri: &Url,
) -> Result<Option<DocumentSymbolResponse>> {
    let file_path = uri
        .to_file_path()
        .map_err(|()| tower_lsp_server::jsonrpc::Error::invalid_params("invalid file URI"))?;

    let source = match std::fs::read_to_string(&file_path) {
        Ok(s) => s,
        Err(_) => return Ok(None),
    };

    let ext = file_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    let Some(config) = ripvec_core::languages::config_for_extension(ext) else {
        return Ok(None);
    };

    // Parse and extract definitions using tree-sitter
    let chunks = ripvec_core::chunk::chunk_file(
        &file_path,
        &source,
        &config,
        &ripvec_core::chunk::ChunkConfig::default(),
    );

    let symbols: Vec<DocumentSymbol> = chunks
        .iter()
        .filter(|c| c.kind != "window" && c.kind != "file")
        .map(|chunk| {
            let kind = match chunk.kind.as_str() {
                "function_item" | "function_definition" | "function_declaration"
                | "method_definition" | "method" => SymbolKind::FUNCTION,
                "struct_item" | "class_definition" | "class_declaration"
                | "class_specifier" | "class" => SymbolKind::CLASS,
                "enum_item" => SymbolKind::ENUM,
                "type_item" | "type_alias_declaration" => SymbolKind::TYPE_PARAMETER,
                "interface_declaration" | "trait_definition" | "protocol_declaration" => {
                    SymbolKind::INTERFACE
                }
                "module" | "object_definition" | "object_declaration" => SymbolKind::MODULE,
                "block" => SymbolKind::NAMESPACE, // HCL blocks
                _ => SymbolKind::VARIABLE,
            };

            let range = Range {
                start: Position {
                    line: (chunk.start_line - 1) as u32,
                    character: 0,
                },
                end: Position {
                    line: (chunk.end_line - 1) as u32,
                    character: 0,
                },
            };

            #[expect(deprecated, reason = "DocumentSymbol requires deprecated field")]
            DocumentSymbol {
                name: chunk.name.clone(),
                detail: Some(chunk.kind.clone()),
                kind,
                tags: None,
                deprecated: None,
                range,
                selection_range: range,
                children: None,
            }
        })
        .collect();

    if symbols.is_empty() {
        Ok(None)
    } else {
        Ok(Some(DocumentSymbolResponse::Nested(symbols)))
    }
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo check -p ripvec-mcp`

- [ ] **Step 3: Commit**

```bash
git add crates/ripvec-mcp/src/lsp/symbols.rs
git commit -m "feat(lsp): implement documentSymbol via tree-sitter chunks"
```

---

### Task 4: Implement workspaceSymbol

**Files:**
- Modify: `crates/ripvec-mcp/src/lsp/symbols.rs` — implement `workspace_symbol`

- [ ] **Step 1: Implement workspace_symbol**

Replace the stub `workspace_symbol` function:

```rust
pub async fn workspace_symbol(
    query: &str,
    index: &tokio::sync::RwLock<Option<ripvec_core::hybrid::HybridIndex>>,
    repo_graph: &std::sync::RwLock<Option<ripvec_core::repo_map::RepoGraph>>,
    root: &std::path::Path,
) -> Result<Option<Vec<SymbolInformation>>> {
    if query.is_empty() {
        return Ok(None);
    }

    let idx_guard = index.read().await;
    let Some(hybrid) = idx_guard.as_ref() else {
        return Ok(None);
    };

    // We need a backend + tokenizer for query embedding.
    // For workspace symbol, fall back to BM25 keyword search (no embedding needed).
    let mut results = hybrid.search(
        &[], // empty embedding — BM25 only via keyword mode
        query,
        20,
        0.0,
        ripvec_core::hybrid::SearchMode::Keyword,
    );

    // Apply PageRank boost if available
    if let Ok(rg) = repo_graph.read() {
        if let Some(graph) = rg.as_ref() {
            let pr = ripvec_core::hybrid::pagerank_lookup(graph);
            ripvec_core::hybrid::boost_with_pagerank(
                &mut results,
                hybrid.chunks(),
                &pr,
                graph.alpha,
            );
        }
    }

    let symbols: Vec<SymbolInformation> = results
        .into_iter()
        .filter_map(|(idx, _score)| {
            let chunk = hybrid.chunks().get(idx)?;
            let path = std::path::Path::new(&chunk.file_path);
            let uri = Url::from_file_path(if path.is_absolute() {
                path.to_path_buf()
            } else {
                root.join(path)
            })
            .ok()?;

            let kind = match chunk.kind.as_str() {
                k if k.contains("function") || k.contains("method") => SymbolKind::FUNCTION,
                k if k.contains("class") || k.contains("struct") => SymbolKind::CLASS,
                k if k.contains("enum") => SymbolKind::ENUM,
                k if k.contains("interface") || k.contains("trait") || k.contains("protocol") => {
                    SymbolKind::INTERFACE
                }
                k if k.contains("module") || k.contains("object") => SymbolKind::MODULE,
                _ => SymbolKind::VARIABLE,
            };

            #[expect(deprecated, reason = "SymbolInformation requires deprecated field")]
            Some(SymbolInformation {
                name: chunk.name.clone(),
                kind,
                tags: None,
                deprecated: None,
                location: Location {
                    uri,
                    range: Range {
                        start: Position {
                            line: (chunk.start_line - 1) as u32,
                            character: 0,
                        },
                        end: Position {
                            line: (chunk.end_line - 1) as u32,
                            character: 0,
                        },
                    },
                },
                container_name: None,
            })
        })
        .collect();

    if symbols.is_empty() {
        Ok(None)
    } else {
        Ok(Some(symbols))
    }
}
```

NOTE: `workspace/symbol` uses BM25 keyword search (not semantic) because we don't have an embedding backend loaded in LSP mode. The query is a symbol name, not a natural language question — BM25 is appropriate. Semantic search can be added later when embedding infrastructure is shared.

- [ ] **Step 2: Verify it compiles**

Run: `cargo check -p ripvec-mcp`

- [ ] **Step 3: Commit**

```bash
git add crates/ripvec-mcp/src/lsp/symbols.rs
git commit -m "feat(lsp): implement workspaceSymbol with BM25 + PageRank boost"
```

---

### Task 5: Implement goToDefinition

**Files:**
- Modify: `crates/ripvec-mcp/src/lsp/navigation.rs` — implement `goto_definition`

- [ ] **Step 1: Add helper to extract word at position**

Add to `navigation.rs`:

```rust
/// Extract the word (identifier) at the given line/character position.
fn word_at_position(source: &str, line: u32, character: u32) -> Option<String> {
    let target_line = source.lines().nth(line as usize)?;
    let col = character as usize;
    if col > target_line.len() {
        return None;
    }

    // Expand outward from cursor position to find the word boundary
    let bytes = target_line.as_bytes();
    let is_ident = |b: u8| b.is_ascii_alphanumeric() || b == b'_';

    let mut start = col;
    while start > 0 && is_ident(bytes[start - 1]) {
        start -= 1;
    }
    let mut end = col;
    while end < bytes.len() && is_ident(bytes[end]) {
        end += 1;
    }

    if start == end {
        return None;
    }
    Some(target_line[start..end].to_string())
}
```

- [ ] **Step 2: Implement goto_definition**

Replace the stub:

```rust
pub async fn goto_definition(
    params: GotoDefinitionParams,
    index: &tokio::sync::RwLock<Option<ripvec_core::hybrid::HybridIndex>>,
    repo_graph: &std::sync::RwLock<Option<ripvec_core::repo_map::RepoGraph>>,
    root: &std::path::Path,
) -> Result<Option<GotoDefinitionResponse>> {
    let pos = &params.text_document_position_params;
    let file_path = pos
        .text_document
        .uri
        .to_file_path()
        .map_err(|()| tower_lsp_server::jsonrpc::Error::invalid_params("invalid file URI"))?;

    let source = match std::fs::read_to_string(&file_path) {
        Ok(s) => s,
        Err(_) => return Ok(None),
    };

    let word = match word_at_position(&source, pos.position.line, pos.position.character) {
        Some(w) => w,
        None => return Ok(None),
    };

    let idx_guard = index.read().await;
    let Some(hybrid) = idx_guard.as_ref() else {
        return Ok(None);
    };

    // Search for the identifier using BM25 keyword matching
    let mut results = hybrid.search(
        &[],
        &word,
        10,
        0.0,
        ripvec_core::hybrid::SearchMode::Keyword,
    );

    // Apply PageRank boost
    if let Ok(rg) = repo_graph.read() {
        if let Some(graph) = rg.as_ref() {
            let pr = ripvec_core::hybrid::pagerank_lookup(graph);
            ripvec_core::hybrid::boost_with_pagerank(
                &mut results,
                hybrid.chunks(),
                &pr,
                graph.alpha,
            );
        }
    }

    // Find the best match where the chunk name matches the word
    let best = results
        .iter()
        .filter_map(|&(idx, _)| {
            let chunk = hybrid.chunks().get(idx)?;
            // Prefer exact name matches
            if chunk.name == word {
                let path = std::path::Path::new(&chunk.file_path);
                let abs = if path.is_absolute() {
                    path.to_path_buf()
                } else {
                    root.join(path)
                };
                let uri = Url::from_file_path(abs).ok()?;
                Some(Location {
                    uri,
                    range: Range {
                        start: Position {
                            line: (chunk.start_line - 1) as u32,
                            character: 0,
                        },
                        end: Position {
                            line: (chunk.end_line - 1) as u32,
                            character: 0,
                        },
                    },
                })
            } else {
                None
            }
        })
        .next();

    Ok(best.map(GotoDefinitionResponse::Scalar))
}
```

- [ ] **Step 3: Verify it compiles**

Run: `cargo check -p ripvec-mcp`

- [ ] **Step 4: Commit**

```bash
git add crates/ripvec-mcp/src/lsp/navigation.rs
git commit -m "feat(lsp): implement goToDefinition with BM25 + PageRank"
```

---

### Task 6: Implement findReferences

**Files:**
- Modify: `crates/ripvec-mcp/src/lsp/navigation.rs` — implement `find_references`

- [ ] **Step 1: Implement find_references**

Replace the stub:

```rust
pub async fn find_references(
    params: ReferenceParams,
    index: &tokio::sync::RwLock<Option<ripvec_core::hybrid::HybridIndex>>,
    repo_graph: &std::sync::RwLock<Option<ripvec_core::repo_map::RepoGraph>>,
    root: &std::path::Path,
) -> Result<Option<Vec<Location>>> {
    let pos = &params.text_document_position;
    let file_path = pos
        .text_document
        .uri
        .to_file_path()
        .map_err(|()| tower_lsp_server::jsonrpc::Error::invalid_params("invalid file URI"))?;

    let source = match std::fs::read_to_string(&file_path) {
        Ok(s) => s,
        Err(_) => return Ok(None),
    };

    let word = match word_at_position(&source, pos.position.line, pos.position.character) {
        Some(w) => w,
        None => return Ok(None),
    };

    let idx_guard = index.read().await;
    let Some(hybrid) = idx_guard.as_ref() else {
        return Ok(None);
    };

    // Search for references using BM25 keyword matching
    let mut results = hybrid.search(
        &[],
        &word,
        30,
        0.0,
        ripvec_core::hybrid::SearchMode::Keyword,
    );

    // Apply PageRank boost
    if let Ok(rg) = repo_graph.read() {
        if let Some(graph) = rg.as_ref() {
            let pr = ripvec_core::hybrid::pagerank_lookup(graph);
            ripvec_core::hybrid::boost_with_pagerank(
                &mut results,
                hybrid.chunks(),
                &pr,
                graph.alpha,
            );
        }
    }

    let locations: Vec<Location> = results
        .into_iter()
        .filter_map(|(idx, _)| {
            let chunk = hybrid.chunks().get(idx)?;
            // Include chunks that contain the word (name match or content match)
            if !chunk.name.contains(&word) && !chunk.content.contains(&word) {
                return None;
            }
            let path = std::path::Path::new(&chunk.file_path);
            let abs = if path.is_absolute() {
                path.to_path_buf()
            } else {
                root.join(path)
            };
            let uri = Url::from_file_path(abs).ok()?;
            Some(Location {
                uri,
                range: Range {
                    start: Position {
                        line: (chunk.start_line - 1) as u32,
                        character: 0,
                    },
                    end: Position {
                        line: (chunk.end_line - 1) as u32,
                        character: 0,
                    },
                },
            })
        })
        .collect();

    if locations.is_empty() {
        Ok(None)
    } else {
        Ok(Some(locations))
    }
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo check -p ripvec-mcp`

- [ ] **Step 3: Commit**

```bash
git add crates/ripvec-mcp/src/lsp/navigation.rs
git commit -m "feat(lsp): implement findReferences with BM25 + PageRank"
```

---

### Task 7: Implement hover

**Files:**
- Modify: `crates/ripvec-mcp/src/lsp/hover.rs` — implement `hover`

- [ ] **Step 1: Implement hover**

Replace the stub:

```rust
//! Hover provider — shows enriched content for the chunk under cursor.

use tower_lsp_server::jsonrpc::Result;
use tower_lsp_server::lsp_types::*;

pub async fn hover(
    params: HoverParams,
    index: &tokio::sync::RwLock<Option<ripvec_core::hybrid::HybridIndex>>,
    root: &std::path::Path,
) -> Result<Option<Hover>> {
    let pos = &params.text_document_position_params;
    let line = pos.position.line as usize + 1; // chunks use 1-based lines

    let idx_guard = index.read().await;
    let Some(hybrid) = idx_guard.as_ref() else {
        return Ok(None);
    };

    let file_path = pos
        .text_document
        .uri
        .to_file_path()
        .map_err(|()| tower_lsp_server::jsonrpc::Error::invalid_params("invalid file URI"))?;
    let file_str = file_path.display().to_string();

    // Find the chunk that contains this line in this file
    let chunk = hybrid.chunks().iter().find(|c| {
        c.file_path == file_str && c.start_line <= line && c.end_line >= line
    });

    // Also check relative path
    let rel_path = file_path
        .strip_prefix(root)
        .unwrap_or(&file_path)
        .display()
        .to_string();
    let chunk = chunk.or_else(|| {
        hybrid.chunks().iter().find(|c| {
            c.file_path == rel_path && c.start_line <= line && c.end_line >= line
        })
    });

    let Some(chunk) = chunk else {
        return Ok(None);
    };

    let markdown = format!(
        "**{}** `{}`\n\n```\n{}\n```",
        chunk.kind, chunk.name, chunk.enriched_content
    );

    Ok(Some(Hover {
        contents: HoverContents::Markup(MarkupContent {
            kind: MarkupKind::Markdown,
            value: markdown,
        }),
        range: Some(Range {
            start: Position {
                line: (chunk.start_line - 1) as u32,
                character: 0,
            },
            end: Position {
                line: (chunk.end_line - 1) as u32,
                character: 0,
            },
        }),
    }))
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo check -p ripvec-mcp`

- [ ] **Step 3: Commit**

```bash
git add crates/ripvec-mcp/src/lsp/hover.rs
git commit -m "feat(lsp): implement hover showing enriched chunk content"
```

---

### Task 8: Implement tree-sitter diagnostics

**Files:**
- Modify: `crates/ripvec-mcp/src/lsp/diagnostics.rs` — implement `publish`

- [ ] **Step 1: Implement diagnostics**

Replace the stub:

```rust
//! Tree-sitter syntax diagnostics.
//!
//! After each file open/change, re-parse with tree-sitter and report
//! ERROR/MISSING nodes as diagnostics. Provides syntax checking for
//! all 21 supported languages without a dedicated language server.

use tower_lsp_server::Client;
use tower_lsp_server::lsp_types::*;

/// Parse the file with tree-sitter and publish diagnostics for error nodes.
pub async fn publish(client: &Client, uri: &Url, text: &str) {
    let file_path = match uri.to_file_path() {
        Ok(p) => p,
        Err(()) => return,
    };

    let ext = file_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    let Some(config) = ripvec_core::languages::config_for_extension(ext) else {
        // Unsupported language — no diagnostics
        client.publish_diagnostics(uri.clone(), vec![], None).await;
        return;
    };

    let mut parser = tree_sitter::Parser::new();
    if parser.set_language(&config.language).is_err() {
        return;
    }

    let Some(tree) = parser.parse(text, None) else {
        return;
    };

    let mut diagnostics = Vec::new();
    collect_errors(tree.root_node(), text, &mut diagnostics);

    client
        .publish_diagnostics(uri.clone(), diagnostics, None)
        .await;
}

/// Walk the AST collecting ERROR and MISSING nodes.
fn collect_errors(
    node: tree_sitter::Node<'_>,
    source: &str,
    diagnostics: &mut Vec<Diagnostic>,
) {
    if node.is_error() || node.is_missing() {
        let start = node.start_position();
        let end = node.end_position();

        let message = if node.is_missing() {
            format!("Missing: expected {}", node.kind())
        } else {
            let text = &source[node.start_byte()..node.end_byte().min(source.len())];
            let preview = if text.len() > 40 {
                format!("{}...", &text[..40])
            } else {
                text.to_string()
            };
            format!("Syntax error near: {preview}")
        };

        diagnostics.push(Diagnostic {
            range: Range {
                start: Position {
                    line: start.row as u32,
                    character: start.column as u32,
                },
                end: Position {
                    line: end.row as u32,
                    character: end.column as u32,
                },
            },
            severity: Some(DiagnosticSeverity::WARNING),
            source: Some("ripvec".to_string()),
            message,
            ..Default::default()
        });
    }

    // Recurse into children
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_errors(child, source, diagnostics);
    }
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo check -p ripvec-mcp`

- [ ] **Step 3: Commit**

```bash
git add crates/ripvec-mcp/src/lsp/diagnostics.rs
git commit -m "feat(lsp): implement tree-sitter syntax diagnostics for 21 languages"
```

---

### Task 9: Add .lsp.json to the plugin

**Files:**
- Create: `../my-claude-plugins/plugins/ripvec/.lsp.json`

- [ ] **Step 1: Create the LSP configuration**

Create `/Users/rwaugh/src/mine/my-claude-plugins/plugins/ripvec/.lsp.json`:

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

- [ ] **Step 2: Commit and push**

```bash
cd /Users/rwaugh/src/mine/my-claude-plugins
git add plugins/ripvec/.lsp.json
git commit -m "feat(ripvec): add .lsp.json for Claude Code LSP integration"
git push
```

---

### Task 10: Full verification

**Files:** None (verification only)

- [ ] **Step 1: Format check**

Run: `cargo fmt --check`

- [ ] **Step 2: Clippy**

Run: `cargo clippy --all-targets -- -D warnings`

- [ ] **Step 3: Tests**

Run: `cargo test --workspace`
Expected: all existing tests pass (LSP module has no tests yet — it needs integration testing with an actual LSP client, which is out of scope for v1)

- [ ] **Step 4: Manual smoke test**

Test the LSP server responds to initialize:

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"capabilities":{}}}' | cargo run -p ripvec-mcp -- --lsp 2>/dev/null
```

Expected: JSON response with `capabilities` including `documentSymbolProvider`, `definitionProvider`, etc.

- [ ] **Step 5: Fix any issues, commit**

```bash
git commit -m "fix: address issues from LSP verification"
```
