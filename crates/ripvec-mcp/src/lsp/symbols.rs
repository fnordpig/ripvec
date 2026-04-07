//! Stub symbol handlers for the LSP server.
//!
//! Will eventually use tree-sitter to extract document symbols
//! and the semantic index for workspace-wide symbol search.

use tower_lsp_server::jsonrpc::Result;
use tower_lsp_server::ls_types::{
    DocumentSymbolParams, DocumentSymbolResponse, WorkspaceSymbolParams, WorkspaceSymbolResponse,
};

/// Return document symbols (outline) for a single file.
/// Currently a stub that returns `Ok(None)`.
#[expect(clippy::unused_async, reason = "will use async once implemented")]
pub async fn document_symbol(
    _params: DocumentSymbolParams,
) -> Result<Option<DocumentSymbolResponse>> {
    Ok(None)
}

/// Search for symbols across the entire workspace.
/// Currently a stub that returns `Ok(None)`.
#[expect(clippy::unused_async, reason = "will use async once implemented")]
pub async fn workspace_symbol(
    _params: WorkspaceSymbolParams,
) -> Result<Option<WorkspaceSymbolResponse>> {
    Ok(None)
}
