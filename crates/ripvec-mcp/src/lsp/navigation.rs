//! Stub navigation handlers for the LSP server.
//!
//! Will eventually resolve definitions and references using
//! the semantic index and tree-sitter symbol tables.

use tower_lsp_server::jsonrpc::Result;
use tower_lsp_server::ls_types::{
    GotoDefinitionParams, GotoDefinitionResponse, Location, ReferenceParams,
};

/// Resolve the definition of the symbol at the given position.
/// Currently a stub that returns `Ok(None)`.
#[expect(clippy::unused_async, reason = "will use async once implemented")]
pub async fn goto_definition(
    _params: GotoDefinitionParams,
) -> Result<Option<GotoDefinitionResponse>> {
    Ok(None)
}

/// Find all references to the symbol at the given position.
/// Currently a stub that returns `Ok(None)`.
#[expect(clippy::unused_async, reason = "will use async once implemented")]
pub async fn find_references(_params: ReferenceParams) -> Result<Option<Vec<Location>>> {
    Ok(None)
}
