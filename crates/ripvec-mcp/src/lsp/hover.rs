//! Stub hover handler for the LSP server.
//!
//! Will eventually return semantic search context or tree-sitter
//! scope information on hover.

use tower_lsp_server::jsonrpc::Result;
use tower_lsp_server::ls_types::{Hover, HoverParams};

/// Return hover information for the symbol at the given position.
/// Currently a stub that returns `Ok(None)`.
#[expect(clippy::unused_async, reason = "will use async once implemented")]
pub async fn hover(_params: HoverParams) -> Result<Option<Hover>> {
    Ok(None)
}
