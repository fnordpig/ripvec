//! Stub diagnostics publisher for the LSP server.
//!
//! Will eventually run tree-sitter parse on open/changed documents
//! and publish syntax error diagnostics.

use tower_lsp_server::Client;
use tower_lsp_server::ls_types::Uri;

/// Publish diagnostics for a document. Currently a no-op stub.
#[expect(clippy::unused_async, reason = "will use async once implemented")]
pub async fn publish(_client: &Client, _uri: &Uri, _text: &str) {}
