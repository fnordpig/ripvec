//! Tree-sitter syntax diagnostics publisher for the LSP server.
//!
//! Parses open documents with tree-sitter and publishes `ERROR` /
//! `MISSING` nodes as LSP diagnostics so editors can surface parse
//! errors without a full language server.

use tower_lsp_server::Client;
use tower_lsp_server::ls_types::{Diagnostic, DiagnosticSeverity, Position, Range, Uri};

use ripvec_core::languages::config_for_extension;

/// Maximum bytes of source to include in the "syntax error near:" preview.
const PREVIEW_MAX_BYTES: usize = 40;

/// Publish tree-sitter syntax diagnostics for a document.
///
/// Parses `text` with the tree-sitter grammar matched by the URI's file
/// extension. Any `ERROR` or `MISSING` nodes are converted to LSP
/// diagnostics at severity `WARNING` (tree-sitter errors are heuristic,
/// not authoritative). For unsupported extensions, publishes an empty
/// diagnostic list to clear stale markers.
pub async fn publish(client: &Client, uri: &Uri, text: &str) {
    let ext = uri
        .to_file_path()
        .as_deref()
        .and_then(|p| p.extension())
        .and_then(|e| e.to_str())
        .map(str::to_owned);

    let Some(ext) = ext else {
        // Unknown URI scheme or no extension -- clear any stale diagnostics.
        client
            .publish_diagnostics(uri.clone(), Vec::new(), None)
            .await;
        return;
    };

    let Some(config) = config_for_extension(&ext) else {
        // Unsupported extension -- clear stale diagnostics.
        client
            .publish_diagnostics(uri.clone(), Vec::new(), None)
            .await;
        return;
    };

    let mut parser = tree_sitter::Parser::new();
    if parser.set_language(&config.language).is_err() {
        client
            .publish_diagnostics(uri.clone(), Vec::new(), None)
            .await;
        return;
    }

    let Some(tree) = parser.parse(text, None) else {
        client
            .publish_diagnostics(uri.clone(), Vec::new(), None)
            .await;
        return;
    };

    let mut diagnostics = Vec::new();
    collect_errors(tree.root_node(), text, &mut diagnostics);

    client
        .publish_diagnostics(uri.clone(), diagnostics, None)
        .await;
}

/// Recursively walk the AST collecting `ERROR` and `MISSING` nodes.
fn collect_errors(node: tree_sitter::Node<'_>, source: &str, out: &mut Vec<Diagnostic>) {
    if node.is_error() {
        let preview = node_preview(node, source);
        out.push(make_diagnostic(
            node,
            format!("Syntax error near: {preview}"),
        ));
    } else if node.is_missing() {
        let kind = node.kind();
        out.push(make_diagnostic(node, format!("Missing: expected {kind}")));
    }

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_errors(child, source, out);
    }
}

/// Saturating conversion from `usize` to `u32` for LSP line/column numbers.
fn pos_to_u32(n: usize) -> u32 {
    u32::try_from(n).unwrap_or(u32::MAX)
}

/// Build a `Diagnostic` from a tree-sitter node.
fn make_diagnostic(node: tree_sitter::Node<'_>, message: String) -> Diagnostic {
    let start = node.start_position();
    let end = node.end_position();

    Diagnostic {
        range: Range {
            start: Position {
                line: pos_to_u32(start.row),
                character: pos_to_u32(start.column),
            },
            end: Position {
                line: pos_to_u32(end.row),
                character: pos_to_u32(end.column),
            },
        },
        severity: Some(DiagnosticSeverity::WARNING),
        source: Some("ripvec".to_owned()),
        message,
        ..Default::default()
    }
}

/// Extract a short preview of the source text covered by a node.
fn node_preview(node: tree_sitter::Node<'_>, source: &str) -> String {
    let text = &source[node.start_byte()..node.end_byte()];
    // Take up to PREVIEW_MAX_BYTES, truncating at a char boundary.
    let truncated = if text.len() > PREVIEW_MAX_BYTES {
        let end = text
            .char_indices()
            .take_while(|(i, _)| *i < PREVIEW_MAX_BYTES)
            .last()
            .map_or(0, |(i, c)| i + c.len_utf8());
        format!("{}...", &text[..end])
    } else {
        text.to_owned()
    };
    // Collapse whitespace for readability.
    truncated
        .chars()
        .map(|c| if c.is_whitespace() { ' ' } else { c })
        .collect()
}
