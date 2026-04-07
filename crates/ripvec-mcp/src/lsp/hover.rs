//! Hover handler for the LSP server.
//!
//! Returns the enriched content of the semantic chunk that contains
//! the cursor position, formatted as Markdown for rich editor display.

use std::path::Path;
use std::sync::Arc;

use tower_lsp_server::jsonrpc::Result;
use tower_lsp_server::ls_types::{Hover, HoverContents, HoverParams, MarkupContent, MarkupKind};

use ripvec_core::hybrid::HybridIndex;

/// Return hover information for the symbol at the given position.
///
/// Finds the chunk in the index whose file and line range contain the
/// cursor, then returns its enriched content as Markdown.
pub async fn hover(
    params: HoverParams,
    index: &Arc<tokio::sync::RwLock<Option<HybridIndex>>>,
    root: &Path,
) -> Result<Option<Hover>> {
    let pos = &params.text_document_position_params;
    let Some(path_cow) = pos.text_document.uri.to_file_path() else {
        return Ok(None);
    };
    let file_path = path_cow.into_owned();

    let line_1based = (pos.position.line as usize) + 1;

    let guard = index.read().await;
    let Some(hybrid) = guard.as_ref() else {
        return Ok(None);
    };

    let chunks = hybrid.chunks();

    // Try to match the file path as absolute or relative to root.
    let rel_path = file_path
        .strip_prefix(root)
        .unwrap_or(&file_path)
        .to_string_lossy();

    let chunk = chunks.iter().find(|c| {
        let chunk_path = Path::new(&c.file_path);
        let matches_abs = chunk_path == file_path;
        let matches_rel = c.file_path == rel_path;
        (matches_abs || matches_rel) && c.start_line <= line_1based && line_1based <= c.end_line
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
        range: None,
    }))
}
