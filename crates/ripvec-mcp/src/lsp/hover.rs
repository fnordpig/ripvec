//! Hover handler for the LSP server.
//!
//! Returns the enriched content of the semantic chunk that contains
//! the cursor position, formatted as Markdown for rich editor display.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use tower_lsp_server::jsonrpc::Result;
use tower_lsp_server::ls_types::{Hover, HoverContents, HoverParams, MarkupContent, MarkupKind};

use ripvec_core::chunk::CodeChunk;
use ripvec_core::hybrid::HybridIndex;

/// Find the chunk containing `file_path` at `line_1based` in the given index.
fn find_hover_chunk<'a>(
    chunks: &'a [CodeChunk],
    file_path: &Path,
    search_root: &Path,
    line_1based: usize,
) -> Option<&'a CodeChunk> {
    let rel_path = file_path
        .strip_prefix(search_root)
        .unwrap_or(file_path)
        .to_string_lossy();

    chunks.iter().find(|c| {
        let chunk_path = Path::new(&c.file_path);
        let matches_abs = chunk_path == file_path;
        let matches_rel = c.file_path == *rel_path;
        (matches_abs || matches_rel) && c.start_line <= line_1based && line_1based <= c.end_line
    })
}

/// Format a chunk as a Markdown hover response.
fn hover_response(chunk: &CodeChunk) -> Hover {
    let markdown = format!(
        "**{}** `{}`\n\n```\n{}\n```",
        chunk.kind, chunk.name, chunk.enriched_content
    );
    Hover {
        contents: HoverContents::Markup(MarkupContent {
            kind: MarkupKind::Markdown,
            value: markdown,
        }),
        range: None,
    }
}

/// Return hover information for the symbol at the given position.
///
/// Finds the chunk in the index whose file and line range contain the
/// cursor, then returns its enriched content as Markdown.
///
/// Tries the default index first; if the file isn't found there, checks
/// on-demand root indices for a matching root.
pub async fn hover(
    params: HoverParams,
    index: &Arc<tokio::sync::RwLock<Option<HybridIndex>>>,
    root_indices: &Arc<tokio::sync::RwLock<HashMap<PathBuf, HybridIndex>>>,
    root: &Path,
) -> Result<Option<Hover>> {
    let pos = &params.text_document_position_params;
    let Some(path_cow) = pos.text_document.uri.to_file_path() else {
        return Ok(None);
    };
    let file_path = path_cow.into_owned();

    let line_1based = (pos.position.line as usize) + 1;

    // Try the default index first.
    {
        let guard = index.read().await;
        if let Some(hybrid) = guard.as_ref()
            && let Some(chunk) = find_hover_chunk(hybrid.chunks(), &file_path, root, line_1based)
        {
            return Ok(Some(hover_response(chunk)));
        }
    }

    // Fall back to on-demand root indices.
    let ri_guard = root_indices.read().await;
    if let Some((alt_root, hybrid)) = ri_guard.iter().find(|(r, _)| file_path.starts_with(r))
        && let Some(chunk) = find_hover_chunk(hybrid.chunks(), &file_path, alt_root, line_1based)
    {
        return Ok(Some(hover_response(chunk)));
    }

    Ok(None)
}
