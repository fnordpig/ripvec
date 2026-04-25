//! Hover handler for the LSP server.
//!
//! Returns the enriched content of the semantic chunk that contains
//! the cursor position, formatted as Markdown for rich editor display.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use tower_lsp_server::jsonrpc::Result;
use tower_lsp_server::ls_types::{Hover, HoverContents, HoverParams, MarkupContent, MarkupKind};

use ripvec_core::cache::reindex::load_cached_index;
use ripvec_core::chunk::{ChunkConfig, CodeChunk, chunk_file};
use ripvec_core::hybrid::HybridIndex;
use ripvec_core::languages::config_for_extension;

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

    chunks
        .iter()
        .filter(|c| {
            let chunk_path = Path::new(&c.file_path);
            let matches_abs = chunk_path == file_path;
            let matches_rel = c.file_path == *rel_path;
            (matches_abs || matches_rel) && c.start_line <= line_1based && line_1based <= c.end_line
        })
        .min_by_key(|c| c.end_line.saturating_sub(c.start_line))
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

/// Parse the requested file directly when no shared index/cache has hover data.
async fn hover_chunk_from_file(file_path: &Path, line_1based: usize) -> Option<CodeChunk> {
    let source = tokio::fs::read_to_string(file_path).await.ok()?;
    let ext = file_path.extension().and_then(|e| e.to_str()).unwrap_or("");
    let config = config_for_extension(ext)?;
    let chunks = chunk_file(file_path, &source, &config, &ChunkConfig::default());
    let root = file_path.parent().unwrap_or_else(|| Path::new(""));
    find_hover_chunk(&chunks, file_path, root, line_1based).cloned()
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
    drop(ri_guard);

    // Fall back to disk cache (cross-process).
    let model_repo = "nomic-ai/modernbert-embed-base";
    let mut candidate = file_path.parent();
    while let Some(dir) = candidate {
        if (dir.join(".git").exists() || dir.join(".ripvec").exists())
            && let Some(cached) = load_cached_index(dir, model_repo)
            && let Some(chunk) = find_hover_chunk(cached.chunks(), &file_path, dir, line_1based)
        {
            return Ok(Some(hover_response(chunk)));
        }
        candidate = dir.parent();
    }

    if let Some(chunk) = hover_chunk_from_file(&file_path, line_1based).await {
        return Ok(Some(hover_response(&chunk)));
    }

    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};
    use tower_lsp_server::ls_types::{
        HoverContents, Position, TextDocumentIdentifier, TextDocumentPositionParams, Uri,
        WorkDoneProgressParams,
    };

    fn chunk(file_path: &str, name: &str, start_line: usize, end_line: usize) -> CodeChunk {
        CodeChunk {
            file_path: file_path.to_string(),
            name: name.to_string(),
            kind: "function_definition".to_string(),
            start_line,
            end_line,
            content: format!("def {name}(): pass"),
            enriched_content: format!("def {name}(): pass"),
        }
    }

    fn temp_dir(name: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("ripvec-{name}-{}-{unique}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn find_hover_chunk_prefers_narrowest_matching_chunk() {
        let root = Path::new("/tmp/project");
        let file_path = root.join("sample.py");
        let chunks = vec![
            chunk("sample.py", "Container", 1, 20),
            chunk("sample.py", "snapshot", 5, 7),
        ];

        let found = find_hover_chunk(&chunks, &file_path, root, 6).unwrap();

        assert_eq!(found.name, "snapshot");
    }

    #[tokio::test]
    async fn hover_parses_current_file_when_no_index_exists() {
        let root = temp_dir("hover-empty-index");
        let file_path = root.join("sample.py");
        tokio::fs::write(&file_path, "def snapshot():\n    return 42\n")
            .await
            .unwrap();

        let params = HoverParams {
            text_document_position_params: TextDocumentPositionParams {
                text_document: TextDocumentIdentifier {
                    uri: Uri::from_file_path(&file_path).unwrap(),
                },
                position: Position {
                    line: 0,
                    character: 4,
                },
            },
            work_done_progress_params: WorkDoneProgressParams::default(),
        };
        let index = Arc::new(tokio::sync::RwLock::new(None::<HybridIndex>));
        let root_indices = Arc::new(tokio::sync::RwLock::new(HashMap::new()));

        let result = hover(params, &index, &root_indices, &root).await.unwrap();

        std::fs::remove_dir_all(&root).unwrap();

        let hover = result.expect("hover should be available from current file");
        match hover.contents {
            HoverContents::Markup(markup) => {
                assert!(markup.value.contains("snapshot"));
            }
            other => panic!("expected markup hover, got {other:?}"),
        }
    }
}
