//! Go-to-definition and find-references handlers for the LSP server.
//!
//! Extracts the identifier under the cursor, searches the BM25 keyword
//! index with PageRank boosting, and returns matching chunk locations.

use std::path::Path;
use std::sync::Arc;

use tower_lsp_server::jsonrpc::Result;
use tower_lsp_server::ls_types::{
    GotoDefinitionParams, GotoDefinitionResponse, Location, Position, Range, ReferenceParams, Uri,
};

use ripvec_core::chunk::CodeChunk;
use ripvec_core::hybrid::{HybridIndex, SearchMode, boost_with_pagerank, pagerank_lookup};
use ripvec_core::repo_map::RepoGraph;

/// Extract the identifier (word) at the given 0-based line and character.
///
/// Expands outward from the cursor position to include all contiguous
/// identifier characters (`[a-zA-Z0-9_]`).  Returns `None` if the cursor
/// is not on an identifier character or the line doesn't exist.
fn word_at_position(source: &str, line: u32, character: u32) -> Option<String> {
    let target_line = source.lines().nth(line as usize)?;
    let col = character as usize;

    // Cursor may be just past the last char (e.g. at EOL).
    if col >= target_line.len() {
        return None;
    }

    let bytes = target_line.as_bytes();
    if !bytes[col].is_ascii_alphanumeric() && bytes[col] != b'_' {
        return None;
    }

    // Expand left.
    let start = (0..=col)
        .rev()
        .take_while(|&i| bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_')
        .last()?;

    // Expand right.
    let end = (col..bytes.len())
        .take_while(|&i| bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_')
        .last()
        .map_or(col, |i| i + 1);

    let word = &target_line[start..end];
    if word.is_empty() {
        None
    } else {
        Some(word.to_string())
    }
}

/// Saturating conversion from `usize` to `u32`.
fn line_to_u32(n: usize) -> u32 {
    u32::try_from(n).unwrap_or(u32::MAX)
}

/// Build an LSP `Range` from a chunk's 1-based line numbers.
fn chunk_range(chunk: &CodeChunk) -> Range {
    Range {
        start: Position {
            line: line_to_u32(chunk.start_line.saturating_sub(1)),
            character: 0,
        },
        end: Position {
            line: line_to_u32(chunk.end_line.saturating_sub(1)),
            character: 0,
        },
    }
}

/// Build a `Uri` from a chunk's file path, resolving against `root` if relative.
fn uri_for_chunk(chunk: &CodeChunk, root: &Path) -> Option<Uri> {
    let path = Path::new(&chunk.file_path);
    let abs = if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    };
    Uri::from_file_path(abs)
}

/// Resolve the definition of the symbol at the given position.
///
/// Reads the file to extract the word under the cursor, searches the
/// BM25 keyword index, applies PageRank boosting, and returns the best
/// match where the chunk name exactly equals the word.
pub async fn goto_definition(
    params: GotoDefinitionParams,
    index: &Arc<tokio::sync::RwLock<Option<HybridIndex>>>,
    repo_graph: &Arc<std::sync::RwLock<Option<RepoGraph>>>,
    root: &Path,
) -> Result<Option<GotoDefinitionResponse>> {
    let pos = &params.text_document_position_params;
    let Some(path_cow) = pos.text_document.uri.to_file_path() else {
        return Ok(None);
    };
    let path = path_cow.into_owned();

    let Ok(source) = tokio::fs::read_to_string(&path).await else {
        return Ok(None);
    };

    let Some(word) = word_at_position(&source, pos.position.line, pos.position.character) else {
        return Ok(None);
    };

    let guard = index.read().await;
    let Some(hybrid) = guard.as_ref() else {
        return Ok(None);
    };

    let mut results = hybrid.search(&[], &word, 20, 0.0, SearchMode::Keyword);

    if let Ok(graph_guard) = repo_graph.read()
        && let Some(graph) = graph_guard.as_ref()
    {
        let pr = pagerank_lookup(graph);
        boost_with_pagerank(&mut results, hybrid.chunks(), &pr, 0.3);
    }

    let chunks = hybrid.chunks();

    // Prefer an exact name match; fall back to the top-ranked result.
    let best = results
        .iter()
        .find(|(idx, _)| chunks.get(*idx).is_some_and(|c| c.name == word))
        .or_else(|| results.first());

    let Some(&(idx, _)) = best else {
        return Ok(None);
    };
    let Some(chunk) = chunks.get(idx) else {
        return Ok(None);
    };
    let Some(uri) = uri_for_chunk(chunk, root) else {
        return Ok(None);
    };

    Ok(Some(GotoDefinitionResponse::Scalar(Location {
        uri,
        range: chunk_range(chunk),
    })))
}

/// Find all references to the symbol at the given position.
///
/// Extracts the word under the cursor, searches the BM25 keyword index
/// with PageRank boosting, and returns all chunks whose name or content
/// contains the word.
pub async fn find_references(
    params: ReferenceParams,
    index: &Arc<tokio::sync::RwLock<Option<HybridIndex>>>,
    repo_graph: &Arc<std::sync::RwLock<Option<RepoGraph>>>,
    root: &Path,
) -> Result<Option<Vec<Location>>> {
    let pos = &params.text_document_position;
    let Some(path_cow) = pos.text_document.uri.to_file_path() else {
        return Ok(None);
    };
    let path = path_cow.into_owned();

    let Ok(source) = tokio::fs::read_to_string(&path).await else {
        return Ok(None);
    };

    let Some(word) = word_at_position(&source, pos.position.line, pos.position.character) else {
        return Ok(None);
    };

    let guard = index.read().await;
    let Some(hybrid) = guard.as_ref() else {
        return Ok(None);
    };

    let mut results = hybrid.search(&[], &word, 50, 0.0, SearchMode::Keyword);

    if let Ok(graph_guard) = repo_graph.read()
        && let Some(graph) = graph_guard.as_ref()
    {
        let pr = pagerank_lookup(graph);
        boost_with_pagerank(&mut results, hybrid.chunks(), &pr, 0.3);
    }

    let chunks = hybrid.chunks();
    let locations: Vec<Location> = results
        .into_iter()
        .filter_map(|(idx, _score)| {
            let chunk = chunks.get(idx)?;
            if !chunk.name.contains(&word) && !chunk.content.contains(&word) {
                return None;
            }
            let uri = uri_for_chunk(chunk, root)?;
            Some(Location {
                uri,
                range: chunk_range(chunk),
            })
        })
        .collect();

    if locations.is_empty() {
        Ok(None)
    } else {
        Ok(Some(locations))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_at_position_extracts_identifier() {
        let src = "fn hello_world() {\n    let x = 42;\n}";
        assert_eq!(word_at_position(src, 0, 3), Some("hello_world".to_string()));
        assert_eq!(
            word_at_position(src, 0, 10),
            Some("hello_world".to_string())
        );
        assert_eq!(word_at_position(src, 0, 0), Some("fn".to_string()));
    }

    #[test]
    fn word_at_position_returns_none_on_non_identifier() {
        let src = "fn foo() {}";
        // '(' is at position 6
        assert_eq!(word_at_position(src, 0, 6), None);
    }

    #[test]
    fn word_at_position_returns_none_past_eol() {
        let src = "fn foo";
        assert_eq!(word_at_position(src, 0, 100), None);
    }

    #[test]
    fn word_at_position_bad_line() {
        let src = "one line";
        assert_eq!(word_at_position(src, 5, 0), None);
    }
}
