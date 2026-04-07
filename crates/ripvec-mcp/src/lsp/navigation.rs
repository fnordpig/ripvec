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

/// Information about a symbol extracted from a cursor position via tree-sitter.
struct SymbolAtPosition {
    /// The symbol text (e.g., "boost_with_pagerank").
    name: String,
    /// The tree-sitter node kind (e.g., "identifier", "type_identifier").
    #[expect(dead_code, reason = "will be used for smarter search routing")]
    node_kind: String,
    /// The kind of the parent node (e.g., "call_expression", "use_declaration").
    #[expect(dead_code, reason = "will be used for smarter search routing")]
    parent_kind: String,
}

/// Extract the symbol at the given position using tree-sitter.
///
/// Parses the file, finds the smallest named node at the cursor, and returns
/// its text plus structural context (node kind, parent kind). Falls back to
/// text-based word extraction for unsupported extensions.
fn symbol_at_position(source: &str, line: u32, character: u32, ext: &str) -> Option<SymbolAtPosition> {
    let config = ripvec_core::languages::config_for_extension(ext)?;
    let mut parser = tree_sitter::Parser::new();
    parser.set_language(&config.language).ok()?;
    let tree = parser.parse(source, None)?;

    // Convert line/character to byte offset
    let mut byte_offset = 0;
    for (i, line_str) in source.lines().enumerate() {
        if i == line as usize {
            byte_offset += (character as usize).min(line_str.len());
            break;
        }
        byte_offset += line_str.len() + 1; // +1 for newline
    }

    // Find the smallest node at this position
    let mut node = tree
        .root_node()
        .descendant_for_byte_range(byte_offset, byte_offset)?;

    // Walk up to the nearest identifier-like node (skip operators, punctuation, keywords)
    loop {
        match node.kind() {
            "identifier" | "type_identifier" | "field_identifier" | "property_identifier"
            | "simple_identifier" | "word" | "constant" | "bare_key" | "command_name"
            | "string_content" => break,
            _ => {
                node = node.parent()?;
            }
        }
    }

    let name = source[node.start_byte()..node.end_byte()].to_string();
    let node_kind = node.kind().to_string();
    let parent_kind = node
        .parent()
        .map(|p| p.kind().to_string())
        .unwrap_or_default();

    Some(SymbolAtPosition {
        name,
        node_kind,
        parent_kind,
    })
}

/// Extract the identifier (word) at the given 0-based line and character.
///
/// Text-based fallback when tree-sitter parsing isn't available.
/// Expands outward from the cursor to include all contiguous
/// identifier characters (`[a-zA-Z0-9_]`).
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

/// Extract the symbol name at a cursor position, using tree-sitter with text fallback.
fn extract_symbol(source: &str, line: u32, character: u32, ext: &str) -> Option<String> {
    // Try tree-sitter first for precise symbol extraction
    if let Some(sym) = symbol_at_position(source, line, character, ext) {
        return Some(sym.name);
    }
    // Fall back to text-based word extraction
    word_at_position(source, line, character)
}

/// Resolve the definition of the symbol at the given position.
///
/// Uses tree-sitter to identify the symbol under the cursor, then searches
/// the BM25 keyword index with PageRank boosting. Returns the best match
/// where the chunk name exactly equals the symbol.
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

    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    let Some(word) = extract_symbol(&source, pos.position.line, pos.position.character, ext)
    else {
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
/// Uses tree-sitter to identify the symbol under the cursor, then searches
/// the BM25 keyword index with PageRank boosting. Returns all chunks whose
/// name or content contains the symbol.
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

    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    let Some(word) = extract_symbol(&source, pos.position.line, pos.position.character, ext)
    else {
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
