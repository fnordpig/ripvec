//! LSP-compatible result types for MCP tool responses.
//!
//! Converts ripvec-core [`CodeChunk`] data into a structured JSON format
//! with 0-based line/character positions matching the LSP spec.

use ripvec_core::chunk::CodeChunk;
use serde::Serialize;

/// LSP-compatible source location.
///
/// Lines and characters are 0-based, matching the Language Server Protocol
/// convention so Claude Code can jump directly to results.
#[derive(Serialize)]
pub struct LspLocation {
    /// Absolute or relative file path.
    pub file_path: String,
    /// 0-based start line.
    pub start_line: usize,
    /// 0-based start character (always 0 for chunk-level results).
    pub start_character: usize,
    /// 0-based end line.
    pub end_line: usize,
    /// 0-based end character (always 0 for chunk-level results).
    pub end_character: usize,
}

/// A single search result with location, score, and preview.
#[derive(Serialize)]
pub struct SearchResultItem {
    /// LSP-compatible source location.
    pub lsp_location: LspLocation,
    /// Symbol or chunk name (e.g. function name).
    pub symbol_name: String,
    /// Cosine similarity score (0.0 to 1.0).
    pub similarity: f32,
    /// Truncated source preview (~200 chars).
    pub preview: String,
}

/// Top-level search response containing ranked results.
#[derive(Serialize)]
pub struct SearchResponse {
    /// Ranked search results, best match first.
    pub results: Vec<SearchResultItem>,
}

impl SearchResultItem {
    /// Convert a [`CodeChunk`] (1-based lines) to an LSP-compatible result item.
    ///
    /// Lines are converted from 1-based (ripvec-core) to 0-based (LSP).
    /// The preview is truncated to approximately 200 characters.
    pub fn from_chunk(chunk: &CodeChunk, similarity: f32) -> Self {
        let preview = if chunk.content.len() > 200 {
            let mut end = 200;
            // Don't split in the middle of a multi-byte character
            while !chunk.content.is_char_boundary(end) && end < chunk.content.len() {
                end += 1;
            }
            format!("{}...", &chunk.content[..end])
        } else {
            chunk.content.clone()
        };

        Self {
            lsp_location: LspLocation {
                file_path: chunk.file_path.clone(),
                start_line: chunk.start_line.saturating_sub(1),
                start_character: 0,
                end_line: chunk.end_line.saturating_sub(1),
                end_character: 0,
            },
            symbol_name: chunk.name.clone(),
            similarity,
            preview,
        }
    }
}
