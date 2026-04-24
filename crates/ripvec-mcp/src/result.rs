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
    /// Full code content in a markdown fenced block with language annotation.
    pub content: String,
}

/// Top-level search response containing ranked results.
#[derive(Serialize)]
pub struct SearchResponse {
    /// Ranked search results, best match first.
    pub results: Vec<SearchResultItem>,
}

/// Map a file extension to a markdown language identifier.
fn ext_to_lang(ext: &str) -> &str {
    match ext {
        "rs" => "rust",
        "py" | "pyi" => "python",
        "js" | "jsx" => "javascript",
        "ts" | "tsx" => "typescript",
        "go" => "go",
        "java" => "java",
        "c" | "h" => "c",
        "cpp" | "hpp" | "cc" => "cpp",
        "sql" => "sql",
        "toml" => "toml",
        "yaml" | "yml" => "yaml",
        "json" => "json",
        "sh" | "bash" | "zsh" => "bash",
        "rb" => "ruby",
        _ => ext,
    }
}

impl SearchResultItem {
    /// Convert a [`CodeChunk`] (1-based lines) to an LSP-compatible result item.
    ///
    /// Lines are converted from 1-based (ripvec-core) to 0-based (LSP).
    /// The preview is truncated to approximately 200 characters.
    /// The content field contains the full code in a markdown fenced block.
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

        let ext = std::path::Path::new(&chunk.file_path)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");
        let lang = ext_to_lang(ext);
        let content = format!("```{lang}\n{}\n```", chunk.content);

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
            content,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ripvec_core::chunk::CodeChunk;

    fn make_chunk(content: &str, start_line: usize, end_line: usize) -> CodeChunk {
        CodeChunk {
            file_path: "src/main.rs".to_string(),
            name: "test_fn".to_string(),
            kind: "function_item".to_string(),
            start_line,
            end_line,
            enriched_content: content.to_string(),
            content: content.to_string(),
        }
    }

    #[test]
    fn test_from_chunk_converts_lines_to_zero_based() {
        let chunk = make_chunk("fn hello() {}", 10, 20);
        let item = SearchResultItem::from_chunk(&chunk, 0.95);
        assert_eq!(item.lsp_location.start_line, 9);
        assert_eq!(item.lsp_location.end_line, 19);
    }

    #[test]
    fn test_from_chunk_preview_truncation() {
        let long_content = "a".repeat(300);
        let chunk = make_chunk(&long_content, 1, 5);
        let item = SearchResultItem::from_chunk(&chunk, 0.8);
        assert!(item.preview.ends_with("..."));
        // 200 chars of content + "..."
        assert_eq!(item.preview.len(), 203);
    }

    #[test]
    fn test_from_chunk_short_content_not_truncated() {
        let content = "fn short() {}";
        let chunk = make_chunk(content, 1, 1);
        let item = SearchResultItem::from_chunk(&chunk, 0.9);
        assert_eq!(item.preview, content);
    }

    #[test]
    fn test_from_chunk_content_has_fenced_block() {
        let content = "fn hello() {}";
        let chunk = make_chunk(content, 1, 1);
        let item = SearchResultItem::from_chunk(&chunk, 0.95);
        assert_eq!(item.content, "```rust\nfn hello() {}\n```");
    }

    #[test]
    fn test_from_chunk_content_python_extension() {
        let chunk = CodeChunk {
            file_path: "app/views.py".to_string(),
            name: "index".to_string(),
            kind: "function_definition".to_string(),
            start_line: 1,
            end_line: 3,
            enriched_content: "def index(): pass".to_string(),
            content: "def index(): pass".to_string(),
        };
        let item = SearchResultItem::from_chunk(&chunk, 0.8);
        assert_eq!(item.content, "```python\ndef index(): pass\n```");
    }

    #[test]
    fn test_from_chunk_content_python_stub_extension() {
        let chunk = CodeChunk {
            file_path: "app/views.pyi".to_string(),
            name: "index".to_string(),
            kind: "function_definition".to_string(),
            start_line: 1,
            end_line: 1,
            enriched_content: "def index() -> str: ...".to_string(),
            content: "def index() -> str: ...".to_string(),
        };
        let item = SearchResultItem::from_chunk(&chunk, 0.8);
        assert_eq!(item.content, "```python\ndef index() -> str: ...\n```");
    }

    #[test]
    fn test_from_chunk_content_unknown_extension() {
        let chunk = CodeChunk {
            file_path: "data.xyz".to_string(),
            name: "data".to_string(),
            kind: "file".to_string(),
            start_line: 1,
            end_line: 1,
            enriched_content: "some data".to_string(),
            content: "some data".to_string(),
        };
        let item = SearchResultItem::from_chunk(&chunk, 0.7);
        assert_eq!(item.content, "```xyz\nsome data\n```");
    }

    #[test]
    fn test_empty_results_serialization() {
        let response = SearchResponse {
            results: Vec::new(),
        };
        let json = serde_json::to_string(&response).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["results"], serde_json::json!([]));
    }

    #[test]
    fn test_lsp_location_zero_based() {
        let chunk = make_chunk("line 1", 1, 1);
        let item = SearchResultItem::from_chunk(&chunk, 1.0);
        assert_eq!(item.lsp_location.start_line, 0);
        assert_eq!(item.lsp_location.end_line, 0);
    }

    #[test]
    fn test_ext_to_lang_mappings() {
        assert_eq!(ext_to_lang("rs"), "rust");
        assert_eq!(ext_to_lang("py"), "python");
        assert_eq!(ext_to_lang("pyi"), "python");
        assert_eq!(ext_to_lang("js"), "javascript");
        assert_eq!(ext_to_lang("jsx"), "javascript");
        assert_eq!(ext_to_lang("ts"), "typescript");
        assert_eq!(ext_to_lang("tsx"), "typescript");
        assert_eq!(ext_to_lang("go"), "go");
        assert_eq!(ext_to_lang("java"), "java");
        assert_eq!(ext_to_lang("c"), "c");
        assert_eq!(ext_to_lang("h"), "c");
        assert_eq!(ext_to_lang("cpp"), "cpp");
        assert_eq!(ext_to_lang("sh"), "bash");
        assert_eq!(ext_to_lang("unknown"), "unknown");
    }
}
