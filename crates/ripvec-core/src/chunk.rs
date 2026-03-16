//! Tree-sitter code chunking (stub — replaced by Task 6 agent).

use std::path::Path;

use crate::languages::LangConfig;

/// A contiguous chunk of source code extracted by tree-sitter.
#[derive(Debug, Clone)]
pub struct CodeChunk {
    /// Path to the source file.
    pub file_path: String,
    /// Name of the definition (function name, struct name, etc.).
    pub name: String,
    /// Kind of definition (e.g. `"function"`, `"struct"`).
    pub kind: String,
    /// First line of the chunk (1-indexed).
    pub start_line: usize,
    /// Last line of the chunk (1-indexed).
    pub end_line: usize,
    /// Raw source text of the chunk.
    pub content: String,
}

/// Chunk a source file into semantic units using tree-sitter.
///
/// Returns an empty `Vec` if parsing fails or the file has no recognised
/// definitions.
pub fn chunk_file(_path: &Path, _source: &str, _config: &LangConfig) -> Vec<CodeChunk> {
    vec![]
}
