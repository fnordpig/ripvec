//! Tree-sitter based code chunking.
//!
//! Parses source files into ASTs and extracts semantic chunks at
//! function, class, and method boundaries. Falls back to whole-file
//! chunks when no semantic boundaries are found.

use std::path::Path;
use tree_sitter::{Parser, QueryCursor};

/// A semantic chunk extracted from a source file.
#[derive(Debug, Clone)]
pub struct CodeChunk {
    /// Path to the source file.
    pub file_path: String,
    /// Name of the definition (function name, class name, etc.).
    pub name: String,
    /// Kind of syntax node (e.g., "function_item", "class_definition").
    pub kind: String,
    /// 1-based start line number.
    pub start_line: usize,
    /// 1-based end line number.
    pub end_line: usize,
    /// Source text of the chunk.
    pub content: String,
}

/// Extract semantic chunks from a source file.
///
/// Uses tree-sitter to parse the file and extract definitions matching
/// the language's query patterns. Falls back to a single whole-file
/// chunk if no semantic boundaries are found.
pub fn chunk_file(
    path: &Path,
    source: &str,
    config: &crate::languages::LangConfig,
) -> Vec<CodeChunk> {
    let mut parser = Parser::new();
    if parser.set_language(&config.language).is_err() {
        return vec![];
    }

    let tree = match parser.parse(source, None) {
        Some(t) => t,
        None => return vec![],
    };

    let mut cursor = QueryCursor::new();
    let mut chunks = Vec::new();

    for m in cursor.matches(&config.query, tree.root_node(), source.as_bytes()) {
        let mut name = String::new();
        let mut def_node = None;
        for cap in m.captures {
            let cap_name = &config.query.capture_names()[cap.index as usize];
            if cap_name == "name" {
                name = source[cap.node.start_byte()..cap.node.end_byte()].to_string();
            } else if cap_name == "def" {
                def_node = Some(cap.node);
            }
        }
        if let Some(node) = def_node {
            chunks.push(CodeChunk {
                file_path: path.display().to_string(),
                name,
                kind: node.kind().to_string(),
                start_line: node.start_position().row + 1,
                end_line: node.end_position().row + 1,
                content: source[node.start_byte()..node.end_byte()].to_string(),
            });
        }
    }

    // Fallback: whole file as one chunk if no semantic matches
    if chunks.is_empty() && !source.trim().is_empty() {
        chunks.push(CodeChunk {
            file_path: path.display().to_string(),
            name: path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string(),
            kind: "file".to_string(),
            start_line: 1,
            end_line: source.lines().count(),
            content: source.to_string(),
        });
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn chunks_rust_functions_and_structs() {
        let source = "fn hello() { println!(\"hi\"); }\nfn world() {}\nstruct Foo { x: i32 }";
        let config = crate::languages::config_for_extension("rs").unwrap();
        let chunks = chunk_file(Path::new("test.rs"), source, &config);
        assert!(
            chunks.len() >= 2,
            "expected at least 2 chunks, got {}",
            chunks.len()
        );
        assert!(chunks.iter().any(|c| c.name == "hello"));
        assert!(chunks.iter().any(|c| c.name == "world"));
    }

    #[test]
    fn chunks_python_functions_and_classes() {
        let source = "def greet(name):\n    pass\n\nclass Foo:\n    pass\n";
        let config = crate::languages::config_for_extension("py").unwrap();
        let chunks = chunk_file(Path::new("test.py"), source, &config);
        assert!(chunks.len() >= 2);
        assert!(chunks.iter().any(|c| c.name == "greet"));
        assert!(chunks.iter().any(|c| c.name == "Foo"));
    }

    #[test]
    fn fallback_for_empty_query_matches() {
        let source = "let x = 42;\nconsole.log(x);\n";
        let config = crate::languages::config_for_extension("js").unwrap();
        let chunks = chunk_file(Path::new("script.js"), source, &config);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].kind, "file");
    }

    #[test]
    fn empty_file_produces_no_chunks() {
        let config = crate::languages::config_for_extension("rs").unwrap();
        let chunks = chunk_file(Path::new("empty.rs"), "", &config);
        assert!(chunks.is_empty());
    }
}
