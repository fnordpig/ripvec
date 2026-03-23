//! Tree-sitter based code chunking with sliding-window fallback.
//!
//! Parses source files into ASTs and extracts semantic chunks at
//! function, class, and method boundaries. For files without recognized
//! semantic structure (or very large fallback chunks), splits into
//! overlapping sliding windows for uniform embedding sizes.

use std::path::Path;
use streaming_iterator::StreamingIterator;
use tree_sitter::{Parser, QueryCursor};

/// Runtime configuration for the chunking pipeline.
///
/// All size values are in bytes. Pass to [`chunk_file`] to control
/// chunk sizing without recompilation.
#[derive(Debug, Clone)]
pub struct ChunkConfig {
    /// Maximum chunk content length in bytes before splitting into windows.
    /// Chunks larger than this are split even if tree-sitter found them as
    /// a single definition (e.g., a 500-line function).
    pub max_chunk_bytes: usize,
    /// Target window size in bytes for the sliding-window fallback chunker.
    /// ~2KB of source text ≈ 128-256 tokens after BPE, well within the
    /// 512-token model limit and optimal for embedding quality.
    pub window_size: usize,
    /// Overlap between adjacent windows in bytes.
    /// Ensures definitions spanning a window boundary are captured in at
    /// least one window. Defaults to 25% of `window_size`.
    pub window_overlap: usize,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            max_chunk_bytes: 4096,
            window_size: 2048,
            window_overlap: 512,
        }
    }
}

/// A semantic chunk extracted from a source file.
#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct CodeChunk {
    /// Path to the source file.
    pub file_path: String,
    /// Name of the definition (function name, class name, etc.).
    pub name: String,
    /// Kind of syntax node (e.g., `function_item`, `class_definition`).
    pub kind: String,
    /// 1-based start line number.
    pub start_line: usize,
    /// 1-based end line number.
    pub end_line: usize,
    /// Source text of the chunk (raw code for display).
    pub content: String,
    /// Enriched content with scope chain and signature metadata for embedding.
    /// Falls back to `content` if metadata would exceed chunk size limits.
    pub enriched_content: String,
}

/// Walk up the AST parent chain collecting structural container names.
///
/// Produces a scope chain like `"impl_item Foo > fn forward"` by
/// identifying structural containers (impl blocks, classes, modules, namespaces)
/// and extracting their names. Tries the `name` field first, then `type`
/// (for Rust `impl_item` which uses `type` instead of `name`).
#[must_use]
pub fn build_scope_chain(node: tree_sitter::Node<'_>, source: &str) -> String {
    /// Node kinds that represent structural containers, by language.
    const CONTAINER_KINDS: &[&str] = &[
        // Rust
        "impl_item",
        "trait_item",
        "mod_item",
        // Python
        "class_definition",
        "module",
        // JS/TS
        "class_declaration",
        // Java
        // "class_declaration" already covered above
        // Go
        "type_declaration",
        // C++
        "namespace_definition",
        "class_specifier",
    ];

    /// Field names to try when extracting the container name.
    /// `impl_item` uses `type` instead of `name`; Go `type_declaration`
    /// has no fields, so we fall back to the node kind.
    const NAME_FIELDS: &[&str] = &["name", "type"];

    let mut parts = Vec::new();
    let mut current = node.parent();
    while let Some(parent) = current {
        let kind = parent.kind();
        if CONTAINER_KINDS.contains(&kind) {
            let name = NAME_FIELDS
                .iter()
                .find_map(|field| parent.child_by_field_name(field))
                .map_or(kind, |n| &source[n.start_byte()..n.end_byte()]);
            parts.push(format!("{kind} {name}"));
        }
        current = parent.parent();
    }
    parts.reverse();
    parts.join(" > ")
}

/// Extract the function/method signature from a definition node.
///
/// Returns the text from the function name to the start of the body,
/// which captures the parameter list and return type (if any).
/// Returns `None` if the node has no `name` or `body`/`block` field.
#[must_use]
pub fn extract_signature(node: tree_sitter::Node<'_>, source: &str) -> Option<String> {
    let name_node = node.child_by_field_name("name")?;
    let body_node = node
        .child_by_field_name("body")
        .or_else(|| node.child_by_field_name("block"))?;
    let start = name_node.start_byte();
    let end = body_node.start_byte();
    if start >= end {
        return None;
    }
    let sig = source[start..end].trim();
    if sig.is_empty() {
        None
    } else {
        Some(sig.to_string())
    }
}

/// Build the enriched content header for a code chunk.
///
/// Prepends scope chain and signature metadata as a comment line.
/// If the header + content would exceed `max_bytes`, returns `content` unchanged.
fn build_enriched_content(
    path: &Path,
    node: tree_sitter::Node<'_>,
    source: &str,
    content: &str,
    max_bytes: usize,
) -> String {
    let scope = build_scope_chain(node, source);
    let sig = extract_signature(node, source).unwrap_or_default();
    let rel_path = path.display().to_string();

    let header = if scope.is_empty() && sig.is_empty() {
        format!("// {rel_path}\n")
    } else if scope.is_empty() {
        format!("// {rel_path} | defines: {sig}\n")
    } else if sig.is_empty() {
        format!("// {rel_path} | {scope}\n")
    } else {
        format!("// {rel_path} | {scope} | defines: {sig}\n")
    };

    if header.len() + content.len() > max_bytes {
        content.to_string()
    } else {
        format!("{header}{content}")
    }
}

/// Extract semantic chunks from a source file.
///
/// Uses tree-sitter to parse the file and extract definitions matching
/// the language's query patterns. For files with no semantic matches,
/// falls back to overlapping sliding windows. Large individual chunks
/// are also split into windows.
///
/// Pass a [`ChunkConfig`] to control chunk sizing at runtime.
#[must_use]
pub fn chunk_file(
    path: &Path,
    source: &str,
    config: &crate::languages::LangConfig,
    chunk_config: &ChunkConfig,
) -> Vec<CodeChunk> {
    let mut parser = Parser::new();
    if parser.set_language(&config.language).is_err() {
        return sliding_windows(path, source, chunk_config);
    }

    let Some(tree) = parser.parse(source, None) else {
        return sliding_windows(path, source, chunk_config);
    };

    let mut cursor = QueryCursor::new();
    let mut chunks = Vec::new();
    let mut matches = cursor.matches(&config.query, tree.root_node(), source.as_bytes());

    while let Some(m) = matches.next() {
        let mut name = String::new();
        let mut def_node = None;
        for cap in m.captures {
            let cap_name = &config.query.capture_names()[cap.index as usize];
            if *cap_name == "name" {
                name = source[cap.node.start_byte()..cap.node.end_byte()].to_string();
            } else if *cap_name == "def" {
                def_node = Some(cap.node);
            }
        }
        if let Some(node) = def_node {
            let content = &source[node.start_byte()..node.end_byte()];
            let start_line = node.start_position().row + 1;

            // Split oversized chunks into windows
            if content.len() > chunk_config.max_chunk_bytes {
                chunks.extend(sliding_windows_with_name(
                    path,
                    content,
                    &name,
                    start_line,
                    chunk_config,
                ));
            } else {
                let enriched = build_enriched_content(
                    path,
                    node,
                    source,
                    content,
                    chunk_config.max_chunk_bytes,
                );
                chunks.push(CodeChunk {
                    file_path: path.display().to_string(),
                    name,
                    kind: node.kind().to_string(),
                    start_line,
                    end_line: node.end_position().row + 1,
                    enriched_content: enriched,
                    content: content.to_string(),
                });
            }
        }
    }

    // Fallback: sliding windows if no semantic matches
    if chunks.is_empty() && !source.trim().is_empty() {
        return sliding_windows(path, source, chunk_config);
    }

    chunks
}

/// Split source text into overlapping sliding windows.
///
/// Each window is `chunk_config.window_size` bytes with `chunk_config.window_overlap` bytes of
/// overlap. Window boundaries are adjusted to line breaks to avoid
/// splitting mid-line.
///
/// This is used as the fallback for files without tree-sitter support
/// (plain text, unknown extensions) and for large semantic chunks that
/// exceed `max_chunk_bytes`.
#[must_use]
pub fn chunk_text(path: &Path, source: &str, chunk_config: &ChunkConfig) -> Vec<CodeChunk> {
    sliding_windows(path, source, chunk_config)
}

/// Internal sliding-window implementation.
fn sliding_windows(path: &Path, source: &str, chunk_config: &ChunkConfig) -> Vec<CodeChunk> {
    if source.trim().is_empty() {
        return vec![];
    }

    // Small enough for a single chunk
    if source.len() <= chunk_config.max_chunk_bytes {
        let content = source.to_string();
        return vec![CodeChunk {
            file_path: path.display().to_string(),
            name: path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string(),
            kind: "file".to_string(),
            start_line: 1,
            end_line: source.lines().count(),
            enriched_content: content.clone(),
            content,
        }];
    }

    let file_name = path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    sliding_window_chunks(source, path, &file_name, 1, chunk_config)
}

/// Split a named definition into overlapping windows.
///
/// Used when a single tree-sitter match (e.g., a large function) exceeds
/// `chunk_config.max_chunk_bytes`. Windows carry the definition name for search context.
fn sliding_windows_with_name(
    path: &Path,
    content: &str,
    name: &str,
    base_line: usize,
    chunk_config: &ChunkConfig,
) -> Vec<CodeChunk> {
    sliding_window_chunks(content, path, name, base_line, chunk_config)
}

/// Shared sliding-window loop used by both [`sliding_windows`] and
/// [`sliding_windows_with_name`].
///
/// Splits `source` into overlapping windows of `chunk_config.window_size` bytes,
/// snapping boundaries to line breaks. Each chunk is tagged with `name_prefix`
/// and an index suffix (e.g., `"main[0]"`, `"main[1]"`).
fn sliding_window_chunks(
    source: &str,
    file_path: &Path,
    name_prefix: &str,
    base_line: usize,
    chunk_config: &ChunkConfig,
) -> Vec<CodeChunk> {
    let step = chunk_config
        .window_size
        .saturating_sub(chunk_config.window_overlap)
        .max(1);
    let bytes = source.as_bytes();
    let mut chunks = Vec::new();
    let mut offset = 0;
    let mut window_idx = 0;

    while offset < bytes.len() {
        let raw_end = (offset + chunk_config.window_size).min(bytes.len());

        // Snap end to a line boundary (don't split mid-line)
        let end = if raw_end < bytes.len() {
            match bytes[offset..raw_end].iter().rposition(|&b| b == b'\n') {
                Some(pos) => offset + pos + 1,
                None => raw_end, // no newline found, use raw end
            }
        } else {
            raw_end
        };

        // Extract window as str (skip invalid UTF-8)
        if let Ok(window) = std::str::from_utf8(&bytes[offset..end])
            && !window.trim().is_empty()
        {
            let start_line = base_line + source[..offset].matches('\n').count();
            let content_lines = window.lines().count().max(1);
            let end_line = start_line + content_lines - 1;
            let content = window.to_string();
            chunks.push(CodeChunk {
                file_path: file_path.display().to_string(),
                name: format!("{name_prefix}[{window_idx}]"),
                kind: "window".to_string(),
                start_line,
                end_line,
                enriched_content: content.clone(),
                content,
            });
            window_idx += 1;
        }

        offset += step;
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Write as _;
    use std::path::Path;

    #[test]
    fn chunks_rust_functions_and_structs() {
        let source = "fn hello() { println!(\"hi\"); }\nfn world() {}\nstruct Foo { x: i32 }";
        let config = crate::languages::config_for_extension("rs").unwrap();
        let chunks = chunk_file(
            Path::new("test.rs"),
            source,
            &config,
            &ChunkConfig::default(),
        );
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
        let chunks = chunk_file(
            Path::new("test.py"),
            source,
            &config,
            &ChunkConfig::default(),
        );
        assert!(chunks.len() >= 2);
        assert!(chunks.iter().any(|c| c.name == "greet"));
        assert!(chunks.iter().any(|c| c.name == "Foo"));
    }

    #[test]
    fn fallback_small_file_single_chunk() {
        let source = "let x = 42;\nconsole.log(x);\n";
        let config = crate::languages::config_for_extension("js").unwrap();
        let chunks = chunk_file(
            Path::new("script.js"),
            source,
            &config,
            &ChunkConfig::default(),
        );
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].kind, "file");
    }

    #[test]
    fn fallback_large_file_produces_windows() {
        // Create a file larger than default max_chunk_bytes with no function declarations
        let line = "console.log('hello world, this is a long line of javascript code');\n";
        let source: String = line.repeat(200); // ~13KB
        let chunk_config = ChunkConfig::default();
        assert!(source.len() > chunk_config.max_chunk_bytes);

        let config = crate::languages::config_for_extension("js").unwrap();
        let chunks = chunk_file(Path::new("big.js"), &source, &config, &chunk_config);
        assert!(
            chunks.len() > 1,
            "expected multiple windows, got {}",
            chunks.len()
        );
        assert!(chunks.iter().all(|c| c.kind == "window"));
        assert!(chunks[0].name.contains("[0]"));
    }

    #[test]
    fn large_definition_is_windowed() {
        // A Rust function larger than default max_chunk_bytes
        let mut source = String::from("fn big_function() {\n");
        for i in 0..200 {
            writeln!(source, "    let var_{i} = {i} * 2 + 1; // some computation").unwrap();
        }
        source.push_str("}\n");
        let chunk_config = ChunkConfig::default();
        assert!(source.len() > chunk_config.max_chunk_bytes);

        let config = crate::languages::config_for_extension("rs").unwrap();
        let chunks = chunk_file(Path::new("test.rs"), &source, &config, &chunk_config);
        assert!(
            chunks.len() > 1,
            "expected windowed chunks, got {}",
            chunks.len()
        );
        assert!(chunks[0].name.starts_with("big_function["));
    }

    #[test]
    fn empty_file_produces_no_chunks() {
        let config = crate::languages::config_for_extension("rs").unwrap();
        let chunks = chunk_file(Path::new("empty.rs"), "", &config, &ChunkConfig::default());
        assert!(chunks.is_empty());
    }

    // --- T1 enrichment tests ---

    /// Helper: parse source with tree-sitter and return the first `@def` node.
    fn first_def_node(
        source: &str,
        ext: &str,
    ) -> (
        tree_sitter::Tree,
        std::sync::Arc<crate::languages::LangConfig>,
    ) {
        let config = crate::languages::config_for_extension(ext).unwrap();
        let mut parser = Parser::new();
        parser.set_language(&config.language).unwrap();
        let tree = parser.parse(source, None).unwrap();
        (tree, config)
    }

    #[test]
    fn scope_chain_rust_impl_method() {
        let source = "impl Foo {\n    fn bar(&self) {}\n}";
        let (tree, config) = first_def_node(source, "rs");
        let mut cursor = QueryCursor::new();
        let mut matches = cursor.matches(&config.query, tree.root_node(), source.as_bytes());

        let mut def_node = None;
        while let Some(m) = StreamingIterator::next(&mut matches) {
            for cap in m.captures {
                let cap_name = &config.query.capture_names()[cap.index as usize];
                if *cap_name == "def" {
                    def_node = Some(cap.node);
                }
            }
        }
        let node = def_node.expect("should find a @def node");
        let scope = build_scope_chain(node, source);
        assert!(
            scope.contains("impl_item"),
            "scope should contain impl_item, got: {scope}"
        );
        assert!(
            scope.contains("Foo"),
            "scope should contain 'Foo', got: {scope}"
        );
    }

    #[test]
    fn scope_chain_python_class_method() {
        let source = "class Greeter:\n    def say_hello(self):\n        pass\n";
        let (tree, config) = first_def_node(source, "py");
        let mut cursor = QueryCursor::new();
        let mut matches = cursor.matches(&config.query, tree.root_node(), source.as_bytes());

        // Find the function_definition @def (say_hello), not the class @def
        let mut fn_node = None;
        while let Some(m) = StreamingIterator::next(&mut matches) {
            for cap in m.captures {
                let cap_name = &config.query.capture_names()[cap.index as usize];
                if *cap_name == "def" && cap.node.kind() == "function_definition" {
                    fn_node = Some(cap.node);
                }
            }
        }
        let node = fn_node.expect("should find say_hello @def node");
        let scope = build_scope_chain(node, source);
        assert!(
            scope.contains("class_definition"),
            "scope should contain class_definition, got: {scope}"
        );
        assert!(
            scope.contains("Greeter"),
            "scope should contain 'Greeter', got: {scope}"
        );
    }

    #[test]
    fn extract_signature_rust_function() {
        let source = "fn greet(name: &str) -> String { name.to_string() }";
        let (tree, config) = first_def_node(source, "rs");
        let mut cursor = QueryCursor::new();
        let mut matches = cursor.matches(&config.query, tree.root_node(), source.as_bytes());

        let mut def_node = None;
        while let Some(m) = StreamingIterator::next(&mut matches) {
            for cap in m.captures {
                let cap_name = &config.query.capture_names()[cap.index as usize];
                if *cap_name == "def" {
                    def_node = Some(cap.node);
                }
            }
        }
        let node = def_node.expect("should find @def node");
        let sig = extract_signature(node, source).expect("should extract signature");
        assert!(
            sig.contains("greet"),
            "signature should contain 'greet', got: {sig}"
        );
        assert!(
            sig.contains("name: &str"),
            "signature should contain parameter, got: {sig}"
        );
        assert!(
            sig.contains("-> String"),
            "signature should contain return type, got: {sig}"
        );
    }

    #[test]
    fn enriched_content_has_header() {
        let source = "fn hello() { println!(\"hi\"); }";
        let config = crate::languages::config_for_extension("rs").unwrap();
        let chunks = chunk_file(
            Path::new("src/main.rs"),
            source,
            &config,
            &ChunkConfig::default(),
        );
        assert!(!chunks.is_empty());
        let chunk = &chunks[0];
        assert!(
            chunk.enriched_content.starts_with("//"),
            "enriched_content should start with '//' header, got: {}",
            &chunk.enriched_content[..chunk.enriched_content.len().min(80)]
        );
        assert!(
            chunk.enriched_content.contains("src/main.rs"),
            "enriched_content should contain file path"
        );
        // Raw content should NOT have the header
        assert!(
            !chunk.content.starts_with("//"),
            "raw content should not start with header"
        );
    }

    #[test]
    fn sliding_window_enriched_equals_content() {
        let source = "let x = 42;\nconsole.log(x);\n";
        let chunks = chunk_text(Path::new("test.txt"), source, &ChunkConfig::default());
        assert!(!chunks.is_empty());
        for chunk in &chunks {
            assert_eq!(
                chunk.enriched_content, chunk.content,
                "sliding window chunks should have enriched_content == content"
            );
        }
    }

    #[test]
    fn header_dropped_when_exceeding_max_bytes() {
        // Create a chunk that barely fits in max_chunk_bytes, so adding
        // a header would push it over the limit.
        let tiny_config = ChunkConfig {
            max_chunk_bytes: 60,
            window_size: 30,
            window_overlap: 10,
        };
        // Source is exactly at max_chunk_bytes — any header would exceed it
        let source = "fn f() { let x = 42; return x; }";
        assert!(source.len() <= tiny_config.max_chunk_bytes);

        let config = crate::languages::config_for_extension("rs").unwrap();
        let chunks = chunk_file(
            Path::new("long/path/to/file.rs"),
            source,
            &config,
            &tiny_config,
        );
        assert!(!chunks.is_empty());
        let chunk = &chunks[0];
        // Header ("// long/path/to/file.rs | defines: ...") + content > 60 bytes
        // So enriched_content should fall back to raw content
        assert_eq!(
            chunk.enriched_content, chunk.content,
            "header should be dropped when it would exceed max_chunk_bytes"
        );
    }
}
