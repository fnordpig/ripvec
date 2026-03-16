//! Language registry mapping file extensions to tree-sitter grammars.
//!
//! Each supported language has a grammar and a tree-sitter query that
//! extracts function, class, and method definitions.

use tree_sitter::{Language, Query};

/// Configuration for a supported source language.
pub struct LangConfig {
    /// The tree-sitter Language grammar.
    pub language: Language,
    /// Query that extracts semantic chunks (`@def` captures with `@name`).
    pub query: Query,
}

/// Look up the language configuration for a file extension.
///
/// Returns `None` for unsupported extensions.
#[must_use]
pub fn config_for_extension(ext: &str) -> Option<LangConfig> {
    let (lang, query_str): (Language, &str) = match ext {
        "rs" => (
            tree_sitter_rust::LANGUAGE.into(),
            concat!(
                "(function_item name: (identifier) @name) @def\n",
                "(struct_item name: (type_identifier) @name) @def\n",
                "(impl_item) @def\n",
                "(trait_item name: (type_identifier) @name) @def",
            ),
        ),
        "py" => (
            tree_sitter_python::LANGUAGE.into(),
            concat!(
                "(function_definition name: (identifier) @name) @def\n",
                "(class_definition name: (identifier) @name) @def",
            ),
        ),
        "js" | "jsx" => (
            tree_sitter_javascript::LANGUAGE.into(),
            concat!(
                "(function_declaration name: (identifier) @name) @def\n",
                "(class_declaration name: (identifier) @name) @def\n",
                "(method_definition name: (property_identifier) @name) @def",
            ),
        ),
        "ts" => (
            tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            concat!(
                "(function_declaration name: (identifier) @name) @def\n",
                "(class_declaration name: (type_identifier) @name) @def\n",
                "(method_definition name: (property_identifier) @name) @def\n",
                "(interface_declaration name: (type_identifier) @name) @def",
            ),
        ),
        "tsx" => (
            tree_sitter_typescript::LANGUAGE_TSX.into(),
            concat!(
                "(function_declaration name: (identifier) @name) @def\n",
                "(class_declaration name: (type_identifier) @name) @def\n",
                "(method_definition name: (property_identifier) @name) @def\n",
                "(interface_declaration name: (type_identifier) @name) @def",
            ),
        ),
        "go" => (
            tree_sitter_go::LANGUAGE.into(),
            concat!(
                "(function_declaration name: (identifier) @name) @def\n",
                "(method_declaration name: (field_identifier) @name) @def",
            ),
        ),
        "java" => (
            tree_sitter_java::LANGUAGE.into(),
            concat!(
                "(method_declaration name: (identifier) @name) @def\n",
                "(class_declaration name: (identifier) @name) @def\n",
                "(interface_declaration name: (identifier) @name) @def",
            ),
        ),
        "c" | "h" => (
            tree_sitter_c::LANGUAGE.into(),
            "(function_definition declarator: (function_declarator declarator: (identifier) @name)) @def",
        ),
        "cpp" | "cc" | "cxx" | "hpp" => (
            tree_sitter_cpp::LANGUAGE.into(),
            concat!(
                "(function_definition declarator: (function_declarator declarator: (identifier) @name)) @def\n",
                "(class_specifier name: (type_identifier) @name) @def",
            ),
        ),
        _ => return None,
    };
    let query = Query::new(&lang, query_str).ok()?;
    Some(LangConfig {
        language: lang,
        query,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rust_extension_resolves() {
        assert!(config_for_extension("rs").is_some());
    }

    #[test]
    fn python_extension_resolves() {
        assert!(config_for_extension("py").is_some());
    }

    #[test]
    fn unknown_extension_returns_none() {
        assert!(config_for_extension("xyz").is_none());
    }

    #[test]
    fn all_supported_extensions() {
        let exts = [
            "rs", "py", "js", "jsx", "ts", "tsx", "go", "java", "c", "h", "cpp", "cc", "cxx", "hpp",
        ];
        for ext in &exts {
            assert!(config_for_extension(ext).is_some(), "failed for {ext}");
        }
    }
}
