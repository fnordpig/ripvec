//! Language registry mapping file extensions to tree-sitter grammars.
//!
//! Each supported language has a grammar and a tree-sitter query that
//! extracts function, class, and method definitions. Compiled queries
//! are cached so that repeated calls for the same extension are free.

use std::sync::{Arc, OnceLock};

use tree_sitter::{Language, Query};

/// Configuration for a supported source language.
///
/// Wrapped in [`Arc`] so it can be shared across threads and returned
/// from the cache without cloning the compiled [`Query`].
pub struct LangConfig {
    /// The tree-sitter Language grammar.
    pub language: Language,
    /// Query that extracts semantic chunks (`@def` captures with `@name`).
    pub query: Query,
}

/// Look up the language configuration for a file extension.
///
/// Compiled queries are cached per extension so repeated calls are free.
/// Returns `None` for unsupported extensions.
#[must_use]
pub fn config_for_extension(ext: &str) -> Option<Arc<LangConfig>> {
    // Cache of compiled configs, keyed by canonical extension.
    static CACHE: OnceLock<std::collections::HashMap<&'static str, Arc<LangConfig>>> =
        OnceLock::new();

    let cache = CACHE.get_or_init(|| {
        let mut m = std::collections::HashMap::new();
        // Pre-compile all supported extensions
        for &ext in &[
            "rs", "py", "js", "jsx", "ts", "tsx", "go", "java", "c", "h", "cpp", "cc", "cxx", "hpp",
        ] {
            if let Some(cfg) = compile_config(ext) {
                m.insert(ext, Arc::new(cfg));
            }
        }
        m
    });

    cache.get(ext).cloned()
}

/// Compile a [`LangConfig`] for the given extension (uncached).
fn compile_config(ext: &str) -> Option<LangConfig> {
    let (lang, query_str): (Language, &str) = match ext {
        // Rust: standalone functions, structs, and methods INSIDE impl/trait blocks.
        // impl_item and trait_item are NOT captured as wholes — we extract their
        // individual function_item children for method-level granularity.
        "rs" => (
            tree_sitter_rust::LANGUAGE.into(),
            concat!(
                "(function_item name: (identifier) @name) @def\n",
                "(struct_item name: (type_identifier) @name) @def\n",
                "(enum_item name: (type_identifier) @name) @def\n",
                "(type_item name: (type_identifier) @name) @def",
            ),
        ),
        // Python: top-level functions AND methods inside classes (function_definition
        // matches at any nesting depth, so methods are captured individually).
        "py" => (
            tree_sitter_python::LANGUAGE.into(),
            concat!(
                "(function_definition name: (identifier) @name) @def\n",
                "(class_definition name: (identifier) @name body: (block) @def)",
            ),
        ),
        // JS: functions, methods, and arrow functions assigned to variables.
        "js" | "jsx" => (
            tree_sitter_javascript::LANGUAGE.into(),
            concat!(
                "(function_declaration name: (identifier) @name) @def\n",
                "(method_definition name: (property_identifier) @name) @def\n",
                "(class_declaration name: (identifier) @name) @def",
            ),
        ),
        "ts" => (
            tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            concat!(
                "(function_declaration name: (identifier) @name) @def\n",
                "(method_definition name: (property_identifier) @name) @def\n",
                "(class_declaration name: (type_identifier) @name) @def\n",
                "(interface_declaration name: (type_identifier) @name) @def",
            ),
        ),
        "tsx" => (
            tree_sitter_typescript::LANGUAGE_TSX.into(),
            concat!(
                "(function_declaration name: (identifier) @name) @def\n",
                "(method_definition name: (property_identifier) @name) @def\n",
                "(class_declaration name: (type_identifier) @name) @def\n",
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
        // Java: methods are already captured individually (method_declaration
        // matches inside class bodies). Keep class for the signature/fields.
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
        // C++: functions at any level, plus class signatures.
        "cpp" | "cc" | "cxx" | "hpp" => (
            tree_sitter_cpp::LANGUAGE.into(),
            concat!(
                "(function_definition declarator: (function_declarator declarator: (identifier) @name)) @def\n",
                "(class_specifier name: (type_identifier) @name) @def",
            ),
        ),
        _ => return None,
    };
    let query = match Query::new(&lang, query_str) {
        Ok(q) => q,
        Err(e) => {
            tracing::warn!(ext, %e, "tree-sitter query compilation failed — language may be ABI-incompatible");
            return None;
        }
    };
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
