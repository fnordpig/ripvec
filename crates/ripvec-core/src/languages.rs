//! Language registry mapping file extensions to tree-sitter grammars.
//!
//! Each supported language has a grammar and a tree-sitter query that
//! extracts function, class, and method definitions. Compiled queries
//! are cached so that repeated calls for the same extension are free.

use std::sync::{Arc, OnceLock};

use tree_sitter::{Language, Query};

/// Configuration for extracting function calls from a language.
///
/// Wrapped in [`Arc`] so it can be shared across threads and returned
/// from the cache without cloning the compiled [`Query`].
pub struct CallConfig {
    /// The tree-sitter Language grammar.
    pub language: Language,
    /// Query that extracts call sites (`@callee` captures).
    pub query: Query,
}

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
            "rs", "py", "pyi", "js", "jsx", "ts", "tsx", "go", "java", "c", "h", "cpp", "cc",
            "cxx", "hpp", "sh", "bash", "bats", "rb", "tf", "tfvars", "hcl", "kt", "kts", "swift",
            "scala", "toml", "json", "yaml", "yml", "md",
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
#[expect(
    clippy::too_many_lines,
    reason = "one match arm per language — flat by design"
)]
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
                "(type_item name: (type_identifier) @name) @def\n",
                "(field_declaration name: (field_identifier) @name) @def\n",
                "(enum_variant name: (identifier) @name) @def\n",
                "(impl_item type: (type_identifier) @name) @def\n",
                "(trait_item name: (type_identifier) @name) @def\n",
                "(const_item name: (identifier) @name) @def\n",
                "(static_item name: (identifier) @name) @def\n",
                "(mod_item name: (identifier) @name) @def",
            ),
        ),
        // Python: top-level functions AND methods inside classes (function_definition
        // matches at any nesting depth, so methods are captured individually).
        "py" | "pyi" => (
            tree_sitter_python::LANGUAGE.into(),
            concat!(
                "(function_definition name: (identifier) @name) @def\n",
                "(class_definition name: (identifier) @name body: (block) @def)\n",
                "(assignment left: (identifier) @name) @def",
            ),
        ),
        // JS: functions, methods, and arrow functions assigned to variables.
        "js" | "jsx" => (
            tree_sitter_javascript::LANGUAGE.into(),
            concat!(
                "(function_declaration name: (identifier) @name) @def\n",
                "(method_definition name: (property_identifier) @name) @def\n",
                "(class_declaration name: (identifier) @name) @def\n",
                "(variable_declarator name: (identifier) @name) @def",
            ),
        ),
        "ts" => (
            tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            concat!(
                "(function_declaration name: (identifier) @name) @def\n",
                "(method_definition name: (property_identifier) @name) @def\n",
                "(class_declaration name: (type_identifier) @name) @def\n",
                "(interface_declaration name: (type_identifier) @name) @def\n",
                "(variable_declarator name: (identifier) @name) @def\n",
                "(type_alias_declaration name: (type_identifier) @name) @def\n",
                "(enum_declaration name: (identifier) @name) @def",
            ),
        ),
        "tsx" => (
            tree_sitter_typescript::LANGUAGE_TSX.into(),
            concat!(
                "(function_declaration name: (identifier) @name) @def\n",
                "(method_definition name: (property_identifier) @name) @def\n",
                "(class_declaration name: (type_identifier) @name) @def\n",
                "(interface_declaration name: (type_identifier) @name) @def\n",
                "(variable_declarator name: (identifier) @name) @def\n",
                "(type_alias_declaration name: (type_identifier) @name) @def\n",
                "(enum_declaration name: (identifier) @name) @def",
            ),
        ),
        "go" => (
            tree_sitter_go::LANGUAGE.into(),
            concat!(
                "(function_declaration name: (identifier) @name) @def\n",
                "(method_declaration name: (field_identifier) @name) @def\n",
                "(type_declaration (type_spec name: (type_identifier) @name)) @def\n",
                "(const_spec name: (identifier) @name) @def",
            ),
        ),
        // Java: methods are already captured individually (method_declaration
        // matches inside class bodies). Keep class for the signature/fields.
        "java" => (
            tree_sitter_java::LANGUAGE.into(),
            concat!(
                "(method_declaration name: (identifier) @name) @def\n",
                "(class_declaration name: (identifier) @name) @def\n",
                "(interface_declaration name: (identifier) @name) @def\n",
                "(field_declaration declarator: (variable_declarator name: (identifier) @name)) @def\n",
                "(enum_constant name: (identifier) @name) @def\n",
                "(enum_declaration name: (identifier) @name) @def\n",
                "(constructor_declaration name: (identifier) @name) @def",
            ),
        ),
        "c" | "h" => (
            tree_sitter_c::LANGUAGE.into(),
            concat!(
                "(function_definition declarator: (function_declarator declarator: (identifier) @name)) @def\n",
                "(declaration declarator: (init_declarator declarator: (identifier) @name)) @def\n",
                "(struct_specifier name: (type_identifier) @name) @def\n",
                "(enum_specifier name: (type_identifier) @name) @def\n",
                "(type_definition declarator: (type_identifier) @name) @def",
            ),
        ),
        // C++: functions at any level, plus class signatures.
        "cpp" | "cc" | "cxx" | "hpp" => (
            tree_sitter_cpp::LANGUAGE.into(),
            concat!(
                "(function_definition declarator: (function_declarator declarator: (identifier) @name)) @def\n",
                "(class_specifier name: (type_identifier) @name) @def\n",
                "(declaration declarator: (init_declarator declarator: (identifier) @name)) @def\n",
                "(struct_specifier name: (type_identifier) @name) @def\n",
                "(enum_specifier name: (type_identifier) @name) @def\n",
                "(type_definition declarator: (type_identifier) @name) @def\n",
                "(namespace_definition name: (namespace_identifier) @name) @def\n",
                "(field_declaration declarator: (field_identifier) @name) @def",
            ),
        ),
        // Bash: function definitions (.bats = Bash Automated Testing System).
        "sh" | "bash" | "bats" => (
            tree_sitter_bash::LANGUAGE.into(),
            concat!(
                "(function_definition name: (word) @name) @def\n",
                "(variable_assignment name: (variable_name) @name) @def",
            ),
        ),
        // Ruby: methods, classes, and modules.
        "rb" => (
            tree_sitter_ruby::LANGUAGE.into(),
            concat!(
                "(method name: (identifier) @name) @def\n",
                "(class name: (constant) @name) @def\n",
                "(module name: (constant) @name) @def\n",
                "(assignment left: (identifier) @name) @def\n",
                "(assignment left: (constant) @name) @def",
            ),
        ),
        // HCL (Terraform): resource, data, variable, and output blocks.
        "tf" | "tfvars" | "hcl" => (
            tree_sitter_hcl::LANGUAGE.into(),
            "(block (identifier) @name) @def",
        ),
        // Kotlin: functions, classes, and objects.
        "kt" | "kts" => (
            tree_sitter_kotlin_ng::LANGUAGE.into(),
            concat!(
                "(function_declaration name: (identifier) @name) @def\n",
                "(class_declaration name: (identifier) @name) @def\n",
                "(object_declaration name: (identifier) @name) @def\n",
                "(property_declaration (identifier) @name) @def\n",
                "(enum_entry (identifier) @name) @def",
            ),
        ),
        // Swift: functions, classes, structs, enums, and protocols.
        "swift" => (
            tree_sitter_swift::LANGUAGE.into(),
            concat!(
                "(function_declaration name: (simple_identifier) @name) @def\n",
                "(class_declaration name: (type_identifier) @name) @def\n",
                "(protocol_declaration name: (type_identifier) @name) @def\n",
                "(property_declaration name: (pattern bound_identifier: (simple_identifier) @name)) @def\n",
                "(typealias_declaration name: (type_identifier) @name) @def",
            ),
        ),
        // Scala: functions, classes, traits, and objects.
        "scala" => (
            tree_sitter_scala::LANGUAGE.into(),
            concat!(
                "(function_definition name: (identifier) @name) @def\n",
                "(class_definition name: (identifier) @name) @def\n",
                "(trait_definition name: (identifier) @name) @def\n",
                "(object_definition name: (identifier) @name) @def\n",
                "(val_definition pattern: (identifier) @name) @def\n",
                "(var_definition pattern: (identifier) @name) @def\n",
                "(type_definition name: (type_identifier) @name) @def",
            ),
        ),
        // TOML: table headers (sections).
        "toml" => (
            tree_sitter_toml_ng::LANGUAGE.into(),
            concat!(
                "(table (bare_key) @name) @def\n",
                "(pair (bare_key) @name) @def",
            ),
        ),
        // JSON: key-value pairs, capturing the key string content.
        "json" => (
            tree_sitter_json::LANGUAGE.into(),
            "(pair key: (string (string_content) @name)) @def",
        ),
        // YAML: block mapping pairs with plain scalar keys.
        "yaml" | "yml" => (
            tree_sitter_yaml::LANGUAGE.into(),
            "(block_mapping_pair key: (flow_node (plain_scalar (string_scalar) @name))) @def",
        ),
        // Markdown: ATX headings (# through ######), capturing the heading text.
        "md" => (
            tree_sitter_md::LANGUAGE.into(),
            "(atx_heading heading_content: (inline) @name) @def",
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

/// Look up the call-extraction query for a file extension.
///
/// Compiled queries are cached per extension so repeated calls are free.
/// Returns `None` for unsupported extensions (including TOML, which has
/// no function calls).
#[must_use]
pub fn call_query_for_extension(ext: &str) -> Option<Arc<CallConfig>> {
    static CACHE: OnceLock<std::collections::HashMap<&'static str, Arc<CallConfig>>> =
        OnceLock::new();

    let cache = CACHE.get_or_init(|| {
        let mut m = std::collections::HashMap::new();
        // Pre-compile for all extensions that have callable constructs.
        // TOML is deliberately excluded — it has no function calls.
        for &ext in &[
            "rs", "py", "pyi", "js", "jsx", "ts", "tsx", "go", "java", "c", "h", "cpp", "cc",
            "cxx", "hpp", "sh", "bash", "bats", "rb", "tf", "tfvars", "hcl", "kt", "kts", "swift",
            "scala",
        ] {
            if let Some(cfg) = compile_call_config(ext) {
                m.insert(ext, Arc::new(cfg));
            }
        }
        m
    });

    cache.get(ext).cloned()
}

/// Compile a [`CallConfig`] for the given extension (uncached).
///
/// Each query extracts the callee identifier (`@callee`) from function
/// and method calls, plus the whole call expression (`@call`).
#[expect(
    clippy::too_many_lines,
    reason = "one match arm per language — flat by design"
)]
fn compile_call_config(ext: &str) -> Option<CallConfig> {
    let (lang, query_str): (Language, &str) = match ext {
        // Rust: free calls, method calls, and scoped (path) calls.
        "rs" => (
            tree_sitter_rust::LANGUAGE.into(),
            concat!(
                "(call_expression function: (identifier) @callee) @call\n",
                "(call_expression function: (field_expression field: (field_identifier) @callee)) @call\n",
                "(call_expression function: (scoped_identifier name: (identifier) @callee)) @call",
            ),
        ),
        // Python: simple calls and attribute (method) calls.
        "py" | "pyi" => (
            tree_sitter_python::LANGUAGE.into(),
            concat!(
                "(call function: (identifier) @callee) @call\n",
                "(call function: (attribute attribute: (identifier) @callee)) @call",
            ),
        ),
        // JavaScript: function calls and member expression calls.
        "js" | "jsx" => (
            tree_sitter_javascript::LANGUAGE.into(),
            concat!(
                "(call_expression function: (identifier) @callee) @call\n",
                "(call_expression function: (member_expression property: (property_identifier) @callee)) @call",
            ),
        ),
        // TypeScript: same patterns as JavaScript.
        "ts" => (
            tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            concat!(
                "(call_expression function: (identifier) @callee) @call\n",
                "(call_expression function: (member_expression property: (property_identifier) @callee)) @call",
            ),
        ),
        // TSX: same patterns as JavaScript.
        "tsx" => (
            tree_sitter_typescript::LANGUAGE_TSX.into(),
            concat!(
                "(call_expression function: (identifier) @callee) @call\n",
                "(call_expression function: (member_expression property: (property_identifier) @callee)) @call",
            ),
        ),
        // Go: function calls and selector (method) calls.
        "go" => (
            tree_sitter_go::LANGUAGE.into(),
            concat!(
                "(call_expression function: (identifier) @callee) @call\n",
                "(call_expression function: (selector_expression field: (field_identifier) @callee)) @call",
            ),
        ),
        // Java: method invocations.
        "java" => (
            tree_sitter_java::LANGUAGE.into(),
            "(method_invocation name: (identifier) @callee) @call",
        ),
        // C: function calls and field-expression calls (function pointers).
        "c" | "h" => (
            tree_sitter_c::LANGUAGE.into(),
            concat!(
                "(call_expression function: (identifier) @callee) @call\n",
                "(call_expression function: (field_expression field: (field_identifier) @callee)) @call",
            ),
        ),
        // C++: same patterns as C.
        "cpp" | "cc" | "cxx" | "hpp" => (
            tree_sitter_cpp::LANGUAGE.into(),
            concat!(
                "(call_expression function: (identifier) @callee) @call\n",
                "(call_expression function: (field_expression field: (field_identifier) @callee)) @call",
            ),
        ),
        // Bash: command invocations (.bats = Bash Automated Testing System).
        "sh" | "bash" | "bats" => (
            tree_sitter_bash::LANGUAGE.into(),
            "(command name: (command_name (word) @callee)) @call",
        ),
        // Ruby: method calls.
        "rb" => (
            tree_sitter_ruby::LANGUAGE.into(),
            "(call method: (identifier) @callee) @call",
        ),
        // HCL (Terraform): built-in function calls.
        "tf" | "tfvars" | "hcl" => (
            tree_sitter_hcl::LANGUAGE.into(),
            "(function_call (identifier) @callee) @call",
        ),
        // Kotlin: call expressions — grammar uses unnamed children, so match
        // identifier as first child of call_expression.
        "kt" | "kts" => (
            tree_sitter_kotlin_ng::LANGUAGE.into(),
            "(call_expression (identifier) @callee) @call",
        ),
        // Swift: call expressions with simple identifiers.
        "swift" => (
            tree_sitter_swift::LANGUAGE.into(),
            "(call_expression (simple_identifier) @callee) @call",
        ),
        // Scala: function calls and field-expression (method) calls.
        "scala" => (
            tree_sitter_scala::LANGUAGE.into(),
            concat!(
                "(call_expression function: (identifier) @callee) @call\n",
                "(call_expression function: (field_expression field: (identifier) @callee)) @call",
            ),
        ),
        _ => return None,
    };
    let query = match Query::new(&lang, query_str) {
        Ok(q) => q,
        Err(e) => {
            tracing::warn!(ext, %e, "tree-sitter call query compilation failed");
            return None;
        }
    };
    Some(CallConfig {
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
    fn python_stub_extension_resolves() {
        assert!(config_for_extension("pyi").is_some());
    }

    #[test]
    fn unknown_extension_returns_none() {
        assert!(config_for_extension("xyz").is_none());
    }

    #[test]
    fn all_supported_extensions() {
        let exts = [
            "rs", "py", "pyi", "js", "jsx", "ts", "tsx", "go", "java", "c", "h", "cpp", "cc",
            "cxx", "hpp", "sh", "bash", "bats", "rb", "tf", "tfvars", "hcl", "kt", "kts", "swift",
            "scala", "toml", "json", "yaml", "yml", "md",
        ];
        for ext in &exts {
            assert!(config_for_extension(ext).is_some(), "failed for {ext}");
        }
    }

    #[test]
    fn all_call_query_extensions() {
        let exts = [
            "rs", "py", "pyi", "js", "jsx", "ts", "tsx", "go", "java", "c", "h", "cpp", "cc",
            "cxx", "hpp", "sh", "bash", "bats", "rb", "tf", "tfvars", "hcl", "kt", "kts", "swift",
            "scala",
        ];
        for ext in &exts {
            assert!(
                call_query_for_extension(ext).is_some(),
                "call query failed for {ext}"
            );
        }
    }

    #[test]
    fn toml_has_no_call_query() {
        assert!(call_query_for_extension("toml").is_none());
    }
}
