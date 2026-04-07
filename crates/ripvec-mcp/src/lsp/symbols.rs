//! Document and workspace symbol providers for the LSP server.
//!
//! Uses tree-sitter via `ripvec_core::chunk::chunk_file` to extract
//! semantic code chunks and maps them to LSP `DocumentSymbol` entries.

use tower_lsp_server::jsonrpc::Result;
use tower_lsp_server::ls_types::{
    DocumentSymbol, DocumentSymbolParams, DocumentSymbolResponse, Position, Range, SymbolKind,
    WorkspaceSymbolParams, WorkspaceSymbolResponse,
};

use ripvec_core::chunk::{ChunkConfig, chunk_file};
use ripvec_core::languages::config_for_extension;

/// Map a tree-sitter node kind string to an LSP `SymbolKind`.
///
/// Explicit function kinds are listed for documentation even though the
/// catch-all also returns `FUNCTION` — this keeps the mapping auditable
/// when new languages are added.
#[expect(
    clippy::match_same_arms,
    reason = "explicit function kinds listed for documentation/auditability"
)]
fn symbol_kind_for(kind: &str) -> SymbolKind {
    match kind {
        // Functions (Rust, Python, JS/TS, Go, Scala, Bash, Kotlin, Swift, C/C++)
        "function_item" | "function_definition" | "function_declaration" => SymbolKind::FUNCTION,

        // Methods (JS/TS, Go, Java, Ruby)
        "method_definition" | "method_declaration" | "method" => SymbolKind::METHOD,

        // Structs (Rust)
        "struct_item" => SymbolKind::STRUCT,

        // Enums (Rust)
        "enum_item" => SymbolKind::ENUM,

        // Classes (Rust type alias, Python, JS/TS, Java, C++, Ruby)
        "type_item" | "class_definition" | "class_declaration" | "class_specifier" | "class" => {
            SymbolKind::CLASS
        }

        // Interfaces (TS, Java, Swift protocol, Scala trait)
        "interface_declaration" | "protocol_declaration" | "trait_definition" => {
            SymbolKind::INTERFACE
        }

        // Objects (Kotlin, Scala)
        "object_declaration" | "object_definition" => SymbolKind::OBJECT,

        // Modules (Ruby)
        "module" => SymbolKind::MODULE,

        // HCL / TOML blocks
        "block" | "table" => SymbolKind::NAMESPACE,

        // Catch-all: default to FUNCTION for unknown node kinds
        _ => SymbolKind::FUNCTION,
    }
}

/// Return document symbols (outline) for a single file.
///
/// Reads the file from disk, runs tree-sitter chunking, and maps each
/// semantic chunk to a `DocumentSymbol`. Fallback text chunks (kind
/// "window" or "file") are excluded since they don't represent named
/// code constructs.
pub async fn document_symbol(
    params: DocumentSymbolParams,
) -> Result<Option<DocumentSymbolResponse>> {
    let uri = &params.text_document.uri;

    let Some(path_cow) = uri.to_file_path() else {
        return Ok(None);
    };
    let path = path_cow.into_owned();

    let Ok(source) = tokio::fs::read_to_string(&path).await else {
        return Ok(None);
    };

    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default()
        .to_string();

    let Some(config) = config_for_extension(&ext) else {
        return Ok(None);
    };

    let chunks = chunk_file(&path, &source, &config, &ChunkConfig::default());

    let symbols: Vec<DocumentSymbol> = chunks
        .into_iter()
        .filter(|c| c.kind != "window" && c.kind != "file")
        .map(|c| chunk_to_symbol(&c))
        .collect();

    if symbols.is_empty() {
        Ok(None)
    } else {
        Ok(Some(DocumentSymbolResponse::Nested(symbols)))
    }
}

/// Saturating conversion from `usize` to `u32` for LSP line/column numbers.
///
/// Files larger than 4 GiB of lines are implausible; clamp to `u32::MAX`.
fn line_to_u32(n: usize) -> u32 {
    u32::try_from(n).unwrap_or(u32::MAX)
}

/// Convert a `CodeChunk` to a `DocumentSymbol`.
///
/// Chunk lines are 1-based; LSP positions are 0-based.
#[expect(
    deprecated,
    reason = "DocumentSymbol has a deprecated `deprecated` field"
)]
fn chunk_to_symbol(chunk: &ripvec_core::chunk::CodeChunk) -> DocumentSymbol {
    let start_line = line_to_u32(chunk.start_line.saturating_sub(1));
    let end_line = line_to_u32(chunk.end_line.saturating_sub(1));

    let range = Range {
        start: Position {
            line: start_line,
            character: 0,
        },
        end: Position {
            line: end_line,
            character: 0,
        },
    };

    // Selection range covers just the first line (the name/signature).
    let selection_range = Range {
        start: Position {
            line: start_line,
            character: 0,
        },
        end: Position {
            line: start_line,
            character: 0,
        },
    };

    DocumentSymbol {
        name: chunk.name.clone(),
        detail: None,
        kind: symbol_kind_for(&chunk.kind),
        tags: None,
        deprecated: None,
        range,
        selection_range,
        children: None,
    }
}

/// Search for symbols across the entire workspace.
/// Currently a stub that returns `Ok(None)`.
#[expect(clippy::unused_async, reason = "will use async once implemented")]
pub async fn workspace_symbol(
    _params: WorkspaceSymbolParams,
) -> Result<Option<WorkspaceSymbolResponse>> {
    Ok(None)
}
