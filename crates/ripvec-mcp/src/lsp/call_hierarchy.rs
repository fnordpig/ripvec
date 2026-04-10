//! Call hierarchy provider for the LSP server.
//!
//! Backed by the definition-level call graph in [`RepoGraph`]. The `prepare`
//! handler finds the definition at the cursor position; `incoming` and
//! `outgoing` traverse `def_callers` / `def_callees` respectively.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::json;
use tower_lsp_server::jsonrpc::Result;
use tower_lsp_server::ls_types::{
    CallHierarchyIncomingCall, CallHierarchyIncomingCallsParams, CallHierarchyItem,
    CallHierarchyOutgoingCall, CallHierarchyOutgoingCallsParams, CallHierarchyPrepareParams,
    Position, Range, Uri,
};

use ripvec_core::repo_map::{Definition, RepoGraph};

use super::symbols::symbol_kind_for;

/// Saturating conversion from `u32` to LSP 0-based line number.
///
/// Definition line numbers are 1-based; LSP positions are 0-based.
fn def_line_to_lsp(one_based: u32) -> u32 {
    one_based.saturating_sub(1)
}

/// Build an LSP `Range` from a [`Definition`]'s 1-based line numbers.
fn def_range(def: &Definition) -> Range {
    Range {
        start: Position {
            line: def_line_to_lsp(def.start_line),
            character: 0,
        },
        end: Position {
            line: def_line_to_lsp(def.end_line),
            character: 0,
        },
    }
}

/// Build a `Uri` from a `FileNode` path, resolving against `root` if relative.
fn uri_for_file(file_path: &str, root: &Path) -> Option<Uri> {
    let path = Path::new(file_path);
    let abs = if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    };
    Uri::from_file_path(abs)
}

/// Build a [`CallHierarchyItem`] from a graph definition.
///
/// Stores `file_idx` and `def_idx` in the `data` field so that
/// `incoming` / `outgoing` can recover them without re-searching.
fn def_to_item(
    graph: &RepoGraph,
    file_idx: usize,
    def_idx: usize,
    root: &Path,
) -> Option<CallHierarchyItem> {
    let file = graph.files.get(file_idx)?;
    let def = file.defs.get(def_idx)?;
    let uri = uri_for_file(&file.path, root)?;

    let range = def_range(def);
    let selection_range = Range {
        start: range.start,
        end: Position {
            line: range.start.line,
            character: 0,
        },
    };

    Some(CallHierarchyItem {
        name: def.name.clone(),
        kind: symbol_kind_for(&def.kind),
        tags: None,
        detail: def.signature.clone(),
        uri,
        range,
        selection_range,
        data: Some(json!({
            "file_idx": file_idx,
            "def_idx": def_idx,
            "root": root.to_string_lossy(),
        })),
    })
}

/// Find the definition at cursor in a single graph, returning the item if found.
fn prepare_in_graph(
    graph: &RepoGraph,
    req_path: &Path,
    search_root: &Path,
    cursor_line: u32,
) -> Option<CallHierarchyItem> {
    let rel_path = make_relative(req_path, search_root);
    let file_idx = graph
        .files
        .iter()
        .position(|f| Path::new(&f.path) == rel_path)?;

    let file = &graph.files[file_idx];
    let mut best: Option<(usize, u32)> = None;
    for (di, def) in file.defs.iter().enumerate() {
        let start = def_line_to_lsp(def.start_line);
        let end = def_line_to_lsp(def.end_line);
        if cursor_line >= start && cursor_line <= end {
            let span = end - start;
            if best.is_none() || span < best.unwrap().1 {
                best = Some((di, span));
            }
        }
    }

    let (def_idx, _) = best?;
    def_to_item(graph, file_idx, def_idx, search_root)
}

/// Find the definition at the cursor position and return it as a
/// [`CallHierarchyItem`].
///
/// Searches definitions in the file matching the cursor's URI for one
/// whose line range contains the cursor line. Tries the default repo
/// graph first, then on-demand root graphs.
#[expect(
    clippy::unused_async,
    reason = "LanguageServer trait requires async signature"
)]
pub async fn prepare(
    params: CallHierarchyPrepareParams,
    repo_graph: &Arc<std::sync::RwLock<Option<RepoGraph>>>,
    root_graphs: &Arc<std::sync::RwLock<HashMap<PathBuf, RepoGraph>>>,
    root: &Path,
) -> Result<Option<Vec<CallHierarchyItem>>> {
    let pos = &params.text_document_position_params;
    let Some(path_cow) = pos.text_document.uri.to_file_path() else {
        return Ok(None);
    };
    let req_path = path_cow.into_owned();
    let cursor_line = pos.position.line; // 0-based

    // Try the default repo graph first.
    if let Ok(graph_guard) = repo_graph.read()
        && let Some(graph) = graph_guard.as_ref()
        && let Some(item) = prepare_in_graph(graph, &req_path, root, cursor_line)
    {
        return Ok(Some(vec![item]));
    }

    // Fall back to on-demand root graphs.
    if let Ok(rg_guard) = root_graphs.read() {
        for (alt_root, graph) in rg_guard.iter() {
            if req_path.starts_with(alt_root)
                && let Some(item) = prepare_in_graph(graph, &req_path, alt_root, cursor_line)
            {
                return Ok(Some(vec![item]));
            }
        }
    }

    // Fall back: build graph on demand from disk for the file's project root.
    let mut candidate = req_path.parent();
    while let Some(dir) = candidate {
        if dir.join(".git").exists() || dir.join(".ripvec").exists() {
            if let Ok(graph) = ripvec_core::repo_map::build_graph(dir)
                && let Some(item) = prepare_in_graph(&graph, &req_path, dir, cursor_line)
            {
                return Ok(Some(vec![item]));
            }
            break;
        }
        candidate = dir.parent();
    }

    Ok(None)
}

/// Collect incoming calls from a single graph.
fn incoming_from_graph(
    graph: &RepoGraph,
    file_idx: usize,
    def_idx: usize,
    search_root: &Path,
) -> Vec<CallHierarchyIncomingCall> {
    let Some(flat) = flat_index(graph, file_idx, def_idx) else {
        return Vec::new();
    };
    let Some(callers) = graph.def_callers.get(flat) else {
        return Vec::new();
    };

    let mut results = Vec::with_capacity(callers.len());
    for &(caller_file, caller_def) in callers {
        let caller_file = caller_file as usize;
        let caller_def = caller_def as usize;

        let Some(item) = def_to_item(graph, caller_file, caller_def, search_root) else {
            continue;
        };

        let from_ranges = call_site_ranges(graph, caller_file, caller_def, file_idx, def_idx);
        results.push(CallHierarchyIncomingCall {
            from: item,
            from_ranges,
        });
    }
    results
}

/// Return callers of the given definition.
///
/// Reads `file_idx` / `def_idx` from the item's `data` field, computes
/// the flat index, and maps each entry in `graph.def_callers[flat]` to a
/// [`CallHierarchyIncomingCall`].
///
/// Tries the default repo graph first; falls back to on-demand root graphs.
#[expect(
    clippy::unused_async,
    reason = "LanguageServer trait requires async signature"
)]
pub async fn incoming(
    params: CallHierarchyIncomingCallsParams,
    repo_graph: &Arc<std::sync::RwLock<Option<RepoGraph>>>,
    root_graphs: &Arc<std::sync::RwLock<HashMap<PathBuf, RepoGraph>>>,
    root: &Path,
) -> Result<Option<Vec<CallHierarchyIncomingCall>>> {
    let Some((file_idx, def_idx)) = extract_ids(&params.item) else {
        return Ok(None);
    };

    // Try the default repo graph first.
    if let Ok(graph_guard) = repo_graph.read()
        && let Some(graph) = graph_guard.as_ref()
    {
        let results = incoming_from_graph(graph, file_idx, def_idx, root);
        if !results.is_empty() {
            return Ok(Some(results));
        }
    }

    // Fall back to on-demand root graphs.
    if let Ok(rg_guard) = root_graphs.read() {
        for (alt_root, graph) in rg_guard.iter() {
            let results = incoming_from_graph(graph, file_idx, def_idx, alt_root);
            if !results.is_empty() {
                return Ok(Some(results));
            }
        }
    }

    // Fall back: rebuild graph from disk using root stored in data.
    if let Some(graph_root) = extract_root(&params.item)
        && let Ok(graph) = ripvec_core::repo_map::build_graph(&graph_root)
    {
        let results = incoming_from_graph(&graph, file_idx, def_idx, &graph_root);
        if !results.is_empty() {
            return Ok(Some(results));
        }
    }

    Ok(None)
}

/// Collect outgoing calls from a single graph.
fn outgoing_from_graph(
    graph: &RepoGraph,
    file_idx: usize,
    def_idx: usize,
    search_root: &Path,
) -> Vec<CallHierarchyOutgoingCall> {
    let Some(flat) = flat_index(graph, file_idx, def_idx) else {
        return Vec::new();
    };
    let Some(callees) = graph.def_callees.get(flat) else {
        return Vec::new();
    };

    let mut results = Vec::with_capacity(callees.len());
    for &(callee_file, callee_def) in callees {
        let callee_file = callee_file as usize;
        let callee_def = callee_def as usize;

        let Some(item) = def_to_item(graph, callee_file, callee_def, search_root) else {
            continue;
        };

        let from_ranges = call_site_ranges(graph, file_idx, def_idx, callee_file, callee_def);
        results.push(CallHierarchyOutgoingCall {
            to: item,
            from_ranges,
        });
    }
    results
}

/// Return callees of the given definition.
///
/// Same as `incoming` but traverses `graph.def_callees[flat]` and returns
/// [`CallHierarchyOutgoingCall`] entries.
///
/// Tries the default repo graph first; falls back to on-demand root graphs.
#[expect(
    clippy::unused_async,
    reason = "LanguageServer trait requires async signature"
)]
pub async fn outgoing(
    params: CallHierarchyOutgoingCallsParams,
    repo_graph: &Arc<std::sync::RwLock<Option<RepoGraph>>>,
    root_graphs: &Arc<std::sync::RwLock<HashMap<PathBuf, RepoGraph>>>,
    root: &Path,
) -> Result<Option<Vec<CallHierarchyOutgoingCall>>> {
    let Some((file_idx, def_idx)) = extract_ids(&params.item) else {
        return Ok(None);
    };

    // Try the default repo graph first.
    if let Ok(graph_guard) = repo_graph.read()
        && let Some(graph) = graph_guard.as_ref()
    {
        let results = outgoing_from_graph(graph, file_idx, def_idx, root);
        if !results.is_empty() {
            return Ok(Some(results));
        }
    }

    // Fall back to on-demand root graphs.
    if let Ok(rg_guard) = root_graphs.read() {
        for (alt_root, graph) in rg_guard.iter() {
            let results = outgoing_from_graph(graph, file_idx, def_idx, alt_root);
            if !results.is_empty() {
                return Ok(Some(results));
            }
        }
    }

    // Fall back: rebuild graph from disk using root stored in data.
    if let Some(graph_root) = extract_root(&params.item)
        && let Ok(graph) = ripvec_core::repo_map::build_graph(&graph_root)
    {
        let results = outgoing_from_graph(&graph, file_idx, def_idx, &graph_root);
        if !results.is_empty() {
            return Ok(Some(results));
        }
    }

    Ok(None)
}

// ── Helpers ────────────────────────────────────────────────────────────

/// Extract `(file_idx, def_idx)` from a `CallHierarchyItem`'s JSON `data`.
fn extract_ids(item: &CallHierarchyItem) -> Option<(usize, usize)> {
    let data = item.data.as_ref()?;
    let file_idx = usize::try_from(data.get("file_idx")?.as_u64()?).ok()?;
    let def_idx = usize::try_from(data.get("def_idx")?.as_u64()?).ok()?;
    Some((file_idx, def_idx))
}

/// Extract the project root from the item's `data` field.
fn extract_root(item: &CallHierarchyItem) -> Option<PathBuf> {
    let data = item.data.as_ref()?;
    let root_str = data.get("root")?.as_str()?;
    Some(PathBuf::from(root_str))
}

/// Compute the flat index from `(file_idx, def_idx)`.
fn flat_index(graph: &RepoGraph, file_idx: usize, def_idx: usize) -> Option<usize> {
    let offset = *graph.def_offsets.get(file_idx)?;
    let flat = offset + def_idx;
    // Bounds-check against the flattened arrays.
    if flat < graph.def_callers.len() {
        Some(flat)
    } else {
        None
    }
}

/// Make `abs_path` relative to `root`, falling back to the full path.
fn make_relative<'a>(abs_path: &'a Path, root: &Path) -> &'a Path {
    abs_path.strip_prefix(root).unwrap_or(abs_path)
}

/// Find call-site ranges within `caller` that target `(target_file, target_def)`.
///
/// Scans the caller definition's `calls` for entries whose `resolved` field
/// matches the target. Each matching call site is mapped to a zero-width
/// range at the call's byte offset (converted to a line number via the
/// caller definition's start line as a rough approximation).
///
/// If no resolved call sites match, returns a single fallback range
/// covering the caller's first line (the LSP spec requires at least one
/// `from_range` entry).
fn call_site_ranges(
    graph: &RepoGraph,
    caller_file: usize,
    caller_def: usize,
    target_file: usize,
    target_def: usize,
) -> Vec<Range> {
    // DefId components are known to fit: file_idx < files.len() (u32 range),
    // def_idx < defs.len() per file (u16 range).
    #[expect(
        clippy::cast_possible_truncation,
        reason = "file_idx and def_idx originate from graph indices within DefId bounds"
    )]
    let target_id = (target_file as u32, target_def as u16);

    let Some(file) = graph.files.get(caller_file) else {
        return vec![fallback_range(0)];
    };
    let Some(def) = file.defs.get(caller_def) else {
        return vec![fallback_range(0)];
    };

    let ranges: Vec<Range> = def
        .calls
        .iter()
        .filter(|c| c.resolved == Some(target_id))
        .map(|c| {
            // Approximate line from byte offset relative to definition start.
            // We don't have source text, so use the call byte offset to
            // estimate a line within the definition's range.
            let def_start_line = def_line_to_lsp(def.start_line);
            let def_start_byte = def.start_byte;
            // Rough heuristic: average ~40 bytes per line.
            let offset_bytes = c.byte_offset.saturating_sub(def_start_byte);
            let approx_line = def_start_line + offset_bytes / 40;
            // Clamp to definition range.
            let line = approx_line.min(def_line_to_lsp(def.end_line));
            Range {
                start: Position { line, character: 0 },
                end: Position { line, character: 0 },
            }
        })
        .collect();

    if ranges.is_empty() {
        // Fallback: point at the caller's first line.
        vec![fallback_range(def_line_to_lsp(def.start_line))]
    } else {
        ranges
    }
}

/// A zero-width range at the given line (used when precise call site info
/// is unavailable).
fn fallback_range(line: u32) -> Range {
    Range {
        start: Position { line, character: 0 },
        end: Position { line, character: 0 },
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use ripvec_core::repo_map::{CallRef, FileNode};

    /// Build a minimal `RepoGraph` for testing.
    fn test_graph() -> RepoGraph {
        // Two files, each with one definition.
        // File 0, def 0: "main" calls "helper"
        // File 1, def 0: "helper" (called by main)
        let files = vec![
            FileNode {
                path: "src/main.rs".to_string(),
                defs: vec![Definition {
                    name: "main".to_string(),
                    kind: "function_item".to_string(),
                    start_line: 1,
                    end_line: 10,
                    scope: String::new(),
                    signature: Some("fn main()".to_string()),
                    start_byte: 0,
                    end_byte: 200,
                    calls: vec![CallRef {
                        name: "helper".to_string(),
                        byte_offset: 50,
                        resolved: Some((1, 0)),
                    }],
                }],
                imports: vec![],
            },
            FileNode {
                path: "src/lib.rs".to_string(),
                defs: vec![Definition {
                    name: "helper".to_string(),
                    kind: "function_item".to_string(),
                    start_line: 5,
                    end_line: 15,
                    scope: String::new(),
                    signature: Some("fn helper() -> i32".to_string()),
                    start_byte: 0,
                    end_byte: 300,
                    calls: vec![],
                }],
                imports: vec![],
            },
        ];

        // def_offsets: file 0 starts at flat 0, file 1 starts at flat 1
        let def_offsets = vec![0, 1];

        // def_callers[flat=0] = [] (nobody calls main)
        // def_callers[flat=1] = [(0,0)] (main calls helper)
        let def_callers = vec![vec![], vec![(0u32, 0u16)]];

        // def_callees[flat=0] = [(1,0)] (main calls helper)
        // def_callees[flat=1] = [] (helper calls nobody)
        let def_callees = vec![vec![(1u32, 0u16)], vec![]];

        RepoGraph {
            files,
            edges: vec![],
            base_ranks: vec![0.5, 0.5],
            callers: vec![vec![], vec![0]],
            callees: vec![vec![1], vec![]],
            def_edges: vec![((0, 0), (1, 0), 1)],
            def_ranks: vec![0.5, 0.5],
            def_callers,
            def_callees,
            def_offsets,
            alpha: 0.3,
        }
    }

    #[test]
    fn extract_ids_round_trip() {
        let graph = test_graph();
        let root = PathBuf::from("/tmp/project");
        let item = def_to_item(&graph, 0, 0, &root).unwrap();

        let (fi, di) = extract_ids(&item).unwrap();
        assert_eq!(fi, 0);
        assert_eq!(di, 0);
    }

    #[test]
    fn flat_index_in_bounds() {
        let graph = test_graph();
        assert_eq!(flat_index(&graph, 0, 0), Some(0));
        assert_eq!(flat_index(&graph, 1, 0), Some(1));
        // Out of bounds.
        assert_eq!(flat_index(&graph, 2, 0), None);
    }

    #[test]
    fn call_site_ranges_finds_resolved_call() {
        let graph = test_graph();
        let ranges = call_site_ranges(&graph, 0, 0, 1, 0);
        assert_eq!(ranges.len(), 1);
        // Should point somewhere within main's range (lines 0..9 in 0-based).
        assert!(ranges[0].start.line <= 9);
    }

    #[test]
    fn call_site_ranges_fallback_when_no_match() {
        let graph = test_graph();
        // helper has no calls, so looking for any call site should produce fallback.
        let ranges = call_site_ranges(&graph, 1, 0, 0, 0);
        assert_eq!(ranges.len(), 1);
        // Fallback should be helper's start line (0-based: 4).
        assert_eq!(ranges[0].start.line, 4);
    }

    #[test]
    fn def_to_item_populates_fields() {
        let graph = test_graph();
        let root = PathBuf::from("/tmp/project");
        let item = def_to_item(&graph, 1, 0, &root).unwrap();
        assert_eq!(item.name, "helper");
        assert_eq!(item.detail.as_deref(), Some("fn helper() -> i32"));
        assert!(item.data.is_some());
    }
}
