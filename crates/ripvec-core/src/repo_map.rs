//! `PageRank`-weighted structural overview of a codebase.
//!
//! Builds a dependency graph from tree-sitter definition and import extraction,
//! ranks files by importance using `PageRank` (standard or topic-sensitive), and
//! renders a budget-constrained overview with tiered detail levels.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use streaming_iterator::StreamingIterator;
use tree_sitter::{Parser, Query, QueryCursor};

use crate::languages;
use crate::walk;

// ── Data Structures ──────────────────────────────────────────────────

/// Persisted dependency graph with `PageRank` scores.
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize)]
pub struct RepoGraph {
    /// Files in the repository with definitions and imports.
    pub files: Vec<FileNode>,
    /// Dependency edges: `(importer_idx, definer_idx, weight)`.
    pub edges: Vec<(u32, u32, u32)>,
    /// Standard `PageRank` scores (one per file).
    pub base_ranks: Vec<f32>,
    /// Top callers per file (indices into `files`).
    pub callers: Vec<Vec<u32>>,
    /// Top callees per file (indices into `files`).
    pub callees: Vec<Vec<u32>>,
    /// Auto-tuned alpha for search boost.
    pub alpha: f32,
}

/// A file in the repository with its definitions and imports.
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize)]
pub struct FileNode {
    /// Relative path from the repository root.
    pub path: String,
    /// Definitions (functions, structs, classes, etc.) extracted from this file.
    pub defs: Vec<Definition>,
    /// Import references extracted from this file.
    pub imports: Vec<ImportRef>,
}

/// A definition extracted from a source file.
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize)]
pub struct Definition {
    /// Name of the definition (e.g., function name, class name).
    pub name: String,
    /// Kind of syntax node (e.g., `function_item`, `class_definition`).
    pub kind: String,
    /// 1-based start line number.
    pub start_line: u32,
    /// 1-based end line number.
    pub end_line: u32,
    /// Scope chain (e.g., `"impl_item Foo > fn bar"`).
    pub scope: String,
    /// Function/method signature, if available.
    pub signature: Option<String>,
}

/// An import reference extracted from a source file.
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize)]
pub struct ImportRef {
    /// Raw import path as written in source (e.g., `crate::foo::bar`).
    pub raw_path: String,
    /// Resolved file index in [`RepoGraph::files`], if resolution succeeded.
    pub resolved_idx: Option<u32>,
}

// ── Constants ────────────────────────────────────────────────────────

/// `PageRank` damping factor.
const DAMPING: f32 = 0.85;

/// `PageRank` convergence threshold.
const EPSILON: f32 = 1e-6;

/// Maximum `PageRank` iterations.
const MAX_ITERATIONS: usize = 100;

/// Maximum callers/callees stored per file.
const MAX_NEIGHBORS: usize = 5;

/// Approximate characters per token for budget estimation.
const CHARS_PER_TOKEN: usize = 4;

// ── Import Queries ───────────────────────────────────────────────────

/// Compile a tree-sitter import query for the given extension.
///
/// Returns `None` for unsupported extensions.
fn import_query_for_extension(ext: &str) -> Option<(tree_sitter::Language, Query)> {
    let (lang, query_str): (tree_sitter::Language, &str) = match ext {
        "rs" => (
            tree_sitter_rust::LANGUAGE.into(),
            "(use_declaration) @import",
        ),
        "py" => (
            tree_sitter_python::LANGUAGE.into(),
            concat!(
                "(import_statement) @import\n",
                "(import_from_statement) @import",
            ),
        ),
        "js" | "jsx" => (
            tree_sitter_javascript::LANGUAGE.into(),
            "(import_statement source: (string) @import_path) @import",
        ),
        "ts" => (
            tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            "(import_statement source: (string) @import_path) @import",
        ),
        "tsx" => (
            tree_sitter_typescript::LANGUAGE_TSX.into(),
            "(import_statement source: (string) @import_path) @import",
        ),
        "go" => (
            tree_sitter_go::LANGUAGE.into(),
            "(import_spec path: (interpreted_string_literal) @import_path) @import",
        ),
        _ => return None,
    };
    let query = match Query::new(&lang, query_str) {
        Ok(q) => q,
        Err(e) => {
            tracing::warn!(ext, %e, "import query compilation failed — language may be ABI-incompatible");
            return None;
        }
    };
    Some((lang, query))
}

/// Extract import paths from source using tree-sitter.
fn extract_imports(
    source: &str,
    lang: &tree_sitter::Language,
    import_query: &Query,
) -> Vec<String> {
    let mut parser = Parser::new();
    if parser.set_language(lang).is_err() {
        return vec![];
    }
    let Some(tree) = parser.parse(source, None) else {
        return vec![];
    };

    let mut cursor = QueryCursor::new();
    let mut imports = Vec::new();
    let mut matches = cursor.matches(import_query, tree.root_node(), source.as_bytes());

    while let Some(m) = matches.next() {
        // Prefer @import_path capture (JS/TS/Go), fall back to full @import text
        let mut import_path_text = None;
        let mut import_text = None;

        for cap in m.captures {
            let cap_name = &import_query.capture_names()[cap.index as usize];
            let text = &source[cap.node.start_byte()..cap.node.end_byte()];
            if *cap_name == "import_path" {
                import_path_text = Some(text.trim_matches(|c| c == '"' || c == '\''));
            } else if *cap_name == "import" {
                import_text = Some(text);
            }
        }

        if let Some(path) = import_path_text {
            imports.push(path.to_string());
        } else if let Some(text) = import_text {
            imports.push(text.to_string());
        }
    }

    imports
}

// ── Import Resolution ────────────────────────────────────────────────

/// Resolve a Rust `use` path to a file index in the file map.
///
/// Handles `crate::`, `self::`, and `super::` prefixes. External crate
/// imports are dropped (returns `None`).
fn resolve_rust_import(
    raw: &str,
    file_path: &Path,
    root: &Path,
    file_index: &HashMap<PathBuf, usize>,
) -> Option<usize> {
    // Extract the module path from `use crate::foo::bar;` or `use crate::foo::bar::Baz;`
    let trimmed = raw
        .trim()
        .trim_start_matches("use ")
        .trim_end_matches(';')
        .trim();

    let segments: Vec<&str> = trimmed.split("::").collect();
    if segments.is_empty() {
        return None;
    }

    // Determine the base directory and skip prefix segments
    let (base, skip) = match segments[0] {
        "crate" => {
            // Find the nearest Cargo.toml ancestor to determine the crate root.
            // In a workspace, `crate::foo` resolves relative to the crate's src/,
            // not the workspace root.
            let mut dir = file_path.parent();
            let crate_root = loop {
                match dir {
                    Some(d) if d.join("Cargo.toml").exists() => break d.join("src"),
                    Some(d) => dir = d.parent(),
                    None => break root.join("src"), // fallback
                }
            };
            (crate_root, 1)
        }
        "self" => {
            let dir = file_path.parent()?;
            (dir.to_path_buf(), 1)
        }
        "super" => {
            let dir = file_path.parent()?.parent()?;
            (dir.to_path_buf(), 1)
        }
        // External crate — drop
        _ => return None,
    };

    // Build candidate paths from the remaining segments.
    // Try progressively shorter prefixes since the last segments
    // may be items (struct, fn) rather than modules.
    let path_segments = &segments[skip..];
    for end in (1..=path_segments.len()).rev() {
        let mut candidate = base.clone();
        for seg in &path_segments[..end] {
            // Strip glob patterns like `{Foo, Bar}`
            let clean = seg.split('{').next().unwrap_or(seg).trim();
            if !clean.is_empty() {
                candidate.push(clean);
            }
        }

        // Try file.rs
        let as_file = candidate.with_extension("rs");
        if let Some(&idx) = file_index.get(&as_file) {
            return Some(idx);
        }

        // Try dir/mod.rs
        let as_mod = candidate.join("mod.rs");
        if let Some(&idx) = file_index.get(&as_mod) {
            return Some(idx);
        }
    }

    None
}

/// Resolve an import path to a file index based on file extension.
fn resolve_import(
    raw: &str,
    ext: &str,
    file_path: &Path,
    root: &Path,
    file_index: &HashMap<PathBuf, usize>,
) -> Option<usize> {
    match ext {
        "rs" => resolve_rust_import(raw, file_path, root, file_index),
        "py" => resolve_python_import(raw, root, file_index),
        "js" | "jsx" | "ts" | "tsx" => resolve_js_import(raw, file_path, file_index),
        // Go imports use full package paths — skip local resolution
        _ => None,
    }
}

/// Resolve a Python import to a file index.
///
/// Handles `import foo.bar` and `from foo.bar import baz` patterns.
fn resolve_python_import(
    raw: &str,
    root: &Path,
    file_index: &HashMap<PathBuf, usize>,
) -> Option<usize> {
    let module_path = if let Some(rest) = raw.strip_prefix("from ") {
        rest.split_whitespace().next()?
    } else if let Some(rest) = raw.strip_prefix("import ") {
        rest.split_whitespace().next()?
    } else {
        return None;
    };

    let rel_path: PathBuf = module_path.split('.').collect();
    let as_file = root.join(&rel_path).with_extension("py");
    if let Some(&idx) = file_index.get(&as_file) {
        return Some(idx);
    }

    let as_init = root.join(&rel_path).join("__init__.py");
    file_index.get(&as_init).copied()
}

/// Resolve a JS/TS import to a file index.
///
/// Handles relative paths like `./foo` or `../bar`.
fn resolve_js_import(
    raw: &str,
    file_path: &Path,
    file_index: &HashMap<PathBuf, usize>,
) -> Option<usize> {
    if !raw.starts_with('.') {
        return None;
    }

    let dir = file_path.parent()?;
    let candidate = dir.join(raw);

    for ext in &["js", "jsx", "ts", "tsx"] {
        let with_ext = candidate.with_extension(ext);
        if let Some(&idx) = file_index.get(&with_ext) {
            return Some(idx);
        }
    }

    for ext in &["js", "jsx", "ts", "tsx"] {
        let index_file = candidate.join("index").with_extension(ext);
        if let Some(&idx) = file_index.get(&index_file) {
            return Some(idx);
        }
    }

    None
}

// ── Extraction ───────────────────────────────────────────────────────

/// Extract definitions from a source file using tree-sitter.
fn extract_definitions(source: &str, config: &languages::LangConfig) -> Vec<Definition> {
    let mut parser = Parser::new();
    if parser.set_language(&config.language).is_err() {
        return vec![];
    }
    let Some(tree) = parser.parse(source, None) else {
        return vec![];
    };

    let mut cursor = QueryCursor::new();
    let mut defs = Vec::new();
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
            let scope = crate::chunk::build_scope_chain(node, source);
            let signature = crate::chunk::extract_signature(node, source);
            #[expect(clippy::cast_possible_truncation, reason = "line numbers fit in u32")]
            let start_line = node.start_position().row as u32 + 1;
            #[expect(clippy::cast_possible_truncation, reason = "line numbers fit in u32")]
            let end_line = node.end_position().row as u32 + 1;
            defs.push(Definition {
                name,
                kind: node.kind().to_string(),
                start_line,
                end_line,
                scope,
                signature,
            });
        }
    }

    defs
}

// ── PageRank ─────────────────────────────────────────────────────────

/// Compute `PageRank` scores for a graph.
///
/// If `focus` is `Some(idx)`, computes topic-sensitive `PageRank` biased
/// toward file `idx`. Otherwise computes standard (uniform) `PageRank`.
///
/// Returns one score per node, summing to 1.0.
#[expect(
    clippy::cast_precision_loss,
    reason = "node count fits comfortably in f32"
)]
fn pagerank(n: usize, edges: &[(u32, u32, u32)], focus: Option<usize>) -> Vec<f32> {
    if n == 0 {
        return vec![];
    }

    // Build adjacency: out_edges[src] = [(dst, weight)]
    let mut out_edges: Vec<Vec<(usize, f32)>> = vec![vec![]; n];
    let mut out_weight: Vec<f32> = vec![0.0; n];

    for &(src, dst, w) in edges {
        let (s, d) = (src as usize, dst as usize);
        if s < n && d < n {
            #[expect(clippy::cast_possible_truncation, reason = "edge weights are small")]
            let wf = f64::from(w) as f32;
            out_edges[s].push((d, wf));
            out_weight[s] += wf;
        }
    }

    // Personalization vector: for topic-sensitive PageRank, blend
    // 70% focus on the target file with 30% uniform. Pure focus
    // (100%) starves unreachable nodes to rank=0 in sparse graphs.
    let bias: Vec<f32> = if let Some(idx) = focus {
        let uniform = 1.0 / n as f32;
        let mut b = vec![0.3 * uniform; n];
        if idx < n {
            b[idx] += 0.7;
        }
        // Normalize to sum=1
        let sum: f32 = b.iter().sum();
        for v in &mut b {
            *v /= sum;
        }
        b
    } else {
        vec![1.0 / n as f32; n]
    };

    let mut rank = vec![1.0 / n as f32; n];
    let mut next_rank = vec![0.0_f32; n];

    for _ in 0..MAX_ITERATIONS {
        // Collect dangling mass (nodes with no outgoing edges)
        let dangling: f32 = rank
            .iter()
            .enumerate()
            .filter(|&(i, _)| out_edges[i].is_empty())
            .map(|(_, &r)| r)
            .sum();

        // Distribute rank
        for (i, nr) in next_rank.iter_mut().enumerate() {
            *nr = (1.0 - DAMPING).mul_add(bias[i], DAMPING * dangling * bias[i]);
        }

        for (src, edges_list) in out_edges.iter().enumerate() {
            if edges_list.is_empty() {
                continue;
            }
            let src_rank = rank[src];
            let total_w = out_weight[src];
            for &(dst, w) in edges_list {
                next_rank[dst] += DAMPING * src_rank * (w / total_w);
            }
        }

        // Check convergence
        let diff: f32 = rank
            .iter()
            .zip(next_rank.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        std::mem::swap(&mut rank, &mut next_rank);

        if diff < EPSILON {
            break;
        }
    }

    rank
}

// ── Graph Building ───────────────────────────────────────────────────

/// Build a dependency graph from a repository root.
///
/// Walks the directory tree, parses each supported file with tree-sitter,
/// extracts definitions and imports, resolves import paths to files, runs
/// `PageRank`, and builds caller/callee lists.
///
/// # Errors
///
/// Returns an error if file walking or reading fails.
pub fn build_graph(root: &Path) -> crate::Result<RepoGraph> {
    let root = root.canonicalize().map_err(|e| crate::Error::Io {
        path: root.display().to_string(),
        source: e,
    })?;

    let all_files = walk::collect_files(&root, None);

    // Build file index mapping canonical paths to indices
    let mut file_index: HashMap<PathBuf, usize> = HashMap::new();
    let mut files: Vec<FileNode> = Vec::new();
    let mut raw_sources: Vec<(usize, String, String)> = Vec::new(); // (idx, ext, source)

    for path in &all_files {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or_default()
            .to_string();

        // Only process files with known language support
        if languages::config_for_extension(&ext).is_none()
            && import_query_for_extension(&ext).is_none()
        {
            continue;
        }

        let Ok(source) = std::fs::read_to_string(path) else {
            continue; // Skip binary/unreadable files
        };

        let rel_path = path
            .strip_prefix(&root)
            .unwrap_or(path)
            .display()
            .to_string();

        let idx = files.len();
        file_index.insert(path.clone(), idx);
        files.push(FileNode {
            path: rel_path,
            defs: vec![],
            imports: vec![],
        });
        raw_sources.push((idx, ext, source));
    }

    // Extract definitions and imports
    for (idx, ext, source) in &raw_sources {
        // Extract definitions
        if let Some(config) = languages::config_for_extension(ext) {
            files[*idx].defs = extract_definitions(source, &config);
        }

        // Extract imports
        if let Some((lang, import_query)) = import_query_for_extension(ext) {
            let raw_imports = extract_imports(source, &lang, &import_query);
            let file_path = root.join(&files[*idx].path);

            files[*idx].imports = raw_imports
                .into_iter()
                .map(|raw| {
                    let resolved_idx = resolve_import(&raw, ext, &file_path, &root, &file_index)
                        .and_then(|i| u32::try_from(i).ok());
                    ImportRef {
                        raw_path: raw,
                        resolved_idx,
                    }
                })
                .collect();
        }
    }

    // Build edge list from resolved imports
    let mut edge_map: HashMap<(u32, u32), u32> = HashMap::new();
    for (importer_idx, file) in files.iter().enumerate() {
        for import in &file.imports {
            if let Some(definer_idx) = import.resolved_idx
                && let Ok(src) = u32::try_from(importer_idx)
            {
                *edge_map.entry((src, definer_idx)).or_insert(0) += 1;
            }
        }
    }
    let edges: Vec<(u32, u32, u32)> = edge_map
        .into_iter()
        .map(|((src, dst), w)| (src, dst, w))
        .collect();

    // Compute PageRank
    let n = files.len();
    let base_ranks = pagerank(n, &edges, None);

    // Build caller/callee lists
    let (inbound, outbound) = build_neighbor_lists(n, &edges);

    // Auto-tune alpha based on graph density
    #[expect(clippy::cast_precision_loss, reason = "graph sizes fit in f32")]
    let density = if n > 1 {
        edges.len() as f32 / (n as f32 * (n as f32 - 1.0))
    } else {
        0.0
    };
    let alpha = 0.3f32.mul_add(density.min(1.0), 0.5);

    Ok(RepoGraph {
        files,
        edges,
        base_ranks,
        callers: inbound,
        callees: outbound,
        alpha,
    })
}

/// Build top-N caller and callee lists for each file.
fn build_neighbor_lists(n: usize, edges: &[(u32, u32, u32)]) -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
    let mut incoming: Vec<Vec<(u32, u32)>> = vec![vec![]; n];
    let mut outgoing: Vec<Vec<(u32, u32)>> = vec![vec![]; n];

    for &(src, dst, w) in edges {
        let (s, d) = (src as usize, dst as usize);
        if s < n && d < n {
            incoming[d].push((src, w));
            outgoing[s].push((dst, w));
        }
    }

    // Sort by weight descending, keep top N
    let trim = |lists: &mut [Vec<(u32, u32)>]| -> Vec<Vec<u32>> {
        lists
            .iter_mut()
            .map(|list| {
                list.sort_by(|a, b| b.1.cmp(&a.1));
                list.iter()
                    .take(MAX_NEIGHBORS)
                    .map(|(idx, _)| *idx)
                    .collect()
            })
            .collect()
    };

    (trim(&mut incoming), trim(&mut outgoing))
}

// ── Rendering ────────────────────────────────────────────────────────

/// Render a budget-constrained overview of the repository.
///
/// Files are sorted by `PageRank` (or topic-sensitive rank if `focus` is
/// `Some`). Output uses four tiers of decreasing detail:
///
/// - **Tier 0** (top 10%): full path, rank, callers/callees, signatures with scopes
/// - **Tier 1** (next 20%): full path, rank, signatures
/// - **Tier 2** (next 40%): full path, rank, definition names and kinds
/// - **Tier 3** (bottom 30%): file path only
///
/// Stops accumulating output when the estimated token count exceeds
/// `max_tokens`.
#[must_use]
pub fn render(graph: &RepoGraph, max_tokens: usize, focus: Option<usize>) -> String {
    let n = graph.files.len();
    if n == 0 {
        return String::new();
    }

    // Compute ranks (recompute topic-sensitive if focus is given)
    let ranks = if focus.is_some() {
        pagerank(n, &graph.edges, focus)
    } else {
        graph.base_ranks.clone()
    };

    // Sort file indices by rank descending
    let mut sorted: Vec<usize> = (0..n).collect();
    sorted.sort_by(|&a, &b| ranks[b].total_cmp(&ranks[a]));

    let mut output = String::new();
    let mut used_tokens = 0;
    let max_chars = max_tokens * CHARS_PER_TOKEN;

    for (rank_pos, &file_idx) in sorted.iter().enumerate() {
        if used_tokens >= max_tokens {
            break;
        }

        let file = &graph.files[file_idx];
        let score = ranks[file_idx];
        #[expect(clippy::cast_precision_loss, reason = "file counts fit in f32")]
        let percentile = (rank_pos as f32) / (n as f32);

        let section = if percentile < 0.1 {
            render_tier0(graph, file_idx, file, score)
        } else if percentile < 0.3 {
            render_tier1(file, score)
        } else if percentile < 0.7 {
            render_tier2(file, score)
        } else {
            render_tier3(file)
        };

        let section_chars = section.len();
        if used_tokens > 0 && used_tokens + section_chars / CHARS_PER_TOKEN > max_tokens {
            // Would exceed budget — try to fit at least the path
            let path_line = format!("{}\n", file.path);
            let path_tokens = path_line.len() / CHARS_PER_TOKEN;
            if used_tokens + path_tokens <= max_tokens {
                output.push_str(&path_line);
            }
            break;
        }

        output.push_str(&section);
        used_tokens = output.len().min(max_chars) / CHARS_PER_TOKEN;
    }

    output
}

/// Render tier 0: full detail with callers, callees, and signatures.
fn render_tier0(graph: &RepoGraph, file_idx: usize, file: &FileNode, score: f32) -> String {
    let mut out = format!("## {} (rank: {score:.4})\n", file.path);

    // Callers
    if file_idx < graph.callers.len() && !graph.callers[file_idx].is_empty() {
        let _ = write!(out, "  called by: ");
        let names: Vec<&str> = graph.callers[file_idx]
            .iter()
            .filter_map(|&idx| graph.files.get(idx as usize).map(|f| f.path.as_str()))
            .collect();
        let _ = writeln!(out, "{}", names.join(", "));
    }

    // Callees
    if file_idx < graph.callees.len() && !graph.callees[file_idx].is_empty() {
        let _ = write!(out, "  calls: ");
        let names: Vec<&str> = graph.callees[file_idx]
            .iter()
            .filter_map(|&idx| graph.files.get(idx as usize).map(|f| f.path.as_str()))
            .collect();
        let _ = writeln!(out, "{}", names.join(", "));
    }

    // Definitions with scope and signature
    for def in &file.defs {
        let scope_prefix = if def.scope.is_empty() {
            String::new()
        } else {
            format!("{} > ", def.scope)
        };
        if let Some(sig) = &def.signature {
            let _ = writeln!(out, "  {scope_prefix}{} {sig}", def.kind);
        } else {
            let _ = writeln!(out, "  {scope_prefix}{} {}", def.kind, def.name);
        }
    }
    let _ = writeln!(out);
    out
}

/// Render tier 1: file path, rank, and signatures.
fn render_tier1(file: &FileNode, score: f32) -> String {
    let mut out = format!("## {} (rank: {score:.4})\n", file.path);
    for def in &file.defs {
        if let Some(sig) = &def.signature {
            let _ = writeln!(out, "  {sig}");
        } else {
            let _ = writeln!(out, "  {} {}", def.kind, def.name);
        }
    }
    let _ = writeln!(out);
    out
}

/// Render tier 2: file path, rank, and definition names/kinds.
fn render_tier2(file: &FileNode, score: f32) -> String {
    let mut out = format!("{} (rank: {score:.4})", file.path);
    if !file.defs.is_empty() {
        let names: Vec<String> = file
            .defs
            .iter()
            .map(|d| format!("{}:{}", d.kind, d.name))
            .collect();
        let _ = write!(out, " -- {}", names.join(", "));
    }
    let _ = writeln!(out);
    out
}

/// Render tier 3: file path only.
fn render_tier3(file: &FileNode) -> String {
    format!("{}\n", file.path)
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pagerank_simple() {
        // 3-node graph: 0 -> 1 -> 2, 2 -> 0 (cycle)
        let edges = vec![(0, 1, 1), (1, 2, 1), (2, 0, 1)];
        let ranks = pagerank(3, &edges, None);

        // All nodes in a symmetric cycle should have equal rank
        assert_eq!(ranks.len(), 3);
        let sum: f32 = ranks.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "ranks should sum to ~1.0, got {sum}"
        );

        // In a perfect cycle, all ranks should be approximately equal
        let expected = 1.0 / 3.0;
        for (i, &r) in ranks.iter().enumerate() {
            assert!(
                (r - expected).abs() < 0.05,
                "rank[{i}] = {r}, expected ~{expected}"
            );
        }
    }

    #[test]
    fn test_pagerank_star() {
        // Star graph: 0,1,2 all point to 3
        let edges = vec![(0, 3, 1), (1, 3, 1), (2, 3, 1)];
        let ranks = pagerank(4, &edges, None);

        assert_eq!(ranks.len(), 4);
        // Node 3 should have the highest rank
        let max_idx = ranks
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        assert_eq!(max_idx, 3, "node 3 should have highest rank");
        assert!(
            ranks[3] > ranks[0],
            "rank[3]={} should be > rank[0]={}",
            ranks[3],
            ranks[0]
        );
    }

    #[test]
    fn test_pagerank_topic_sensitive() {
        // 3-node chain: 0 -> 1 -> 2
        let edges = vec![(0, 1, 1), (1, 2, 1)];
        let uniform_ranks = pagerank(3, &edges, None);
        let biased_ranks = pagerank(3, &edges, Some(0));

        // With focus on node 0, it should get a higher rank than uniform
        assert!(
            biased_ranks[0] > uniform_ranks[0],
            "focused rank[0]={} should be > uniform rank[0]={}",
            biased_ranks[0],
            uniform_ranks[0]
        );
    }

    #[test]
    fn test_pagerank_empty() {
        let ranks = pagerank(0, &[], None);
        assert!(ranks.is_empty());
    }

    #[test]
    fn test_render_tiers() {
        // Build a small graph with 10 files to exercise all tiers
        let files: Vec<FileNode> = (0..10)
            .map(|i| FileNode {
                path: format!("src/file_{i}.rs"),
                defs: vec![Definition {
                    name: format!("func_{i}"),
                    kind: "function_item".to_string(),
                    start_line: 1,
                    end_line: 5,
                    scope: String::new(),
                    signature: Some(format!("func_{i}(x: i32) -> i32")),
                }],
                imports: vec![],
            })
            .collect();

        // Create a star graph: files 1-9 all import from file 0
        let edges: Vec<(u32, u32, u32)> = (1..10).map(|i| (i, 0, 1)).collect();
        let base_ranks = pagerank(10, &edges, None);
        let (top_callers, top_callees) = build_neighbor_lists(10, &edges);

        let graph = RepoGraph {
            files,
            edges,
            base_ranks,
            callers: top_callers,
            callees: top_callees,
            alpha: 0.5,
        };

        // Large budget: should include all files
        let full = render(&graph, 10_000, None);
        assert!(
            full.contains("file_0"),
            "output should contain the top-ranked file"
        );
        // file_0 should appear as tier 0 (highest rank)
        assert!(
            full.contains("## src/file_0.rs"),
            "top file should have tier 0 heading"
        );

        // Tiny budget: should only fit a few files
        let small = render(&graph, 10, None);
        assert!(
            !small.is_empty(),
            "even tiny budget should produce some output"
        );
        // Should have fewer entries than full render
        let full_lines = full.lines().count();
        let small_lines = small.lines().count();
        assert!(
            small_lines < full_lines,
            "small budget ({small_lines} lines) should have fewer lines than full ({full_lines})"
        );
    }

    #[test]
    fn test_render_empty_graph() {
        let graph = RepoGraph {
            files: vec![],
            edges: vec![],
            base_ranks: vec![],
            callers: vec![],
            callees: vec![],
            alpha: 0.5,
        };
        let output = render(&graph, 1000, None);
        assert!(output.is_empty(), "empty graph should render empty string");
    }

    #[test]
    fn test_build_graph_on_fixtures() {
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("tests")
            .join("fixtures");

        let graph = build_graph(&fixtures).expect("build_graph should succeed on fixtures");

        // Should find at least the 3 fixture files
        assert!(
            !graph.files.is_empty(),
            "graph should contain files from fixtures"
        );

        // Should find definitions in the Rust fixture
        let rs_file = graph.files.iter().find(|f| f.path.ends_with("sample.rs"));
        assert!(rs_file.is_some(), "should find sample.rs");
        let rs_file = rs_file.unwrap();
        assert!(
            !rs_file.defs.is_empty(),
            "sample.rs should have definitions"
        );
        assert!(
            rs_file.defs.iter().any(|d| d.name == "hello"),
            "should find 'hello' function in sample.rs"
        );

        // Should find definitions in the Python fixture
        let py_file = graph.files.iter().find(|f| f.path.ends_with("sample.py"));
        assert!(py_file.is_some(), "should find sample.py");
        let py_file = py_file.unwrap();
        assert!(
            !py_file.defs.is_empty(),
            "sample.py should have definitions"
        );
        assert!(
            py_file.defs.iter().any(|d| d.name == "greet"),
            "should find 'greet' function in sample.py"
        );

        // PageRank scores should be computed
        assert_eq!(graph.base_ranks.len(), graph.files.len());
        let sum: f32 = graph.base_ranks.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "PageRank scores should sum to ~1.0, got {sum}"
        );
    }

    #[test]
    fn test_extract_imports_rust() {
        let source = "use crate::foo::bar;\nuse std::collections::HashMap;\n";
        let (lang, query) = import_query_for_extension("rs").unwrap();
        let imports = extract_imports(source, &lang, &query);
        assert_eq!(imports.len(), 2);
        assert!(imports[0].contains("crate::foo::bar"));
    }

    #[test]
    fn test_resolve_rust_crate_import() {
        let root = PathBuf::from("/project");
        let file_path = PathBuf::from("/project/src/main.rs");
        let mut file_index = HashMap::new();
        file_index.insert(PathBuf::from("/project/src/foo/bar.rs"), 1);
        file_index.insert(PathBuf::from("/project/src/main.rs"), 0);

        let result = resolve_rust_import("use crate::foo::bar;", &file_path, &root, &file_index);
        assert_eq!(result, Some(1));
    }

    #[test]
    fn test_resolve_rust_external_crate_dropped() {
        let root = PathBuf::from("/project");
        let file_path = PathBuf::from("/project/src/main.rs");
        let file_index = HashMap::new();

        let result = resolve_rust_import(
            "use std::collections::HashMap;",
            &file_path,
            &root,
            &file_index,
        );
        assert_eq!(result, None, "external crate imports should be dropped");
    }

    #[test]
    fn test_neighbor_lists() {
        // 0 -> 1, 0 -> 2, 1 -> 2
        let edges = vec![(0, 1, 1), (0, 2, 1), (1, 2, 1)];
        let (incoming, outgoing) = build_neighbor_lists(3, &edges);

        // Node 2 should be called by 0 and 1
        assert!(incoming[2].contains(&0));
        assert!(incoming[2].contains(&1));

        // Node 0 should call 1 and 2
        assert!(outgoing[0].contains(&1));
        assert!(outgoing[0].contains(&2));
    }

    #[test]
    #[ignore = "runs on full ripvec codebase; use --nocapture to see output"]
    fn test_full_repo_map() {
        use std::time::Instant;

        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap();

        // Phase 1: build_graph (walk + parse + import resolve + PageRank)
        let t0 = Instant::now();
        let graph = build_graph(root).expect("build_graph on ripvec root");
        let build_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Phase 2: render (default, no focus)
        let t1 = Instant::now();
        let rendered = render(&graph, 2000, None);
        let render_ms = t1.elapsed().as_secs_f64() * 1000.0;

        // Phase 3: render (topic-sensitive, focused on highest-ranked file)
        let t2 = Instant::now();
        let focus_idx = graph
            .base_ranks
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i);
        let focused = render(&graph, 2000, focus_idx);
        let focus_ms = t2.elapsed().as_secs_f64() * 1000.0;

        eprintln!("\n=== Repo Map Performance ===");
        eprintln!(
            "Files: {}, Edges: {}, Defs: {}",
            graph.files.len(),
            graph.edges.len(),
            graph.files.iter().map(|f| f.defs.len()).sum::<usize>()
        );
        eprintln!("build_graph:     {build_ms:.1}ms (walk + parse + resolve + PageRank)");
        eprintln!(
            "render(default): {render_ms:.3}ms ({} chars, ~{} tokens)",
            rendered.len(),
            rendered.len() / 4
        );
        eprintln!(
            "render(focused): {focus_ms:.3}ms ({} chars, ~{} tokens)",
            focused.len(),
            focused.len() / 4
        );

        eprintln!("\nTop 5 by PageRank:");
        let mut ranked: Vec<(usize, f32)> = graph.base_ranks.iter().copied().enumerate().collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        for (i, rank) in ranked.iter().take(5) {
            eprintln!("  {:.4} {}", rank, graph.files[*i].path);
        }

        eprintln!("\n=== Default Render ===\n{rendered}");
        eprintln!(
            "\n=== Focused Render (on {}) ===\n{focused}",
            focus_idx
                .map(|i| graph.files[i].path.as_str())
                .unwrap_or("none")
        );
    }
}
