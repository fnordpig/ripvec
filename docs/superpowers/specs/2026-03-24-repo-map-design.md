# Repo Map: PageRank-Weighted Structural Overview

**Date:** 2026-03-24
**Status:** Draft

## Goal

Build an aider-style repo map that provides importance-weighted structural
context for Claude Code. The map serves as both ambient context (MCP resource)
and a ranking signal for semantic search.

## Architecture

### Data Model

```rust
/// Persisted graph — rebuilt on reindex, stored via rkyv + mmap.
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
struct RepoGraph {
    files: Vec<FileNode>,
    edges: Vec<(u32, u32, u32)>,       // (importer, definer, weight)
    base_ranks: Vec<f32>,              // standard PageRank scores
    callers: Vec<Vec<u32>>,            // top callers per file (indices)
    callees: Vec<Vec<u32>>,            // top callees per file (indices)
}

#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
struct FileNode {
    path: String,
    definitions: Vec<Definition>,
    imports: Vec<ImportRef>,
}

#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
struct Definition {
    name: String,
    kind: String,       // "fn", "struct", "trait", "impl", "class", "module"
    signature: Option<String>,
    scope: String,      // from build_scope_chain()
    line: u32,
}

#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
struct ImportRef {
    raw: String,        // "crate::backend::EmbedBackend"
    resolved_file: Option<u32>,  // index into files[]
}
```

### Pipeline

**Stage 1: Extract definitions + imports (tree-sitter)**

Reuse `languages.rs` definition queries. Add import queries per language:

| Language | Query pattern |
|---|---|
| Rust | `(use_declaration) @import` |
| Python | `(import_statement) @import`, `(import_from_statement) @import` |
| JS/TS | `(import_statement) @import` |
| Go | `(import_declaration) @import` |
| Java | `(import_declaration) @import` |
| C/C++ | `(preproc_include) @import` |

**Stage 2: Resolve imports to files**

For Rust: map `use crate::backend::mod` to `src/backend/mod.rs` via
module path → file path convention. For other languages: resolve relative
paths from the importing file's directory.

Unresolved imports (external crates, stdlib) are dropped — they don't
contribute to the internal dependency graph.

**Stage 2b: Build caller/callee lists**

From the edge list, compute per-file:
- `callers[f]`: files that import from `f`, sorted by edge weight descending
- `callees[f]`: files that `f` imports from, sorted by edge weight descending

Store top-5 each. Used in the zeroth rendering tier.

**Stage 3: PageRank**

Standard iterative PageRank, damping=0.85, 30 iterations. ~30 lines.
Also supports topic-sensitive variant: restart probability concentrated
on a focus file instead of uniform.

```rust
fn pagerank(graph: &RepoGraph, focus: Option<usize>, iterations: usize) -> Vec<f32> {
    let n = graph.files.len();
    let d = 0.85_f32;
    let mut rank = vec![1.0 / n as f32; n];
    let mut out_degree = vec![0u32; n];
    for &(src, _, w) in &graph.edges { out_degree[src as usize] += w; }

    for _ in 0..iterations {
        let restart = match focus {
            Some(f) => { let mut v = vec![0.0; n]; v[f] = 1.0; v }
            None => vec![1.0 / n as f32; n],
        };
        let mut new_rank = restart.iter().map(|r| (1.0 - d) * r).collect::<Vec<_>>();
        for &(src, dst, w) in &graph.edges {
            let s = src as usize;
            if out_degree[s] > 0 {
                new_rank[dst as usize] += d * rank[s] * w as f32 / out_degree[s] as f32;
            }
        }
        rank = new_rank;
    }
    rank
}
```

**Stage 4: Budget-constrained rendering**

Sort files by PageRank descending. Four rendering tiers:

| Tier | Budget zone | Detail level |
|---|---|---|
| **Tier 0** | Top 10% | Full signatures + scope chains + "called by: X, Y" / "calls: A, B" |
| **Tier 1** | Next 20% | Full signatures with scope chains |
| **Tier 2** | Next 40% | Definition names + kinds only |
| **Tier 3** | Bottom 30% | File name only |

Token estimation: line count * 10 chars / 4 ≈ tokens.

Output format: indented text tree (aider-style):

```
crates/ripvec-core/src/embed.rs  [rank: 0.12]
  called by: main.rs, server.rs, tools.rs
  calls: backend/mod.rs, chunk.rs, similarity.rs, walk.rs
  fn embed_all(root, backends, tokenizer, cfg, profiler) -> (Vec<CodeChunk>, Vec<Vec<f32>>)
  fn search(root, query, backends, tokenizer, top_k, cfg, profiler) -> Vec<SearchResult>
  fn embed_distributed(tokenized, backends, batch_size, profiler) -> Vec<Vec<f32>>
crates/ripvec-core/src/backend/mod.rs  [rank: 0.09]
  called by: embed.rs, main.rs, server.rs, mlx.rs, metal.rs, cpu.rs
  calls: mlx.rs, metal.rs, cpu.rs, cuda.rs
  trait EmbedBackend: Send + Sync
    fn embed_batch(&self, encodings: &[Encoding]) -> Result<Vec<Vec<f32>>>
    fn is_gpu(&self) -> bool
  fn detect_backends(model_repo: &str) -> Result<Vec<Box<dyn EmbedBackend>>>
...
crates/ripvec-core/src/backend/blas_info.rs  [rank: 0.001]
```

---

## Integration with Semantic Search

### Score normalization + PageRank boost

Different models return different similarity scales (CodeRankEmbed: 0.3-0.8,
BGE-small: 0.5-0.9). Raw PageRank boost would overwhelm low-scale models.

Normalize scores to [0, 1] before applying the structural boost. Use SIMD
for the hot path (NEON on Apple Silicon, SSE on x86):

```rust
fn rank_with_structure(
    similarities: &mut [f32],
    file_ranks: &[f32],
    alpha: f32,
) {
    // SIMD min/max pass (process 4 floats at a time)
    let (min, max) = simd_minmax(similarities);
    let inv_range = 1.0 / (max - min).max(1e-12);

    // SIMD normalize + boost (fused: avoids second pass over memory)
    simd_normalize_boost(similarities, file_ranks, min, inv_range, alpha);
}

// Fallback scalar for non-SIMD targets:
// for (sim, pr) in similarities.iter_mut().zip(file_ranks) {
//     *sim = (*sim - min) * inv_range + alpha * pr;
// }
```

On Apple Silicon: `std::arch::aarch64::vminq_f32` / `vmaxq_f32` for
reduction, `vfmaq_f32` for fused multiply-add in the normalize+boost pass.
4-wide NEON processes 600 chunks in ~150ns.

### Alpha tuning via automated recall hill-climbing

Instead of a hardcoded `alpha = 0.3`, run an automated parameter sweep
at index build time:

1. After building the repo graph + embeddings, sample 10 test queries
   from the definition names (e.g., "EmbedBackend trait", "chunk_file
   function", "SearchIndex ranking")
2. For each alpha in [0.0, 0.1, 0.2, ..., 1.0], compute Recall@10
   against the full-dim unmodified ranking as reference
3. Pick the alpha that maximizes average Recall@10
4. Store the optimal alpha in the cached `RepoGraph`

This auto-tunes per codebase: a highly modular codebase with clear
dependency structure benefits from higher alpha; a flat codebase with
few cross-file references gets lower alpha.

The sweep runs once per reindex (~100ms for 10 queries × 11 alpha values).

### Where it applies

- `search()` in `embed.rs`: after cosine ranking, apply PageRank boost
- `rank_cascade()` in `index.rs`: apply boost in the re-rank phase (not pre-filter)
- MCP `search_code` tool: boost is always active when repo map exists

### Topic-sensitive search

When the MCP server knows which file the user is editing (future: from
editor context), run topic-sensitive PageRank concentrated on that file.
Results from the file's dependency neighborhood get higher structural boost.

---

## MCP Integration

### Resource (ambient context)

```rust
// In ServerCapabilities:
.enable_resources()

// Resource:
uri: "ripvec://repo-map"
name: "Repository Structure Map"
mime_type: "text/markdown"
```

Served on every `resources/read`. ~1K tokens. Rebuilt on reindex,
client notified via `notifications/resources/updated`.

### Tool (focused exploration)

```rust
#[tool(description = "Get a PageRank-weighted structural overview. \
    Use FIRST when exploring unfamiliar code or asked about architecture.")]
async fn get_repo_map(
    &self,
    /// Maximum tokens in the output (default: 2000)
    max_tokens: Option<usize>,
    /// Glob pattern to filter files (e.g. "backend/**")
    file_pattern: Option<String>,
    /// Focus file for topic-sensitive PageRank
    focus_file: Option<String>,
) -> CallToolResult
```

---

## Caching

- **Format**: rkyv zero-copy serialization + mmap (instant load, no deserialization)
- **Key**: manifest root hash (same as search index)
- **Location**: ObjectStore (`cache/store.rs`)
- **Size**: ~10-20KB for ripvec-sized codebase
- **Invalidation**: manifest hash change triggers rebuild
- **Stored fields**: graph edges, definitions, base ranks, caller/callee
  lists, auto-tuned alpha
- **Rendered text**: NOT cached (cheap to regenerate from graph)

Topic-sensitive PageRank runs on the fly from the mmap'd graph (~1ms
for <1000 files).

---

## Files

| File | What |
|---|---|
| `crates/ripvec-core/src/repo_map.rs` | Graph building, PageRank, rendering, alpha tuning (~400 LOC) |
| `crates/ripvec-core/src/languages.rs` | Add import query patterns (~50 LOC) |
| `crates/ripvec-core/src/embed.rs` | SIMD score normalization + PageRank boost (~40 LOC) |
| `crates/ripvec-core/src/index.rs` | Store file PageRank scores alongside embeddings |
| `crates/ripvec-mcp/src/tools.rs` | `get_repo_map` tool handler |
| `crates/ripvec-mcp/src/server.rs` | Resource handler + `enable_resources()` |

---

## Out of Scope

- Full reference resolution (use imports only, not identifier matching)
- Cross-language references (each file's language is independent)
- Visualization / graph rendering
