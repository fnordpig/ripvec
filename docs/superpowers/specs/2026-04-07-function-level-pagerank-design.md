# Function-Level PageRank + Call Hierarchy

**Date**: 2026-04-07
**Status**: Approved

## Problem

PageRank is computed per-file from import edges. This is coarse — a file with one critical function and ten unused helpers gets a single rank. Search results can't distinguish the critical function from the helpers. The LSP call hierarchy operations (`incomingCalls`, `outgoingCalls`) need function-level edges that don't exist.

## Solution

Replace file-level PageRank with definition-level PageRank. Extract call expressions from each definition body using tree-sitter, resolve callee names to definitions, build a call graph between definitions, and compute PageRank on that graph. File-level rank becomes a derived aggregate (`sum of definition ranks`).

## Architecture

### Graph model

**Current**: `FileNode` → `ImportRef` → file-level edges → file-level PageRank

**New**: `Definition` → `CallRef` → definition-level edges → definition-level PageRank → file rank is `sum(def_ranks)`

Each `Definition` gains a `calls: Vec<CallRef>` field. `CallRef` stores the callee name and an optional resolved `DefId`:

```rust
pub type DefId = (u32, u16);  // (file_idx, def_idx within file)

pub struct CallRef {
    pub name: String,
    pub resolved: Option<DefId>,
}
```

### Call expression extraction

15 grammars need call queries (TOML excluded — no calls). The queries fall into 3 categories:

**Category 1 — `function` field** (Rust, Python, JS, TS, TSX, Go, C, C++, Scala):
```
(call_expression function: (identifier) @callee) @call
```
Plus method variants: `field_expression`, `member_expression`, `attribute`, `selector_expression`.

**Category 2 — named field varies** (Java, Ruby, Bash):
```
(method_invocation name: (identifier) @callee) @call     // Java
(call method: (identifier) @callee) @call                // Ruby
(command name: (command_name (word) @callee)) @call      // Bash
```

**Category 3 — positional children** (HCL, Kotlin, Swift):
```
(function_call (identifier) @callee) @call               // HCL
(call_expression (simple_identifier) @callee) @call      // Kotlin, Swift
```

Each language gets a `call_query_for_extension(ext)` function (parallel to `config_for_extension` and `import_query_for_extension`) that returns the compiled call query.

### Call resolution

For each callee name extracted from a definition body:

1. **Same-file lookup**: Search definitions in the same file by name. Prefer exact match, then case-insensitive.
2. **Imported-file lookup**: For files with resolved imports, search definitions in those files by name.
3. **Unresolved**: External crates, dynamic dispatch, etc. → `resolved: None`. These don't create edges but the callee name is preserved for future resolution.

Resolution is name-based, not type-based. This is fuzzy but sufficient for PageRank — even approximate edges produce meaningful structural importance signals.

### PageRank computation

Run the same `pagerank()` algorithm on definition-level nodes:

- **Nodes**: All definitions across all files. Each gets a unique `DefId = (file_idx, def_idx)`.
- **Edges**: `(caller_def_id, callee_def_id, weight)` from resolved `CallRef`s.
- **Output**: `def_ranks: Vec<f32>` parallel to a flattened definitions list.

File-level rank is derived: `file_rank[i] = sum(def_ranks for defs in file[i])`. No separate file-level PageRank computation.

### Alpha auto-tuning

Same formula as today but computed on the definition-level graph density:
```
density = edges.len() / (n_defs * (n_defs - 1))
alpha = 0.3 * min(density, 1.0) + 0.5
```

### Chunk-level boost

Each `CodeChunk` maps 1:1 to a `Definition` (same file, same name, same line range). The `boost_with_pagerank` function looks up the chunk's definition rank directly — no file-level indirection.

New lookup: `pagerank_by_def: HashMap<(String, String), f32>` keyed by `(file_path, def_name)` → definition rank.

### LSP call hierarchy

With definition-level edges, the three LSP call hierarchy operations become straightforward:

- `prepareCallHierarchy(position)`: Find the definition at the cursor → return as `CallHierarchyItem`
- `incomingCalls(item)`: Look up callers of this definition in the edge list
- `outgoingCalls(item)`: Look up callees of this definition in the edge list

### RepoGraph changes

```rust
pub struct RepoGraph {
    pub files: Vec<FileNode>,
    // REMOVED: pub edges: Vec<(u32, u32, u32)>,      // file-level edges
    // REMOVED: pub base_ranks: Vec<f32>,               // file-level ranks
    // REMOVED: pub callers: Vec<Vec<u32>>,             // file-level callers
    // REMOVED: pub callees: Vec<Vec<u32>>,             // file-level callees
    pub def_edges: Vec<(DefId, DefId, u32)>,            // definition-level edges
    pub def_ranks: Vec<f32>,                             // definition-level ranks (flattened)
    pub def_callers: Vec<Vec<DefId>>,                   // per-def incoming callers
    pub def_callees: Vec<Vec<DefId>>,                   // per-def outgoing callees
    pub alpha: f32,
}
```

File-level accessors become methods:
```rust
impl RepoGraph {
    pub fn file_rank(&self, file_idx: usize) -> f32 { ... }
    pub fn file_callers(&self, file_idx: usize) -> Vec<u32> { ... }
    pub fn file_callees(&self, file_idx: usize) -> Vec<u32> { ... }
}
```

The existing `render()` function and `get_repo_map` MCP tool continue to work via these methods — no visible API change for consumers that only need file-level information.

## Backward compatibility

- `pagerank_lookup()` in `hybrid.rs` currently builds `HashMap<String, f32>` from file paths. It will build from `(file_path, def_name)` pairs instead, falling back to file-level aggregation for chunks that don't match a specific definition.
- `boost_with_pagerank()` signature unchanged — it takes a `HashMap` and applies multiplicatively.
- MCP `get_repo_map` rendering uses file-level callers/callees — provided via accessor methods on the new struct.
- LSP `workspaceSymbol`, `goToDefinition`, `findReferences` — no change needed, they already use the chunk-level boost.
- Cache format: `RepoGraph` is serialized with rkyv. The struct change requires a cache version bump or separate versioning for the graph cache.

## Scope

### v2 (this spec)
1. Add `CallRef` to `Definition`, `call_query_for_extension()` to `languages.rs`
2. Extract call expressions during `build_graph()`
3. Resolve calls to definitions (same-file, then imported-file)
4. Build definition-level edge list
5. Compute definition-level PageRank
6. Add file-level accessor methods to `RepoGraph`
7. Update `pagerank_lookup()` to use definition-level ranks
8. Add LSP `prepareCallHierarchy`, `incomingCalls`, `outgoingCalls`
9. Update `render()` to use definition-level data

### Not in scope
- Cross-language call resolution (e.g., Python calling Rust via FFI)
- Type-based resolution (would need a type system per language)
- Dynamic dispatch resolution (would need runtime analysis)
