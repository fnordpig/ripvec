# Function-Level PageRank Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace file-level import-edge PageRank with definition-level call-edge PageRank. Each function/class gets its own rank; file rank is the aggregate. Enables chunk-level search boost and LSP call hierarchy.

**Architecture:** Add call queries to `languages.rs` (15 grammars). Extract call expressions within each definition body during `build_graph()`. Resolve callee names to `DefId`s using same-file-first, then imported-file strategy. Build def-level edges, compute PageRank on the definition graph, aggregate to file level. Update `RepoGraph` struct with def-level fields and file-level accessor methods. Wire LSP call hierarchy through the def-level graph.

**Tech Stack:** tree-sitter (call expression queries), existing `pagerank()` function (already generic over node count)

---

### Task 1: Add call query infrastructure to languages.rs

**Files:**
- Modify: `crates/ripvec-core/src/languages.rs` — add `call_query_for_extension()` function

- [ ] **Step 1: Add the `CallConfig` struct and cache**

In `crates/ripvec-core/src/languages.rs`, add after the existing `LangConfig` and `config_for_extension`:

```rust
/// Configuration for extracting function calls from a language.
pub struct CallConfig {
    /// The tree-sitter Language grammar.
    pub language: Language,
    /// Query that extracts call sites (`@callee` captures with `@call`).
    pub query: Query,
}

/// Look up the call extraction configuration for a file extension.
///
/// Compiled queries are cached per extension. Returns `None` for
/// unsupported extensions (e.g., TOML has no call expressions).
#[must_use]
pub fn call_query_for_extension(ext: &str) -> Option<Arc<CallConfig>> {
    static CACHE: OnceLock<std::collections::HashMap<&'static str, Arc<CallConfig>>> =
        OnceLock::new();

    let cache = CACHE.get_or_init(|| {
        let mut m = std::collections::HashMap::new();
        for &ext in &[
            "rs", "py", "js", "jsx", "ts", "tsx", "go", "java", "c", "h", "cpp", "cc", "cxx",
            "hpp", "sh", "bash", "bats", "rb", "tf", "tfvars", "hcl", "kt", "kts", "swift",
            "scala",
            // Note: "toml" excluded — TOML has no function calls.
        ] {
            if let Some(cfg) = compile_call_config(ext) {
                m.insert(ext, Arc::new(cfg));
            }
        }
        m
    });

    cache.get(ext).cloned()
}
```

- [ ] **Step 2: Implement `compile_call_config`**

Add the per-language call queries. These extract the callee identifier from call expressions:

```rust
fn compile_call_config(ext: &str) -> Option<CallConfig> {
    let (lang, query_str): (Language, &str) = match ext {
        // Rust: function calls and method calls.
        "rs" => (
            tree_sitter_rust::LANGUAGE.into(),
            concat!(
                "(call_expression function: (identifier) @callee) @call\n",
                "(call_expression function: (field_expression field: (field_identifier) @callee)) @call\n",
                "(call_expression function: (scoped_identifier name: (identifier) @callee)) @call",
            ),
        ),
        // Python: function and method calls.
        "py" => (
            tree_sitter_python::LANGUAGE.into(),
            concat!(
                "(call function: (identifier) @callee) @call\n",
                "(call function: (attribute attribute: (identifier) @callee)) @call",
            ),
        ),
        // JavaScript: function and method calls.
        "js" | "jsx" => (
            tree_sitter_javascript::LANGUAGE.into(),
            concat!(
                "(call_expression function: (identifier) @callee) @call\n",
                "(call_expression function: (member_expression property: (property_identifier) @callee)) @call",
            ),
        ),
        "ts" => (
            tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            concat!(
                "(call_expression function: (identifier) @callee) @call\n",
                "(call_expression function: (member_expression property: (property_identifier) @callee)) @call",
            ),
        ),
        "tsx" => (
            tree_sitter_typescript::LANGUAGE_TSX.into(),
            concat!(
                "(call_expression function: (identifier) @callee) @call\n",
                "(call_expression function: (member_expression property: (property_identifier) @callee)) @call",
            ),
        ),
        // Go: function and method calls.
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
        // C: function calls.
        "c" | "h" => (
            tree_sitter_c::LANGUAGE.into(),
            concat!(
                "(call_expression function: (identifier) @callee) @call\n",
                "(call_expression function: (field_expression field: (field_identifier) @callee)) @call",
            ),
        ),
        // C++: function and method calls.
        "cpp" | "cc" | "cxx" | "hpp" => (
            tree_sitter_cpp::LANGUAGE.into(),
            concat!(
                "(call_expression function: (identifier) @callee) @call\n",
                "(call_expression function: (field_expression field: (field_identifier) @callee)) @call",
            ),
        ),
        // Bash: command invocations.
        "sh" | "bash" | "bats" => (
            tree_sitter_bash::LANGUAGE.into(),
            "(command name: (command_name (word) @callee)) @call",
        ),
        // Ruby: method calls.
        "rb" => (
            tree_sitter_ruby::LANGUAGE.into(),
            "(call method: (identifier) @callee) @call",
        ),
        // HCL: function calls in expressions.
        "tf" | "tfvars" | "hcl" => (
            tree_sitter_hcl::LANGUAGE.into(),
            "(function_call (identifier) @callee) @call",
        ),
        // Kotlin: function and method calls.
        "kt" | "kts" => (
            tree_sitter_kotlin_ng::LANGUAGE.into(),
            "(call_expression (simple_identifier) @callee) @call",
        ),
        // Swift: function calls.
        "swift" => (
            tree_sitter_swift::LANGUAGE.into(),
            "(call_expression (simple_identifier) @callee) @call",
        ),
        // Scala: function and method calls.
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
```

NOTE: These query patterns are based on grammar node-types.json analysis. Some may fail to compile against the actual grammars. After adding them, run `cargo test -p ripvec-core languages::tests` — the test should verify all call-supporting extensions return `Some`. If any fail, check the grammar source and adjust the query pattern.

- [ ] **Step 3: Add test**

Add to the test module in `languages.rs`:

```rust
#[test]
fn all_call_query_extensions() {
    // All extensions except toml should have call queries
    let exts = [
        "rs", "py", "js", "jsx", "ts", "tsx", "go", "java", "c", "h", "cpp", "cc", "cxx",
        "hpp", "sh", "bash", "bats", "rb", "tf", "tfvars", "hcl", "kt", "kts", "swift",
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
```

- [ ] **Step 4: Verify**

Run: `cargo test -p ripvec-core languages::tests`
Expected: all tests pass including the new call query tests.

Run: `cargo clippy -p ripvec-core -- -D warnings`

- [ ] **Step 5: Commit**

```bash
git add crates/ripvec-core/src/languages.rs
git commit -m "feat: add call_query_for_extension for 15 grammars"
```

---

### Task 2: Extract call sites within definitions

**Files:**
- Modify: `crates/ripvec-core/src/repo_map.rs` — add `CallRef`, update `Definition`, add extraction function

- [ ] **Step 1: Add `DefId` type alias and `CallRef` struct**

Add after the `ImportRef` struct in `repo_map.rs`:

```rust
/// Unique identifier for a definition: (file index, definition index within file).
pub type DefId = (u32, u16);

/// A call site extracted from a definition body.
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize)]
pub struct CallRef {
    /// Callee function/method name.
    pub name: String,
    /// Byte offset of the call in the source file (for scoping to definitions).
    pub byte_offset: u32,
    /// Resolved target definition, if resolution succeeded.
    pub resolved: Option<DefId>,
}
```

- [ ] **Step 2: Add `calls` field to `Definition`**

Add to the `Definition` struct:

```rust
pub struct Definition {
    pub name: String,
    pub kind: String,
    pub start_line: u32,
    pub end_line: u32,
    pub scope: String,
    pub signature: Option<String>,
    /// Byte range of this definition in the source file (for scoping call extraction).
    pub start_byte: u32,
    pub end_byte: u32,
    /// Call sites within this definition's body.
    pub calls: Vec<CallRef>,
}
```

Update `extract_definitions` to populate `start_byte` and `end_byte` from `node.start_byte()` / `node.end_byte()`. Initialize `calls: vec![]`.

- [ ] **Step 3: Add `extract_calls` function**

Add a new function that extracts all call sites from a file, then assigns each to the nearest enclosing definition:

```rust
/// Extract call sites from a source file and assign them to definitions.
///
/// Uses the language's call query to find all call expressions, then
/// assigns each call to the definition whose byte range contains it.
/// Calls outside any definition body (module-level) are ignored.
fn extract_calls(
    source: &str,
    call_config: &languages::CallConfig,
    defs: &mut [Definition],
) {
    let mut parser = Parser::new();
    if parser.set_language(&call_config.language).is_err() {
        return;
    }
    let Some(tree) = parser.parse(source, None) else {
        return;
    };

    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&call_config.query, tree.root_node(), source.as_bytes());

    while let Some(m) = matches.next() {
        let mut callee_name = None;
        let mut call_byte = 0u32;

        for cap in m.captures {
            let cap_name = &call_config.query.capture_names()[cap.index as usize];
            if *cap_name == "callee" {
                callee_name =
                    Some(source[cap.node.start_byte()..cap.node.end_byte()].to_string());
                #[expect(clippy::cast_possible_truncation, reason = "byte offsets fit in u32")]
                { call_byte = cap.node.start_byte() as u32; }
            }
        }

        if let Some(name) = callee_name {
            // Assign to the enclosing definition by byte range
            if let Some(def) = defs.iter_mut().find(|d| {
                d.start_byte <= call_byte && call_byte < d.end_byte
            }) {
                // Skip self-recursive calls
                if def.name != name {
                    def.calls.push(CallRef {
                        name,
                        byte_offset: call_byte,
                        resolved: None,
                    });
                }
            }
            // Calls outside any definition are ignored (module-level init)
        }
    }
}
```

- [ ] **Step 4: Update `extract_definitions` to populate byte ranges**

In `extract_definitions`, add `start_byte` and `end_byte` to the `Definition` construction:

```rust
#[expect(clippy::cast_possible_truncation, reason = "byte offsets fit in u32")]
let start_byte = node.start_byte() as u32;
#[expect(clippy::cast_possible_truncation, reason = "byte offsets fit in u32")]
let end_byte = node.end_byte() as u32;
defs.push(Definition {
    name,
    kind: node.kind().to_string(),
    start_line,
    end_line,
    scope,
    signature,
    start_byte,
    end_byte,
    calls: vec![],
});
```

- [ ] **Step 5: Wire call extraction into `build_graph`**

In `build_graph()`, after the existing definition extraction loop (line ~548), add call extraction:

```rust
// Extract calls within definitions
if let Some(call_config) = languages::call_query_for_extension(ext) {
    extract_calls(source, &call_config, &mut files[*idx].defs);
}
```

- [ ] **Step 6: Verify**

Run: `cargo check -p ripvec-core`
Run: `cargo test -p ripvec-core repo_map::tests`

- [ ] **Step 7: Commit**

```bash
git add crates/ripvec-core/src/repo_map.rs
git commit -m "feat: extract call sites within definitions using tree-sitter"
```

---

### Task 3: Resolve calls and build definition-level graph

**Files:**
- Modify: `crates/ripvec-core/src/repo_map.rs` — add resolution logic, update `build_graph`

- [ ] **Step 1: Add definition name index builder**

Add a helper function:

```rust
/// Build an index from definition name → list of DefIds.
fn build_def_index(files: &[FileNode]) -> HashMap<String, Vec<DefId>> {
    let mut index: HashMap<String, Vec<DefId>> = HashMap::new();
    for (file_idx, file) in files.iter().enumerate() {
        for (def_idx, def) in file.defs.iter().enumerate() {
            #[expect(clippy::cast_possible_truncation)]
            let did: DefId = (file_idx as u32, def_idx as u16);
            index.entry(def.name.clone()).or_default().push(did);
        }
    }
    index
}
```

- [ ] **Step 2: Add call resolution function**

```rust
/// Resolve call references to target definitions.
///
/// Strategy:
/// 1. Same-file: prefer definitions in the caller's own file.
/// 2. Imported-file: check definitions in files this file imports.
/// 3. Unresolved: leave `resolved` as `None`.
fn resolve_calls(
    files: &mut [FileNode],
    def_index: &HashMap<String, Vec<DefId>>,
) {
    // Pre-compute imported file sets for each file
    let imported_files: Vec<std::collections::HashSet<u32>> = files
        .iter()
        .map(|f| {
            f.imports
                .iter()
                .filter_map(|imp| imp.resolved_idx)
                .collect()
        })
        .collect();

    for file_idx in 0..files.len() {
        for def_idx in 0..files[file_idx].defs.len() {
            for call_idx in 0..files[file_idx].defs[def_idx].calls.len() {
                let call_name = files[file_idx].defs[def_idx].calls[call_idx].name.clone();

                let Some(candidates) = def_index.get(&call_name) else {
                    continue;
                };

                // Priority 1: same file
                #[expect(clippy::cast_possible_truncation)]
                let file_idx_u32 = file_idx as u32;
                if let Some(&did) = candidates.iter().find(|(f, _)| *f == file_idx_u32) {
                    files[file_idx].defs[def_idx].calls[call_idx].resolved = Some(did);
                    continue;
                }

                // Priority 2: imported file
                if let Some(&did) = candidates
                    .iter()
                    .find(|(f, _)| imported_files[file_idx].contains(f))
                {
                    files[file_idx].defs[def_idx].calls[call_idx].resolved = Some(did);
                }
                // Priority 3: unresolved — leave as None
            }
        }
    }
}
```

- [ ] **Step 3: Build def-level edges from resolved calls**

Add to `build_graph()`, after call extraction and resolution:

```rust
// Build definition-level edge list
let mut def_edge_map: HashMap<(DefId, DefId), u32> = HashMap::new();
for (file_idx, file) in files.iter().enumerate() {
    for (def_idx, def) in file.defs.iter().enumerate() {
        #[expect(clippy::cast_possible_truncation)]
        let caller_id: DefId = (file_idx as u32, def_idx as u16);
        for call in &def.calls {
            if let Some(callee_id) = call.resolved {
                *def_edge_map.entry((caller_id, callee_id)).or_insert(0) += 1;
            }
        }
    }
}
let def_edges: Vec<(DefId, DefId, u32)> = def_edge_map
    .into_iter()
    .map(|((src, dst), w)| (src, dst, w))
    .collect();
```

- [ ] **Step 4: Compute def-level PageRank**

The existing `pagerank()` function takes `(n, edges: &[(u32, u32, u32)], focus)`. For def-level, we need to flatten DefIds to a single index space. Add a helper:

```rust
/// Compute a prefix-sum offset table for flattening DefIds to linear indices.
fn def_offsets(files: &[FileNode]) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(files.len() + 1);
    offsets.push(0);
    for file in files {
        offsets.push(offsets.last().unwrap() + file.defs.len());
    }
    offsets
}

/// Flatten a DefId to a linear index using the offset table.
fn flatten_def_id(offsets: &[usize], did: DefId) -> usize {
    offsets[did.0 as usize] + did.1 as usize
}
```

Then in `build_graph()`:

```rust
// Compute def-level PageRank
let offsets = def_offsets(&files);
let n_defs = *offsets.last().unwrap_or(&0);

// Convert def-level edges to flat indices for pagerank()
let flat_def_edges: Vec<(u32, u32, u32)> = def_edges
    .iter()
    .map(|(src, dst, w)| {
        #[expect(clippy::cast_possible_truncation)]
        (
            flatten_def_id(&offsets, *src) as u32,
            flatten_def_id(&offsets, *dst) as u32,
            *w,
        )
    })
    .collect();

let def_ranks = pagerank(n_defs, &flat_def_edges, None);
```

- [ ] **Step 5: Derive file-level ranks from def-level ranks**

```rust
// Aggregate def ranks to file level
let base_ranks: Vec<f32> = files
    .iter()
    .enumerate()
    .map(|(i, file)| {
        let start = offsets[i];
        let end = start + file.defs.len();
        def_ranks[start..end].iter().sum()
    })
    .collect();
```

- [ ] **Step 6: Build def-level caller/callee lists**

```rust
// Build def-level neighbor lists
let (def_callers, def_callees) = build_def_neighbor_lists(n_defs, &flat_def_edges, &offsets);
```

Add the function:

```rust
fn build_def_neighbor_lists(
    n: usize,
    edges: &[(u32, u32, u32)],
    offsets: &[usize],
) -> (Vec<Vec<DefId>>, Vec<Vec<DefId>>) {
    let mut incoming: Vec<Vec<(u32, u32)>> = vec![vec![]; n];
    let mut outgoing: Vec<Vec<(u32, u32)>> = vec![vec![]; n];

    for &(src, dst, w) in edges {
        let (s, d) = (src as usize, dst as usize);
        if s < n && d < n {
            incoming[d].push((src, w));
            outgoing[s].push((dst, w));
        }
    }

    // Helper to convert flat index back to DefId
    let to_def_id = |flat: u32| -> DefId {
        let flat = flat as usize;
        let file_idx = offsets.partition_point(|&o| o <= flat) - 1;
        let def_idx = flat - offsets[file_idx];
        #[expect(clippy::cast_possible_truncation)]
        (file_idx as u32, def_idx as u16)
    };

    let callers = incoming
        .into_iter()
        .map(|mut v| {
            v.sort_by(|a, b| b.1.cmp(&a.1));
            v.truncate(MAX_NEIGHBORS);
            v.into_iter().map(|(idx, _)| to_def_id(idx)).collect()
        })
        .collect();

    let callees = outgoing
        .into_iter()
        .map(|mut v| {
            v.sort_by(|a, b| b.1.cmp(&a.1));
            v.truncate(MAX_NEIGHBORS);
            v.into_iter().map(|(idx, _)| to_def_id(idx)).collect()
        })
        .collect();

    (callers, callees)
}
```

- [ ] **Step 7: Commit**

```bash
git add crates/ripvec-core/src/repo_map.rs
git commit -m "feat: definition-level call resolution, edges, and PageRank"
```

---

### Task 4: Update RepoGraph struct and all consumers

**Files:**
- Modify: `crates/ripvec-core/src/repo_map.rs` — update struct, add accessors, update render
- Modify: `crates/ripvec-core/src/hybrid.rs` — update `pagerank_lookup` for def-level

- [ ] **Step 1: Update RepoGraph struct**

Replace the current `RepoGraph` fields:

```rust
pub struct RepoGraph {
    /// Files in the repository with definitions, imports, and calls.
    pub files: Vec<FileNode>,
    /// File-level import edges (kept for backward compat and topic-sensitive rendering).
    pub edges: Vec<(u32, u32, u32)>,
    /// File-level PageRank (aggregated from def-level).
    pub base_ranks: Vec<f32>,
    /// File-level callers (aggregated from def-level).
    pub callers: Vec<Vec<u32>>,
    /// File-level callees (aggregated from def-level).
    pub callees: Vec<Vec<u32>>,
    /// Definition-level call edges: (caller_def, callee_def, weight).
    pub def_edges: Vec<(DefId, DefId, u32)>,
    /// Definition-level PageRank scores (flattened: offsets[file_idx] + def_idx).
    pub def_ranks: Vec<f32>,
    /// Definition-level callers (flattened, parallel to def_ranks).
    pub def_callers: Vec<Vec<DefId>>,
    /// Definition-level callees (flattened, parallel to def_ranks).
    pub def_callees: Vec<Vec<DefId>>,
    /// Prefix-sum offsets for flattening DefId → linear index.
    pub def_offsets: Vec<usize>,
    /// Auto-tuned alpha for search boost.
    pub alpha: f32,
}
```

Keep `edges`, `base_ranks`, `callers`, `callees` as derived fields — they're still needed by `render()` (topic-sensitive recomputation) and existing consumers. They're now computed FROM def-level data during `build_graph()`.

- [ ] **Step 2: Aggregate file-level callers/callees from def-level**

Derive file-level callers/callees by aggregating the def-level edges. In `build_graph()`:

```rust
// Derive file-level callers/callees from def-level edges
let mut file_edge_map: HashMap<(u32, u32), u32> = HashMap::new();
for &(src, dst, w) in &def_edges {
    let src_file = src.0;
    let dst_file = dst.0;
    if src_file != dst_file {
        *file_edge_map.entry((src_file, dst_file)).or_insert(0) += w;
    }
}
let edges: Vec<(u32, u32, u32)> = file_edge_map
    .into_iter()
    .map(|((src, dst), w)| (src, dst, w))
    .collect();

let (callers, callees) = build_neighbor_lists(files.len(), &edges);
```

This replaces the old import-edge-based file edges with call-edge-based file edges — structurally more accurate.

- [ ] **Step 3: Add `def_rank` accessor method**

```rust
impl RepoGraph {
    /// Get the PageRank score for a specific definition.
    #[must_use]
    pub fn def_rank(&self, did: DefId) -> f32 {
        let flat = self.def_offsets[did.0 as usize] + did.1 as usize;
        self.def_ranks.get(flat).copied().unwrap_or(0.0)
    }

    /// Look up a definition by file path and name. Returns the first match.
    #[must_use]
    pub fn find_def(&self, file_path: &str, def_name: &str) -> Option<DefId> {
        for (file_idx, file) in self.files.iter().enumerate() {
            if file.path == file_path {
                for (def_idx, def) in file.defs.iter().enumerate() {
                    if def.name == def_name {
                        #[expect(clippy::cast_possible_truncation)]
                        return Some((file_idx as u32, def_idx as u16));
                    }
                }
            }
        }
        None
    }
}
```

- [ ] **Step 4: Update `pagerank_lookup` in hybrid.rs for def-level boost**

Replace the current `pagerank_lookup` with a version that maps `(file_path, def_name)` → normalized rank:

```rust
/// Build a normalized PageRank lookup table from a [`RepoGraph`].
///
/// Returns a map from `"file_path::def_name"` to definition-level PageRank
/// normalized to [0, 1]. For chunks that don't match a specific definition,
/// `boost_with_pagerank` falls back to file-level rank.
#[must_use]
pub fn pagerank_lookup(graph: &crate::repo_map::RepoGraph) -> HashMap<String, f32> {
    let max_rank = graph.def_ranks.iter().copied().fold(0.0_f32, f32::max);
    if max_rank <= f32::EPSILON {
        return HashMap::new();
    }

    let mut map = HashMap::new();

    // Definition-level entries: "path::name" → def_rank
    for (file_idx, file) in graph.files.iter().enumerate() {
        for (def_idx, def) in file.defs.iter().enumerate() {
            let flat = graph.def_offsets[file_idx] + def_idx;
            if let Some(&rank) = graph.def_ranks.get(flat) {
                let key = format!("{}::{}", file.path, def.name);
                map.insert(key, rank / max_rank);
            }
        }
        // Also insert file-level aggregate for fallback
        map.insert(file.path.clone(), graph.base_ranks[file_idx] / max_rank);
    }

    map
}
```

- [ ] **Step 5: Update `boost_with_pagerank` to try def-level first**

In `hybrid.rs`, update `boost_with_pagerank` to look up `"path::name"` first, falling back to `"path"`:

```rust
pub fn boost_with_pagerank<S: std::hash::BuildHasher>(
    results: &mut [(usize, f32)],
    chunks: &[CodeChunk],
    pagerank_by_file: &HashMap<String, f32, S>,
    alpha: f32,
) {
    for (idx, score) in results.iter_mut() {
        if let Some(chunk) = chunks.get(*idx) {
            // Try definition-level lookup first
            let def_key = format!("{}::{}", chunk.file_path, chunk.name);
            let rank = pagerank_by_file
                .get(&def_key)
                .or_else(|| pagerank_by_file.get(&chunk.file_path))
                .copied()
                .unwrap_or(0.0);
            *score *= 1.0 + alpha * rank;
        }
    }
    results.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
}
```

- [ ] **Step 6: Update tests**

Update existing `RepoGraph` construction in tests to include the new fields. For tests that construct `RepoGraph` manually, add default empty values for the new fields:

```rust
def_edges: vec![],
def_ranks: vec![],
def_callers: vec![],
def_callees: vec![],
def_offsets: vec![0],
```

- [ ] **Step 7: Verify**

Run: `cargo test -p ripvec-core`
Run: `cargo clippy --all-targets -- -D warnings`
Run: `cargo test -p ripvec-mcp`

- [ ] **Step 8: Commit**

```bash
git add crates/ripvec-core/src/repo_map.rs crates/ripvec-core/src/hybrid.rs
git commit -m "feat: definition-level PageRank with file-level aggregation"
```

---

### Task 5: LSP call hierarchy operations

**Files:**
- Create: `crates/ripvec-mcp/src/lsp/call_hierarchy.rs`
- Modify: `crates/ripvec-mcp/src/lsp/mod.rs` — add module, wire trait methods

- [ ] **Step 1: Create call_hierarchy.rs**

Create `crates/ripvec-mcp/src/lsp/call_hierarchy.rs`:

```rust
//! LSP call hierarchy: prepareCallHierarchy, incomingCalls, outgoingCalls.
//!
//! Backed by the definition-level call graph in [`RepoGraph`].

use std::path::Path;
use std::sync::Arc;

use tower_lsp_server::jsonrpc::Result;
use tower_lsp_server::ls_types::*;

use ripvec_core::repo_map::{DefId, RepoGraph};

/// Prepare a call hierarchy item for the definition at the cursor.
pub async fn prepare(
    params: CallHierarchyPrepareParams,
    repo_graph: &Arc<std::sync::RwLock<Option<RepoGraph>>>,
    root: &Path,
) -> Result<Option<Vec<CallHierarchyItem>>> {
    let pos = &params.text_document_position_params;
    let Some(path_cow) = pos.text_document.uri.to_file_path() else {
        return Ok(None);
    };
    let path = path_cow.into_owned();
    let line = pos.position.line as u32 + 1; // LSP 0-based → 1-based

    let Ok(guard) = repo_graph.read() else {
        return Ok(None);
    };
    let Some(graph) = guard.as_ref() else {
        return Ok(None);
    };

    let rel_path = path
        .strip_prefix(root)
        .unwrap_or(&path)
        .display()
        .to_string();

    // Find the definition containing this line
    for (file_idx, file) in graph.files.iter().enumerate() {
        if file.path != rel_path {
            continue;
        }
        for (def_idx, def) in file.defs.iter().enumerate() {
            if def.start_line <= line && line <= def.end_line {
                let uri = Uri::from_file_path(if path.is_absolute() {
                    path.clone()
                } else {
                    root.join(&path)
                });
                let Some(uri) = uri else {
                    return Ok(None);
                };

                let item = CallHierarchyItem {
                    name: def.name.clone(),
                    kind: super::symbols::symbol_kind_for(&def.kind),
                    tags: None,
                    detail: def.signature.clone(),
                    uri,
                    range: Range {
                        start: Position { line: def.start_line - 1, character: 0 },
                        end: Position { line: def.end_line - 1, character: 0 },
                    },
                    selection_range: Range {
                        start: Position { line: def.start_line - 1, character: 0 },
                        end: Position { line: def.start_line - 1, character: 0 },
                    },
                    data: Some(serde_json::json!({
                        "file_idx": file_idx,
                        "def_idx": def_idx,
                    })),
                };

                return Ok(Some(vec![item]));
            }
        }
    }

    Ok(None)
}

/// Find all functions that call the given function (incoming calls).
pub async fn incoming(
    params: CallHierarchyIncomingCallsParams,
    repo_graph: &Arc<std::sync::RwLock<Option<RepoGraph>>>,
    root: &Path,
) -> Result<Option<Vec<CallHierarchyIncomingCall>>> {
    let (file_idx, def_idx) = extract_def_id(&params.item)?;

    let Ok(guard) = repo_graph.read() else {
        return Ok(None);
    };
    let Some(graph) = guard.as_ref() else {
        return Ok(None);
    };

    let flat = graph.def_offsets.get(file_idx).copied().unwrap_or(0) + def_idx;
    let Some(callers) = graph.def_callers.get(flat) else {
        return Ok(None);
    };

    let items: Vec<CallHierarchyIncomingCall> = callers
        .iter()
        .filter_map(|did| {
            let file = graph.files.get(did.0 as usize)?;
            let def = file.defs.get(did.1 as usize)?;
            let uri = Uri::from_file_path(root.join(&file.path))?;
            Some(CallHierarchyIncomingCall {
                from: CallHierarchyItem {
                    name: def.name.clone(),
                    kind: super::symbols::symbol_kind_for(&def.kind),
                    tags: None,
                    detail: def.signature.clone(),
                    uri,
                    range: Range {
                        start: Position { line: def.start_line - 1, character: 0 },
                        end: Position { line: def.end_line - 1, character: 0 },
                    },
                    selection_range: Range {
                        start: Position { line: def.start_line - 1, character: 0 },
                        end: Position { line: def.start_line - 1, character: 0 },
                    },
                    data: None,
                },
                from_ranges: vec![Range {
                    start: Position { line: def.start_line - 1, character: 0 },
                    end: Position { line: def.end_line - 1, character: 0 },
                }],
            })
        })
        .collect();

    if items.is_empty() { Ok(None) } else { Ok(Some(items)) }
}

/// Find all functions called by the given function (outgoing calls).
pub async fn outgoing(
    params: CallHierarchyOutgoingCallsParams,
    repo_graph: &Arc<std::sync::RwLock<Option<RepoGraph>>>,
    root: &Path,
) -> Result<Option<Vec<CallHierarchyOutgoingCall>>> {
    let (file_idx, def_idx) = extract_def_id(&params.item)?;

    let Ok(guard) = repo_graph.read() else {
        return Ok(None);
    };
    let Some(graph) = guard.as_ref() else {
        return Ok(None);
    };

    let flat = graph.def_offsets.get(file_idx).copied().unwrap_or(0) + def_idx;
    let Some(callees) = graph.def_callees.get(flat) else {
        return Ok(None);
    };

    let items: Vec<CallHierarchyOutgoingCall> = callees
        .iter()
        .filter_map(|did| {
            let file = graph.files.get(did.0 as usize)?;
            let def = file.defs.get(did.1 as usize)?;
            let uri = Uri::from_file_path(root.join(&file.path))?;
            Some(CallHierarchyOutgoingCall {
                to: CallHierarchyItem {
                    name: def.name.clone(),
                    kind: super::symbols::symbol_kind_for(&def.kind),
                    tags: None,
                    detail: def.signature.clone(),
                    uri,
                    range: Range {
                        start: Position { line: def.start_line - 1, character: 0 },
                        end: Position { line: def.end_line - 1, character: 0 },
                    },
                    selection_range: Range {
                        start: Position { line: def.start_line - 1, character: 0 },
                        end: Position { line: def.start_line - 1, character: 0 },
                    },
                    data: None,
                },
                from_ranges: vec![Range {
                    start: Position { line: def.start_line - 1, character: 0 },
                    end: Position { line: def.end_line - 1, character: 0 },
                }],
            })
        })
        .collect();

    if items.is_empty() { Ok(None) } else { Ok(Some(items)) }
}

/// Extract DefId from the `data` field of a CallHierarchyItem.
fn extract_def_id(item: &CallHierarchyItem) -> Result<(usize, usize)> {
    let data = item.data.as_ref().ok_or_else(|| {
        tower_lsp_server::jsonrpc::Error::invalid_params("missing data in CallHierarchyItem")
    })?;
    let file_idx = data["file_idx"]
        .as_u64()
        .ok_or_else(|| tower_lsp_server::jsonrpc::Error::invalid_params("missing file_idx"))? as usize;
    let def_idx = data["def_idx"]
        .as_u64()
        .ok_or_else(|| tower_lsp_server::jsonrpc::Error::invalid_params("missing def_idx"))? as usize;
    Ok((file_idx, def_idx))
}
```

- [ ] **Step 2: Wire into LanguageServer trait in mod.rs**

Add `mod call_hierarchy;` and add these methods to the `LanguageServer` impl:

In `initialize`, add call hierarchy capability:
```rust
call_hierarchy_provider: Some(CallHierarchyServerCapability::Simple(true)),
```

Add trait method implementations:
```rust
async fn prepare_call_hierarchy(
    &self,
    params: CallHierarchyPrepareParams,
) -> Result<Option<Vec<CallHierarchyItem>>> {
    call_hierarchy::prepare(params, &self.repo_graph, &self.project_root).await
}

async fn incoming_call(
    &self,
    params: CallHierarchyIncomingCallsParams,
) -> Result<Option<Vec<CallHierarchyIncomingCall>>> {
    call_hierarchy::incoming(params, &self.repo_graph, &self.project_root).await
}

async fn outgoing_call(
    &self,
    params: CallHierarchyOutgoingCallsParams,
) -> Result<Option<Vec<CallHierarchyOutgoingCall>>> {
    call_hierarchy::outgoing(params, &self.repo_graph, &self.project_root).await
}
```

Also make `symbols::symbol_kind_for` `pub(crate)` so `call_hierarchy.rs` can use it.

- [ ] **Step 3: Verify**

Run: `cargo check -p ripvec-mcp`
Run: `cargo clippy --all-targets -- -D warnings`

- [ ] **Step 4: Commit**

```bash
git add crates/ripvec-mcp/src/lsp/
git commit -m "feat(lsp): implement call hierarchy backed by def-level graph"
```

---

### Task 6: Full verification and push

**Files:** None (verification only)

- [ ] **Step 1: Format check**

Run: `cargo fmt --check`

- [ ] **Step 2: Clippy**

Run: `cargo clippy --all-targets -- -D warnings`

- [ ] **Step 3: Tests**

Run: `cargo test -p ripvec-core --lib 2>&1 | tail -5`
Run: `cargo test -p ripvec -p ripvec-mcp 2>&1 | tail -5`

- [ ] **Step 4: Push**

Run: `git push`

- [ ] **Step 5: Fix any issues, re-commit**
