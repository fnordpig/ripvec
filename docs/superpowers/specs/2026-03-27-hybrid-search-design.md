# Phase 4: Hybrid Search Design

**Date:** 2026-03-27
**Status:** Approved
**Goal:** Add BM25 keyword search alongside semantic search, fused via Reciprocal Rank Fusion. Catches identifiers that BERT tokenizers butcher (`parseJsonConfig` -> `parse`, `##J`, `##son`, etc).

---

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| BM25 tokenizer | Custom `CodeTokenizer` registered in tantivy | camelCase/snake_case splitting as tantivy `Tokenizer` impl |
| Index construction | At `SearchIndex::new()` time, concurrent with GPU embedding | Hides BM25 index cost in idle CPU budget |
| Score fusion | Reciprocal Rank Fusion (k=60) | Industry standard, no normalization needed, robust |
| Tantivy fields | `name` (3x boost), `file_path` (1.5x), `body` (1x) | Name boost is where BM25 shines for identifier search |
| Result merging | Union — chunk from either list contributes | Ensures exact matches surface even when semantic misses |
| Search modes | `--mode hybrid\|semantic\|keyword`, default hybrid | Explicit control for CLI and MCP; hybrid is the default |

---

## Architecture

### HybridIndex

Wraps existing `SearchIndex` + tantivy `Index`. Single construction point.

```
                 +---------------+
  chunks ------->| HybridIndex   |<--- query + mode
                 |               |
          +------+   search()   +------+
          |      +---------------+      |
          v                             v
   SearchIndex.rank()         tantivy BM25 query
   -> Vec<(idx, score)>       -> Vec<(idx, score)>
          |                             |
          +----------+------------------+
                     v
              RRF fusion (k=60)
              -> Vec<(idx, rrf_score)>
```

### New Types

```rust
// crates/ripvec-core/src/index.rs

pub enum SearchMode {
    /// RRF fusion of semantic + BM25 (default)
    Hybrid,
    /// Embedding cosine similarity only
    Semantic,
    /// BM25 keyword scoring only
    Keyword,
}

pub struct HybridIndex {
    /// Existing semantic index (embeddings, chunks, compressed, MRL cascade)
    pub semantic: SearchIndex,
    /// tantivy full-text index (in-memory or on-disk for --index mode)
    bm25: tantivy::Index,
    /// tantivy reader (reusable across queries)
    reader: tantivy::IndexReader,
    /// Schema field handles
    field_name: tantivy::schema::Field,
    field_path: tantivy::schema::Field,
    field_body: tantivy::schema::Field,
    field_chunk_id: tantivy::schema::Field,
}
```

### Custom Tantivy Tokenizer: `CodeTokenizer`

Registered as `"code"` on the tantivy index. Implements `tantivy::tokenizer::Tokenizer`.

Pipeline per input string:
1. Split on whitespace + punctuation (ASCII word boundaries)
2. For each token, detect camelCase and snake_case boundaries
3. Emit the original lowercased form AND all sub-tokens
4. Examples:
   - `parseJsonConfig` -> `[parsejsonconfig, parse, json, config]`
   - `my_func_name` -> `[my_func_name, my, func, name]`
   - `MetalDriver` -> `[metaldriver, metal, driver]`
   - `HTML5Parser` -> `[html5parser, html5, parser]` (digits stay attached to preceding alpha)
   - `std::vec::Vec` -> after punctuation split: `[std, vec, vec]`

Implementation: a `TokenFilter` wrapping tantivy's `SimpleTokenizer` + `LowerCaser`, then emitting additional sub-tokens from camel/snake splitting.

### Tantivy Schema

```rust
fn build_schema() -> (Schema, Field, Field, Field, Field) {
    let mut builder = Schema::builder();
    let text_opts = TextOptions::default().set_indexing_options(
        TextFieldIndexing::default()
            .set_tokenizer("code")
            .set_index_option(IndexRecordOption::WithFreqsAndPositions),
    );
    let name = builder.add_text_field("name", text_opts.clone());
    let file_path = builder.add_text_field("file_path", text_opts.clone());
    let body = builder.add_text_field("body", text_opts);
    let chunk_id = builder.add_u64_field("chunk_id", INDEXED | STORED);
    (builder.build(), name, file_path, body, chunk_id)
}
```

No content stored in tantivy — chunks already live in `SearchIndex.chunks`. Tantivy is scoring-only. The `chunk_id` field maps back to the `Vec<CodeChunk>` index.

### RRF Fusion

```rust
/// Reciprocal Rank Fusion: combine two ranked lists.
///
/// Each result's score = sum of 1/(k + rank) across the lists it appears in.
/// Union semantics: a chunk from either list contributes.
fn rrf_fuse(
    semantic: &[(usize, f32)],  // (chunk_idx, cosine_sim), sorted desc
    bm25: &[(usize, f32)],      // (chunk_idx, bm25_score), sorted desc
    k: f32,                      // default 60.0
    top_n: usize,
) -> Vec<(usize, f32)> {
    let mut scores: HashMap<usize, f32> = HashMap::new();
    for (rank, &(idx, _)) in semantic.iter().enumerate() {
        *scores.entry(idx).or_default() += 1.0 / (k + rank as f32 + 1.0);
    }
    for (rank, &(idx, _)) in bm25.iter().enumerate() {
        *scores.entry(idx).or_default() += 1.0 / (k + rank as f32 + 1.0);
    }
    let mut fused: Vec<_> = scores.into_iter().collect();
    fused.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    fused.truncate(top_n);
    fused
}
```

### Concurrency: Build BM25 During Embedding

```
Timeline:
  CPU: [walk] [chunk] [tokenize] [build tantivy --------] [fuse+rank]
  GPU:                            [embed batch 1][batch 2]...[query]
```

After chunking + tokenization, spawn the tantivy index build on a rayon thread via `rayon::join()`:

```rust
let (embeddings, bm25_index) = rayon::join(
    || embed_distributed(backend, &encodings, profiler),
    || build_tantivy_index(&chunks),
);
```

Zero added wall time on GPU-bound workloads. The tantivy build is CPU-only (~50-100ms for 1K chunks) and the GPU embedding takes 3-16s.

---

## Integration Points

### SearchConfig changes

```rust
// crates/ripvec-core/src/embed.rs
pub struct SearchConfig {
    // ... existing fields ...

    /// Search mode: hybrid (default), semantic, or keyword.
    pub mode: SearchMode,
}
```

### CLI changes

```rust
// crates/ripvec/src/cli.rs
/// Search mode: hybrid (semantic + BM25), semantic-only, or keyword-only.
#[arg(long, value_enum, default_value_t = SearchMode::Hybrid)]
pub mode: SearchMode,
```

### MCP changes

```rust
// crates/ripvec-mcp/src/tools.rs — search_code params
pub struct SearchParams {
    pub query: String,
    pub top_k: Option<usize>,
    pub threshold: Option<f32>,
    pub offset: Option<usize>,
    pub mode: Option<SearchMode>,  // NEW: default "hybrid"
}
```

### HybridIndex API

```rust
impl HybridIndex {
    /// Build from chunks + embeddings. Constructs tantivy index from chunk content.
    pub fn new(
        chunks: Vec<CodeChunk>,
        embeddings: Vec<Vec<f32>>,
        cascade_dim: Option<usize>,
    ) -> Self { ... }

    /// Build from chunks + embeddings + pre-built tantivy index.
    /// Used when loading from persistent cache.
    pub fn from_parts(
        semantic: SearchIndex,
        bm25: tantivy::Index,
    ) -> Self { ... }

    /// Search with the given mode.
    pub fn search(
        &self,
        query_embedding: &[f32],
        query_text: &str,
        top_k: usize,
        threshold: f32,
        mode: SearchMode,
    ) -> Vec<(usize, f32)> {
        match mode {
            SearchMode::Semantic => self.semantic.rank(query_embedding, threshold),
            SearchMode::Keyword => self.bm25_rank(query_text, top_k),
            SearchMode::Hybrid => {
                let sem = self.semantic.rank(query_embedding, threshold);
                let bm25 = self.bm25_rank(query_text, top_k.max(100));
                rrf_fuse(&sem, &bm25, 60.0, top_k)
            }
        }
    }

    /// BM25 keyword search via tantivy.
    fn bm25_rank(&self, query_text: &str, top_k: usize) -> Vec<(usize, f32)> {
        // Build a BooleanQuery with per-field TermQueries:
        //   name terms: boost 3.0
        //   file_path terms: boost 1.5
        //   body terms: boost 1.0
        // tantivy applies boosts at query time via BoostQuery wrapper.
        // Return (chunk_idx, bm25_score) sorted descending
    }
}
```

### Persistent Index (--index mode)

The tantivy index serializes to disk natively. Store alongside the existing rkyv-serialized embedding cache:

```
.ripvec/
  index.bin          # existing: rkyv-serialized SearchIndex
  tantivy/           # NEW: tantivy index directory
    meta.json
    *.managed
```

On load, reconstruct `HybridIndex::from_parts()` from both. On model change (detected via `model_repo` field), invalidate both.

---

## New Dependency

```toml
# crates/ripvec-core/Cargo.toml
[dependencies]
tantivy = "0.25"
```

No aho-corasick needed — tantivy handles multi-term queries internally with its own optimized posting list intersection.

---

## What This Does NOT Include

- Query syntax (boolean operators, field-specific queries) — tantivy supports this but we parse the query as a simple string for now
- Learned ranking / LambdaMART — RRF is sufficient
- Separate `--alpha` weight tuning — RRF doesn't need it
- Stemming — code identifiers shouldn't be stemmed (`parsing` != `parse` in code)
- Stop word removal — code has different stop words than prose, skip for now

---

## Success Criteria

1. `ripvec "parseJsonConfig"` returns the function definition as #1 result (currently fails with semantic-only)
2. `ripvec "error handling in the Metal driver"` still returns relevant semantic results (BM25 alone wouldn't catch this)
3. `ripvec "MetalDriver" --mode keyword` returns exact matches instantly without embedding
4. No regression in semantic-only search quality (--mode semantic = current behavior)
5. Wall time overhead < 5% for hybrid vs semantic-only (tantivy build hidden in GPU idle)
6. MCP search_code tool works with all three modes
