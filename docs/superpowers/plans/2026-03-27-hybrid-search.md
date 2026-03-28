# Phase 4: Hybrid Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add BM25 keyword search via tantivy alongside semantic search, fused with Reciprocal Rank Fusion, so identifier searches like `parseJsonConfig` return exact matches.

**Architecture:** New `bm25.rs` module owns the `CodeTokenizer`, tantivy schema, and `Bm25Index`. New `hybrid.rs` wraps `SearchIndex` + `Bm25Index` into `HybridIndex` with RRF fusion. `SearchMode` enum gates behavior. Tantivy index built concurrently with GPU embedding via `rayon::join()`.

**Tech Stack:** tantivy 0.25, Rust 2024 edition, existing ripvec-core infrastructure

**Spec:** `docs/superpowers/specs/2026-03-27-hybrid-search-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `crates/ripvec-core/Cargo.toml` | Modify | Add `tantivy = "0.25"` |
| `crates/ripvec-core/src/lib.rs` | Modify | Add `pub mod bm25; pub mod hybrid;` |
| `crates/ripvec-core/src/bm25.rs` | Create | `CodeTokenizer`, tantivy schema, `Bm25Index` |
| `crates/ripvec-core/src/hybrid.rs` | Create | `SearchMode`, `HybridIndex`, RRF fusion |
| `crates/ripvec-core/src/embed.rs` | Modify | Add `mode: SearchMode` to `SearchConfig`, concurrent build |
| `crates/ripvec/src/cli.rs` | Modify | Add `--mode` flag |
| `crates/ripvec/src/main.rs` | Modify | Pass mode through pipeline |
| `crates/ripvec-mcp/src/tools.rs` | Modify | Add `mode` parameter to search tools |

---

## Task 1: Add tantivy dependency

**Agent:** Haiku (trivial change)
**Parallel:** Yes — independent

**Files:**
- Modify: `crates/ripvec-core/Cargo.toml`

- [ ] **Step 1: Add tantivy to dependencies**

Add after the `ndarray = "0.17"` line (line 27):

```toml
tantivy = "0.25"
```

- [ ] **Step 2: Verify compilation**

Run: `cargo check -p ripvec-core`

- [ ] **Step 3: Commit**

```bash
git add crates/ripvec-core/Cargo.toml
git commit -m "deps: add tantivy 0.25 for BM25 hybrid search"
```

---

## Task 2: Implement `CodeTokenizer` and `Bm25Index`

**Agent:** Sonnet (focused module, clear spec)
**Parallel:** Yes — independent of Tasks 1, 3

**Files:**
- Create: `crates/ripvec-core/src/bm25.rs`
- Modify: `crates/ripvec-core/src/lib.rs` (add `pub mod bm25;`)

- [ ] **Step 1: Add module declaration**

In `crates/ripvec-core/src/lib.rs`, add after `pub mod backend;` (line 7):

```rust
pub mod bm25;
```

- [ ] **Step 2: Create `bm25.rs` with CodeTokenizer**

Create `crates/ripvec-core/src/bm25.rs`:

```rust
//! BM25 full-text search via tantivy with code-aware tokenization.
//!
//! [`CodeTokenizer`] splits camelCase and snake_case identifiers into
//! sub-tokens so that `parseJsonConfig` matches queries for `parse`,
//! `json`, or `config`. Registered as the `"code"` tokenizer on the
//! tantivy index.

use tantivy::schema::{
    Field, IndexRecordOption, Schema, TextFieldIndexing, TextOptions, INDEXED, STORED,
};
use tantivy::tokenizer::{
    LowerCaser, SimpleTokenizer, TextAnalyzer, Token, TokenFilter, TokenStream, Tokenizer,
};
use tantivy::{
    collector::TopDocs,
    query::{BooleanQuery, BoostQuery, Occur, QueryParser},
    Index, IndexReader, IndexWriter, ReloadPolicy, TantivyDocument,
};

use crate::chunk::CodeChunk;

/// A token filter that splits camelCase and snake_case identifiers
/// into sub-tokens while preserving the original.
///
/// `parseJsonConfig` → `[parsejsonconfig, parse, json, config]`
/// `my_func_name` → `[my_func_name, my, func, name]`
#[derive(Clone)]
struct CodeSplitFilter;

impl TokenFilter for CodeSplitFilter {
    type Tokenizer<T: Tokenizer> = CodeSplitFilterWrapper<T>;

    fn transform<T: Tokenizer>(self, tokenizer: T) -> Self::Tokenizer<T> {
        CodeSplitFilterWrapper {
            inner: tokenizer,
        }
    }
}

#[derive(Clone)]
struct CodeSplitFilterWrapper<T> {
    inner: T,
}

impl<T: Tokenizer> Tokenizer for CodeSplitFilterWrapper<T> {
    type TokenStream<'a> = CodeSplitTokenStream<T::TokenStream<'a>>;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> Self::TokenStream<'a> {
        CodeSplitTokenStream {
            inner: self.inner.token_stream(text),
            pending: Vec::new(),
            current: Token::default(),
        }
    }
}

struct CodeSplitTokenStream<S> {
    inner: S,
    pending: Vec<String>,
    current: Token,
}

/// Split a token into camelCase / snake_case sub-parts.
/// Returns sub-parts only (caller keeps the original).
fn split_code_identifier(text: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();

    let chars: Vec<char> = text.chars().collect();
    for i in 0..chars.len() {
        let c = chars[i];
        if c == '_' {
            // snake_case boundary
            if !current.is_empty() {
                parts.push(std::mem::take(&mut current));
            }
        } else if c.is_ascii_uppercase() && i > 0 && chars[i - 1].is_ascii_lowercase() {
            // camelCase boundary: aB
            if !current.is_empty() {
                parts.push(std::mem::take(&mut current));
            }
            current.push(c.to_ascii_lowercase());
        } else if c.is_ascii_uppercase()
            && i + 1 < chars.len()
            && chars[i + 1].is_ascii_lowercase()
            && i > 0
            && chars[i - 1].is_ascii_uppercase()
        {
            // Acronym boundary: ABc → [A, Bc] — split before the last uppercase
            if !current.is_empty() {
                parts.push(std::mem::take(&mut current));
            }
            current.push(c.to_ascii_lowercase());
        } else {
            current.push(c.to_ascii_lowercase());
        }
    }
    if !current.is_empty() {
        parts.push(current);
    }

    // Only return sub-parts if we actually split something
    if parts.len() <= 1 {
        Vec::new()
    } else {
        parts
    }
}

impl<S: TokenStream> TokenStream for CodeSplitTokenStream<S> {
    fn advance(&mut self) -> bool {
        // Drain pending sub-tokens first
        if let Some(text) = self.pending.pop() {
            self.current.text = text;
            return true;
        }

        // Advance the inner stream
        if !self.inner.advance() {
            return false;
        }

        let token = self.inner.token();
        self.current = token.clone();

        // Generate sub-tokens from camelCase/snake_case
        let sub_parts = split_code_identifier(&token.text);
        if !sub_parts.is_empty() {
            // Push sub-tokens to pending (reversed so pop returns in order)
            for part in sub_parts.into_iter().rev() {
                self.pending.push(part);
            }
        }

        true
    }

    fn token(&self) -> &Token {
        &self.current
    }

    fn token_mut(&mut self) -> &mut Token {
        &mut self.current
    }
}

/// Build the code-aware text analyzer: `SimpleTokenizer` → `LowerCaser` → `CodeSplitFilter`.
fn code_analyzer() -> TextAnalyzer {
    TextAnalyzer::builder(SimpleTokenizer::default())
        .filter(LowerCaser)
        .filter(CodeSplitFilter)
        .build()
}

/// Schema fields for the BM25 index.
pub struct BM25Fields {
    pub name: Field,
    pub file_path: Field,
    pub body: Field,
    pub chunk_id: Field,
}

/// Build the tantivy schema with code-aware tokenization.
fn build_schema() -> (Schema, BM25Fields) {
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
    let schema = builder.build();
    (schema, BM25Fields { name, file_path, body, chunk_id })
}

/// BM25 full-text index backed by tantivy.
pub struct Bm25Index {
    index: Index,
    reader: IndexReader,
    fields: BM25Fields,
}

impl Bm25Index {
    /// Build a BM25 index from code chunks.
    ///
    /// Registers the `"code"` tokenizer, indexes each chunk's `name`,
    /// `file_path`, and `content` fields, and returns a searchable index.
    pub fn build(chunks: &[CodeChunk]) -> crate::Result<Self> {
        let (schema, fields) = build_schema();
        let index = Index::create_in_ram(schema);
        index.tokenizers().register("code", code_analyzer());

        let mut writer: IndexWriter = index
            .writer(50_000_000) // 50MB heap
            .map_err(|e| crate::Error::Other(e.into()))?;

        for (i, chunk) in chunks.iter().enumerate() {
            let mut doc = TantivyDocument::new();
            doc.add_text(fields.name, &chunk.name);
            doc.add_text(fields.file_path, &chunk.file_path);
            doc.add_text(fields.body, &chunk.content);
            doc.add_u64(fields.chunk_id, i as u64);
            writer
                .add_document(doc)
                .map_err(|e| crate::Error::Other(e.into()))?;
        }

        writer
            .commit()
            .map_err(|e| crate::Error::Other(e.into()))?;

        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()
            .map_err(|e| crate::Error::Other(e.into()))?;

        Ok(Self { index, reader, fields })
    }

    /// Search for chunks matching `query_text`, returning `(chunk_idx, score)`.
    ///
    /// Searches across `name` (3× boost), `file_path` (1.5× boost), and
    /// `body` (1× boost) using tantivy's BM25 scoring.
    pub fn search(&self, query_text: &str, top_k: usize) -> Vec<(usize, f32)> {
        let searcher = self.reader.searcher();

        // Build per-field queries with boosts
        let name_parser = QueryParser::for_index(&self.index, vec![self.fields.name]);
        let path_parser = QueryParser::for_index(&self.index, vec![self.fields.file_path]);
        let body_parser = QueryParser::for_index(&self.index, vec![self.fields.body]);

        let mut subqueries: Vec<(Occur, Box<dyn tantivy::query::Query>)> = Vec::new();

        if let Ok(q) = name_parser.parse_query(query_text) {
            subqueries.push((Occur::Should, Box::new(BoostQuery::new(q, 3.0))));
        }
        if let Ok(q) = path_parser.parse_query(query_text) {
            subqueries.push((Occur::Should, Box::new(BoostQuery::new(q, 1.5))));
        }
        if let Ok(q) = body_parser.parse_query(query_text) {
            subqueries.push((Occur::Should, Box::new(BoostQuery::new(q, 1.0))));
        }

        if subqueries.is_empty() {
            return Vec::new();
        }

        let query = BooleanQuery::new(subqueries);
        let top_docs = match searcher.search(&query, &TopDocs::with_limit(top_k)) {
            Ok(docs) => docs,
            Err(_) => return Vec::new(),
        };

        top_docs
            .into_iter()
            .filter_map(|(score, doc_addr)| {
                let doc: TantivyDocument = searcher.doc(doc_addr).ok()?;
                let chunk_id = doc.get_first(self.fields.chunk_id)?.as_u64()? as usize;
                Some((chunk_id, score))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_camel_case() {
        assert_eq!(
            split_code_identifier("parsejsonconfig"),
            Vec::<String>::new(), // no split — all lowercase
        );
        assert_eq!(
            split_code_identifier("parseJsonConfig"),
            vec!["parse", "json", "config"],
        );
    }

    #[test]
    fn split_snake_case() {
        assert_eq!(
            split_code_identifier("my_func_name"),
            vec!["my", "func", "name"],
        );
    }

    #[test]
    fn split_screaming_snake() {
        assert_eq!(
            split_code_identifier("MAX_BATCH_SIZE"),
            vec!["max", "batch", "size"],
        );
    }

    #[test]
    fn split_mixed() {
        assert_eq!(
            split_code_identifier("MetalDriver"),
            vec!["metal", "driver"],
        );
    }

    #[test]
    fn no_split_single_word() {
        assert_eq!(
            split_code_identifier("parser"),
            Vec::<String>::new(),
        );
    }

    #[test]
    fn bm25_index_search() {
        let chunks = vec![
            CodeChunk {
                file_path: "src/parser.rs".into(),
                name: "parseJsonConfig".into(),
                kind: "function_item".into(),
                start_line: 10,
                end_line: 20,
                content: "fn parseJsonConfig(input: &str) -> Config { }".into(),
                enriched_content: String::new(),
            },
            CodeChunk {
                file_path: "src/main.rs".into(),
                name: "main".into(),
                kind: "function_item".into(),
                start_line: 1,
                end_line: 5,
                content: "fn main() { println!(\"hello\"); }".into(),
                enriched_content: String::new(),
            },
        ];

        let index = Bm25Index::build(&chunks).unwrap();
        let results = index.search("parseJsonConfig", 10);

        // The function named parseJsonConfig should rank first
        assert!(!results.is_empty());
        assert_eq!(results[0].0, 0); // chunk index 0
    }

    #[test]
    fn bm25_camel_case_subtoken_match() {
        let chunks = vec![
            CodeChunk {
                file_path: "src/parser.rs".into(),
                name: "parseJsonConfig".into(),
                kind: "function_item".into(),
                start_line: 10,
                end_line: 20,
                content: "fn parseJsonConfig(input: &str) -> Config { }".into(),
                enriched_content: String::new(),
            },
        ];

        let index = Bm25Index::build(&chunks).unwrap();

        // Searching for "json" should match via camelCase sub-token split
        let results = index.search("json", 10);
        assert!(!results.is_empty());
        assert_eq!(results[0].0, 0);
    }
}
```

- [ ] **Step 3: Verify compilation and run tests**

Run: `cargo test -p ripvec-core -- bm25 --nocapture`
Expected: all 6 tests pass

- [ ] **Step 4: Commit**

```bash
git add crates/ripvec-core/src/bm25.rs crates/ripvec-core/src/lib.rs
git commit -m "feat(bm25): CodeTokenizer + Bm25Index with camelCase/snake_case splitting"
```

---

## Task 3: Implement `HybridIndex` with RRF fusion

**Agent:** Sonnet (focused module, clear interfaces)
**Parallel:** Yes — independent of Tasks 1, 2

**Files:**
- Create: `crates/ripvec-core/src/hybrid.rs`
- Modify: `crates/ripvec-core/src/lib.rs` (add `pub mod hybrid;`)

- [ ] **Step 1: Add module declaration**

In `crates/ripvec-core/src/lib.rs`, add after `pub mod bm25;`:

```rust
pub mod hybrid;
```

- [ ] **Step 2: Create `hybrid.rs`**

Create `crates/ripvec-core/src/hybrid.rs`:

```rust
//! Hybrid search combining semantic embeddings with BM25 keyword search.
//!
//! [`HybridIndex`] wraps a [`SearchIndex`] and a [`Bm25Index`], fusing
//! results via Reciprocal Rank Fusion (RRF).

use std::cmp::Ordering;
use std::collections::HashMap;

use crate::bm25::Bm25Index;
use crate::chunk::CodeChunk;
use crate::index::SearchIndex;

/// Search mode: how to combine semantic and keyword results.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SearchMode {
    /// RRF fusion of semantic + BM25 (default).
    #[default]
    Hybrid,
    /// Embedding cosine similarity only.
    Semantic,
    /// BM25 keyword scoring only.
    Keyword,
}

impl std::fmt::Display for SearchMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Hybrid => write!(f, "hybrid"),
            Self::Semantic => write!(f, "semantic"),
            Self::Keyword => write!(f, "keyword"),
        }
    }
}

impl std::str::FromStr for SearchMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "hybrid" => Ok(Self::Hybrid),
            "semantic" => Ok(Self::Semantic),
            "keyword" => Ok(Self::Keyword),
            other => Err(format!("unknown search mode: {other}")),
        }
    }
}

/// Combined semantic + BM25 search index.
pub struct HybridIndex {
    /// Semantic embedding index.
    pub semantic: SearchIndex,
    /// BM25 keyword index.
    bm25: Bm25Index,
}

impl HybridIndex {
    /// Build from chunks + embeddings. Constructs both semantic and BM25 indices.
    pub fn new(
        chunks: Vec<CodeChunk>,
        embeddings: Vec<Vec<f32>>,
        cascade_dim: Option<usize>,
    ) -> crate::Result<Self> {
        let bm25 = Bm25Index::build(&chunks)?;
        let semantic = SearchIndex::new(chunks, embeddings, cascade_dim);
        Ok(Self { semantic, bm25 })
    }

    /// Build from pre-built components.
    pub fn from_parts(semantic: SearchIndex, bm25: Bm25Index) -> Self {
        Self { semantic, bm25 }
    }

    /// Search with the given mode.
    ///
    /// - `query_embedding`: L2-normalized embedding of the query (unused in Keyword mode)
    /// - `query_text`: raw query string (unused in Semantic mode)
    /// - `top_k`: max results to return
    /// - `threshold`: minimum cosine similarity for semantic results
    /// - `mode`: how to combine results
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
            SearchMode::Keyword => self.bm25.search(query_text, top_k),
            SearchMode::Hybrid => {
                let sem = self.semantic.rank(query_embedding, threshold);
                // Over-retrieve BM25 for better RRF fusion
                let bm25 = self.bm25.search(query_text, top_k.max(100));
                let mut fused = rrf_fuse(&sem, &bm25, 60.0);
                fused.truncate(top_k);
                fused
            }
        }
    }

    /// Access the chunks.
    pub fn chunks(&self) -> &[CodeChunk] {
        &self.semantic.chunks
    }
}

/// Reciprocal Rank Fusion: combine two ranked lists.
///
/// Each result's score = sum of `1/(k + rank)` across the lists it appears in.
/// Union semantics: a chunk from either list contributes.
fn rrf_fuse(
    semantic: &[(usize, f32)],
    bm25: &[(usize, f32)],
    k: f32,
) -> Vec<(usize, f32)> {
    let mut scores: HashMap<usize, f32> =
        HashMap::with_capacity(semantic.len() + bm25.len());
    for (rank, &(idx, _)) in semantic.iter().enumerate() {
        *scores.entry(idx).or_default() += 1.0 / (k + rank as f32 + 1.0);
    }
    for (rank, &(idx, _)) in bm25.iter().enumerate() {
        *scores.entry(idx).or_default() += 1.0 / (k + rank as f32 + 1.0);
    }
    let mut fused: Vec<(usize, f32)> = scores.into_iter().collect();
    fused.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    fused
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rrf_union_semantics() {
        // Semantic ranks: chunk 0, chunk 1, chunk 2
        let sem = vec![(0, 0.9), (1, 0.8), (2, 0.7)];
        // BM25 ranks: chunk 3, chunk 0, chunk 4
        let bm25 = vec![(3, 5.0), (0, 3.0), (4, 1.0)];

        let fused = rrf_fuse(&sem, &bm25, 60.0);

        // Chunk 0 appears in both → highest RRF score
        assert_eq!(fused[0].0, 0);
        // All 5 chunks should appear (union)
        assert_eq!(fused.len(), 5);
    }

    #[test]
    fn rrf_single_list() {
        let sem = vec![(0, 0.9), (1, 0.8)];
        let bm25 = vec![];

        let fused = rrf_fuse(&sem, &bm25, 60.0);
        assert_eq!(fused.len(), 2);
        assert_eq!(fused[0].0, 0);
    }

    #[test]
    fn search_mode_roundtrip() {
        assert_eq!("hybrid".parse::<SearchMode>().unwrap(), SearchMode::Hybrid);
        assert_eq!("semantic".parse::<SearchMode>().unwrap(), SearchMode::Semantic);
        assert_eq!("keyword".parse::<SearchMode>().unwrap(), SearchMode::Keyword);
        assert!("invalid".parse::<SearchMode>().is_err());
    }
}
```

- [ ] **Step 3: Verify compilation and run tests**

Run: `cargo test -p ripvec-core -- hybrid --nocapture`
Expected: all 3 tests pass

- [ ] **Step 4: Commit**

```bash
git add crates/ripvec-core/src/hybrid.rs crates/ripvec-core/src/lib.rs
git commit -m "feat(hybrid): HybridIndex with RRF fusion + SearchMode enum"
```

---

## Task 4: Wire into embed pipeline + CLI + MCP

**Agent:** Opus (touches multiple files, requires understanding of data flow)
**Parallel:** No — depends on Tasks 1, 2, 3

**Files:**
- Modify: `crates/ripvec-core/src/embed.rs:39-78` (SearchConfig + search function)
- Modify: `crates/ripvec/src/cli.rs:12-42` (add --mode flag)
- Modify: `crates/ripvec/src/main.rs` (pass mode + query_text through)
- Modify: `crates/ripvec-mcp/src/tools.rs:180-210` (add mode param to run_search)

- [ ] **Step 1: Add `mode` to `SearchConfig`**

In `crates/ripvec-core/src/embed.rs`, add to the `SearchConfig` struct after the `file_type` field (line 65):

```rust
    /// Search mode: hybrid (default), semantic, or keyword.
    pub mode: crate::hybrid::SearchMode,
```

And in the `Default` impl (line 68-78), add:

```rust
            mode: crate::hybrid::SearchMode::Hybrid,
```

- [ ] **Step 2: Change `search()` to return `HybridIndex` and accept query_text**

In `crates/ripvec-core/src/embed.rs`, modify the `search()` function (line 228-295) to:
1. Build `HybridIndex` instead of doing inline ranking
2. Use `rayon::join()` to build tantivy concurrently with embedding
3. Use `HybridIndex::search()` with the configured mode

Replace the search function body from line 243 (after the `embed_all` call) through the end of the function with:

```rust
    // Build HybridIndex: semantic index from embeddings, BM25 from chunks
    // BM25 build runs on CPU concurrent with GPU embedding (already complete here)
    let bm25 = crate::bm25::Bm25Index::build(&chunks)?;
    let hybrid = crate::hybrid::HybridIndex::from_parts(
        crate::index::SearchIndex::new(chunks, embeddings, cfg.cascade_dim),
        bm25,
    );

    let t_query_start = std::time::Instant::now();

    // Phase 5: Embed query (using the primary backend) — skip if keyword-only
    let query_embedding = if cfg.mode == crate::hybrid::SearchMode::Keyword {
        vec![0.0; backends[0].hidden_dim()]
    } else {
        let _span = info_span!("embed_query").entered();
        let _guard = profiler.phase("embed_query");
        let t_tok = std::time::Instant::now();
        let enc = tokenize(query, tokenizer, cfg.max_tokens, backends[0].max_tokens())?;
        let tok_ms = t_tok.elapsed().as_secs_f64() * 1000.0;
        let t_emb = std::time::Instant::now();
        let mut results = backends[0].embed_batch(&[enc])?;
        let emb_ms = t_emb.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "[search] query: tokenize={tok_ms:.1}ms embed={emb_ms:.1}ms total_since_embed_all={:.1}ms",
            t_query_start.elapsed().as_secs_f64() * 1000.0
        );
        results.pop().ok_or_else(|| {
            crate::Error::Other(anyhow::anyhow!("backend returned no embedding for query"))
        })?
    };

    // Phase 6: Rank using hybrid search
    let ranked = {
        let _span = info_span!("rank", chunk_count = hybrid.chunks().len()).entered();
        let guard = profiler.phase("rank");

        let results = hybrid.search(
            &query_embedding,
            query,
            top_k,
            threshold,
            cfg.mode,
        );

        guard.set_detail(format!(
            "top {} from {} (mode: {})",
            top_k.min(results.len()),
            hybrid.chunks().len(),
            cfg.mode,
        ));
        results
    };

    let results: Vec<SearchResult> = ranked
        .into_iter()
        .map(|(idx, score)| SearchResult {
            chunk: hybrid.chunks()[idx].clone(),
            similarity: score,
        })
        .collect();

    Ok(results)
```

Note: the `threshold` parameter needs to be passed in. Read the current function signature — it should already have `threshold` via the CLI. If not, check `run_oneshot` in `main.rs` to see where threshold is passed.

- [ ] **Step 3: Add `--mode` flag to CLI**

In `crates/ripvec/src/cli.rs`, add after the `profile` field (around line 104):

```rust
    /// Search mode: hybrid (semantic + BM25), semantic-only, or keyword-only.
    ///
    /// Hybrid (default) fuses results from both using Reciprocal Rank Fusion.
    /// Semantic is the traditional embedding-only search.
    /// Keyword uses BM25 full-text scoring (no embeddings needed for query).
    #[arg(long, default_value = "hybrid")]
    pub mode: String,
```

- [ ] **Step 4: Pass mode through in `main.rs`**

In `crates/ripvec/src/main.rs`, in the `load_pipeline` or search config construction, add:

```rust
    let mode: ripvec_core::hybrid::SearchMode = args.mode.parse().unwrap_or_default();
```

And set it on the `SearchConfig`:

```rust
    search_cfg.mode = mode;
```

- [ ] **Step 5: Add mode to MCP search tool**

In `crates/ripvec-mcp/src/tools.rs`, the `run_search` method needs to accept a `mode` parameter. Add `mode: &str` to the signature and parse it:

```rust
    let mode: ripvec_core::hybrid::SearchMode = mode.parse().unwrap_or_default();
```

Then use it when ranking. The MCP server builds its own index — update the ranking call to use `HybridIndex::search()` instead of `SearchIndex::rank()`.

- [ ] **Step 6: Verify full workspace compiles**

Run: `cargo check --workspace`

- [ ] **Step 7: Run all tests**

Run: `cargo test --workspace`

- [ ] **Step 8: Smoke test**

```bash
cargo build --release
# Hybrid (default)
./target/release/ripvec "parseJsonConfig" -n 5
# Keyword only
./target/release/ripvec "MetalDriver" --mode keyword -n 5
# Semantic only (should match previous behavior)
./target/release/ripvec "error handling in driver" --mode semantic -n 5
```

- [ ] **Step 9: Commit**

```bash
git add crates/ripvec-core/src/embed.rs crates/ripvec/src/cli.rs crates/ripvec/src/main.rs crates/ripvec-mcp/src/tools.rs
git commit -m "feat: wire hybrid search into CLI, embed pipeline, and MCP server"
```

---

## Task 5: Concurrent BM25 build during embedding

**Agent:** Sonnet (targeted optimization in embed.rs)
**Parallel:** No — depends on Task 4

**Files:**
- Modify: `crates/ripvec-core/src/embed.rs` (embed_all function)

- [ ] **Step 1: Move BM25 build to run concurrent with embedding**

In `embed_all()` (or the `search()` function), change from sequential:

```rust
let (chunks, embeddings) = embed_all(...)?;
let bm25 = Bm25Index::build(&chunks)?;
```

To concurrent via `rayon::join()`:

```rust
// After chunking + tokenization, build BM25 concurrently with GPU embedding
let chunks_for_bm25 = chunks.clone();
let ((chunks, embeddings), bm25) = rayon::join(
    || embed_all_inner(backends, &all_encodings, &chunks, profiler),
    || crate::bm25::Bm25Index::build(&chunks_for_bm25),
);
let bm25 = bm25?;
```

This hides the ~50-100ms BM25 build inside the 3-16s GPU embedding phase.

Note: this requires restructuring `embed_all` to separate the walk+chunk+tokenize phase from the embed phase. The walk+chunk+tokenize must complete first (BM25 needs chunks), then embedding and BM25 build run in parallel.

- [ ] **Step 2: Benchmark**

```bash
cargo build --release
# Compare wall time with --profile
./target/release/ripvec "test query" --mode hybrid --profile 2>&1 | grep 'total:'
./target/release/ripvec "test query" --mode semantic --profile 2>&1 | grep 'total:'
```

Expected: hybrid wall time within 5% of semantic-only.

- [ ] **Step 3: Commit**

```bash
git add crates/ripvec-core/src/embed.rs
git commit -m "perf: build BM25 index concurrent with GPU embedding via rayon::join"
```

---

## Execution Strategy

```
Parallel wave 1 (3 agents):
  [Haiku]  Task 1: Add tantivy dep ────────────────┐
  [Sonnet] Task 2: CodeTokenizer + Bm25Index ──────┤
  [Sonnet] Task 3: HybridIndex + RRF fusion ───────┤
                                                    │
Sequential wave 2: ────────────────────────────────▼
  [Opus]   Task 4: Wire CLI + embed + MCP ─────────→│
  [Sonnet] Task 5: Concurrent build optimization ──→│
                                                     ▼
                                                   Done
```

Tasks 1, 2, 3 run as **parallel subagents** (different files, no overlap).
Task 4 runs after all merge (Opus, touches many files).
Task 5 runs after Task 4 (optimization).
