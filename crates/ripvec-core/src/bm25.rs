//! BM25 keyword search index for code chunks.
//!
//! Provides camelCase/snake_case-aware tokenization via [`CodeSplitFilter`]
//! and an in-RAM tantivy index ([`Bm25Index`]) that supports per-field
//! boosted queries so identifier sub-tokens (e.g. `json` from
//! `parseJsonConfig`) are matched correctly.

use tantivy::schema::{
    Field, INDEXED, IndexRecordOption, STORED, Schema, TextFieldIndexing, TextOptions, Value,
};
use tantivy::tokenizer::{
    LowerCaser, SimpleTokenizer, TextAnalyzer, Token, TokenFilter, TokenStream, Tokenizer,
};
use tantivy::{
    Index, IndexReader, ReloadPolicy, TantivyDocument,
    collector::TopDocs,
    query::{BooleanQuery, BoostQuery, Occur, QueryParser},
};

use crate::chunk::CodeChunk;

// ──────────────────────────────────────────────────────────────────────────────
// Identifier splitting
// ──────────────────────────────────────────────────────────────────────────────

/// Split a code identifier into its constituent sub-words.
///
/// Handles camelCase, PascalCase, snake_case, SCREAMING_SNAKE_CASE, and mixed
/// forms (e.g. `HTML5Parser`). Returns the lowercased parts; if there is only
/// one part (i.e. the token cannot be split further) an empty `Vec` is returned
/// so callers know no expansion is needed.
///
/// # Examples
/// ```
/// # use ripvec_core::bm25::split_code_identifier;
/// assert_eq!(split_code_identifier("parseJsonConfig"), vec!["parse", "json", "config"]);
/// assert_eq!(split_code_identifier("my_func_name"),    vec!["my", "func", "name"]);
/// assert_eq!(split_code_identifier("HTML5Parser"),     vec!["html5", "parser"]);
/// assert_eq!(split_code_identifier("parser"),          Vec::<String>::new());
/// ```
#[must_use]
pub fn split_code_identifier(text: &str) -> Vec<String> {
    // First split on underscores (handles snake_case / SCREAMING_SNAKE).
    let underscore_parts: Vec<&str> = text.split('_').filter(|s| !s.is_empty()).collect();

    let mut parts: Vec<String> = Vec::new();

    for segment in &underscore_parts {
        // Within each segment apply camelCase splitting.
        // State machine: accumulate a "run" of chars, flush when the boundary
        // rule triggers.
        let chars: Vec<char> = segment.chars().collect();
        let n = chars.len();
        let mut start = 0usize;

        let mut i = 0usize;
        while i < n {
            // Detect camelCase / acronym boundaries.
            // Treat digits as "non-upper" (like lowercase) for boundary
            // detection so that "HTML5Parser" splits into ["html5", "parser"].
            if i > start {
                let prev = chars[i - 1];
                let cur = chars[i];

                // lowercase/digit → uppercase: "parseJson"/"HTML5Parser" → split before cur.
                let lower_to_upper =
                    (prev.is_lowercase() || prev.is_ascii_digit()) && cur.is_uppercase();

                // uppercase-run → lowercase: "HTMLParser" → the 'P' starts a new word.
                // The split point is before the last uppercase in the run.
                // Digits are NOT treated as terminators here so "HTML5" stays intact.
                let upper_run_to_lower = i >= 2
                    && prev.is_uppercase()
                    && cur.is_lowercase()
                    && chars[i - 2].is_uppercase();

                if lower_to_upper {
                    parts.push(chars[start..i].iter().collect::<String>().to_lowercase());
                    start = i;
                } else if upper_run_to_lower {
                    // Flush everything up to (but not including) prev.
                    parts.push(
                        chars[start..i - 1]
                            .iter()
                            .collect::<String>()
                            .to_lowercase(),
                    );
                    start = i - 1;
                }
            }
            i += 1;
        }
        // Flush remaining
        if start < n {
            parts.push(chars[start..n].iter().collect::<String>().to_lowercase());
        }
    }

    // If we ended up with a single part that equals the lowercased original,
    // there was nothing to split — return empty to signal "no expansion".
    if parts.len() <= 1 {
        return Vec::new();
    }

    parts
}

// ──────────────────────────────────────────────────────────────────────────────
// Tantivy token filter
// ──────────────────────────────────────────────────────────────────────────────

/// Token stream produced by [`CodeSplitFilterWrapper`].
///
/// For each token from the upstream stream the original token is emitted first,
/// then any sub-tokens produced by [`split_code_identifier`].
pub struct CodeSplitTokenStream<'a, T> {
    /// Upstream token stream (already lowercased by `LowerCaser`).
    tail: T,
    /// Buffer of pending sub-tokens; filled in reverse so `pop()` yields them
    /// in order.
    pending: &'a mut Vec<Token>,
}

impl<T: TokenStream> TokenStream for CodeSplitTokenStream<'_, T> {
    fn advance(&mut self) -> bool {
        // Drain any buffered sub-tokens first.
        if let Some(tok) = self.pending.pop() {
            *self.tail.token_mut() = tok;
            return true;
        }

        // Advance the upstream stream.
        if !self.tail.advance() {
            return false;
        }

        let upstream = self.tail.token().clone();
        let sub_tokens = split_code_identifier(&upstream.text);

        // Queue sub-tokens in reverse order so pop() gives them in order.
        let position_offset = upstream.position;
        for (idx, sub) in sub_tokens.iter().enumerate().rev() {
            let mut t = upstream.clone();
            t.text.clone_from(sub);
            t.position = position_offset + idx + 1;
            self.pending.push(t);
        }

        // The upstream token is already current — nothing extra needed.
        true
    }

    fn token(&self) -> &Token {
        self.tail.token()
    }

    fn token_mut(&mut self) -> &mut Token {
        self.tail.token_mut()
    }
}

/// Tantivy [`TokenFilter`] that emits sub-tokens for camelCase/snake_case
/// identifiers in addition to the original token.
#[derive(Clone)]
pub struct CodeSplitFilter;

impl TokenFilter for CodeSplitFilter {
    type Tokenizer<T: Tokenizer> = CodeSplitFilterWrapper<T>;

    fn transform<T: Tokenizer>(self, tokenizer: T) -> CodeSplitFilterWrapper<T> {
        CodeSplitFilterWrapper {
            inner: tokenizer,
            pending: Vec::new(),
        }
    }
}

/// Wrapper tokenizer produced by [`CodeSplitFilter::transform`].
#[derive(Clone)]
pub struct CodeSplitFilterWrapper<T> {
    inner: T,
    pending: Vec<Token>,
}

impl<T: Tokenizer> Tokenizer for CodeSplitFilterWrapper<T> {
    type TokenStream<'a> = CodeSplitTokenStream<'a, T::TokenStream<'a>>;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> Self::TokenStream<'a> {
        self.pending.clear();
        CodeSplitTokenStream {
            tail: self.inner.token_stream(text),
            pending: &mut self.pending,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Analyzer
// ──────────────────────────────────────────────────────────────────────────────

/// Build a tantivy [`TextAnalyzer`] that tokenizes, expands camelCase/snake_case
/// identifiers into sub-tokens, then lowercases everything.
///
/// `CodeSplitFilter` must run **before** `LowerCaser` so that camelCase boundaries
/// (uppercase letters) are still visible when splitting.
#[must_use]
pub fn code_analyzer() -> TextAnalyzer {
    TextAnalyzer::builder(SimpleTokenizer::default())
        .filter(CodeSplitFilter)
        .filter(LowerCaser)
        .build()
}

// ──────────────────────────────────────────────────────────────────────────────
// Schema
// ──────────────────────────────────────────────────────────────────────────────

/// Handles to the tantivy schema fields used by [`Bm25Index`].
pub struct BM25Fields {
    /// Chunk name (function/struct name, high-value signal).
    pub name: Field,
    /// Relative file path of the source file.
    pub file_path: Field,
    /// Full chunk content (body text).
    pub body: Field,
    /// Monotonic index into the original `chunks` slice — stored for retrieval.
    pub chunk_id: Field,
}

/// Construct the tantivy [`Schema`] and return field handles.
#[must_use]
pub fn build_schema() -> (Schema, BM25Fields) {
    let mut builder = Schema::builder();

    let code_indexing = TextFieldIndexing::default()
        .set_tokenizer("code")
        .set_index_option(IndexRecordOption::WithFreqsAndPositions);

    let text_opts = TextOptions::default()
        .set_indexing_options(code_indexing)
        .set_stored();

    let name = builder.add_text_field("name", text_opts.clone());
    let file_path = builder.add_text_field("file_path", text_opts.clone());
    let body = builder.add_text_field("body", text_opts);
    let chunk_id = builder.add_u64_field("chunk_id", INDEXED | STORED);

    let schema = builder.build();
    (
        schema,
        BM25Fields {
            name,
            file_path,
            body,
            chunk_id,
        },
    )
}

// ──────────────────────────────────────────────────────────────────────────────
// Bm25Index
// ──────────────────────────────────────────────────────────────────────────────

/// In-RAM BM25 index over a slice of [`CodeChunk`]s.
///
/// Built with [`Bm25Index::build`]; query with [`Bm25Index::search`].
pub struct Bm25Index {
    index: Index,
    reader: IndexReader,
    fields: BM25Fields,
}

impl Bm25Index {
    /// Build a fresh in-RAM index from the given chunks.
    ///
    /// Registers the `"code"` tokenizer, indexes each chunk's `name`,
    /// `file_path`, and `content`, then commits.
    pub fn build(chunks: &[CodeChunk]) -> crate::Result<Self> {
        let (schema, fields) = build_schema();

        let index = Index::create_in_ram(schema.clone());

        // Register our custom tokenizer under the name "code".
        index.tokenizers().register("code", code_analyzer());

        let mut writer = index
            .writer(50_000_000)
            .map_err(|e| crate::Error::Other(e.into()))?;

        for (idx, chunk) in chunks.iter().enumerate() {
            let mut doc = TantivyDocument::default();
            doc.add_text(fields.name, &chunk.name);
            doc.add_text(fields.file_path, &chunk.file_path);
            doc.add_text(fields.body, &chunk.content);
            doc.add_u64(fields.chunk_id, idx as u64);
            writer
                .add_document(doc)
                .map_err(|e| crate::Error::Other(e.into()))?;
        }

        writer.commit().map_err(|e| crate::Error::Other(e.into()))?;

        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()
            .map_err(|e| crate::Error::Other(e.into()))?;

        Ok(Self {
            index,
            reader,
            fields,
        })
    }

    /// Search the index for `query_text`, returning up to `top_k` results.
    ///
    /// Fields are boosted: `name` ×3.0, `file_path` ×1.5, `body` ×1.0.
    ///
    /// Returns a `Vec<(chunk_idx, bm25_score)>` sorted by descending score.
    #[must_use]
    pub fn search(&self, query_text: &str, top_k: usize) -> Vec<(usize, f32)> {
        let searcher = self.reader.searcher();

        // Build per-field boosted sub-queries and combine with BooleanQuery.
        let make_sub = |field: Field, boost: f32| -> Box<dyn tantivy::query::Query> {
            let mut parser = QueryParser::for_index(&self.index, vec![field]);
            parser.set_field_boost(field, boost);
            let q = parser.parse_query(query_text).unwrap_or_else(|_| {
                // Fallback: empty query that matches nothing.
                Box::new(tantivy::query::AllQuery)
            });
            Box::new(BoostQuery::new(q, boost))
        };

        let sub_queries: Vec<(Occur, Box<dyn tantivy::query::Query>)> = vec![
            (Occur::Should, make_sub(self.fields.name, 3.0)),
            (Occur::Should, make_sub(self.fields.file_path, 1.5)),
            (Occur::Should, make_sub(self.fields.body, 1.0)),
        ];

        let combined = BooleanQuery::new(sub_queries);

        let Ok(top_docs) = searcher.search(&combined, &TopDocs::with_limit(top_k).order_by_score())
        else {
            return vec![];
        };

        let mut results = Vec::with_capacity(top_docs.len());
        for (score, doc_addr) in top_docs {
            let Ok(doc) = searcher.doc::<TantivyDocument>(doc_addr) else {
                continue;
            };
            let Some(id_val) = doc.get_first(self.fields.chunk_id) else {
                continue;
            };
            let Some(id) = id_val.as_u64() else {
                continue;
            };
            results.push((usize::try_from(id).unwrap_or(usize::MAX), score));
        }

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_chunk(name: &str, file_path: &str, content: &str) -> CodeChunk {
        CodeChunk {
            file_path: file_path.to_string(),
            name: name.to_string(),
            kind: "function_item".to_string(),
            start_line: 1,
            end_line: 10,
            content: content.to_string(),
            enriched_content: content.to_string(),
        }
    }

    #[test]
    fn split_camel_case() {
        let parts = split_code_identifier("parseJsonConfig");
        assert_eq!(parts, vec!["parse", "json", "config"]);
    }

    #[test]
    fn split_snake_case() {
        let parts = split_code_identifier("my_func_name");
        assert_eq!(parts, vec!["my", "func", "name"]);
    }

    #[test]
    fn split_screaming_snake() {
        let parts = split_code_identifier("MAX_BATCH_SIZE");
        assert_eq!(parts, vec!["max", "batch", "size"]);
    }

    #[test]
    fn split_mixed() {
        let parts = split_code_identifier("MetalDriver");
        assert_eq!(parts, vec!["metal", "driver"]);
    }

    #[test]
    fn no_split_single_word() {
        let parts = split_code_identifier("parser");
        assert!(parts.is_empty(), "expected empty vec, got {parts:?}");
    }

    #[test]
    fn bm25_index_search() {
        let chunks = vec![
            make_chunk(
                "parseJsonConfig",
                "src/config.rs",
                "fn parseJsonConfig(data: &str) -> Config { ... }",
            ),
            make_chunk(
                "renderHtml",
                "src/render.rs",
                "fn renderHtml(template: &str) -> String { ... }",
            ),
        ];

        let index = Bm25Index::build(&chunks).expect("index build failed");
        let results = index.search("parseJsonConfig", 5);

        println!("results: {results:?}");
        assert!(!results.is_empty(), "expected at least one result");
        assert_eq!(results[0].0, 0, "chunk 0 should rank first");
    }

    #[test]
    fn bm25_camel_case_subtoken_match() {
        let chunks = vec![
            make_chunk(
                "parseJsonConfig",
                "src/config.rs",
                "fn parseJsonConfig(data: &str) -> Config { ... }",
            ),
            make_chunk(
                "renderHtml",
                "src/render.rs",
                "fn renderHtml(template: &str) -> String { ... }",
            ),
        ];

        let index = Bm25Index::build(&chunks).expect("index build failed");
        // "json" is a sub-token of "parseJsonConfig" — should match chunk 0.
        let results = index.search("json", 5);

        println!("subtoken results: {results:?}");
        assert!(!results.is_empty(), "expected results for sub-token 'json'");
        assert_eq!(results[0].0, 0, "parseJsonConfig chunk should match 'json'");
    }
}
