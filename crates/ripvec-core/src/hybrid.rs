//! Hybrid semantic + keyword search with Reciprocal Rank Fusion (RRF).
//!
//! [`HybridIndex`] wraps a [`SearchIndex`] (dense vector search) and a
//! [`Bm25Index`] (BM25 keyword search) and fuses their ranked results via
//! Reciprocal Rank Fusion so that chunks appearing high in either list
//! bubble to the top of the combined ranking.

use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;

use crate::bm25::Bm25Index;
use crate::chunk::CodeChunk;
use crate::index::SearchIndex;

/// Controls which retrieval strategy is used during search.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SearchMode {
    /// Fuse semantic (vector) and keyword (BM25) results via RRF.
    #[default]
    Hybrid,
    /// Dense vector cosine-similarity ranking only.
    Semantic,
    /// BM25 keyword ranking only.
    Keyword,
}

impl fmt::Display for SearchMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hybrid => f.write_str("hybrid"),
            Self::Semantic => f.write_str("semantic"),
            Self::Keyword => f.write_str("keyword"),
        }
    }
}

/// Error returned when a `SearchMode` string cannot be parsed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseSearchModeError(String);

impl fmt::Display for ParseSearchModeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "unknown search mode {:?}; expected hybrid, semantic, or keyword",
            self.0
        )
    }
}

impl std::error::Error for ParseSearchModeError {}

impl FromStr for SearchMode {
    type Err = ParseSearchModeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "hybrid" => Ok(Self::Hybrid),
            "semantic" => Ok(Self::Semantic),
            "keyword" => Ok(Self::Keyword),
            other => Err(ParseSearchModeError(other.to_string())),
        }
    }
}

/// Combined semantic + keyword search index with RRF fusion.
///
/// Build once from chunks and pre-computed embeddings; query repeatedly
/// via [`search`](Self::search).
pub struct HybridIndex {
    /// Semantic (dense vector) search index.
    pub semantic: SearchIndex,
    /// BM25 keyword search index.
    bm25: Bm25Index,
}

impl HybridIndex {
    /// Build a `HybridIndex` from raw chunks and their pre-computed embeddings.
    ///
    /// Constructs both the [`SearchIndex`] and [`Bm25Index`] in one call.
    /// `cascade_dim` is forwarded to [`SearchIndex::new`] for optional MRL
    /// cascade pre-filtering.
    ///
    /// # Errors
    ///
    /// Returns an error if the BM25 index cannot be built (e.g., tantivy
    /// schema or writer failure).
    pub fn new(
        chunks: Vec<CodeChunk>,
        embeddings: Vec<Vec<f32>>,
        cascade_dim: Option<usize>,
    ) -> crate::Result<Self> {
        let bm25 = Bm25Index::build(&chunks)?;
        let semantic = SearchIndex::new(chunks, &embeddings, cascade_dim);
        Ok(Self { semantic, bm25 })
    }

    /// Assemble a `HybridIndex` from pre-built components.
    ///
    /// Useful when the caller has already constructed the sub-indices
    /// separately (e.g., loaded from a cache).
    #[must_use]
    pub fn from_parts(semantic: SearchIndex, bm25: Bm25Index) -> Self {
        Self { semantic, bm25 }
    }

    /// Search the index and return `(chunk_index, score)` pairs.
    ///
    /// Dispatches based on `mode`:
    /// - [`SearchMode::Semantic`] — pure dense vector search via
    ///   [`SearchIndex::rank`].
    /// - [`SearchMode::Keyword`] — pure BM25 keyword search, truncated to
    ///   `top_k`.
    /// - [`SearchMode::Hybrid`] — retrieves both ranked lists, fuses them
    ///   with [`rrf_fuse`], then truncates to `top_k`.
    ///
    /// Scores in the returned vec have different scales depending on mode:
    /// semantic scores are cosine similarities in `[0, 1]`; keyword and
    /// hybrid scores are RRF scores (`≤ 1 / (k + 1) * 2` for hybrid).
    #[must_use]
    pub fn search(
        &self,
        query_embedding: &[f32],
        query_text: &str,
        top_k: usize,
        threshold: f32,
        mode: SearchMode,
    ) -> Vec<(usize, f32)> {
        match mode {
            SearchMode::Semantic => {
                let mut results = self
                    .semantic
                    .rank_turboquant(query_embedding, top_k, threshold);
                results.truncate(top_k);
                results
            }
            SearchMode::Keyword => self.bm25.search(query_text, top_k),
            SearchMode::Hybrid => {
                // TurboQuant 4-bit scan for semantic candidates (5× faster than BLAS).
                // threshold=0 because RRF ranks by position, not score magnitude.
                let sem = self
                    .semantic
                    .rank_turboquant(query_embedding, top_k.max(100), 0.0);
                let kw = self.bm25.search(query_text, top_k.max(100));
                let mut fused = rrf_fuse(&sem, &kw, 60.0);
                fused.truncate(top_k);
                fused
            }
        }
    }

    /// All chunks in the index.
    #[must_use]
    pub fn chunks(&self) -> &[CodeChunk] {
        &self.semantic.chunks
    }
}

/// Reciprocal Rank Fusion of two ranked lists.
///
/// Each entry in `semantic` and `bm25` is `(chunk_index, _score)`.
/// The fused score for a chunk is the sum of `1 / (k + rank + 1)` across
/// every list the chunk appears in, where `rank` is 0-based.
///
/// Returns all chunks that appear in either list, sorted descending by
/// fused RRF score.
///
/// `k` should typically be 60.0 — a conventional constant that smooths the
/// ranking boost for the very top results.
#[must_use]
pub fn rrf_fuse(semantic: &[(usize, f32)], bm25: &[(usize, f32)], k: f32) -> Vec<(usize, f32)> {
    let mut scores: HashMap<usize, f32> = HashMap::new();

    for (rank, &(idx, _)) in semantic.iter().enumerate() {
        *scores.entry(idx).or_insert(0.0) += 1.0 / (k + rank as f32 + 1.0);
    }
    for (rank, &(idx, _)) in bm25.iter().enumerate() {
        *scores.entry(idx).or_insert(0.0) += 1.0 / (k + rank as f32 + 1.0);
    }

    let mut results: Vec<(usize, f32)> = scores.into_iter().collect();
    results.sort_unstable_by(|a, b| {
        b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)) // stable tie-break by chunk index
    });
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rrf_union_semantics() {
        // sem: [0, 1, 2], bm25: [3, 0, 4]
        // Chunk 0 appears in both lists → highest RRF score.
        // Chunks 1, 2, 3, 4 appear in exactly one list → all five appear.
        let sem = vec![(0, 0.9), (1, 0.8), (2, 0.7)];
        let bm25 = vec![(3, 10.0), (0, 8.0), (4, 6.0)];

        let fused = rrf_fuse(&sem, &bm25, 60.0);

        let indices: Vec<usize> = fused.iter().map(|&(i, _)| i).collect();

        // All 5 unique chunks must appear
        for expected in [0, 1, 2, 3, 4] {
            assert!(
                indices.contains(&expected),
                "chunk {expected} missing from fused results"
            );
        }
        assert_eq!(fused.len(), 5);

        // Chunk 0 must rank first (double-list bonus)
        assert_eq!(indices[0], 0, "chunk 0 should rank first");
    }

    #[test]
    fn rrf_single_list() {
        // Only semantic results; BM25 is empty.
        let sem = vec![(0, 0.9), (1, 0.8)];
        let bm25: Vec<(usize, f32)> = vec![];

        let fused = rrf_fuse(&sem, &bm25, 60.0);

        assert_eq!(fused.len(), 2);
        // Chunk 0 ranked first in sem list → higher RRF score than chunk 1
        assert_eq!(fused[0].0, 0);
        assert_eq!(fused[1].0, 1);
        assert!(fused[0].1 > fused[1].1);
    }

    #[test]
    fn search_mode_roundtrip() {
        assert_eq!("hybrid".parse::<SearchMode>().unwrap(), SearchMode::Hybrid);
        assert_eq!(
            "semantic".parse::<SearchMode>().unwrap(),
            SearchMode::Semantic
        );
        assert_eq!(
            "keyword".parse::<SearchMode>().unwrap(),
            SearchMode::Keyword
        );

        let err = "invalid".parse::<SearchMode>();
        assert!(err.is_err(), "expected parse error for 'invalid'");
        let msg = err.unwrap_err().to_string();
        assert!(
            msg.contains("invalid"),
            "error message should echo the bad input"
        );
    }

    #[test]
    fn search_mode_display() {
        assert_eq!(SearchMode::Hybrid.to_string(), "hybrid");
        assert_eq!(SearchMode::Semantic.to_string(), "semantic");
        assert_eq!(SearchMode::Keyword.to_string(), "keyword");
    }
}
