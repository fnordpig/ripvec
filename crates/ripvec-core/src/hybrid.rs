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
        embeddings: &[Vec<f32>],
        cascade_dim: Option<usize>,
    ) -> crate::Result<Self> {
        let bm25 = Bm25Index::build(&chunks)?;
        let semantic = SearchIndex::new(chunks, embeddings, cascade_dim);
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
    /// Scores are min-max normalized to `[0, 1]` regardless of mode, so
    /// a threshold of 0.5 always means "above midpoint of the score range"
    /// whether the underlying scores are cosine similarity, BM25, or RRF.
    #[must_use]
    pub fn search(
        &self,
        query_embedding: &[f32],
        query_text: &str,
        top_k: usize,
        threshold: f32,
        mode: SearchMode,
    ) -> Vec<(usize, f32)> {
        let mut raw = match mode {
            SearchMode::Semantic => {
                // Fetch more than top_k so normalization has a meaningful range.
                self.semantic
                    .rank_turboquant(query_embedding, top_k.max(100), 0.0)
            }
            SearchMode::Keyword => self.bm25.search(query_text, top_k.max(100)),
            SearchMode::Hybrid => {
                let sem = self
                    .semantic
                    .rank_turboquant(query_embedding, top_k.max(100), 0.0);
                let kw = self.bm25.search(query_text, top_k.max(100));
                rrf_fuse(&sem, &kw, 60.0)
            }
        };

        // Min-max normalize scores to [0, 1] so threshold is model-agnostic.
        if let (Some(max), Some(min)) = (raw.first().map(|(_, s)| *s), raw.last().map(|(_, s)| *s))
        {
            let range = max - min;
            if range > f32::EPSILON {
                for (_, score) in &mut raw {
                    *score = (*score - min) / range;
                }
            } else {
                // All scores identical — normalize to 1.0
                for (_, score) in &mut raw {
                    *score = 1.0;
                }
            }
        }

        // Apply threshold on normalized scores, then truncate
        raw.retain(|(_, score)| *score >= threshold);
        raw.truncate(top_k);
        raw
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

/// Logarithmic saturation steepness for PageRank boost.
///
/// Controls how quickly the boost curve flattens. With `PAGERANK_BETA=10`:
/// - rank 0.01 → 4% of max boost (barely boosted)
/// - rank 0.10 → 29% of max boost
/// - rank 0.50 → 75% of max boost
/// - rank 1.00 → 100% of max boost
///
/// This prevents the single highest-ranked definition from getting a
/// disproportionate boost relative to the second-highest.
const PAGERANK_BETA: f32 = 10.0;

/// Apply a multiplicative PageRank boost to search results.
///
/// For each result, looks up the chunk's PageRank score and applies a
/// log-saturated boost:
///
///   `boosted = score * (1 + alpha * log(1 + beta * rank) / log(1 + beta))`
///
/// The logarithmic saturation compresses high PageRank values so the
/// top-ranked definition doesn't dominate. This models a Bayesian prior
/// where structural importance multiplies with query relevance.
///
/// Results are re-sorted after boosting.
///
/// `pagerank_by_file` maps relative file paths to their PageRank scores
/// (pre-normalized to [0, 1] by dividing by max rank).
/// `alpha` controls overall boost strength. The `alpha` field from
/// [`RepoGraph`] is recommended (auto-tuned from graph density).
pub fn boost_with_pagerank<S: std::hash::BuildHasher>(
    results: &mut [(usize, f32)],
    chunks: &[CodeChunk],
    pagerank_by_file: &HashMap<String, f32, S>,
    alpha: f32,
) {
    let log_denom = (1.0 + PAGERANK_BETA).ln();
    if log_denom <= f32::EPSILON {
        return;
    }

    for (idx, score) in results.iter_mut() {
        if let Some(chunk) = chunks.get(*idx) {
            // Try definition-level lookup first, fall back to file-level
            let def_key = format!("{}::{}", chunk.file_path, chunk.name);
            let rank = pagerank_by_file
                .get(&def_key)
                .or_else(|| pagerank_by_file.get(&chunk.file_path))
                .copied()
                .unwrap_or(0.0);
            let saturated = (1.0 + PAGERANK_BETA * rank).ln() / log_denom;
            *score *= 1.0 + alpha * saturated;
        }
    }
    // Re-sort descending by boosted score
    results.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
}

/// Build a normalized PageRank lookup table from a [`RepoGraph`].
///
/// Returns a map from `"file_path::def_name"` to definition-level PageRank
/// normalized to `[0, 1]`. Also inserts file-level entries (`"file_path"`)
/// as aggregated fallback for chunks that don't match a specific definition.
#[must_use]
pub fn pagerank_lookup(graph: &crate::repo_map::RepoGraph) -> HashMap<String, f32> {
    let max_rank = graph.def_ranks.iter().copied().fold(0.0_f32, f32::max);
    if max_rank <= f32::EPSILON {
        // Fall back to file-level ranks if no def-level data
        let file_max = graph.base_ranks.iter().copied().fold(0.0_f32, f32::max);
        if file_max <= f32::EPSILON {
            return HashMap::new();
        }
        return graph
            .files
            .iter()
            .zip(graph.base_ranks.iter())
            .map(|(file, &rank)| (file.path.clone(), rank / file_max))
            .collect();
    }

    let mut map = HashMap::new();

    // Definition-level entries: "path::name" -> def_rank
    for (file_idx, file) in graph.files.iter().enumerate() {
        for (def_idx, def) in file.defs.iter().enumerate() {
            let flat = graph.def_offsets[file_idx] + def_idx;
            if let Some(&rank) = graph.def_ranks.get(flat) {
                let key = format!("{}::{}", file.path, def.name);
                map.insert(key, rank / max_rank);
            }
        }
        // File-level aggregate for fallback
        if file_idx < graph.base_ranks.len() {
            map.insert(file.path.clone(), graph.base_ranks[file_idx] / max_rank);
        }
    }

    map
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

    #[test]
    fn pagerank_boost_amplifies_relevant() {
        let chunks = vec![
            CodeChunk {
                file_path: "important.rs".into(),
                name: "a".into(),
                kind: "function".into(),
                start_line: 1,
                end_line: 10,
                content: String::new(),
                enriched_content: String::new(),
            },
            CodeChunk {
                file_path: "obscure.rs".into(),
                name: "b".into(),
                kind: "function".into(),
                start_line: 1,
                end_line: 10,
                content: String::new(),
                enriched_content: String::new(),
            },
        ];

        // Both start with same score; important.rs has high PageRank
        let mut results = vec![(0, 0.8_f32), (1, 0.8)];
        let mut pr = HashMap::new();
        pr.insert("important.rs".to_string(), 1.0); // max PageRank
        pr.insert("obscure.rs".to_string(), 0.1); // low PageRank

        boost_with_pagerank(&mut results, &chunks, &pr, 0.3);

        // important.rs should now rank higher
        assert_eq!(
            results[0].0, 0,
            "important.rs should rank first after boost"
        );
        assert!(results[0].1 > results[1].1);

        // Verify the math with log saturation (beta=10):
        // rank=1.0: saturated = ln(11)/ln(11) = 1.0 → 0.8 * (1 + 0.3 * 1.0) = 1.04
        assert!(
            (results[0].1 - 1.04).abs() < 0.01,
            "rank=1.0 boost: expected ~1.04, got {}",
            results[0].1
        );
        // rank=0.1: saturated = ln(2)/ln(11) ≈ 0.289 → 0.8 * (1 + 0.3 * 0.289) ≈ 0.869
        assert!(
            (results[1].1 - 0.869).abs() < 0.01,
            "rank=0.1 boost: expected ~0.869, got {}",
            results[1].1
        );
    }

    #[test]
    fn pagerank_boost_zero_relevance_stays_zero() {
        let chunks = vec![CodeChunk {
            file_path: "important.rs".into(),
            name: "a".into(),
            kind: "function".into(),
            start_line: 1,
            end_line: 10,
            content: String::new(),
            enriched_content: String::new(),
        }];

        let mut results = vec![(0, 0.0_f32)];
        let mut pr = HashMap::new();
        pr.insert("important.rs".to_string(), 1.0);

        boost_with_pagerank(&mut results, &chunks, &pr, 0.3);

        // Zero score stays zero regardless of PageRank
        assert!(results[0].1.abs() < f32::EPSILON);
    }

    #[test]
    fn pagerank_boost_unknown_file_no_effect() {
        let chunks = vec![CodeChunk {
            file_path: "unknown.rs".into(),
            name: "a".into(),
            kind: "function".into(),
            start_line: 1,
            end_line: 10,
            content: String::new(),
            enriched_content: String::new(),
        }];

        let mut results = vec![(0, 0.5_f32)];
        let pr = HashMap::new(); // empty — no PageRank data

        boost_with_pagerank(&mut results, &chunks, &pr, 0.3);

        // No PageRank data → no boost
        assert!((results[0].1 - 0.5).abs() < f32::EPSILON);
    }
}
