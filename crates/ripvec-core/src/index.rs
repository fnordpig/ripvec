//! In-memory search index for real-time re-ranking.
//!
//! Stores all chunk embeddings as a contiguous ndarray matrix so that
//! re-ranking is a single BLAS matrix-vector multiply via [`crate::similarity::rank_all`].
//!
//! Optionally uses [`TurboQuant`](turbo_quant) compression for fast approximate
//! scanning at monorepo scale (100K+ chunks). `TurboQuant` compresses 768-dim
//! embeddings from 3072 bytes (FP32) to ~386 bytes (4-bit), giving ~5× faster
//! scan via sequential memory access + centroid table lookup.

use ndarray::{Array1, Array2};

use crate::chunk::CodeChunk;
use crate::turbo_quant::{CompressedCorpus, PolarCodec};

/// Pre-computed embedding matrix for fast re-ranking.
///
/// Stores all chunk embeddings as a contiguous `[num_chunks, hidden_dim]`
/// ndarray matrix. Re-ranking is a single BLAS matrix-vector multiply.
///
/// When constructed with a `cascade_dim`, also stores a truncated and
/// re-normalized `[num_chunks, cascade_dim]` matrix for two-phase MRL
/// cascade search: fast pre-filter at reduced dimension, then full-dim
/// re-rank of the top candidates.
pub struct SearchIndex {
    /// All chunks with metadata.
    pub chunks: Vec<CodeChunk>,
    /// Embedding matrix `[num_chunks, hidden_dim]`.
    embeddings: Array2<f32>,
    /// Truncated + re-normalized embedding matrix for MRL cascade pre-filter.
    /// `None` when cascade search is disabled.
    truncated: Option<Array2<f32>>,
    /// `TurboQuant`-compressed embeddings for fast approximate scanning.
    /// At 4-bit: 386 bytes/vector vs 3072 bytes FP32 (8× compression).
    /// Scan is ~5× faster than FP32 BLAS at 100K+ chunks.
    compressed: Option<CompressedIndex>,
    /// Hidden dimension size.
    pub hidden_dim: usize,
    /// Truncated dimension size, if cascade search is enabled.
    truncated_dim: Option<usize>,
}

/// `PolarQuant`-compressed embedding index for fast approximate scanning.
///
/// Uses SoA flat layout ([`CompressedCorpus`]) for cache-friendly streaming scans.
struct CompressedIndex {
    /// The codec (holds rotation matrix + centroid tables).
    codec: PolarCodec,
    /// Flat SoA corpus: radii + indices packed contiguously.
    corpus: CompressedCorpus,
}

impl SearchIndex {
    /// Build an index from `embed_all` output.
    ///
    /// Flattens the per-chunk embedding vectors into a contiguous `Array2`
    /// for BLAS-accelerated matrix-vector products at query time.
    ///
    /// When `cascade_dim` is `Some(d)`, also builds a truncated and
    /// L2-re-normalized `[N, d]` matrix for two-phase MRL cascade search.
    /// The truncated dimension is clamped to `hidden_dim`.
    ///
    /// # Panics
    ///
    /// Panics if the flattened embedding data cannot form a valid
    /// `[num_chunks, hidden_dim]` matrix (should never happen when
    /// embeddings come from `embed_all`).
    pub fn new(
        chunks: Vec<CodeChunk>,
        raw_embeddings: &[Vec<f32>],
        cascade_dim: Option<usize>,
    ) -> Self {
        let hidden_dim = raw_embeddings.first().map_or(384, Vec::len);
        let n = chunks.len();

        // Flatten into contiguous array for BLAS
        let mut flat = Vec::with_capacity(n * hidden_dim);
        for emb in raw_embeddings {
            if emb.len() == hidden_dim {
                flat.extend_from_slice(emb);
            } else {
                // Pad/truncate to hidden_dim (shouldn't happen, but be safe)
                flat.extend(emb.iter().take(hidden_dim));
                flat.resize(flat.len() + hidden_dim.saturating_sub(emb.len()), 0.0);
            }
        }

        let embeddings =
            Array2::from_shape_vec((n, hidden_dim), flat).expect("embedding matrix shape mismatch");

        // Build truncated + re-normalized matrix for MRL cascade pre-filter.
        // Nomic MRL models require layer-norm before truncation:
        //   1. Layer-norm over the FULL embedding (mean-center, scale by inv_std)
        //   2. Truncate to first d dimensions
        //   3. L2 renormalize the truncated slice
        let truncated_dim = cascade_dim.map(|d| d.min(hidden_dim));
        let truncated = truncated_dim.map(|d| {
            let mut trunc = Array2::zeros((n, d));
            for (i, row) in embeddings.rows().into_iter().enumerate() {
                let full = row.as_slice().expect("embedding row not contiguous");

                // Step 1: Layer-norm over FULL embedding
                let len = full.len() as f32;
                let mean: f32 = full.iter().sum::<f32>() / len;
                let var: f32 = full.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / len;
                let inv_std = 1.0 / (var + 1e-5).sqrt();

                // Step 2: Truncate first d dims of layer-normed embedding
                // Step 3: L2 renormalize the truncated slice
                let norm: f32 = full[..d]
                    .iter()
                    .map(|x| {
                        let ln = (x - mean) * inv_std;
                        ln * ln
                    })
                    .sum::<f32>()
                    .sqrt()
                    .max(1e-12);
                for (j, &v) in full[..d].iter().enumerate() {
                    trunc[[i, j]] = (v - mean) * inv_std / norm;
                }
            }
            trunc
        });

        // Compress embeddings with PolarQuant (4-bit).
        // At 768-dim: ~1920 bytes/vector vs 3072 FP32. 8× compression with bit-packing.
        let compressed = if hidden_dim >= 64 && hidden_dim.is_multiple_of(2) {
            let codec = PolarCodec::new(hidden_dim, 4, 42);
            let corpus = codec.encode_batch(&embeddings);
            Some(CompressedIndex { codec, corpus })
        } else {
            None
        };

        Self {
            chunks,
            embeddings,
            truncated,
            compressed,
            hidden_dim,
            truncated_dim,
        }
    }

    /// Rank all chunks against a query embedding.
    ///
    /// Returns `(chunk_index, similarity_score)` pairs sorted by descending
    /// score, filtered by `threshold`.
    #[must_use]
    pub fn rank(&self, query_embedding: &[f32], threshold: f32) -> Vec<(usize, f32)> {
        if query_embedding.len() != self.hidden_dim || self.chunks.is_empty() {
            return vec![];
        }
        let query = Array1::from_vec(query_embedding.to_vec());
        let scores = crate::similarity::rank_all(&self.embeddings, &query);

        let mut results: Vec<(usize, f32)> = scores
            .into_iter()
            .enumerate()
            .filter(|(_, score)| *score >= threshold)
            .collect();
        results.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        results
    }

    /// `TurboQuant`-accelerated ranking: compressed approximate scan → exact re-rank.
    ///
    /// 1. Estimate inner products for ALL vectors via `TurboQuant` (~5× faster than BLAS).
    /// 2. Take top `pre_filter_k` approximate candidates.
    /// 3. Re-rank with exact FP32 dot products on the full embedding matrix.
    ///
    /// Falls back to [`rank`] when no compressed index is available.
    #[must_use]
    pub fn rank_turboquant(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        threshold: f32,
    ) -> Vec<(usize, f32)> {
        let Some(ref comp) = self.compressed else {
            return self.rank(query_embedding, threshold);
        };

        if comp.corpus.n != self.chunks.len() {
            return self.rank(query_embedding, threshold);
        }

        // Phase 1: SoA corpus scan — sequential streaming, centroid table in L1.
        let pre_filter_k = (top_k * 10).min(comp.corpus.n);
        let query_state = comp.codec.prepare_query(query_embedding);
        let scores = comp.codec.scan_corpus(&comp.corpus, &query_state);
        let mut approx_scores: Vec<(usize, f32)> = scores.into_iter().enumerate().collect();
        approx_scores.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        approx_scores.truncate(pre_filter_k);

        // Phase 2: exact re-rank top candidates
        let query = Array1::from_vec(query_embedding.to_vec());
        let mut results: Vec<(usize, f32)> = approx_scores
            .iter()
            .map(|&(idx, _)| {
                let exact = self.embeddings.row(idx).dot(&query);
                (idx, exact)
            })
            .filter(|(_, score)| *score >= threshold)
            .collect();
        results.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        results.truncate(top_k);
        results
    }

    /// Two-phase MRL cascade ranking: fast pre-filter then full re-rank.
    ///
    /// 1. Layer-norms the query over its full dimension, truncates to
    ///    `truncated_dim`, L2-normalizes, and computes dot products against
    ///    the truncated matrix to find the top `pre_filter_k` candidates.
    /// 2. Re-ranks those candidates using full-dimension dot products.
    ///
    /// Falls back to [`rank`] when no truncated matrix is available.
    #[must_use]
    pub fn rank_cascade(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        threshold: f32,
    ) -> Vec<(usize, f32)> {
        let Some(ref trunc_matrix) = self.truncated else {
            return self.rank(query_embedding, threshold);
        };
        if query_embedding.len() != self.hidden_dim || self.chunks.is_empty() {
            return vec![];
        }

        let trunc_dim = trunc_matrix.shape()[1];
        let pre_filter_k = 100_usize.max(top_k * 3); // over-retrieve for re-ranking

        // Phase 1: fast pre-filter at truncated dimension
        // Apply layer-norm over full query before truncation (matches corpus processing)
        let len = query_embedding.len() as f32;
        let mean: f32 = query_embedding.iter().sum::<f32>() / len;
        let var: f32 = query_embedding
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>()
            / len;
        let inv_std = 1.0 / (var + 1e-5).sqrt();
        let trunc_query: Vec<f32> = query_embedding[..trunc_dim]
            .iter()
            .map(|x| (x - mean) * inv_std)
            .collect();
        let norm: f32 = trunc_query
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt()
            .max(1e-12);
        let trunc_query_norm: Vec<f32> = trunc_query.iter().map(|x| x / norm).collect();
        let trunc_q = Array1::from_vec(trunc_query_norm);
        let scores = trunc_matrix.dot(&trunc_q);

        // Get top pre_filter_k indices
        let mut candidates: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
        candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        candidates.truncate(pre_filter_k);

        // Phase 2: re-rank candidates with full-dimension dot products
        let query_arr = Array1::from_vec(query_embedding.to_vec());
        let mut reranked: Vec<(usize, f32)> = candidates
            .into_iter()
            .map(|(idx, _)| {
                let full_score = self.embeddings.row(idx).dot(&query_arr);
                (idx, full_score)
            })
            .filter(|(_, s)| *s >= threshold)
            .collect();
        reranked.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        reranked.truncate(top_k);
        reranked
    }

    /// Return a clone of the embedding vector for chunk `idx`.
    ///
    /// Returns `None` if `idx` is out of bounds.
    #[must_use]
    pub fn embedding(&self, idx: usize) -> Option<Vec<f32>> {
        if idx >= self.chunks.len() {
            return None;
        }
        Some(self.embeddings.row(idx).to_vec())
    }

    /// Number of chunks in the index.
    #[must_use]
    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    /// Whether the index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// The truncated dimension used for cascade pre-filtering, if enabled.
    #[must_use]
    pub fn truncated_dim(&self) -> Option<usize> {
        self.truncated_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a dummy `CodeChunk` for testing.
    fn dummy_chunk(name: &str) -> CodeChunk {
        let content = format!("fn {name}() {{}}");
        CodeChunk {
            file_path: "test.rs".to_string(),
            name: name.to_string(),
            kind: "function".to_string(),
            start_line: 1,
            end_line: 10,
            enriched_content: content.clone(),
            content,
        }
    }

    #[test]
    fn new_builds_correct_matrix_shape() {
        let chunks = vec![dummy_chunk("a"), dummy_chunk("b"), dummy_chunk("c")];
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];

        let index = SearchIndex::new(chunks, &embeddings, None);

        assert_eq!(index.len(), 3);
        assert_eq!(index.hidden_dim, 3);
        assert!(!index.is_empty());
    }

    #[test]
    fn rank_returns_sorted_results_above_threshold() {
        let chunks = vec![dummy_chunk("low"), dummy_chunk("high"), dummy_chunk("mid")];
        // Embeddings designed so dot product with [1, 0] gives known scores:
        // chunk 0: 0.2, chunk 1: 0.9, chunk 2: 0.5
        let embeddings = vec![vec![0.2, 0.8], vec![0.9, 0.1], vec![0.5, 0.5]];

        let index = SearchIndex::new(chunks, &embeddings, None);
        let results = index.rank(&[1.0, 0.0], 0.3);

        // Should exclude chunk 0 (score 0.2 < threshold 0.3)
        assert_eq!(results.len(), 2);
        // Should be sorted descending: chunk 1 (0.9), then chunk 2 (0.5)
        assert_eq!(results[0].0, 1);
        assert_eq!(results[1].0, 2);
        assert!(results[0].1 > results[1].1);
    }

    #[test]
    fn rank_with_wrong_dimension_returns_empty() {
        let chunks = vec![dummy_chunk("a")];
        let embeddings = vec![vec![1.0, 0.0, 0.0]];

        let index = SearchIndex::new(chunks, &embeddings, None);
        // Query has wrong dimension (2 instead of 3)
        let results = index.rank(&[1.0, 0.0], 0.0);

        assert!(results.is_empty());
    }

    #[test]
    fn rank_with_empty_query_returns_empty() {
        let chunks = vec![dummy_chunk("a")];
        let embeddings = vec![vec![1.0, 0.0, 0.0]];

        let index = SearchIndex::new(chunks, &embeddings, None);
        let results = index.rank(&[], 0.0);

        assert!(results.is_empty());
    }

    #[test]
    fn rank_handles_empty_index() {
        let index = SearchIndex::new(vec![], &[], None);

        // hidden_dim defaults to 384 for empty input
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);

        let results = index.rank(&[1.0; 384], 0.0);
        assert!(results.is_empty());
    }

    /// L2-normalize a vector in-place.
    fn l2_normalize(v: &mut [f32]) {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        for x in v.iter_mut() {
            *x /= norm;
        }
    }

    #[test]
    #[expect(
        clippy::cast_precision_loss,
        reason = "test values are small counts and indices"
    )]
    fn cascade_recall_at_10_vs_full_rank() {
        // Build 200 chunks with 8-dim random-ish embeddings (L2-normalized).
        // Use a deterministic pattern so the test is reproducible.
        let n = 200;
        let dim = 8;
        let cascade_dim = 4;

        let mut chunks = Vec::with_capacity(n);
        let mut embeddings = Vec::with_capacity(n);
        for i in 0..n {
            chunks.push(dummy_chunk(&format!("chunk_{i}")));
            // Deterministic pseudo-random: use sin/cos of index
            let mut emb: Vec<f32> = (0..dim).map(|d| ((i * 7 + d * 13) as f32).sin()).collect();
            l2_normalize(&mut emb);
            embeddings.push(emb);
        }

        // Query: L2-normalized
        let mut query: Vec<f32> = (0..dim).map(|d| ((42 * 7 + d * 13) as f32).sin()).collect();
        l2_normalize(&mut query);

        // Build index without cascade (reference)
        let index_full = SearchIndex::new(chunks.clone(), &embeddings, None);
        let full_results = index_full.rank(&query, 0.0);
        let full_top10: Vec<usize> = full_results.iter().take(10).map(|(idx, _)| *idx).collect();

        // Build index with cascade
        let index_cascade = SearchIndex::new(chunks, &embeddings, Some(cascade_dim));
        assert_eq!(index_cascade.truncated_dim(), Some(cascade_dim));
        let cascade_results = index_cascade.rank_cascade(&query, 10, 0.0);
        let cascade_top10: Vec<usize> = cascade_results.iter().map(|(idx, _)| *idx).collect();

        // Recall@10: how many of full-dim top-10 appear in cascade top-10
        let overlap = full_top10
            .iter()
            .filter(|i| cascade_top10.contains(i))
            .count();
        let recall = overlap as f32 / 10.0;

        assert!(
            recall >= 0.7,
            "cascade Recall@10 = {recall} ({overlap}/10), expected >= 0.7"
        );
    }

    #[test]
    fn cascade_falls_back_without_truncated_matrix() {
        let chunks = vec![dummy_chunk("a"), dummy_chunk("b")];
        let embeddings = vec![vec![0.9, 0.1], vec![0.1, 0.9]];

        // No cascade_dim → rank_cascade should behave like rank
        let index = SearchIndex::new(chunks, &embeddings, None);
        let cascade = index.rank_cascade(&[1.0, 0.0], 10, 0.0);
        let plain = index.rank(&[1.0, 0.0], 0.0);

        assert_eq!(cascade.len(), plain.len());
        for (c, p) in cascade.iter().zip(plain.iter()) {
            assert_eq!(c.0, p.0);
            assert!((c.1 - p.1).abs() < 1e-6);
        }
    }

    #[test]
    fn cascade_respects_threshold() {
        let chunks = vec![dummy_chunk("high"), dummy_chunk("low")];
        // Embeddings: chunk 0 aligns with query, chunk 1 is orthogonal
        let embeddings = vec![vec![1.0, 0.0], vec![0.0, 1.0]];

        let index = SearchIndex::new(chunks, &embeddings, Some(1));
        let results = index.rank_cascade(&[1.0, 0.0], 10, 0.5);

        // Only chunk 0 should pass the 0.5 threshold
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn turboquant_recall_vs_exact() {
        // Generate 200 random 768-dim L2-normalized embeddings.
        let dim = 768;
        let n = 200;
        let mut embeddings: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                let mut v: Vec<f32> = (0..dim).map(|d| ((i * 17 + d * 31) as f32).sin()).collect();
                l2_normalize(&mut v);
                v
            })
            .collect();

        let chunks: Vec<CodeChunk> = (0..n).map(|i| dummy_chunk(&format!("chunk_{i}"))).collect();
        let mut query: Vec<f32> = (0..dim).map(|d| ((42 * 7 + d * 13) as f32).sin()).collect();
        l2_normalize(&mut query);

        let index = SearchIndex::new(chunks, &embeddings, None);

        // Exact ranking
        let exact = index.rank(&query, 0.0);
        let exact_top10: Vec<usize> = exact.iter().take(10).map(|(idx, _)| *idx).collect();

        // TurboQuant ranking
        let tq = index.rank_turboquant(&query, 10, 0.0);
        let tq_top10: Vec<usize> = tq.iter().take(10).map(|(idx, _)| *idx).collect();

        // Recall@10: how many of exact top-10 appear in TQ top-10
        let recall = exact_top10.iter().filter(|i| tq_top10.contains(i)).count();
        eprintln!("TurboQuant Recall@10: {recall}/10");
        assert!(
            recall >= 7,
            "TurboQuant recall should be >= 7/10, got {recall}/10"
        );
    }
}
