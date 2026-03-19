//! In-memory search index for real-time re-ranking.
//!
//! Stores all chunk embeddings as a contiguous ndarray matrix so that
//! re-ranking is a single BLAS matrix-vector multiply via [`ripvec_core::similarity::rank_all`].

use ndarray::{Array1, Array2};
use ripvec_core::chunk::CodeChunk;

/// Pre-computed embedding matrix for fast re-ranking.
///
/// Stores all chunk embeddings as a contiguous `[num_chunks, hidden_dim]`
/// ndarray matrix. Re-ranking is a single BLAS matrix-vector multiply.
pub struct SearchIndex {
    /// All chunks with metadata.
    pub chunks: Vec<CodeChunk>,
    /// Embedding matrix `[num_chunks, hidden_dim]`.
    embeddings: Array2<f32>,
    /// Hidden dimension size.
    pub hidden_dim: usize,
}

impl SearchIndex {
    /// Build an index from `embed_all` output.
    ///
    /// Flattens the per-chunk embedding vectors into a contiguous `Array2`
    /// for BLAS-accelerated matrix-vector products at query time.
    pub fn new(chunks: Vec<CodeChunk>, raw_embeddings: &[Vec<f32>]) -> Self {
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

        Self {
            chunks,
            embeddings,
            hidden_dim,
        }
    }

    /// Rank all chunks against a query embedding.
    ///
    /// Returns `(chunk_index, similarity_score)` pairs sorted by descending
    /// score, filtered by `threshold`.
    pub fn rank(&self, query_embedding: &[f32], threshold: f32) -> Vec<(usize, f32)> {
        if query_embedding.len() != self.hidden_dim || self.chunks.is_empty() {
            return vec![];
        }
        let query = Array1::from_vec(query_embedding.to_vec());
        let scores = ripvec_core::similarity::rank_all(&self.embeddings, &query);

        let mut results: Vec<(usize, f32)> = scores
            .into_iter()
            .enumerate()
            .filter(|(_, score)| *score >= threshold)
            .collect();
        results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Number of chunks in the index.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "public API for future TUI features")
    )]
    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    /// Whether the index is empty.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "public API for future TUI features")
    )]
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a dummy `CodeChunk` for testing.
    fn dummy_chunk(name: &str) -> CodeChunk {
        CodeChunk {
            file_path: "test.rs".to_string(),
            name: name.to_string(),
            kind: "function".to_string(),
            start_line: 1,
            end_line: 10,
            content: format!("fn {name}() {{}}"),
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

        let index = SearchIndex::new(chunks, &embeddings);

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

        let index = SearchIndex::new(chunks, &embeddings);
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

        let index = SearchIndex::new(chunks, &embeddings);
        // Query has wrong dimension (2 instead of 3)
        let results = index.rank(&[1.0, 0.0], 0.0);

        assert!(results.is_empty());
    }

    #[test]
    fn rank_with_empty_query_returns_empty() {
        let chunks = vec![dummy_chunk("a")];
        let embeddings = vec![vec![1.0, 0.0, 0.0]];

        let index = SearchIndex::new(chunks, &embeddings);
        let results = index.rank(&[], 0.0);

        assert!(results.is_empty());
    }

    #[test]
    fn rank_handles_empty_index() {
        let index = SearchIndex::new(vec![], &[]);

        // hidden_dim defaults to 384 for empty input
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);

        let results = index.rank(&[1.0; 384], 0.0);
        assert!(results.is_empty());
    }
}
