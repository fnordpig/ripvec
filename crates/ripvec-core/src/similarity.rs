//! Cosine similarity computation and ranking.
//!
//! Since all embeddings are L2-normalized, cosine similarity equals
//! the dot product — no square roots needed at query time.

/// Cosine similarity between two L2-normalized vectors (= dot product).
#[must_use]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "dot_product: vector length mismatch");
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Rank all chunks against a query embedding using matrix-vector multiply.
///
/// Computes `embeddings @ query` where `embeddings` is `[num_chunks, hidden_dim]`
/// and `query` is `[hidden_dim]`. Returns similarity scores in chunk order.
///
/// Uses ndarray's optimized matmul (SIMD-accelerated via `matrixmultiply` crate).
#[must_use]
pub fn rank_all(embeddings: &ndarray::Array2<f32>, query: &ndarray::Array1<f32>) -> Vec<f32> {
    embeddings.dot(query).to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    #[test]
    fn identical_normalized_vectors() {
        let v = vec![0.5773, 0.5773, 0.5773];
        let sim = dot_product(&v, &v);
        assert!((sim - 1.0).abs() < 0.01);
    }

    #[test]
    fn orthogonal_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = dot_product(&a, &b);
        assert!((sim).abs() < 1e-6);
    }

    #[test]
    fn opposite_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = dot_product(&a, &b);
        assert!((sim + 1.0).abs() < 1e-6);
    }

    #[test]
    fn rank_all_matches_scalar_dot_product() {
        // 4 chunks, 3-dimensional embeddings
        let data = vec![
            1.0, 0.0, 0.0, // chunk 0
            0.0, 1.0, 0.0, // chunk 1
            0.5773, 0.5773, 0.5773, // chunk 2
            -1.0, 0.0, 0.0, // chunk 3
        ];
        let embeddings = Array2::from_shape_vec((4, 3), data.clone()).unwrap();
        let query = Array1::from_vec(vec![1.0, 0.0, 0.0]);

        let scores = rank_all(&embeddings, &query);

        // Compare against scalar dot_product over each row
        for (i, score) in scores.iter().enumerate() {
            let row = &data[i * 3..(i + 1) * 3];
            let expected = dot_product(row, query.as_slice().unwrap());
            assert!(
                (score - expected).abs() < 1e-6,
                "mismatch at chunk {i}: rank_all={score}, dot_product={expected}"
            );
        }
    }

    #[test]
    fn rank_all_empty_matrix() {
        let embeddings = Array2::from_shape_vec((0, 384), vec![]).unwrap();
        let query = Array1::zeros(384);
        let scores = rank_all(&embeddings, &query);
        assert!(scores.is_empty());
    }

    #[test]
    fn rank_all_known_values() {
        // 2x2 matrix: [[1, 2], [3, 4]] dot [1, 0] = [1, 3]
        let embeddings = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let query = Array1::from_vec(vec![1.0, 0.0]);
        let scores = rank_all(&embeddings, &query);
        assert!((scores[0] - 1.0).abs() < 1e-6);
        assert!((scores[1] - 3.0).abs() < 1e-6);

        // Same matrix dot [0, 1] = [2, 4]
        let query2 = Array1::from_vec(vec![0.0, 1.0]);
        let scores2 = rank_all(&embeddings, &query2);
        assert!((scores2[0] - 2.0).abs() < 1e-6);
        assert!((scores2[1] - 4.0).abs() < 1e-6);
    }
}
