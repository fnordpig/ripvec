//! Cosine similarity computation and ranking.
//!
//! Since all embeddings are L2-normalized, cosine similarity equals
//! the dot product — no square roots needed at query time.

/// Cosine similarity between two L2-normalized vectors (= dot product).
#[must_use]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
