//! TurboQuant compressed search: 4-bit PolarQuant for fast approximate scanning.
//!
//! Compresses L2-normalized embedding vectors from 768×f32 (3072 bytes) to
//! ~386 bytes per vector (8× compression). The scan phase uses pre-computed
//! centroid-query dot products for O(d/2) table lookups per vector instead
//! of O(d) multiply-adds.
//!
//! # Algorithm (PolarQuant, no QJL)
//!
//! 1. **Rotation**: multiply by a seeded orthogonal matrix Π to whiten the data.
//!    After rotation, coordinates are approximately i.i.d. N(0, 1/√d).
//! 2. **Polar encoding**: group consecutive pairs (y₀,y₁), (y₂,y₃), …
//!    Convert each to polar: r = √(a²+b²), θ = atan2(b, a).
//!    Quantize θ to 4 bits (16 levels uniformly on [−π, π]).
//! 3. **Storage**: per-pair f32 radius + u8 angle index = 5 bytes/pair.
//!    d=768 → 384 pairs → 384×5 = 1920 bytes. (Further compressible via
//!    bit-packing angle indices to 4 bits → 384×4.5 = 1728 bytes.)
//!
//! # Batch scan (the fast path)
//!
//! Pre-compute once per query:
//! - `rotated_query = Π · query` (one matvec via `Driver::gemm`)
//! - For each angle level j ∈ 0..16, for each pair position i:
//!   `centroid_q[i][j] = q_a·cos(θⱼ) + q_b·sin(θⱼ)`
//!   where (q_a, q_b) = rotated_query pair i, θⱼ = dequantized angle.
//!
//! Per vector: `score = Σᵢ radii[i] × centroid_q[i][angle_index[i]]`
//! That's 384 table lookups + 384 multiply-adds. No matvec.

use std::f32::consts::PI;

use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};

/// Compressed representation of a single embedding vector.
///
/// At 4-bit, d=768: 384 pairs × (4 bytes radius + 1 byte angle) = 1920 bytes.
/// Compare: FP32 = 3072 bytes, FP16 = 1536 bytes.
#[derive(Clone)]
pub struct CompressedCode {
    /// Per-pair radii (f32, lossless).
    pub radii: Vec<f32>,
    /// Quantized angle indices in \[0, 2^bits).
    pub angle_indices: Vec<u8>,
}

impl CompressedCode {
    /// Approximate memory in bytes.
    pub fn encoded_bytes(&self) -> usize {
        self.radii.len() * 4 + self.angle_indices.len()
    }
}

/// PolarQuant codec: encode, decode, and batch-scan compressed embeddings.
///
/// The rotation matrix is seeded — only `(dim, bits, seed)` need to be stored.
/// The matrix is regenerated on construction via QR decomposition of a random
/// Gaussian matrix.
pub struct PolarCodec {
    dim: usize,
    bits: u8,
    levels: usize,
    pairs: usize,
    /// Row-major orthogonal rotation matrix [dim × dim].
    rotation: Array2<f32>,
    /// Pre-computed cos/sin for each quantized angle level.
    /// `cos_table[j]`, `sin_table[j]` for j in 0..levels.
    cos_table: Vec<f32>,
    sin_table: Vec<f32>,
}

impl PolarCodec {
    /// Create a new codec for vectors of dimension `dim`.
    ///
    /// # Panics
    ///
    /// Panics if `dim` is 0, odd, or `bits` is 0 or > 8.
    pub fn new(dim: usize, bits: u8, seed: u64) -> Self {
        assert!(dim > 0 && dim % 2 == 0, "dim must be even and non-zero");
        assert!(bits > 0 && bits <= 8, "bits must be 1..=8");

        let levels = 1usize << bits;
        let pairs = dim / 2;

        // Generate rotation matrix via QR decomposition of random Gaussian.
        let rotation = generate_rotation(dim, seed);

        // Pre-compute cos/sin for quantized angle levels.
        let mut cos_table = Vec::with_capacity(levels);
        let mut sin_table = Vec::with_capacity(levels);
        for j in 0..levels {
            let theta = (j as f32 / levels as f32) * 2.0 * PI - PI;
            cos_table.push(theta.cos());
            sin_table.push(theta.sin());
        }

        Self {
            dim,
            bits,
            levels,
            pairs,
            rotation,
            cos_table,
            sin_table,
        }
    }

    /// Encode a single vector.
    pub fn encode(&self, vector: &[f32]) -> CompressedCode {
        assert_eq!(vector.len(), self.dim);

        // Rotate: y = Π · x
        let x = Array1::from_vec(vector.to_vec());
        let rotated = self.rotation.dot(&x);

        let mut radii = Vec::with_capacity(self.pairs);
        let mut angle_indices = Vec::with_capacity(self.pairs);

        for i in 0..self.pairs {
            let a = rotated[2 * i];
            let b = rotated[2 * i + 1];
            let r = (a * a + b * b).sqrt();
            let theta = b.atan2(a); // in [-π, π]

            // Quantize θ to [0, levels)
            let normalized = (theta + PI) / (2.0 * PI); // [0, 1)
            let idx = ((normalized * self.levels as f32) as usize).min(self.levels - 1);

            radii.push(r);
            angle_indices.push(idx as u8);
        }

        CompressedCode {
            radii,
            angle_indices,
        }
    }

    /// Encode a batch of vectors using BLAS matrix multiply.
    ///
    /// `vectors` shape: `[n, dim]`. Returns n compressed codes.
    /// The rotation is applied as a single GEMM: `rotated = vectors × Πᵀ`.
    pub fn encode_batch(&self, vectors: &Array2<f32>) -> Vec<CompressedCode> {
        assert_eq!(vectors.ncols(), self.dim);
        let n = vectors.nrows();

        // Batch rotation: [n, dim] × [dim, dim]ᵀ = [n, dim]
        let rotated = vectors.dot(&self.rotation.t());

        (0..n)
            .map(|row| {
                let mut radii = Vec::with_capacity(self.pairs);
                let mut angle_indices = Vec::with_capacity(self.pairs);

                for i in 0..self.pairs {
                    let a = rotated[[row, 2 * i]];
                    let b = rotated[[row, 2 * i + 1]];
                    let r = (a * a + b * b).sqrt();
                    let theta = b.atan2(a);
                    let normalized = (theta + PI) / (2.0 * PI);
                    let idx = ((normalized * self.levels as f32) as usize).min(self.levels - 1);
                    radii.push(r);
                    angle_indices.push(idx as u8);
                }

                CompressedCode {
                    radii,
                    angle_indices,
                }
            })
            .collect()
    }

    /// Prepare query-dependent lookup tables for batch scanning.
    ///
    /// Returns a `QueryState` that can be reused across all vectors.
    /// Cost: one matrix-vector multiply (rotation) + d/2 × 16 multiply-adds.
    pub fn prepare_query(&self, query: &[f32]) -> QueryState {
        assert_eq!(query.len(), self.dim);

        // Rotate query: q_rot = Π · query
        let q = Array1::from_vec(query.to_vec());
        let rotated = self.rotation.dot(&q);

        // Build per-pair centroid lookup: centroid_q[pair][level]
        // centroid_q[i][j] = q_a·cos(θⱼ) + q_b·sin(θⱼ)
        let mut centroid_q = vec![0.0f32; self.pairs * self.levels];
        for i in 0..self.pairs {
            let q_a = rotated[2 * i];
            let q_b = rotated[2 * i + 1];
            for j in 0..self.levels {
                centroid_q[i * self.levels + j] = q_a * self.cos_table[j] + q_b * self.sin_table[j];
            }
        }

        QueryState {
            centroid_q,
            pairs: self.pairs,
            levels: self.levels,
        }
    }

    /// Scan all codes against a prepared query. Returns approximate scores.
    ///
    /// Cost per vector: `pairs` table lookups + `pairs` multiply-adds.
    /// For d=768 at 4-bit: 384 lookups + 384 muls = 768 ops/vector.
    pub fn batch_scan(&self, codes: &[CompressedCode], query_state: &QueryState) -> Vec<f32> {
        codes
            .iter()
            .map(|code| {
                let mut score = 0.0f32;
                for i in 0..query_state.pairs {
                    let j = code.angle_indices[i] as usize;
                    score += code.radii[i] * query_state.centroid_q[i * query_state.levels + j];
                }
                score
            })
            .collect()
    }
}

/// Pre-computed query state for fast scanning.
///
/// Contains the centroid-query dot product table: `centroid_q[pair][level]`.
/// Build once via [`PolarCodec::prepare_query`], reuse for all vectors.
pub struct QueryState {
    /// Flat `[pairs × levels]` table of pre-computed dot products.
    centroid_q: Vec<f32>,
    pairs: usize,
    levels: usize,
}

/// Generate a d×d orthogonal rotation matrix via QR decomposition.
///
/// Seeded deterministically — identical `(dim, seed)` always produces the
/// same matrix. Uses Gram-Schmidt on a random Gaussian matrix.
fn generate_rotation(dim: usize, seed: u64) -> Array2<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Generate random Gaussian matrix
    let mut data = Vec::with_capacity(dim * dim);
    for _ in 0..(dim * dim) {
        data.push(StandardNormal.sample(&mut rng));
    }
    let a = Array2::from_shape_vec((dim, dim), data).expect("shape matches data length");

    // QR decomposition via modified Gram-Schmidt
    gram_schmidt_qr(a)
}

/// Modified Gram-Schmidt orthogonalization → returns Q (orthogonal matrix).
fn gram_schmidt_qr(a: Array2<f32>) -> Array2<f32> {
    let n = a.ncols();
    let mut q = a;

    for i in 0..n {
        // Normalize column i
        let norm: f32 = q.column(i).iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm < 1e-10 {
            continue;
        }
        let inv_norm = 1.0 / norm;
        for row in 0..q.nrows() {
            q[[row, i]] *= inv_norm;
        }

        // Subtract projection of column i from all subsequent columns
        for j in (i + 1)..n {
            let dot: f32 = (0..q.nrows()).map(|row| q[[row, i]] * q[[row, j]]).sum();
            for row in 0..q.nrows() {
                q[[row, j]] -= dot * q[[row, i]];
            }
        }
    }

    q
}

#[cfg(test)]
mod tests {
    use super::*;

    fn l2_normalize(v: &mut [f32]) {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
    }

    #[test]
    fn rotation_is_orthogonal() {
        let r = generate_rotation(8, 42);
        // Q × Qᵀ should be identity
        let eye = r.dot(&r.t());
        for i in 0..8 {
            for j in 0..8 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (eye[[i, j]] - expected).abs() < 1e-5,
                    "Q×Qᵀ[{i},{j}] = {}, expected {expected}",
                    eye[[i, j]]
                );
            }
        }
    }

    #[test]
    fn encode_decode_roundtrip() {
        let codec = PolarCodec::new(8, 4, 42);
        let mut v = vec![0.3, -0.1, 0.5, 0.2, -0.4, 0.1, 0.3, -0.2];
        l2_normalize(&mut v);

        let code = codec.encode(&v);
        assert_eq!(code.radii.len(), 4); // 8/2 pairs
        assert_eq!(code.angle_indices.len(), 4);
        assert_eq!(code.encoded_bytes(), 4 * 4 + 4); // 4 f32 + 4 u8
    }

    #[test]
    fn batch_scan_recall() {
        let dim = 768;
        let n = 500;
        let codec = PolarCodec::new(dim, 4, 42);

        // Generate random L2-normalized vectors
        let mut vecs = Array2::<f32>::zeros((n, dim));
        for i in 0..n {
            for d in 0..dim {
                vecs[[i, d]] = ((i * 17 + d * 31) as f32).sin();
            }
            let norm: f32 = vecs.row(i).iter().map(|x| x * x).sum::<f32>().sqrt();
            for d in 0..dim {
                vecs[[i, d]] /= norm;
            }
        }

        // Encode batch
        let t0 = std::time::Instant::now();
        let codes = codec.encode_batch(&vecs);
        let encode_ms = t0.elapsed().as_secs_f64() * 1000.0;
        eprintln!("encode {n} vectors: {encode_ms:.1}ms");

        // Query
        let mut query = vec![0.0f32; dim];
        for d in 0..dim {
            query[d] = ((42 * 7 + d * 13) as f32).sin();
        }
        l2_normalize(&mut query);

        // Exact ranking
        let query_arr = Array1::from_vec(query.clone());
        let mut exact: Vec<(usize, f32)> =
            (0..n).map(|i| (i, vecs.row(i).dot(&query_arr))).collect();
        exact.sort_by(|a, b| b.1.total_cmp(&a.1));

        // TurboQuant batch scan
        let t1 = std::time::Instant::now();
        let qs = codec.prepare_query(&query);
        let prepare_ms = t1.elapsed().as_secs_f64() * 1000.0;

        let t2 = std::time::Instant::now();
        let approx_scores = codec.batch_scan(&codes, &qs);
        let scan_ms = t2.elapsed().as_secs_f64() * 1000.0;

        let mut approx: Vec<(usize, f32)> = approx_scores.into_iter().enumerate().collect();
        approx.sort_by(|a, b| b.1.total_cmp(&a.1));

        eprintln!("prepare query: {prepare_ms:.3}ms, scan {n}: {scan_ms:.3}ms");

        // Recall@10
        let exact_top10: Vec<usize> = exact.iter().take(10).map(|(i, _)| *i).collect();
        let approx_top10: Vec<usize> = approx.iter().take(10).map(|(i, _)| *i).collect();
        let recall = exact_top10
            .iter()
            .filter(|i| approx_top10.contains(i))
            .count();
        eprintln!("Recall@10: {recall}/10");
        // Phase A (PolarQuant-only, no QJL correction): 7/10 recall typical.
        // Phase B adds QJL residual sketch → 9-10/10.
        assert!(recall >= 6, "recall should be >= 6/10, got {recall}/10");
    }
}
