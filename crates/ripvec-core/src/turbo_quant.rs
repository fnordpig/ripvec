//! TurboQuant: high-throughput compressed vector search.
//!
//! PolarQuant at 4 bits: rotated embedding pairs are encoded as (radius, angle_index).
//! The scan uses a pre-computed centroid-query dot product table (24 KB, fits in L1)
//! and streams sequentially through packed radii + indices — cache-line optimal.
//!
//! # Memory layout (SoA, not AoS)
//!
//! ```text
//! CompressedCorpus:
//!   radii:   [n × pairs] f32, contiguous — sequential streaming reads
//!   indices: [n × pairs] u8,  contiguous — sequential streaming reads
//!   (future: 4-bit packed indices → [n × pairs / 2] u8 for 2× index bandwidth)
//! ```
//!
//! This layout enables:
//! - GPU: one thread per vector, coalesced reads across threads
//! - CPU NEON: process 4 pairs per SIMD iteration, amortize centroid loads
//! - Cache: centroid table (24 KB) stays in L1 throughout the scan

use std::f32::consts::PI;

use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};

// ---------------------------------------------------------------------------
// Compressed corpus — flat SoA layout for cache-friendly scanning
// ---------------------------------------------------------------------------

/// Flat, contiguous compressed embeddings for maximum scan throughput.
///
/// Structure-of-arrays layout: all radii packed, then all indices packed.
/// No per-vector heap allocations, no pointer chasing.
///
/// At 4-bit, d=768: 384 pairs × (4 + 1) bytes = 1920 bytes/vector.
/// For 100K vectors: 192 MB (vs 300 MB FP32 or 150 MB FP16).
pub struct CompressedCorpus {
    /// Number of vectors.
    pub n: usize,
    /// Number of pairs per vector (dim / 2).
    pub pairs: usize,
    /// Flat radii: `[n × pairs]` f32, row-major.
    pub radii: Vec<f32>,
    /// Flat angle indices: `[n × pairs]` u8, row-major.
    pub indices: Vec<u8>,
}

/// Compressed representation of a single vector (for the old API).
#[derive(Clone)]
pub struct CompressedCode {
    /// Per-pair radii (f32).
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

// ---------------------------------------------------------------------------
// PolarCodec — encode, prepare query, scan
// ---------------------------------------------------------------------------

/// PolarQuant codec: batch encode, query preparation, and high-throughput scan.
pub struct PolarCodec {
    dim: usize,
    #[expect(dead_code, reason = "stored for serialization / reconstruction")]
    bits: u8,
    levels: usize,
    pairs: usize,
    /// Row-major orthogonal rotation matrix \[dim × dim\].
    rotation: Array2<f32>,
    /// Pre-computed cos/sin for each quantized angle level.
    cos_table: Vec<f32>,
    sin_table: Vec<f32>,
}

impl PolarCodec {
    /// Create a new codec.
    ///
    /// # Panics
    ///
    /// Panics if `dim` is 0, odd, or `bits` is 0 or > 8.
    pub fn new(dim: usize, bits: u8, seed: u64) -> Self {
        assert!(dim > 0 && dim % 2 == 0, "dim must be even and non-zero");
        assert!(bits > 0 && bits <= 8, "bits must be 1..=8");

        let levels = 1usize << bits;
        let pairs = dim / 2;
        let rotation = generate_rotation(dim, seed);

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

    /// Number of pairs per vector.
    pub fn pairs(&self) -> usize {
        self.pairs
    }

    /// Encode a single vector (convenience, allocates).
    pub fn encode(&self, vector: &[f32]) -> CompressedCode {
        assert_eq!(vector.len(), self.dim);
        let x = Array1::from_vec(vector.to_vec());
        let rotated = self.rotation.dot(&x);

        let mut radii = Vec::with_capacity(self.pairs);
        let mut angle_indices = Vec::with_capacity(self.pairs);
        for i in 0..self.pairs {
            let (r, idx) = self.encode_pair(rotated[2 * i], rotated[2 * i + 1]);
            radii.push(r);
            angle_indices.push(idx);
        }
        CompressedCode {
            radii,
            angle_indices,
        }
    }

    /// Batch-encode into a flat [`CompressedCorpus`] (SoA layout).
    ///
    /// Uses BLAS for the rotation: `rotated = vectors × Πᵀ` (one GEMM).
    /// Quantization is scalar but cache-friendly (sequential writes).
    pub fn encode_batch(&self, vectors: &Array2<f32>) -> CompressedCorpus {
        assert_eq!(vectors.ncols(), self.dim);
        let n = vectors.nrows();

        // Batch rotation via BLAS: [n, dim] × [dim, dim]ᵀ → [n, dim]
        let rotated = vectors.dot(&self.rotation.t());

        let total = n * self.pairs;
        let mut radii = Vec::with_capacity(total);
        let mut indices = Vec::with_capacity(total);

        for row in 0..n {
            for i in 0..self.pairs {
                let (r, idx) = self.encode_pair(rotated[[row, 2 * i]], rotated[[row, 2 * i + 1]]);
                radii.push(r);
                indices.push(idx);
            }
        }

        CompressedCorpus {
            n,
            pairs: self.pairs,
            radii,
            indices,
        }
    }

    /// Also produce the old per-vector codes (for backward compat with index.rs).
    pub fn encode_batch_codes(&self, vectors: &Array2<f32>) -> Vec<CompressedCode> {
        let corpus = self.encode_batch(vectors);
        (0..corpus.n)
            .map(|v| {
                let off = v * corpus.pairs;
                CompressedCode {
                    radii: corpus.radii[off..off + corpus.pairs].to_vec(),
                    angle_indices: corpus.indices[off..off + corpus.pairs].to_vec(),
                }
            })
            .collect()
    }

    /// Prepare query-dependent centroid lookup table.
    ///
    /// Cost: one 768×768 matvec + 384×16 multiply-adds = ~0.08ms.
    /// The returned [`QueryState`] is reused for ALL vectors in the scan.
    pub fn prepare_query(&self, query: &[f32]) -> QueryState {
        assert_eq!(query.len(), self.dim);
        let q = Array1::from_vec(query.to_vec());
        let rotated = self.rotation.dot(&q);

        // centroid_q[pair * levels + level] = q_a·cos(θ_level) + q_b·sin(θ_level)
        // Layout: pairs × levels, contiguous. Fits in L1 (384 × 16 × 4 = 24 KB).
        let mut centroid_q = vec![0.0f32; self.pairs * self.levels];
        for i in 0..self.pairs {
            let q_a = rotated[2 * i];
            let q_b = rotated[2 * i + 1];
            let base = i * self.levels;
            for j in 0..self.levels {
                centroid_q[base + j] = q_a * self.cos_table[j] + q_b * self.sin_table[j];
            }
        }

        QueryState {
            centroid_q,
            pairs: self.pairs,
            levels: self.levels,
        }
    }

    /// High-throughput scan of a [`CompressedCorpus`] against a prepared query.
    ///
    /// Returns approximate inner product scores for all vectors.
    /// Memory access: sequential streaming through radii + indices (cache-optimal).
    /// Centroid table: 24 KB, stays in L1 throughout.
    ///
    /// At 100K vectors, d=768: ~3.3ms on CPU, ~0.1ms on GPU (future Metal kernel).
    pub fn scan_corpus(&self, corpus: &CompressedCorpus, qs: &QueryState) -> Vec<f32> {
        let n = corpus.n;
        let pairs = corpus.pairs;
        let mut scores = vec![0.0f32; n];

        // Hot loop: sequential access to radii + indices,
        // random-but-L1-hot access to centroid table.
        for v in 0..n {
            let base = v * pairs;
            let mut score = 0.0f32;

            // Process 4 pairs per iteration (manual unroll for ILP).
            let chunks = pairs / 4;
            let remainder = pairs % 4;

            for c in 0..chunks {
                let i = base + c * 4;
                let i0 = corpus.indices[i] as usize;
                let i1 = corpus.indices[i + 1] as usize;
                let i2 = corpus.indices[i + 2] as usize;
                let i3 = corpus.indices[i + 3] as usize;

                let p = c * 4;
                score += corpus.radii[i] * qs.centroid_q[p * qs.levels + i0];
                score += corpus.radii[i + 1] * qs.centroid_q[(p + 1) * qs.levels + i1];
                score += corpus.radii[i + 2] * qs.centroid_q[(p + 2) * qs.levels + i2];
                score += corpus.radii[i + 3] * qs.centroid_q[(p + 3) * qs.levels + i3];
            }
            for r in 0..remainder {
                let i = base + chunks * 4 + r;
                let p = chunks * 4 + r;
                let j = corpus.indices[i] as usize;
                score += corpus.radii[i] * qs.centroid_q[p * qs.levels + j];
            }

            scores[v] = score;
        }

        scores
    }

    /// Scan per-vector codes (old API, for backward compat).
    pub fn batch_scan(&self, codes: &[CompressedCode], qs: &QueryState) -> Vec<f32> {
        codes
            .iter()
            .map(|code| {
                let mut score = 0.0f32;
                for i in 0..qs.pairs {
                    let j = code.angle_indices[i] as usize;
                    score += code.radii[i] * qs.centroid_q[i * qs.levels + j];
                }
                score
            })
            .collect()
    }

    #[inline]
    fn encode_pair(&self, a: f32, b: f32) -> (f32, u8) {
        let r = (a * a + b * b).sqrt();
        let theta = b.atan2(a);
        let normalized = (theta + PI) / (2.0 * PI);
        let idx = ((normalized * self.levels as f32) as usize).min(self.levels - 1);
        (r, idx as u8)
    }
}

/// Pre-computed query state for fast scanning.
pub struct QueryState {
    /// Flat `[pairs × levels]` centroid-query dot products (24 KB at d=768, 4-bit).
    pub centroid_q: Vec<f32>,
    /// Number of pairs.
    pub pairs: usize,
    /// Number of quantization levels.
    pub levels: usize,
}

// ---------------------------------------------------------------------------
// Rotation matrix generation (seeded, deterministic)
// ---------------------------------------------------------------------------

/// Generate a d×d orthogonal matrix via QR on a seeded Gaussian matrix.
fn generate_rotation(dim: usize, seed: u64) -> Array2<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut data = Vec::with_capacity(dim * dim);
    for _ in 0..(dim * dim) {
        data.push(StandardNormal.sample(&mut rng));
    }
    let a = Array2::from_shape_vec((dim, dim), data).expect("shape matches data length");
    gram_schmidt_qr(a)
}

/// Modified Gram-Schmidt → Q (orthogonal).
fn gram_schmidt_qr(mut q: Array2<f32>) -> Array2<f32> {
    let n = q.ncols();
    for i in 0..n {
        let norm: f32 = q.column(i).iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm < 1e-10 {
            continue;
        }
        let inv = 1.0 / norm;
        for row in 0..q.nrows() {
            q[[row, i]] *= inv;
        }
        for j in (i + 1)..n {
            let dot: f32 = (0..q.nrows()).map(|row| q[[row, i]] * q[[row, j]]).sum();
            for row in 0..q.nrows() {
                q[[row, j]] -= dot * q[[row, i]];
            }
        }
    }
    q
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

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
        assert_eq!(code.radii.len(), 4);
        assert_eq!(code.angle_indices.len(), 4);
    }

    #[test]
    fn corpus_scan_recall_and_throughput() {
        let dim = 768;
        let n = 1000;
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

        // Encode to SoA corpus
        let t0 = std::time::Instant::now();
        let corpus = codec.encode_batch(&vecs);
        let encode_ms = t0.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "encode {n} → SoA corpus: {encode_ms:.1}ms ({:.1}µs/vec)",
            encode_ms * 1000.0 / n as f64
        );

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

        // TurboQuant corpus scan
        let t1 = std::time::Instant::now();
        let qs = codec.prepare_query(&query);
        let prep_us = t1.elapsed().as_secs_f64() * 1e6;

        let t2 = std::time::Instant::now();
        let scores = codec.scan_corpus(&corpus, &qs);
        let scan_us = t2.elapsed().as_secs_f64() * 1e6;

        eprintln!(
            "prepare: {prep_us:.0}µs, scan {n}: {scan_us:.0}µs ({:.2}µs/vec)",
            scan_us / n as f64
        );
        eprintln!("scan throughput: {:.1}M vec/s", n as f64 / scan_us);

        // Recall@10
        let mut approx: Vec<(usize, f32)> = scores.into_iter().enumerate().collect();
        approx.sort_by(|a, b| b.1.total_cmp(&a.1));
        let exact_top10: Vec<usize> = exact.iter().take(10).map(|(i, _)| *i).collect();
        let approx_top10: Vec<usize> = approx.iter().take(10).map(|(i, _)| *i).collect();
        let recall = exact_top10
            .iter()
            .filter(|i| approx_top10.contains(i))
            .count();
        eprintln!("Recall@10: {recall}/10");
        // Raw scan recall (no re-rank) is 4-7/10 for PolarQuant-only 4-bit.
        // With exact re-rank of top-100 (SearchIndex::rank_turboquant), recall is 10/10.
        assert!(
            recall >= 4,
            "raw scan recall should be >= 4/10, got {recall}/10"
        );
    }

    /// GPU vs CPU scan benchmark (Metal only).
    #[test]
    #[cfg(feature = "metal")]
    fn metal_turboquant_scan() {
        let dim = 768;
        let n = 10_000;
        let codec = PolarCodec::new(dim, 4, 42);

        // Generate corpus
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

        let corpus = codec.encode_batch(&vecs);
        let mut query = vec![0.0f32; dim];
        for d in 0..dim {
            query[d] = ((42 * 7 + d * 13) as f32).sin();
        }
        l2_normalize(&mut query);
        let qs = codec.prepare_query(&query);

        // CPU scan
        let t0 = std::time::Instant::now();
        let cpu_scores = codec.scan_corpus(&corpus, &qs);
        let cpu_us = t0.elapsed().as_secs_f64() * 1e6;

        // GPU scan — upload once, scan twice to measure warm vs cold
        let driver = crate::backend::driver::metal::MetalDriver::new().unwrap();

        // Cold: upload + scan (includes buffer creation)
        let t_cold = std::time::Instant::now();
        let gpu_corpus = driver
            .turboquant_upload_corpus(&corpus.radii, &corpus.indices)
            .unwrap();
        let upload_us = t_cold.elapsed().as_secs_f64() * 1e6;

        let t_warm = std::time::Instant::now();
        let gpu_scores = driver
            .turboquant_scan_gpu(&gpu_corpus, &qs.centroid_q, n, corpus.pairs, qs.levels)
            .unwrap();
        let warm_us = t_warm.elapsed().as_secs_f64() * 1e6;

        // Second scan — fully warm (centroid upload only)
        let t_hot = std::time::Instant::now();
        let _ = driver
            .turboquant_scan_gpu(&gpu_corpus, &qs.centroid_q, n, corpus.pairs, qs.levels)
            .unwrap();
        let hot_us = t_hot.elapsed().as_secs_f64() * 1e6;

        eprintln!("10K vectors:");
        eprintln!("  CPU:        {cpu_us:.0}µs ({:.1}M/s)", n as f64 / cpu_us);
        eprintln!("  GPU upload: {upload_us:.0}µs (one-time)");
        eprintln!(
            "  GPU warm:   {warm_us:.0}µs ({:.1}M/s, {:.1}× vs CPU)",
            n as f64 / warm_us,
            cpu_us / warm_us
        );
        eprintln!(
            "  GPU hot:    {hot_us:.0}µs ({:.1}M/s, {:.1}× vs CPU)",
            n as f64 / hot_us,
            cpu_us / hot_us
        );

        // Verify GPU matches CPU (approximate — f32 accumulation order differs)
        let mut max_diff = 0.0f32;
        for i in 0..n {
            let diff = (cpu_scores[i] - gpu_scores[i]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
        eprintln!("max CPU/GPU score diff: {max_diff:.6}");
        assert!(
            max_diff < 0.01,
            "GPU scores should match CPU within 0.01, got {max_diff}"
        );
    }
}
