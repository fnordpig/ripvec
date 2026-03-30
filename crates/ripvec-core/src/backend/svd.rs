//! Low-rank SVD decomposition for FFN Wi weight matrices.
//!
//! Decomposes `Wi [rows × cols]` into two factors `A [k × cols]` and `B [rows × k]`
//! such that `Wi ≈ B @ A`. The original GEMM `Y = X @ Wi^T` becomes two smaller
//! GEMMs: `Z = X @ A^T` then `Y = Z @ B^T`.

use crate::embed::SvdRank;

/// Result of decomposing one Wi weight matrix.
pub struct SvdFactors {
    /// First factor `[k × cols]` — stored row-major for `gemm(X, A, Z, M, k, cols, true)`.
    pub a: Vec<f32>,
    /// Second factor `[rows × k]` — stored row-major for `gemm(Z, B, Y, M, rows, k, true)`.
    pub b: Vec<f32>,
    /// Effective rank used.
    pub k: usize,
    /// Frobenius reconstruction error ratio `‖Wi - B@A‖_F / ‖Wi‖_F`.
    pub error: f32,
}

/// Find the smallest rank `k` such that the reconstruction error is within `threshold`.
///
/// Error ratio = `sqrt(sum(s[k:]²) / sum(s[:]²))` where `s` are singular values.
fn auto_rank(singular_values: &[f32], threshold: f32) -> usize {
    let total_sq: f64 = singular_values
        .iter()
        .map(|&s| (s as f64) * (s as f64))
        .sum();
    if total_sq == 0.0 {
        return singular_values.len();
    }
    let threshold_sq = (threshold as f64) * (threshold as f64) * total_sq;
    let mut residual_sq = total_sq;
    for (k, &s) in singular_values.iter().enumerate() {
        residual_sq -= (s as f64) * (s as f64);
        if residual_sq <= threshold_sq {
            return k + 1;
        }
    }
    singular_values.len()
}

/// Decompose a Wi weight matrix `[rows × cols]` into low-rank factors.
///
/// Returns `None` if `svd_rank` is `Disabled` or the rank equals full rank
/// (no benefit to factoring).
pub fn decompose_wi(
    wi_data: &[f32],
    rows: usize,
    cols: usize,
    svd_rank: &SvdRank,
) -> Option<SvdFactors> {
    if matches!(svd_rank, SvdRank::Disabled) {
        return None;
    }

    // Build faer matrix from flat row-major data
    let wi = faer::Mat::from_fn(rows, cols, |i, j| wi_data[i * cols + j] as f64);

    // Compute thin SVD: Wi = U @ diag(S) @ V^T
    let svd = wi.thin_svd().expect("SVD computation failed");
    let u = svd.U();
    let s = svd.S();
    let v = svd.V();

    let min_dim = rows.min(cols);

    // Collect singular values
    let svals: Vec<f32> = s.column_vector().iter().map(|&val| val as f32).collect();

    // Determine effective rank
    let effective_k = match svd_rank {
        SvdRank::Auto => auto_rank(&svals, 0.01), // 1% threshold
        SvdRank::Fixed(k) => (*k).min(min_dim),
        SvdRank::Disabled => unreachable!(),
    };

    if effective_k >= min_dim {
        return None; // Full rank — no FLOP savings
    }

    // Compute reconstruction error
    let total_sq: f64 = svals.iter().map(|&s| (s as f64) * (s as f64)).sum();
    let residual_sq: f64 = svals[effective_k..]
        .iter()
        .map(|&s| (s as f64) * (s as f64))
        .sum();
    let error = if total_sq > 0.0 {
        (residual_sq / total_sq).sqrt() as f32
    } else {
        0.0
    };

    // Build factors using col_iter for faer 0.21 compatibility:
    // A_weight = diag(sqrt(S_k)) @ V_k^T  →  [k, cols]
    // B_weight = U_k @ diag(sqrt(S_k))    →  [rows, k]
    let u_cols: Vec<Vec<f64>> = u
        .col_iter()
        .take(effective_k)
        .map(|c| c.iter().copied().collect())
        .collect();
    let v_cols: Vec<Vec<f64>> = v
        .col_iter()
        .take(effective_k)
        .map(|c| c.iter().copied().collect())
        .collect();

    let mut a_data = vec![0.0f32; effective_k * cols];
    let mut b_data = vec![0.0f32; rows * effective_k];

    for ki in 0..effective_k {
        let sqrt_s = (svals[ki] as f64).sqrt();
        // A row ki = sqrt(s[ki]) * V[:, ki]^T  →  [cols]
        for (j, &vval) in v_cols[ki].iter().enumerate() {
            a_data[ki * cols + j] = (sqrt_s * vval) as f32;
        }
        // B column ki = sqrt(s[ki]) * U[:, ki]  →  [rows]
        for (i, &uval) in u_cols[ki].iter().enumerate() {
            b_data[i * effective_k + ki] = (sqrt_s * uval) as f32;
        }
    }

    Some(SvdFactors {
        a: a_data,
        b: b_data,
        k: effective_k,
        error,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn svd_reconstruction_within_threshold() {
        let rows = 64;
        let cols = 32;
        let mut data = vec![0.0f32; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                data[i * cols + j] =
                    (i as f32) * (j as f32) + 0.5 * ((i + 1) as f32) * ((j + 2) as f32);
            }
        }

        let factors = decompose_wi(&data, rows, cols, &SvdRank::Fixed(4)).unwrap();
        assert_eq!(factors.k, 4);
        assert!(
            factors.error < 0.01,
            "error {} should be < 1%",
            factors.error
        );

        // Verify reconstruction: Wi ≈ B @ A
        for i in 0..rows {
            for j in 0..cols {
                let mut reconstructed = 0.0f32;
                for ki in 0..factors.k {
                    reconstructed += factors.b[i * factors.k + ki] * factors.a[ki * cols + j];
                }
                let original = data[i * cols + j];
                let diff = (reconstructed - original).abs();
                let scale = original.abs().max(1.0);
                assert!(
                    diff / scale < 0.05,
                    "reconstruction error too large at [{i},{j}]: orig={original}, got={reconstructed}"
                );
            }
        }
    }

    #[test]
    fn auto_rank_finds_correct_rank() {
        let svals = vec![100.0, 50.0, 25.0, 0.1, 0.05, 0.01];
        let k = super::auto_rank(&svals, 0.01);
        assert!(k <= 4, "auto_rank should find k ≤ 4, got {k}");
        assert!(k >= 3, "auto_rank should find k ≥ 3, got {k}");
    }

    #[test]
    fn disabled_returns_none() {
        let data = vec![1.0; 4 * 3];
        assert!(decompose_wi(&data, 4, 3, &SvdRank::Disabled).is_none());
    }
}
