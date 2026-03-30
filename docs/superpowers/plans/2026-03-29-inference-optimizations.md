# ModernBERT Inference Optimizations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Increase ModernBERT embedding throughput from 73.6/s to ≥100/s while staying within recall budgets (≤10% semantic Recall@10 loss, ≤2% hybrid).

**Architecture:** Four stacked optimizations applied in the forward pass: (1) `fast::exp()` in softmax kernels (always-on), (2) low-rank FFN via SVD at load time replacing one Wi GEMM with two smaller GEMMs, (3) layer skipping via a skip set in the layer loop, (4) token pruning at layer 11 dropping low-information tokens. Each optimization has a runtime flag; all are disabled by default except `fast::exp()`.

**Tech Stack:** Rust 2024, Metal Shading Language, `faer` (pure-Rust SVD), existing `ndarray`/Metal/MPS infrastructure.

**Spec:** `docs/superpowers/specs/2026-03-29-inference-optimizations-design.md`

**Baselines:**
- bench.py: `scripts/bench/results/20260329T220434.json` (MPS 73.6/s, compute 59.0/s, CPU 72.3/s)
- tracemeld: `mps-fp16-22L-baseline`, `compute-22L-baseline`, `cpu-22L-baseline`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `crates/ripvec-core/src/backend/metal_kernels.rs` | Modify | `fast::exp()` in 4 softmax kernels; 2 new token-pruning kernels |
| `crates/ripvec-core/src/backend/arch/modern_bert.rs` | Modify | SVD fields on layer weights, SVD-aware `ffn_sublayer`/`ffn_sublayer_f16`, layer skip + token pruning in `forward()` |
| `crates/ripvec-core/src/backend/driver/metal.rs` | Modify | SVD at load time, `ensure_fp16` on factors, token-pruning dispatch methods |
| `crates/ripvec-core/src/backend/driver/cpu.rs` | Modify | SVD at load time for CPU path |
| `crates/ripvec-core/src/backend/driver/mod.rs` | Modify | `compute_token_distances` + `gather_tokens` trait methods |
| `crates/ripvec-core/src/backend/mod.rs` | Modify | Pass `svd_rank` through `load_modernbert_metal`/`load_modernbert_cpu` |
| `crates/ripvec-core/src/backend/svd.rs` | Create | Pure-Rust SVD decomposition utility (shared by Metal + CPU) |
| `crates/ripvec-core/src/embed.rs` | Modify | `SearchConfig` gets `svd_rank`, `prune_ratio`, `skip_layers` |
| `crates/ripvec/src/cli.rs` | Modify | `--svd-rank`, `--prune-ratio`, `--skip-layers` CLI flags |
| `crates/ripvec/src/main.rs` | Modify | Wire new CLI flags into `SearchConfig` and `load_backend` |
| `crates/ripvec-core/Cargo.toml` | Modify | Add `faer` dependency |

---

### Task 1: `fast::exp()` in Softmax Kernels

**Files:**
- Modify: `crates/ripvec-core/src/backend/metal_kernels.rs:304,971,1338,1400`

- [ ] **Step 1: Change `exp()` to `fast::exp()` in all 4 softmax kernels**

In `crates/ripvec-core/src/backend/metal_kernels.rs`, make these 4 substitutions:

Line 304 (in `fused_scale_mask_softmax_kernel`):
```metal
// BEFORE:
float val = exp(row_data[i] - row_max);
// AFTER:
float val = fast::exp(row_data[i] - row_max);
```

Line 971 (in `fused_scale_mask_softmax_windowed_kernel`):
```metal
// BEFORE:
float val = exp(row_data[i] - row_max);
// AFTER:
float val = fast::exp(row_data[i] - row_max);
```

Line 1338 (in `fused_scale_mask_softmax_f16_kernel`):
```metal
// BEFORE:
float val = exp(float(row_data[i]) - row_max);
// AFTER:
float val = fast::exp(float(row_data[i]) - row_max);
```

Line 1400 (in `fused_scale_mask_softmax_windowed_f16_kernel`):
```metal
// BEFORE:
float val = exp(float(row_data[i]) - row_max);
// AFTER:
float val = fast::exp(float(row_data[i]) - row_max);
```

- [ ] **Step 2: Build and verify compilation**

```bash
cargo build --release 2>&1 | tail -5
```

Expected: `Finished release profile` — MSL compiles at runtime but syntax errors would surface in the raw string.

- [ ] **Step 3: Run correctness validation**

```bash
uv run scripts/bench/bench.py --configs mps compute --layers 22 --validate --no-build
```

Expected: "All configs produce identical rankings." — `fast::exp()` preserves softmax ordering.

- [ ] **Step 4: Commit**

```bash
git add crates/ripvec-core/src/backend/metal_kernels.rs
git commit -m "perf(metal): fast::exp() in all softmax kernels

Reduces softmax latency ~2-3% by using Metal's reduced-precision
hardware path (~11-bit mantissa). Ranking preserved — only near-zero
attention weight magnitudes affected."
```

---

### Task 2: CLI Flags + Config Plumbing

**Files:**
- Modify: `crates/ripvec/src/cli.rs`
- Modify: `crates/ripvec-core/src/embed.rs`
- Modify: `crates/ripvec/src/main.rs`
- Modify: `crates/ripvec-core/src/backend/mod.rs`
- Modify: `crates/ripvec-core/src/backend/arch/modern_bert.rs`

- [ ] **Step 1: Add `SvdRank` enum to `embed.rs`**

In `crates/ripvec-core/src/embed.rs`, add before the `SearchConfig` struct:

```rust
/// SVD rank selection for low-rank FFN approximation.
#[derive(Debug, Clone, Default)]
pub enum SvdRank {
    /// Disabled — use original Wi weight (default).
    #[default]
    Disabled,
    /// Per-layer rank from Frobenius norm threshold (1% reconstruction error).
    Auto,
    /// Fixed rank for all layers.
    Fixed(usize),
}
```

- [ ] **Step 2: Add fields to `SearchConfig`**

In `crates/ripvec-core/src/embed.rs`, add three fields to the `SearchConfig` struct:

```rust
    /// SVD rank for low-rank FFN approximation. Consumed at model load time.
    pub svd_rank: SvdRank,

    /// Token pruning ratio at layer 11 (0.0 = disabled, 0.5 = drop 50%).
    pub prune_ratio: f32,

    /// Layer indices to skip entirely in the encoder.
    pub skip_layers: Vec<usize>,
```

Update the `Default` impl if there is one, or ensure these default to `SvdRank::Disabled`, `0.0`, and `vec![]`.

- [ ] **Step 3: Add CLI flags to `cli.rs`**

In `crates/ripvec/src/cli.rs`, add after the `layers` field (~line 91):

```rust
    /// SVD rank for low-rank FFN approximation.
    /// 0 = disabled (default), "auto" = per-layer from Frobenius threshold.
    #[arg(long, default_value = "0")]
    pub svd_rank: String,

    /// Token pruning ratio at layer 11 (0.0 = disabled, 0.5 = drop 50%).
    #[arg(long, default_value_t = 0.0)]
    pub prune_ratio: f32,

    /// Comma-separated layer indices to skip (e.g., "6,7,13,14").
    #[arg(long, default_value = "")]
    pub skip_layers: String,
```

- [ ] **Step 4: Wire CLI flags into `SearchConfig` in `main.rs`**

In `crates/ripvec/src/main.rs`, in `load_pipeline()` where `SearchConfig` is constructed (~line 184):

```rust
    let svd_rank = match args.svd_rank.as_str() {
        "0" => ripvec_core::embed::SvdRank::Disabled,
        "auto" => ripvec_core::embed::SvdRank::Auto,
        n => ripvec_core::embed::SvdRank::Fixed(
            n.parse::<usize>().expect("--svd-rank must be 0, 'auto', or an integer"),
        ),
    };
    let skip_layers: Vec<usize> = args
        .skip_layers
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| s.trim().parse::<usize>().expect("--skip-layers must be comma-separated integers"))
        .collect();

    let search_cfg = ripvec_core::embed::SearchConfig {
        batch_size: args.batch_size,
        max_tokens: args.max_tokens,
        chunk: ripvec_core::chunk::ChunkConfig {
            max_chunk_bytes: args.max_chunk_bytes,
            window_size: args.window_size,
            window_overlap: args.window_overlap,
        },
        text_mode: args.text_mode,
        cascade_dim: None,
        file_type: args.file_type.clone(),
        mode,
        svd_rank,
        prune_ratio: args.prune_ratio,
        skip_layers,
    };
```

- [ ] **Step 5: Add `skip_layers` and `prune_ratio` to `ModernBertArch`**

In `crates/ripvec-core/src/backend/arch/modern_bert.rs`, add to the `ModernBertArch` struct:

```rust
pub struct ModernBertArch<T> {
    pub weights: ModernBertWeights<T>,
    pub global_rope: RopeCache<T>,
    pub local_rope: RopeCache<T>,
    pub max_layers: Option<usize>,
    /// Layer indices to skip entirely.
    pub skip_layers: std::collections::HashSet<usize>,
    /// Token pruning ratio at layer 11 (0.0 = disabled).
    pub prune_ratio: f32,
}
```

Update all construction sites of `ModernBertArch` to include `skip_layers: Default::default()` and `prune_ratio: 0.0`.

- [ ] **Step 6: Pass `svd_rank` through `load_modernbert_metal` / `load_modernbert_cpu`**

In `crates/ripvec-core/src/backend/mod.rs`, change `load_modernbert_metal` and `load_modernbert_cpu` signatures to accept `svd_rank`:

```rust
#[cfg(feature = "metal")]
pub fn load_modernbert_metal(
    model_repo: &str,
    max_layers: Option<usize>,
    svd_rank: &crate::embed::SvdRank,
) -> crate::Result<Box<dyn EmbedBackend>> {
```

Similarly for `load_modernbert_cpu`. After `arch.max_layers = max_layers;`, add:

```rust
    // SVD rank is consumed at load time — no need to store on arch.
    // The SVD factors are already in the layer weights at this point.
```

(SVD decomposition will be added in Task 3.)

Update all callers of these functions in `detect_backends()` and `load_backend()` to pass `&SvdRank::Disabled` for now.

- [ ] **Step 7: Pass `skip_layers` and `prune_ratio` from SearchConfig to arch**

In `crates/ripvec-core/src/backend/mod.rs`, after `arch.max_layers = max_layers;` in both Metal and CPU loaders:

```rust
    // skip_layers and prune_ratio will be set by the caller via SearchConfig.
    // For now, defaults are applied. The search() function in embed.rs will
    // configure these on the GenericBackend before inference.
```

The actual plumbing of runtime flags (skip_layers, prune_ratio) from `SearchConfig` to the arch will happen through the `EmbedBackend` trait. For now, ensure `SearchConfig` carries the values and they reach `load_pipeline()`.

- [ ] **Step 8: Build and verify**

```bash
cargo check --workspace
```

Expected: Compiles cleanly. No runtime behavior change yet — all new flags default to disabled.

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "feat: add --svd-rank, --prune-ratio, --skip-layers CLI flags

Plumbs optimization config from CLI through SearchConfig to
ModernBertArch. All disabled by default. No behavioral change."
```

---

### Task 3: Low-Rank FFN via SVD

**Files:**
- Create: `crates/ripvec-core/src/backend/svd.rs`
- Modify: `crates/ripvec-core/Cargo.toml`
- Modify: `crates/ripvec-core/src/backend/mod.rs` (add `pub mod svd;`)
- Modify: `crates/ripvec-core/src/backend/arch/modern_bert.rs`
- Modify: `crates/ripvec-core/src/backend/driver/metal.rs`
- Modify: `crates/ripvec-core/src/backend/driver/cpu.rs`

- [ ] **Step 1: Add `faer` dependency**

In `crates/ripvec-core/Cargo.toml`, add under `[dependencies]`:

```toml
faer = "0.21"
```

Run `cargo check -p ripvec-core` to verify it resolves.

- [ ] **Step 2: Create SVD utility module**

Create `crates/ripvec-core/src/backend/svd.rs`:

```rust
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
    let total_sq: f64 = singular_values.iter().map(|&s| (s as f64) * (s as f64)).sum();
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
    let k = match svd_rank {
        SvdRank::Disabled => return None,
        SvdRank::Auto => {
            // Compute full SVD to determine auto rank
            0 // placeholder — will be set after SVD
        }
        SvdRank::Fixed(k) => {
            if *k >= cols.min(rows) {
                return None; // Full rank — no benefit
            }
            *k
        }
    };

    // Build faer matrix from flat row-major data
    let wi = faer::Mat::from_fn(rows, cols, |i, j| wi_data[i * cols + j]);

    // Compute thin SVD: Wi = U @ diag(S) @ V^T
    // U: [rows, min(rows,cols)], S: [min(rows,cols)], V: [cols, min(rows,cols)]
    let svd = wi.thin_svd();
    let u = svd.u();
    let s = svd.s_diagonal();
    let v = svd.v();

    let min_dim = rows.min(cols);

    // Collect singular values
    let svals: Vec<f32> = (0..min_dim).map(|i| s[(i, i)] as f32).collect();

    // Determine effective rank
    let effective_k = match svd_rank {
        SvdRank::Auto => auto_rank(&svals, 0.01), // 1% threshold
        SvdRank::Fixed(k) => *k,
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

    // Build factors:
    // A_weight = diag(sqrt(S_k)) @ V_k^T  →  [k, cols]
    // B_weight = U_k @ diag(sqrt(S_k))    →  [rows, k]
    let mut a_data = vec![0.0f32; effective_k * cols];
    let mut b_data = vec![0.0f32; rows * effective_k];

    for ki in 0..effective_k {
        let sqrt_s = (svals[ki] as f64).sqrt();
        // A row ki = sqrt(s[ki]) * V[:, ki]^T  →  [cols]
        for j in 0..cols {
            a_data[ki * cols + j] = (sqrt_s * v[(j, ki)]) as f32;
        }
        // B column ki = sqrt(s[ki]) * U[:, ki]  →  [rows]
        for i in 0..rows {
            b_data[i * effective_k + ki] = (sqrt_s * u[(i, ki)]) as f32;
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
        // Create a rank-2 matrix: outer product of two pairs
        let rows = 64;
        let cols = 32;
        let mut data = vec![0.0f32; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                data[i * cols + j] = (i as f32) * (j as f32) + 0.5 * ((i + 1) as f32) * ((j + 2) as f32);
            }
        }

        let factors = decompose_wi(&data, rows, cols, &SvdRank::Fixed(4)).unwrap();
        assert!(factors.k == 4);
        assert!(factors.error < 0.01, "error {} should be < 1%", factors.error);

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
        // Singular values with sharp dropoff after rank 3
        let svals = vec![100.0, 50.0, 25.0, 0.1, 0.05, 0.01];
        let k = super::auto_rank(&svals, 0.01); // 1% threshold
        assert!(k <= 4, "auto_rank should find k ≤ 4, got {k}");
        assert!(k >= 3, "auto_rank should find k ≥ 3, got {k}");
    }

    #[test]
    fn disabled_returns_none() {
        let data = vec![1.0; 4 * 3];
        assert!(decompose_wi(&data, 4, 3, &SvdRank::Disabled).is_none());
    }
}
```

- [ ] **Step 3: Register the module**

In `crates/ripvec-core/src/backend/mod.rs`, add:

```rust
pub mod svd;
```

- [ ] **Step 4: Run SVD unit tests**

```bash
cargo test -p ripvec-core svd -- --nocapture
```

Expected: All 3 tests pass.

- [ ] **Step 5: Add optional SVD factor fields to `ModernBertLayerWeights`**

In `crates/ripvec-core/src/backend/arch/modern_bert.rs`, add to `ModernBertLayerWeights`:

```rust
pub struct ModernBertLayerWeights<T> {
    pub qkv_weight: T,
    pub output_weight: T,
    pub attn_norm_weight: Option<T>,
    pub mlp_wi_weight: T,
    pub mlp_wo_weight: T,
    pub mlp_norm_weight: T,
    pub is_global: bool,
    /// Low-rank SVD factor A for Wi: `[k, hidden]`. Present when `--svd-rank` is set.
    pub svd_wi_a: Option<T>,
    /// Low-rank SVD factor B for Wi: `[2*intermediate, k]`. Present when `--svd-rank` is set.
    pub svd_wi_b: Option<T>,
    /// SVD rank (k) for this layer. 0 if SVD not applied.
    pub svd_rank: usize,
}
```

Update all construction sites to include `svd_wi_a: None, svd_wi_b: None, svd_rank: 0`.

- [ ] **Step 6: SVD at Metal load time**

In `crates/ripvec-core/src/backend/driver/metal.rs`, in `load_modern_bert_weights()`, after the layer weight construction loop but before FP16 pre-conversion (~line 1234):

```rust
        // SVD decomposition of Wi weights (if requested).
        if !matches!(svd_rank, crate::embed::SvdRank::Disabled) {
            let intermediate = config.intermediate_size;
            let wi_rows = 2 * intermediate;
            let wi_cols = hidden;

            for (li, layer) in weights.layers.iter_mut().enumerate() {
                // Read Wi data from mmap (FP32)
                let wi_offset = layer.mlp_wi_weight.offset;
                let wi_bytes = wi_rows * wi_cols * 4;
                let wi_slice = &mmap[wi_offset..wi_offset + wi_bytes];
                let wi_data: Vec<f32> = wi_slice
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();

                if let Some(factors) = crate::backend::svd::decompose_wi(
                    &wi_data, wi_rows, wi_cols, svd_rank,
                ) {
                    tracing::info!(
                        layer = li,
                        rank = factors.k,
                        error = format!("{:.4}%", factors.error * 100.0),
                        "SVD Wi decomposition"
                    );

                    // Create Metal buffers from factor data
                    let a_buf = self.create_buffer_from_f32(&factors.a)?;
                    let b_buf = self.create_buffer_from_f32(&factors.b)?;

                    layer.svd_wi_a = Some(a_buf);
                    layer.svd_wi_b = Some(b_buf);
                    layer.svd_rank = factors.k;
                }
            }
        }
```

Add a helper method to `MetalDriver`:

```rust
    /// Create a Metal buffer from FP32 data (for SVD factors and similar computed weights).
    fn create_buffer_from_f32(&self, data: &[f32]) -> crate::Result<MetalTensor> {
        let size = (data.len() * 4) as objc2_metal::NSUInteger;
        let buffer = unsafe {
            self.device.newBufferWithBytes_length_options(
                data.as_ptr() as *const core::ffi::c_void,
                size,
                objc2_metal::MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or_else(|| crate::Error::Metal("failed to create buffer from f32 data".into()))?;
        Ok(MetalTensor::new(buffer, 0))
    }
```

Then update the FP16 pre-conversion loop to also convert SVD factors:

```rust
        // After existing ensure_fp16 calls for each layer:
        if layer.svd_wi_a.is_some() {
            let a = layer.svd_wi_a.as_ref().unwrap();
            let a_elems = layer.svd_rank * hidden;
            self.ensure_fp16(a, a_elems)?;

            let b = layer.svd_wi_b.as_ref().unwrap();
            let b_elems = 2 * intermediate * layer.svd_rank;
            self.ensure_fp16(b, b_elems)?;
        }
```

The function signature must accept `svd_rank`:

```rust
    pub fn load_modern_bert_weights(
        &self,
        weights_path: &Path,
        config: &ModernBertConfig,
        svd_rank: &crate::embed::SvdRank,
    ) -> crate::Result<(ModernBertArch<MetalTensor>, memmap2::Mmap)> {
```

Update the caller in `load_modernbert_metal()` in `backend/mod.rs`.

- [ ] **Step 7: SVD at CPU load time**

In `crates/ripvec-core/src/backend/driver/cpu.rs`, in `load_modern_bert_weights()`, after layer construction:

```rust
        // SVD decomposition of Wi weights (if requested).
        if !matches!(svd_rank, crate::embed::SvdRank::Disabled) {
            let wi_rows = 2 * config.intermediate_size;
            let wi_cols = config.hidden_size;

            for (li, layer) in layers.iter_mut().enumerate() {
                if let Some(factors) = crate::backend::svd::decompose_wi(
                    &layer.mlp_wi_weight, wi_rows, wi_cols, svd_rank,
                ) {
                    tracing::info!(
                        layer = li,
                        rank = factors.k,
                        error = format!("{:.4}%", factors.error * 100.0),
                        "SVD Wi decomposition"
                    );
                    layer.svd_wi_a = Some(factors.a);
                    layer.svd_wi_b = Some(factors.b);
                    layer.svd_rank = factors.k;
                }
            }
        }
```

Update the CPU function signature to also accept `svd_rank`.

- [ ] **Step 8: Split Wi GEMM in `ffn_sublayer` when SVD factors are present**

In `crates/ripvec-core/src/backend/arch/modern_bert.rs`, replace the Wi GEMM block in `ffn_sublayer` (lines 377-388):

```rust
    // Wi projection: either one full GEMM or two low-rank GEMMs.
    let double_inter = 2 * g.intermediate;
    let mut wi_out = driver.alloc_zeros(g.total_tokens * double_inter)?;

    if let (Some(ref a), Some(ref b)) = (&layer.svd_wi_a, &layer.svd_wi_b) {
        // Low-rank: Z = X @ A^T  [M, hidden] @ [k, hidden]^T → [M, k]
        let k = layer.svd_rank;
        let mut z = driver.alloc_zeros(g.total_tokens * k)?;
        driver.gemm(&mlp_normed, a, &mut z, g.total_tokens, k, g.hidden, true)?;

        // Y = Z @ B^T  [M, k] @ [2*inter, k]^T → [M, 2*inter]
        driver.gemm(&z, b, &mut wi_out, g.total_tokens, double_inter, k, true)?;
    } else {
        // Original full-rank GEMM
        driver.gemm(
            &mlp_normed, &layer.mlp_wi_weight, &mut wi_out,
            g.total_tokens, double_inter, g.hidden, true,
        )?;
    }
```

- [ ] **Step 9: Split Wi GEMM in `ffn_sublayer_f16`**

Same pattern in `ffn_sublayer_f16` (lines 676-687), using `gemm_f16` and `alloc_zeros_f16`:

```rust
    let double_inter = 2 * g.intermediate;
    let mut wi_out = driver.alloc_zeros_f16(g.total_tokens * double_inter)?;

    if let (Some(ref a), Some(ref b)) = (&layer.svd_wi_a, &layer.svd_wi_b) {
        let k = layer.svd_rank;
        let mut z = driver.alloc_zeros_f16(g.total_tokens * k)?;
        driver.gemm_f16(&mlp_normed, a, &mut z, g.total_tokens, k, g.hidden, true)?;
        driver.gemm_f16(&z, b, &mut wi_out, g.total_tokens, double_inter, k, true)?;
    } else {
        driver.gemm_f16(
            &mlp_normed, &layer.mlp_wi_weight, &mut wi_out,
            g.total_tokens, double_inter, g.hidden, true,
        )?;
    }
```

- [ ] **Step 10: Build and verify correctness**

```bash
cargo check --workspace && cargo test --workspace
```

Then validate search results still match:

```bash
uv run scripts/bench/bench.py --configs mps cpu --layers 22 --validate --no-build
```

Expected: Rankings match (SVD is disabled by default).

- [ ] **Step 11: Test SVD-enabled correctness**

Run with SVD auto to verify it doesn't crash and produces reasonable results:

```bash
./target/release/ripvec "error handling" tests/corpus/code/flask/ --svd-rank auto --layers 22 -n 3
```

Expected: Returns results (may differ slightly from baseline — that's expected).

- [ ] **Step 12: Commit**

```bash
git add -A
git commit -m "feat: low-rank FFN via SVD at load time (--svd-rank)

Decomposes each layer's Wi [2304,768] into two factors A [k,768] and
B [2304,k] using truncated SVD. Replaces one large GEMM with two
smaller ones through a rank-k bottleneck.

--svd-rank auto: per-layer k from 1% Frobenius threshold
--svd-rank N: fixed rank for all layers
--svd-rank 0: disabled (default)

At k=384: 33% Wi FLOP reduction, ~9% total throughput gain."
```

---

### Task 4: Phase A Benchmark + Tracemeld

**Files:** None modified — measurement only.

- [ ] **Step 1: Benchmark `fast::exp()` alone**

```bash
uv run scripts/bench/bench.py --configs mps --layers 22 --no-build
```

Record throughput. Expected: ~74-76/s (small gain over 73.6/s baseline).

- [ ] **Step 2: Benchmark `fast::exp()` + SVD auto**

```bash
RIPVEC_SVD_RANK=auto uv run scripts/bench/bench.py --configs mps --layers 22 --no-build
```

Wait — CLI flags aren't env vars. We need to pass `--svd-rank auto` to ripvec. bench.py doesn't support this yet. Run manually:

```bash
./target/release/ripvec session tests/corpus/code/flask/ --svd-rank auto --layers 0 -n 1 --profile
```

Record throughput. Expected: ~80-85/s.

- [ ] **Step 3: Tracemeld capture + diff**

```bash
xctrace record --template 'Metal System Trace' \
  --output /tmp/phase-a-svd-auto.trace \
  --launch -- ./target/release/ripvec "error handling middleware" tests/corpus/code/flask/ \
  --layers 0 --batch-size 32 --mode semantic --svd-rank auto
```

Import and diff:
```
import_profile(source: "/tmp/phase-a-svd-auto.trace", format: "xctrace")
diff_profile(baseline: "mps-fp16-22L-baseline")
save_baseline(name: "phase-a-svd-auto", checkpoint: "after", task: "fast::exp + SVD auto")
```

- [ ] **Step 4: Validate recall (manual)**

Run baseline and SVD-auto, compare top-10 results for 3 queries:

```bash
# Baseline
./target/release/ripvec "error handling" tests/corpus/code/flask/ -n 10 --format json > /tmp/baseline.json
./target/release/ripvec "database connection" tests/corpus/code/flask/ -n 10 --format json >> /tmp/baseline.json
./target/release/ripvec "request routing" tests/corpus/code/flask/ -n 10 --format json >> /tmp/baseline.json

# SVD auto
./target/release/ripvec "error handling" tests/corpus/code/flask/ -n 10 --format json --svd-rank auto > /tmp/svd.json
./target/release/ripvec "database connection" tests/corpus/code/flask/ -n 10 --format json --svd-rank auto >> /tmp/svd.json
./target/release/ripvec "request routing" tests/corpus/code/flask/ -n 10 --format json --svd-rank auto >> /tmp/svd.json
```

Compare overlap manually or with a script. Target: ≥90% overlap (≤10% recall loss).

---

### Task 5: Layer Skipping

**Files:**
- Modify: `crates/ripvec-core/src/backend/arch/modern_bert.rs`

- [ ] **Step 1: Add skip logic to FP16 layer loop**

In `forward()`, change the FP16 layer loop (lines 809-824) to use `.enumerate()` and check skip set:

```rust
        for (li, layer) in w.layers[..num_layers].iter().enumerate() {
            if self.skip_layers.contains(&li) {
                continue;
            }
            let saved = driver.save_pool_cursor();

            let rope = if layer.is_global {
                &self.global_rope
            } else {
                &self.local_rope
            };

            let (q, k, v) =
                attn_prenorm_qkv_f16(driver, &hidden_f16, layer, &g, &w.zero_bias, rope)?;
            let attn_output =
                attn_scores_residual_f16(driver, &q, &k, &v, &hidden_f16, layer, &inputs, &g)?;
            hidden_f16 = ffn_sublayer_f16(driver, &attn_output, layer, &g, &w.zero_bias)?;
            driver.restore_pool_cursor(saved);
        }
```

- [ ] **Step 2: Add skip logic to FP32 layer loop**

The FP32 loop (lines 832-853) already has `enumerate`. Add the skip check:

```rust
        for (li, layer) in w.layers[..num_layers].iter().enumerate() {
            if self.skip_layers.contains(&li) {
                continue;
            }
            let saved = driver.save_pool_cursor();
            // ... rest unchanged ...
        }
```

Remove the `let _ = li;` line since `li` is now used.

- [ ] **Step 3: Wire `skip_layers` from CLI to arch**

In `crates/ripvec-core/src/backend/mod.rs`, in `load_modernbert_metal()` after `arch.max_layers = max_layers;`:

The skip_layers come from SearchConfig, but the arch is created during load_backend which happens before SearchConfig is fully wired. We need to pass skip_layers into the load function.

Update `load_modernbert_metal` and `load_modernbert_cpu` to accept `skip_layers` and `prune_ratio`:

```rust
pub fn load_modernbert_metal(
    model_repo: &str,
    max_layers: Option<usize>,
    svd_rank: &crate::embed::SvdRank,
    skip_layers: std::collections::HashSet<usize>,
    prune_ratio: f32,
) -> crate::Result<Box<dyn EmbedBackend>> {
    // ...
    arch.max_layers = max_layers;
    arch.skip_layers = skip_layers;
    arch.prune_ratio = prune_ratio;
    // ...
}
```

Update callers in `detect_backends()` and `load_backend()` to pass these through from the function parameters. Update the `detect_backends()` and `load_backend()` function signatures to accept them as well.

In `main.rs`, pass the parsed values when calling `detect_backends()` or `load_backend()`.

- [ ] **Step 4: Build and verify**

```bash
cargo check --workspace
```

Test with skip:

```bash
./target/release/ripvec "error handling" tests/corpus/code/flask/ --skip-layers 6,7,13,14 -n 3
```

Expected: Returns results (may differ from baseline).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: layer skipping via --skip-layers flag

Skips specified encoder layers entirely. Layer N-1 output feeds
directly to layer N+1. Pre-norm + residual connections dampen the
distribution shift.

Example: --skip-layers 6,7,13,14 (middle local-attention layers)"
```

---

### Task 6: Token Pruning — Metal Kernels

**Files:**
- Modify: `crates/ripvec-core/src/backend/metal_kernels.rs`
- Modify: `crates/ripvec-core/src/backend/driver/metal.rs` (KernelPipelines + dispatch)
- Modify: `crates/ripvec-core/src/backend/driver/mod.rs` (Driver trait)

- [ ] **Step 1: Add `compute_token_distances` kernel to MSL**

In `crates/ripvec-core/src/backend/metal_kernels.rs`, add in the `KERNELS` string (near the other element-wise kernels):

```metal
/// Compute squared L2 distance of each token from the batch mean.
/// Input: hidden_states [total_tokens, hidden] (FP32)
/// Output: distances [total_tokens] (FP32)
kernel void compute_token_distances_kernel(
    const device float* hidden      [[buffer(0)]],
    device float* distances          [[buffer(1)]],
    constant int& total_tokens       [[buffer(2)]],
    constant int& hidden_dim         [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= uint(total_tokens)) return;

    // Compute mean on-the-fly (each thread computes full mean — simple, correct)
    // For production: precompute mean in a separate reduction kernel.
    // With total_tokens ~2000 and hidden=768, this is ~1.5M FLOPs per token.
    // At 50% pruning that's ~1.5B FLOPs total — small vs GEMM cost.
    float dist_sq = 0.0;
    for (int d = 0; d < hidden_dim; d++) {
        // Compute mean for dimension d
        float sum = 0.0;
        for (int t = 0; t < total_tokens; t++) {
            sum += hidden[t * hidden_dim + d];
        }
        float mean_d = sum / float(total_tokens);

        float diff = hidden[tid * hidden_dim + d] - mean_d;
        dist_sq += diff * diff;
    }
    distances[tid] = dist_sq;
}
```

- [ ] **Step 2: Add FP16 version**

```metal
/// FP16 version: reads FP16 hidden states, outputs FP32 distances.
kernel void compute_token_distances_f16_kernel(
    const device half* hidden        [[buffer(0)]],
    device float* distances          [[buffer(1)]],
    constant int& total_tokens       [[buffer(2)]],
    constant int& hidden_dim         [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= uint(total_tokens)) return;

    float dist_sq = 0.0;
    for (int d = 0; d < hidden_dim; d++) {
        float sum = 0.0;
        for (int t = 0; t < total_tokens; t++) {
            sum += float(hidden[t * hidden_dim + d]);
        }
        float mean_d = sum / float(total_tokens);

        float diff = float(hidden[tid * hidden_dim + d]) - mean_d;
        dist_sq += diff * diff;
    }
    distances[tid] = dist_sq;
}
```

- [ ] **Step 3: Add `gather_tokens` kernels**

```metal
/// Compact hidden states by gathering selected token indices.
/// Input: src [old_total, hidden], indices [new_total]
/// Output: dst [new_total, hidden]
kernel void gather_tokens_kernel(
    const device float* src          [[buffer(0)]],
    device float* dst                [[buffer(1)]],
    const device int* indices        [[buffer(2)]],
    constant int& new_total          [[buffer(3)]],
    constant int& hidden_dim         [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= uint(new_total * hidden_dim)) return;
    int token = int(tid) / hidden_dim;
    int dim = int(tid) % hidden_dim;
    int src_token = indices[token];
    dst[tid] = src[src_token * hidden_dim + dim];
}

/// FP16 version of gather_tokens.
kernel void gather_tokens_f16_kernel(
    const device half* src           [[buffer(0)]],
    device half* dst                 [[buffer(1)]],
    const device int* indices        [[buffer(2)]],
    constant int& new_total          [[buffer(3)]],
    constant int& hidden_dim         [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= uint(new_total * hidden_dim)) return;
    int token = int(tid) / hidden_dim;
    int dim = int(tid) % hidden_dim;
    int src_token = indices[token];
    dst[tid] = src[src_token * hidden_dim + dim];
}
```

- [ ] **Step 4: Add pipeline states to `KernelPipelines`**

In `crates/ripvec-core/src/backend/driver/metal.rs`, add to the `KernelPipelines` struct:

```rust
    compute_token_distances: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    compute_token_distances_f16: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    gather_tokens: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    gather_tokens_f16: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
```

Add corresponding `make_pipeline!` calls where the other kernels are compiled (in the `KernelPipelines::new` or equivalent initialization).

- [ ] **Step 5: Add Driver trait methods**

In `crates/ripvec-core/src/backend/driver/mod.rs`, add:

```rust
    /// Compute squared L2 distance of each token from the batch mean.
    fn compute_token_distances(
        &self,
        _hidden: &Self::Tensor,
        _output: &mut Self::Tensor,
        _total_tokens: usize,
        _hidden_dim: usize,
    ) -> crate::Result<()> {
        Err(crate::Error::Other(anyhow::anyhow!("compute_token_distances not implemented")))
    }

    /// FP16 version of compute_token_distances.
    fn compute_token_distances_f16(
        &self,
        _hidden: &Self::Tensor,
        _output: &mut Self::Tensor,
        _total_tokens: usize,
        _hidden_dim: usize,
    ) -> crate::Result<()> {
        Err(crate::Error::Other(anyhow::anyhow!("compute_token_distances_f16 not implemented")))
    }

    /// Compact hidden states by gathering selected token indices.
    fn gather_tokens(
        &self,
        _src: &Self::Tensor,
        _dst: &mut Self::Tensor,
        _indices: &Self::Tensor,
        _new_total: usize,
        _hidden_dim: usize,
    ) -> crate::Result<()> {
        Err(crate::Error::Other(anyhow::anyhow!("gather_tokens not implemented")))
    }

    /// FP16 version of gather_tokens.
    fn gather_tokens_f16(
        &self,
        _src: &Self::Tensor,
        _dst: &mut Self::Tensor,
        _indices: &Self::Tensor,
        _new_total: usize,
        _hidden_dim: usize,
    ) -> crate::Result<()> {
        Err(crate::Error::Other(anyhow::anyhow!("gather_tokens_f16 not implemented")))
    }
```

- [ ] **Step 6: Implement MetalDriver dispatch methods**

In `crates/ripvec-core/src/backend/driver/metal.rs`, add dispatch implementations:

```rust
    fn compute_token_distances(
        &self,
        hidden: &MetalTensor,
        output: &mut MetalTensor,
        total_tokens: usize,
        hidden_dim: usize,
    ) -> crate::Result<()> {
        self.run_compute("token-distances", |enc| {
            enc.setComputePipelineState(&self.kernels.compute_token_distances);
            set_buffer(enc, &hidden.buffer, hidden.offset, 0);
            set_buffer(enc, &output.buffer, output.offset, 1);
            set_i32_param(enc, total_tokens as i32, 2);
            set_i32_param(enc, hidden_dim as i32, 3);
            dispatch_1d(enc, &self.kernels.compute_token_distances, total_tokens);
            Ok(())
        })
    }

    fn gather_tokens(
        &self,
        src: &MetalTensor,
        dst: &mut MetalTensor,
        indices: &MetalTensor,
        new_total: usize,
        hidden_dim: usize,
    ) -> crate::Result<()> {
        self.run_compute("gather-tokens", |enc| {
            enc.setComputePipelineState(&self.kernels.gather_tokens);
            set_buffer(enc, &src.buffer, src.offset, 0);
            set_buffer(enc, &dst.buffer, dst.offset, 1);
            set_buffer(enc, &indices.buffer, indices.offset, 2);
            set_i32_param(enc, new_total as i32, 3);
            set_i32_param(enc, hidden_dim as i32, 4);
            dispatch_1d(enc, &self.kernels.gather_tokens, new_total * hidden_dim);
            Ok(())
        })
    }
```

Add FP16 versions using `compute_token_distances_f16` and `gather_tokens_f16` pipeline states, reading from `.fp16` fields or main buffer as appropriate for FP16 tensors.

- [ ] **Step 7: Build and verify compilation**

```bash
cargo check --workspace
```

Expected: Compiles. Kernels are not yet called from forward().

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "feat(metal): token pruning kernels — distance computation + gather

compute_token_distances: L2 distance from mean for each token
gather_tokens: compact hidden states using selected indices
Both FP32 and FP16 variants. Not yet wired into forward pass."
```

---

### Task 7: Token Pruning in Forward Pass

**Files:**
- Modify: `crates/ripvec-core/src/backend/arch/modern_bert.rs`
- Modify: `crates/ripvec-core/src/backend/driver/metal.rs` (buffer upload helper)

- [ ] **Step 1: Add index upload helper to MetalDriver**

In `crates/ripvec-core/src/backend/driver/metal.rs`, add:

```rust
    /// Create a Metal buffer from i32 data (for token indices).
    pub fn create_buffer_from_i32(&self, data: &[i32]) -> crate::Result<MetalTensor> {
        let size = (data.len() * 4) as objc2_metal::NSUInteger;
        let buffer = unsafe {
            self.device.newBufferWithBytes_length_options(
                data.as_ptr() as *const core::ffi::c_void,
                size,
                objc2_metal::MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or_else(|| crate::Error::Metal("failed to create buffer from i32 data".into()))?;
        Ok(MetalTensor::new(buffer, 0))
    }
```

Also add a `to_host_f32` method to read back distances:

```rust
    /// Read FP32 data from a Metal buffer back to CPU.
    pub fn read_f32(&self, tensor: &MetalTensor, n: usize) -> crate::Result<Vec<f32>> {
        let ptr = tensor.buffer.contents() as *const f32;
        let offset_elems = tensor.offset / 4;
        let slice = unsafe { std::slice::from_raw_parts(ptr.add(offset_elems), n) };
        Ok(slice.to_vec())
    }
```

- [ ] **Step 2: Add `prune_tokens` helper function**

In `crates/ripvec-core/src/backend/arch/modern_bert.rs`, add a helper:

```rust
/// Prune tokens at layer 11 boundary. Returns new hidden states and updated geometry.
///
/// Computes L2 distance from mean for each token, drops the bottom `ratio` fraction,
/// and compacts the hidden states and sequence metadata.
fn prune_tokens<D: Driver>(
    driver: &D,
    hidden: &D::Tensor,
    g: &EncoderGeometry,
    ratio: f32,
    use_f16: bool,
) -> crate::Result<(D::Tensor, EncoderGeometry)> {
    let total = g.total_tokens;
    let hidden_dim = g.hidden;

    // 1. Compute per-token distances from mean
    let mut distances = driver.alloc_zeros(total)?;
    if use_f16 {
        driver.compute_token_distances_f16(hidden, &mut distances, total, hidden_dim)?;
    } else {
        driver.compute_token_distances(hidden, &mut distances, total, hidden_dim)?;
    }

    // 2. Flush GPU work and read distances to CPU
    driver.flush_batch()?;
    let dist_vec = driver.to_host_flat_f32(&distances, total)?;

    // 3. Build per-sequence token indices, sort by distance, keep top (1-ratio)
    let mut cursor = 0usize;
    let mut new_seq_lengths = Vec::with_capacity(g.seq_lengths.len());
    let mut selected_indices = Vec::new();

    for &seq_len in &g.seq_lengths {
        let seq_start = cursor;
        let seq_end = cursor + seq_len;

        // Pair each token with its distance
        let mut token_dists: Vec<(usize, f32)> = (seq_start..seq_end)
            .map(|i| (i, dist_vec[i]))
            .collect();

        // Sort descending by distance (keep tokens farthest from mean)
        token_dists.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Keep top (1 - ratio) tokens, minimum 1
        let keep = ((seq_len as f32 * (1.0 - ratio)).ceil() as usize).max(1);
        let mut kept: Vec<usize> = token_dists[..keep].iter().map(|(idx, _)| *idx).collect();
        // Sort by original position to preserve order
        kept.sort_unstable();

        new_seq_lengths.push(keep);
        selected_indices.extend(kept);
        cursor = seq_end;
    }

    let new_total = selected_indices.len();

    // 4. Upload indices to GPU and gather
    let indices_i32: Vec<i32> = selected_indices.iter().map(|&i| i as i32).collect();
    let indices_tensor = driver.create_index_tensor(&indices_i32)?;

    let mut new_hidden = if use_f16 {
        let mut h = driver.alloc_zeros_f16(new_total * hidden_dim)?;
        driver.gather_tokens_f16(hidden, &mut h, &indices_tensor, new_total, hidden_dim)?;
        h
    } else {
        let mut h = driver.alloc_zeros(new_total * hidden_dim)?;
        driver.gather_tokens(hidden, &mut h, &indices_tensor, new_total, hidden_dim)?;
        h
    };

    // 5. Build new geometry
    let new_max_seq = *new_seq_lengths.iter().max().unwrap_or(&1);
    let new_g = EncoderGeometry {
        batch: g.batch,
        max_seq: new_max_seq,
        total_tokens: new_total,
        padded_tokens: g.batch * new_max_seq,
        seq_lengths: new_seq_lengths,
        hidden: g.hidden,
        num_heads: g.num_heads,
        head_dim: g.head_dim,
        intermediate: g.intermediate,
        local_window: g.local_window,
        scale: g.scale,
        eps: g.eps,
    };

    Ok((new_hidden, new_g))
}
```

This function requires two additions to the Driver trait:

```rust
    /// Read FP32 data from a tensor back to host.
    fn to_host_flat_f32(&self, tensor: &Self::Tensor, n: usize) -> crate::Result<Vec<f32>>;

    /// Create a tensor from i32 index data.
    fn create_index_tensor(&self, data: &[i32]) -> crate::Result<Self::Tensor>;
```

Add default implementations that return errors, implement on MetalDriver using the helpers from Step 1, and implement on CpuDriver (trivial — just wrap the Vec).

- [ ] **Step 3: Add pruning to the FP16 layer loop**

In `forward()`, insert pruning between layer 11 and layer 12:

```rust
        // FP16 layer loop with pruning at layer 11
        let mut g = g;  // Make geometry mutable for pruning
        for (li, layer) in w.layers[..num_layers].iter().enumerate() {
            if self.skip_layers.contains(&li) {
                continue;
            }
            let saved = driver.save_pool_cursor();

            let rope = if layer.is_global {
                &self.global_rope
            } else {
                &self.local_rope
            };

            let (q, k, v) =
                attn_prenorm_qkv_f16(driver, &hidden_f16, layer, &g, &w.zero_bias, rope)?;
            let attn_output =
                attn_scores_residual_f16(driver, &q, &k, &v, &hidden_f16, layer, &inputs, &g)?;
            hidden_f16 = ffn_sublayer_f16(driver, &attn_output, layer, &g, &w.zero_bias)?;
            driver.restore_pool_cursor(saved);

            // Token pruning at layer 11 boundary
            if li == 10 && self.prune_ratio > 0.0 {
                let (pruned, new_g) =
                    prune_tokens(driver, &hidden_f16, &g, self.prune_ratio, true)?;
                hidden_f16 = pruned;
                g = new_g;
                // Rebuild inputs for new geometry
                // (pad/unpad in subsequent layers will use new seq_lengths)
            }
        }
```

Note: `li == 10` because layers are 0-indexed, so after processing layer index 10 (the 11th layer), we prune before layer 11 (index 11) starts.

Apply the same pattern to the FP32 loop.

- [ ] **Step 4: Update `inputs` for pruned attention**

The `inputs` struct (`BatchInputs`) contains `pooling_mask` and other attention masks. After pruning, attention in layers 12-21 needs masks rebuilt for the new sequence lengths. Update the pruning code to rebuild `inputs`:

```rust
            if li == 10 && self.prune_ratio > 0.0 {
                let (pruned, new_g) =
                    prune_tokens(driver, &hidden_f16, &g, self.prune_ratio, true)?;
                hidden_f16 = pruned;
                g = new_g;
                // Rebuild batch inputs for pruned geometry
                inputs = driver.rebuild_batch_inputs(&g.seq_lengths, g.max_seq)?;
            }
```

This requires a `rebuild_batch_inputs` method on the Driver trait that constructs new masks from the pruned sequence lengths.

- [ ] **Step 5: Build and test**

```bash
cargo check --workspace
```

Test pruning:

```bash
./target/release/ripvec "error handling" tests/corpus/code/flask/ --prune-ratio 0.3 -n 3
```

Expected: Returns results. Some quality loss is expected.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: token pruning at layer 11 (--prune-ratio)

After layer 11, computes each token's L2 distance from the running
mean. Drops the bottom N% (closest to mean = least informative).
Layers 12-21 process fewer tokens.

--prune-ratio 0.5: drop 50% of tokens at layer 11
--prune-ratio 0.0: disabled (default)"
```

---

### Task 8: Phase B Benchmark + Quality Evaluation

**Files:** None modified — measurement only.

- [ ] **Step 1: Layer skip throughput sweep**

```bash
# Skip 2 layers (least aggressive)
./target/release/ripvec session tests/corpus/code/flask/ --skip-layers 6,7 -n 1 --profile

# Skip 4 layers
./target/release/ripvec session tests/corpus/code/flask/ --skip-layers 6,7,13,14 -n 1 --profile

# Skip 6 layers (most aggressive)
./target/release/ripvec session tests/corpus/code/flask/ --skip-layers 5,6,7,13,14,15 -n 1 --profile
```

Record throughput for each.

- [ ] **Step 2: Token pruning throughput sweep**

```bash
for ratio in 0.1 0.2 0.3 0.4 0.5; do
  echo "=== prune-ratio $ratio ==="
  ./target/release/ripvec session tests/corpus/code/flask/ --prune-ratio $ratio -n 1 --profile
done
```

Record throughput for each ratio.

- [ ] **Step 3: Quality evaluation — ranking comparison**

For each optimization config, run against 5 test queries and compare top-10 against baseline:

```bash
QUERIES=("error handling" "database connection" "request routing" "session management" "template rendering")

# Baseline
for q in "${QUERIES[@]}"; do
  ./target/release/ripvec "$q" tests/corpus/code/flask/ -n 10 --format json --mode semantic
done > /tmp/baseline_recall.jsonl

# Each config
for q in "${QUERIES[@]}"; do
  ./target/release/ripvec "$q" tests/corpus/code/flask/ -n 10 --format json --mode semantic --skip-layers 6,7,13,14
done > /tmp/skip4_recall.jsonl

# Compute Recall@10 = intersection(baseline_top10, config_top10) / 10
```

Parse JSON and count file path overlap. Target: semantic ≥90%, hybrid ≥98%.

- [ ] **Step 4: Combined optimization sweep**

Run the best combination:

```bash
./target/release/ripvec session tests/corpus/code/flask/ \
  --svd-rank auto --prune-ratio 0.3 --skip-layers 6,7,13,14 \
  -n 1 --profile
```

Expected: ≥100/s. If not, adjust parameters:
- Lower prune ratio (0.3 → 0.2)
- Skip fewer layers (4 → 2)
- Increase SVD rank (auto → 512)

- [ ] **Step 5: Tracemeld final diff**

```bash
xctrace record --template 'Metal System Trace' \
  --output /tmp/combined-optimized.trace \
  --launch -- ./target/release/ripvec "error handling middleware" tests/corpus/code/flask/ \
  --layers 0 --batch-size 32 --mode semantic \
  --svd-rank auto --prune-ratio 0.3 --skip-layers 6,7,13,14
```

```
import_profile(source: "/tmp/combined-optimized.trace", format: "xctrace")
diff_profile(baseline: "mps-fp16-22L-baseline")
save_baseline(name: "combined-optimized", checkpoint: "after", task: "All optimizations: fast::exp + SVD auto + prune 0.3 + skip 4")
```

- [ ] **Step 6: Final bench.py run**

```bash
uv run scripts/bench/bench.py --configs mps compute cpu --layers 22 --no-build
```

Compare against baseline results in `scripts/bench/results/20260329T220434.json`.

- [ ] **Step 7: Commit results**

```bash
git add scripts/bench/results/
git commit -m "bench: inference optimization results — Phase A + B

Baseline: MPS 73.6/s, compute 59.0/s, CPU 72.3/s
Optimized: [fill in actual numbers]"
```

---

## Success Criteria

| Metric | Target | How to Verify |
|--------|--------|---------------|
| Throughput (MPS, 22L, Flask) | ≥100/s | bench.py |
| Semantic Recall@10 | ≥90% | Task 8 quality evaluation |
| Hybrid Recall@10 | ≥98% | Task 8 quality evaluation |
| Model load time | <3s (including SVD) | `--profile` output |
| No MPS regression | Baseline ≥73/s without flags | bench.py |
| tracemeld diff | Positive wall_ms reduction | diff_profile |
