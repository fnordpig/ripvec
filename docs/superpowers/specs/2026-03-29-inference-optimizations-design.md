# ModernBERT Inference Optimizations Design

**Date:** 2026-03-29
**Status:** Approved

**Outcome (2026-03-30):** Implemented and tested all four optimizations.
Only `fast::exp()` survived — SVD, token pruning, and layer skipping were
all removed after failing quality or throughput validation:
- **fast::exp()**: Shipped (always on). +0.3% throughput, zero recall loss.
- **SVD**: Removed. 4.5s load penalty exceeded FLOP savings (-8% net).
- **Token pruning**: Removed. Even 10% pruning caused 11% semantic recall loss.
- **Layer skipping**: Removed. Per-layer ablation showed no safe layers to skip.
- **--layers early exit**: Also removed. All 22 layers required for quality.

**Goal:** Maximize ModernBERT embedding throughput while staying within recall budgets: ≤10% semantic Recall@10 loss, ≤2% hybrid Recall@10 loss.

---

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Optimization gating | Runtime flags per optimization | Maximum flexibility for benchmarking; rationalize UX later |
| fast::exp() | Always-on (no flag) | Zero quality risk, 1 line per kernel |
| Low-rank FFN | `--svd-rank <k>` flag | Per-layer rank from Frobenius norm threshold |
| Token pruning | `--prune-ratio <0.0-1.0>` flag | Continuous knob for quality/speed tradeoff |
| Layer skipping | `--skip-layers <list>` flag | Comma-separated layer indices |
| Head pruning | Deferred | High effort (QKV reshape), low payoff (8-15%) |
| Quality budget | 10% semantic, 2% hybrid | Hybrid BM25 compensates for semantic degradation |

---

## Phase A: Zero Risk (Always-On)

### 1. fast::exp() in Softmax Kernels

Replace `exp()` with `fast::exp()` in all softmax MSL kernels. Metal's `fast::exp()` uses a reduced-precision hardware path (~11-bit mantissa vs 23-bit) that is faster for the same throughput of transcendental operations.

**Files:** `crates/ripvec-core/src/backend/metal_kernels.rs`

**Kernels to modify (4 total):**
- `fused_scale_mask_softmax_kernel` (FP32)
- `fused_scale_mask_softmax_f16_kernel` (FP16)
- `fused_scale_mask_softmax_windowed_kernel` (FP32 local attention)
- `fused_scale_mask_softmax_windowed_f16_kernel` (FP16 local attention)

**Change per kernel:** `exp(val - row_max)` → `fast::exp(val - row_max)`

**Quality impact:** Zero. Softmax ranking is preserved — the reduced precision affects only the magnitude of near-zero attention weights, not which tokens attend to which.

**Expected:** +2-3% throughput.

### 2. Low-Rank FFN via SVD at Load Time

Each ModernBERT layer's FFN has a Wi projection `[2304, 768]` that maps hidden→intermediate. Many of these weight matrices are low-rank — their singular value spectrum drops off sharply. Truncating to rank `k` replaces one large GEMM with two smaller ones through a bottleneck.

**Math:**
```
Original: Y = X @ Wi^T          where Wi is [2304, 768]
          FLOPs: M × 768 × 2304 = M × 1,769,472

SVD:      Wi ≈ U_k @ S_k @ V_k^T    where U_k is [2304, k], V_k is [768, k]
          A = V_k @ diag(sqrt(S_k))  → [768, k]    (absorbed into first GEMM weight)
          B = U_k @ diag(sqrt(S_k))  → [2304, k]   (absorbed into second GEMM weight)

Low-rank: Z = X @ A^T              FLOPs: M × 768 × k
          Y = Z @ B^T              FLOPs: M × k × 2304
          Total: M × k × (768 + 2304) = M × k × 3072

At k=384: M × 384 × 3072 = M × 1,179,648  → 33% FLOP reduction on Wi GEMM
```

**Quality guarantee:** The Frobenius norm ratio `‖Wi - A@B^T‖_F / ‖Wi‖_F` is computed at load time. If it exceeds the threshold (default 1%), increase k for that layer. This is a hard, measurable bound — not a guess.

**Files:**
- `crates/ripvec-core/src/backend/driver/metal.rs` — SVD computation at weight load time
- `crates/ripvec-core/src/backend/arch/modern_bert.rs` — split `ffn_sublayer` Wi GEMM into two
- `crates/ripvec/src/cli.rs` — `--svd-rank` flag

**Runtime flag:** `--svd-rank <k>`
- `0` (default): disabled, use original Wi weight
- `auto`: compute per-layer k from Frobenius threshold (1% reconstruction error)
- Explicit integer: use that rank for all layers

**SVD computation:** Uses `ndarray` (already a dependency) at model load time. ~1-2s for 22 matrices. Produces two weight tensors per layer stored alongside the original.

**Expected:** +9% total throughput (33% Wi FLOP reduction × 55% FFN share × ~50% GEMM share of total).

---

## Phase B: Quality-Benchmarked

### 3. Token Pruning at Layer 11

After layer 11 (halfway through the 22-layer stack), compute each token's L2 distance from the running mean embedding. Tokens closest to the mean contribute least to the final mean-pooled output — they pull the mean toward where it already is. Drop the bottom N% and process layers 12-21 with fewer tokens.

**Why layer 11:** Early layers (0-10) build syntactic and lexical representations — pruning here loses structural information. Late layers (12-21) refine semantic representations — the tokens that will matter have already differentiated from the mean by layer 11. The 11/22 split is a heuristic; bench.py `--recall` validates it empirically.

**Why this works for code:** Code has many semantically empty tokens (`{`, `}`, `;`, `(`, `)`, whitespace) that contextualize to near-mean embeddings after 11 layers. Natural language has fewer of these, so the pruning ratio should be lower for text corpora.

**Implementation:**

1. **Distance kernel** (`compute_token_distances`): For each token in `[total_tokens, hidden]`, compute `‖token_i - mean‖²` where `mean = sum(tokens) / total_tokens`. Output: `[total_tokens]` float distances.

2. **CPU-side sort + compact:** Read distances back, sort, select top (1-ratio) × total_tokens indices. Rebuild `seq_lengths` (per-sequence count of surviving tokens), `cu_seqlens`, `total_tokens`.

3. **Gather kernel** (`gather_tokens`): Compact `hidden_states` from `[old_total, hidden]` to `[new_total, hidden]` using the selected indices.

4. **Resume forward pass:** Layers 12-21 see smaller `total_tokens`. Pad/unpad for attention uses new `seq_lengths`.

**Runtime flag:** `--prune-ratio <0.0-1.0>` (default 0.0 = disabled)
- 0.5 = drop 50% of tokens at layer 11
- Applied once, not per-layer

**Files:**
- `crates/ripvec-core/src/backend/metal_kernels.rs` — distance + gather kernels
- `crates/ripvec-core/src/backend/driver/metal.rs` — dispatch methods
- `crates/ripvec-core/src/backend/driver/mod.rs` — Driver trait methods
- `crates/ripvec-core/src/backend/arch/modern_bert.rs` — pruning logic at layer 11
- `crates/ripvec/src/cli.rs` — `--prune-ratio` flag

**Expected at 50%:** +36% throughput (10/22 layers at half M × 80% GEMM share).

**Quality risk:** Unknown — must benchmark. Target: ≤10% semantic Recall@10 loss, ≤2% hybrid.

### 4. Layer Skipping

Skip specified encoder layers entirely. Layer N-1's output feeds directly to layer N+1. Pre-norm + residual connections dampen the distribution shift, but without distillation, the quality impact is unpredictable.

**Implementation:** Filter the layer loop:
```rust
for (li, layer) in w.layers[..num_layers].iter().enumerate() {
    if skip_set.contains(&li) { continue; }
    // ... layer compute ...
}
```

Skipped layers' weights are loaded but not used (memory cost but zero compute cost).

**Runtime flag:** `--skip-layers <list>` (default empty)
- Example: `--skip-layers 6,7,13,14`
- Layers 6,7,13,14 are local-attention middle layers — most redundant per distillation literature

**Files:**
- `crates/ripvec-core/src/backend/arch/modern_bert.rs` — skip logic in forward()
- `crates/ripvec/src/cli.rs` — `--skip-layers` flag
- `crates/ripvec-core/src/embed.rs` — pass skip set through SearchConfig

**Expected at 4 layers:** +18% throughput (4/22 layers removed).

**Quality risk:** Unknown without distillation. Target: ≤10% semantic, ≤2% hybrid (combined with other optimizations).

---

## Phase C: Deferred

### 5. Head Pruning

Deferred. Requires offline head importance analysis, invasive QKV weight reshaping (changing GEMM dimensions throughout the Driver trait), and yields only 8-15% for high implementation cost. The same throughput gain is achievable more safely through token pruning.

---

## Evaluation Protocol

### Benchmark Matrix

Each optimization measured independently AND in combination using bench.py:

```bash
# Baseline
uv run scripts/bench/bench.py --configs mps --no-build --layers 22

# fast::exp() (always on after Phase A)
# Rebuild with fast::exp(), benchmark

# SVD
uv run scripts/bench/bench.py --configs mps --no-build --layers 22
# with --svd-rank auto

# Token pruning sweep
for ratio in 0.1 0.2 0.3 0.4 0.5; do
  # --prune-ratio $ratio
done

# Layer skip sweep
# --skip-layers 6,7
# --skip-layers 6,7,13,14
# --skip-layers 5,6,7,13,14,15

# Combined best
# --svd-rank auto --prune-ratio 0.3 --skip-layers 6,7,13,14
```

### Recall Evaluation

Using bench.py `--recall` with 10 queries on Flask corpus:

```
Pass criteria:
  Semantic Recall@10 ≥ 90% (≤10% loss)
  Hybrid Recall@10 ≥ 98% (≤2% loss)
```

If any configuration exceeds the budget, reduce parameters:
- Lower `--prune-ratio` (0.5 → 0.3 → 0.2)
- Skip fewer layers (4 → 2 → 0)
- Increase SVD rank (384 → 512 → 640)

### Success Criteria

| Metric | Target |
|--------|--------|
| Throughput (22L, Flask) | ≥100/s (1.4× baseline) |
| Semantic Recall@10 | ≥90% |
| Hybrid Recall@10 | ≥98% |
| Model load time | <3s (including SVD) |
| No MPS regression | ✓ |

---

## Stacking Order

Optimizations are applied in order during the forward pass:

```
1. fast::exp() — in softmax kernels (always active)
2. Low-rank FFN — in ffn_sublayer (when --svd-rank is set)
3. Layer skip — in layer loop (when --skip-layers is set)
4. Token pruning — at layer 11 boundary (when --prune-ratio > 0)
   → Layers 12-21 benefit from BOTH pruning AND SVD
```

The stacking is multiplicative for throughput:
- SVD reduces per-GEMM FLOPs for ALL layers
- Layer skip removes entire layers (including their SVD-optimized GEMMs)
- Token pruning reduces M for layers 12-21 (including skipped ones — no, skipped layers don't execute)

Corrected stacking estimate:
```
Baseline:                    72/s
+ fast::exp():               ~74/s  (+3%)
+ SVD (k=384):               ~81/s  (+9%)
+ Prune 50% at L11:          ~113/s (+40% on top of SVD)
+ Skip 4 layers:             ~130/s (+15% on remaining layers)
```

Target: ≥100/s. The combined stack should reach this if quality permits.
