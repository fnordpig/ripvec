# ModernBERT Metal Inference Optimization Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Double ModernBERT throughput from 60 → 130 chunks/s on M2 Max by activating the dormant unpadding infrastructure, reordering the output projection, and adding configurable early exit.

**Architecture:** The ModernBERT forward pass currently calls `prepare_batch` (padded) when `prepare_batch_unpadded` is already implemented in MetalDriver. This means `total_tokens == padded_tokens == batch * max_seq` — all linear ops wastefully process padding. Additionally, the output projection operates on `padded_tokens` then unpads, when unpadding first would be cheaper. Combined with a `--layers` flag for configurable early exit, these changes yield ~2× throughput.

**Tech Stack:** Rust 2024, objc2-metal, MPS, Metal compute kernels (MSL)

**Subagent Strategy:** Tasks 1, 2, 4 run in parallel (independent, different files). Task 3 depends on 1+2 merging. Task 5 depends on 3+4.

---

## File Map

| File | Changes |
|------|---------|
| `crates/ripvec-core/src/backend/driver/metal.rs:1982-2044` | Fix `prepare_batch_unpadded` to build proper padded masks |
| `crates/ripvec-core/src/backend/arch/modern_bert.rs:233-346` | Reorder output proj in FP32 attention path |
| `crates/ripvec-core/src/backend/arch/modern_bert.rs:537-645` | Reorder output proj in FP16 attention path |
| `crates/ripvec-core/src/backend/arch/modern_bert.rs:737-880` | Wire `prepare_batch_unpadded` into `forward()` |
| `crates/ripvec/src/cli.rs` | Add `--layers` flag |
| `crates/ripvec/src/main.rs` | Plumb `--layers` to `ModernBertArch.max_layers` |

---

### Task 1: Fix `prepare_batch_unpadded` masks

**Agent:** Sonnet (focused code change, clear spec)
**Parallel:** Yes — independent of Tasks 2-4

**Problem:** `prepare_batch_unpadded` (metal.rs:2026-2029) creates dummy zeroed `float_mask` and `attention_mask`. Attention softmax needs a proper padded `float_mask` `[batch, max_seq]` with `0.0` for real tokens and `-1e9` for padding positions. Mean pooling needs a padded `pooling_mask` with `1.0`/`0.0`.

**Files:**
- Modify: `crates/ripvec-core/src/backend/driver/metal.rs:1982-2044`

- [ ] **Step 1: Read the existing `prepare_batch_unpadded` and `prepare_batch` implementations**

Read metal.rs lines 1908-1980 (`prepare_batch`) to understand how `float_mask` and `pooling_mask` are built in the padded path — this is the reference for what we need to replicate.

- [ ] **Step 2: Replace dummy masks with proper padded masks**

In `prepare_batch_unpadded` (metal.rs:2026-2044), replace the dummy mask construction with proper padded masks built from `seq_lengths`. The token tensors (`input_ids`, `token_type_ids`, `position_ids`) stay flat/unpadded. Only the masks need padded layout.

Replace lines 2026-2043:
```rust
        // Build padded attention mask from seq_lengths: [batch * max_seq]
        // 1 for real tokens, 0 for padding positions.
        let padded_total = batch * max_seq;
        let mut attn_mask_int = vec![0_i32; padded_total];
        for (b, &len) in seq_lengths.iter().enumerate() {
            let offset = b * max_seq;
            for i in 0..len {
                attn_mask_int[offset + i] = 1;
            }
        }
        let attn_mask_int_buf = make_i32_buffer(&self.device, &attn_mask_int)?;

        // Build float attention bias mask on GPU (0.0 for real, -1e9 for pad)
        let float_mask_buf = alloc_f32_buffer(&self.device, padded_total)?;
        self.run_compute("build-attn-mask", |enc| {
            enc.setComputePipelineState(&self.kernels.build_attn_mask);
            set_buffer(enc, &float_mask_buf, 0, 0);
            set_buffer(enc, &attn_mask_int_buf, 0, 1);
            set_i32_param(enc, padded_total as i32, 2);
            dispatch_1d(enc, &self.kernels.build_attn_mask, padded_total);
            Ok(())
        })?;

        // Build padded pooling mask (1.0 for real, 0.0 for pad)
        let pooling_mask_padded: Vec<f32> = attn_mask_int
            .iter()
            .map(|&m| if m == 1 { 1.0 } else { 0.0 })
            .collect();
        let pooling_mask_buf = make_f32_buffer(&self.device, &pooling_mask_padded)?;

        Ok(BatchInputs {
            input_ids: MetalTensor::new(input_ids_buf, 0),
            attention_mask: MetalTensor::new(attn_mask_int_buf, 0),
            token_type_ids: MetalTensor::new(token_type_ids_buf, 0),
            position_ids: MetalTensor::new(position_ids_buf, 0),
            float_mask: MetalTensor::new(float_mask_buf, 0),
            pooling_mask: MetalTensor::new(pooling_mask_buf, 0),
            batch,
            max_seq,
            total_tokens,
            seq_lengths,
            cu_seqlens: Some(cu_seqlens),
        })
```

- [ ] **Step 3: Verify compilation**

Run: `cargo check -p ripvec-core`
Expected: compiles with no new errors (existing warnings OK)

- [ ] **Step 4: Commit**

```bash
git add crates/ripvec-core/src/backend/driver/metal.rs
git commit -m "fix(metal): build proper padded masks in prepare_batch_unpadded"
```

---

### Task 2: Reorder output projection in BOTH attention paths (FP32 + FP16)

**Agent:** Sonnet (focused refactor, clear before/after, both paths in same file)
**Parallel:** Yes — independent of Tasks 1, 5

**Problem:** Both `attn_scores_residual` (modern_bert.rs:315-335) and `attn_scores_residual_f16` (modern_bert.rs:615-644) run the output projection on `padded_tokens` then unpad. Unpadding first makes the GEMM operate on `total_tokens` — up to 5× fewer rows. Both functions are in the same file so they MUST be changed together.

**Files:**
- Modify: `crates/ripvec-core/src/backend/arch/modern_bert.rs:304-345` (FP32)
- Modify: `crates/ripvec-core/src/backend/arch/modern_bert.rs:604-644` (FP16)

- [ ] **Step 1: Read both attention functions**

Read modern_bert.rs lines 233-346 (`attn_scores_residual`) and 537-645 (`attn_scores_residual_f16`) to understand the full flow in both precision paths.

- [ ] **Step 2a: Reorder FP32 path — unpad BEFORE output projection**

Replace lines 304-345 with:
```rust
    // Reshape heads back to [padded_tokens, hidden] (still padded).
    let mut context = driver.alloc_zeros(g.padded_tokens * g.hidden)?;
    driver.attn_reshape(
        &mut context,
        &attn_out,
        g.batch,
        g.max_seq,
        g.num_heads,
        g.head_dim,
    )?;

    // Unpad FIRST: [padded_tokens, H] → [total_tokens, H].
    // Output projection is per-token — unpadding before GEMM is valid and
    // avoids processing batch*max_seq rows when only total_tokens are real.
    let mut context_unpacked = driver.alloc_zeros(g.total_tokens * g.hidden)?;
    driver.unpad_from_batch(
        &context,
        &mut context_unpacked,
        &g.seq_lengths,
        g.max_seq,
        g.hidden,
    )?;

    // Output projection on unpadded layout: [total_tokens, H] × [H, H].
    let mut projected = driver.alloc_zeros(g.total_tokens * g.hidden)?;
    driver.gemm(
        &context_unpacked,
        &layer.output_weight,
        &mut projected,
        g.total_tokens,
        g.hidden,
        g.hidden,
        true,
    )?;

    // Residual add (no bias in ModernBERT). Both are [total_tokens, H].
    let mut output = driver.alloc_zeros(g.total_tokens * g.hidden)?;
    driver.residual_add(
        &mut output,
        &projected,
        hidden_states,
        g.total_tokens * g.hidden,
    )?;
    Ok(output)
```

- [ ] **Step 2b: Reorder FP16 path — unpad BEFORE output projection**

Replace lines 604-644 with:
```rust
    // Reshape heads — FP16.
    let mut context = driver.alloc_zeros_f16(g.padded_tokens * g.hidden)?;
    driver.attn_reshape_f16(
        &mut context,
        &attn_out,
        g.batch,
        g.max_seq,
        g.num_heads,
        g.head_dim,
    )?;

    // Unpad FIRST — FP16: [padded_tokens, H] → [total_tokens, H].
    let mut context_unpacked = driver.alloc_zeros_f16(g.total_tokens * g.hidden)?;
    driver.unpad_from_batch_f16(
        &context,
        &mut context_unpacked,
        &g.seq_lengths,
        g.max_seq,
        g.hidden,
    )?;

    // Output projection on unpadded — FP16: [total_tokens, H] × [H, H].
    let mut projected = driver.alloc_zeros_f16(g.total_tokens * g.hidden)?;
    driver.gemm_f16(
        &context_unpacked,
        &layer.output_weight,
        &mut projected,
        g.total_tokens,
        g.hidden,
        g.hidden,
        true,
    )?;

    // Residual add — FP16.
    let mut output = driver.alloc_zeros_f16(g.total_tokens * g.hidden)?;
    driver.residual_add_f16(
        &mut output,
        &projected,
        hidden_states,
        g.total_tokens * g.hidden,
    )?;
    Ok(output)
```

- [ ] **Step 3: Verify compilation**

Run: `cargo check -p ripvec-core`
Expected: compiles (note: until Task 3 wires `prepare_batch_unpadded`, `total_tokens == padded_tokens` so behavior is unchanged — this is a safe no-op refactor)

- [ ] **Step 4: Commit**

```bash
git add crates/ripvec-core/src/backend/arch/modern_bert.rs
git commit -m "perf(modernbert): unpad before output projection in FP32 + FP16 paths"
```

---

### Task 3: Wire `prepare_batch_unpadded` into ModernBERT `forward()`

**Agent:** Opus (requires understanding interaction of total_tokens/padded_tokens across entire forward pass)
**Parallel:** No — depends on Tasks 1 and 2 being merged first

**Problem:** `forward()` (modern_bert.rs:742-749) calls `prepare_batch` and sets `total_tokens = batch * max_seq`. Must switch to `prepare_batch_unpadded` and use `inputs.total_tokens` (actual sum of sequence lengths) for linear ops while keeping `padded_tokens = batch * max_seq` for attention.

**Files:**
- Modify: `crates/ripvec-core/src/backend/arch/modern_bert.rs:737-878`

- [ ] **Step 1: Read the full `forward()` function**

Read modern_bert.rs lines 728-880 to understand every use of `total_tokens`.

- [ ] **Step 2: Switch to `prepare_batch_unpadded` and fix geometry**

Replace lines 742-773 with:
```rust
        let inputs = driver.prepare_batch_unpadded(encodings)?;
        let max_seq = inputs.max_seq;
        let total_tokens = inputs.total_tokens;

        // Enter batched mode: all GPU ops encode into ONE command buffer.
        driver.begin_batch()?;

        // Embedding (FP32): tok_embeddings + LayerNorm.
        // input_ids is flat [total_tokens] — no padding in linear layers.
        let mut hidden_states =
            driver.embedding_lookup(&inputs.input_ids, &w.tok_embeddings, total_tokens, hidden)?;
        let emb_input = driver.clone_tensor(&hidden_states, total_tokens * hidden)?;
        driver.layer_norm(
            &mut hidden_states,
            &emb_input,
            &w.emb_norm_weight,
            &w.zero_bias,
            total_tokens,
            hidden,
            w.layer_norm_eps,
        )?;

        let g = EncoderGeometry {
            batch,
            max_seq,
            total_tokens,
            padded_tokens: batch * max_seq,
            seq_lengths: inputs.seq_lengths.clone(),
            hidden,
            num_heads: w.num_heads,
            head_dim: w.head_dim,
            intermediate: w.intermediate_dim,
            local_window: w.local_window,
            scale: 1.0 / (w.head_dim as f32).sqrt(),
            eps: w.layer_norm_eps,
        };
```

Key change: `total_tokens = inputs.total_tokens` (sum of actual lengths) instead of `batch * max_seq`. Now `total_tokens != padded_tokens` — linear ops process only real tokens, `pad_to_batch` actually pads before attention, `unpad_from_batch` actually strips padding after.

- [ ] **Step 3: Fix final pooling to use padded masks**

The final pooling (lines 853-873) pads `hidden_states` to `[batch, max_seq, hidden]` then calls `mean_pool` with `inputs.pooling_mask`. After Task 1, `pooling_mask` is in padded `[batch * max_seq]` layout. Verify this code needs NO changes — it already pads before pooling and uses `max_seq` for the pool kernel.

Read lines 841-878 to confirm. The current code:
```rust
        let mut padded_for_pool = driver.alloc_zeros(batch * max_seq * hidden)?;
        driver.pad_to_batch(
            &hidden_states,
            &mut padded_for_pool,
            &inputs.seq_lengths,
            max_seq,
            hidden,
        )?;
        let mut pooled = driver.alloc_zeros(batch * hidden)?;
        driver.mean_pool(
            &mut pooled,
            &padded_for_pool,
            &inputs.pooling_mask,
            batch,
            max_seq,
            hidden,
        )?;
```

This is correct — `pad_to_batch` pads from `[total_tokens, hidden]` to `[batch*max_seq, hidden]`, and `mean_pool` uses the padded `pooling_mask`. No changes needed.

- [ ] **Step 4: Verify compilation**

Run: `cargo check -p ripvec-core`
Expected: compiles with no new errors

- [ ] **Step 5: Run correctness test**

Run the existing ModernBERT test (requires model download):
```bash
cargo test -p ripvec-core -- modernbert --ignored 2>&1 | tail -20
```

If the model is available, verify cosine similarity is ≥ 0.98 against known-good embeddings. If not available, run a quick smoke test:
```bash
./target/release/ripvec "buffer pool aliasing" --modern -n 3
```
Verify results are sensible (non-empty, scores > 0.5).

- [ ] **Step 6: Commit**

```bash
git add crates/ripvec-core/src/backend/arch/modern_bert.rs
git commit -m "perf(modernbert): activate unpadded batch — linear ops skip padding tokens"
```

---

### Task 4: Add `--layers` CLI flag for configurable early exit

**Agent:** Haiku (mechanical plumbing, no complex logic)
**Parallel:** Yes — independent of Tasks 1-3

**Problem:** `ModernBertArch.max_layers` exists but has no CLI path. Need a `--layers <N>` flag that's easy to configure.

**Files:**
- Modify: `crates/ripvec/src/cli.rs`
- Modify: `crates/ripvec/src/main.rs`

- [ ] **Step 1: Add `--layers` flag to CLI**

In `crates/ripvec/src/cli.rs`, add after the `max_tokens` field (line 82):

```rust
    /// Number of encoder layers to run (ModernBERT only, 1-22).
    ///
    /// Fewer layers = faster inference at the cost of embedding quality.
    /// ModernBERT has 22 layers; values 14-18 offer good quality/speed tradeoffs.
    /// 0 means all layers (default).
    #[arg(long, default_value_t = 0)]
    pub layers: usize,
```

- [ ] **Step 2: Wire `--layers` through to `ModernBertArch.max_layers`**

In `crates/ripvec/src/main.rs`, find where the backend/model is constructed. After the `ModernBertArch` is created, set `max_layers`. Search for where `modernbert` or `model_repo` is used to create the backend.

The `load_backend` function in `crates/ripvec-core/src/backend/mod.rs` creates the arch. Find the line that sets `arch.max_layers = None` for the Metal+ModernBERT path and change it.

In `main.rs`, after `load_backend` is called, check if there's a way to pass `layers` through. The cleanest approach: add a `max_layers: Option<usize>` parameter to `load_backend` (or a config struct).

If `load_backend` already accepts options, add `layers` there. Otherwise, the simplest approach is to modify the `BackendConfig` or equivalent to carry `max_layers`.

Read `crates/ripvec-core/src/backend/mod.rs` around line 848 to see how `max_layers` is set. Currently it's hardcoded to `None`. Change the function signature to accept an optional `max_layers` parameter, and pass `args.layers` from main.rs (converting 0 to None, N to Some(N)).

- [ ] **Step 3: Verify compilation**

Run: `cargo check --workspace`
Expected: compiles

- [ ] **Step 4: Quick test**

```bash
./target/release/ripvec "test query" --modern --layers 16 -n 3
```
Expected: results return (possibly different quality than full model, but no crash).

- [ ] **Step 5: Commit**

```bash
git add crates/ripvec/src/cli.rs crates/ripvec/src/main.rs crates/ripvec-core/src/backend/mod.rs
git commit -m "feat(cli): add --layers flag for configurable ModernBERT early exit"
```

---

### Task 5: Benchmark, verify correctness, and profile

**Agent:** Opus (requires analysis and judgment)
**Parallel:** No — depends on Tasks 3 and 4

**Problem:** Must verify the optimizations are correct (cosine similarity preserved) and measure the actual throughput improvement.

**Files:**
- No new files — uses existing benchmark infrastructure

- [ ] **Step 1: Build release with profiling**

```bash
RUSTFLAGS="-C force-frame-pointers=yes" cargo build --release
```

- [ ] **Step 2: Baseline correctness — full model**

Run on the test corpus with full 22 layers:
```bash
./target/release/ripvec "metal buffer pool" --modern -n 5 --profile
```

Record: query time, result files, similarity scores. Compare against pre-optimization results to verify no regression.

- [ ] **Step 3: Benchmark throughput — full model**

Run the full corpus embedding to measure chunks/s:
```bash
time ./target/release/ripvec "embedding optimization" --modern -n 1 --profile
```

Record the `embed_all` time and compute chunks/s. Target: ≥ 85 chunks/s (was 60).

- [ ] **Step 4: Benchmark throughput — early exit at 16 layers**

```bash
time ./target/release/ripvec "embedding optimization" --modern --layers 16 -n 1 --profile
```

Record chunks/s. Target: ≥ 110 chunks/s.

- [ ] **Step 5: Quality evaluation — early exit sweep**

Run the existing early exit quality test if available, or manually:
```bash
cargo test -p ripvec-core -- early_exit --ignored 2>&1
```

Or run queries at different layer counts and compare top-5 results:
```bash
for layers in 14 16 18 20 22; do
  echo "=== layers=$layers ==="
  ./target/release/ripvec "metal buffer pool" --modern --layers $layers -n 5 2>&1 | head -10
done
```

Verify that results at 16-18 layers are substantially similar to full model.

- [ ] **Step 6: Metal System Trace profile**

Record a trace to verify the unpadding is working (compute commands should be shorter):
```bash
xcrun xctrace record \
  --template 'Metal System Trace' \
  --output /tmp/ripvec_modernbert_optimized.trace \
  --time-limit 30s \
  --no-prompt \
  --env MTL_CAPTURE_ENABLED=1 \
  --target-stdout /dev/null \
  --launch -- ./target/release/ripvec "embedding pipeline" --modern -n 3
```

Import into tracemeld and verify:
- GPU idle % decreased from 11%
- Driver overhead decreased from 7%
- Compute spans are shorter (fewer padded tokens processed)

- [ ] **Step 7: Commit results summary**

Add a brief performance note to the commit:
```bash
git commit --allow-empty -m "perf(modernbert): unpadding + output reorder measured

Before: 60 chunks/s, 23% peak TFLOPS, 11% GPU idle
After:  <measured> chunks/s, <measured>% peak, <measured>% idle
Early exit (16 layers): <measured> chunks/s"
```

---

## Execution Strategy

```
Time →
────────────────────────────────────────────────────────

Parallel wave 1 (3 agents):
  [Sonnet] Task 1: Fix unpadded masks ──────────┐
  [Sonnet] Task 2: Reorder output FP32+FP16 ────┤
  [Haiku]  Task 4: --layers CLI flag ────────────┤
                                                 │
Sequential wave 2:                               ▼
  [Opus]   Task 3: Wire forward() ──────────────→│
                                                 │
  [Opus]   Task 5: Benchmark + verify ──────────→│
                                                 ▼
                                               Done
```

Tasks 1, 2, 4 run as **parallel subagents** (3 agents, ~2-5 min each).
Task 3 runs after 1+2 merge (Opus, ~5 min).
Task 5 runs after 3+4 merge (Opus, ~5 min with trace recording).

**Total estimated wall time with parallel agents:** ~15 min.
