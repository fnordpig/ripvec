//! `ModernBERT` architecture (`nomic-ai/modernbert-embed-base`).
//!
//! 22-layer transformer with alternating local/global attention, gated GELU
//! (`GeGLU`) MLP, two `RoPE` frequency caches, and pre-norm layer structure.
//! No biases anywhere, no position embeddings (`RoPE` only), mean pooling.
//!
//! Weight structures are generic over the tensor type `T`, which is
//! [`Driver::Tensor`](super::super::driver::Driver::Tensor) when wired to a
//! backend. The [`ModelArch`](super::ModelArch) implementation composes
//! [`Driver`](super::super::driver::Driver) primitives into the full forward
//! pass.

use super::super::Encoding;
use super::super::driver::{BatchInputs, Driver};
use super::ModelArch;

// ---------------------------------------------------------------------------
// Weight structures
// ---------------------------------------------------------------------------

/// Weights for one `ModernBERT` encoder layer.
///
/// All projections are bias-free. The QKV weight is a fused `[3*hidden, hidden]`
/// matrix. The MLP `Wi` is `[2*intermediate, hidden]` and chunks into value +
/// gate halves for the `GeGLU` activation.
pub struct ModernBertLayerWeights<T> {
    /// Fused Q+K+V projection weight `[3*hidden, hidden]` -- no bias.
    pub qkv_weight: T,
    /// Attention output projection weight `[hidden, hidden]` -- no bias.
    pub output_weight: T,
    /// Pre-attention `LayerNorm` weight `[hidden]` -- `None` for layer 0 (identity).
    pub attn_norm_weight: Option<T>,
    /// Gated MLP input weight `[2*intermediate, hidden]` -- chunks to
    /// `value[intermediate]` + `gate[intermediate]` for `GeGLU`.
    pub mlp_wi_weight: T,
    /// MLP output projection weight `[hidden, intermediate]` -- no bias.
    pub mlp_wo_weight: T,
    /// Pre-MLP `LayerNorm` weight `[hidden]`.
    pub mlp_norm_weight: T,
    /// Whether this layer uses global (full) or local (sliding window) attention.
    pub is_global: bool,
}

/// Full `ModernBERT` model weights, generic over tensor type.
///
/// Includes embedding table, per-layer encoder weights, final norm, and model
/// geometry. The tensor type `T` becomes
/// [`Driver::Tensor`](super::super::driver::Driver::Tensor) when loaded onto a
/// specific backend.
pub struct ModernBertWeights<T> {
    /// Word embedding table `[vocab_size, hidden]`.
    pub tok_embeddings: T,
    /// Post-embedding `LayerNorm` weight `[hidden]` (no bias).
    pub emb_norm_weight: T,
    /// Final `LayerNorm` weight `[hidden]` applied before pooling (no bias).
    pub final_norm_weight: T,
    /// A zero-filled tensor `[hidden]` used as dummy bias for `LayerNorm` calls.
    ///
    /// The [`Driver::layer_norm`] API requires a bias tensor; `ModernBERT` has
    /// none, so we pass this zero buffer instead.
    pub zero_bias: T,
    /// Per-layer encoder weights.
    pub layers: Vec<ModernBertLayerWeights<T>>,
    /// Number of attention heads (12 for modernbert-embed-base).
    pub num_heads: usize,
    /// Dimension per attention head (`hidden / num_heads`, 64).
    pub head_dim: usize,
    /// Hidden dimension (768 for modernbert-embed-base).
    pub hidden_dim: usize,
    /// MLP intermediate dimension (1152 for modernbert-embed-base).
    pub intermediate_dim: usize,
    /// Layer normalization epsilon (1e-5).
    pub layer_norm_eps: f32,
    /// Sliding window size for local attention layers (128).
    pub local_window: usize,
}

/// Pre-computed `RoPE` cos/sin cache for one frequency base.
///
/// `ModernBERT` uses two caches: one for global layers (theta=160000) and one
/// for local layers (theta=10000).
pub struct RopeCache<T> {
    /// Cosine table `[max_seq, head_dim/2]`.
    pub cos: T,
    /// Sine table `[max_seq, head_dim/2]`.
    pub sin: T,
}

/// `ModernBERT` architecture: `nomic-ai/modernbert-embed-base`.
///
/// 22 layers, alternating local/global attention, `GeGLU` MLP, `RoPE` (two
/// theta values), no biases, mean pooling. Composes [`Driver`] primitives into
/// the full forward pass.
pub struct ModernBertArch<T> {
    /// Model weights on device.
    pub weights: ModernBertWeights<T>,
    /// `RoPE` cos/sin cache for global attention layers (theta=160000).
    pub global_rope: RopeCache<T>,
    /// `RoPE` cos/sin cache for local attention layers (theta=10000).
    pub local_rope: RopeCache<T>,
    /// Optional early exit: run only this many encoder layers.
    /// `None` runs all 22 layers.
    pub max_layers: Option<usize>,
}

// ---------------------------------------------------------------------------
// Encoder geometry
// ---------------------------------------------------------------------------

/// Encoder geometry passed to sublayer helpers to avoid repeating fields.
struct EncoderGeometry {
    batch: usize,
    max_seq: usize,
    /// Actual tokens across all sequences (no padding). Used for linear ops.
    total_tokens: usize,
    /// Padded total: `batch * max_seq`. Used for attention layout.
    padded_tokens: usize,
    /// Per-sequence lengths for pad/unpad.
    seq_lengths: Vec<usize>,
    hidden: usize,
    num_heads: usize,
    head_dim: usize,
    intermediate: usize,
    local_window: usize,
    scale: f32,
    eps: f32,
}

// ---------------------------------------------------------------------------
// Attention sublayer — pre-norm + QKV + RoPE
// ---------------------------------------------------------------------------

/// Pre-norm + QKV projection + split + `RoPE`.
///
/// Returns `(q, k, v)` each `[batch*num_heads, seq, head_dim]`.
fn attn_prenorm_qkv<D: Driver>(
    driver: &D,
    hidden_states: &D::Tensor,
    layer: &ModernBertLayerWeights<D::Tensor>,
    g: &EncoderGeometry,
    zero_bias: &D::Tensor,
    rope: &RopeCache<D::Tensor>,
) -> crate::Result<(D::Tensor, D::Tensor, D::Tensor)> {
    // Pre-attention norm (identity for layer 0).
    let normed = if let Some(ref norm_w) = layer.attn_norm_weight {
        let mut n = driver.alloc_zeros(g.total_tokens * g.hidden)?;
        driver.layer_norm(
            &mut n,
            hidden_states,
            norm_w,
            zero_bias,
            g.total_tokens,
            g.hidden,
            g.eps,
        )?;
        n
    } else {
        // Layer 0: identity -- just clone the input.
        driver.clone_tensor(hidden_states, g.total_tokens * g.hidden)?
    };

    // QKV projection: [total_tokens, hidden] @ [3*hidden, hidden]^T
    // Uses total_tokens (unpadded) — no wasted compute on padding.
    let mut qkv = driver.alloc_zeros(g.total_tokens * 3 * g.hidden)?;
    driver.gemm(
        &normed,
        &layer.qkv_weight,
        &mut qkv,
        g.total_tokens,
        3 * g.hidden,
        g.hidden,
        true,
    )?;

    // Pad QKV from [total_tokens, 3H] to [batch*max_seq, 3H] for attention.
    // qkv_split needs the padded batch×seq layout to reshape into per-head tensors.
    let mut qkv_padded = driver.alloc_zeros(g.padded_tokens * 3 * g.hidden)?;
    driver.pad_to_batch(
        &qkv,
        &mut qkv_padded,
        &g.seq_lengths,
        g.max_seq,
        3 * g.hidden,
    )?;

    // Split into Q, K, V each [batch * num_heads, seq, head_dim].
    let padded = g.padded_tokens;
    let mut q = driver.alloc_zeros(padded * g.hidden)?;
    let mut k = driver.alloc_zeros(padded * g.hidden)?;
    let mut v = driver.alloc_zeros(padded * g.hidden)?;
    driver.qkv_split(
        &mut q,
        &mut k,
        &mut v,
        &qkv_padded,
        g.batch,
        g.max_seq,
        g.hidden,
        g.num_heads,
        g.head_dim,
    )?;

    // Apply RoPE to Q and K with appropriate theta.
    let num_rows = g.batch * g.num_heads * g.max_seq;
    driver.apply_rope(
        &mut q,
        &rope.cos,
        &rope.sin,
        num_rows,
        g.max_seq,
        g.head_dim,
        g.num_heads,
    )?;
    driver.apply_rope(
        &mut k,
        &rope.cos,
        &rope.sin,
        num_rows,
        g.max_seq,
        g.head_dim,
        g.num_heads,
    )?;

    Ok((q, k, v))
}

// ---------------------------------------------------------------------------
// Attention sublayer — scores + output projection + residual
// ---------------------------------------------------------------------------

/// Attention scores + output projection + residual add.
#[expect(clippy::too_many_arguments, reason = "Q/K/V must be separate tensors")]
fn attn_scores_residual<D: Driver>(
    driver: &D,
    q: &D::Tensor,
    k: &D::Tensor,
    v: &D::Tensor,
    hidden_states: &D::Tensor,
    layer: &ModernBertLayerWeights<D::Tensor>,
    inputs: &BatchInputs<D::Tensor>,
    g: &EncoderGeometry,
) -> crate::Result<D::Tensor> {
    let batch_heads = g.batch * g.num_heads;
    let stride_qk = g.max_seq * g.head_dim;

    // Full Q@K^T for all layers. Local layers mask via windowed softmax.
    //
    // Banded attention kernels (banded_qk/sv) compute 4× fewer elements for
    // local layers but are scalar — 35% slower than hardware simdgroup GEMM.
    // The GEMM + windowed mask approach wastes compute on masked positions but
    // Apple's hardware matmul throughput more than compensates at seq≤512.
    // TODO: banded attention wins at seq>2048 where O(seq²) dominates.
    let mut scores = driver.alloc_zeros(batch_heads * g.max_seq * g.max_seq)?;
    driver.gemm_batched(
        q,
        k,
        &mut scores,
        g.max_seq,
        g.max_seq,
        g.head_dim,
        true,
        stride_qk,
        stride_qk,
        g.max_seq * g.max_seq,
        batch_heads,
    )?;

    if layer.is_global {
        driver.fused_scale_mask_softmax(
            &mut scores,
            &inputs.float_mask,
            g.batch,
            g.num_heads,
            g.max_seq,
            g.scale,
        )?;
    } else {
        driver.fused_scale_mask_softmax_windowed(
            &mut scores,
            &inputs.float_mask,
            g.batch,
            g.num_heads,
            g.max_seq,
            g.scale,
            g.local_window,
        )?;
    }

    let mut attn_out = driver.alloc_zeros(g.padded_tokens * g.hidden)?;
    driver.gemm_batched(
        &scores,
        v,
        &mut attn_out,
        g.max_seq,
        g.head_dim,
        g.max_seq,
        false,
        g.max_seq * g.max_seq,
        stride_qk,
        stride_qk,
        batch_heads,
    )?;

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
}

// ---------------------------------------------------------------------------
// Feed-forward (GeGLU MLP) sublayer
// ---------------------------------------------------------------------------

/// Run the gated GELU MLP sublayer for one `ModernBERT` encoder layer.
///
/// Pre-MLP norm -> Wi projection -> split into value+gate -> `GeGLU` ->
/// Wo projection -> residual add.
fn ffn_sublayer<D: Driver>(
    driver: &D,
    attn_output: &D::Tensor,
    layer: &ModernBertLayerWeights<D::Tensor>,
    g: &EncoderGeometry,
    zero_bias: &D::Tensor,
) -> crate::Result<D::Tensor> {
    // Pre-MLP LayerNorm.
    let mut mlp_normed = driver.alloc_zeros(g.total_tokens * g.hidden)?;
    driver.layer_norm(
        &mut mlp_normed,
        attn_output,
        &layer.mlp_norm_weight,
        zero_bias,
        g.total_tokens,
        g.hidden,
        g.eps,
    )?;

    // Wi projection: [total_tokens, hidden] @ [2*inter, hidden]^T => [total_tokens, 2*inter]
    let double_inter = 2 * g.intermediate;
    let mut wi_out = driver.alloc_zeros(g.total_tokens * double_inter)?;
    driver.gemm(
        &mlp_normed,
        &layer.mlp_wi_weight,
        &mut wi_out,
        g.total_tokens,
        double_inter,
        g.hidden,
        true,
    )?;

    // Split Wi output into value [total_tokens, inter] and gate [total_tokens, inter].
    let n_elements = g.total_tokens * g.intermediate;
    let mut value = driver.alloc_zeros(n_elements)?;
    let mut gate = driver.alloc_zeros(n_elements)?;
    driver.split_gate_value(
        &mut value,
        &mut gate,
        &wi_out,
        g.total_tokens,
        g.intermediate,
    )?;

    // GeGLU: output = gelu(value) * gate
    let mut activated = driver.alloc_zeros(n_elements)?;
    driver.geglu(&value, &gate, &mut activated, n_elements)?;

    // Wo projection: [total_tokens, inter] @ [hidden, inter]^T => [total_tokens, hidden]
    let mut mlp_out = driver.alloc_zeros(g.total_tokens * g.hidden)?;
    driver.gemm(
        &activated,
        &layer.mlp_wo_weight,
        &mut mlp_out,
        g.total_tokens,
        g.hidden,
        g.intermediate,
        true,
    )?;

    // Residual add (no bias).
    let mut output = driver.alloc_zeros(g.total_tokens * g.hidden)?;
    driver.residual_add(
        &mut output,
        &mlp_out,
        attn_output,
        g.total_tokens * g.hidden,
    )?;
    Ok(output)
}

// ---------------------------------------------------------------------------
// FP16 attention sublayer — pre-norm + QKV + RoPE (all half precision)
// ---------------------------------------------------------------------------

/// FP16 pre-norm + QKV projection + split + `RoPE`.
///
/// All tensors are half precision. RoPE cos/sin tables stay FP32 (the kernel
/// reads half Q/K, does FP32 trig, writes half).
/// Returns `(q, k, v)` each `[batch*num_heads, seq, head_dim]` in FP16.
fn attn_prenorm_qkv_f16<D: Driver>(
    driver: &D,
    hidden_states: &D::Tensor,
    layer: &ModernBertLayerWeights<D::Tensor>,
    g: &EncoderGeometry,
    zero_bias: &D::Tensor,
    rope: &RopeCache<D::Tensor>,
) -> crate::Result<(D::Tensor, D::Tensor, D::Tensor)> {
    // Pre-attention norm (identity for layer 0). FP16 in/out.
    // Layer 0 uses hidden_states directly (GEMM is read-only, no clone needed).
    let normed: Option<D::Tensor>;
    let normed_ref = if let Some(ref norm_w) = layer.attn_norm_weight {
        let mut n = driver.alloc_zeros_f16(g.total_tokens * g.hidden)?;
        driver.layer_norm_f16(
            &mut n,
            hidden_states,
            norm_w,
            zero_bias,
            g.total_tokens,
            g.hidden,
            g.eps,
        )?;
        normed = Some(n);
        normed.as_ref().unwrap()
    } else {
        // Layer 0: identity — pass through directly. GEMM reads, does not modify.
        hidden_states
    };

    // QKV: [total_tokens, hidden] @ [3*hidden, hidden]^T — all FP16.
    let mut qkv = driver.alloc_zeros_f16(g.total_tokens * 3 * g.hidden)?;
    driver.gemm_f16(
        normed_ref,
        &layer.qkv_weight,
        &mut qkv,
        g.total_tokens,
        3 * g.hidden,
        g.hidden,
        true,
    )?;

    // Pad QKV for attention layout.
    let mut qkv_padded = driver.alloc_zeros_f16(g.padded_tokens * 3 * g.hidden)?;
    driver.pad_to_batch_f16(
        &qkv,
        &mut qkv_padded,
        &g.seq_lengths,
        g.max_seq,
        3 * g.hidden,
    )?;

    // Split into Q, K, V — all FP16.
    let padded = g.padded_tokens;
    let mut q = driver.alloc_zeros_f16(padded * g.hidden)?;
    let mut k = driver.alloc_zeros_f16(padded * g.hidden)?;
    let mut v = driver.alloc_zeros_f16(padded * g.hidden)?;
    driver.qkv_split_f16(
        &mut q,
        &mut k,
        &mut v,
        &qkv_padded,
        g.batch,
        g.max_seq,
        g.hidden,
        g.num_heads,
        g.head_dim,
    )?;

    // RoPE: half Q/K, float cos/sin tables.
    let num_rows = g.batch * g.num_heads * g.max_seq;
    driver.rope_encode_f16(
        &mut q,
        &rope.cos,
        &rope.sin,
        num_rows,
        g.max_seq,
        g.head_dim,
        g.num_heads,
    )?;
    driver.rope_encode_f16(
        &mut k,
        &rope.cos,
        &rope.sin,
        num_rows,
        g.max_seq,
        g.head_dim,
        g.num_heads,
    )?;

    Ok((q, k, v))
}

// ---------------------------------------------------------------------------
// FP16 attention sublayer — scores + output projection + residual
// ---------------------------------------------------------------------------

/// FP16 attention scores + output projection + residual add.
///
/// All tensors FP16. The softmax kernel uses FP32 accumulators internally.
/// The `float_mask` from `BatchInputs` stays FP32 (softmax kernel reads it).
#[expect(clippy::too_many_arguments, reason = "Q/K/V must be separate tensors")]
fn attn_scores_residual_f16<D: Driver>(
    driver: &D,
    q: &D::Tensor,
    k: &D::Tensor,
    v: &D::Tensor,
    hidden_states: &D::Tensor,
    layer: &ModernBertLayerWeights<D::Tensor>,
    inputs: &BatchInputs<D::Tensor>,
    g: &EncoderGeometry,
) -> crate::Result<D::Tensor> {
    let batch_heads = g.batch * g.num_heads;
    let stride_qk = g.max_seq * g.head_dim;

    // Q@K^T — FP16 batched GEMM.
    let mut scores = driver.alloc_zeros_f16(batch_heads * g.max_seq * g.max_seq)?;
    driver.gemm_batched_f16(
        q,
        k,
        &mut scores,
        g.max_seq,
        g.max_seq,
        g.head_dim,
        true,
        stride_qk,
        stride_qk,
        g.max_seq * g.max_seq,
        batch_heads,
    )?;

    // Softmax — FP16 scores, FP32 mask, FP32 accumulators inside kernel.
    if layer.is_global {
        driver.fused_scale_mask_softmax_f16(
            &mut scores,
            &inputs.float_mask,
            g.batch,
            g.num_heads,
            g.max_seq,
            g.scale,
        )?;
    } else {
        driver.fused_scale_mask_softmax_windowed_f16(
            &mut scores,
            &inputs.float_mask,
            g.batch,
            g.num_heads,
            g.max_seq,
            g.scale,
            g.local_window,
        )?;
    }

    // scores @ V — FP16 batched GEMM.
    let mut attn_out = driver.alloc_zeros_f16(g.padded_tokens * g.hidden)?;
    driver.gemm_batched_f16(
        &scores,
        v,
        &mut attn_out,
        g.max_seq,
        g.head_dim,
        g.max_seq,
        false,
        g.max_seq * g.max_seq,
        stride_qk,
        stride_qk,
        batch_heads,
    )?;

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
}

// ---------------------------------------------------------------------------
// FP16 feed-forward (GeGLU MLP) sublayer
// ---------------------------------------------------------------------------

/// FP16 gated GELU MLP sublayer.
///
/// All tensors FP16. `GeGLU` kernel uses FP32 GELU compute internally.
fn ffn_sublayer_f16<D: Driver>(
    driver: &D,
    attn_output: &D::Tensor,
    layer: &ModernBertLayerWeights<D::Tensor>,
    g: &EncoderGeometry,
    zero_bias: &D::Tensor,
) -> crate::Result<D::Tensor> {
    // Pre-MLP LayerNorm — FP16.
    let mut mlp_normed = driver.alloc_zeros_f16(g.total_tokens * g.hidden)?;
    driver.layer_norm_f16(
        &mut mlp_normed,
        attn_output,
        &layer.mlp_norm_weight,
        zero_bias,
        g.total_tokens,
        g.hidden,
        g.eps,
    )?;

    // Wi projection — FP16 GEMM.
    let double_inter = 2 * g.intermediate;
    let mut wi_out = driver.alloc_zeros_f16(g.total_tokens * double_inter)?;
    driver.gemm_f16(
        &mlp_normed,
        &layer.mlp_wi_weight,
        &mut wi_out,
        g.total_tokens,
        double_inter,
        g.hidden,
        true,
    )?;

    // Split + GeGLU — FP16.
    let n_elements = g.total_tokens * g.intermediate;
    let mut value = driver.alloc_zeros_f16(n_elements)?;
    let mut gate = driver.alloc_zeros_f16(n_elements)?;
    driver.split_gate_value_f16(
        &mut value,
        &mut gate,
        &wi_out,
        g.total_tokens,
        g.intermediate,
    )?;

    let mut activated = driver.alloc_zeros_f16(n_elements)?;
    driver.geglu_f16(&value, &gate, &mut activated, n_elements)?;

    // Wo projection — FP16 GEMM.
    let mut mlp_out = driver.alloc_zeros_f16(g.total_tokens * g.hidden)?;
    driver.gemm_f16(
        &activated,
        &layer.mlp_wo_weight,
        &mut mlp_out,
        g.total_tokens,
        g.hidden,
        g.intermediate,
        true,
    )?;

    // Residual add — FP16.
    let mut output = driver.alloc_zeros_f16(g.total_tokens * g.hidden)?;
    driver.residual_add_f16(
        &mut output,
        &mlp_out,
        attn_output,
        g.total_tokens * g.hidden,
    )?;
    Ok(output)
}

// ---------------------------------------------------------------------------
// ModelArch implementation
// ---------------------------------------------------------------------------

impl<D: Driver> ModelArch<D> for ModernBertArch<D::Tensor> {
    #[expect(
        clippy::cast_precision_loss,
        reason = "head_dim is small (64); sqrt is exact at this size"
    )]
    #[expect(
        clippy::many_single_char_names,
        reason = "w, g are standard geometry names; q, k, v are standard attention names"
    )]
    fn forward(&self, driver: &D, encodings: &[Encoding]) -> crate::Result<Vec<Vec<f32>>> {
        let w = &self.weights;
        let batch = encodings.len();
        let hidden = w.hidden_dim;

        let inputs = driver.prepare_batch_unpadded(encodings)?;
        let max_seq = inputs.max_seq;
        let total_tokens = inputs.total_tokens;

        // Enter batched mode: all GPU ops encode into ONE command buffer.
        driver.begin_batch()?;

        // Embedding (FP32): tok_embeddings + LayerNorm.
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

        let num_layers = self
            .max_layers
            .unwrap_or(w.layers.len())
            .min(w.layers.len());

        // FP16 path: f32_to_f16 ONCE → all layers in FP16 → f16_to_f32 ONCE.
        // Falls back to FP32 if the driver doesn't support FP16 ops.
        //
        // MPS FP16 GEMM uses Apple's proprietary AMX coprocessor and achieves
        // RIPVEC_NO_MPS=1: force FP32 activations + compute GEMM path.
        // The gemm_f16w_f32a_kernel uses native simdgroup ops with FP16 weights
        // and FP32 activations — no MFA wrapper, no type conversion at store.
        let force_fp32 = std::env::var("RIPVEC_NO_MPS").is_ok_and(|v| v == "1")
            || std::env::var("RIPVEC_FP32").is_ok_and(|v| v == "1");
        let use_f16 = if force_fp32 {
            false
        } else {
            driver.alloc_zeros_f16(1).map(|_| true).unwrap_or(false)
        };

        if use_f16 {
            // === FP16 PATH: zero F32↔F16 conversions in layer loop ===

            // ONLY conversion #1: F32 → F16 after embedding LN.
            let mut hidden_f16 = driver.alloc_zeros_f16(total_tokens * hidden)?;
            driver.f32_to_f16(&mut hidden_f16, &hidden_states, total_tokens * hidden)?;

            // 22 layers — ALL in FP16. Zero conversions.
            for layer in &w.layers[..num_layers] {
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

            // ONLY conversion #2: F16 → F32 before final LN + pooling.
            let mut hidden_f32 = driver.alloc_zeros(total_tokens * hidden)?;
            driver.f16_to_f32(&mut hidden_f32, &hidden_f16, total_tokens * hidden)?;
            hidden_states = hidden_f32;
        } else {
            // === FP32 PATH ===
            for (li, layer) in w.layers[..num_layers].iter().enumerate() {
                let saved = driver.save_pool_cursor();

                let rope = if layer.is_global {
                    &self.global_rope
                } else {
                    &self.local_rope
                };

                let (q, k, v) =
                    attn_prenorm_qkv(driver, &hidden_states, layer, &g, &w.zero_bias, rope)?;
                let attn_output =
                    attn_scores_residual(driver, &q, &k, &v, &hidden_states, layer, &inputs, &g)?;
                hidden_states = ffn_sublayer(driver, &attn_output, layer, &g, &w.zero_bias)?;

                driver.restore_pool_cursor(saved);

                // Segment the compute encoder every 3 layers to prevent
                // encoder state overflow. This closes and reopens the encoder
                // within the same command buffer — zero sync, zero GPU idle.
                // Segment after EVERY layer (~19 dispatches per encoder)
                driver.segment_encoder();
            }
        }

        // Final LayerNorm (FP32) before pooling.
        let final_input = driver.clone_tensor(&hidden_states, total_tokens * hidden)?;
        driver.layer_norm(
            &mut hidden_states,
            &final_input,
            &w.final_norm_weight,
            &w.zero_bias,
            total_tokens,
            hidden,
            w.layer_norm_eps,
        )?;

        // Pad back to [batch, max_seq, hidden] for mean_pool kernel.
        let mut padded_for_pool = driver.alloc_zeros(batch * max_seq * hidden)?;
        driver.pad_to_batch(
            &hidden_states,
            &mut padded_for_pool,
            &inputs.seq_lengths,
            max_seq,
            hidden,
        )?;

        // Mean pooling + L2 normalize (FP32).
        let mut pooled = driver.alloc_zeros(batch * hidden)?;
        driver.mean_pool(
            &mut pooled,
            &padded_for_pool,
            &inputs.pooling_mask,
            batch,
            max_seq,
            hidden,
        )?;
        driver.l2_normalize(&mut pooled, batch, hidden)?;

        // End batched mode — commit all GPU work, wait for completion.
        driver.end_batch()?;

        driver.to_host(&pooled, batch, hidden)
    }
}
