//! `NomicBert` architecture (`nomic-ai/CodeRankEmbed`).
//!
//! 12-layer transformer with fused QKV (no bias), `RoPE` position encoding,
//! `SwiGLU` activation, and mean pooling. All attention is global (no sliding
//! window). `LayerNorm` layers have bias (unlike `ModernBERT`). Uses post-norm
//! (residual + `LayerNorm` after each sublayer), matching the original BERT
//! architecture.
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

/// Weights for one `NomicBert` encoder layer.
///
/// No bias on linear projections. QKV is a single fused `[3*hidden, hidden]`
/// weight. The FFN uses two separate projections (`fc11`=up, `fc12`=gate) for
/// `SwiGLU`, not a single fused matrix.
pub struct NomicBertLayerWeights<T> {
    /// Fused Q+K+V projection weight `[3*hidden, hidden]` -- no bias.
    pub qkv_weight: T,
    /// Attention output projection weight `[hidden, hidden]` -- no bias.
    pub output_weight: T,
    /// Post-attention `LayerNorm` weight `[hidden]` (`norm1`).
    pub output_ln_weight: T,
    /// Post-attention `LayerNorm` bias `[hidden]` (`norm1`).
    pub output_ln_bias: T,
    /// FFN up/value projection weight `[intermediate, hidden]` (`fc11`).
    pub ffn_up_weight: T,
    /// FFN gate projection weight `[intermediate, hidden]` (`fc12`).
    pub ffn_gate_weight: T,
    /// FFN down/output projection weight `[hidden, intermediate]` (`fc2`).
    pub ffn_down_weight: T,
    /// Post-FFN `LayerNorm` weight `[hidden]` (`norm2`).
    pub ffn_ln_weight: T,
    /// Post-FFN `LayerNorm` bias `[hidden]` (`norm2`).
    pub ffn_ln_bias: T,
}

/// Full `NomicBert` model weights, generic over tensor type.
///
/// Includes embedding table, embedding `LayerNorm`, per-layer encoder weights,
/// `RoPE` cache, and model geometry. No final norm before pooling (unlike
/// `ModernBERT`).
pub struct NomicBertWeights<T> {
    /// Word embedding table `[vocab_size, hidden]`.
    pub tok_embeddings: T,
    /// Token type embedding table `[2, hidden]` (optional but present in CodeRankEmbed).
    pub token_type_embeddings: Option<T>,
    /// Post-embedding `LayerNorm` weight `[hidden]`.
    pub emb_ln_weight: T,
    /// Post-embedding `LayerNorm` bias `[hidden]`.
    pub emb_ln_bias: T,
    /// Per-layer encoder weights.
    pub layers: Vec<NomicBertLayerWeights<T>>,
    /// Pre-computed `RoPE` cosine table `[max_seq, head_dim/2]`.
    pub rope_cos: T,
    /// Pre-computed `RoPE` sine table `[max_seq, head_dim/2]`.
    pub rope_sin: T,
    /// Number of attention heads (12 for `CodeRankEmbed`).
    pub num_heads: usize,
    /// Dimension per attention head (`hidden / num_heads`, 64).
    pub head_dim: usize,
    /// Hidden dimension (768 for `CodeRankEmbed`).
    pub hidden_dim: usize,
    /// FFN intermediate dimension (3072 for `CodeRankEmbed`).
    pub intermediate_dim: usize,
    /// Layer normalization epsilon (typically 1e-12).
    pub layer_norm_eps: f32,
}

/// `NomicBert` architecture: `nomic-ai/CodeRankEmbed`.
///
/// 12 layers, `RoPE` (single theta), `SwiGLU` activation, post-norm, no final
/// norm, mean pooling. Composes [`Driver`] primitives into the full forward pass.
pub struct NomicBertArch<T> {
    /// Model weights on device.
    pub weights: NomicBertWeights<T>,
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
    scale: f32,
    eps: f32,
}

// ---------------------------------------------------------------------------
// Attention sublayer — QKV + RoPE + scores + output + residual + LN
// ---------------------------------------------------------------------------

/// QKV projection (unpadded) + pad + split + `RoPE` (no pre-norm -- post-norm architecture).
///
/// Computes QKV directly from `hidden_states` without a preceding `LayerNorm`.
/// QKV GEMM runs on `total_tokens` (unpadded), then pads to `batch*max_seq`
/// for `qkv_split` and attention.
/// Returns `(q, k, v)` each `[batch*num_heads, seq, head_dim]`.
fn attn_qkv_rope<D: Driver>(
    driver: &D,
    hidden_states: &D::Tensor,
    layer: &NomicBertLayerWeights<D::Tensor>,
    g: &EncoderGeometry,
    weights: &NomicBertWeights<D::Tensor>,
) -> crate::Result<(D::Tensor, D::Tensor, D::Tensor)> {
    // QKV projection: [total_tokens, hidden] @ [3*hidden, hidden]^T -- no bias.
    // Uses total_tokens (unpadded) — no wasted compute on padding.
    let mut qkv = driver.alloc_zeros(g.total_tokens * 3 * g.hidden)?;
    driver.gemm(
        hidden_states,
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

    // Apply RoPE to Q and K (single theta).
    let num_rows = g.batch * g.num_heads * g.max_seq;
    driver.apply_rope(
        &mut q,
        &weights.rope_cos,
        &weights.rope_sin,
        num_rows,
        g.max_seq,
        g.head_dim,
        g.num_heads,
    )?;
    driver.apply_rope(
        &mut k,
        &weights.rope_cos,
        &weights.rope_sin,
        num_rows,
        g.max_seq,
        g.head_dim,
        g.num_heads,
    )?;

    Ok((q, k, v))
}

/// Attention scores + output projection (padded) + unpad + fused residual + `LayerNorm`.
///
/// Post-norm architecture: `output = LN(output_proj + hidden_states)`.
/// Attention operates in padded layout; output projection unpads back to
/// `[total_tokens, H]` before the residual + `LayerNorm`.
#[expect(clippy::too_many_arguments, reason = "Q/K/V must be separate tensors")]
fn attn_scores_residual<D: Driver>(
    driver: &D,
    q: &D::Tensor,
    k: &D::Tensor,
    v: &D::Tensor,
    hidden_states: &D::Tensor,
    layer: &NomicBertLayerWeights<D::Tensor>,
    inputs: &BatchInputs<D::Tensor>,
    g: &EncoderGeometry,
) -> crate::Result<D::Tensor> {
    let padded = g.padded_tokens;

    // Attention scores: Q @ K^T => [batch * num_heads, seq, seq]
    let mut scores = driver.alloc_zeros(g.batch * g.num_heads * g.max_seq * g.max_seq)?;
    driver.gemm_batched(
        q,
        k,
        &mut scores,
        g.max_seq,
        g.max_seq,
        g.head_dim,
        true,
        g.max_seq * g.head_dim,
        g.max_seq * g.head_dim,
        g.max_seq * g.max_seq,
        g.batch * g.num_heads,
    )?;

    // Scale + mask + softmax (always global -- no sliding window).
    driver.fused_scale_mask_softmax(
        &mut scores,
        &inputs.float_mask,
        g.batch,
        g.num_heads,
        g.max_seq,
        g.scale,
    )?;

    // Weighted sum: scores @ V => [batch * num_heads, seq, head_dim]
    let mut attn_out = driver.alloc_zeros(padded * g.hidden)?;
    driver.gemm_batched(
        &scores,
        v,
        &mut attn_out,
        g.max_seq,
        g.head_dim,
        g.max_seq,
        false,
        g.max_seq * g.max_seq,
        g.max_seq * g.head_dim,
        g.max_seq * g.head_dim,
        g.batch * g.num_heads,
    )?;

    // Reshape heads back to [padded_tokens, hidden] (still padded).
    let mut context = driver.alloc_zeros(padded * g.hidden)?;
    driver.attn_reshape(
        &mut context,
        &attn_out,
        g.batch,
        g.max_seq,
        g.num_heads,
        g.head_dim,
    )?;

    // Output projection on padded layout, then unpad.
    let mut projected_padded = driver.alloc_zeros(padded * g.hidden)?;
    driver.gemm(
        &context,
        &layer.output_weight,
        &mut projected_padded,
        padded,
        g.hidden,
        g.hidden,
        true,
    )?;

    // Unpad: [padded_tokens, H] → [total_tokens, H]
    let mut projected = driver.alloc_zeros(g.total_tokens * g.hidden)?;
    driver.unpad_from_batch(
        &projected_padded,
        &mut projected,
        &g.seq_lengths,
        g.max_seq,
        g.hidden,
    )?;

    // Post-norm: output = LN(projected + hidden_states, norm1_weight, norm1_bias).
    let mut output = driver.alloc_zeros(g.total_tokens * g.hidden)?;
    driver.fused_residual_layernorm(
        &mut output,
        &projected,
        hidden_states,
        &layer.output_ln_weight,
        &layer.output_ln_bias,
        g.total_tokens,
        g.hidden,
        g.eps,
    )?;
    Ok(output)
}

// ---------------------------------------------------------------------------
// Feed-forward (SwiGLU MLP) sublayer
// ---------------------------------------------------------------------------

/// Run the `SwiGLU` MLP sublayer for one `NomicBert` encoder layer.
///
/// Post-norm architecture: up GEMM + gate GEMM -> `SwiGLU` ->
/// down GEMM -> fused residual + `LayerNorm`. No pre-FFN norm -- the
/// `LayerNorm` is applied *after* the residual add.
fn ffn_sublayer<D: Driver>(
    driver: &D,
    attn_output: &D::Tensor,
    layer: &NomicBertLayerWeights<D::Tensor>,
    g: &EncoderGeometry,
) -> crate::Result<D::Tensor> {
    let n_elements = g.total_tokens * g.intermediate;

    // Up/value projection: [total_tokens, hidden] @ [inter, hidden]^T => [total_tokens, inter]
    // Reads from attn_output directly (no pre-norm in post-norm architecture).
    let mut up_out = driver.alloc_zeros(n_elements)?;
    driver.gemm(
        attn_output,
        &layer.ffn_up_weight,
        &mut up_out,
        g.total_tokens,
        g.intermediate,
        g.hidden,
        true,
    )?;

    // Gate projection: [total_tokens, hidden] @ [inter, hidden]^T => [total_tokens, inter]
    let mut gate_out = driver.alloc_zeros(n_elements)?;
    driver.gemm(
        attn_output,
        &layer.ffn_gate_weight,
        &mut gate_out,
        g.total_tokens,
        g.intermediate,
        g.hidden,
        true,
    )?;

    // SwiGLU: output = up_out * silu(gate_out)
    let mut activated = driver.alloc_zeros(n_elements)?;
    driver.swiglu(&up_out, &gate_out, &mut activated, n_elements)?;

    // Down/output projection: [total_tokens, inter] @ [hidden, inter]^T => [total_tokens, hidden]
    let mut ffn_out = driver.alloc_zeros(g.total_tokens * g.hidden)?;
    driver.gemm(
        &activated,
        &layer.ffn_down_weight,
        &mut ffn_out,
        g.total_tokens,
        g.hidden,
        g.intermediate,
        true,
    )?;

    // Post-norm: output = LN(ffn_out + attn_output, norm2_weight, norm2_bias).
    let mut output = driver.alloc_zeros(g.total_tokens * g.hidden)?;
    driver.fused_residual_layernorm(
        &mut output,
        &ffn_out,
        attn_output,
        &layer.ffn_ln_weight,
        &layer.ffn_ln_bias,
        g.total_tokens,
        g.hidden,
        g.eps,
    )?;
    Ok(output)
}

// ---------------------------------------------------------------------------
// ModelArch implementation
// ---------------------------------------------------------------------------

impl<D: Driver> ModelArch<D> for NomicBertArch<D::Tensor> {
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

        // Unpadded mode: tokens concatenated without padding.
        // Linear layers (GEMM, LN, SwiGLU) process total_tokens rows — no wasted compute.
        // Attention pads/unpads around per-head operations via pad_to_batch/unpad_from_batch.
        let inputs = driver.prepare_batch_unpadded(encodings)?;
        let total_tokens = inputs.total_tokens;
        let max_seq = inputs.max_seq;

        // Enter batched mode: all GPU ops encode into ONE command buffer.
        driver.begin_batch()?;

        // Embedding: tok_embeddings + token_type + LayerNorm.
        let mut hidden_states =
            driver.embedding_lookup(&inputs.input_ids, &w.tok_embeddings, total_tokens, hidden)?;
        if let Some(ref tok_type_emb) = w.token_type_embeddings {
            driver.add_embeddings(
                &mut hidden_states,
                tok_type_emb,
                &inputs.token_type_ids,
                total_tokens,
                hidden,
            )?;
        }
        let emb_input = driver.clone_tensor(&hidden_states, total_tokens * hidden)?;
        driver.layer_norm(
            &mut hidden_states,
            &emb_input,
            &w.emb_ln_weight,
            &w.emb_ln_bias,
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
            scale: 1.0 / (w.head_dim as f32).sqrt(),
            eps: w.layer_norm_eps,
        };

        // Encoder layers (all 12, post-norm architecture).
        for layer in &w.layers {
            driver.reset_layer_workspace();
            let (q, k, v) = attn_qkv_rope(driver, &hidden_states, layer, &g, w)?;
            let attn_output =
                attn_scores_residual(driver, &q, &k, &v, &hidden_states, layer, &inputs, &g)?;
            hidden_states = ffn_sublayer(driver, &attn_output, layer, &g)?;
        }

        // No final norm before pooling (unlike ModernBERT).

        // Pad back to [batch, max_seq, hidden] for mean_pool kernel.
        let mut padded_for_pool = driver.alloc_zeros(batch * max_seq * hidden)?;
        driver.pad_to_batch(
            &hidden_states,
            &mut padded_for_pool,
            &inputs.seq_lengths,
            max_seq,
            hidden,
        )?;

        // Mean pooling + L2 normalize.
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

        // End batched mode -- commit all GPU work, wait for completion.
        driver.end_batch()?;

        driver.to_host(&pooled, batch, hidden)
    }
}
