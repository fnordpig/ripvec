//! `ClassicBert` architecture (BGE-small-en-v1.5).
//!
//! 12-layer BERT with learned position embeddings, GELU activation, fused QKV
//! projections, and CLS pooling. This is the original BERT architecture used
//! by BGE-small.
//!
//! Weight structures are generic over the tensor type `T`, which is
//! [`Driver::Tensor`](super::super::driver::Driver::Tensor) when wired to a
//! backend. The [`ModelArch`](super::ModelArch) implementation composes
//! [`Driver`](super::super::driver::Driver) primitives into the full forward
//! pass.

use super::super::Encoding;
use super::super::driver::{BatchInputs, Driver};
use super::ModelArch;

/// Weights for one `ClassicBert` encoder layer.
///
/// All projections include bias (unlike `NomicBert`). The QKV weight is a fused
/// `[3*hidden, hidden]` matrix that produces Q, K, V in a single GEMM.
pub struct ClassicBertLayerWeights<T> {
    /// Fused Q+K+V projection weight `[3*hidden, hidden]`.
    pub qkv_weight: T,
    /// Fused Q+K+V projection bias `[3*hidden]`.
    pub qkv_bias: T,
    /// Attention output projection weight `[hidden, hidden]`.
    pub output_weight: T,
    /// Attention output projection bias `[hidden]`.
    pub output_bias: T,
    /// Post-attention `LayerNorm` weight `[hidden]`.
    pub output_ln_weight: T,
    /// Post-attention `LayerNorm` bias `[hidden]`.
    pub output_ln_bias: T,
    /// FFN intermediate projection weight `[intermediate, hidden]`.
    pub ffn_inter_weight: T,
    /// FFN intermediate projection bias `[intermediate]`.
    pub ffn_inter_bias: T,
    /// FFN output projection weight `[hidden, intermediate]`.
    pub ffn_out_weight: T,
    /// FFN output projection bias `[hidden]`.
    pub ffn_out_bias: T,
    /// Post-FFN `LayerNorm` weight `[hidden]`.
    pub ffn_ln_weight: T,
    /// Post-FFN `LayerNorm` bias `[hidden]`.
    pub ffn_ln_bias: T,
}

/// Full `ClassicBert` model weights, generic over tensor type.
///
/// Includes embedding tables, per-layer encoder weights, and model geometry.
/// The tensor type `T` becomes [`Driver::Tensor`](super::super::driver::Driver::Tensor)
/// when loaded onto a specific backend.
pub struct ClassicBertWeights<T> {
    /// Word embedding table `[vocab_size, hidden]`.
    pub word_embeddings: T,
    /// Learned position embedding table `[max_position, hidden]`.
    pub position_embeddings: T,
    /// Token type embedding table `[2, hidden]`.
    pub token_type_embeddings: T,
    /// Post-embedding `LayerNorm` weight `[hidden]`.
    pub emb_ln_weight: T,
    /// Post-embedding `LayerNorm` bias `[hidden]`.
    pub emb_ln_bias: T,
    /// Per-layer encoder weights.
    pub layers: Vec<ClassicBertLayerWeights<T>>,
    /// Number of attention heads (e.g., 12 for BGE-small).
    pub num_heads: usize,
    /// Dimension per attention head (`hidden / num_heads`).
    pub head_dim: usize,
    /// Hidden dimension (e.g., 384 for BGE-small).
    pub hidden_dim: usize,
    /// FFN intermediate dimension (e.g., 1536 for BGE-small).
    pub intermediate_dim: usize,
    /// Layer normalization epsilon (typically 1e-12).
    pub layer_norm_eps: f32,
}

/// `ClassicBert` architecture: BGE-small-en-v1.5.
///
/// 12 layers, learned position embeddings, GELU activation, CLS pooling.
/// Composes [`Driver`] primitives into the full forward pass.
pub struct ClassicBertArch<T> {
    /// Model weights on device.
    pub weights: ClassicBertWeights<T>,
}

/// Encoder geometry passed to sublayer helpers to avoid repeating fields.
struct EncoderGeometry {
    batch: usize,
    max_seq: usize,
    total_tokens: usize,
    hidden: usize,
    num_heads: usize,
    head_dim: usize,
    intermediate: usize,
    scale: f32,
    eps: f32,
}

/// Run the self-attention sublayer for one encoder layer.
///
/// QKV projection -> split heads -> scaled dot-product attention ->
/// reshape -> output projection -> bias + residual + `LayerNorm`.
fn attention_sublayer<D: Driver>(
    driver: &D,
    hidden_states: &D::Tensor,
    layer: &ClassicBertLayerWeights<D::Tensor>,
    inputs: &BatchInputs<D::Tensor>,
    g: &EncoderGeometry,
) -> crate::Result<D::Tensor> {
    // QKV projection: [total_tokens, hidden] @ [3*hidden, hidden]^T
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
    driver.add_bias(&mut qkv, &layer.qkv_bias, g.total_tokens, 3 * g.hidden)?;

    // Split into Q, K, V each [batch * num_heads, seq, head_dim].
    let mut q = driver.alloc_zeros(g.total_tokens * g.hidden)?;
    let mut k = driver.alloc_zeros(g.total_tokens * g.hidden)?;
    let mut v = driver.alloc_zeros(g.total_tokens * g.hidden)?;
    driver.qkv_split(
        &mut q,
        &mut k,
        &mut v,
        &qkv,
        g.batch,
        g.max_seq,
        g.hidden,
        g.num_heads,
        g.head_dim,
    )?;

    // Attention scores: Q @ K^T => [batch * num_heads, seq, seq]
    let mut scores = driver.alloc_zeros(g.batch * g.num_heads * g.max_seq * g.max_seq)?;
    driver.gemm_batched(
        &q,
        &k,
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
    driver.fused_scale_mask_softmax(
        &mut scores,
        &inputs.float_mask,
        g.batch,
        g.num_heads,
        g.max_seq,
        g.scale,
    )?;

    // Weighted sum: scores @ V => [batch * num_heads, seq, head_dim]
    let mut attn_out = driver.alloc_zeros(g.total_tokens * g.hidden)?;
    driver.gemm_batched(
        &scores,
        &v,
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

    // Reshape heads back to [total_tokens, hidden].
    let mut context = driver.alloc_zeros(g.total_tokens * g.hidden)?;
    driver.attn_reshape(
        &mut context,
        &attn_out,
        g.batch,
        g.max_seq,
        g.num_heads,
        g.head_dim,
    )?;

    // Output projection + bias + residual + LayerNorm.
    let mut projected = driver.alloc_zeros(g.total_tokens * g.hidden)?;
    driver.gemm(
        &context,
        &layer.output_weight,
        &mut projected,
        g.total_tokens,
        g.hidden,
        g.hidden,
        true,
    )?;
    driver.add_bias(&mut projected, &layer.output_bias, g.total_tokens, g.hidden)?;

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

/// Run the feed-forward sublayer for one encoder layer.
///
/// Intermediate GEMM -> bias + GELU -> output GEMM -> bias + residual + `LayerNorm`.
fn ffn_sublayer<D: Driver>(
    driver: &D,
    attn_output: &D::Tensor,
    layer: &ClassicBertLayerWeights<D::Tensor>,
    g: &EncoderGeometry,
) -> crate::Result<D::Tensor> {
    // Intermediate: [total_tokens, hidden] @ [inter, hidden]^T => [total_tokens, inter]
    let mut intermediate = driver.alloc_zeros(g.total_tokens * g.intermediate)?;
    driver.gemm(
        attn_output,
        &layer.ffn_inter_weight,
        &mut intermediate,
        g.total_tokens,
        g.intermediate,
        g.hidden,
        true,
    )?;
    driver.fused_bias_gelu(
        &mut intermediate,
        &layer.ffn_inter_bias,
        g.total_tokens,
        g.intermediate,
    )?;

    // Output: [total_tokens, inter] @ [hidden, inter]^T => [total_tokens, hidden]
    let mut ffn_out = driver.alloc_zeros(g.total_tokens * g.hidden)?;
    driver.gemm(
        &intermediate,
        &layer.ffn_out_weight,
        &mut ffn_out,
        g.total_tokens,
        g.hidden,
        g.intermediate,
        true,
    )?;
    driver.add_bias(&mut ffn_out, &layer.ffn_out_bias, g.total_tokens, g.hidden)?;

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

impl<D: Driver> ModelArch<D> for ClassicBertArch<D::Tensor> {
    #[expect(
        clippy::cast_precision_loss,
        reason = "head_dim is small (32-64); sqrt is exact at these sizes"
    )]
    fn forward(&self, driver: &D, encodings: &[Encoding]) -> crate::Result<Vec<Vec<f32>>> {
        let w = &self.weights;
        let batch = encodings.len();
        let max_seq = encodings
            .iter()
            .map(|e| e.input_ids.len())
            .max()
            .unwrap_or(0)
            .next_multiple_of(8); // Pad for GEMM alignment.
        let total_tokens = batch * max_seq;
        let hidden = w.hidden_dim;

        // Prepare batch inputs on device.
        let inputs = driver.prepare_batch(encodings, max_seq)?;

        // Embedding: word + position + token_type + LayerNorm.
        let mut hidden_states =
            driver.embedding_lookup(&inputs.input_ids, &w.word_embeddings, total_tokens, hidden)?;
        driver.add_embeddings(
            &mut hidden_states,
            &w.position_embeddings,
            &inputs.position_ids,
            total_tokens,
            hidden,
        )?;
        driver.add_embeddings(
            &mut hidden_states,
            &w.token_type_embeddings,
            &inputs.token_type_ids,
            total_tokens,
            hidden,
        )?;
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
            hidden,
            num_heads: w.num_heads,
            head_dim: w.head_dim,
            intermediate: w.intermediate_dim,
            scale: 1.0 / (w.head_dim as f32).sqrt(),
            eps: w.layer_norm_eps,
        };

        // Encoder layers.
        for layer in &w.layers {
            let attn_output = attention_sublayer(driver, &hidden_states, layer, &inputs, &g)?;
            hidden_states = ffn_sublayer(driver, &attn_output, layer, &g)?;
        }

        // CLS pooling + L2 normalize.
        let mut pooled = driver.alloc_zeros(batch * hidden)?;
        driver.cls_pool(&mut pooled, &hidden_states, batch, max_seq, hidden)?;
        driver.l2_normalize(&mut pooled, batch, hidden)?;
        driver.to_host(&pooled, batch, hidden)
    }
}
