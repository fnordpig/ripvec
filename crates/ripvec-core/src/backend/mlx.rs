//! MLX embedding backend for Apple Silicon.
//!
//! Implements BERT inference using Apple's MLX framework via [`mlx_rs`].
//! MLX uses unified memory and Metal compute shaders, avoiding the CPU
//! bottlenecks (software GELU, allocation overhead, CPU-GPU copies) that
//! limit the Candle backend on Apple Silicon.
//!
//! Supports two model families:
//! - **`ClassicBert`** (BGE models): learned position embeddings, GELU, QKV with bias.
//! - **`NomicBert`** (CodeRankEmbed, nomic-embed-text): RoPE, SwiGLU, no bias.
//!
//! Weights are loaded from safetensors files downloaded via `hf-hub` and
//! manually assigned to a hand-rolled BERT model. The model is wrapped in
//! `Arc<Mutex<_>>` because MLX's `Array` is `Send` but not `Sync`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use hf_hub::api::sync::Api;
use mlx_rs::Array;
use mlx_rs::ops::indexing::TryIndexOp;

use super::{DeviceHint, EmbedBackend, Encoding};

/// Convert an MLX exception into our crate error type.
fn mlx_err(e: impl std::fmt::Display) -> crate::Error {
    crate::Error::Other(anyhow::anyhow!("MLX: {e}"))
}

// ---------------------------------------------------------------------------
// Model variant detection
// ---------------------------------------------------------------------------

/// Which BERT variant the loaded weights correspond to.
///
/// `ClassicBert` uses learned position embeddings, GELU activation, and
/// biased QKV projections. `NomicBert` uses RoPE, SwiGLU, and unbiased
/// projections.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelVariant {
    /// Standard BERT / BGE models (e.g. `BAAI/bge-small-en-v1.5`).
    ClassicBert,
    /// NomicBert models (e.g. `nomic-ai/CodeRankEmbed`, `nomic-embed-text-v1.5`).
    NomicBert,
}

/// Detect the model variant by inspecting weight names.
///
/// `ClassicBert` has `embeddings.position_embeddings.weight`; `NomicBert`
/// does not (it uses rotary position encoding instead).
fn detect_variant(weights: &HashMap<String, Array>) -> ModelVariant {
    if weights.contains_key("embeddings.position_embeddings.weight") {
        ModelVariant::ClassicBert
    } else {
        ModelVariant::NomicBert
    }
}

// ---------------------------------------------------------------------------
// BERT model configuration
// ---------------------------------------------------------------------------

/// Configuration for a BERT-style encoder model.
///
/// Matches the `config.json` schema from HuggingFace model repos.
/// Supports both `ClassicBert` and `NomicBert` config key names.
#[derive(Debug, Clone)]
struct BertConfig {
    /// Which variant this config describes.
    variant: ModelVariant,
    /// Hidden dimension (384 for bge-small, 768 for nomic).
    hidden_size: i32,
    /// Number of transformer layers.
    num_hidden_layers: i32,
    /// Number of attention heads.
    num_attention_heads: i32,
    /// Maximum sequence length (512 for classic, 8192 for nomic).
    max_position_embeddings: i32,
    /// Base for rotary embeddings (only used by `NomicBert`).
    rotary_emb_base: f32,
    /// Layer norm epsilon.
    layer_norm_eps: f32,
}

impl BertConfig {
    /// Parse from a `config.json` value, dispatching on `variant`.
    fn from_json(v: &serde_json::Value, variant: ModelVariant) -> crate::Result<Self> {
        let get_i32 = |key: &str| -> crate::Result<i32> {
            v.get(key)
                .and_then(serde_json::Value::as_i64)
                .map(|n| n as i32)
                .ok_or_else(|| crate::Error::Other(anyhow::anyhow!("missing config key: {key}")))
        };
        let get_f64 = |key: &str| -> crate::Result<f64> {
            v.get(key)
                .and_then(serde_json::Value::as_f64)
                .ok_or_else(|| crate::Error::Other(anyhow::anyhow!("missing config key: {key}")))
        };

        #[expect(
            clippy::cast_possible_truncation,
            reason = "layer_norm_eps is always a small float"
        )]
        let layer_norm_eps =
            get_f64("layer_norm_epsilon").or_else(|_| get_f64("layer_norm_eps"))? as f32;

        match variant {
            ModelVariant::ClassicBert => Ok(Self {
                variant,
                hidden_size: get_i32("hidden_size")?,
                num_hidden_layers: get_i32("num_hidden_layers")?,
                num_attention_heads: get_i32("num_attention_heads")?,
                max_position_embeddings: get_i32("max_position_embeddings").unwrap_or(512),
                rotary_emb_base: 10000.0,
                layer_norm_eps,
            }),
            ModelVariant::NomicBert => {
                // NomicBert uses different config key names
                let hidden_size = get_i32("n_embd").or_else(|_| get_i32("hidden_size"))?;
                let num_hidden_layers =
                    get_i32("n_layer").or_else(|_| get_i32("num_hidden_layers"))?;
                let num_attention_heads =
                    get_i32("n_head").or_else(|_| get_i32("num_attention_heads"))?;
                let max_position_embeddings = get_i32("n_positions")
                    .or_else(|_| get_i32("max_position_embeddings"))
                    .unwrap_or(8192);
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "rotary_emb_base is a moderate float"
                )]
                let rotary_emb_base = get_f64("rotary_emb_base")
                    .map(|v| v as f32)
                    .unwrap_or(10000.0);

                Ok(Self {
                    variant,
                    hidden_size,
                    num_hidden_layers,
                    num_attention_heads,
                    max_position_embeddings,
                    rotary_emb_base,
                    layer_norm_eps,
                })
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Rotary position embeddings (RoPE)
// ---------------------------------------------------------------------------

/// Pre-computed cosine and sine tables for rotary position embeddings.
///
/// Built once at model load time for the full `max_position_embeddings`
/// length, then sliced to the actual sequence length during each forward
/// pass. This avoids recomputing the frequency table on every call.
///
/// Each table has shape `[max_seq, half_dim]`.
#[derive(Debug, Clone)]
struct RopeCache {
    /// `cos(pos * theta)` table, shape `[max_seq, half_dim]`.
    cos: Array,
    /// `sin(pos * theta)` table, shape `[max_seq, half_dim]`.
    sin: Array,
}

impl RopeCache {
    /// Build the RoPE cache for the given head dimension, base, and max
    /// sequence length.
    fn new(head_dim: i32, base: f32, max_seq: i32) -> crate::Result<Self> {
        let half_dim = head_dim / 2;

        // theta_i = base^(-2i / head_dim) for i in [0, half_dim)
        let exponents = Array::arange::<_, f32>(None, half_dim, None).map_err(mlx_err)?;
        let neg_two_over_d = Array::from_slice(&[-2.0_f32 / head_dim as f32], &[1]);
        let exponents = mlx_rs::ops::multiply(&exponents, &neg_two_over_d).map_err(mlx_err)?;
        let base_arr = Array::from_slice(&[base], &[1]);
        let theta = base_arr.power(&exponents).map_err(mlx_err)?; // [half_dim]

        // Position indices: [0, 1, ..., max_seq-1]
        let positions = Array::arange::<_, f32>(None, max_seq, None).map_err(mlx_err)?;
        let positions = mlx_rs::ops::reshape(&positions, &[max_seq, 1]).map_err(mlx_err)?;
        let theta = mlx_rs::ops::reshape(&theta, &[1, half_dim]).map_err(mlx_err)?;

        // Outer product: [max_seq, half_dim]
        let angles = mlx_rs::ops::multiply(&positions, &theta).map_err(mlx_err)?;

        let cos = angles.cos().map_err(mlx_err)?;
        let sin = angles.sin().map_err(mlx_err)?;

        // Eagerly evaluate so the tables are materialized once
        cos.eval().map_err(mlx_err)?;
        sin.eval().map_err(mlx_err)?;

        Ok(Self { cos, sin })
    }
}

/// Apply rotary position embeddings using pre-computed cos/sin tables.
///
/// Slices the cached tables to `seq_len`, reshapes to
/// `[1, 1, seq_len, half_dim]` for broadcasting, then applies the rotation:
/// `[x1*cos - x2*sin, x1*sin + x2*cos]` to each (first half, second half)
/// pair of the head dimension.
///
/// Inputs `q` and `k` are `[batch, num_heads, seq_len, head_dim]`.
fn apply_rope(q: &Array, k: &Array, cache: &RopeCache) -> crate::Result<(Array, Array)> {
    let seq_len = q.shape()[2];
    let half_dim = cache.cos.shape()[1];

    // Slice cached tables to actual sequence length: [seq_len, half_dim]
    let cos_vals = cache.cos.try_index(..seq_len).map_err(mlx_err)?;
    let sin_vals = cache.sin.try_index(..seq_len).map_err(mlx_err)?;

    // Broadcast to [1, 1, seq_len, half_dim] for batch/head broadcasting
    let cos_vals = mlx_rs::ops::reshape(&cos_vals, &[1, 1, seq_len, half_dim]).map_err(mlx_err)?;
    let sin_vals = mlx_rs::ops::reshape(&sin_vals, &[1, 1, seq_len, half_dim]).map_err(mlx_err)?;

    let rotate = |x: &Array| -> crate::Result<Array> {
        // Split x [..., head_dim] into two halves [..., half_dim] each
        let parts = mlx_rs::ops::split(x, 2, -1).map_err(mlx_err)?;
        let first = &parts[0]; // [batch, heads, seq, half_dim]
        let second = &parts[1];

        // rotated_first = first * cos - second * sin
        let fc = mlx_rs::ops::multiply(first, &cos_vals).map_err(mlx_err)?;
        let ss = mlx_rs::ops::multiply(second, &sin_vals).map_err(mlx_err)?;
        let rotated_first = mlx_rs::ops::subtract(&fc, &ss).map_err(mlx_err)?;

        // rotated_second = first * sin + second * cos
        let fs = mlx_rs::ops::multiply(first, &sin_vals).map_err(mlx_err)?;
        let sc = mlx_rs::ops::multiply(second, &cos_vals).map_err(mlx_err)?;
        let rotated_second = mlx_rs::ops::add(&fs, &sc).map_err(mlx_err)?;

        mlx_rs::ops::concatenate_axis(&[&rotated_first, &rotated_second], -1).map_err(mlx_err)
    };

    let q_rot = rotate(q)?;
    let k_rot = rotate(k)?;
    Ok((q_rot, k_rot))
}

// ---------------------------------------------------------------------------
// BERT building blocks (manual weight assignment, no derive macros)
// ---------------------------------------------------------------------------

/// BERT embeddings layer: word + optional(position + token_type) + LayerNorm.
///
/// For `ClassicBert`, position and token_type embeddings are summed with word
/// embeddings. For `NomicBert`, only word embeddings + LayerNorm are used
/// (positions are handled by RoPE in each attention layer).
#[derive(Debug)]
struct BertEmbeddings {
    word_embeddings: Array,
    /// Learned position embeddings (`ClassicBert` only).
    position_embeddings: Option<Array>,
    /// Token type embeddings (`ClassicBert` only).
    token_type_embeddings: Option<Array>,
    layer_norm_weight: Array,
    layer_norm_bias: Array,
    layer_norm_eps: f32,
}

impl BertEmbeddings {
    /// Forward pass: look up embeddings, sum, and normalize.
    fn forward(&self, input_ids: &Array, token_type_ids: &Array) -> crate::Result<Array> {
        let seq_len = input_ids.shape()[1];

        // Embedding lookups via indexing
        let mut sum = self.word_embeddings.try_index(input_ids).map_err(mlx_err)?;

        // Add position embeddings if present (ClassicBert)
        if let Some(ref pos_emb_table) = self.position_embeddings {
            let position_ids =
                Array::from_slice(&(0..seq_len).collect::<Vec<i32>>(), &[1, seq_len]);
            let pos_emb = pos_emb_table.try_index(&position_ids).map_err(mlx_err)?;
            sum = mlx_rs::ops::add(&sum, &pos_emb).map_err(mlx_err)?;
        }

        // Add token type embeddings if present (ClassicBert)
        if let Some(ref tok_emb_table) = self.token_type_embeddings {
            let tok_emb = tok_emb_table.try_index(token_type_ids).map_err(mlx_err)?;
            sum = mlx_rs::ops::add(&sum, &tok_emb).map_err(mlx_err)?;
        }

        // Layer norm
        let normed = mlx_rs::fast::layer_norm(
            &sum,
            Some(&self.layer_norm_weight),
            Some(&self.layer_norm_bias),
            self.layer_norm_eps,
        )
        .map_err(mlx_err)?;

        Ok(normed)
    }
}

/// Self-attention sub-layer within a BERT encoder layer.
///
/// For `ClassicBert`, QKV projections include bias terms and no rotary
/// encoding. For `NomicBert`, projections are unbiased and RoPE is applied
/// to Q and K after reshaping to head layout.
#[derive(Debug)]
struct BertSelfAttention {
    query_weight: Array,
    query_bias: Option<Array>,
    key_weight: Array,
    key_bias: Option<Array>,
    value_weight: Array,
    value_bias: Option<Array>,
    output_weight: Array,
    output_bias: Option<Array>,
    output_ln_weight: Array,
    output_ln_bias: Array,
    num_heads: i32,
    head_dim: i32,
    layer_norm_eps: f32,
    /// Pre-computed RoPE cos/sin tables (`NomicBert` only).
    rope_cache: Option<RopeCache>,
}

/// Compute a linear projection, optionally adding a bias.
///
/// With bias: `addmm(bias, input, weight.T)`.
/// Without bias: `matmul(input, weight.T)`.
fn linear(input: &Array, weight: &Array, bias: Option<&Array>) -> crate::Result<Array> {
    match bias {
        Some(b) => mlx_rs::ops::addmm(b, input, weight.t(), None, None).map_err(mlx_err),
        None => mlx_rs::ops::matmul(input, weight.t()).map_err(mlx_err),
    }
}

impl BertSelfAttention {
    /// Scaled dot-product multi-head attention with residual + LayerNorm.
    ///
    /// Both variants use post-norm: attention → residual → LayerNorm.
    /// NomicBert config has `prenorm: false` (same as ClassicBert).
    /// The only NomicBert differences are RoPE and no bias.
    fn forward(&self, hidden: &Array, attention_mask: &Array) -> crate::Result<Array> {
        let batch = hidden.shape()[0];
        let seq_len = hidden.shape()[1];

        // Q, K, V projections
        let q = linear(hidden, &self.query_weight, self.query_bias.as_ref())?;
        let k = linear(hidden, &self.key_weight, self.key_bias.as_ref())?;
        let v = linear(hidden, &self.value_weight, self.value_bias.as_ref())?;

        // Reshape to [batch, seq, num_heads, head_dim] then transpose to [batch, num_heads, seq, head_dim]
        let q = mlx_rs::ops::reshape(&q, &[batch, seq_len, self.num_heads, self.head_dim])
            .map_err(mlx_err)?;
        let mut q = mlx_rs::ops::transpose_axes(&q, &[0, 2, 1, 3]).map_err(mlx_err)?;

        let k = mlx_rs::ops::reshape(&k, &[batch, seq_len, self.num_heads, self.head_dim])
            .map_err(mlx_err)?;
        let mut k = mlx_rs::ops::transpose_axes(&k, &[0, 2, 1, 3]).map_err(mlx_err)?;

        let v = mlx_rs::ops::reshape(&v, &[batch, seq_len, self.num_heads, self.head_dim])
            .map_err(mlx_err)?;
        let v = mlx_rs::ops::transpose_axes(&v, &[0, 2, 1, 3]).map_err(mlx_err)?;

        // Apply RoPE for NomicBert (using pre-computed tables)
        if let Some(ref cache) = self.rope_cache {
            let (q_rot, k_rot) = apply_rope(&q, &k, cache)?;
            q = q_rot;
            k = k_rot;
        }

        // Scaled dot-product attention with mask
        let scale = (self.head_dim as f32).sqrt().recip();
        let attn_out =
            mlx_rs::fast::scaled_dot_product_attention(&q, &k, &v, scale, attention_mask)
                .map_err(mlx_err)?;

        // Reshape back: [batch, num_heads, seq, head_dim] -> [batch, seq, hidden]
        let attn_out = mlx_rs::ops::transpose_axes(&attn_out, &[0, 2, 1, 3]).map_err(mlx_err)?;
        let hidden_dim = self.num_heads * self.head_dim;
        let attn_out =
            mlx_rs::ops::reshape(&attn_out, &[batch, seq_len, hidden_dim]).map_err(mlx_err)?;

        // Output projection
        let projected = linear(&attn_out, &self.output_weight, self.output_bias.as_ref())?;

        // Residual + LayerNorm (post-norm for both variants)
        let residual = mlx_rs::ops::add(hidden, &projected).map_err(mlx_err)?;
        let normed = mlx_rs::fast::layer_norm(
            &residual,
            Some(&self.output_ln_weight),
            Some(&self.output_ln_bias),
            self.layer_norm_eps,
        )
        .map_err(mlx_err)?;

        Ok(normed)
    }
}

/// Feed-forward network sub-layer within a BERT encoder layer.
///
/// `ClassicBert`: Linear -> GELU -> Linear (all with bias).
/// `NomicBert`: Linear -> SwiGLU split -> Linear (no bias).
/// The intermediate weight for `NomicBert` is `[hidden, 2*intermediate]`;
/// the output is split into gate and value, then `SiLU(gate) * value`.
#[derive(Debug)]
struct BertFfn {
    intermediate_weight: Array,
    intermediate_bias: Option<Array>,
    output_weight: Array,
    output_bias: Option<Array>,
    output_ln_weight: Array,
    output_ln_bias: Array,
    layer_norm_eps: f32,
    /// Model variant (determines GELU vs SwiGLU activation).
    variant: ModelVariant,
}

impl BertFfn {
    /// FFN forward pass, dispatching on variant for activation function.
    ///
    /// Both variants use post-norm: FFN → residual → LayerNorm.
    /// ClassicBert: GELU activation.
    /// NomicBert: SwiGLU (value * SiLU(gate), where fc11=value, fc12=gate).
    fn forward(&self, hidden: &Array) -> crate::Result<Array> {
        // Intermediate projection
        let intermediate = linear(
            hidden,
            &self.intermediate_weight,
            self.intermediate_bias.as_ref(),
        )?;

        // Activation
        let activated = match self.variant {
            ModelVariant::ClassicBert => mlx_rs::nn::gelu(&intermediate).map_err(mlx_err)?,
            ModelVariant::NomicBert => {
                // SwiGLU: intermediate has shape [b, s, 2*inter] from [fc11; fc12]
                // fc11 = value (up), fc12 = gate
                // Output = value * SiLU(gate) = fc11(x) * SiLU(fc12(x))
                let parts = mlx_rs::ops::split(&intermediate, 2, -1).map_err(mlx_err)?;
                let value = &parts[0]; // fc11 output
                let gate = &parts[1]; // fc12 output
                let gate_activated = mlx_rs::nn::silu(gate).map_err(mlx_err)?;
                mlx_rs::ops::multiply(value, &gate_activated).map_err(mlx_err)?
            }
        };

        // Output projection
        let output = linear(&activated, &self.output_weight, self.output_bias.as_ref())?;

        // Residual + LayerNorm (post-norm for both variants)
        let residual = mlx_rs::ops::add(hidden, &output).map_err(mlx_err)?;
        let normed = mlx_rs::fast::layer_norm(
            &residual,
            Some(&self.output_ln_weight),
            Some(&self.output_ln_bias),
            self.layer_norm_eps,
        )
        .map_err(mlx_err)?;

        Ok(normed)
    }
}

/// A single BERT encoder layer (self-attention + FFN).
#[derive(Debug)]
struct BertLayer {
    attention: BertSelfAttention,
    ffn: BertFfn,
}

impl BertLayer {
    fn forward(&self, hidden: &Array, attention_mask: &Array) -> crate::Result<Array> {
        let after_attn = self.attention.forward(hidden, attention_mask)?;
        self.ffn.forward(&after_attn)
    }
}

/// Remove a weight from the map by name, returning an error if missing.
///
/// Uses `HashMap::remove` to move the `Array` out instead of cloning,
/// avoiding unnecessary GPU buffer copies.
fn take_weight(weights: &mut HashMap<String, Array>, name: &str) -> crate::Result<Array> {
    weights
        .remove(name)
        .ok_or_else(|| crate::Error::Other(anyhow::anyhow!("missing weight: {name}")))
}

/// Complete BERT model for embedding extraction.
#[derive(Debug)]
struct BertModel {
    embeddings: BertEmbeddings,
    layers: Vec<BertLayer>,
}

impl BertModel {
    /// Run the full BERT forward pass, returning hidden states `[batch, seq, hidden]`.
    fn forward(
        &self,
        input_ids: &Array,
        token_type_ids: &Array,
        attention_mask: &Array,
    ) -> crate::Result<Array> {
        let mut hidden = self.embeddings.forward(input_ids, token_type_ids)?;

        // Build padding attention mask for BERT (bidirectional, not causal).
        // Converts 0/1 mask to additive bias: mask = (1.0 - mask) * -1e9
        // broadcast to [batch, 1, 1, seq] so padding tokens get ~-inf scores.
        let ones = Array::ones::<f32>(attention_mask.shape()).map_err(mlx_err)?;
        let inverted = mlx_rs::ops::subtract(&ones, attention_mask).map_err(mlx_err)?;
        let large_neg = Array::from_slice(&[-1e9_f32], &[1]);
        let mask_bias = mlx_rs::ops::multiply(&inverted, &large_neg).map_err(mlx_err)?;

        // Expand to [batch, 1, 1, seq] for broadcasting over heads and query positions
        let mask_bias = mlx_rs::ops::expand_dims(&mask_bias, 1).map_err(mlx_err)?;
        let mask_bias = mlx_rs::ops::expand_dims(&mask_bias, 1).map_err(mlx_err)?;

        for layer in &self.layers {
            hidden = layer.forward(&hidden, &mask_bias)?;
        }

        Ok(hidden)
    }

    /// Load model weights from a safetensors `HashMap`.
    ///
    /// Uses [`take_weight`] to move arrays out of the map instead of cloning,
    /// avoiding unnecessary GPU buffer copies.
    fn from_weights(
        mut weights: HashMap<String, Array>,
        config: &BertConfig,
    ) -> crate::Result<Self> {
        let w = &mut weights;

        let embeddings = match config.variant {
            ModelVariant::ClassicBert => BertEmbeddings {
                word_embeddings: take_weight(w, "embeddings.word_embeddings.weight")?,
                position_embeddings: Some(take_weight(w, "embeddings.position_embeddings.weight")?),
                token_type_embeddings: Some(take_weight(
                    w,
                    "embeddings.token_type_embeddings.weight",
                )?),
                layer_norm_weight: take_weight(w, "embeddings.LayerNorm.weight")?,
                layer_norm_bias: take_weight(w, "embeddings.LayerNorm.bias")?,
                layer_norm_eps: config.layer_norm_eps,
            },
            ModelVariant::NomicBert => BertEmbeddings {
                word_embeddings: take_weight(w, "embeddings.word_embeddings.weight")?,
                position_embeddings: None,
                token_type_embeddings: w.remove("embeddings.token_type_embeddings.weight"),
                layer_norm_weight: take_weight(w, "emb_ln.weight")?,
                layer_norm_bias: take_weight(w, "emb_ln.bias")?,
                layer_norm_eps: config.layer_norm_eps,
            },
        };

        // Validate embedding dimension matches config
        let emb_dim = embeddings.word_embeddings.shape()[1] as i32;
        if emb_dim != config.hidden_size {
            return Err(crate::Error::Other(anyhow::anyhow!(
                "model hidden_size mismatch: config says {} but word_embeddings has dim {}",
                config.hidden_size,
                emb_dim
            )));
        }

        // Pre-compute RoPE cache once for all NomicBert layers (shared params)
        let rope_cache = if config.variant == ModelVariant::NomicBert {
            let head_dim = config.hidden_size / config.num_attention_heads;
            Some(RopeCache::new(
                head_dim,
                config.rotary_emb_base,
                config.max_position_embeddings,
            )?)
        } else {
            None
        };

        let mut layers = Vec::with_capacity(config.num_hidden_layers as usize);
        for i in 0..config.num_hidden_layers {
            let (attention, ffn) = match config.variant {
                ModelVariant::ClassicBert => {
                    let prefix = format!("encoder.layer.{i}");
                    let attention = BertSelfAttention {
                        query_weight: take_weight(
                            w,
                            &format!("{prefix}.attention.self.query.weight"),
                        )?,
                        query_bias: w.remove(&format!("{prefix}.attention.self.query.bias")),
                        key_weight: take_weight(w, &format!("{prefix}.attention.self.key.weight"))?,
                        key_bias: w.remove(&format!("{prefix}.attention.self.key.bias")),
                        value_weight: take_weight(
                            w,
                            &format!("{prefix}.attention.self.value.weight"),
                        )?,
                        value_bias: w.remove(&format!("{prefix}.attention.self.value.bias")),
                        output_weight: take_weight(
                            w,
                            &format!("{prefix}.attention.output.dense.weight"),
                        )?,
                        output_bias: w.remove(&format!("{prefix}.attention.output.dense.bias")),
                        output_ln_weight: take_weight(
                            w,
                            &format!("{prefix}.attention.output.LayerNorm.weight"),
                        )?,
                        output_ln_bias: take_weight(
                            w,
                            &format!("{prefix}.attention.output.LayerNorm.bias"),
                        )?,
                        num_heads: config.num_attention_heads,
                        head_dim: config.hidden_size / config.num_attention_heads,
                        layer_norm_eps: config.layer_norm_eps,
                        rope_cache: None,
                    };
                    let ffn = BertFfn {
                        intermediate_weight: take_weight(
                            w,
                            &format!("{prefix}.intermediate.dense.weight"),
                        )?,
                        intermediate_bias: w.remove(&format!("{prefix}.intermediate.dense.bias")),
                        output_weight: take_weight(w, &format!("{prefix}.output.dense.weight"))?,
                        output_bias: w.remove(&format!("{prefix}.output.dense.bias")),
                        output_ln_weight: take_weight(
                            w,
                            &format!("{prefix}.output.LayerNorm.weight"),
                        )?,
                        output_ln_bias: take_weight(w, &format!("{prefix}.output.LayerNorm.bias"))?,
                        layer_norm_eps: config.layer_norm_eps,
                        variant: config.variant,
                    };
                    (attention, ffn)
                }
                ModelVariant::NomicBert => {
                    let prefix = format!("encoder.layers.{i}");
                    // NomicBert uses fused QKV: Wqkv is [3*hidden, hidden]
                    // Split into three [hidden, hidden] chunks along dim 0
                    let wqkv = take_weight(w, &format!("{prefix}.attn.Wqkv.weight"))?;
                    let hidden = config.hidden_size;
                    let qkv_chunks = wqkv.split(3, 0).map_err(mlx_err)?;

                    let attention = BertSelfAttention {
                        query_weight: qkv_chunks[0].clone(),
                        query_bias: None,
                        key_weight: qkv_chunks[1].clone(),
                        key_bias: None,
                        value_weight: qkv_chunks[2].clone(),
                        value_bias: None,
                        output_weight: take_weight(w, &format!("{prefix}.attn.out_proj.weight"))?,
                        output_bias: None,
                        // NomicBert uses post-norm (prenorm: false): norm1 is after attention
                        output_ln_weight: take_weight(w, &format!("{prefix}.norm1.weight"))?,
                        output_ln_bias: take_weight(w, &format!("{prefix}.norm1.bias"))?,
                        num_heads: config.num_attention_heads,
                        head_dim: hidden / config.num_attention_heads,
                        layer_norm_eps: config.layer_norm_eps,
                        rope_cache: rope_cache.clone(),
                    };
                    // NomicBert SwiGLU: fc11 = value/up, fc12 = gate, fc2 = down
                    // Concatenate [fc11; fc12] -> [2*inter, hidden]; split in forward
                    // to compute fc11(x) * SiLU(fc12(x))
                    let fc11 = take_weight(w, &format!("{prefix}.mlp.fc11.weight"))?;
                    let fc12 = take_weight(w, &format!("{prefix}.mlp.fc12.weight"))?;
                    let gate_up =
                        mlx_rs::ops::concatenate_axis(&[&fc11, &fc12], 0).map_err(mlx_err)?;

                    let ffn = BertFfn {
                        intermediate_weight: gate_up,
                        intermediate_bias: None,
                        output_weight: take_weight(w, &format!("{prefix}.mlp.fc2.weight"))?,
                        output_bias: None,
                        output_ln_weight: take_weight(w, &format!("{prefix}.norm2.weight"))?,
                        output_ln_bias: take_weight(w, &format!("{prefix}.norm2.bias"))?,
                        layer_norm_eps: config.layer_norm_eps,
                        variant: config.variant,
                    };
                    (attention, ffn)
                }
            };

            layers.push(BertLayer { attention, ffn });
        }

        Ok(Self { embeddings, layers })
    }
}

// ---------------------------------------------------------------------------
// Public backend wrapper
// ---------------------------------------------------------------------------

/// MLX-based BERT embedding backend for Apple Silicon.
///
/// Uses Apple's MLX framework via [`mlx_rs`] for inference. MLX leverages
/// unified memory and Metal compute shaders, avoiding the CPU bottlenecks
/// (software GELU, allocation overhead, explicit copies) seen with Candle.
///
/// Supports both `ClassicBert` (BGE) and `NomicBert` (CodeRankEmbed) model
/// families, detected automatically from weight names.
///
/// The inner [`BertModel`] is wrapped in `Arc<Mutex<_>>` because MLX's
/// [`Array`] type is `Send` but not `Sync`. The mutex ensures safe access
/// from the `&self` methods required by [`EmbedBackend`].
pub struct MlxBackend {
    /// The BERT model (mutex-protected because `Array` is not `Sync`).
    model: Arc<Mutex<BertModel>>,
    /// Hidden dimension for output vector size validation.
    hidden_size: i32,
    /// Maximum sequence length supported by the model.
    max_position_embeddings: i32,
}

impl std::fmt::Debug for MlxBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MlxBackend")
            .field("hidden_size", &self.hidden_size)
            .field("max_position_embeddings", &self.max_position_embeddings)
            .finish()
    }
}

impl MlxBackend {
    /// Load a BERT/BGE/NomicBert embedding model from `HuggingFace`.
    ///
    /// Downloads `model.safetensors` and `config.json` on first call;
    /// subsequent calls use the `hf-hub` cache. The model variant
    /// (`ClassicBert` or `NomicBert`) is auto-detected from weight names.
    ///
    /// MLX always runs on the GPU via Metal -- the `device_hint` is accepted
    /// for API compatibility but ignored (MLX manages its own device placement).
    ///
    /// # Errors
    ///
    /// Returns an error if the model cannot be downloaded, the config
    /// cannot be parsed, or the weights fail to load.
    pub fn load(model_repo: &str, _device_hint: &DeviceHint) -> crate::Result<Self> {
        let api = Api::new().map_err(|e| crate::Error::Download(e.to_string()))?;
        let repo = api.model(model_repo.to_string());

        // Download config and weights
        let config_path = repo
            .get("config.json")
            .map_err(|e| crate::Error::Download(e.to_string()))?;
        let weights_path = repo
            .get("model.safetensors")
            .map_err(|e| crate::Error::Download(e.to_string()))?;

        // Load safetensors weights into MLX arrays (before config, to detect variant)
        let weights = Array::load_safetensors(weights_path).map_err(mlx_err)?;
        let variant = detect_variant(&weights);

        // Parse config with variant awareness
        let config_str = std::fs::read_to_string(&config_path).map_err(|e| crate::Error::Io {
            path: config_path.display().to_string(),
            source: e,
        })?;
        let config_json: serde_json::Value = serde_json::from_str(&config_str)
            .map_err(|e| crate::Error::Other(anyhow::anyhow!("config parse error: {e}")))?;
        let config = BertConfig::from_json(&config_json, variant)?;

        let hidden_size = config.hidden_size;
        let max_position_embeddings = config.max_position_embeddings;
        let model = BertModel::from_weights(weights, &config)?;

        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            hidden_size,
            max_position_embeddings,
        })
    }
}

impl EmbedBackend for MlxBackend {
    /// Embed a batch of pre-tokenized inputs using CLS pooling and L2
    /// normalization.
    ///
    /// Builds padded `[batch, seq]` tensors, runs the BERT forward pass on
    /// Metal via MLX, CLS-pools the output, and L2-normalizes each vector.
    ///
    /// # Errors
    ///
    /// Returns an error if tensor construction or the forward pass fails.
    fn embed_batch(&self, encodings: &[Encoding]) -> crate::Result<Vec<Vec<f32>>> {
        if encodings.is_empty() {
            return Ok(vec![]);
        }

        let batch_size = encodings.len() as i32;
        let max_len = encodings
            .iter()
            .map(|e| e.input_ids.len())
            .max()
            .unwrap_or(0) as i32;

        // Build padded tensors [batch_size, max_len]
        let mut ids_flat = vec![0i32; (batch_size * max_len) as usize];
        let mut mask_flat = vec![0.0_f32; (batch_size * max_len) as usize];
        let mut types_flat = vec![0i32; (batch_size * max_len) as usize];

        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "token IDs from tokenizer are always non-negative and fit in i32"
        )]
        for (i, enc) in encodings.iter().enumerate() {
            let offset = i * max_len as usize;
            for (j, (&id, (&mask, &typ))) in enc
                .input_ids
                .iter()
                .zip(enc.attention_mask.iter().zip(enc.token_type_ids.iter()))
                .enumerate()
            {
                ids_flat[offset + j] = id as i32;
                mask_flat[offset + j] = mask as f32;
                types_flat[offset + j] = typ as i32;
            }
        }

        let input_ids = Array::from_slice(&ids_flat, &[batch_size, max_len]);
        let attention_mask = Array::from_slice(&mask_flat, &[batch_size, max_len]);
        let token_type_ids = Array::from_slice(&types_flat, &[batch_size, max_len]);

        // Forward pass through BERT (lock model for thread-safety since Array is not Sync)
        let model = self
            .model
            .lock()
            .map_err(|e| crate::Error::Other(anyhow::anyhow!("MLX mutex poisoned: {e}")))?;
        let hidden = model.forward(&input_ids, &token_type_ids, &attention_mask)?;
        drop(model);

        // CLS pooling: take first token [batch, hidden]
        // Slice hidden[:, 0:1, :] then squeeze
        let cls = hidden.try_index((.., 0, ..)).map_err(mlx_err)?;

        // L2 normalize each vector (clamp norm to avoid NaN on zero vectors)
        let sq = cls.square().map_err(mlx_err)?;
        let norms = sq
            .sum_axis(-1, true)
            .map_err(mlx_err)?
            .sqrt()
            .map_err(mlx_err)?;
        let eps = Array::from_slice(&[1e-12_f32], &[1]);
        let norms = mlx_rs::ops::maximum(&norms, &eps).map_err(mlx_err)?;
        let normalized = mlx_rs::ops::divide(&cls, &norms).map_err(mlx_err)?;

        // Evaluate and extract to Vec<Vec<f32>>
        normalized.eval().map_err(mlx_err)?;

        let shape = normalized.shape();
        let flat: &[f32] = normalized.as_slice::<f32>();
        let hidden_dim = self.hidden_size as usize;

        let mut results = Vec::with_capacity(batch_size as usize);
        for i in 0..shape[0] as usize {
            let start = i * hidden_dim;
            results.push(flat[start..start + hidden_dim].to_vec());
        }

        Ok(results)
    }

    /// MLX manages its own parallelism -- cloning is not needed.
    fn supports_clone(&self) -> bool {
        false
    }

    /// MLX backends do not support per-thread cloning.
    ///
    /// # Panics
    ///
    /// Always panics -- callers must check [`supports_clone`](EmbedBackend::supports_clone) first.
    fn clone_backend(&self) -> Box<dyn EmbedBackend> {
        unimplemented!("clone_backend() called on MlxBackend -- MLX manages its own parallelism")
    }

    /// MLX always runs on the GPU via Metal.
    fn is_gpu(&self) -> bool {
        true
    }

    /// Maximum tokens from model config (512 for BERT, 8192 for NomicBert).
    fn max_tokens(&self) -> usize {
        self.max_position_embeddings as usize
    }
}

// NOTE: MLX tests must run single-threaded (`--test-threads=1`) because
// MLX's Metal runtime segfaults when multiple model instances run in
// parallel across test threads. This is fine for production (GPU backend
// uses single-threaded pipelined scheduler).

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Encoding;

    const BGE_SMALL: &str = "BAAI/bge-small-en-v1.5";

    #[test]
    fn mlx_backend_loads_model() {
        // Isolate: does model loading segfault?
        let _backend = MlxBackend::load(BGE_SMALL, &DeviceHint::Auto).unwrap();
    }

    #[test]
    fn mlx_backend_loads_and_embeds() {
        let backend = MlxBackend::load(BGE_SMALL, &DeviceHint::Auto).unwrap();
        let enc = Encoding {
            input_ids: vec![101, 7592, 102],
            attention_mask: vec![1, 1, 1],
            token_type_ids: vec![0, 0, 0],
        };
        let results = backend.embed_batch(&[enc]).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].len(), 384);
        let norm: f32 = results[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "L2 norm should be ~1.0, got {norm}"
        );
    }

    #[test]
    fn mlx_backend_empty_batch() {
        let backend = MlxBackend::load(BGE_SMALL, &DeviceHint::Auto).unwrap();
        let results = backend.embed_batch(&[]).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn mlx_backend_is_gpu() {
        let backend = MlxBackend::load(BGE_SMALL, &DeviceHint::Auto).unwrap();
        assert!(backend.is_gpu());
        assert!(!backend.supports_clone());
    }

    #[test]
    fn mlx_backend_batch_of_two() {
        let backend = MlxBackend::load(BGE_SMALL, &DeviceHint::Auto).unwrap();
        let enc1 = Encoding {
            input_ids: vec![101, 7592, 102],
            attention_mask: vec![1, 1, 1],
            token_type_ids: vec![0, 0, 0],
        };
        let enc2 = Encoding {
            input_ids: vec![101, 2023, 2003, 1037, 3231, 102],
            attention_mask: vec![1, 1, 1, 1, 1, 1],
            token_type_ids: vec![0, 0, 0, 0, 0, 0],
        };
        let results = backend.embed_batch(&[enc1, enc2]).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].len(), 384);
        assert_eq!(results[1].len(), 384);

        // Both should be L2 normalized
        for (i, emb) in results.iter().enumerate() {
            let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-4,
                "embedding {i}: L2 norm should be ~1.0, got {norm}"
            );
        }

        // Different inputs should produce different embeddings
        let dot: f32 = results[0]
            .iter()
            .zip(results[1].iter())
            .map(|(a, b)| a * b)
            .sum();
        assert!(
            dot < 0.999,
            "different inputs should produce different embeddings, dot={dot}"
        );
    }

    #[test]
    fn detect_variant_classic_bert() {
        // BGE model should detect as ClassicBert
        let backend = MlxBackend::load(BGE_SMALL, &DeviceHint::Auto).unwrap();
        // ClassicBert max_position_embeddings is 512
        assert_eq!(backend.max_position_embeddings, 512);
    }

    #[test]
    #[ignore = "loads CodeRankEmbed model; run with `cargo test -- --ignored`"]
    fn nomic_bert_loads_and_embeds() {
        let backend = MlxBackend::load("nomic-ai/CodeRankEmbed", &DeviceHint::Auto).unwrap();
        assert_eq!(backend.max_tokens(), 8192);
        let enc = Encoding {
            input_ids: vec![101, 7592, 102], // [CLS] hello [SEP]
            attention_mask: vec![1, 1, 1],
            token_type_ids: vec![0, 0, 0],
        };
        let results = backend.embed_batch(&[enc]).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].len(), 768); // CodeRankEmbed hidden size

        // Verify L2 norm
        let norm: f32 = results[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "L2 norm should be ~1.0, got {norm}"
        );
    }
}
