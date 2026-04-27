//! MLX embedding backend for Apple Silicon.
//!
//! Implements BERT inference using Apple's MLX framework via [`mlx_rs`].
//! MLX uses unified memory and Metal compute shaders, avoiding the CPU
//! bottlenecks (software GELU, allocation overhead, CPU-GPU copies) that
//! limit the Candle backend on Apple Silicon.
//!
//! Supports the `ClassicBert` model family (BGE models): learned position
//! embeddings, GELU activation, QKV with bias.
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

/// Convert an array to FP16 for reduced memory bandwidth on Apple Silicon.
///
/// Halves the memory footprint of weight matrices, which speeds up the
/// memory-bound matmul operations that dominate BERT inference.
fn to_fp16(arr: &Array) -> crate::Result<Array> {
    arr.as_dtype(mlx_rs::Dtype::Float16).map_err(mlx_err)
}

/// Optionally convert an array to FP16 (for `Option<Array>` fields).
fn opt_to_fp16(arr: Option<Array>) -> crate::Result<Option<Array>> {
    arr.map(|a| to_fp16(&a)).transpose()
}

// ---------------------------------------------------------------------------
// Model variant detection
// ---------------------------------------------------------------------------

/// Validate that the loaded weights are a recognized `ClassicBert` model.
///
/// Returns an error for unknown architectures (e.g. models without learned
/// position embeddings are not supported in the MLX backend).
fn detect_variant(weights: &HashMap<String, Array>) -> crate::Result<()> {
    if weights.contains_key("embeddings.position_embeddings.weight") {
        Ok(())
    } else {
        Err(crate::Error::Other(anyhow::anyhow!(
            "unknown model architecture: expected ClassicBert (embeddings.position_embeddings.weight not found)"
        )))
    }
}

// ---------------------------------------------------------------------------
// BERT model configuration
// ---------------------------------------------------------------------------

/// Configuration for a BERT-style encoder model.
///
/// Matches the `config.json` schema from `HuggingFace` model repos.
#[derive(Debug, Clone)]
struct BertConfig {
    /// Hidden dimension (384 for bge-small).
    hidden_size: i32,
    /// Number of transformer layers.
    num_hidden_layers: i32,
    /// Number of attention heads.
    num_attention_heads: i32,
    /// Maximum sequence length (512 for ClassicBert).
    max_position_embeddings: i32,
    /// Layer norm epsilon.
    layer_norm_eps: f32,
}

impl BertConfig {
    /// Parse from a `config.json` value.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "HuggingFace config ints (hidden_size, num_layers, etc.) always fit in i32"
    )]
    fn from_json(v: &serde_json::Value) -> crate::Result<Self> {
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

        Ok(Self {
            hidden_size: get_i32("hidden_size")?,
            num_hidden_layers: get_i32("num_hidden_layers")?,
            num_attention_heads: get_i32("num_attention_heads")?,
            max_position_embeddings: get_i32("max_position_embeddings").unwrap_or(512),
            layer_norm_eps,
        })
    }
}

// ---------------------------------------------------------------------------
// BERT building blocks (manual weight assignment, no derive macros)
// ---------------------------------------------------------------------------

/// BERT embeddings layer: word + position + `token_type` + `LayerNorm`.
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
/// Uses a fused QKV projection: a single `[3*hidden, hidden]` weight matrix
/// produces Q, K, V in one matmul, then splits the result. This eliminates
/// 2 kernel launches per layer (24 total for 12-layer BERT).
///
/// Projections include bias terms. No rotary position encoding.
#[derive(Debug)]
struct BertSelfAttention {
    /// Fused Q/K/V weight matrix `[3*hidden, hidden]`.
    qkv_weight: Array,
    /// Fused Q/K/V bias `[3*hidden]`.
    qkv_bias: Option<Array>,
    output_weight: Array,
    output_bias: Option<Array>,
    output_ln_weight: Array,
    output_ln_bias: Array,
    num_heads: i32,
    head_dim: i32,
    layer_norm_eps: f32,
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
    /// Scaled dot-product multi-head attention with residual + `LayerNorm`.
    ///
    /// Uses post-norm: attention → residual → `LayerNorm`.
    #[expect(
        clippy::cast_precision_loss,
        reason = "head_dim is always small (≤ 128 for BERT); i32 → f32 is lossless here"
    )]
    fn forward(&self, hidden: &Array, attention_mask: &Array) -> crate::Result<Array> {
        let batch = hidden.shape()[0];
        let seq_len = hidden.shape()[1];

        // Fused Q/K/V projection: one matmul instead of three
        let qkv = linear(hidden, &self.qkv_weight, self.qkv_bias.as_ref())?;
        let parts = mlx_rs::ops::split(&qkv, 3, -1).map_err(mlx_err)?;
        let (q, k, v) = (&parts[0], &parts[1], &parts[2]);

        // Reshape to [batch, seq, num_heads, head_dim] then transpose to [batch, num_heads, seq, head_dim]
        let q = mlx_rs::ops::reshape(q, &[batch, seq_len, self.num_heads, self.head_dim])
            .map_err(mlx_err)?;
        let q = mlx_rs::ops::transpose_axes(&q, &[0, 2, 1, 3]).map_err(mlx_err)?;

        let k = mlx_rs::ops::reshape(k, &[batch, seq_len, self.num_heads, self.head_dim])
            .map_err(mlx_err)?;
        let k = mlx_rs::ops::transpose_axes(&k, &[0, 2, 1, 3]).map_err(mlx_err)?;

        let v = mlx_rs::ops::reshape(v, &[batch, seq_len, self.num_heads, self.head_dim])
            .map_err(mlx_err)?;
        let v = mlx_rs::ops::transpose_axes(&v, &[0, 2, 1, 3]).map_err(mlx_err)?;

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
/// Linear -> GELU -> Linear (all with bias).
#[derive(Debug)]
struct BertFfn {
    intermediate_weight: Array,
    intermediate_bias: Option<Array>,
    output_weight: Array,
    output_bias: Option<Array>,
    output_ln_weight: Array,
    output_ln_bias: Array,
    layer_norm_eps: f32,
}

impl BertFfn {
    /// FFN forward pass: intermediate projection -> GELU -> output projection -> residual + `LayerNorm`.
    fn forward(&self, hidden: &Array) -> crate::Result<Array> {
        // Intermediate projection
        let intermediate = linear(
            hidden,
            &self.intermediate_weight,
            self.intermediate_bias.as_ref(),
        )?;

        // GELU activation
        let activated = mlx_rs::nn::gelu(&intermediate).map_err(mlx_err)?;

        // Output projection
        let output = linear(&activated, &self.output_weight, self.output_bias.as_ref())?;

        // Residual + LayerNorm (post-norm)
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
        // Converts 0/1 mask to additive bias: mask = (1.0 - mask) * -65504
        // broadcast to [batch, 1, 1, seq] so padding tokens get ~-inf scores.
        // Uses -65504 instead of -1e9 because FP16 max is ~65504.
        let ones = Array::ones::<f32>(attention_mask.shape())
            .map_err(mlx_err)?
            .as_dtype(mlx_rs::Dtype::Float16)
            .map_err(mlx_err)?;
        let inverted = mlx_rs::ops::subtract(&ones, attention_mask).map_err(mlx_err)?;
        let large_neg = Array::from_slice(&[-65504.0_f32], &[1])
            .as_dtype(mlx_rs::Dtype::Float16)
            .map_err(mlx_err)?;
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

        let embeddings = BertEmbeddings {
            word_embeddings: to_fp16(&take_weight(w, "embeddings.word_embeddings.weight")?)?,
            position_embeddings: Some(to_fp16(&take_weight(
                w,
                "embeddings.position_embeddings.weight",
            )?)?),
            token_type_embeddings: Some(to_fp16(&take_weight(
                w,
                "embeddings.token_type_embeddings.weight",
            )?)?),
            // LayerNorm stays FP32 — mean/variance computation needs full precision
            layer_norm_weight: take_weight(w, "embeddings.LayerNorm.weight")?,
            layer_norm_bias: take_weight(w, "embeddings.LayerNorm.bias")?,
            layer_norm_eps: config.layer_norm_eps,
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

        let mut layers = Vec::with_capacity(usize::try_from(config.num_hidden_layers).unwrap());
        for i in 0..config.num_hidden_layers {
            let prefix = format!("encoder.layer.{i}");

            // Load separate Q/K/V weights then fuse into single matrix
            let query_weight = take_weight(w, &format!("{prefix}.attention.self.query.weight"))?;
            let key_weight = take_weight(w, &format!("{prefix}.attention.self.key.weight"))?;
            let value_weight = take_weight(w, &format!("{prefix}.attention.self.value.weight"))?;
            let qkv_weight = to_fp16(
                &mlx_rs::ops::concatenate_axis(&[&query_weight, &key_weight, &value_weight], 0)
                    .map_err(mlx_err)?,
            )?;

            // Fuse biases if present (FP16)
            let query_bias = w.remove(&format!("{prefix}.attention.self.query.bias"));
            let key_bias = w.remove(&format!("{prefix}.attention.self.key.bias"));
            let value_bias = w.remove(&format!("{prefix}.attention.self.value.bias"));
            let qkv_bias = match (&query_bias, &key_bias, &value_bias) {
                (Some(qb), Some(kb), Some(vb)) => Some(to_fp16(
                    &mlx_rs::ops::concatenate_axis(&[qb, kb, vb], 0).map_err(mlx_err)?,
                )?),
                _ => None,
            };

            let attention = BertSelfAttention {
                qkv_weight,
                qkv_bias,
                output_weight: to_fp16(&take_weight(
                    w,
                    &format!("{prefix}.attention.output.dense.weight"),
                )?)?,
                output_bias: opt_to_fp16(
                    w.remove(&format!("{prefix}.attention.output.dense.bias")),
                )?,
                // LayerNorm stays FP32
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
            };
            let ffn = BertFfn {
                intermediate_weight: to_fp16(&take_weight(
                    w,
                    &format!("{prefix}.intermediate.dense.weight"),
                )?)?,
                intermediate_bias: opt_to_fp16(
                    w.remove(&format!("{prefix}.intermediate.dense.bias")),
                )?,
                output_weight: to_fp16(&take_weight(w, &format!("{prefix}.output.dense.weight"))?)?,
                output_bias: opt_to_fp16(w.remove(&format!("{prefix}.output.dense.bias")))?,
                // LayerNorm stays FP32
                output_ln_weight: take_weight(w, &format!("{prefix}.output.LayerNorm.weight"))?,
                output_ln_bias: take_weight(w, &format!("{prefix}.output.LayerNorm.bias"))?,
                layer_norm_eps: config.layer_norm_eps,
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
/// Supports the `ClassicBert` (BGE) model family.
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
            .finish_non_exhaustive()
    }
}

impl MlxBackend {
    /// Load a `ClassicBert` (BGE) embedding model from `HuggingFace`.
    ///
    /// Downloads `model.safetensors` and `config.json` on first call;
    /// subsequent calls use the `hf-hub` cache.
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

        // Load safetensors weights into MLX arrays (before config, to validate architecture)
        let weights = Array::load_safetensors(weights_path).map_err(mlx_err)?;
        detect_variant(&weights)?;

        // Parse config
        let config_str = std::fs::read_to_string(&config_path).map_err(|e| crate::Error::Io {
            path: config_path.display().to_string(),
            source: e,
        })?;
        let config_json: serde_json::Value = serde_json::from_str(&config_str)
            .map_err(|e| crate::Error::Other(anyhow::anyhow!("config parse error: {e}")))?;
        let config = BertConfig::from_json(&config_json)?;

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

/// Build padded `[batch, seq]` MLX tensors from pre-tokenized encodings.
///
/// This is pure CPU + memory work (no model needed) so it can run outside
/// the model mutex.  Returns `(input_ids, attention_mask, token_type_ids)`.
fn prepare_batch_tensors(encodings: &[Encoding]) -> (Array, Array, Array) {
    let batch_size = encodings.len();
    let max_len = encodings
        .iter()
        .map(|e| e.input_ids.len())
        .max()
        .unwrap_or(0);

    let total = batch_size * max_len;
    let mut ids_flat = vec![0i32; total];
    let mut mask_flat = vec![0.0_f32; total];
    let mut types_flat = vec![0i32; total];

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        reason = "token IDs, masks, and type IDs from tokenizer are always small non-negative values"
    )]
    for (i, enc) in encodings.iter().enumerate() {
        let offset = i * max_len;
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

    let batch_i32 = i32::try_from(batch_size).expect("batch size fits in i32");
    let len_i32 = i32::try_from(max_len).expect("sequence length fits in i32");
    let input_ids = Array::from_slice(&ids_flat, &[batch_i32, len_i32]);
    let attention_mask = Array::from_slice(&mask_flat, &[batch_i32, len_i32]);
    let token_type_ids = Array::from_slice(&types_flat, &[batch_i32, len_i32]);

    (input_ids, attention_mask, token_type_ids)
}

impl EmbedBackend for MlxBackend {
    /// Embed a batch of pre-tokenized inputs using CLS pooling and L2
    /// normalization.
    ///
    /// Tensor preparation and post-processing (CLS pooling, L2 normalize,
    /// eval, extraction) run outside the model lock. Only the forward pass
    /// holds the mutex, minimising contention.
    ///
    /// # Errors
    ///
    /// Returns an error if tensor construction or the forward pass fails.
    fn embed_batch(&self, encodings: &[Encoding]) -> crate::Result<Vec<Vec<f32>>> {
        // Sub-batch to reduce padding waste. With 128 sequences sorted by
        // descending length, a single batch pads all to the longest (~512).
        // Sub-batching into 64-sequence groups gives tighter per-group padding.
        const MLX_MAX_BATCH: usize = 64;

        if encodings.is_empty() {
            return Ok(vec![]);
        }
        if encodings.len() > MLX_MAX_BATCH {
            let mut all_results = Vec::with_capacity(encodings.len());
            for chunk in encodings.chunks(MLX_MAX_BATCH) {
                let mut results = self.embed_batch(chunk)?;
                all_results.append(&mut results);
            }
            return Ok(all_results);
        }

        // Phase 1: Tensor prep (no lock needed)
        let (input_ids, attention_mask, token_type_ids) = prepare_batch_tensors(encodings);
        // Attention mask to FP16 — float tensor used in matmul-heavy attention
        let attention_mask = attention_mask
            .as_dtype(mlx_rs::Dtype::Float16)
            .map_err(mlx_err)?;

        // Phase 2: Forward pass (lock needed)
        let hidden = {
            let model = self
                .model
                .lock()
                .map_err(|e| crate::Error::Other(anyhow::anyhow!("MLX mutex poisoned: {e}")))?;
            model.forward(&input_ids, &token_type_ids, &attention_mask)?
        }; // lock released here

        // Phase 3: Post-process (no lock needed)
        // CLS pooling: take first token [batch, hidden]
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

        // Back to FP32 for host extraction (as_slice requires matching type)
        let normalized = normalized
            .as_dtype(mlx_rs::Dtype::Float32)
            .map_err(mlx_err)?;

        // Evaluate and extract to Vec<Vec<f32>>
        normalized.eval().map_err(mlx_err)?;

        let shape = normalized.shape();
        let flat: &[f32] = normalized.as_slice::<f32>();
        let hidden_dim = usize::try_from(self.hidden_size).expect("hidden_size is positive");
        let batch_out = usize::try_from(shape[0]).expect("batch dimension is non-negative");

        let mut results = Vec::with_capacity(encodings.len());
        for i in 0..batch_out {
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

    fn name(&self) -> &'static str {
        "MLX"
    }

    /// Maximum tokens from model config (512 for `ClassicBert`).
    fn max_tokens(&self) -> usize {
        usize::try_from(self.max_position_embeddings).expect("max_position_embeddings is positive")
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
    #[ignore = "requires MLX/Metal runtime"]
    fn mlx_supports_fp16() {
        let arr = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]);
        let half = arr.as_dtype(mlx_rs::Dtype::Float16);
        assert!(
            half.is_ok(),
            "MLX doesn't support Float16: {:?}",
            half.err()
        );
    }

    #[test]
    #[ignore = "requires MLX/Metal runtime and model download"]
    fn mlx_backend_loads_model() {
        // Isolate: does model loading segfault?
        let _backend = MlxBackend::load(BGE_SMALL, &DeviceHint::Auto).unwrap();
    }

    #[test]
    #[ignore = "requires MLX/Metal runtime and model download"]
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
    #[ignore = "requires MLX/Metal runtime and model download"]
    fn mlx_backend_empty_batch() {
        let backend = MlxBackend::load(BGE_SMALL, &DeviceHint::Auto).unwrap();
        let results = backend.embed_batch(&[]).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    #[ignore = "requires MLX/Metal runtime and model download"]
    fn mlx_backend_is_gpu() {
        let backend = MlxBackend::load(BGE_SMALL, &DeviceHint::Auto).unwrap();
        assert!(backend.is_gpu());
        assert!(!backend.supports_clone());
    }

    #[test]
    #[ignore = "requires MLX/Metal runtime and model download"]
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
    #[ignore = "requires MLX/Metal runtime and model download"]
    fn detect_variant_classic_bert() {
        // BGE model should detect as ClassicBert
        let backend = MlxBackend::load(BGE_SMALL, &DeviceHint::Auto).unwrap();
        // ClassicBert max_position_embeddings is 512
        assert_eq!(backend.max_position_embeddings, 512);
    }
}
