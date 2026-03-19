//! MLX embedding backend for Apple Silicon.
//!
//! Implements BERT inference using Apple's MLX framework via [`mlx_rs`].
//! MLX uses unified memory and Metal compute shaders, avoiding the CPU
//! bottlenecks (software GELU, allocation overhead, CPU↔GPU copies) that
//! limit the Candle backend on Apple Silicon.
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
// BERT model configuration
// ---------------------------------------------------------------------------

/// Configuration for a BERT-style encoder model.
///
/// Matches the `config.json` schema from HuggingFace model repos.
#[derive(Debug, Clone)]
struct BertConfig {
    /// Hidden dimension (384 for bge-small).
    hidden_size: i32,
    /// Number of transformer layers.
    num_hidden_layers: i32,
    /// Number of attention heads.
    num_attention_heads: i32,
    /// Layer norm epsilon.
    layer_norm_eps: f32,
}

impl BertConfig {
    /// Parse from a `config.json` value.
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
            layer_norm_eps,
        })
    }
}

// ---------------------------------------------------------------------------
// BERT building blocks (manual weight assignment, no derive macros)
// ---------------------------------------------------------------------------

/// BERT embeddings layer: word + position + token_type + LayerNorm.
#[derive(Debug)]
struct BertEmbeddings {
    word_embeddings: Array,
    position_embeddings: Array,
    token_type_embeddings: Array,
    layer_norm_weight: Array,
    layer_norm_bias: Array,
    layer_norm_eps: f32,
}

impl BertEmbeddings {
    /// Forward pass: look up embeddings, sum, and normalize.
    fn forward(&self, input_ids: &Array, token_type_ids: &Array) -> crate::Result<Array> {
        let seq_len = input_ids.shape()[1];

        // Position IDs: [0, 1, ..., seq_len-1]
        let position_ids = Array::from_slice(&(0..seq_len).collect::<Vec<i32>>(), &[1, seq_len]);

        // Embedding lookups via indexing
        let word_emb = self.word_embeddings.try_index(input_ids).map_err(mlx_err)?;
        let pos_emb = self
            .position_embeddings
            .try_index(&position_ids)
            .map_err(mlx_err)?;
        let tok_emb = self
            .token_type_embeddings
            .try_index(token_type_ids)
            .map_err(mlx_err)?;

        // Sum embeddings
        let sum = mlx_rs::ops::add(&word_emb, &pos_emb).map_err(mlx_err)?;
        let sum = mlx_rs::ops::add(&sum, &tok_emb).map_err(mlx_err)?;

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
#[derive(Debug)]
struct BertSelfAttention {
    query_weight: Array,
    query_bias: Array,
    key_weight: Array,
    key_bias: Array,
    value_weight: Array,
    value_bias: Array,
    output_weight: Array,
    output_bias: Array,
    output_ln_weight: Array,
    output_ln_bias: Array,
    num_heads: i32,
    head_dim: i32,
    layer_norm_eps: f32,
}

impl BertSelfAttention {
    /// Scaled dot-product multi-head attention with residual + LayerNorm.
    fn forward(&self, hidden: &Array, attention_mask: &Array) -> crate::Result<Array> {
        let batch = hidden.shape()[0];
        let seq_len = hidden.shape()[1];

        // Q, K, V projections: [batch, seq, hidden] × [hidden, hidden]^T + bias
        let q = mlx_rs::ops::addmm(&self.query_bias, hidden, self.query_weight.t(), None, None)
            .map_err(mlx_err)?;
        let k = mlx_rs::ops::addmm(&self.key_bias, hidden, self.key_weight.t(), None, None)
            .map_err(mlx_err)?;
        let v = mlx_rs::ops::addmm(&self.value_bias, hidden, self.value_weight.t(), None, None)
            .map_err(mlx_err)?;

        // Reshape to [batch, seq, num_heads, head_dim] then transpose to [batch, num_heads, seq, head_dim]
        let q = mlx_rs::ops::reshape(&q, &[batch, seq_len, self.num_heads, self.head_dim])
            .map_err(mlx_err)?;
        let q = mlx_rs::ops::transpose_axes(&q, &[0, 2, 1, 3]).map_err(mlx_err)?;

        let k = mlx_rs::ops::reshape(&k, &[batch, seq_len, self.num_heads, self.head_dim])
            .map_err(mlx_err)?;
        let k = mlx_rs::ops::transpose_axes(&k, &[0, 2, 1, 3]).map_err(mlx_err)?;

        let v = mlx_rs::ops::reshape(&v, &[batch, seq_len, self.num_heads, self.head_dim])
            .map_err(mlx_err)?;
        let v = mlx_rs::ops::transpose_axes(&v, &[0, 2, 1, 3]).map_err(mlx_err)?;

        // Scaled dot-product attention with mask
        // MLX's fast SDPA dispatches to optimized Metal kernels.
        let scale = (self.head_dim as f32).sqrt().recip();
        let attn_out =
            mlx_rs::fast::scaled_dot_product_attention(&q, &k, &v, scale, attention_mask)
                .map_err(mlx_err)?;

        // Reshape back: [batch, num_heads, seq, head_dim] → [batch, seq, hidden]
        let attn_out = mlx_rs::ops::transpose_axes(&attn_out, &[0, 2, 1, 3]).map_err(mlx_err)?;
        let hidden_dim = self.num_heads * self.head_dim;
        let attn_out =
            mlx_rs::ops::reshape(&attn_out, &[batch, seq_len, hidden_dim]).map_err(mlx_err)?;

        // Output projection
        let projected = mlx_rs::ops::addmm(
            &self.output_bias,
            &attn_out,
            self.output_weight.t(),
            None,
            None,
        )
        .map_err(mlx_err)?;

        // Residual + LayerNorm
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
#[derive(Debug)]
struct BertFfn {
    intermediate_weight: Array,
    intermediate_bias: Array,
    output_weight: Array,
    output_bias: Array,
    output_ln_weight: Array,
    output_ln_bias: Array,
    layer_norm_eps: f32,
}

impl BertFfn {
    /// FFN: Linear → GELU → Linear → residual → LayerNorm.
    fn forward(&self, hidden: &Array) -> crate::Result<Array> {
        // Intermediate: [batch, seq, hidden] → [batch, seq, intermediate]
        let intermediate = mlx_rs::ops::addmm(
            &self.intermediate_bias,
            hidden,
            self.intermediate_weight.t(),
            None,
            None,
        )
        .map_err(mlx_err)?;

        // GELU activation (Metal compute shader, not software erff)
        let activated = mlx_rs::nn::gelu(&intermediate).map_err(mlx_err)?;

        // Output: [batch, seq, intermediate] → [batch, seq, hidden]
        let output = mlx_rs::ops::addmm(
            &self.output_bias,
            &activated,
            self.output_weight.t(),
            None,
            None,
        )
        .map_err(mlx_err)?;

        // Residual + LayerNorm
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

        // Build causal-style attention mask for BERT.
        // BERT uses: mask = (1.0 - mask) * -1e9 broadcast to [batch, 1, 1, seq]
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
    fn from_weights(weights: HashMap<String, Array>, config: &BertConfig) -> crate::Result<Self> {
        let get = |name: &str| -> crate::Result<Array> {
            weights
                .get(name)
                .cloned()
                .ok_or_else(|| crate::Error::Other(anyhow::anyhow!("missing weight: {name}")))
        };

        let embeddings = BertEmbeddings {
            word_embeddings: get("embeddings.word_embeddings.weight")?,
            position_embeddings: get("embeddings.position_embeddings.weight")?,
            token_type_embeddings: get("embeddings.token_type_embeddings.weight")?,
            layer_norm_weight: get("embeddings.LayerNorm.weight")?,
            layer_norm_bias: get("embeddings.LayerNorm.bias")?,
            layer_norm_eps: config.layer_norm_eps,
        };

        let mut layers = Vec::with_capacity(config.num_hidden_layers as usize);
        for i in 0..config.num_hidden_layers {
            let prefix = format!("encoder.layer.{i}");

            let attention = BertSelfAttention {
                query_weight: get(&format!("{prefix}.attention.self.query.weight"))?,
                query_bias: get(&format!("{prefix}.attention.self.query.bias"))?,
                key_weight: get(&format!("{prefix}.attention.self.key.weight"))?,
                key_bias: get(&format!("{prefix}.attention.self.key.bias"))?,
                value_weight: get(&format!("{prefix}.attention.self.value.weight"))?,
                value_bias: get(&format!("{prefix}.attention.self.value.bias"))?,
                output_weight: get(&format!("{prefix}.attention.output.dense.weight"))?,
                output_bias: get(&format!("{prefix}.attention.output.dense.bias"))?,
                output_ln_weight: get(&format!("{prefix}.attention.output.LayerNorm.weight"))?,
                output_ln_bias: get(&format!("{prefix}.attention.output.LayerNorm.bias"))?,
                num_heads: config.num_attention_heads,
                head_dim: config.hidden_size / config.num_attention_heads,
                layer_norm_eps: config.layer_norm_eps,
            };

            let ffn = BertFfn {
                intermediate_weight: get(&format!("{prefix}.intermediate.dense.weight"))?,
                intermediate_bias: get(&format!("{prefix}.intermediate.dense.bias"))?,
                output_weight: get(&format!("{prefix}.output.dense.weight"))?,
                output_bias: get(&format!("{prefix}.output.dense.bias"))?,
                output_ln_weight: get(&format!("{prefix}.output.LayerNorm.weight"))?,
                output_ln_bias: get(&format!("{prefix}.output.LayerNorm.bias"))?,
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
/// The inner [`BertModel`] is wrapped in `Arc<Mutex<_>>` because MLX's
/// [`Array`] type is `Send` but not `Sync`. The mutex ensures safe access
/// from the `&self` methods required by [`EmbedBackend`].
pub struct MlxBackend {
    /// The BERT model (mutex-protected because `Array` is not `Sync`).
    model: Arc<Mutex<BertModel>>,
    /// Hidden dimension for output vector size validation.
    hidden_size: i32,
}

impl std::fmt::Debug for MlxBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MlxBackend")
            .field("hidden_size", &self.hidden_size)
            .finish()
    }
}

impl MlxBackend {
    /// Load a BERT/BGE embedding model from `HuggingFace`.
    ///
    /// Downloads `model.safetensors` and `config.json` on first call;
    /// subsequent calls use the `hf-hub` cache. MLX always runs on the
    /// GPU via Metal — the `device_hint` is accepted for API compatibility
    /// but ignored (MLX manages its own device placement).
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

        // Parse config
        let config_str = std::fs::read_to_string(&config_path).map_err(|e| crate::Error::Io {
            path: config_path.display().to_string(),
            source: e,
        })?;
        let config_json: serde_json::Value = serde_json::from_str(&config_str)
            .map_err(|e| crate::Error::Other(anyhow::anyhow!("config parse error: {e}")))?;
        let config = BertConfig::from_json(&config_json)?;

        // Load safetensors weights into MLX arrays
        let weights = Array::load_safetensors(weights_path).map_err(mlx_err)?;

        let hidden_size = config.hidden_size;
        let model = BertModel::from_weights(weights, &config)?;

        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            hidden_size,
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

    /// MLX manages its own parallelism — cloning is not needed.
    fn supports_clone(&self) -> bool {
        false
    }

    /// MLX backends do not support per-thread cloning.
    ///
    /// # Panics
    ///
    /// Always panics — callers must check [`supports_clone`](EmbedBackend::supports_clone) first.
    fn clone_backend(&self) -> Box<dyn EmbedBackend> {
        panic!("clone_backend() called on MlxBackend — MLX manages its own parallelism");
    }

    /// MLX always runs on the GPU via Metal.
    fn is_gpu(&self) -> bool {
        true
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
}
