//! CPU embedding backend using ndarray + system BLAS.
//!
//! Implements BERT inference on CPU using [`ndarray`] with system BLAS for
//! matrix operations. Weights are loaded directly from safetensors files
//! downloaded via `hf-hub`.
//!
//! Supports two model families:
//! - **`ClassicBert`** (BGE models): learned position embeddings, GELU, QKV with bias.
//! - **`NomicBert`** (`CodeRankEmbed`, nomic-embed-text): `RoPE`, `SwiGLU`, no bias.

// Ensure the BLAS linker symbols are pulled in.
extern crate blas_src;
extern crate openblas_src;

use std::sync::Arc;

use hf_hub::api::sync::Api;
use ndarray::{Array1, Array2, ArrayView1, Axis};
use safetensors::SafeTensors;

use super::{DeviceHint, EmbedBackend, Encoding};

// ---------------------------------------------------------------------------
// Model variant detection
// ---------------------------------------------------------------------------

/// Which BERT variant the loaded weights correspond to.
///
/// `ClassicBert` uses learned position embeddings, GELU activation, and
/// biased QKV projections. `NomicBert` uses `RoPE`, `SwiGLU`, and unbiased
/// projections.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelVariant {
    /// Standard BERT / BGE models (e.g. `BAAI/bge-small-en-v1.5`).
    ClassicBert,
    /// `NomicBert` models (e.g. `nomic-ai/CodeRankEmbed`, `nomic-embed-text-v1.5`).
    NomicBert,
}

/// Detect the model variant by inspecting weight names.
///
/// `ClassicBert` has `embeddings.position_embeddings.weight`; `NomicBert`
/// does not (it uses rotary position encoding instead).
fn detect_variant(tensors: &SafeTensors<'_>) -> ModelVariant {
    if tensors
        .tensor("embeddings.position_embeddings.weight")
        .is_ok()
    {
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
/// Matches the `config.json` schema from `HuggingFace` model repos.
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
    #[expect(
        clippy::cast_possible_truncation,
        reason = "config values are small ints/floats that fit in i32/f32"
    )]
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
                let hidden_size = get_i32("n_embd").or_else(|_| get_i32("hidden_size"))?;
                let num_hidden_layers =
                    get_i32("n_layer").or_else(|_| get_i32("num_hidden_layers"))?;
                let num_attention_heads =
                    get_i32("n_head").or_else(|_| get_i32("num_attention_heads"))?;
                let max_position_embeddings = get_i32("n_positions")
                    .or_else(|_| get_i32("max_position_embeddings"))
                    .unwrap_or(8192);
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
// Safetensors -> ndarray helpers
// ---------------------------------------------------------------------------

/// Load a named tensor from safetensors as `Array2<f32>`.
///
/// The tensor must be stored in `f32` (little-endian) format. Returns an
/// error if the tensor is missing or the byte count does not match the shape.
fn load_tensor2(tensors: &SafeTensors<'_>, name: &str) -> crate::Result<Array2<f32>> {
    let tensor = tensors
        .tensor(name)
        .map_err(|_| crate::Error::Other(anyhow::anyhow!("missing weight: {name}")))?;
    let shape = tensor.shape();
    if shape.len() != 2 {
        return Err(crate::Error::Other(anyhow::anyhow!(
            "expected 2D tensor for {name}, got {}D",
            shape.len()
        )));
    }
    let data: Vec<f32> = tensor
        .data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    Array2::from_shape_vec((shape[0], shape[1]), data)
        .map_err(|e| crate::Error::Other(anyhow::anyhow!("shape error for {name}: {e}")))
}

/// Load a named tensor from safetensors as `Array1<f32>`.
///
/// The tensor must be stored in `f32` (little-endian) format.
fn load_tensor1(tensors: &SafeTensors<'_>, name: &str) -> crate::Result<Array1<f32>> {
    let tensor = tensors
        .tensor(name)
        .map_err(|_| crate::Error::Other(anyhow::anyhow!("missing weight: {name}")))?;
    let shape = tensor.shape();
    if shape.len() != 1 {
        return Err(crate::Error::Other(anyhow::anyhow!(
            "expected 1D tensor for {name}, got {}D",
            shape.len()
        )));
    }
    let data: Vec<f32> = tensor
        .data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    Ok(Array1::from_vec(data))
}

/// Optionally load a 2D tensor -- returns `None` if missing.
fn try_load_tensor2(tensors: &SafeTensors<'_>, name: &str) -> crate::Result<Option<Array2<f32>>> {
    if tensors.tensor(name).is_ok() {
        Ok(Some(load_tensor2(tensors, name)?))
    } else {
        Ok(None)
    }
}

/// Optionally load a 1D tensor -- returns `None` if missing.
fn try_load_tensor1(tensors: &SafeTensors<'_>, name: &str) -> crate::Result<Option<Array1<f32>>> {
    if tensors.tensor(name).is_ok() {
        Ok(Some(load_tensor1(tensors, name)?))
    } else {
        Ok(None)
    }
}

// ---------------------------------------------------------------------------
// Layer norm helper
// ---------------------------------------------------------------------------

/// Apply layer normalization to a 1D hidden vector (single token).
///
/// Computes: `(x - mean) / sqrt(var + eps) * weight + bias`
fn layer_norm(
    x: &ArrayView1<'_, f32>,
    weight: &Array1<f32>,
    bias: &Array1<f32>,
    eps: f32,
) -> Array1<f32> {
    let mean = x.mean().unwrap_or(0.0);
    let var = x.mapv(|v| (v - mean).powi(2)).mean().unwrap_or(0.0);
    let inv_std = 1.0 / (var + eps).sqrt();
    (x.mapv(|v| (v - mean) * inv_std) * weight) + bias
}

// ---------------------------------------------------------------------------
// BERT embeddings layer
// ---------------------------------------------------------------------------

/// BERT embeddings layer: word + optional(position + `token_type`) + `LayerNorm`.
///
/// For `ClassicBert`, position and `token_type` embeddings are summed with word
/// embeddings. For `NomicBert`, only word embeddings + `LayerNorm` are used
/// (positions are handled by `RoPE` in each attention layer).
#[derive(Debug)]
struct CpuBertEmbeddings {
    /// Word embedding table `[vocab_size, hidden]`.
    word_embeddings: Array2<f32>,
    /// Learned position embeddings (`ClassicBert` only) `[max_seq, hidden]`.
    position_embeddings: Option<Array2<f32>>,
    /// Token type embeddings (`ClassicBert` only) `[2, hidden]`.
    token_type_embeddings: Option<Array2<f32>>,
    /// Layer norm weight `[hidden]`.
    layer_norm_weight: Array1<f32>,
    /// Layer norm bias `[hidden]`.
    layer_norm_bias: Array1<f32>,
    /// Layer norm epsilon.
    layer_norm_eps: f32,
}

impl CpuBertEmbeddings {
    /// Forward pass: look up embeddings, sum, and normalize.
    ///
    /// Returns one `[seq, hidden]` matrix per batch item.
    #[expect(
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        reason = "token IDs from tokenizer are always non-negative and fit in usize"
    )]
    fn forward(&self, encodings: &[Encoding]) -> Vec<Array2<f32>> {
        let hidden = self.word_embeddings.shape()[1];

        encodings
            .iter()
            .map(|enc| {
                let seq_len = enc.input_ids.len();
                let mut output = Array2::<f32>::zeros((seq_len, hidden));

                for (t, &id) in enc.input_ids.iter().enumerate() {
                    let word_row = self.word_embeddings.row(id as usize);
                    output.row_mut(t).assign(&word_row);

                    if let Some(ref pos_emb) = self.position_embeddings {
                        let pos_row = pos_emb.row(t);
                        output.row_mut(t).zip_mut_with(&pos_row, |o, &p| *o += p);
                    }

                    if let Some(ref tok_emb) = self.token_type_embeddings {
                        let type_id = enc.token_type_ids[t] as usize;
                        let tok_row = tok_emb.row(type_id);
                        output.row_mut(t).zip_mut_with(&tok_row, |o, &p| *o += p);
                    }

                    let normed = layer_norm(
                        &output.row(t),
                        &self.layer_norm_weight,
                        &self.layer_norm_bias,
                        self.layer_norm_eps,
                    );
                    output.row_mut(t).assign(&normed);
                }

                output
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Self-attention (struct only -- forward() in a later task)
// ---------------------------------------------------------------------------

/// Self-attention sub-layer within a BERT encoder layer.
///
/// Uses a fused QKV projection for both variants: a single `[3*hidden, hidden]`
/// weight matrix produces Q, K, V in one matmul, then splits the result.
///
/// For `ClassicBert`, projections include bias terms and no rotary encoding.
/// For `NomicBert`, projections are unbiased and `RoPE` is applied to Q and K.
#[expect(dead_code, reason = "scaffold -- forward() will use these fields")]
#[derive(Debug)]
struct CpuBertSelfAttention {
    /// Fused Q/K/V weight matrix `[3*hidden, hidden]`.
    qkv_weight: Array2<f32>,
    /// Fused Q/K/V bias `[3*hidden]` (`ClassicBert` only).
    qkv_bias: Option<Array1<f32>>,
    /// Output projection weight `[hidden, hidden]`.
    output_weight: Array2<f32>,
    /// Output projection bias `[hidden]` (`ClassicBert` only).
    output_bias: Option<Array1<f32>>,
    /// Post-attention `LayerNorm` weight `[hidden]`.
    output_ln_weight: Array1<f32>,
    /// Post-attention `LayerNorm` bias `[hidden]`.
    output_ln_bias: Array1<f32>,
    /// Number of attention heads.
    num_heads: i32,
    /// Dimension per head (`hidden / num_heads`).
    head_dim: i32,
    /// Layer norm epsilon.
    layer_norm_eps: f32,
    /// Rotary embedding base (`NomicBert` only).
    rotary_emb_base: Option<f32>,
}

// ---------------------------------------------------------------------------
// Feed-forward network (struct only -- forward() in a later task)
// ---------------------------------------------------------------------------

/// Feed-forward network sub-layer within a BERT encoder layer.
///
/// `ClassicBert`: Linear -> GELU -> Linear (all with bias).
/// `NomicBert`: Linear -> `SwiGLU` split -> Linear (no bias).
#[expect(dead_code, reason = "scaffold -- forward() will use these fields")]
#[derive(Debug)]
struct CpuBertFfn {
    /// Intermediate projection weight.
    ///
    /// `ClassicBert`: `[intermediate, hidden]`.
    /// `NomicBert`: `[2*intermediate, hidden]` (gate+value fused).
    intermediate_weight: Array2<f32>,
    /// Intermediate projection bias (`ClassicBert` only).
    intermediate_bias: Option<Array1<f32>>,
    /// Output projection weight `[hidden, intermediate]`.
    output_weight: Array2<f32>,
    /// Output projection bias (`ClassicBert` only).
    output_bias: Option<Array1<f32>>,
    /// Post-FFN `LayerNorm` weight `[hidden]`.
    output_ln_weight: Array1<f32>,
    /// Post-FFN `LayerNorm` bias `[hidden]`.
    output_ln_bias: Array1<f32>,
    /// Layer norm epsilon.
    layer_norm_eps: f32,
    /// Model variant (determines GELU vs `SwiGLU` activation).
    variant: ModelVariant,
}

// ---------------------------------------------------------------------------
// Encoder layer
// ---------------------------------------------------------------------------

/// A single BERT encoder layer (self-attention + FFN).
#[expect(dead_code, reason = "scaffold -- forward() will use these fields")]
#[derive(Debug)]
struct CpuBertLayer {
    /// Self-attention sub-layer.
    attention: CpuBertSelfAttention,
    /// Feed-forward sub-layer.
    ffn: CpuBertFfn,
}

// ---------------------------------------------------------------------------
// Full model
// ---------------------------------------------------------------------------

/// Complete BERT model for embedding extraction.
#[derive(Debug)]
struct CpuBertModel {
    /// Embeddings layer (word + position + `token_type` + `LayerNorm`).
    embeddings: CpuBertEmbeddings,
    /// Transformer encoder layers.
    #[expect(dead_code, reason = "scaffold -- forward() will iterate layers")]
    layers: Vec<CpuBertLayer>,
}

/// Load `ClassicBert` encoder layers from safetensors.
fn load_classic_layer(
    tensors: &SafeTensors<'_>,
    i: i32,
    config: &BertConfig,
) -> crate::Result<(CpuBertSelfAttention, CpuBertFfn)> {
    let prefix = format!("encoder.layer.{i}");

    // Load separate Q/K/V weights then fuse via concatenation
    let q_weight = load_tensor2(tensors, &format!("{prefix}.attention.self.query.weight"))?;
    let k_weight = load_tensor2(tensors, &format!("{prefix}.attention.self.key.weight"))?;
    let v_weight = load_tensor2(tensors, &format!("{prefix}.attention.self.value.weight"))?;
    let qkv_weight = ndarray::concatenate(
        Axis(0),
        &[q_weight.view(), k_weight.view(), v_weight.view()],
    )
    .map_err(|e| crate::Error::Other(anyhow::anyhow!("QKV concat error layer {i}: {e}")))?;

    // Fuse biases if present
    let q_bias = try_load_tensor1(tensors, &format!("{prefix}.attention.self.query.bias"))?;
    let k_bias = try_load_tensor1(tensors, &format!("{prefix}.attention.self.key.bias"))?;
    let v_bias = try_load_tensor1(tensors, &format!("{prefix}.attention.self.value.bias"))?;
    let qkv_bias = match (&q_bias, &k_bias, &v_bias) {
        (Some(qb), Some(kb), Some(vb)) => Some(
            ndarray::concatenate(Axis(0), &[qb.view(), kb.view(), vb.view()]).map_err(|e| {
                crate::Error::Other(anyhow::anyhow!("QKV bias concat error layer {i}: {e}"))
            })?,
        ),
        _ => None,
    };

    let attention = CpuBertSelfAttention {
        qkv_weight,
        qkv_bias,
        output_weight: load_tensor2(tensors, &format!("{prefix}.attention.output.dense.weight"))?,
        output_bias: try_load_tensor1(tensors, &format!("{prefix}.attention.output.dense.bias"))?,
        output_ln_weight: load_tensor1(
            tensors,
            &format!("{prefix}.attention.output.LayerNorm.weight"),
        )?,
        output_ln_bias: load_tensor1(
            tensors,
            &format!("{prefix}.attention.output.LayerNorm.bias"),
        )?,
        num_heads: config.num_attention_heads,
        head_dim: config.hidden_size / config.num_attention_heads,
        layer_norm_eps: config.layer_norm_eps,
        rotary_emb_base: None,
    };
    let ffn = CpuBertFfn {
        intermediate_weight: load_tensor2(tensors, &format!("{prefix}.intermediate.dense.weight"))?,
        intermediate_bias: try_load_tensor1(tensors, &format!("{prefix}.intermediate.dense.bias"))?,
        output_weight: load_tensor2(tensors, &format!("{prefix}.output.dense.weight"))?,
        output_bias: try_load_tensor1(tensors, &format!("{prefix}.output.dense.bias"))?,
        output_ln_weight: load_tensor1(tensors, &format!("{prefix}.output.LayerNorm.weight"))?,
        output_ln_bias: load_tensor1(tensors, &format!("{prefix}.output.LayerNorm.bias"))?,
        layer_norm_eps: config.layer_norm_eps,
        variant: config.variant,
    };
    Ok((attention, ffn))
}

/// Load `NomicBert` encoder layers from safetensors.
fn load_nomic_layer(
    tensors: &SafeTensors<'_>,
    i: i32,
    config: &BertConfig,
) -> crate::Result<(CpuBertSelfAttention, CpuBertFfn)> {
    let prefix = format!("encoder.layers.{i}");

    let qkv_weight = load_tensor2(tensors, &format!("{prefix}.attn.Wqkv.weight"))?;

    let attention = CpuBertSelfAttention {
        qkv_weight,
        qkv_bias: None,
        output_weight: load_tensor2(tensors, &format!("{prefix}.attn.out_proj.weight"))?,
        output_bias: None,
        output_ln_weight: load_tensor1(tensors, &format!("{prefix}.norm1.weight"))?,
        output_ln_bias: load_tensor1(tensors, &format!("{prefix}.norm1.bias"))?,
        num_heads: config.num_attention_heads,
        head_dim: config.hidden_size / config.num_attention_heads,
        layer_norm_eps: config.layer_norm_eps,
        rotary_emb_base: Some(config.rotary_emb_base),
    };

    // SwiGLU: fc11 = value/up, fc12 = gate, fc2 = down
    let fc11 = load_tensor2(tensors, &format!("{prefix}.mlp.fc11.weight"))?;
    let fc12 = load_tensor2(tensors, &format!("{prefix}.mlp.fc12.weight"))?;
    let gate_up = ndarray::concatenate(Axis(0), &[fc11.view(), fc12.view()])
        .map_err(|e| crate::Error::Other(anyhow::anyhow!("SwiGLU concat error layer {i}: {e}")))?;

    let ffn = CpuBertFfn {
        intermediate_weight: gate_up,
        intermediate_bias: None,
        output_weight: load_tensor2(tensors, &format!("{prefix}.mlp.fc2.weight"))?,
        output_bias: None,
        output_ln_weight: load_tensor1(tensors, &format!("{prefix}.norm2.weight"))?,
        output_ln_bias: load_tensor1(tensors, &format!("{prefix}.norm2.bias"))?,
        layer_norm_eps: config.layer_norm_eps,
        variant: config.variant,
    };
    Ok((attention, ffn))
}

impl CpuBertModel {
    /// Load model weights from a safetensors file.
    ///
    /// Parses all weights into `ndarray` arrays. Fuses separate Q/K/V
    /// weight matrices into a single `[3*hidden, hidden]` matrix for
    /// `ClassicBert` (matching the MLX backend pattern).
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_possible_wrap,
        reason = "hidden_size and num_layers are small positive ints from config"
    )]
    fn from_safetensors(tensors: &SafeTensors<'_>, config: &BertConfig) -> crate::Result<Self> {
        let embeddings = match config.variant {
            ModelVariant::ClassicBert => CpuBertEmbeddings {
                word_embeddings: load_tensor2(tensors, "embeddings.word_embeddings.weight")?,
                position_embeddings: Some(load_tensor2(
                    tensors,
                    "embeddings.position_embeddings.weight",
                )?),
                token_type_embeddings: Some(load_tensor2(
                    tensors,
                    "embeddings.token_type_embeddings.weight",
                )?),
                layer_norm_weight: load_tensor1(tensors, "embeddings.LayerNorm.weight")?,
                layer_norm_bias: load_tensor1(tensors, "embeddings.LayerNorm.bias")?,
                layer_norm_eps: config.layer_norm_eps,
            },
            ModelVariant::NomicBert => CpuBertEmbeddings {
                word_embeddings: load_tensor2(tensors, "embeddings.word_embeddings.weight")?,
                position_embeddings: None,
                token_type_embeddings: try_load_tensor2(
                    tensors,
                    "embeddings.token_type_embeddings.weight",
                )?,
                layer_norm_weight: load_tensor1(tensors, "emb_ln.weight")?,
                layer_norm_bias: load_tensor1(tensors, "emb_ln.bias")?,
                layer_norm_eps: config.layer_norm_eps,
            },
        };

        let emb_dim = embeddings.word_embeddings.shape()[1] as i32;
        if emb_dim != config.hidden_size {
            return Err(crate::Error::Other(anyhow::anyhow!(
                "model hidden_size mismatch: config says {} but word_embeddings has dim {}",
                config.hidden_size,
                emb_dim
            )));
        }

        let mut layers = Vec::with_capacity(config.num_hidden_layers as usize);
        for i in 0..config.num_hidden_layers {
            let (attention, ffn) = match config.variant {
                ModelVariant::ClassicBert => load_classic_layer(tensors, i, config)?,
                ModelVariant::NomicBert => load_nomic_layer(tensors, i, config)?,
            };
            layers.push(CpuBertLayer { attention, ffn });
        }

        Ok(Self { embeddings, layers })
    }
}

// ---------------------------------------------------------------------------
// Public backend wrapper
// ---------------------------------------------------------------------------

/// CPU-based BERT embedding backend using ndarray + system BLAS.
///
/// Uses [`ndarray`] with system BLAS for matrix operations. All computation
/// runs on the CPU, making this backend portable to any platform with a
/// system BLAS library (`OpenBLAS`, MKL, Accelerate).
///
/// Supports both `ClassicBert` (BGE) and `NomicBert` (`CodeRankEmbed`) model
/// families, detected automatically from weight names.
///
/// The inner model is wrapped in [`Arc`] so the backend can be cheaply
/// cloned for per-thread use in rayon.
pub struct CpuBackend {
    /// The BERT model (all weights as ndarray arrays).
    model: Arc<CpuBertModel>,
    /// Hidden dimension for output vector size validation.
    hidden_size: i32,
    /// Maximum sequence length supported by the model.
    max_position_embeddings: i32,
}

impl std::fmt::Debug for CpuBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpuBackend")
            .field("hidden_size", &self.hidden_size)
            .field("max_position_embeddings", &self.max_position_embeddings)
            .finish_non_exhaustive()
    }
}

impl CpuBackend {
    /// Load a BERT/BGE/`NomicBert` embedding model from `HuggingFace`.
    ///
    /// Downloads `model.safetensors` and `config.json` on first call;
    /// subsequent calls use the `hf-hub` cache. The model variant
    /// (`ClassicBert` or `NomicBert`) is auto-detected from weight names.
    ///
    /// # Errors
    ///
    /// Returns an error if the model cannot be downloaded, the config
    /// cannot be parsed, or the weights fail to load.
    pub fn load(model_repo: &str, _device_hint: &DeviceHint) -> crate::Result<Self> {
        let api = Api::new().map_err(|e| crate::Error::Download(e.to_string()))?;
        let repo = api.model(model_repo.to_string());

        let config_path = repo
            .get("config.json")
            .map_err(|e| crate::Error::Download(e.to_string()))?;
        let weights_path = repo
            .get("model.safetensors")
            .map_err(|e| crate::Error::Download(e.to_string()))?;

        let model_bytes = std::fs::read(&weights_path).map_err(|e| crate::Error::Io {
            path: weights_path.display().to_string(),
            source: e,
        })?;

        let tensors = SafeTensors::deserialize(&model_bytes)
            .map_err(|e| crate::Error::Other(anyhow::anyhow!("safetensors parse error: {e}")))?;
        let variant = detect_variant(&tensors);

        let config_str = std::fs::read_to_string(&config_path).map_err(|e| crate::Error::Io {
            path: config_path.display().to_string(),
            source: e,
        })?;
        let config_json: serde_json::Value = serde_json::from_str(&config_str)
            .map_err(|e| crate::Error::Other(anyhow::anyhow!("config parse error: {e}")))?;
        let config = BertConfig::from_json(&config_json, variant)?;

        let hidden_size = config.hidden_size;
        let max_position_embeddings = config.max_position_embeddings;
        let model = CpuBertModel::from_safetensors(&tensors, &config)?;

        Ok(Self {
            model: Arc::new(model),
            hidden_size,
            max_position_embeddings,
        })
    }
}

impl EmbedBackend for CpuBackend {
    /// Embed a batch of pre-tokenized inputs using CLS pooling and L2 norm.
    ///
    /// Currently runs only the embeddings layer (word + position +
    /// `token_type` + `layer_norm`). The full forward pass (attention + FFN
    /// layers) will be added in a follow-up task.
    fn embed_batch(&self, encodings: &[Encoding]) -> crate::Result<Vec<Vec<f32>>> {
        if encodings.is_empty() {
            return Ok(vec![]);
        }

        // Run embeddings layer (attention layers not yet wired up)
        let hidden_states = self.model.embeddings.forward(encodings);

        // CLS pooling + L2 normalize
        let mut results = Vec::with_capacity(encodings.len());
        for emb in &hidden_states {
            // CLS token is the first row
            let cls = emb.row(0);
            let norm = cls.mapv(|v| v * v).sum().sqrt().max(1e-12);
            let normalized: Vec<f32> = cls.iter().map(|&v| v / norm).collect();
            results.push(normalized);
        }

        Ok(results)
    }

    /// CPU backend supports cheap cloning via `Arc`.
    fn supports_clone(&self) -> bool {
        true
    }

    /// Clone the backend for per-thread use in rayon.
    fn clone_backend(&self) -> Box<dyn EmbedBackend> {
        Box::new(Self {
            model: Arc::clone(&self.model),
            hidden_size: self.hidden_size,
            max_position_embeddings: self.max_position_embeddings,
        })
    }

    /// CPU backend does not use GPU.
    fn is_gpu(&self) -> bool {
        false
    }

    /// Maximum tokens from model config (512 for BERT, 8192 for `NomicBert`).
    #[expect(
        clippy::cast_sign_loss,
        reason = "max_position_embeddings is always positive from config"
    )]
    fn max_tokens(&self) -> usize {
        self.max_position_embeddings as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BGE_SMALL: &str = "BAAI/bge-small-en-v1.5";

    #[test]
    fn config_from_json_classic() {
        let json: serde_json::Value = serde_json::json!({
            "hidden_size": 384,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "max_position_embeddings": 512,
            "layer_norm_eps": 1e-12
        });
        let config = BertConfig::from_json(&json, ModelVariant::ClassicBert).unwrap();
        assert_eq!(config.hidden_size, 384);
        assert_eq!(config.num_hidden_layers, 12);
        assert_eq!(config.num_attention_heads, 12);
        assert_eq!(config.max_position_embeddings, 512);
    }

    #[test]
    fn config_from_json_nomic() {
        let json: serde_json::Value = serde_json::json!({
            "n_embd": 768,
            "n_layer": 12,
            "n_head": 12,
            "n_positions": 8192,
            "rotary_emb_base": 10000.0,
            "layer_norm_epsilon": 1e-5
        });
        let config = BertConfig::from_json(&json, ModelVariant::NomicBert).unwrap();
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.num_hidden_layers, 12);
        assert_eq!(config.max_position_embeddings, 8192);
    }

    #[test]
    fn config_missing_key_errors() {
        let json: serde_json::Value = serde_json::json!({});
        let result = BertConfig::from_json(&json, ModelVariant::ClassicBert);
        assert!(result.is_err());
    }

    #[test]
    fn cpu_backend_loads_model() {
        let backend = CpuBackend::load(BGE_SMALL, &DeviceHint::Cpu).unwrap();
        assert_eq!(backend.hidden_size, 384);
        assert_eq!(backend.max_position_embeddings, 512);
        assert!(!backend.is_gpu());
        assert!(backend.supports_clone());
        assert_eq!(backend.max_tokens(), 512);
    }

    #[test]
    fn cpu_backend_embeddings_forward() {
        let backend = CpuBackend::load(BGE_SMALL, &DeviceHint::Cpu).unwrap();
        let enc = Encoding {
            input_ids: vec![101, 2023, 2003, 1037, 3231, 102],
            attention_mask: vec![1, 1, 1, 1, 1, 1],
            token_type_ids: vec![0, 0, 0, 0, 0, 0],
        };
        let outputs = backend.model.embeddings.forward(&[enc]);
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].shape(), &[6, 384]);
        let sum: f32 = outputs[0].iter().map(|v| v.abs()).sum();
        assert!(sum > 0.0, "embeddings output should not be all zeros");
    }

    #[test]
    fn cpu_backend_clone() {
        let backend = CpuBackend::load(BGE_SMALL, &DeviceHint::Cpu).unwrap();
        let cloned = backend.clone_backend();
        assert!(!cloned.is_gpu());
        assert!(cloned.supports_clone());
        assert_eq!(cloned.max_tokens(), 512);
    }
}
