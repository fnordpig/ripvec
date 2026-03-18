//! Embedding model using candle (pure Rust ML framework).
//!
//! Loads BERT/BGE models from `HuggingFace` in safetensors format.
//! Supports CPU, Metal (macOS GPU), and CUDA (NVIDIA GPU) via feature flags.

use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use hf_hub::api::sync::Api;

/// Inference device selection.
#[derive(Debug, Clone, Copy, Default)]
pub enum DeviceKind {
    /// CPU inference (works everywhere).
    #[default]
    Cpu,
    /// Apple Metal GPU (macOS).
    Metal,
    /// NVIDIA CUDA GPU.
    Cuda,
}

/// Select the candle [`Device`] based on the requested kind.
///
/// # Errors
///
/// Returns an error if the requested device is not available or
/// the required feature flag was not enabled at compile time.
pub fn select_device(kind: DeviceKind) -> crate::Result<Device> {
    match kind {
        DeviceKind::Cpu => Ok(Device::Cpu),
        #[cfg(feature = "metal")]
        DeviceKind::Metal => Ok(Device::new_metal(0)?),
        #[cfg(not(feature = "metal"))]
        DeviceKind::Metal => Err(crate::Error::Other(anyhow::anyhow!(
            "Metal support requires: cargo build --features metal"
        ))),
        #[cfg(feature = "cuda")]
        DeviceKind::Cuda => Ok(Device::new_cuda(0)?),
        #[cfg(not(feature = "cuda"))]
        DeviceKind::Cuda => Err(crate::Error::Other(anyhow::anyhow!(
            "CUDA support requires: cargo build --features cuda"
        ))),
    }
}

/// An embedding model backed by candle's BERT implementation.
///
/// The inner [`BertModel`] is wrapped in [`Arc`] so that [`Clone`] is cheap
/// (shared weights, no data copy). CPU parallelism clones one model per
/// rayon thread.
#[derive(Clone)]
pub struct EmbeddingModel {
    /// The candle BERT model (shared via Arc for cheap cloning).
    model: Arc<BertModel>,
    /// The device the model lives on.
    device: Device,
}

impl EmbeddingModel {
    /// Load a BERT/BGE embedding model from `HuggingFace`.
    ///
    /// Downloads `model.safetensors` and `config.json` on first call;
    /// subsequent calls use the `hf-hub` cache.
    ///
    /// # Errors
    ///
    /// Returns an error if the model cannot be downloaded, the config
    /// cannot be parsed, or the weights fail to load.
    pub fn load(model_repo: &str, device: &Device) -> crate::Result<Self> {
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
        let config: Config = serde_json::from_str(&config_str)
            .map_err(|e| crate::Error::Other(anyhow::anyhow!("config parse error: {e}")))?;

        // Load weights via mmap (zero-copy, handled by candle internally).
        // SAFETY: The safetensors file is read-only and not modified while mapped.
        #[expect(unsafe_code, reason = "mmap of read-only safetensors model file")]
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, device)? };

        let model = BertModel::load(vb, &config)?;

        Ok(Self {
            model: Arc::new(model),
            device: device.clone(),
        })
    }

    /// Get a reference to the device.
    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }
}

/// Pre-tokenized encoding ready for inference.
pub struct Encoding {
    /// Token IDs.
    pub input_ids: Vec<i64>,
    /// Attention mask (1 for real tokens, 0 for padding).
    pub attention_mask: Vec<i64>,
    /// Token type IDs (0 for single-sequence models).
    pub token_type_ids: Vec<i64>,
}

/// Embed a batch of tokenized inputs, returning L2-normalized vectors.
///
/// Uses CLS pooling (first token embedding) for BGE models.
///
/// # Errors
///
/// Returns an error if tensor construction or the forward pass fails.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "token IDs from tokenizer are always non-negative and fit in u32"
)]
pub fn embed_batch(model: &EmbeddingModel, encodings: &[Encoding]) -> crate::Result<Vec<Vec<f32>>> {
    if encodings.is_empty() {
        return Ok(vec![]);
    }

    let batch_size = encodings.len();
    let max_len = encodings
        .iter()
        .map(|e| e.input_ids.len())
        .max()
        .unwrap_or(0);

    // Build padded tensors [batch_size, max_len]
    let mut ids_flat = vec![0u32; batch_size * max_len];
    let mut mask_flat = vec![0u32; batch_size * max_len];
    let mut types_flat = vec![0u32; batch_size * max_len];

    for (i, enc) in encodings.iter().enumerate() {
        let offset = i * max_len;
        let len = enc.input_ids.len();
        for j in 0..len {
            ids_flat[offset + j] = enc.input_ids[j] as u32;
            mask_flat[offset + j] = enc.attention_mask[j] as u32;
            types_flat[offset + j] = enc.token_type_ids[j] as u32;
        }
    }

    let device = &model.device;
    let input_ids = Tensor::from_vec(ids_flat, (batch_size, max_len), device)?;
    let token_type_ids = Tensor::from_vec(types_flat, (batch_size, max_len), device)?;
    let attention_mask = Tensor::from_vec(mask_flat, (batch_size, max_len), device)?;

    // Forward pass through BERT
    let embeddings = model
        .model
        .forward(&input_ids, &token_type_ids, Some(&attention_mask))?;

    // CLS pooling: take first token [batch_size, hidden_dim]
    let cls = embeddings.narrow(1, 0, 1)?.squeeze(1)?;

    // L2 normalize
    let norms = cls.sqr()?.sum_keepdim(1)?.sqrt()?;
    let normalized = cls.broadcast_div(&norms)?;

    // Extract to Vec<Vec<f32>>
    let data = normalized.to_vec2::<f32>()?;

    Ok(data)
}

/// Embed a single input (convenience wrapper).
///
/// # Errors
///
/// Returns an error if tensor construction or the forward pass fails.
pub fn embed(
    model: &EmbeddingModel,
    input_ids: &[i64],
    attention_mask: &[i64],
    token_type_ids: &[i64],
) -> crate::Result<Vec<f32>> {
    let enc = Encoding {
        input_ids: input_ids.to_vec(),
        attention_mask: attention_mask.to_vec(),
        token_type_ids: token_type_ids.to_vec(),
    };
    let mut results = embed_batch(model, &[enc])?;
    Ok(results.pop().unwrap_or_default())
}

#[cfg(test)]
mod tests {
    #[test]
    fn l2_normalize_unit_vector() {
        let v = [1.0f32, 0.0, 0.0];
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn l2_normalize_arbitrary_vector() {
        let v = [3.0f32, 4.0];
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let normalized: Vec<f32> = v.iter().map(|x| x / norm).collect();
        let result_norm: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((result_norm - 1.0).abs() < 1e-6);
    }
}
