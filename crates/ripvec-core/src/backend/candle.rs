//! Candle embedding backend (pure-Rust, CPU + Metal + CUDA).
//!
//! Wraps a [`BertModel`] behind the [`EmbedBackend`] trait. The model weights
//! are shared via [`Arc`] so that CPU clones are cheap (one copy of the
//! weights, many inference threads).

use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use hf_hub::api::sync::Api;

use super::{DeviceHint, EmbedBackend, Encoding};

/// Candle-based BERT embedding backend.
///
/// Supports CPU, Metal (macOS), and CUDA (Linux/Windows) inference.
/// The inner [`BertModel`] is wrapped in [`Arc`] so
/// [`clone_backend`](EmbedBackend::clone_backend) is a cheap pointer bump on
/// CPU.
pub struct CandleBackend {
    /// The candle BERT model (shared via Arc for cheap cloning).
    model: Arc<BertModel>,
    /// The device the model lives on.
    device: Device,
}

impl CandleBackend {
    /// Load a BERT/BGE embedding model from `HuggingFace`.
    ///
    /// Downloads `model.safetensors` and `config.json` on first call;
    /// subsequent calls use the `hf-hub` cache. The `device_hint` selects
    /// the inference device (CPU, Metal, or CUDA).
    ///
    /// # Errors
    ///
    /// Returns an error if the model cannot be downloaded, the config
    /// cannot be parsed, or the weights fail to load.
    pub fn load(model_repo: &str, device_hint: &DeviceHint) -> crate::Result<Self> {
        let device = match device_hint {
            DeviceHint::Cpu => Device::Cpu,
            DeviceHint::Auto | DeviceHint::Gpu => Self::select_gpu_or_cpu(),
        };

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
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)? };

        let model = BertModel::load(vb, &config)?;

        Ok(Self {
            model: Arc::new(model),
            device,
        })
    }

    /// Try to select a GPU device, falling back to CPU if unavailable.
    fn select_gpu_or_cpu() -> Device {
        // Try Metal first (macOS), then CUDA (Linux/Windows), then CPU.
        #[cfg(feature = "metal")]
        {
            if let Ok(dev) = Device::new_metal(0) {
                return dev;
            }
        }
        #[cfg(feature = "cuda")]
        {
            if let Ok(dev) = Device::new_cuda(0) {
                return dev;
            }
        }
        Device::Cpu
    }
}

impl EmbedBackend for CandleBackend {
    /// Embed a batch of pre-tokenized inputs using CLS pooling and L2
    /// normalization.
    ///
    /// # Errors
    ///
    /// Returns an error if tensor construction or the forward pass fails.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "token IDs from tokenizer are always non-negative and fit in u32"
    )]
    fn embed_batch(&self, encodings: &[Encoding]) -> crate::Result<Vec<Vec<f32>>> {
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

        let input_ids = Tensor::from_vec(ids_flat, (batch_size, max_len), &self.device)?;
        let token_type_ids = Tensor::from_vec(types_flat, (batch_size, max_len), &self.device)?;
        let attention_mask = Tensor::from_vec(mask_flat, (batch_size, max_len), &self.device)?;

        // Forward pass through BERT
        let embeddings = self
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

    /// Candle CPU backends support cheap cloning (shared `Arc<BertModel>`).
    fn supports_clone(&self) -> bool {
        matches!(self.device, Device::Cpu)
    }

    /// Create a cheap clone sharing the same `Arc<BertModel>`.
    ///
    /// # Panics
    ///
    /// Panics if called on a GPU backend where `supports_clone()` is `false`.
    fn clone_backend(&self) -> Box<dyn EmbedBackend> {
        assert!(
            self.supports_clone(),
            "clone_backend() called on GPU CandleBackend"
        );
        Box::new(Self {
            model: Arc::clone(&self.model),
            device: self.device.clone(),
        })
    }

    /// Whether inference runs on a GPU (Metal or CUDA).
    fn is_gpu(&self) -> bool {
        !matches!(self.device, Device::Cpu)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BGE_SMALL: &str = "BAAI/bge-small-en-v1.5";

    #[test]
    fn candle_backend_loads_and_embeds() {
        let backend = CandleBackend::load(BGE_SMALL, &DeviceHint::Cpu).unwrap();
        let enc = Encoding {
            input_ids: vec![101, 7592, 102],
            attention_mask: vec![1, 1, 1],
            token_type_ids: vec![0, 0, 0],
        };
        let results = backend.embed_batch(&[enc]).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].len(), 384);

        // Verify L2 norm is approximately 1.0
        let norm: f32 = results[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "L2 norm should be ~1.0, got {norm}"
        );
    }

    #[test]
    fn candle_backend_supports_clone_on_cpu() {
        let backend = CandleBackend::load(BGE_SMALL, &DeviceHint::Cpu).unwrap();
        assert!(backend.supports_clone());
        assert!(!backend.is_gpu());
    }

    #[test]
    fn candle_backend_empty_batch() {
        let backend = CandleBackend::load(BGE_SMALL, &DeviceHint::Cpu).unwrap();
        let results = backend.embed_batch(&[]).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn candle_backend_clone_produces_same_results() {
        let backend = CandleBackend::load(BGE_SMALL, &DeviceHint::Cpu).unwrap();
        let cloned = backend.clone_backend();

        let enc = Encoding {
            input_ids: vec![101, 7592, 102],
            attention_mask: vec![1, 1, 1],
            token_type_ids: vec![0, 0, 0],
        };

        let results_orig = backend.embed_batch(&[enc.clone()]).unwrap();
        let results_cloned = cloned.embed_batch(&[enc]).unwrap();

        assert_eq!(results_orig.len(), results_cloned.len());
        for (a, b) in results_orig[0].iter().zip(results_cloned[0].iter()) {
            assert!((a - b).abs() < 1e-6, "embeddings should match: {a} vs {b}");
        }
    }
}
