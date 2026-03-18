//! ORT (ONNX Runtime) embedding backend.
//!
//! Wraps an ONNX model file behind the [`EmbedBackend`] trait. Model bytes are
//! shared via [`Arc`] so that per-thread sessions are cheap to create — each
//! thread gets its own [`ort::session::Session`] reading from the same
//! in-memory model.

use std::sync::Arc;

use hf_hub::api::sync::Api;
use ndarray::Axis;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::TensorRef;

use super::{DeviceHint, EmbedBackend, Encoding};

/// Convert any ort error (regardless of recovery type) into our crate error.
fn ort_err(e: impl std::fmt::Display) -> crate::Error {
    crate::Error::Ort(e.to_string())
}

/// ORT-based ONNX embedding backend.
///
/// Stores the raw ONNX model bytes in an [`Arc`] so that cloning is cheap
/// (shared bytes, independent sessions). Each call to [`embed_batch`] creates
/// a fresh [`ort::session::Session`] from the shared bytes. For hot-path usage,
/// the caller's scheduling layer (ring buffer or rayon clone) amortizes session
/// creation cost.
///
/// [`embed_batch`]: EmbedBackend::embed_batch
pub struct OrtBackend {
    /// ONNX model file contents, shared across clones.
    model_bytes: Arc<Vec<u8>>,
    /// Device hint for configuring execution providers.
    device_hint: DeviceHint,
}

impl OrtBackend {
    /// Load an ONNX embedding model from a `HuggingFace` repository.
    ///
    /// Downloads `onnx/model.onnx` on first call; subsequent calls use the
    /// `hf-hub` cache. The model file is read into memory for session creation.
    ///
    /// # Errors
    ///
    /// Returns an error if the model cannot be downloaded or read.
    pub fn load(model_repo: &str, device_hint: &DeviceHint) -> crate::Result<Self> {
        let api = Api::new().map_err(|e| crate::Error::Download(e.to_string()))?;
        let repo = api.model(model_repo.to_string());
        let model_path = repo
            .get("onnx/model.onnx")
            .map_err(|e| crate::Error::Download(e.to_string()))?;

        let model_bytes = std::fs::read(&model_path).map_err(|e| crate::Error::Io {
            path: model_path.display().to_string(),
            source: e,
        })?;

        Ok(Self {
            model_bytes: Arc::new(model_bytes),
            device_hint: *device_hint,
        })
    }

    /// Create an ONNX Runtime session from the in-memory model bytes.
    ///
    /// Configures execution providers based on the [`DeviceHint`]. Each thread
    /// should create its own session — sessions are not `Send`.
    ///
    /// # Errors
    ///
    /// Returns an error if session initialization fails.
    fn create_session(&self) -> crate::Result<Session> {
        let mut builder = Session::builder().map_err(ort_err)?;

        builder = builder
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(ort_err)?
            .with_intra_threads(1)
            .map_err(ort_err)?;

        // Configure execution provider based on device hint.
        match self.device_hint {
            DeviceHint::Cpu => {} // default CPU EP
            DeviceHint::Auto | DeviceHint::Gpu => {
                // Try platform-appropriate GPU EP, fall back to CPU.
                #[cfg(target_os = "macos")]
                {
                    builder = builder
                        .with_execution_providers([ort::ep::CoreML::default().build()])
                        .map_err(ort_err)?;
                }
                #[cfg(not(target_os = "macos"))]
                {
                    builder = builder
                        .with_execution_providers([ort::ep::CUDA::default().build()])
                        .map_err(ort_err)?;
                }
            }
        }

        let session = builder
            .commit_from_memory(&self.model_bytes)
            .map_err(ort_err)?;

        Ok(session)
    }
}

impl EmbedBackend for OrtBackend {
    /// Embed a batch of pre-tokenized inputs using CLS pooling and L2
    /// normalization.
    ///
    /// Creates a per-call ORT session, builds padded `[batch_size, max_len]`
    /// tensors, runs inference, CLS-pools the output, and L2-normalizes each
    /// embedding vector.
    ///
    /// # Errors
    ///
    /// Returns an error if tensor construction or the forward pass fails.
    fn embed_batch(&self, encodings: &[Encoding]) -> crate::Result<Vec<Vec<f32>>> {
        if encodings.is_empty() {
            return Ok(vec![]);
        }

        let mut session = self.create_session()?;

        let batch_size = encodings.len();
        let max_len = encodings
            .iter()
            .map(|e| e.input_ids.len())
            .max()
            .unwrap_or(0);

        // Build padded [batch_size, max_len] tensors as flat i64 vecs.
        let mut ids_flat = vec![0i64; batch_size * max_len];
        let mut mask_flat = vec![0i64; batch_size * max_len];
        let mut types_flat = vec![0i64; batch_size * max_len];

        for (i, enc) in encodings.iter().enumerate() {
            let offset = i * max_len;
            let len = enc.input_ids.len();
            ids_flat[offset..offset + len].copy_from_slice(&enc.input_ids);
            mask_flat[offset..offset + len].copy_from_slice(&enc.attention_mask);
            types_flat[offset..offset + len].copy_from_slice(&enc.token_type_ids);
        }

        let ids =
            ndarray::Array2::from_shape_vec((batch_size, max_len), ids_flat).map_err(ort_err)?;
        let mask =
            ndarray::Array2::from_shape_vec((batch_size, max_len), mask_flat).map_err(ort_err)?;
        let types =
            ndarray::Array2::from_shape_vec((batch_size, max_len), types_flat).map_err(ort_err)?;

        let ids_ref = TensorRef::from_array_view(&ids).map_err(ort_err)?;
        let mask_ref = TensorRef::from_array_view(&mask).map_err(ort_err)?;
        let types_ref = TensorRef::from_array_view(&types).map_err(ort_err)?;

        let outputs = session
            .run(ort::inputs![
                "input_ids" => ids_ref,
                "attention_mask" => mask_ref,
                "token_type_ids" => types_ref,
            ])
            .map_err(ort_err)?;

        // Output shape: [batch_size, seq_len, hidden_dim]
        // CLS pooling: take first token (index 0) of each sequence.
        let output = &outputs[0];
        let array = output.try_extract_array::<f32>().map_err(ort_err)?;

        let mut results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let seq = array.index_axis(Axis(0), i);
            let cls = seq.index_axis(Axis(0), 0); // first token = CLS
            let embedding: Vec<f32> = cls.iter().copied().collect();
            results.push(l2_normalize(&embedding));
        }

        Ok(results)
    }

    /// ORT backends support cloning (model bytes shared via `Arc`).
    fn supports_clone(&self) -> bool {
        true
    }

    /// Create a cheap clone sharing the same model bytes.
    fn clone_backend(&self) -> Box<dyn EmbedBackend> {
        Box::new(Self {
            model_bytes: Arc::clone(&self.model_bytes),
            device_hint: self.device_hint,
        })
    }

    /// Whether inference runs on a GPU.
    ///
    /// Returns `true` for GPU device hints (`CoreML` or CUDA), `false` for CPU.
    fn is_gpu(&self) -> bool {
        matches!(self.device_hint, DeviceHint::Gpu)
    }
}

/// L2-normalize a vector. Returns the zero vector if norm is zero.
fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Encoding;

    #[test]
    fn ort_backend_loads_and_embeds() {
        let backend = OrtBackend::load("BAAI/bge-small-en-v1.5", &DeviceHint::Cpu).unwrap();
        let enc = Encoding {
            input_ids: vec![101, 7592, 102],
            attention_mask: vec![1, 1, 1],
            token_type_ids: vec![0, 0, 0],
        };
        let results = backend.embed_batch(&[enc]).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].len(), 384);
        let norm: f32 = results[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4);
    }

    #[test]
    fn ort_backend_empty_batch() {
        let backend = OrtBackend::load("BAAI/bge-small-en-v1.5", &DeviceHint::Cpu).unwrap();
        let results = backend.embed_batch(&[]).unwrap();
        assert!(results.is_empty());
    }
}
