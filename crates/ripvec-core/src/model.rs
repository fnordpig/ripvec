//! ONNX embedding model loading and inference.
//!
//! Downloads model weights from `HuggingFace` (<https://huggingface.co>),
//! memory-maps the ONNX file for zero-copy access, and provides thread-safe
//! embedding inference with CLS pooling and L2 normalization.
//!
//! # Thread safety
//!
//! [`EmbeddingModel`] stores a memory-mapped model file. Call
//! [`create_session`](EmbeddingModel::create_session) to obtain a per-thread
//! [`ort::session::InMemorySession`] that reads directly from the mmap.
//! Each thread gets its own session, enabling fully parallel inference.

use std::fs::File;
use std::path::PathBuf;

use hf_hub::api::sync::Api;
use memmap2::Mmap;
use ndarray::Axis;
use ort::session::InMemorySession;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::TensorRef;

/// Get a platform-appropriate cache directory for ripvec data.
fn dirs_path(app: &str, sub: &str) -> Option<PathBuf> {
    let base = dirs_base()?;
    let path = base.join(app).join(sub);
    std::fs::create_dir_all(&path).ok()?;
    Some(path)
}

/// Platform cache base directory.
fn dirs_base() -> Option<PathBuf> {
    #[cfg(target_os = "macos")]
    {
        dirs_macos()
    }
    #[cfg(not(target_os = "macos"))]
    {
        std::env::var("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .ok()
            .or_else(|| {
                std::env::var("HOME")
                    .map(|h| PathBuf::from(h).join(".cache"))
                    .ok()
            })
    }
}

#[cfg(target_os = "macos")]
fn dirs_macos() -> Option<PathBuf> {
    std::env::var("HOME")
        .map(|h| PathBuf::from(h).join("Library").join("Caches"))
        .ok()
}

/// Inference device for ONNX Runtime execution providers.
#[derive(Debug, Clone, Copy, Default)]
pub enum Device {
    /// CPU inference (works everywhere).
    #[default]
    Cpu,
    /// Apple `CoreML` (Neural Engine + GPU on macOS).
    CoreML,
    /// NVIDIA CUDA GPU.
    Cuda,
}

/// An embedding model backed by a memory-mapped ONNX file.
///
/// The model file is mmap'd once; per-thread sessions read directly from
/// the mapped memory (zero-copy via `commit_from_memory_directly`).
pub struct EmbeddingModel {
    mmap: Mmap,
    device: Device,
}

// SAFETY: Mmap is Send + Sync. The model struct holds no mutable state.
// Sessions are created per-thread and are not shared.
#[expect(
    unsafe_code,
    reason = "Mmap is Send+Sync; model holds no mutable state"
)]
unsafe impl Send for EmbeddingModel {}
#[expect(
    unsafe_code,
    reason = "Mmap is Send+Sync; model holds no mutable state"
)]
unsafe impl Sync for EmbeddingModel {}

impl EmbeddingModel {
    /// Load an ONNX embedding model from a `HuggingFace` repository.
    ///
    /// Downloads the model file on first call; subsequent calls use the cache.
    /// The file is memory-mapped for zero-copy session creation.
    ///
    /// # Errors
    ///
    /// Returns an error if the model cannot be downloaded or memory-mapped.
    pub fn load(model_repo: &str, model_file: &str, device: Device) -> crate::Result<Self> {
        let api = Api::new().map_err(|e| crate::Error::Download(e.to_string()))?;
        let repo = api.model(model_repo.to_string());
        let model_path = repo
            .get(model_file)
            .map_err(|e| crate::Error::Download(e.to_string()))?;

        let file = File::open(&model_path).map_err(|e| crate::Error::Io {
            path: model_path.display().to_string(),
            source: e,
        })?;

        // SAFETY: The file is read-only and not modified while mapped.
        #[expect(unsafe_code, reason = "mmap of read-only model file")]
        // Safety: file is read-only, not modified while mapped
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| crate::Error::Io {
            path: model_path.display().to_string(),
            source: e,
        })?;

        Ok(Self { mmap, device })
    }

    /// Create an ONNX Runtime session that reads directly from the mmap.
    ///
    /// Each rayon thread should create its own session via
    /// [`rayon::iter::ParallelIterator::map_init`]. Sessions share the
    /// underlying mmap memory — only execution metadata is per-session.
    ///
    /// # Errors
    ///
    /// Returns an error if the ONNX session fails to initialize.
    pub fn create_session(&self) -> crate::Result<InMemorySession<'_>> {
        let mut builder = ort::session::Session::builder().map_err(|e| anyhow::anyhow!("{e}"))?;

        builder = builder
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_intra_threads(1)
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        // Configure execution provider based on device
        match self.device {
            Device::Cpu => {} // default CPU EP, no config needed
            Device::CoreML => {
                let cache_dir = dirs_path("ripvec", "coreml-cache");
                let mut ep = ort::ep::CoreML::default()
                    .with_compute_units(ort::ep::coreml::ComputeUnits::All)
                    .with_model_format(ort::ep::coreml::ModelFormat::MLProgram);
                if let Some(dir) = &cache_dir {
                    ep = ep.with_model_cache_dir(dir.display().to_string());
                }
                builder = builder
                    .with_execution_providers([ep.build()])
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
            }
            Device::Cuda => {
                builder = builder
                    .with_execution_providers([ort::ep::CUDA::default().build()])
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
            }
        }

        let session = builder.commit_from_memory_directly(&self.mmap)?;
        Ok(session)
    }
}

/// Pre-tokenized encoding ready for ONNX inference.
pub struct Encoding {
    /// Token IDs.
    pub input_ids: Vec<i64>,
    /// Attention mask (1 for real tokens, 0 for padding).
    pub attention_mask: Vec<i64>,
    /// Token type IDs (0 for single-sequence models).
    pub token_type_ids: Vec<i64>,
}

/// Produce L2-normalized embedding vectors for a batch of tokenized inputs.
///
/// Pads all sequences to the longest in the batch, runs a single ONNX
/// inference call with shape `[batch_size, max_seq_len]`, then extracts
/// per-sequence CLS embeddings. This amortizes per-call overhead and
/// enables SIMD across the batch dimension.
///
/// # Errors
///
/// Returns an error if the ONNX inference fails or the output tensor
/// cannot be extracted.
pub fn embed_batch(
    session: &mut InMemorySession<'_>,
    encodings: &[Encoding],
) -> crate::Result<Vec<Vec<f32>>> {
    if encodings.is_empty() {
        return Ok(vec![]);
    }

    let batch_size = encodings.len();
    let max_len = encodings
        .iter()
        .map(|e| e.input_ids.len())
        .max()
        .unwrap_or(0);

    // Build padded [batch_size, max_len] tensors
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

    let ids = ndarray::Array2::from_shape_vec((batch_size, max_len), ids_flat)?;
    let mask = ndarray::Array2::from_shape_vec((batch_size, max_len), mask_flat)?;
    let types = ndarray::Array2::from_shape_vec((batch_size, max_len), types_flat)?;

    let ids_ref = TensorRef::from_array_view(&ids)?;
    let mask_ref = TensorRef::from_array_view(&mask)?;
    let types_ref = TensorRef::from_array_view(&types)?;

    let outputs = session.run(ort::inputs![
        "input_ids" => ids_ref,
        "attention_mask" => mask_ref,
        "token_type_ids" => types_ref,
    ])?;

    // Output shape: [batch_size, seq_len, hidden_dim]
    // CLS pooling: take first token (index 0) of each sequence
    let output = &outputs[0];
    let array = output.try_extract_array::<f32>()?;

    let mut results = Vec::with_capacity(batch_size);
    for i in 0..batch_size {
        let seq = array.index_axis(Axis(0), i);
        let cls = seq.index_axis(Axis(0), 0); // first token = CLS
        let embedding: Vec<f32> = cls.iter().copied().collect();
        results.push(l2_normalize(&embedding));
    }

    Ok(results)
}

/// Produce a single L2-normalized embedding vector from tokenized input.
///
/// Convenience wrapper around [`embed_batch`] for single-sequence use.
///
/// # Errors
///
/// Returns an error if the ONNX inference fails.
pub fn embed(
    session: &mut InMemorySession<'_>,
    input_ids: &[i64],
    attention_mask: &[i64],
    token_type_ids: &[i64],
) -> crate::Result<Vec<f32>> {
    let enc = Encoding {
        input_ids: input_ids.to_vec(),
        attention_mask: attention_mask.to_vec(),
        token_type_ids: token_type_ids.to_vec(),
    };
    let mut results = embed_batch(session, &[enc])?;
    Ok(results.pop().unwrap_or_default())
}

/// L2-normalize a vector. Returns zero vector if norm is zero.
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

    #[test]
    fn l2_normalize_unit_vector() {
        let v = vec![1.0, 0.0, 0.0];
        let n = l2_normalize(&v);
        assert!((n[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn l2_normalize_arbitrary_vector() {
        let v = vec![3.0, 4.0];
        let n = l2_normalize(&v);
        let norm: f32 = n.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn l2_normalize_zero_vector() {
        let v = vec![0.0, 0.0, 0.0];
        let n = l2_normalize(&v);
        assert_eq!(n, vec![0.0, 0.0, 0.0]);
    }
}
