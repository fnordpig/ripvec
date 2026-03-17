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

use hf_hub::api::sync::Api;
use memmap2::Mmap;
use ndarray::Axis;
use ort::session::InMemorySession;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::TensorRef;

/// An embedding model backed by a memory-mapped ONNX file.
///
/// The model file is mmap'd once; per-thread sessions read directly from
/// the mapped memory (zero-copy via `commit_from_memory_directly`).
pub struct EmbeddingModel {
    mmap: Mmap,
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
    pub fn load(model_repo: &str, model_file: &str) -> crate::Result<Self> {
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

        Ok(Self { mmap })
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
        let session = ort::session::Session::builder()
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_intra_threads(1) // one ORT thread per session; parallelism via rayon
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .commit_from_memory_directly(&self.mmap)?;
        Ok(session)
    }
}

/// Produce an L2-normalized embedding vector from tokenized input.
///
/// Uses CLS pooling (first token) suitable for BGE models.
///
/// # Errors
///
/// Returns an error if the ONNX inference fails or the output tensor
/// cannot be extracted.
pub fn embed(
    session: &mut InMemorySession<'_>,
    input_ids: &[i64],
    attention_mask: &[i64],
    token_type_ids: &[i64],
) -> crate::Result<Vec<f32>> {
    let len = input_ids.len();
    let ids = ndarray::Array2::from_shape_vec((1, len), input_ids.to_vec())?;
    let mask = ndarray::Array2::from_shape_vec((1, len), attention_mask.to_vec())?;
    let types = ndarray::Array2::from_shape_vec((1, len), token_type_ids.to_vec())?;

    let ids_ref = TensorRef::from_array_view(&ids)?;
    let mask_ref = TensorRef::from_array_view(&mask)?;
    let types_ref = TensorRef::from_array_view(&types)?;

    let outputs = session.run(ort::inputs![
        "input_ids" => ids_ref,
        "attention_mask" => mask_ref,
        "token_type_ids" => types_ref,
    ])?;

    // Model output shape: [1, seq_len, hidden_dim]
    // CLS pooling: take first token embedding
    let output = &outputs[0];
    let array = output.try_extract_array::<f32>()?;
    let cls = array.index_axis(Axis(1), 0);
    let embedding: Vec<f32> = cls.iter().copied().collect();

    // L2 normalize — required for cosine similarity = dot product
    Ok(l2_normalize(&embedding))
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
