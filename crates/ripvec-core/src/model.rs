//! ONNX embedding model loading and inference.
//!
//! Downloads model weights from HuggingFace, creates an ONNX Runtime
//! session, and provides embedding inference with CLS pooling and
//! L2 normalization.

use hf_hub::api::sync::Api;
use ndarray::Axis;
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::TensorRef;

/// An embedding model backed by ONNX Runtime.
pub struct EmbeddingModel {
    session: Session,
}

impl EmbeddingModel {
    /// Load an ONNX embedding model from a HuggingFace repository.
    ///
    /// Downloads the model file on first call; subsequent calls use the cache.
    /// The OS page cache keeps weights hot between invocations.
    pub fn load(model_repo: &str, model_file: &str) -> crate::Result<Self> {
        let api = Api::new().map_err(|e| crate::Error::Download(e.to_string()))?;
        let repo = api.model(model_repo.to_string());
        let model_path = repo
            .get(model_file)
            .map_err(|e| crate::Error::Download(e.to_string()))?;

        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_intra_threads(1) // parallelism via rayon, not ORT threads
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .commit_from_file(&model_path)?;

        Ok(Self { session })
    }

    /// Produce an L2-normalized embedding vector from tokenized input.
    ///
    /// Uses CLS pooling (first token) suitable for BGE models.
    /// Input arrays must all have the same length (token count).
    pub fn embed(
        &mut self,
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

        let outputs = self.session.run(ort::inputs![
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
