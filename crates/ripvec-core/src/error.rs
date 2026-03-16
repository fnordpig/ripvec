//! Error types for ripvec-core.

use thiserror::Error;

/// Errors that can occur in ripvec-core operations.
#[derive(Error, Debug)]
pub enum Error {
    /// Model download or cache retrieval failed.
    #[error("model download failed: {0}")]
    Download(String),

    /// ONNX Runtime inference failed.
    #[error("ONNX inference failed")]
    Inference(#[from] ort::Error),

    /// Tokenization of input text failed.
    #[error("tokenization failed: {0}")]
    Tokenization(String),

    /// File I/O error with path context.
    #[error("I/O error: {path}")]
    Io {
        /// Path that caused the error.
        path: String,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// Unsupported source file language.
    #[error("unsupported language: {0}")]
    UnsupportedLanguage(String),

    /// ndarray shape mismatch.
    #[error("array shape error")]
    Shape(#[from] ndarray::ShapeError),

    /// Catch-all for other errors.
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}
