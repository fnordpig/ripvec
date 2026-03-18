//! Error types for ripvec-core.

use thiserror::Error;

/// Errors that can occur in ripvec-core operations.
#[derive(Error, Debug)]
pub enum Error {
    /// Model download or cache retrieval failed.
    #[error("model download failed: {0}")]
    Download(String),

    /// Candle model inference failed.
    #[error("candle inference failed")]
    Candle(#[from] candle_core::Error),

    /// ONNX Runtime inference failed.
    #[error("ORT inference failed: {0}")]
    Ort(String),

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

    /// Catch-all for other errors.
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}
