//! Core library for ripvec semantic code search.
//!
//! Provides ONNX embedding model loading, tree-sitter code chunking,
//! parallel embedding, and cosine similarity ranking.

pub mod error;
pub mod languages;
pub mod model;
pub mod tokenize;

pub use error::Error;

/// Convenience Result type for ripvec-core.
pub type Result<T> = std::result::Result<T, Error>;
