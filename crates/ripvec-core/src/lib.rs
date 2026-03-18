//! Core library for ripvec semantic code search.
//!
//! Provides candle-based BERT embedding model loading, tree-sitter code chunking,
//! parallel embedding, and cosine similarity ranking.

pub mod backend;
pub mod chunk;
pub mod embed;
pub mod error;
pub mod languages;
pub mod model;
pub mod profile;
pub mod similarity;
pub mod tokenize;
pub mod walk;

pub use error::Error;

/// Convenience Result type for ripvec-core.
pub type Result<T> = std::result::Result<T, Error>;
