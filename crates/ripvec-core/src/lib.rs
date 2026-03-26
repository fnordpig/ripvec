//! Core library for ripvec semantic code search.
//!
//! Provides pluggable embedding backends ([`backend::EmbedBackend`] trait with
//! CPU, CUDA, and MLX implementations), tree-sitter code chunking, parallel
//! embedding, and cosine similarity ranking.

pub mod backend;
pub mod cache;
pub mod chunk;
pub mod embed;
pub mod error;
pub mod index;
pub mod languages;
pub mod profile;
pub mod repo_map;
pub mod similarity;
pub mod tokenize;
pub mod turbo_quant;
pub mod walk;

pub use error::Error;

/// Convenience Result type for ripvec-core.
pub type Result<T> = std::result::Result<T, Error>;
