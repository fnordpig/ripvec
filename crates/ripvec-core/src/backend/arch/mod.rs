//! Model architecture trait and variant enum.
//!
//! The [`ModelArch`] trait defines how a model architecture composes
//! [`Driver`](super::driver::Driver) primitives into a complete forward pass
//! (embeddings -> encoder layers -> pooling -> normalization).
//!
//! Each architecture (ClassicBert, ModernBert) is implemented once
//! and works with any driver backend via generics.

pub mod classic_bert;
pub mod modern_bert;

use super::Encoding;
use super::driver::Driver;

/// Model architecture that composes [`Driver`] primitives into a forward pass.
///
/// Implementations store their weights (on device) and model config, then
/// orchestrate the driver to execute embedding lookup, encoder layers, pooling,
/// and L2 normalization.
///
/// # Type parameter
///
/// `D: Driver` — the hardware backend. Architectures are generic over the
/// driver so they can be monomorphized for each backend (Metal, CUDA, CPU).
pub trait ModelArch<D: Driver> {
    /// Run the full forward pass: embeddings -> encoder layers -> pool -> L2 normalize.
    ///
    /// Returns one L2-normalized embedding vector per input encoding.
    ///
    /// # Errors
    ///
    /// Returns an error if any driver operation fails (buffer allocation,
    /// kernel dispatch, synchronization, etc.).
    fn forward(&self, driver: &D, encodings: &[Encoding]) -> crate::Result<Vec<Vec<f32>>>;
}

/// Supported model architectures.
///
/// Each variant corresponds to a distinct BERT family with different attention
/// mechanisms, activations, position encodings, and pooling strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchVariant {
    /// BGE-small: learned position embeddings, GELU, CLS pooling, bias.
    ClassicBert,
    /// ModernBERT: alternating local/global attention, GeGLU, unpadding.
    ModernBert,
}
