//! Generic backend that pairs a [`Driver`] with a [`ModelArch`].
//!
//! [`GenericBackend`] implements [`EmbedBackend`] by delegating to the
//! architecture's `forward()` method, which composes driver primitives into
//! the full inference pipeline. This decouples weight loading from the
//! backend interface — any `(Driver, ModelArch)` pair can serve as an
//! embedding backend.
//!
//! The `_mmap` field keeps the memory-mapped safetensors file alive as long
//! as the backend exists, since Metal zero-copy buffers reference its pages.

use super::arch::ModelArch;
use super::driver::Driver;
use super::{EmbedBackend, Encoding};

/// Generic backend that pairs a [`Driver`] with a [`ModelArch`].
///
/// Implements [`EmbedBackend`] by calling `arch.forward(driver, encodings)`.
/// The driver provides hardware-specific compute primitives; the architecture
/// orchestrates them into a full forward pass.
///
/// # Lifetime invariant
///
/// `_mmap` **must** be declared after `arch` so it is dropped last. The
/// architecture's weight tensors reference pages in the memory-mapped file
/// via zero-copy Metal buffers; dropping the mmap first would invalidate them.
pub struct GenericBackend<D: Driver, A: ModelArch<D>> {
    /// Hardware compute driver (Metal, CUDA, CPU).
    driver: D,
    /// Model architecture with loaded weights.
    arch: A,
    /// Maximum token count the model supports.
    max_tokens: usize,
    /// Whether this backend runs on a GPU.
    is_gpu: bool,
    /// Keeps the memory-mapped safetensors file alive.
    ///
    /// Must outlive the weight tensors in `arch` — declared last for correct
    /// drop order.
    _mmap: memmap2::Mmap,
}

impl<D: Driver, A: ModelArch<D>> GenericBackend<D, A> {
    /// Create a new generic backend from a driver, architecture, and mmap.
    ///
    /// The `mmap` must be the memory-mapped safetensors file whose pages back
    /// the weight tensors stored in `arch`.
    pub fn new(driver: D, arch: A, max_tokens: usize, is_gpu: bool, mmap: memmap2::Mmap) -> Self {
        Self {
            driver,
            arch,
            max_tokens,
            is_gpu,
            _mmap: mmap,
        }
    }
}

impl<D, A> EmbedBackend for GenericBackend<D, A>
where
    D: Driver + Send + Sync + 'static,
    A: ModelArch<D> + Send + Sync + 'static,
{
    fn embed_batch(&self, encodings: &[Encoding]) -> crate::Result<Vec<Vec<f32>>> {
        // Sub-batch at MAX_BATCH=32 to reduce padding waste.
        // Pre-tokenization sorts by descending length, so consecutive
        // sequences have similar lengths. Smaller sub-batches → tighter
        // per-batch padding → less wasted compute.
        const MAX_BATCH: usize = 32;
        if encodings.len() <= MAX_BATCH {
            return self.arch.forward(&self.driver, encodings);
        }
        let mut all = Vec::with_capacity(encodings.len());
        for chunk in encodings.chunks(MAX_BATCH) {
            let mut results = self.arch.forward(&self.driver, chunk)?;
            all.append(&mut results);
        }
        Ok(all)
    }

    fn supports_clone(&self) -> bool {
        false
    }

    fn clone_backend(&self) -> Box<dyn EmbedBackend> {
        panic!("GenericBackend does not support cloning")
    }

    fn is_gpu(&self) -> bool {
        self.is_gpu
    }

    fn max_tokens(&self) -> usize {
        self.max_tokens
    }
}
