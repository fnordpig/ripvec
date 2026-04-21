//! Generic backend that pairs a [`Driver`] with a [`ModelArch`].
//!
//! [`GenericBackend`] implements [`EmbedBackend`] by delegating to the
//! architecture's `forward()` method, which composes driver primitives into
//! the full inference pipeline. This decouples weight loading from the
//! backend interface — any `(Driver, ModelArch)` pair can serve as an
//! embedding backend.
//!
//! The `_mmap` field keeps the memory-mapped safetensors file alive as long
//! as the backend exists, since Metal zero-copy buffers and CPU `MmapTensor`
//! slices reference its pages.

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
/// via zero-copy Metal buffers or CPU `MmapTensor::Mapped` slices; dropping
/// the mmap first would invalidate them.
pub struct GenericBackend<D: Driver, A: ModelArch<D>> {
    /// Hardware compute driver (Metal, CUDA, CPU).
    driver: D,
    /// Model architecture with loaded weights.
    arch: A,
    /// Maximum token count the model supports.
    max_tokens: usize,
    /// Whether this backend runs on a GPU.
    is_gpu: bool,
    /// Maximum encodings per forward pass. Larger batches saturate GPU SMs better
    /// but use more memory. Default: 32 (Metal-tuned). CUDA can handle 128+.
    max_batch: usize,
    /// Keeps the memory-mapped safetensors file alive.
    ///
    /// Must outlive the weight tensors in `arch` — declared last for correct
    /// drop order.
    _mmap: MmapHolder,
}

/// Holds a memory-mapped file, accepting either an owned `Mmap` or an
/// `Arc<Mmap>`. CPU backends share the `Arc` with `MmapTensor::Mapped`
/// variants; GPU backends pass an owned `Mmap`.
pub enum MmapHolder {
    /// Owned mmap (GPU backends — tensors are device copies, not mmap refs).
    Owned(memmap2::Mmap),
    /// Shared mmap (CPU backend — `MmapTensor::Mapped` variants hold `Arc` clones).
    Shared(std::sync::Arc<memmap2::Mmap>),
}

impl<D: Driver, A: ModelArch<D>> GenericBackend<D, A> {
    /// Create a new generic backend from a driver, architecture, and mmap.
    ///
    /// The `mmap` must be the memory-mapped safetensors file whose pages back
    /// the weight tensors stored in `arch`.
    ///
    /// For GPU backends, runs a warm-up forward pass to prime the buffer pool.
    /// This is skipped for large models (max_tokens > 1024) where the warm-up
    /// cost exceeds the benefit.
    ///
    /// `max_batch` controls how many encodings are sent in each forward pass.
    /// Metal: 32 (optimal for M2 Max AMX). CUDA: 128+ (needs more work to
    /// saturate 128 SMs on RTX 4090).
    pub fn new(driver: D, arch: A, max_tokens: usize, is_gpu: bool, mmap: memmap2::Mmap) -> Self {
        Self::with_max_batch(
            driver,
            arch,
            max_tokens,
            is_gpu,
            MmapHolder::Owned(mmap),
            32,
        )
    }

    /// Create with an `Arc<Mmap>` shared with zero-copy tensors.
    pub fn new_shared(
        driver: D,
        arch: A,
        max_tokens: usize,
        is_gpu: bool,
        mmap: std::sync::Arc<memmap2::Mmap>,
    ) -> Self {
        Self::with_max_batch(
            driver,
            arch,
            max_tokens,
            is_gpu,
            MmapHolder::Shared(mmap),
            32,
        )
    }

    /// Create with explicit max batch size.
    #[expect(clippy::cast_possible_wrap, reason = "warmup seq length is small")]
    pub fn with_max_batch(
        driver: D,
        arch: A,
        max_tokens: usize,
        is_gpu: bool,
        mmap: MmapHolder,
        max_batch: usize,
    ) -> Self {
        let backend = Self {
            driver,
            arch,
            max_tokens,
            is_gpu,
            max_batch,
            _mmap: mmap,
        };
        // Warm up buffer pool: run a dummy forward to pre-allocate Metal buffers.
        // Without this, the first real batch pays 160-330 fresh newBufferWithLength
        // calls. The warm-up fills the pool; subsequent batches with similar
        // dimensions get exact-match hits (within 8× tolerance).
        //
        // Small models (BGE-small, 12L): batch=32 × seq=512, ~80ms.
        // Large models (ModernBERT, 22L): batch=32 × seq=64, ~300ms.
        //   (Smaller seq keeps cost down; 8× pool tolerance covers seq up to 512.)
        if is_gpu && max_tokens <= 1024 {
            let seq = if max_tokens <= 1024 {
                512.min(max_tokens)
            } else {
                64
            };
            let mut dummy = Vec::with_capacity(32);
            for _ in 0..32 {
                let ids: Vec<i64> = (0..seq as i64).collect();
                dummy.push(Encoding {
                    input_ids: ids,
                    attention_mask: vec![1; seq],
                    token_type_ids: vec![0; seq],
                });
            }
            let _ = backend.arch.forward(&backend.driver, &dummy);
        }
        backend
    }
}

impl<D, A> EmbedBackend for GenericBackend<D, A>
where
    D: Driver + Send + Sync + 'static,
    A: ModelArch<D> + Send + Sync + 'static,
{
    fn embed_batch(&self, encodings: &[Encoding]) -> crate::Result<Vec<Vec<f32>>> {
        let max_batch = self.max_batch;
        if encodings.len() <= max_batch {
            return self.arch.forward(&self.driver, encodings);
        }
        let mut all = Vec::with_capacity(encodings.len());
        for chunk in encodings.chunks(max_batch) {
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

    fn name(&self) -> &'static str {
        self.driver.name()
    }
}
