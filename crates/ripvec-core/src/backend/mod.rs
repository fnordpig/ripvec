//! Embedding backend abstraction layer.
//!
//! Defines the [`EmbedBackend`] trait that all embedding backends (Candle, MLX,
//! ORT) implement, plus the [`Encoding`] input type and [`BackendKind`]
//! discriminant. Use [`load_backend`] to construct a backend by kind.

pub mod candle;
#[cfg(feature = "mlx")]
pub mod mlx;
#[cfg(feature = "ort")]
pub mod ort;

/// Pre-tokenized encoding ready for inference.
///
/// Token IDs, attention mask, and token type IDs must all have the same length.
/// Token count is capped at `MODEL_MAX_TOKENS` (512) by the tokenizer before
/// reaching the backend.
#[derive(Debug, Clone)]
pub struct Encoding {
    /// Token IDs produced by the tokenizer.
    pub input_ids: Vec<i64>,
    /// Attention mask (1 for real tokens, 0 for padding).
    pub attention_mask: Vec<i64>,
    /// Token type IDs (0 for single-sequence models).
    pub token_type_ids: Vec<i64>,
}

/// Trait for embedding backends.
///
/// Implementations must be [`Send`] so they can be moved across threads (e.g.
/// into a ring-buffer pipeline). The trait is object-safe — callers use
/// `&dyn EmbedBackend` or `Box<dyn EmbedBackend>`.
///
/// # GPU vs CPU scheduling
///
/// - **CPU backends** (`is_gpu() == false`): cloned per rayon thread via
///   [`clone_backend`](EmbedBackend::clone_backend).
/// - **GPU backends** (`is_gpu() == true`): use a ring-buffer pipeline with
///   `RING_SIZE = 4` for bounded memory.
pub trait EmbedBackend: Send + Sync {
    /// Embed a batch of pre-tokenized inputs, returning L2-normalized vectors.
    ///
    /// Each inner `Vec<f32>` is the embedding for the corresponding
    /// [`Encoding`]. Errors **must** propagate — never silently return
    /// defaults.
    ///
    /// # Errors
    ///
    /// Returns an error if tensor construction or the forward pass fails.
    fn embed_batch(&self, encodings: &[Encoding]) -> crate::Result<Vec<Vec<f32>>>;

    /// Whether this backend supports cheap cloning for per-thread instances.
    ///
    /// CPU backends return `true`; GPU backends typically return `false`.
    fn supports_clone(&self) -> bool;

    /// Create a cheap clone of this backend for per-thread use in rayon.
    ///
    /// # Panics
    ///
    /// May panic if [`supports_clone`](EmbedBackend::supports_clone) returns
    /// `false`. Callers must check `supports_clone()` first.
    fn clone_backend(&self) -> Box<dyn EmbedBackend>;

    /// Whether this backend runs on a GPU.
    ///
    /// GPU backends use a ring-buffer pipelined scheduler (`RING_SIZE = 4`)
    /// for bounded memory usage.
    fn is_gpu(&self) -> bool;
}

/// Available embedding backend implementations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BackendKind {
    /// Candle (pure-Rust, CPU + Metal + CUDA).
    #[default]
    Candle,
    /// MLX (Apple Silicon, macOS only).
    Mlx,
    /// ONNX Runtime (cross-platform, CPU + GPU).
    Ort,
}

impl std::fmt::Display for BackendKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Candle => write!(f, "candle"),
            Self::Mlx => write!(f, "mlx"),
            Self::Ort => write!(f, "ort"),
        }
    }
}

/// Device hint passed to [`load_backend`].
///
/// Backends map this to their native device type. Not all backends support
/// all devices — unsupported combinations return an error.
#[derive(Debug, Clone, Copy, Default)]
pub enum DeviceHint {
    /// Automatically select the best available device.
    #[default]
    Auto,
    /// Force CPU inference.
    Cpu,
    /// Force GPU inference (Metal on macOS, CUDA on Linux/Windows).
    Gpu,
}

/// Construct an embedding backend of the given kind.
///
/// Downloads model weights on first use via `hf-hub`. The `device_hint`
/// is advisory — backends that don't support GPU fall back to CPU.
///
/// # Errors
///
/// Returns an error if the requested backend was not compiled in (missing
/// feature flag) or if model loading fails.
pub fn load_backend(
    kind: BackendKind,
    model_repo: &str,
    device_hint: DeviceHint,
) -> crate::Result<Box<dyn EmbedBackend>> {
    match kind {
        BackendKind::Candle => {
            let backend = candle::CandleBackend::load(model_repo, &device_hint)?;
            Ok(Box::new(backend))
        }
        #[cfg(feature = "mlx")]
        BackendKind::Mlx => {
            let backend = mlx::MlxBackend::load(model_repo, &device_hint)?;
            Ok(Box::new(backend))
        }
        #[cfg(not(feature = "mlx"))]
        BackendKind::Mlx => Err(crate::Error::Other(anyhow::anyhow!(
            "mlx backend requires building with: cargo build --features mlx"
        ))),
        #[cfg(feature = "ort")]
        BackendKind::Ort => {
            let backend = ort::OrtBackend::load(model_repo, &device_hint)?;
            Ok(Box::new(backend))
        }
        #[cfg(not(feature = "ort"))]
        BackendKind::Ort => Err(crate::Error::Other(anyhow::anyhow!(
            "ort backend requires building with: cargo build --features ort"
        ))),
    }
}

/// Detect all available backends and load them.
///
/// Probes for GPU backends (MLX, CUDA) first, then always adds CPU
/// as the baseline. Returns backends in priority order — the first
/// entry is the primary (used for query embedding in interactive mode).
///
/// # Errors
///
/// Returns an error if no backends can be loaded (not even CPU).
pub fn detect_backends(model_repo: &str) -> crate::Result<Vec<Box<dyn EmbedBackend>>> {
    let mut backends: Vec<Box<dyn EmbedBackend>> = Vec::new();

    // Try MLX (Apple Silicon GPU)
    #[cfg(feature = "mlx")]
    if let Ok(b) = mlx::MlxBackend::load(model_repo, &DeviceHint::Auto) {
        backends.push(Box::new(b));
    }

    // Future: enumerate CUDA devices
    // #[cfg(feature = "cuda")]
    // for device_id in 0..cuda_device_count() { ... }

    // Add CPU as fallback only when no GPU backend was loaded.
    // On Apple Silicon, running CPU + MLX concurrently is slower than
    // MLX alone because they share the same physical cores and memory.
    // On discrete GPU systems (CUDA), CPU would be a useful helper.
    let has_gpu = backends.iter().any(|b| b.is_gpu());
    if !has_gpu {
        if let Ok(b) = candle::CandleBackend::load(model_repo, &DeviceHint::Cpu) {
            backends.push(Box::new(b));
        }
    }

    if backends.is_empty() {
        return Err(crate::Error::Other(anyhow::anyhow!(
            "no embedding backends available"
        )));
    }

    Ok(backends)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that `EmbedBackend` is object-safe by constructing a trait object type.
    #[test]
    fn trait_is_object_safe() {
        // If this compiles, the trait is object-safe.
        fn _assert_object_safe(_: &dyn EmbedBackend) {}
    }

    /// Verify that `Box<dyn EmbedBackend>` is `Send`.
    #[test]
    fn trait_object_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<Box<dyn EmbedBackend>>();
    }

    /// Verify that `Box<dyn EmbedBackend>` is `Sync` (needed for `&dyn` across threads).
    #[test]
    fn trait_object_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<Box<dyn EmbedBackend>>();
    }

    /// Verify that `Arc<dyn EmbedBackend>` is `Send` (needed for ring-buffer pipeline).
    #[test]
    fn arc_trait_object_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<std::sync::Arc<dyn EmbedBackend>>();
    }

    #[test]
    fn encoding_construction() {
        let enc = Encoding {
            input_ids: vec![101, 2023, 2003, 1037, 3231, 102],
            attention_mask: vec![1, 1, 1, 1, 1, 1],
            token_type_ids: vec![0, 0, 0, 0, 0, 0],
        };
        assert_eq!(enc.input_ids.len(), 6);
        assert_eq!(enc.attention_mask.len(), 6);
        assert_eq!(enc.token_type_ids.len(), 6);
    }

    #[test]
    fn encoding_clone() {
        let enc = Encoding {
            input_ids: vec![101, 102],
            attention_mask: vec![1, 1],
            token_type_ids: vec![0, 0],
        };
        let cloned = enc.clone();
        assert_eq!(enc.input_ids, cloned.input_ids);
    }

    #[test]
    fn backend_kind_default_is_candle() {
        assert_eq!(BackendKind::default(), BackendKind::Candle);
    }

    #[test]
    fn backend_kind_display() {
        assert_eq!(BackendKind::Candle.to_string(), "candle");
        assert_eq!(BackendKind::Mlx.to_string(), "mlx");
        assert_eq!(BackendKind::Ort.to_string(), "ort");
    }

    #[cfg(not(feature = "mlx"))]
    #[test]
    fn load_backend_mlx_not_compiled() {
        let result = load_backend(BackendKind::Mlx, "test/model", DeviceHint::Cpu);
        assert!(result.is_err());
    }

    #[test]
    fn load_backend_ort_not_implemented() {
        let result = load_backend(BackendKind::Ort, "test/model", DeviceHint::Cpu);
        assert!(result.is_err());
    }

    #[test]
    fn detect_backends_returns_at_least_one() {
        let backends = detect_backends("BAAI/bge-small-en-v1.5").unwrap();
        assert!(!backends.is_empty());
    }

    #[test]
    fn detect_backends_cpu_always_last() {
        let backends = detect_backends("BAAI/bge-small-en-v1.5").unwrap();
        let last = backends.last().unwrap();
        assert!(!last.is_gpu(), "last backend should be CPU fallback");
    }
}
