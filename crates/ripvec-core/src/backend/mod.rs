//! Embedding backend abstraction layer.
//!
//! Defines the [`EmbedBackend`] trait that all embedding backends (CPU, CUDA,
//! MLX) implement, plus the [`Encoding`] input type and [`BackendKind`]
//! discriminant. Use [`load_backend`] to construct a backend by kind.

pub mod arch;
pub mod blas_info;
#[cfg(feature = "cpu")]
pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod driver;
pub mod generic;
#[cfg(feature = "metal")]
pub mod metal;
#[cfg(feature = "metal")]
pub mod metal_kernels;
#[cfg(feature = "mlx")]
pub mod mlx;

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

    /// Maximum token count this model supports (position embedding limit).
    ///
    /// `ClassicBert`: 512. `NomicBert`: 8192. Tokens beyond this are truncated
    /// during tokenization.
    fn max_tokens(&self) -> usize {
        512 // default for classic BERT models
    }
}

/// Available embedding backend implementations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BackendKind {
    /// CUDA (cudarc, NVIDIA GPUs via cuBLAS + custom kernels).
    Cuda,
    /// MLX (Apple Silicon, macOS only).
    Mlx,
    /// CPU (ndarray + system BLAS).
    #[default]
    Cpu,
    /// Metal (Apple Silicon, macOS only, direct Metal GPU).
    Metal,
}

impl std::fmt::Display for BackendKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cuda => write!(f, "cuda"),
            Self::Mlx => write!(f, "mlx"),
            Self::Cpu => write!(f, "cpu"),
            Self::Metal => write!(f, "metal"),
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
    #[cfg_attr(
        not(any(feature = "cuda", feature = "mlx", feature = "cpu", feature = "metal")),
        expect(unused_variables, reason = "used when backend features are enabled")
    )]
    model_repo: &str,
    #[cfg_attr(
        not(any(feature = "cuda", feature = "mlx", feature = "cpu", feature = "metal")),
        expect(unused_variables, reason = "used when backend features are enabled")
    )]
    device_hint: DeviceHint,
) -> crate::Result<Box<dyn EmbedBackend>> {
    match kind {
        #[cfg(feature = "cuda")]
        BackendKind::Cuda => {
            let backend = cuda::CudaBackend::load(model_repo, &device_hint)?;
            Ok(Box::new(backend))
        }
        #[cfg(not(feature = "cuda"))]
        BackendKind::Cuda => Err(crate::Error::Other(anyhow::anyhow!(
            "cuda backend requires building with: cargo build --features cuda"
        ))),
        #[cfg(feature = "mlx")]
        BackendKind::Mlx => {
            let backend = mlx::MlxBackend::load(model_repo, &device_hint)?;
            Ok(Box::new(backend))
        }
        #[cfg(not(feature = "mlx"))]
        BackendKind::Mlx => Err(crate::Error::Other(anyhow::anyhow!(
            "mlx backend requires building with: cargo build --features mlx"
        ))),
        #[cfg(feature = "cpu")]
        BackendKind::Cpu => {
            let backend = cpu::CpuBackend::load(model_repo, &device_hint)?;
            Ok(Box::new(backend))
        }
        #[cfg(not(feature = "cpu"))]
        BackendKind::Cpu => Err(crate::Error::Other(anyhow::anyhow!(
            "cpu backend requires building with: cargo build --features cpu"
        ))),
        #[cfg(feature = "metal")]
        BackendKind::Metal => {
            // Route ModernBERT models through the new driver/arch system.
            if is_modernbert_model(model_repo) {
                return load_modernbert_metal(model_repo);
            }
            let backend = metal::MetalBackend::load(model_repo, &device_hint)?;
            Ok(Box::new(backend))
        }
        #[cfg(not(feature = "metal"))]
        BackendKind::Metal => Err(crate::Error::Other(anyhow::anyhow!(
            "metal backend requires building with: cargo build --features metal"
        ))),
    }
}

/// Detect all available backends and load them.
///
/// Probes for GPU backends (CUDA, MLX) first, then falls back to CPU.
/// Returns backends in priority order — the first entry is the primary
/// (used for query embedding in interactive mode).
///
/// # Errors
///
/// Returns an error if no backends can be loaded (not even CPU).
pub fn detect_backends(
    #[cfg_attr(
        not(any(feature = "cuda", feature = "mlx", feature = "cpu", feature = "metal")),
        expect(unused_variables, reason = "used when backend features are enabled")
    )]
    model_repo: &str,
) -> crate::Result<Vec<Box<dyn EmbedBackend>>> {
    #[cfg_attr(
        not(any(feature = "cuda", feature = "mlx", feature = "cpu", feature = "metal")),
        expect(unused_mut, reason = "mut needed when backend features are enabled")
    )]
    let mut backends: Vec<Box<dyn EmbedBackend>> = Vec::new();

    // Try CUDA (NVIDIA GPU)
    #[cfg(feature = "cuda")]
    if let Ok(b) = cuda::CudaBackend::load(model_repo, &DeviceHint::Gpu) {
        backends.push(Box::new(b));
    }

    // Try Metal (Apple Silicon GPU, preferred over MLX)
    #[cfg(feature = "metal")]
    {
        // Route ModernBERT models through the new driver/arch system.
        if is_modernbert_model(model_repo) {
            if let Ok(b) = load_modernbert_metal(model_repo) {
                backends.push(b);
            }
        } else if let Ok(b) = metal::MetalBackend::load(model_repo, &DeviceHint::Auto) {
            backends.push(Box::new(b));
        }
    }

    // Try MLX (Apple Silicon GPU, fallback if Metal unavailable)
    #[cfg(feature = "mlx")]
    if backends.is_empty()
        && let Ok(b) = mlx::MlxBackend::load(model_repo, &DeviceHint::Auto)
    {
        backends.push(Box::new(b));
    }

    // Add CPU as fallback only when no GPU backend was loaded.
    // On Apple Silicon, running CPU + MLX concurrently is slower than
    // MLX alone because they share the same physical cores and memory.
    // On discrete GPU systems (CUDA), CPU would be a useful helper.
    #[cfg_attr(
        not(feature = "cpu"),
        expect(unused_variables, reason = "used when cpu feature is enabled")
    )]
    let has_gpu = backends.iter().any(|b| b.is_gpu());
    #[cfg(feature = "cpu")]
    if !has_gpu && let Ok(b) = cpu::CpuBackend::load(model_repo, &DeviceHint::Cpu) {
        backends.push(Box::new(b));
    }

    if backends.is_empty() {
        return Err(crate::Error::Other(anyhow::anyhow!(
            "no embedding backends available"
        )));
    }

    Ok(backends)
}

// ---------------------------------------------------------------------------
// ModernBERT loader (driver/arch system)
// ---------------------------------------------------------------------------

/// Load a `ModernBERT` model on the Metal GPU backend.
///
/// Downloads the model from Hugging Face Hub (cached after first download),
/// memory-maps the safetensors weights, and builds a [`GenericBackend`]
/// pairing a [`MetalDriver`](driver::metal::MetalDriver) with a
/// [`ModernBertArch`](arch::modern_bert::ModernBertArch).
///
/// # Errors
///
/// Returns an error if no Metal device is available, the model cannot be
/// downloaded, or weight loading fails.
#[cfg(feature = "metal")]
pub fn load_modernbert_metal(model_repo: &str) -> crate::Result<Box<dyn EmbedBackend>> {
    use driver::metal::{MetalDriver, ModernBertConfig};
    use generic::GenericBackend;
    use hf_hub::api::sync::Api;

    let api = Api::new().map_err(|e| crate::Error::Download(e.to_string()))?;
    let repo = api.model(model_repo.to_string());

    let config_path = repo
        .get("config.json")
        .map_err(|e| crate::Error::Download(e.to_string()))?;
    let weights_path = repo
        .get("model.safetensors")
        .map_err(|e| crate::Error::Download(e.to_string()))?;

    // Parse config.json
    let config_str = std::fs::read_to_string(&config_path).map_err(|e| crate::Error::Io {
        path: config_path.display().to_string(),
        source: e,
    })?;
    let config_json: serde_json::Value = serde_json::from_str(&config_str)
        .map_err(|e| crate::Error::Other(anyhow::anyhow!("config parse error: {e}")))?;
    let config = ModernBertConfig::from_json(&config_json)?;
    let max_tokens = config.max_position_embeddings;

    let driver = MetalDriver::new()?;
    let (arch, mmap) = driver.load_modern_bert_weights(&weights_path, &config)?;

    tracing::info!(
        model_repo,
        hidden = config.hidden_size,
        layers = config.num_hidden_layers,
        heads = config.num_attention_heads,
        intermediate = config.intermediate_size,
        max_tokens,
        "ModernBERT loaded on Metal (driver/arch)"
    );

    Ok(Box::new(GenericBackend::new(
        driver, arch, max_tokens, true, mmap,
    )))
}

/// Check whether a model repo uses the `ModernBERT` architecture.
///
/// Downloads and inspects `config.json` to check for `"model_type": "modernbert"`.
/// Returns `false` on any download or parse error (fail-open for detection).
#[cfg(feature = "metal")]
fn is_modernbert_model(model_repo: &str) -> bool {
    let Ok(api) = hf_hub::api::sync::Api::new() else {
        return false;
    };
    let repo = api.model(model_repo.to_string());
    let Ok(config_path) = repo.get("config.json") else {
        return false;
    };
    let Ok(config_str) = std::fs::read_to_string(&config_path) else {
        return false;
    };
    let Ok(json) = serde_json::from_str::<serde_json::Value>(&config_str) else {
        return false;
    };
    json.get("model_type")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|t| t == "modernbert")
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
    fn backend_kind_default_is_cpu() {
        assert_eq!(BackendKind::default(), BackendKind::Cpu);
    }

    #[test]
    fn backend_kind_display() {
        assert_eq!(BackendKind::Cuda.to_string(), "cuda");
        assert_eq!(BackendKind::Mlx.to_string(), "mlx");
        assert_eq!(BackendKind::Cpu.to_string(), "cpu");
    }

    #[cfg(not(feature = "mlx"))]
    #[test]
    fn load_backend_mlx_not_compiled() {
        let result = load_backend(BackendKind::Mlx, "test/model", DeviceHint::Cpu);
        assert!(result.is_err());
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn detect_backends_returns_at_least_one() {
        let backends = detect_backends("BAAI/bge-small-en-v1.5").unwrap();
        assert!(!backends.is_empty());
    }

    #[cfg(all(feature = "cpu", not(feature = "mlx")))]
    #[test]
    fn detect_backends_cpu_always_last() {
        let backends = detect_backends("BAAI/bge-small-en-v1.5").unwrap();
        let last = backends.last().unwrap();
        assert!(!last.is_gpu(), "last backend should be CPU fallback");
    }

    /// Load `ModernBERT` on Metal and embed a short token sequence.
    ///
    /// Verifies that the full pipeline (weight loading, forward pass, pooling,
    /// L2 normalization) produces a 768-dim unit vector.
    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "requires model download (~570MB)"]
    fn modernbert_loads_and_embeds() {
        let backend = load_modernbert_metal("nomic-ai/modernbert-embed-base").expect("load failed");
        assert!(backend.is_gpu(), "Metal backend should be GPU");
        assert_eq!(backend.max_tokens(), 8192);

        let enc = Encoding {
            input_ids: vec![1, 100, 200, 300, 2],
            attention_mask: vec![1; 5],
            token_type_ids: vec![0; 5],
        };
        let result = backend.embed_batch(&[enc]).expect("embed_batch failed");
        assert_eq!(result.len(), 1, "should produce one embedding");
        assert_eq!(result[0].len(), 768, "embedding should be 768-dim");

        let l2: f32 = result[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (l2 - 1.0).abs() < 0.01,
            "embedding should be L2 normalized, got L2={l2}"
        );
    }
}
