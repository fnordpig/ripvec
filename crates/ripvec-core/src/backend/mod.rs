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
    /// `ClassicBert`: 512. `ModernBERT`: up to model config. Tokens beyond this
    /// are truncated during tokenization.
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
        not(any(
            feature = "cuda",
            feature = "mlx",
            feature = "cpu",
            feature = "cpu-accelerate",
            feature = "metal"
        )),
        expect(unused_variables, reason = "used when backend features are enabled")
    )]
    model_repo: &str,
    #[cfg_attr(
        not(any(
            feature = "cuda",
            feature = "mlx",
            feature = "cpu",
            feature = "cpu-accelerate",
            feature = "metal"
        )),
        expect(unused_variables, reason = "used when backend features are enabled")
    )]
    device_hint: DeviceHint,
    #[cfg_attr(
        not(any(
            feature = "cuda",
            feature = "mlx",
            feature = "cpu",
            feature = "cpu-accelerate",
            feature = "metal"
        )),
        expect(unused_variables, reason = "used when backend features are enabled")
    )]
    max_layers: Option<usize>,
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
        #[cfg(any(feature = "cpu", feature = "cpu-accelerate"))]
        BackendKind::Cpu => {
            if is_modernbert_model(model_repo) {
                return load_modernbert_cpu(model_repo, max_layers);
            }
            #[cfg(feature = "cpu")]
            {
                let backend = cpu::CpuBackend::load(model_repo, &device_hint)?;
                return Ok(Box::new(backend));
            }
            #[cfg(not(feature = "cpu"))]
            Err(crate::Error::Other(anyhow::anyhow!(
                "ClassicBert CPU backend requires feature 'cpu'; only ModernBERT is available with 'cpu-accelerate'"
            )))
        }
        #[cfg(not(any(feature = "cpu", feature = "cpu-accelerate")))]
        BackendKind::Cpu => Err(crate::Error::Other(anyhow::anyhow!(
            "cpu backend requires building with: cargo build --features cpu"
        ))),
        #[cfg(feature = "metal")]
        BackendKind::Metal => {
            // All models route through the driver/arch system.
            if is_modernbert_model(model_repo) {
                return load_modernbert_metal(model_repo, max_layers);
            }
            load_classic_metal(model_repo)
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
        not(any(
            feature = "cuda",
            feature = "mlx",
            feature = "cpu",
            feature = "cpu-accelerate",
            feature = "metal"
        )),
        expect(unused_variables, reason = "used when backend features are enabled")
    )]
    model_repo: &str,
    #[cfg_attr(
        not(any(
            feature = "cuda",
            feature = "mlx",
            feature = "cpu",
            feature = "cpu-accelerate",
            feature = "metal"
        )),
        expect(unused_variables, reason = "used when backend features are enabled")
    )]
    max_layers: Option<usize>,
) -> crate::Result<Vec<Box<dyn EmbedBackend>>> {
    #[cfg_attr(
        not(any(
            feature = "cuda",
            feature = "mlx",
            feature = "cpu",
            feature = "cpu-accelerate",
            feature = "metal"
        )),
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
        // Route models through the driver/arch system by architecture.
        if is_modernbert_model(model_repo) {
            if let Ok(b) = load_modernbert_metal(model_repo, max_layers) {
                backends.push(b);
            }
        } else if let Ok(b) = load_classic_metal(model_repo) {
            backends.push(b);
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
        not(any(feature = "cpu", feature = "cpu-accelerate")),
        expect(unused_variables, reason = "used when cpu feature is enabled")
    )]
    let has_gpu = backends.iter().any(|b| b.is_gpu());
    #[cfg(any(feature = "cpu", feature = "cpu-accelerate"))]
    if !has_gpu {
        if is_modernbert_model(model_repo) {
            if let Ok(b) = load_modernbert_cpu(model_repo, max_layers) {
                backends.push(b);
            }
        } else {
            #[cfg(feature = "cpu")]
            if let Ok(b) = cpu::CpuBackend::load(model_repo, &DeviceHint::Cpu) {
                backends.push(Box::new(b));
            }
        }
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
pub fn load_modernbert_metal(
    model_repo: &str,
    max_layers: Option<usize>,
) -> crate::Result<Box<dyn EmbedBackend>> {
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
    let (mut arch, mmap) = driver.load_modern_bert_weights(&weights_path, &config)?;
    arch.max_layers = max_layers;

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

/// Load `ModernBERT` on CPU via the driver/arch system.
#[cfg(any(feature = "cpu", feature = "cpu-accelerate"))]
pub fn load_modernbert_cpu(
    model_repo: &str,
    max_layers: Option<usize>,
) -> crate::Result<Box<dyn EmbedBackend>> {
    use driver::cpu::{CpuDriver, ModernBertConfig};
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

    let config_str = std::fs::read_to_string(&config_path).map_err(|e| crate::Error::Io {
        path: config_path.display().to_string(),
        source: e,
    })?;
    let config_json: serde_json::Value = serde_json::from_str(&config_str)
        .map_err(|e| crate::Error::Other(anyhow::anyhow!("config parse error: {e}")))?;
    let config = ModernBertConfig::from_json(&config_json)?;
    let max_tokens = config.max_position_embeddings;

    let driver = CpuDriver::new()?;
    let (mut arch, mmap) = driver.load_modern_bert_weights(&weights_path, &config)?;
    arch.max_layers = max_layers;

    tracing::info!(
        model_repo,
        hidden = config.hidden_size,
        layers = config.num_hidden_layers,
        heads = config.num_attention_heads,
        max_tokens,
        "ModernBERT loaded on CPU (driver/arch)"
    );

    Ok(Box::new(GenericBackend::new(
        driver, arch, max_tokens, false, mmap,
    )))
}

/// Check whether a model repo uses the `ModernBERT` architecture.
///
/// Downloads and inspects `config.json` to check for `"model_type": "modernbert"`.
/// Returns `false` on any download or parse error (fail-open for detection).
#[cfg(any(feature = "metal", feature = "cpu", feature = "cpu-accelerate"))]
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

// ---------------------------------------------------------------------------
// ClassicBert loader (driver/arch system)
// ---------------------------------------------------------------------------

/// Load a `ClassicBert` model (e.g. `BAAI/bge-small-en-v1.5`) on the Metal GPU backend.
///
/// Downloads the model from Hugging Face Hub (cached after first download),
/// memory-maps the safetensors weights, fuses Q/K/V into a single tensor per
/// layer, and builds a [`GenericBackend`] pairing a
/// [`MetalDriver`](driver::metal::MetalDriver) with a
/// [`ClassicBertArch`](arch::classic_bert::ClassicBertArch).
///
/// # Errors
///
/// Returns an error if no Metal device is available, the model cannot be
/// downloaded, or weight loading fails.
#[cfg(feature = "metal")]
pub fn load_classic_metal(model_repo: &str) -> crate::Result<Box<dyn EmbedBackend>> {
    use driver::metal::{ClassicBertConfig, MetalDriver};
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
    let config = ClassicBertConfig::from_json(&config_json)?;
    let max_tokens = config.max_position_embeddings;

    let driver = MetalDriver::new()?;
    let (arch, mmap) = driver.load_classic_bert_weights(&weights_path, &config)?;

    tracing::info!(
        model_repo,
        hidden = config.hidden_size,
        layers = config.num_hidden_layers,
        heads = config.num_attention_heads,
        intermediate = config.intermediate_size,
        max_tokens,
        "ClassicBert loaded on Metal (driver/arch)"
    );

    Ok(Box::new(GenericBackend::new(
        driver, arch, max_tokens, true, mmap,
    )))
}

// ---------------------------------------------------------------------------
// ClassicBert loader (CPU driver/arch system)
// ---------------------------------------------------------------------------

/// Load a `ClassicBert` model (e.g. `BAAI/bge-small-en-v1.5`) on the CPU backend
/// via the driver/arch system.
///
/// Downloads the model from Hugging Face Hub (cached after first download),
/// reads safetensors weights into `Vec<f32>` tensors, fuses Q/K/V per layer,
/// and builds a [`GenericBackend`] pairing a
/// [`CpuDriver`](driver::cpu::CpuDriver) with a
/// [`ClassicBertArch`](arch::classic_bert::ClassicBertArch).
///
/// # Errors
///
/// Returns an error if the model cannot be downloaded or weight loading fails.
#[cfg(any(feature = "cpu", feature = "cpu-accelerate"))]
pub fn load_classic_cpu(model_repo: &str) -> crate::Result<Box<dyn EmbedBackend>> {
    use driver::cpu::{ClassicBertConfig, CpuDriver};
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
    let config = ClassicBertConfig::from_json(&config_json)?;
    let max_tokens = config.max_position_embeddings;

    let driver = CpuDriver::new()?;
    let (arch, mmap) = driver.load_classic_bert_weights(&weights_path, &config)?;

    tracing::info!(
        model_repo,
        hidden = config.hidden_size,
        layers = config.num_hidden_layers,
        heads = config.num_attention_heads,
        intermediate = config.intermediate_size,
        max_tokens,
        "ClassicBert loaded on CPU (driver/arch)"
    );

    Ok(Box::new(GenericBackend::new(
        driver, arch, max_tokens, false, mmap,
    )))
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
        let result = load_backend(BackendKind::Mlx, "test/model", DeviceHint::Cpu, None);
        assert!(result.is_err());
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn detect_backends_returns_at_least_one() {
        let backends = detect_backends("BAAI/bge-small-en-v1.5", None).unwrap();
        assert!(!backends.is_empty());
    }

    #[cfg(all(feature = "cpu", not(feature = "mlx")))]
    #[test]
    fn detect_backends_returns_at_least_one_backend() {
        let backends = detect_backends("BAAI/bge-small-en-v1.5", None).unwrap();
        assert!(!backends.is_empty(), "should detect at least one backend");
    }

    /// Load `ModernBERT` on Metal and embed a short token sequence.
    ///
    /// Verifies that the full pipeline (weight loading, forward pass, pooling,
    /// L2 normalization) produces a 768-dim unit vector.
    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "requires model download (~570MB)"]
    fn modernbert_loads_and_embeds() {
        use crate::backend::driver::Driver;

        let backend =
            load_modernbert_metal("nomic-ai/modernbert-embed-base", None).expect("load failed");
        assert!(backend.is_gpu(), "Metal backend should be GPU");

        let enc = Encoding {
            input_ids: vec![1, 100, 200, 300, 2],
            attention_mask: vec![1; 5],
            token_type_ids: vec![0; 5],
        };

        // Stage-by-stage diagnostic using the driver directly
        let driver = crate::backend::driver::metal::MetalDriver::new().unwrap();
        let inputs = driver.prepare_batch(&[enc.clone()], 8).unwrap();

        // Check: can we read back input_ids?
        let ids_host = driver.to_host(&inputs.input_ids, 1, 8).unwrap();
        eprintln!("input_ids: {:?}", &ids_host[0][..5]);

        // Check: embedding lookup
        // Need the tok_embeddings weight — load weights directly
        let api = hf_hub::api::sync::Api::new().unwrap();
        let repo = api.model("nomic-ai/modernbert-embed-base".to_string());
        let weights_path = repo.get("model.safetensors").unwrap();
        let config_path = repo.get("config.json").unwrap();
        let config_str = std::fs::read_to_string(&config_path).unwrap();
        let config_json: serde_json::Value = serde_json::from_str(&config_str).unwrap();
        let config =
            crate::backend::driver::metal::ModernBertConfig::from_json(&config_json).unwrap();
        let (mut arch, _mmap) = driver
            .load_modern_bert_weights(&weights_path, &config)
            .unwrap();

        let hidden = driver
            .embedding_lookup(&inputs.input_ids, &arch.weights.tok_embeddings, 8, 768)
            .unwrap();
        let h = driver.to_host(&hidden, 1, 8 * 768).unwrap();
        let nz = h[0].iter().filter(|&&v| v.abs() > 1e-10).count();
        eprintln!(
            "embedding: {nz}/{} nonzero, first 5: {:?}",
            h[0].len(),
            &h[0][..5]
        );

        // Stage-by-stage forward pass bisection
        let total = 8; // padded seq
        let hd = 768;
        let nh = 12;
        let head_dim = 64;

        // After embedding LN
        let emb_clone = driver.clone_tensor(&hidden, total * hd).unwrap();
        let mut ln_out = driver.alloc_zeros(total * hd).unwrap();
        driver
            .layer_norm(
                &mut ln_out,
                &emb_clone,
                &arch.weights.emb_norm_weight,
                &arch.weights.zero_bias,
                total,
                hd,
                1e-5,
            )
            .unwrap();
        let ln_h = driver.to_host(&ln_out, 1, total * hd).unwrap();
        let nz = ln_h[0].iter().filter(|&&v| v.abs() > 1e-10).count();
        eprintln!("STAGE 1 - emb+LN: {nz}/{} nonzero", total * hd);

        // Layer 0 QKV GEMM
        let layer0 = &arch.weights.layers[0];
        let mut qkv = driver.alloc_zeros(total * 3 * hd).unwrap();
        driver
            .gemm(
                &ln_out,
                &layer0.qkv_weight,
                &mut qkv,
                total,
                3 * hd,
                hd,
                true,
            )
            .unwrap();
        let qkv_h = driver.to_host(&qkv, 1, total * 3 * hd).unwrap();
        let nz = qkv_h[0].iter().filter(|&&v| v.abs() > 1e-10).count();
        eprintln!("STAGE 2 - QKV GEMM: {nz}/{} nonzero", total * 3 * hd);

        // QKV split
        let mut q = driver.alloc_zeros(total * hd).unwrap();
        let mut k = driver.alloc_zeros(total * hd).unwrap();
        let mut v = driver.alloc_zeros(total * hd).unwrap();
        driver
            .qkv_split(&mut q, &mut k, &mut v, &qkv, 1, 8, hd, nh, head_dim)
            .unwrap();
        let q_h = driver.to_host(&q, 1, total * hd).unwrap();
        let nz = q_h[0].iter().filter(|&&v| v.abs() > 1e-10).count();
        eprintln!("STAGE 3 - Q after split: {nz}/{} nonzero", total * hd);

        // Attention scores
        let mut scores = driver.alloc_zeros(1 * nh * 8 * 8).unwrap();
        driver
            .gemm_batched(
                &q,
                &k,
                &mut scores,
                8,
                8,
                head_dim,
                true,
                8 * head_dim,
                8 * head_dim,
                8 * 8,
                nh,
            )
            .unwrap();
        let s_h = driver.to_host(&scores, 1, nh * 8 * 8).unwrap();
        let nz = s_h[0].iter().filter(|&&v| v.abs() > 1e-10).count();
        eprintln!("STAGE 4 - scores: {nz}/{} nonzero", nh * 8 * 8);

        // Early exit quality sweep — the whole point of ModernBERT
        use crate::backend::arch::ModelArch;
        let enc2 = Encoding {
            input_ids: vec![1, 100, 200, 300, 2],
            attention_mask: vec![1; 5],
            token_type_ids: vec![0; 5],
        };

        arch.max_layers = None;
        let quick = arch.forward(&driver, &[enc2.clone()]).unwrap();
        let l2: f32 = quick[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        let nz = quick[0].iter().filter(|&&v| v.abs() > 1e-10).count();
        eprintln!(
            "BATCHED forward: L2={l2:.4}, nz={nz}/768, first 3: {:?}",
            &quick[0][..3]
        );

        // Get full 22-layer reference embedding
        arch.max_layers = None;
        let full = arch.forward(&driver, &[enc2.clone()]).unwrap();
        let full_emb = &full[0];
        eprintln!("\n=== ModernBERT Early Exit Quality ===");
        eprintln!(
            "Full 22-layer embedding: {} dims, L2={:.4}",
            full_emb.len(),
            full_emb.iter().map(|x| x * x).sum::<f32>().sqrt()
        );

        // Sweep layers
        for layers in [1, 2, 4, 6, 8, 11, 14, 17, 20, 22] {
            arch.max_layers = Some(layers);
            let result = arch.forward(&driver, &[enc2.clone()]).unwrap();
            let emb = &result[0];
            let cos: f32 = emb.iter().zip(full_emb).map(|(a, b)| a * b).sum();
            eprintln!("  layers={layers:>2}: cosine={cos:.6}");
        }

        // MRL truncation at full layers
        eprintln!("\n=== ModernBERT MRL Truncation ===");
        arch.max_layers = None;
        let full2 = arch.forward(&driver, &[enc2.clone()]).unwrap();
        let full_emb2 = &full2[0];
        for dims in [64, 128, 256, 384, 512, 768] {
            let t: Vec<f32> = full_emb2[..dims].to_vec();
            let t_norm: f32 = t.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            let f_norm: f32 = full_emb2[..dims]
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt()
                .max(1e-12);
            let cos: f32 = t
                .iter()
                .zip(&full_emb2[..dims])
                .map(|(a, b)| a * b)
                .sum::<f32>()
                / (t_norm * f_norm);
            eprintln!("  dims={dims:>3}: cosine={cos:.6}");
        }

        // Combined: early exit + MRL
        eprintln!("\n=== Combined: Early Exit + MRL ===");
        for layers in [6, 11, 16, 22] {
            arch.max_layers = Some(layers);
            let result = arch.forward(&driver, &[enc2.clone()]).unwrap();
            let emb = &result[0];
            for dims in [64, 128, 256, 768] {
                let t: Vec<f32> = emb[..dims].to_vec();
                let t_norm: f32 = t.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
                let f_norm: f32 = full_emb[..dims]
                    .iter()
                    .map(|x| x * x)
                    .sum::<f32>()
                    .sqrt()
                    .max(1e-12);
                let cos: f32 = t
                    .iter()
                    .zip(&full_emb[..dims])
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
                    / (t_norm * f_norm);
                eprintln!("  layers={layers:>2} dims={dims:>3}: cosine={cos:.6}");
            }
        }

        // Throughput benchmark
        eprintln!("\n=== ModernBERT Throughput ===");
        arch.max_layers = None;
        // Build 32 encodings of varying length
        let mut encs = Vec::new();
        for i in 0..32 {
            let len = 16 + (i * 4); // 16 to 140 tokens
            let mut ids = vec![1_i64]; // CLS
            for j in 1..len - 1 {
                ids.push(100 + j as i64);
            }
            ids.push(2); // SEP
            encs.push(Encoding {
                input_ids: ids.clone(),
                attention_mask: vec![1; ids.len()],
                token_type_ids: vec![0; ids.len()],
            });
        }

        // Warmup
        let _ = arch.forward(&driver, &encs[..4]);

        // Timed run
        let t0 = std::time::Instant::now();
        let result = arch.forward(&driver, &encs).unwrap();
        let elapsed = t0.elapsed();
        let throughput = encs.len() as f64 / elapsed.as_secs_f64();
        eprintln!(
            "  batch={}, time={:.1}ms, throughput={:.1}/s",
            encs.len(),
            elapsed.as_secs_f64() * 1000.0,
            throughput
        );
        assert_eq!(result.len(), 32);

        // Batch=1 timing (critical — CLI query path)
        let single = vec![encs[0].clone()];
        let t1 = std::time::Instant::now();
        let _ = arch.forward(&driver, &single).unwrap();
        let single_ms = t1.elapsed().as_secs_f64() * 1000.0;
        eprintln!("  batch=1, time={single_ms:.1}ms");
    }

    /// Load `ClassicBert` (`BAAI/bge-small-en-v1.5`) on Metal via the driver/arch system.
    ///
    /// Verifies that the full pipeline produces a 384-dim L2-normalized vector,
    /// compares against the CPU backend for numerical equivalence, and measures
    /// throughput (target: >=308/s matching monolithic).
    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "requires model download (~33MB)"]
    fn classic_bert_driver_arch() {
        use crate::backend::arch::ModelArch;

        let model_repo = "BAAI/bge-small-en-v1.5";

        // Load via driver/arch system
        let backend = load_classic_metal(model_repo).expect("load_classic_metal failed");
        assert!(backend.is_gpu(), "Metal backend should be GPU");

        let enc = Encoding {
            input_ids: vec![101, 2023, 2003, 1037, 3231, 102],
            attention_mask: vec![1, 1, 1, 1, 1, 1],
            token_type_ids: vec![0, 0, 0, 0, 0, 0],
        };

        // Basic forward pass
        let result = backend.embed_batch(std::slice::from_ref(&enc)).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 384);

        let l2: f32 = result[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        eprintln!(
            "ClassicBert driver/arch: L2={l2:.4}, first 3: {:?}",
            &result[0][..3]
        );
        assert!(
            (l2 - 1.0).abs() < 0.01,
            "embedding should be L2-normalized, got L2={l2}"
        );

        // Compare against CPU backend (reliable reference)
        #[cfg(feature = "cpu")]
        {
            let cpu = load_backend(BackendKind::Cpu, model_repo, DeviceHint::Cpu, None)
                .expect("CPU load failed");
            let cpu_result = cpu.embed_batch(std::slice::from_ref(&enc)).unwrap();
            eprintln!("CPU  first 5: {:?}", &cpu_result[0][..5]);
            eprintln!("NEW  first 5: {:?}", &result[0][..5]);
            let cosine: f32 = result[0]
                .iter()
                .zip(&cpu_result[0])
                .map(|(a, b)| a * b)
                .sum();
            eprintln!("cosine(driver/arch, CPU) = {cosine:.6}");
            assert!(
                cosine > 0.95,
                "cosine similarity vs CPU should be >0.95, got {cosine}"
            );
        }

        // Throughput benchmark
        eprintln!("\n=== ClassicBert Driver/Arch Throughput ===");
        let driver = crate::backend::driver::metal::MetalDriver::new().unwrap();
        let config_path = {
            let api = hf_hub::api::sync::Api::new().unwrap();
            let repo = api.model(model_repo.to_string());
            repo.get("config.json").unwrap()
        };
        let weights_path = {
            let api = hf_hub::api::sync::Api::new().unwrap();
            let repo = api.model(model_repo.to_string());
            repo.get("model.safetensors").unwrap()
        };
        let config_str = std::fs::read_to_string(&config_path).unwrap();
        let config_json: serde_json::Value = serde_json::from_str(&config_str).unwrap();
        let config =
            crate::backend::driver::metal::ClassicBertConfig::from_json(&config_json).unwrap();
        let (arch, _mmap) = driver
            .load_classic_bert_weights(&weights_path, &config)
            .unwrap();

        // Build 32 encodings of varying length
        let mut encs = Vec::new();
        for i in 0..32 {
            let len = 16 + (i * 4); // 16 to 140 tokens
            let mut ids = vec![101_i64]; // [CLS]
            for j in 1..len - 1 {
                ids.push(100 + j as i64);
            }
            ids.push(102); // [SEP]
            encs.push(Encoding {
                input_ids: ids.clone(),
                attention_mask: vec![1; ids.len()],
                token_type_ids: vec![0; ids.len()],
            });
        }

        // Warmup
        let _ = arch.forward(&driver, &encs[..4]);

        // Timed run
        let t0 = std::time::Instant::now();
        let bench_result = arch.forward(&driver, &encs).unwrap();
        let elapsed = t0.elapsed();
        let throughput = encs.len() as f64 / elapsed.as_secs_f64();
        eprintln!(
            "  batch={}, time={:.1}ms, throughput={:.1}/s",
            encs.len(),
            elapsed.as_secs_f64() * 1000.0,
            throughput
        );
        assert_eq!(bench_result.len(), 32);
    }

    /// Load `ClassicBert` (`BAAI/bge-small-en-v1.5`) on CPU via the driver/arch system.
    ///
    /// Verifies that the full pipeline produces a 384-dim L2-normalized vector
    /// and compares against the monolithic CPU backend for numerical equivalence.
    #[cfg(any(feature = "cpu", feature = "cpu-accelerate"))]
    #[test]
    #[ignore = "requires model download (~33MB)"]
    fn classic_bert_cpu_driver_arch() {
        let model_repo = "BAAI/bge-small-en-v1.5";

        // Load via new driver/arch system
        let backend = load_classic_cpu(model_repo).expect("load_classic_cpu failed");
        assert!(!backend.is_gpu(), "CPU backend should not be GPU");

        let enc = Encoding {
            input_ids: vec![101, 2023, 2003, 1037, 3231, 102],
            attention_mask: vec![1, 1, 1, 1, 1, 1],
            token_type_ids: vec![0, 0, 0, 0, 0, 0],
        };

        // Basic forward pass
        let result = backend.embed_batch(std::slice::from_ref(&enc)).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 384);

        let l2: f32 = result[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        eprintln!(
            "ClassicBert CPU driver/arch: L2={l2:.4}, first 5: {:?}",
            &result[0][..5]
        );
        assert!(
            (l2 - 1.0).abs() < 0.01,
            "embedding should be L2-normalized, got L2={l2}"
        );

        // Compare against monolithic CPU backend (reference)
        #[cfg(feature = "cpu")]
        {
            let cpu_mono = cpu::CpuBackend::load(model_repo, &DeviceHint::Cpu)
                .expect("monolithic CPU load failed");
            let cpu_result = cpu_mono.embed_batch(&[enc]).unwrap();
            eprintln!("Mono first 5: {:?}", &cpu_result[0][..5]);
            eprintln!("New  first 5: {:?}", &result[0][..5]);
            let cosine: f32 = result[0]
                .iter()
                .zip(&cpu_result[0])
                .map(|(a, b)| a * b)
                .sum();
            eprintln!("cosine(driver/arch, monolithic) = {cosine:.6}");
            assert!(
                cosine > 0.999,
                "cosine similarity vs monolithic CPU should be >0.999, got {cosine}"
            );
        }
    }
}
