//! Command-line argument definitions using clap derive.

use clap::Parser;

/// Semantic code search — like ripgrep but for meaning.
#[derive(Parser, Debug)]
#[command(name = "ripvec", version, about)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "CLI flags are naturally boolean"
)]
pub struct Args {
    /// Natural language query to search for.
    ///
    /// Required for one-shot mode; ignored when `--interactive` is set.
    #[arg(default_value = "")]
    pub query: String,

    /// Root directory to search (defaults to current directory).
    #[arg(default_value = ".")]
    pub path: String,

    /// Launch interactive TUI mode (embed once, then search as you type).
    #[arg(short = 'i', long)]
    pub interactive: bool,

    /// Number of results to show (0 = all matches above threshold).
    #[arg(short = 'n', long, default_value_t = 0)]
    pub top_k: usize,

    /// Use code-optimized model (nomic-ai/CodeRankEmbed).
    /// Adds the required query prefix automatically.
    #[arg(long, conflicts_with = "text")]
    pub code: bool,

    /// Use general text model (BAAI/bge-small-en-v1.5). This is the default.
    #[arg(long)]
    pub text: bool,

    /// Override `HuggingFace` model repository (advanced).
    #[arg(long)]
    pub model_repo: Option<String>,

    /// Output format.
    #[arg(short, long, default_value = "color")]
    pub format: OutputFormat,

    /// Minimum similarity threshold (0.0 to 1.0).
    #[arg(short = 't', long, default_value_t = 0.5)]
    pub threshold: f32,

    /// Number of threads for parallel processing (0 = cores).
    #[arg(short = 'j', long, default_value_t = 0)]
    pub threads: usize,

    /// Inference device: cpu, metal (macOS GPU), or cuda (NVIDIA GPU).
    #[arg(long, default_value = "cpu")]
    pub device: DeviceArg,

    /// Embedding backend implementation (auto = detect all available).
    #[arg(long, default_value = "auto")]
    pub backend: BackendArg,

    /// Batch size for embedding inference (chunks per model forward pass).
    #[arg(short = 'b', long, default_value_t = 64)]
    pub batch_size: usize,

    /// Maximum tokens per chunk fed to the model (0 = unlimited).
    ///
    /// Capping tokens controls inference cost: 128 tokens is 7.7× faster
    /// than 512. CLS pooling means early tokens carry the most semantic weight.
    #[arg(long, default_value_t = 0)]
    pub max_tokens: usize,

    /// Maximum chunk size in bytes before splitting into windows.
    #[arg(long, default_value_t = 4096)]
    pub max_chunk_bytes: usize,

    /// Sliding-window size in bytes for the fallback chunker.
    #[arg(long, default_value_t = 2048)]
    pub window_size: usize,

    /// Overlap between adjacent sliding windows in bytes.
    #[arg(long, default_value_t = 512)]
    pub window_overlap: usize,

    /// Treat all files as plain text (sliding-window chunking only).
    /// By default, recognized source files use tree-sitter semantic chunking
    /// and unrecognized files fall back to plain-text windows.
    #[arg(long)]
    pub text_mode: bool,

    /// Enable pipeline profiling output to stderr.
    #[arg(long)]
    pub profile: bool,

    /// Profiling report interval in seconds.
    #[arg(long, default_value_t = 10.0)]
    pub profile_interval: f64,

    /// Write a Chrome trace JSON file (open in `chrome://tracing` or Perfetto).
    #[arg(long, value_name = "FILE")]
    pub trace: Option<String>,

    /// Use persistent index: cache embeddings to disk, re-embed only changed files.
    ///
    /// First run builds the full index. Subsequent runs load from cache and
    /// only re-embed files that changed since last run.
    #[arg(long)]
    pub index: bool,

    /// Force a full rebuild of the persistent index (requires `--index`).
    #[arg(long, requires = "index")]
    pub reindex: bool,

    /// Clear the persistent index cache for this project and exit.
    #[arg(long)]
    pub clear_cache: bool,

    /// Override the cache directory (default: `~/.cache/ripvec/`).
    #[arg(long, value_name = "DIR")]
    pub cache_dir: Option<String>,
}

/// Output format for search results.
#[derive(clap::ValueEnum, Clone, Debug)]
pub enum OutputFormat {
    /// Plain text without color.
    Plain,
    /// JSON output.
    Json,
    /// Colored terminal output (default).
    Color,
}

/// Inference device selection.
#[derive(clap::ValueEnum, Clone, Debug)]
pub enum DeviceArg {
    /// CPU inference (default, works everywhere).
    Cpu,
    /// Apple Metal GPU (macOS).
    Metal,
    /// NVIDIA CUDA GPU.
    Cuda,
}

/// Embedding backend implementation.
#[derive(clap::ValueEnum, Clone, Debug, Default)]
pub enum BackendArg {
    /// Auto-detect all available backends (default).
    #[default]
    Auto,
    /// CPU (ndarray + system BLAS).
    Cpu,
    /// CUDA (cudarc, NVIDIA GPU via cuBLAS).
    Cuda,
    /// MLX (Apple Silicon, macOS only).
    Mlx,
}
