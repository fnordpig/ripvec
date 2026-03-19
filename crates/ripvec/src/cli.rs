//! Command-line argument definitions using clap derive.

use clap::Parser;

/// Semantic code search — like ripgrep but for meaning.
#[derive(Parser, Debug)]
#[command(name = "ripvec", version, about)]
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

    /// `HuggingFace` model repository.
    #[arg(long, default_value = "BAAI/bge-small-en-v1.5")]
    pub model_repo: String,

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

    /// Embedding backend implementation.
    #[arg(long, default_value = "candle")]
    pub backend: BackendArg,

    /// Batch size for embedding inference (chunks per model forward pass).
    #[arg(short = 'b', long, default_value_t = 32)]
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

    /// Chunk scheduling order for parallel embedding batches.
    #[arg(long, default_value = "none")]
    pub sort_order: SortOrderArg,

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
#[derive(clap::ValueEnum, Clone, Debug)]
pub enum BackendArg {
    /// Candle (pure-Rust, CPU + Metal + CUDA).
    Candle,
    /// MLX (Apple Silicon, macOS only).
    Mlx,
    /// ONNX Runtime (cross-platform, CPU + GPU).
    Ort,
}

/// Chunk scheduling order for the embedding pipeline.
#[derive(clap::ValueEnum, Clone, Debug)]
pub enum SortOrderArg {
    /// Longest chunks first (best load balance — default).
    Desc,
    /// Shortest chunks first.
    Asc,
    /// No sorting — process chunks in file-walk order.
    None,
}
