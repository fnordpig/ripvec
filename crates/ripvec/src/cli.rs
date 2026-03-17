//! Command-line argument definitions using clap derive.

use clap::Parser;

/// Semantic code search — like ripgrep but for meaning.
#[derive(Parser, Debug)]
#[command(name = "ripvec", version, about)]
pub struct Args {
    /// Natural language query to search for.
    pub query: String,

    /// Root directory to search (defaults to current directory).
    #[arg(default_value = ".")]
    pub path: String,

    /// Number of results to show.
    #[arg(short = 'n', long, default_value_t = 10)]
    pub top_k: usize,

    /// `HuggingFace` model repository.
    #[arg(long, default_value = "BAAI/bge-small-en-v1.5")]
    pub model_repo: String,

    /// ONNX model filename within the repository.
    #[arg(long, default_value = "onnx/model.onnx")]
    pub model_file: String,

    /// Output format.
    #[arg(short, long, default_value = "color")]
    pub format: OutputFormat,

    /// Minimum similarity threshold (0.0 to 1.0).
    #[arg(short = 't', long, default_value_t = 0.0)]
    pub threshold: f32,

    /// Number of threads for parallel processing (0 = 2x cores).
    #[arg(short = 'j', long, default_value_t = 0)]
    pub threads: usize,

    /// Batch size for embedding inference (chunks per ONNX call).
    #[arg(short = 'b', long, default_value_t = 32)]
    pub batch_size: usize,

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
