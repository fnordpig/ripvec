mod cli;
mod output;
mod progress;
mod tui;

use anyhow::{Context, Result};
use clap::Parser;
use tracing_subscriber::layer::SubscriberExt;

fn main() -> Result<()> {
    let args = cli::Args::parse();

    // Set up Chrome tracing if `--trace <file>` is specified.
    // The guard must be held until the end of main — dropping it flushes the trace file.
    let trace_guard = args.trace.as_ref().map(|trace_path| {
        let (chrome_layer, guard) = tracing_chrome::ChromeLayerBuilder::new()
            .file(trace_path)
            .include_args(true)
            .build();
        tracing::subscriber::set_global_default(tracing_subscriber::registry().with(chrome_layer))
            .expect("failed to set tracing subscriber");
        guard
    });

    // Create profiler
    let profiler = ripvec_core::profile::Profiler::new(
        args.profile,
        std::time::Duration::from_secs_f64(args.profile_interval),
    );

    // Configure thread pool — default to physical core count (empirically optimal)
    let cores = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1);
    let threads = if args.threads > 0 {
        args.threads
    } else {
        cores
    };
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .context("failed to configure thread pool")?;

    // Whether to show indicatif progress bars: TTY stderr and no --profile flag.
    let use_progress = !args.profile && std::io::IsTerminal::is_terminal(&std::io::stderr());

    // Print profiler header
    profiler.header(
        env!("CARGO_PKG_VERSION"),
        &args.model_repo,
        rayon::current_num_threads(),
        cores,
    );

    // Load embedding backend + tokenizer (shared by both modes)
    let (backend, tokenizer, search_cfg) = load_pipeline(&args, use_progress, &profiler)?;

    if args.interactive {
        drop(trace_guard);
        run_interactive(
            backend,
            tokenizer,
            &search_cfg,
            &args,
            use_progress,
            &profiler,
        )?;
    } else {
        run_oneshot(
            &*backend,
            &tokenizer,
            &search_cfg,
            &args,
            use_progress,
            &profiler,
        )?;
        drop(trace_guard);
    }

    Ok(())
}

/// Load the embedding backend, tokenizer, and build the search config.
fn load_pipeline(
    args: &cli::Args,
    use_progress: bool,
    profiler: &ripvec_core::profile::Profiler,
) -> Result<(
    Box<dyn ripvec_core::backend::EmbedBackend>,
    tokenizers::Tokenizer,
    ripvec_core::embed::SearchConfig,
)> {
    let backend = {
        let _guard = profiler.phase("model_load");
        let pb = use_progress.then(|| progress::spinner("Loading model\u{2026}"));
        let kind = match args.backend {
            cli::BackendArg::Candle => ripvec_core::backend::BackendKind::Candle,
            cli::BackendArg::Mlx => ripvec_core::backend::BackendKind::Mlx,
            cli::BackendArg::Ort => ripvec_core::backend::BackendKind::Ort,
        };
        let device_hint = match args.device {
            cli::DeviceArg::Cpu => ripvec_core::backend::DeviceHint::Cpu,
            cli::DeviceArg::Metal | cli::DeviceArg::Cuda => ripvec_core::backend::DeviceHint::Gpu,
        };
        let result = ripvec_core::backend::load_backend(kind, &args.model_repo, device_hint)
            .context("failed to load embedding backend")?;
        if let Some(pb) = pb {
            pb.finish_and_clear();
        }
        result
    };
    let tokenizer = ripvec_core::tokenize::load_tokenizer(&args.model_repo)
        .context("failed to load tokenizer")?;

    let search_cfg = ripvec_core::embed::SearchConfig {
        batch_size: args.batch_size,
        max_tokens: args.max_tokens,
        chunk: ripvec_core::chunk::ChunkConfig {
            max_chunk_bytes: args.max_chunk_bytes,
            window_size: args.window_size,
            window_overlap: args.window_overlap,
        },
        sort_order: match args.sort_order {
            cli::SortOrderArg::Desc => ripvec_core::embed::SortOrder::Descending,
            cli::SortOrderArg::Asc => ripvec_core::embed::SortOrder::Ascending,
            cli::SortOrderArg::None => ripvec_core::embed::SortOrder::None,
        },
        text_mode: args.text_mode,
    };

    Ok((backend, tokenizer, search_cfg))
}

/// Interactive TUI mode: embed the codebase once, then launch the search UI.
fn run_interactive(
    backend: Box<dyn ripvec_core::backend::EmbedBackend>,
    tokenizer: tokenizers::Tokenizer,
    search_cfg: &ripvec_core::embed::SearchConfig,
    args: &cli::Args,
    use_progress: bool,
    profiler: &ripvec_core::profile::Profiler,
) -> Result<()> {
    let pb_embed = use_progress.then(|| progress::spinner("Embedding codebase\u{2026}"));
    let (chunks, embeddings) = ripvec_core::embed::embed_all(
        std::path::Path::new(&args.path),
        backend.as_ref(),
        &tokenizer,
        search_cfg,
        profiler,
    )
    .context("embedding failed")?;
    if let Some(pb) = pb_embed {
        pb.finish_and_clear();
    }
    profiler.finish();

    // Determine hidden dimension from first non-empty embedding
    let hidden_dim = embeddings
        .iter()
        .find(|e| !e.is_empty())
        .map_or(384, Vec::len);

    let app = tui::App {
        query: String::new(),
        selected: 0,
        preview_scroll: 0,
        chunks,
        embeddings,
        results: Vec::new(),
        backend,
        tokenizer,
        hidden_dim,
        threshold: args.threshold,
        rank_time_ms: 0.0,
        should_quit: false,
    };

    tui::run(app)
}

/// One-shot search mode: embed, rank, print results, exit.
fn run_oneshot(
    backend: &dyn ripvec_core::backend::EmbedBackend,
    tokenizer: &tokenizers::Tokenizer,
    search_cfg: &ripvec_core::embed::SearchConfig,
    args: &cli::Args,
    use_progress: bool,
    profiler: &ripvec_core::profile::Profiler,
) -> Result<()> {
    anyhow::ensure!(
        !args.query.is_empty(),
        "query is required (or use --interactive)"
    );

    let pb_search = use_progress.then(|| progress::spinner("Embedding\u{2026}"));
    let results = ripvec_core::embed::search(
        std::path::Path::new(&args.path),
        &args.query,
        backend,
        tokenizer,
        args.top_k,
        search_cfg,
        profiler,
    )
    .context("search failed")?;
    if let Some(pb) = pb_search {
        pb.finish_and_clear();
    }

    profiler.finish();

    let filtered: Vec<_> = results
        .into_iter()
        .filter(|r| r.similarity >= args.threshold)
        .collect();
    output::print_results(&filtered, &args.format);

    Ok(())
}
