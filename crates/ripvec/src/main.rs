mod cli;
mod output;
mod progress;
mod tui;

use anyhow::{Context, Result};
use clap::Parser;
use tracing_subscriber::layer::SubscriberExt;

fn main() -> Result<()> {
    let mut args = cli::Args::parse();

    // In interactive mode the query is typed in the TUI, so the first
    // positional arg is the path, not the query.  Clap sees it as
    // `query` because that's the first positional — swap them.
    if args.interactive && !args.query.is_empty() && args.path == "." {
        args.path = std::mem::take(&mut args.query);
    }

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

    // Load embedding backend(s) + tokenizer (shared by both modes)
    let (backends, tokenizer, search_cfg) = load_pipeline(&args, use_progress, &profiler)?;

    if args.interactive {
        drop(trace_guard);
        run_interactive(
            backends,
            tokenizer,
            &search_cfg,
            &args,
            use_progress,
            &profiler,
        )?;
    } else {
        run_oneshot(
            &backends,
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

/// Load the embedding backend(s), tokenizer, and build the search config.
///
/// When `--backend auto` (the default), probes for all available backends
/// via [`ripvec_core::backend::detect_backends`]. When a specific backend
/// is requested, loads just that one.
#[expect(clippy::type_complexity, reason = "tuple return is clear in context")]
fn load_pipeline(
    args: &cli::Args,
    use_progress: bool,
    profiler: &ripvec_core::profile::Profiler,
) -> Result<(
    Vec<Box<dyn ripvec_core::backend::EmbedBackend>>,
    tokenizers::Tokenizer,
    ripvec_core::embed::SearchConfig,
)> {
    let backends = {
        let _guard = profiler.phase("model_load");
        let pb = use_progress.then(|| progress::spinner("Loading model\u{2026}"));
        let result = match args.backend {
            cli::BackendArg::Auto => ripvec_core::backend::detect_backends(&args.model_repo)
                .context("failed to detect available backends")?,
            ref specific => {
                let kind = match specific {
                    cli::BackendArg::Candle => ripvec_core::backend::BackendKind::Candle,
                    cli::BackendArg::Mlx => ripvec_core::backend::BackendKind::Mlx,
                    cli::BackendArg::Ort => ripvec_core::backend::BackendKind::Ort,
                    cli::BackendArg::Auto => unreachable!(),
                };
                let device_hint = match args.device {
                    cli::DeviceArg::Cpu => ripvec_core::backend::DeviceHint::Cpu,
                    cli::DeviceArg::Metal | cli::DeviceArg::Cuda => {
                        ripvec_core::backend::DeviceHint::Gpu
                    }
                };
                vec![
                    ripvec_core::backend::load_backend(kind, &args.model_repo, device_hint)
                        .context("failed to load embedding backend")?,
                ]
            }
        };
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

    Ok((backends, tokenizer, search_cfg))
}

/// Interactive TUI mode: embed the codebase once, then launch the search UI.
fn run_interactive(
    backends: Vec<Box<dyn ripvec_core::backend::EmbedBackend>>,
    tokenizer: tokenizers::Tokenizer,
    search_cfg: &ripvec_core::embed::SearchConfig,
    args: &cli::Args,
    use_progress: bool,
    _profiler: &ripvec_core::profile::Profiler,
) -> Result<()> {
    // Create a profiler that drives the spinner with live stats
    let pb = if use_progress {
        Some(progress::spinner("Indexing\u{2026}"))
    } else {
        None
    };
    let live_profiler = if let Some(ref pb) = pb {
        let pb = pb.clone();
        ripvec_core::profile::Profiler::with_callback(
            std::time::Duration::from_millis(500),
            move |msg| pb.set_message(msg.to_string()),
        )
    } else {
        ripvec_core::profile::Profiler::noop()
    };

    let backend_refs: Vec<&dyn ripvec_core::backend::EmbedBackend> =
        backends.iter().map(|b| &**b).collect();
    let (chunks, embeddings) = ripvec_core::embed::embed_all(
        std::path::Path::new(&args.path),
        &backend_refs,
        &tokenizer,
        search_cfg,
        &live_profiler,
    )
    .context("embedding failed")?;

    if let Some(pb) = pb {
        let n_files = chunks
            .iter()
            .map(|c| &c.file_path)
            .collect::<std::collections::HashSet<_>>()
            .len();
        pb.finish_with_message(format!(
            "Indexed {} chunks from {} files",
            chunks.len(),
            n_files,
        ));
    }

    // Build index summary: count chunks by file extension
    let index_summary = {
        let mut ext_counts: std::collections::BTreeMap<String, usize> =
            std::collections::BTreeMap::new();
        for chunk in &chunks {
            let ext = std::path::Path::new(&chunk.file_path)
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("other");
            *ext_counts.entry(ext.to_string()).or_default() += 1;
        }
        let breakdown: Vec<String> = ext_counts
            .iter()
            .rev() // largest extensions first (BTreeMap is sorted)
            .map(|(ext, count)| format!("{count} .{ext}"))
            .collect();
        format!("{} chunks \u{2502} {}", chunks.len(), breakdown.join(", "))
    };

    let index = tui::index::SearchIndex::new(chunks, &embeddings);

    let app = tui::App {
        query: String::new(),
        selected: 0,
        preview_scroll: 0,
        index,
        results: Vec::new(),
        backends,
        tokenizer,
        threshold: args.threshold,
        rank_time_ms: 0.0,
        highlighter: tui::highlight::Highlighter::new(),
        should_quit: false,
        open_editor: None,
        index_summary,
    };

    tui::run(app)
}

/// One-shot search mode: embed, rank, print results, exit.
fn run_oneshot(
    backends: &[Box<dyn ripvec_core::backend::EmbedBackend>],
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

    let backend_refs: Vec<&dyn ripvec_core::backend::EmbedBackend> =
        backends.iter().map(|b| &**b).collect();
    let pb_search = use_progress.then(|| progress::spinner("Embedding\u{2026}"));
    let results = ripvec_core::embed::search(
        std::path::Path::new(&args.path),
        &args.query,
        &backend_refs,
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
