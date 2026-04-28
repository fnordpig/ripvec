mod cli;
mod index_tui;
mod output;
mod progress;
mod tui;

use anyhow::{Context, Result};
use clap::Parser;
use tracing_subscriber::layer::SubscriberExt;

/// Map `-v` count to a tracing level.
fn log_level_from_verbosity(verbose: u8) -> Option<&'static str> {
    match verbose {
        0 => None,
        1 => Some("info"),
        2 => Some("debug"),
        _ => Some("trace"),
    }
}

/// Resolve the effective tracing level.
///
/// Priority: explicit `--log-level`, `--debug`, `-v`, `RIPVEC_LOG`, then warn.
fn resolve_log_level(args: &cli::Args) -> String {
    args.log_level
        .clone()
        .or_else(|| args.debug.then(|| "debug".to_string()))
        .or_else(|| log_level_from_verbosity(args.verbose).map(str::to_string))
        .or_else(|| std::env::var("RIPVEC_LOG").ok())
        .unwrap_or_else(|| "warn".to_string())
}

/// Whether stderr should be line-oriented diagnostics instead of a spinner.
fn diagnostics_mode(args: &cli::Args) -> bool {
    args.profile || args.debug || args.verbose > 0
}

/// Whether `--index` should open the full-screen index dashboard.
fn should_run_index_dashboard(args: &cli::Args) -> bool {
    args.index && args.query.is_empty()
}

fn model_repo_for_args(args: &cli::Args) -> String {
    let use_fast = args.fast || args.text;
    args.model_repo.clone().unwrap_or_else(|| {
        if use_fast {
            "BAAI/bge-small-en-v1.5".to_string()
        } else {
            "nomic-ai/modernbert-embed-base".to_string()
        }
    })
}

#[expect(clippy::too_many_lines, reason = "orchestration entry point")]
fn main() -> Result<()> {
    let mut args = cli::Args::parse();

    // In interactive mode the query is typed in the TUI, so the first
    // positional arg is the path, not the query.  Clap sees it as
    // `query` because that's the first positional — swap them.
    if args.interactive && !args.query.is_empty() && args.path == "." {
        args.path = std::mem::take(&mut args.query);
    }

    // Handle --clear-cache early (before loading anything).
    // For repo-local caches, clears .ripvec/cache/ only (preserves config.toml).
    // For user-level caches, clears ALL model variants (removes the project hash dir).
    if args.clear_cache {
        let root = std::path::Path::new(&args.path);
        // Check for repo-local config first
        if let Some(ripvec_dir) = ripvec_core::cache::config::find_repo_config(root) {
            let cache_dir = ripvec_dir.join("cache");
            if cache_dir.exists() {
                std::fs::remove_dir_all(&cache_dir).context("failed to clear repo-local cache")?;
                eprintln!("Repo-local cache cleared: {}", cache_dir.display());
            } else {
                eprintln!("No repo-local cache found at {}", cache_dir.display());
            }
        } else {
            // User-level cache: pass a dummy model to get parent dir
            let version_dir = ripvec_core::cache::reindex::resolve_cache_dir(
                root,
                "dummy",
                args.cache_dir.as_deref().map(std::path::Path::new),
            );
            let cache_dir = version_dir.parent().unwrap_or(&version_dir);
            if cache_dir.exists() {
                std::fs::remove_dir_all(cache_dir).context("failed to clear cache")?;
                eprintln!("Cache cleared: {}", cache_dir.display());
            } else {
                eprintln!("No cache found at {}", cache_dir.display());
            }
        }
        return Ok(());
    }

    // Resolve model repo: default -> ModernBERT, --fast -> BGE-small.
    let model_repo = model_repo_for_args(&args);

    // Default threshold: 0.5 on normalized [0,1] scores (model-agnostic).
    if args.threshold == 0.0 {
        args.threshold = 0.5;
    }

    let dashboard_mode = should_run_index_dashboard(&args);
    let dashboard_logs = dashboard_mode.then(|| index_tui::LogBuffer::new(2_000));

    // Set up tracing: stderr (or --log-file), optional Chrome trace, level filter.
    let level = resolve_log_level(&args);
    let make_filter = || {
        tracing_subscriber::EnvFilter::try_new(&level)
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"))
    };

    let trace_guard = args.trace.as_ref().map(|trace_path| {
        let (chrome_layer, guard) = tracing_chrome::ChromeLayerBuilder::new()
            .file(trace_path)
            .include_args(true)
            .build();
        tracing::subscriber::set_global_default(
            tracing_subscriber::registry()
                .with(make_filter())
                .with(chrome_layer),
        )
        .expect("failed to set tracing subscriber");
        guard
    });

    if trace_guard.is_none() {
        if let Some(logs) = dashboard_logs.clone() {
            let filter = tracing_subscriber::EnvFilter::new("trace");
            if let Some(log_path) = args.log_file.as_ref() {
                let fmt_layer = tracing_subscriber::fmt::layer()
                    .with_target(false)
                    .compact();
                let file = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(log_path)
                    .expect("failed to open log file");
                let _ = tracing::subscriber::set_global_default(
                    tracing_subscriber::registry()
                        .with(filter)
                        .with(fmt_layer.with_writer(std::sync::Mutex::new(file)))
                        .with(index_tui::BufferLayer::new(logs)),
                );
            } else {
                let _ = tracing::subscriber::set_global_default(
                    tracing_subscriber::registry()
                        .with(filter)
                        .with(index_tui::BufferLayer::new(logs)),
                );
            }
        } else if let Some(log_path) = args.log_file.as_ref() {
            let fmt_layer = tracing_subscriber::fmt::layer()
                .with_target(false)
                .compact();
            let file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(log_path)
                .expect("failed to open log file");
            let _ = tracing::subscriber::set_global_default(
                tracing_subscriber::registry()
                    .with(make_filter())
                    .with(fmt_layer.with_writer(std::sync::Mutex::new(file))),
            );
        } else {
            let fmt_layer = tracing_subscriber::fmt::layer()
                .with_target(false)
                .compact();
            let _ = tracing::subscriber::set_global_default(
                tracing_subscriber::registry()
                    .with(make_filter())
                    .with(fmt_layer.with_writer(std::io::stderr)),
            );
        }
    }

    tracing::info!(
        path = %args.path,
        model = %model_repo,
        mode = %args.mode,
        index = args.index,
        repo_level = args.repo_level,
        reindex = args.reindex,
        backend = ?args.backend,
        device = ?args.device,
        threads = args.threads,
        verbose = args.verbose,
        "ripvec starting"
    );

    // Create profiler. --debug and -v modes enable phase diagnostics because
    // tracing alone does not show which long-running pipeline step is active.
    let profiler = if dashboard_mode {
        ripvec_core::profile::Profiler::noop()
    } else {
        ripvec_core::profile::Profiler::new(
            diagnostics_mode(&args),
            std::time::Duration::from_secs_f64(args.profile_interval),
        )
    };

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

    // Whether to show indicatif progress bars. Diagnostic modes deliberately
    // use line-oriented output because spinners and logs share stderr.
    let use_progress = !dashboard_mode
        && !diagnostics_mode(&args)
        && std::io::IsTerminal::is_terminal(&std::io::stderr());

    if dashboard_mode {
        let logs = dashboard_logs.expect("dashboard log buffer exists in dashboard mode");
        run_index_dashboard(&args, model_repo, cores, &logs)?;
        drop(trace_guard);
        return Ok(());
    }

    // Print profiler header
    profiler.header(
        env!("CARGO_PKG_VERSION"),
        &model_repo,
        rayon::current_num_threads(),
        cores,
    );

    // Load embedding backend(s) + tokenizer (shared by both modes)
    let (backends, tokenizer, search_cfg) =
        load_pipeline(&args, &model_repo, use_progress, &profiler)?;

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

/// Prompt the user to enable `pull.autoStash` for a repo-local cache.
///
/// Only prompts if the setting hasn't been configured yet. Reads a single
/// line from stdin (default: yes).
fn prompt_auto_stash(root: &std::path::Path) {
    if let Some(msg) = ripvec_core::cache::reindex::check_auto_stash(root) {
        eprintln!("{msg}");
        eprint!("[Y/n] ");
        let mut input = String::new();
        let enable = if std::io::stdin().read_line(&mut input).is_ok() {
            !input.trim().eq_ignore_ascii_case("n")
        } else {
            true
        };
        if let Err(e) = ripvec_core::cache::reindex::apply_auto_stash(root, enable) {
            eprintln!("ripvec: failed to save auto_stash preference: {e}");
        } else if enable {
            eprintln!("ripvec: pull.autoStash enabled for this repo.");
        } else {
            eprintln!(
                "ripvec: pull.autoStash declined. You can enable it later in .ripvec/config.toml."
            );
        }
    }
}

/// Load the embedding backend(s), tokenizer, and build the search config.
///
/// When `--backend auto` (the default), probes for all available backends
/// via [`ripvec_core::backend::detect_backends`]. When a specific backend
/// is requested, loads just that one.
#[expect(clippy::type_complexity, reason = "tuple return is clear in context")]
fn load_pipeline(
    args: &cli::Args,
    model_repo: &str,
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

        // An explicit --device cuda/metal pins the backend to that GPU family
        // when --backend is left at auto. Without this, auto would silently
        // fall back to CPU on GPU-load failure even though the user asked for
        // a GPU — pinning surfaces the real error instead.
        let effective_backend = match (&args.backend, &args.device) {
            (cli::BackendArg::Auto, cli::DeviceArg::Cuda) => cli::BackendArg::Cuda,
            (cli::BackendArg::Auto, cli::DeviceArg::Metal) => cli::BackendArg::Metal,
            (b, _) => b.clone(),
        };

        tracing::info!(
            model = model_repo,
            requested_backend = ?args.backend,
            effective_backend = ?effective_backend,
            device = ?args.device,
            "loading embedding backend"
        );
        let result = match effective_backend {
            cli::BackendArg::Auto => ripvec_core::backend::detect_backends(model_repo)
                .context("failed to detect available backends")?,
            ref specific => {
                let kind = match specific {
                    cli::BackendArg::Cpu => ripvec_core::backend::BackendKind::Cpu,
                    cli::BackendArg::Cuda => ripvec_core::backend::BackendKind::Cuda,
                    cli::BackendArg::Mlx => ripvec_core::backend::BackendKind::Mlx,
                    cli::BackendArg::Metal => ripvec_core::backend::BackendKind::Metal,
                    cli::BackendArg::Auto => unreachable!(),
                };
                let device_hint = match args.device {
                    cli::DeviceArg::Cpu => ripvec_core::backend::DeviceHint::Cpu,
                    cli::DeviceArg::Metal | cli::DeviceArg::Cuda => {
                        ripvec_core::backend::DeviceHint::Gpu
                    }
                };
                vec![
                    ripvec_core::backend::load_backend(kind, model_repo, device_hint)
                        .context("failed to load embedding backend")?,
                ]
            }
        };
        if let Some(pb) = pb {
            pb.finish_and_clear();
        }
        tracing::info!(backends = result.len(), "embedding backend loaded");
        result
    };
    tracing::info!(model = model_repo, "loading tokenizer");
    let tokenizer =
        ripvec_core::tokenize::load_tokenizer(model_repo).context("failed to load tokenizer")?;
    tracing::info!("tokenizer loaded");

    let mode: ripvec_core::hybrid::SearchMode = args.mode.parse().unwrap_or_default();

    let search_cfg = ripvec_core::embed::SearchConfig {
        batch_size: args.batch_size,
        max_tokens: args.max_tokens,
        chunk: ripvec_core::chunk::ChunkConfig {
            max_chunk_bytes: args.max_chunk_bytes,
            window_size: args.window_size,
            window_overlap: args.window_overlap,
        },
        text_mode: args.text_mode,
        cascade_dim: None,
        file_type: args.file_type.clone(),
        mode,
    };

    Ok((backends, tokenizer, search_cfg))
}

fn run_index_dashboard(
    args: &cli::Args,
    model_repo: String,
    cores: usize,
    logs: &index_tui::LogBuffer,
) -> Result<()> {
    let (sink, receiver) = index_tui::EventSink::channel(logs.clone());
    let worker_args = (*args).clone();
    let worker_model = model_repo.clone();
    let worker_sink = sink.clone();
    let threads = rayon::current_num_threads();

    std::thread::spawn(move || {
        let result =
            build_index_for_dashboard(&worker_args, worker_model, cores, threads, &worker_sink)
                .map_err(|e| format!("{e:#}"));
        worker_sink.finish(result);
    });

    let outcome = index_tui::run(
        &receiver,
        logs,
        index_tui::DashboardConfig {
            root: args.path.clone(),
            model_repo,
            search_after: args.interactive,
        },
    )?;

    match outcome {
        index_tui::DashboardOutcome::Quit => {
            // The indexing worker may be inside backend/model loading, which
            // is not cancellable. The dashboard has already restored the
            // terminal; exiting the process is safer than letting background
            // GPU/FFI work race process teardown after a user-requested quit.
            std::process::exit(0);
        }
        index_tui::DashboardOutcome::Done(build) => {
            if args.repo_level {
                prompt_auto_stash(std::path::Path::new(&args.path));
            }
            eprintln!(
                "Indexed {} chunks in {}ms",
                build.stats.chunks_total, build.stats.duration_ms
            );
            Ok(())
        }
        index_tui::DashboardOutcome::Search(build) => {
            if args.repo_level {
                prompt_auto_stash(std::path::Path::new(&args.path));
            }
            run_search_tui_from_build(build, args)
        }
    }
}

fn build_index_for_dashboard(
    args: &cli::Args,
    model_repo: String,
    cores: usize,
    threads: usize,
    sink: &index_tui::EventSink,
) -> Result<index_tui::IndexBuild> {
    let profiler_sink = (*sink).clone();
    let tick_sink = (*sink).clone();
    let embedding_sink = (*sink).clone();
    let chunk_sink = (*sink).clone();
    let profiler = ripvec_core::profile::Profiler::with_callback(
        std::time::Duration::from_secs_f64(args.profile_interval),
        move |msg| profiler_sink.profiler_message(msg),
    )
    .with_embed_tick(move |progress| tick_sink.embed_tick(progress))
    .with_chunk_batch(move |chunks| chunk_sink.chunks(chunks))
    .with_embedding_batch(move |embeddings| embedding_sink.embeddings(embeddings));

    profiler.header(env!("CARGO_PKG_VERSION"), &model_repo, threads, cores);

    sink.phase("Loading embedding model");
    let (backends, tokenizer, search_cfg) = load_pipeline(args, &model_repo, false, &profiler)?;
    let backend_refs: Vec<&dyn ripvec_core::backend::EmbedBackend> =
        backends.iter().map(|b| &**b).collect();

    if args.reindex {
        let cache_dir = ripvec_core::cache::reindex::resolve_cache_dir(
            std::path::Path::new(&args.path),
            &model_repo,
            args.cache_dir.as_deref().map(std::path::Path::new),
        );
        tracing::info!(cache_dir = %cache_dir.display(), "reindex requested; clearing cache");
        let _ = std::fs::remove_dir_all(&cache_dir);
    }

    sink.phase("Loading persistent index");
    let (index, stats) = ripvec_core::cache::reindex::incremental_index(
        std::path::Path::new(&args.path),
        &backend_refs,
        &tokenizer,
        &search_cfg,
        &profiler,
        &model_repo,
        args.cache_dir.as_deref().map(std::path::Path::new),
        args.repo_level,
    )
    .context("incremental index failed")?;
    profiler.finish();

    sink.chunk_snapshot(index.chunks());
    let index_summary = build_index_summary(index.chunks());

    Ok(index_tui::IndexBuild {
        index,
        stats,
        backends,
        tokenizer,
        index_summary,
        model_repo,
    })
}

fn build_index_summary(chunks: &[ripvec_core::chunk::CodeChunk]) -> String {
    let mut ext_counts: std::collections::BTreeMap<String, usize> =
        std::collections::BTreeMap::new();
    for chunk in chunks {
        let ext = std::path::Path::new(&chunk.file_path)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("other");
        *ext_counts.entry(ext.to_string()).or_default() += 1;
    }
    let mut pairs: Vec<_> = ext_counts.into_iter().collect();
    pairs.sort_by_key(|b| std::cmp::Reverse(b.1));
    let breakdown: Vec<String> = pairs
        .into_iter()
        .map(|(ext, count)| format!("{count} .{ext}"))
        .collect();
    format!("{} chunks │ {}", chunks.len(), breakdown.join(", "))
}

fn run_search_tui_from_build(build: index_tui::IndexBuild, args: &cli::Args) -> Result<()> {
    let mut app = tui::App {
        query: String::new(),
        selected: 0,
        preview_scroll: 0,
        index: build.index,
        results: Vec::new(),
        backends: build.backends,
        tokenizer: build.tokenizer,
        threshold: args.threshold,
        rank_time_ms: 0.0,
        highlighter: tui::highlight::Highlighter::new(),
        should_quit: false,
        open_editor: None,
        index_summary: build.index_summary,
        watcher_rx: None,
        watcher_handle: None,
        status_flash: None,
        force_redraw: false,
        cache_config: Some(tui::CacheConfig {
            root: std::path::PathBuf::from(&args.path),
            model_repo: build.model_repo,
            cache_dir: args.cache_dir.as_ref().map(std::path::PathBuf::from),
        }),
    };
    attach_file_watcher(&mut app, &args.path)?;
    tui::run(app)
}

fn attach_file_watcher(app: &mut tui::App, path: &str) -> Result<()> {
    use notify::{RecursiveMode, Watcher};
    let (tx, rx) = std::sync::mpsc::channel();
    let mut watcher = notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
        if let Ok(event) = res {
            use notify::EventKind;
            if matches!(
                event.kind,
                EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_)
            ) {
                let _ = tx.send(());
            }
        }
    })
    .context("failed to create file watcher")?;
    watcher
        .watch(std::path::Path::new(path), RecursiveMode::Recursive)
        .context("failed to watch directory")?;
    app.watcher_rx = Some(rx);
    app.watcher_handle = Some(watcher);
    Ok(())
}

/// Interactive TUI mode: embed the codebase once, then launch the search UI.
#[expect(
    clippy::too_many_lines,
    reason = "progress bar setup + watcher setup in one function"
)]
fn run_interactive(
    backends: Vec<Box<dyn ripvec_core::backend::EmbedBackend>>,
    tokenizer: tokenizers::Tokenizer,
    search_cfg: &ripvec_core::embed::SearchConfig,
    args: &cli::Args,
    use_progress: bool,
    _profiler: &ripvec_core::profile::Profiler,
) -> Result<()> {
    // Create a profiler that drives progress display with live stats.
    // Starts as a spinner for walk/chunk phases, then switches to a
    // determinate progress bar once the embed phase begins.
    let spinner = if use_progress {
        Some(progress::spinner("Indexing\u{2026}"))
    } else {
        None
    };
    let embed_display: std::sync::Arc<std::sync::Mutex<Option<progress::EmbedDisplay>>> =
        std::sync::Arc::new(std::sync::Mutex::new(None));
    let live_profiler = if let Some(ref spinner) = spinner {
        let spinner_clone = spinner.clone();
        let profiler =
            ripvec_core::profile::Profiler::with_callback(std::time::Duration::ZERO, move |msg| {
                spinner_clone.set_message(msg.to_string());
            });
        let display = embed_display.clone();
        let spinner_for_tick = spinner.clone();
        profiler.with_embed_tick(move |p| {
            let mut guard = display.lock().unwrap();
            let d = guard.get_or_insert_with(|| {
                spinner_for_tick.finish_and_clear();
                progress::EmbedDisplay::new(p.total as u64)
            });
            d.update(p);
        })
    } else {
        ripvec_core::profile::Profiler::noop()
    };

    let backend_refs: Vec<&dyn ripvec_core::backend::EmbedBackend> =
        backends.iter().map(|b| &**b).collect();

    let model_repo = args.model_repo.clone().unwrap_or_else(|| {
        if args.fast || args.text {
            "BAAI/bge-small-en-v1.5".to_string()
        } else {
            "nomic-ai/modernbert-embed-base".to_string()
        }
    });

    let (chunks, embeddings) = if args.index {
        // Persistent index path: load from cache, diff, re-embed only changes
        let (index, _stats) = ripvec_core::cache::reindex::incremental_index(
            std::path::Path::new(&args.path),
            &backend_refs,
            &tokenizer,
            search_cfg,
            &live_profiler,
            &model_repo,
            args.cache_dir.as_deref().map(std::path::Path::new),
            args.repo_level,
        )
        .context("incremental index failed")?;

        if args.repo_level {
            prompt_auto_stash(std::path::Path::new(&args.path));
        }

        // Extract chunks and embeddings from the HybridIndex for the TUI
        let n = index.chunks().len();
        let embs: Vec<Vec<f32>> = (0..n).filter_map(|i| index.semantic.embedding(i)).collect();
        (index.chunks().to_vec(), embs)
    } else {
        // Stateless path: embed everything from scratch
        ripvec_core::embed::embed_all(
            std::path::Path::new(&args.path),
            &backend_refs,
            &tokenizer,
            search_cfg,
            &live_profiler,
        )
        .context("embedding failed")?
    };

    if use_progress {
        let n_files = chunks
            .iter()
            .map(|c| &c.file_path)
            .collect::<std::collections::HashSet<_>>()
            .len();
        // Finish whichever progress widget is active (embed display or spinner).
        let mut guard = embed_display.lock().unwrap();
        if let Some(d) = guard.take() {
            d.finish(chunks.len(), n_files);
        } else if let Some(spinner) = spinner {
            spinner.finish_with_message(format!(
                "Indexed {} chunks from {} files",
                chunks.len(),
                n_files,
            ));
        }
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
        let mut pairs: Vec<_> = ext_counts.into_iter().collect();
        pairs.sort_by_key(|b| std::cmp::Reverse(b.1)); // most-common extension first
        let breakdown: Vec<String> = pairs
            .into_iter()
            .map(|(ext, count)| format!("{count} .{ext}"))
            .collect();
        format!("{} chunks \u{2502} {}", chunks.len(), breakdown.join(", "))
    };

    let index = ripvec_core::hybrid::HybridIndex::new(chunks, &embeddings, None)
        .context("failed to build hybrid index")?;

    let mut app = tui::App {
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
        watcher_rx: None,
        watcher_handle: None,
        status_flash: None,
        force_redraw: false,
        cache_config: if args.index {
            let model = args.model_repo.clone().unwrap_or_else(|| {
                if args.fast || args.text {
                    "BAAI/bge-small-en-v1.5".to_string()
                } else {
                    "nomic-ai/modernbert-embed-base".to_string()
                }
            });
            Some(tui::CacheConfig {
                root: std::path::PathBuf::from(&args.path),
                model_repo: model,
                cache_dir: args.cache_dir.as_ref().map(std::path::PathBuf::from),
            })
        } else {
            None
        },
    };

    // Set up file watcher if --index mode
    if args.index {
        use notify::{RecursiveMode, Watcher};
        let (tx, rx) = std::sync::mpsc::channel();
        let mut watcher = notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
            if let Ok(event) = res {
                use notify::EventKind;
                if matches!(
                    event.kind,
                    EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_)
                ) {
                    let _ = tx.send(());
                }
            }
        })
        .context("failed to create file watcher")?;
        watcher
            .watch(std::path::Path::new(&args.path), RecursiveMode::Recursive)
            .context("failed to watch directory")?;
        app.watcher_rx = Some(rx);
        app.watcher_handle = Some(watcher);
    }

    tui::run(app)
}

/// One-shot search mode: embed, rank, print results, exit.
#[expect(
    clippy::too_many_lines,
    reason = "two code paths (--index vs stateless) in one function"
)]
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

    // Resolve model repo for cache compatibility check
    let model_repo = args.model_repo.clone().unwrap_or_else(|| {
        if args.fast || args.text {
            "BAAI/bge-small-en-v1.5".to_string()
        } else {
            "nomic-ai/modernbert-embed-base".to_string()
        }
    });

    // If --index, use persistent cache; otherwise embed from scratch
    let results = if args.index {
        // Clear and rebuild if --reindex
        if args.reindex {
            let cache_dir = ripvec_core::cache::reindex::resolve_cache_dir(
                std::path::Path::new(&args.path),
                &model_repo,
                args.cache_dir.as_deref().map(std::path::Path::new),
            );
            tracing::info!(cache_dir = %cache_dir.display(), "reindex requested; clearing cache");
            let _ = std::fs::remove_dir_all(&cache_dir);
        }

        let cache_dir = ripvec_core::cache::reindex::resolve_cache_dir(
            std::path::Path::new(&args.path),
            &model_repo,
            args.cache_dir.as_deref().map(std::path::Path::new),
        );
        tracing::info!(
            root = %args.path,
            cache_dir = %cache_dir.display(),
            repo_level = args.repo_level,
            "loading persistent index"
        );

        let pb = use_progress.then(|| progress::spinner("Loading index\u{2026}"));

        let (index, stats) = ripvec_core::cache::reindex::incremental_index(
            std::path::Path::new(&args.path),
            &backend_refs,
            tokenizer,
            search_cfg,
            profiler,
            &model_repo,
            args.cache_dir.as_deref().map(std::path::Path::new),
            args.repo_level,
        )
        .context("incremental index failed")?;

        if args.repo_level {
            prompt_auto_stash(std::path::Path::new(&args.path));
        }

        if let Some(pb) = &pb {
            if stats.chunks_reembedded > 0 {
                pb.finish_with_message(format!(
                    "Indexed {} chunks ({} re-embedded, {} cached)",
                    stats.chunks_total, stats.chunks_reembedded, stats.files_unchanged,
                ));
            } else {
                pb.finish_with_message(format!(
                    "Loaded {} chunks from cache ({}ms)",
                    stats.chunks_total, stats.duration_ms,
                ));
            }
        }

        // incremental_index now returns HybridIndex directly (semantic + BM25)
        let hybrid = index;

        // Embed query (skip for keyword-only mode)
        let query_embedding = if search_cfg.mode == ripvec_core::hybrid::SearchMode::Keyword {
            tracing::info!("keyword mode selected; skipping query embedding");
            vec![0.0f32; hybrid.semantic.hidden_dim]
        } else {
            tracing::info!("embedding query");
            let enc = ripvec_core::tokenize::tokenize_query(
                &args.query,
                tokenizer,
                backend_refs[0].max_tokens(),
            )?;
            let mut query_emb = backend_refs[0].embed_batch(&[enc])?;
            query_emb
                .pop()
                .ok_or_else(|| anyhow::anyhow!("backend returned no embedding"))?
        };

        // `top_k = 0` on the CLI means "all matches above threshold". Earlier
        // revisions passed `usize::MAX` here, which later overflowed inside
        // tantivy's TopDocs and in the semantic `pre_filter_k = top_k * 10`
        // math. Clamp to the actual corpus size — every chunk could match and
        // no more, so anything beyond that is wasted capacity and overflow bait.
        let corpus_size = hybrid.chunks().len().max(1);
        let effective_top_k = if args.top_k > 0 {
            args.top_k.min(corpus_size)
        } else {
            corpus_size
        };
        tracing::info!(
            corpus_size,
            requested_top_k = args.top_k,
            effective_top_k,
            threshold = args.threshold,
            mode = ?search_cfg.mode,
            "ranking search results"
        );
        let ranked = hybrid.search(
            &query_embedding,
            &args.query,
            effective_top_k,
            args.threshold,
            search_cfg.mode,
        );
        ranked
            .into_iter()
            .filter_map(|(idx, score)| {
                hybrid
                    .chunks()
                    .get(idx)
                    .map(|chunk| ripvec_core::embed::SearchResult {
                        chunk: chunk.clone(),
                        similarity: score,
                    })
            })
            .collect::<Vec<_>>()
    } else {
        // Original stateless path
        let spinner = use_progress.then(|| progress::spinner("Searching\u{2026}"));
        let embed_display: std::sync::Arc<std::sync::Mutex<Option<progress::EmbedDisplay>>> =
            std::sync::Arc::new(std::sync::Mutex::new(None));
        let live_profiler;
        let effective_profiler: &ripvec_core::profile::Profiler = if use_progress {
            let spinner_clone = spinner.as_ref().unwrap().clone();
            let display = embed_display.clone();
            live_profiler = ripvec_core::profile::Profiler::with_callback(
                std::time::Duration::ZERO,
                move |_msg| {},
            )
            .with_embed_tick(move |p| {
                let mut guard = display.lock().unwrap();
                let d = guard.get_or_insert_with(|| {
                    spinner_clone.finish_and_clear();
                    progress::EmbedDisplay::new(p.total as u64)
                });
                d.update(p);
            });
            &live_profiler
        } else {
            profiler
        };

        let search_results = ripvec_core::embed::search(
            std::path::Path::new(&args.path),
            &args.query,
            &backend_refs,
            tokenizer,
            args.top_k,
            search_cfg,
            effective_profiler,
        )
        .context("search failed")?;

        // Finish whichever progress widget is active
        {
            let mut guard = embed_display.lock().unwrap();
            if let Some(d) = guard.take() {
                d.finish_and_clear();
            } else if let Some(spinner) = spinner {
                spinner.finish_and_clear();
            }
        }

        search_results
    };

    profiler.finish();

    // Scores are already normalized to [0,1] and threshold-filtered by HybridIndex::search().
    output::print_results(&results, &args.format);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{cli, log_level_from_verbosity, should_run_index_dashboard};
    use clap::Parser;

    #[test]
    fn verbosity_maps_to_increasing_log_levels() {
        assert_eq!(log_level_from_verbosity(0), None);
        assert_eq!(log_level_from_verbosity(1), Some("info"));
        assert_eq!(log_level_from_verbosity(2), Some("debug"));
        assert_eq!(log_level_from_verbosity(3), Some("trace"));
        assert_eq!(log_level_from_verbosity(9), Some("trace"));
    }

    #[test]
    fn index_dashboard_runs_for_index_without_query() {
        let args = cli::Args::try_parse_from(["ripvec", "--index"]).unwrap();
        assert!(should_run_index_dashboard(&args));

        let args = cli::Args::try_parse_from(["ripvec", "--index", "-i"]).unwrap();
        assert!(should_run_index_dashboard(&args));

        let args =
            cli::Args::try_parse_from(["ripvec", "--index", "--reindex", "--repo-level"]).unwrap();
        assert!(should_run_index_dashboard(&args));
    }

    #[test]
    fn index_dashboard_does_not_run_when_query_is_present() {
        let args = cli::Args::try_parse_from(["ripvec", "--index", "cache invalidation"]).unwrap();
        assert!(!should_run_index_dashboard(&args));

        let args = cli::Args::try_parse_from(["ripvec", "-i"]).unwrap();
        assert!(!should_run_index_dashboard(&args));
    }
}
