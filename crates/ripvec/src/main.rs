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

    // Handle --clear-cache early (before loading anything).
    // Clears ALL model variants for this project (removes the project hash dir).
    if args.clear_cache {
        let root = std::path::Path::new(&args.path);
        // Pass a dummy model — we want the parent (project hash) dir.
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
        return Ok(());
    }

    // Resolve model repo: default → ModernBERT, --fast → BGE-small, --code → CodeRankEmbed
    let use_code_model = args.code;
    let use_fast = args.fast || args.text;
    let model_repo = args.model_repo.clone().unwrap_or_else(|| {
        if use_fast {
            "BAAI/bge-small-en-v1.5".to_string()
        } else if use_code_model {
            "nomic-ai/CodeRankEmbed".to_string()
        } else {
            "nomic-ai/modernbert-embed-base".to_string()
        }
    });

    // Default threshold: 0.5 on normalized [0,1] scores (model-agnostic).
    if args.threshold == 0.0 {
        args.threshold = 0.5;
    }

    // For --code mode, prepend the required query prefix (non-interactive only)
    if use_code_model && !args.interactive && !args.query.is_empty() {
        args.query = format!(
            "Represent this query for searching relevant code: {}",
            args.query
        );
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
            use_code_model,
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
        let max_layers = if args.layers > 0 {
            Some(args.layers)
        } else {
            None
        };
        let result = match args.backend {
            cli::BackendArg::Auto => ripvec_core::backend::detect_backends(model_repo, max_layers)
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
                    ripvec_core::backend::load_backend(kind, model_repo, device_hint, max_layers)
                        .context("failed to load embedding backend")?,
                ]
            }
        };
        if let Some(pb) = pb {
            pb.finish_and_clear();
        }
        result
    };
    let tokenizer =
        ripvec_core::tokenize::load_tokenizer(model_repo).context("failed to load tokenizer")?;

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
    use_code_model: bool,
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
        if args.modern {
            "nomic-ai/modernbert-embed-base".to_string()
        } else if use_code_model {
            "nomic-ai/CodeRankEmbed".to_string()
        } else {
            "BAAI/bge-small-en-v1.5".to_string()
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
        )
        .context("incremental index failed")?;

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
        pairs.sort_by(|a, b| b.1.cmp(&a.1)); // most-common extension first
        let breakdown: Vec<String> = pairs
            .into_iter()
            .map(|(ext, count)| format!("{count} .{ext}"))
            .collect();
        format!("{} chunks \u{2502} {}", chunks.len(), breakdown.join(", "))
    };

    let index = ripvec_core::hybrid::HybridIndex::new(chunks, embeddings, None)
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
        query_prefix: if use_code_model {
            "Represent this query for searching relevant code: ".to_string()
        } else {
            String::new()
        },
        watcher_rx: None,
        watcher_handle: None,
        status_flash: None,
        force_redraw: false,
        cache_config: if args.index {
            let model = args.model_repo.clone().unwrap_or_else(|| {
                if use_code_model {
                    "nomic-ai/CodeRankEmbed".to_string()
                } else {
                    "BAAI/bge-small-en-v1.5".to_string()
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
        if args.code {
            "nomic-ai/CodeRankEmbed".to_string()
        } else {
            "BAAI/bge-small-en-v1.5".to_string()
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
            let _ = std::fs::remove_dir_all(&cache_dir);
        }

        let pb = use_progress.then(|| progress::spinner("Loading index\u{2026}"));

        let (index, stats) = ripvec_core::cache::reindex::incremental_index(
            std::path::Path::new(&args.path),
            &backend_refs,
            tokenizer,
            search_cfg,
            profiler,
            &model_repo,
            args.cache_dir.as_deref().map(std::path::Path::new),
        )
        .context("incremental index failed")?;

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
            vec![0.0f32; hybrid.semantic.hidden_dim]
        } else {
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

        let effective_top_k = if args.top_k > 0 {
            args.top_k
        } else {
            usize::MAX
        };
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
