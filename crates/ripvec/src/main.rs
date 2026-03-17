mod cli;
mod output;

use anyhow::{Context, Result};
use clap::Parser;
use tracing_subscriber::layer::SubscriberExt;

fn main() -> Result<()> {
    let args = cli::Args::parse();

    // Set up Chrome tracing if `--trace <file>` is specified.
    // The guard must be held until the end of main — dropping it flushes the trace file.
    // We flush explicitly before returning to avoid TLS destruction ordering issues
    // with rayon threads.
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

    // Configure thread pool — over-subscribe by default (2x cores)
    let cores = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1);
    let threads = if args.threads > 0 {
        args.threads
    } else {
        cores * 2
    };
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .context("failed to configure thread pool")?;

    // Print profiler header
    profiler.header(
        env!("CARGO_PKG_VERSION"),
        &args.model_repo,
        rayon::current_num_threads(),
        cores,
    );

    // Load model (mmap'd, no Mutex needed)
    let model = {
        let _guard = profiler.phase("model_load");
        ripvec_core::model::EmbeddingModel::load(&args.model_repo, &args.model_file)
            .context("failed to load embedding model")?
    };
    let tokenizer = ripvec_core::tokenize::load_tokenizer(&args.model_repo)
        .context("failed to load tokenizer")?;

    // Run search (fully parallel — per-thread sessions, no Mutex)
    let results = ripvec_core::embed::search(
        std::path::Path::new(&args.path),
        &args.query,
        &model,
        &tokenizer,
        args.top_k,
        args.batch_size,
        &profiler,
    )
    .context("search failed")?;

    profiler.finish();

    // Filter by threshold and print
    let filtered: Vec<_> = results
        .into_iter()
        .filter(|r| r.similarity >= args.threshold)
        .collect();
    output::print_results(&filtered, &args.format);

    // Explicitly drop the trace guard before rayon's TLS is destroyed
    drop(trace_guard);

    Ok(())
}
