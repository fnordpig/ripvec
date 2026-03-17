mod cli;
mod output;

use anyhow::{Context, Result};
use clap::Parser;
use std::sync::Mutex;

fn main() -> Result<()> {
    let args = cli::Args::parse();

    // Create profiler
    let profiler = ripvec_core::profile::Profiler::new(
        args.profile,
        std::time::Duration::from_secs_f64(args.profile_interval),
    );

    // Configure thread pool
    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .context("failed to configure thread pool")?;
    }

    // Print profiler header
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    profiler.header(
        env!("CARGO_PKG_VERSION"),
        &args.model_repo,
        rayon::current_num_threads(),
        cores,
    );

    // Load model and tokenizer
    let model = {
        let _guard = profiler.phase("model_load");
        ripvec_core::model::EmbeddingModel::load(&args.model_repo, &args.model_file)
            .context("failed to load embedding model")?
    };
    let model = Mutex::new(model);
    let tokenizer = ripvec_core::tokenize::load_tokenizer(&args.model_repo)
        .context("failed to load tokenizer")?;

    // Run search
    let results = ripvec_core::embed::search(
        std::path::Path::new(&args.path),
        &args.query,
        &model,
        &tokenizer,
        args.top_k,
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

    Ok(())
}
