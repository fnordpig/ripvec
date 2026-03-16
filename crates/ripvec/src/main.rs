mod cli;
mod output;

use anyhow::{Context, Result};
use clap::Parser;
use std::sync::Mutex;

fn main() -> Result<()> {
    let args = cli::Args::parse();

    // Configure thread pool
    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .context("failed to configure thread pool")?;
    }

    // Load model and tokenizer
    let model = ripvec_core::model::EmbeddingModel::load(&args.model_repo, &args.model_file)
        .context("failed to load embedding model")?;
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
    )
    .context("search failed")?;

    // Filter by threshold and print
    let filtered: Vec<_> = results
        .into_iter()
        .filter(|r| r.similarity >= args.threshold)
        .collect();
    output::print_results(&filtered, &args.format);

    Ok(())
}
