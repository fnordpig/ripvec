//! Parallel embedding pipeline.
//!
//! Two-phase architecture: discover files (I/O-bound via `ignore`),
//! then chunk and embed in parallel (CPU-bound via `rayon`).
//!
//! # Thread safety note
//!
//! [`EmbeddingModel::embed`] requires `&mut self` because `ort::Session::run`
//! takes `&mut self`. The [`search`] function therefore accepts the model
//! wrapped in a [`std::sync::Mutex`], serialising inference calls while
//! keeping file I/O and chunking parallel.

use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;

use rayon::prelude::*;

use crate::chunk::CodeChunk;
use crate::model::EmbeddingModel;
use crate::similarity;

/// A search result pairing a code chunk with its similarity score.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// The matched code chunk.
    pub chunk: CodeChunk,
    /// Cosine similarity to the query (0.0 to 1.0).
    pub similarity: f32,
}

/// Search a directory for code chunks semantically similar to a query.
///
/// Walks the directory, chunks all supported files, embeds everything,
/// and returns the top-k results ranked by similarity.
///
/// The model is wrapped in a [`Mutex`] because [`EmbeddingModel::embed`]
/// requires mutable access; chunking and file reads remain parallel.
///
/// Pass a [`crate::profile::Profiler`] to collect per-phase timing; use
/// [`crate::profile::Profiler::noop`] when profiling is not needed.
///
/// # Errors
///
/// Returns an error if the query cannot be tokenized or embedded, or if the
/// model mutex is poisoned.
pub fn search(
    root: &Path,
    query: &str,
    model: &Mutex<EmbeddingModel>,
    tokenizer: &tokenizers::Tokenizer,
    top_k: usize,
    profiler: &crate::profile::Profiler,
) -> crate::Result<Vec<SearchResult>> {
    // Phase 1: Collect files (respects .gitignore, filters by extension)
    let files = {
        let guard = profiler.phase("walk");
        let files = crate::walk::collect_files(root);
        guard.set_detail(format!("{} files", files.len()));
        files
    };

    // Phase 2: Chunk all files in parallel
    let chunk_start = Instant::now();
    let chunks: Vec<CodeChunk> = files
        .par_iter()
        .flat_map(|path| {
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            let Some(config) = crate::languages::config_for_extension(ext) else {
                return vec![];
            };
            let Ok(source) = std::fs::read_to_string(path) else {
                return vec![];
            };
            let chunks = crate::chunk::chunk_file(path, &source, &config);
            profiler.chunk_thread_report(chunks.len());
            chunks
        })
        .collect();
    profiler.chunk_summary(chunks.len(), files.len(), chunk_start.elapsed());

    // Phase 3: Embed the query (mutex-guarded mutable borrow)
    let query_embedding = {
        let _guard = profiler.phase("embed_query");
        embed_text(query, model, tokenizer, profiler)?
    };

    // Phase 4: Embed each chunk and compute cosine similarity.
    // Inference is serialised through the Mutex; chunking was the
    // parallel bottleneck, so total throughput is still good.
    profiler.embed_begin(chunks.len());
    let mut results: Vec<SearchResult> = chunks
        .iter()
        .enumerate()
        .filter_map(|(i, chunk)| {
            let emb = embed_text(&chunk.content, model, tokenizer, profiler).ok()?;
            profiler.embed_tick(i + 1);
            let sim = similarity::dot_product(&query_embedding, &emb);
            Some(SearchResult {
                chunk: chunk.clone(),
                similarity: sim,
            })
        })
        .collect();
    profiler.embed_done();

    // Phase 5: Rank by similarity (descending) and keep top-k
    {
        let guard = profiler.phase("rank");
        results.sort_unstable_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(top_k);
        guard.set_detail(format!(
            "top {} from {}",
            top_k.min(results.len()),
            chunks.len()
        ));
    }

    Ok(results)
}

/// Tokenize `text` and run it through the embedding model.
///
/// Acquires the mutex lock for the duration of the ONNX inference call.
/// Timing for lock acquisition and inference is reported to `profiler`.
fn embed_text(
    text: &str,
    model: &Mutex<EmbeddingModel>,
    tokenizer: &tokenizers::Tokenizer,
    profiler: &crate::profile::Profiler,
) -> crate::Result<Vec<f32>> {
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| crate::Error::Tokenization(e.to_string()))?;
    let ids: Vec<i64> = encoding.get_ids().iter().map(|&x| i64::from(x)).collect();
    let mask: Vec<i64> = encoding
        .get_attention_mask()
        .iter()
        .map(|&x| i64::from(x))
        .collect();
    let type_ids: Vec<i64> = encoding
        .get_type_ids()
        .iter()
        .map(|&x| i64::from(x))
        .collect();

    let lock_start = Instant::now();
    let mut guard = model
        .lock()
        .map_err(|e| crate::Error::Other(anyhow::anyhow!("model mutex poisoned: {e}")))?;
    profiler.embed_lock_wait(lock_start.elapsed());

    let infer_start = Instant::now();
    let result = guard.embed(&ids, &mask, &type_ids);
    profiler.embed_inference(infer_start.elapsed());

    result
}
