//! Parallel embedding pipeline.
//!
//! Three-phase architecture: discover files (I/O-bound via `ignore`),
//! chunk in parallel (CPU-bound via `rayon`), then embed in parallel
//! with per-thread ONNX sessions sharing the same memory-mapped model.
//!
//! # Thread safety
//!
//! Each rayon thread creates its own [`ort::session::InMemorySession`] via
//! [`EmbeddingModel::create_session`]. Sessions share the underlying mmap
//! memory — only execution metadata is per-thread. This enables fully
//! parallel inference without any mutexes.

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
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
/// Walks the directory, chunks all supported files, embeds everything
/// in parallel, and returns the top-k results ranked by similarity.
///
/// Per-thread ONNX sessions are created via [`EmbeddingModel::create_session`],
/// sharing the memory-mapped model weights. No mutexes are needed.
///
/// Pass a [`crate::profile::Profiler`] to collect per-phase timing; use
/// [`crate::profile::Profiler::noop`] when profiling is not needed.
///
/// # Errors
///
/// Returns an error if the query cannot be tokenized or embedded.
///
/// # Panics
///
/// Panics if a per-thread ONNX session cannot be created during parallel
/// embedding (should not happen if the model loaded successfully).
pub fn search(
    root: &Path,
    query: &str,
    model: &EmbeddingModel,
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

    // Phase 3: Embed the query
    let query_embedding = {
        let _guard = profiler.phase("embed_query");
        let mut session = model.create_session()?;
        embed_text(query, &mut session, tokenizer)?
    };

    // Phase 4: Embed all chunks in parallel with per-thread sessions
    profiler.embed_begin(chunks.len());
    let done_counter = AtomicUsize::new(0);

    let mut results: Vec<SearchResult> = chunks
        .par_iter()
        .map_init(
            || {
                model
                    .create_session()
                    .expect("failed to create ONNX session")
            },
            |session, chunk| {
                let Ok(emb) = embed_text(&chunk.content, session, tokenizer) else {
                    return None;
                };
                let done = done_counter.fetch_add(1, Ordering::Relaxed) + 1;
                profiler.embed_tick(done);
                let sim = similarity::dot_product(&query_embedding, &emb);
                Some(SearchResult {
                    chunk: chunk.clone(),
                    similarity: sim,
                })
            },
        )
        .flatten()
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

/// Tokenize `text` and run it through an ONNX session.
fn embed_text(
    text: &str,
    session: &mut ort::session::InMemorySession<'_>,
    tokenizer: &tokenizers::Tokenizer,
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

    crate::model::embed(session, &ids, &mask, &type_ids)
}
