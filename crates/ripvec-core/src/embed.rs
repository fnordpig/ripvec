//! Parallel batch embedding pipeline.
//!
//! Three-phase architecture: discover files (I/O-bound via `ignore`),
//! chunk in parallel (CPU-bound via `rayon`), then embed in parallel
//! batches with per-thread ONNX sessions sharing the same memory-mapped model.
//!
//! # Batch inference
//!
//! Instead of one `session.run()` per chunk, chunks are grouped into batches
//! of configurable size (default 32). Each batch is tokenized, padded to
//! the longest sequence, and run as a single ONNX call with shape
//! `[batch_size, max_seq_len]`. This amortizes per-call overhead and enables
//! SIMD across the batch dimension.

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use rayon::prelude::*;

use crate::chunk::CodeChunk;
use crate::model::{EmbeddingModel, Encoding};
use crate::similarity;

/// Default batch size for embedding inference.
pub const DEFAULT_BATCH_SIZE: usize = 32;

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
/// in parallel batches, and returns the top-k results ranked by similarity.
///
/// `batch_size` controls how many chunks are embedded per ONNX inference call.
/// Larger batches amortize overhead but use more memory. Default is 32.
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
    batch_size: usize,
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
        let enc = tokenize(query, tokenizer)?;
        let mut results = crate::model::embed_batch(&mut session, &[enc])?;
        results.pop().unwrap_or_default()
    };

    // Phase 4: Embed all chunks in parallel batches
    profiler.embed_begin(chunks.len());
    let done_counter = AtomicUsize::new(0);
    let bs = batch_size.max(1);

    let mut results: Vec<SearchResult> = chunks
        .par_chunks(bs)
        .map_init(
            || {
                model
                    .create_session()
                    .expect("failed to create ONNX session")
            },
            |session, batch| {
                // Tokenize entire batch
                let encodings: Vec<Encoding> = batch
                    .iter()
                    .filter_map(|chunk| tokenize(&chunk.content, tokenizer).ok())
                    .collect();

                // Run batch inference
                let Ok(embeddings) = crate::model::embed_batch(session, &encodings) else {
                    return vec![];
                };

                let done = done_counter.fetch_add(batch.len(), Ordering::Relaxed) + batch.len();
                profiler.embed_tick(done);

                // Pair embeddings with chunks and compute similarity
                batch
                    .iter()
                    .zip(embeddings)
                    .map(|(chunk, emb)| {
                        let sim = similarity::dot_product(&query_embedding, &emb);
                        SearchResult {
                            chunk: chunk.clone(),
                            similarity: sim,
                        }
                    })
                    .collect::<Vec<_>>()
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

/// Tokenize text into an [`Encoding`] ready for ONNX inference.
fn tokenize(text: &str, tokenizer: &tokenizers::Tokenizer) -> crate::Result<Encoding> {
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| crate::Error::Tokenization(e.to_string()))?;
    Ok(Encoding {
        input_ids: encoding.get_ids().iter().map(|&x| i64::from(x)).collect(),
        attention_mask: encoding
            .get_attention_mask()
            .iter()
            .map(|&x| i64::from(x))
            .collect(),
        token_type_ids: encoding
            .get_type_ids()
            .iter()
            .map(|&x| i64::from(x))
            .collect(),
    })
}
