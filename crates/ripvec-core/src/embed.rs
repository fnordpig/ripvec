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
//!
//! # Scheduling
//!
//! Chunks are sorted by content length **descending** (longest first) before
//! batching. This ensures the heaviest batches run first while all threads
//! are busy, and short chunks fill in the gaps at the end — classic
//! longest-job-first scheduling for optimal load balancing.

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use memmap2::Mmap;
use rayon::prelude::*;
use tracing::{info_span, instrument};

use crate::chunk::CodeChunk;
use crate::model::{EmbeddingModel, Encoding};
use crate::similarity;

/// Files larger than this are memory-mapped instead of read into a String.
/// Mmap avoids heap allocation and lets the OS page cache handle I/O.
const MMAP_THRESHOLD: u64 = 32 * 1024; // 32 KB

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
#[expect(
    clippy::too_many_lines,
    reason = "multi-phase pipeline is inherently sequential"
)]
#[instrument(skip_all, fields(root = %root.display(), top_k, batch_size))]
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
        let _span = info_span!("walk").entered();
        let guard = profiler.phase("walk");
        let files = crate::walk::collect_files(root);
        guard.set_detail(format!("{} files", files.len()));
        files
    };

    // Phase 2: Chunk all files in parallel.
    // Large files are mmap'd to avoid heap allocation; small files use read_to_string.
    let chunks: Vec<CodeChunk> = {
        let _span = info_span!("chunk", file_count = files.len()).entered();
        let chunk_start = Instant::now();
        let result: Vec<CodeChunk> = files
            .par_iter()
            .flat_map(|path| {
                let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
                let Some(config) = crate::languages::config_for_extension(ext) else {
                    return vec![];
                };
                let Some(source) = read_source(path) else {
                    return vec![];
                };
                let chunks = crate::chunk::chunk_file(path, &source, &config);
                profiler.chunk_thread_report(chunks.len());
                chunks
            })
            .collect();
        profiler.chunk_summary(result.len(), files.len(), chunk_start.elapsed());
        result
    };

    // Phase 3: Embed the query
    let query_embedding = {
        let _span = info_span!("embed_query").entered();
        let _guard = profiler.phase("embed_query");
        let mut session = model.create_session()?;
        let enc = tokenize(query, tokenizer)?;
        let mut results = crate::model::embed_batch(&mut session, &[enc])?;
        results.pop().unwrap_or_default()
    };

    // Phase 4: Embed all chunks in parallel batches.
    // Sort DESCENDING by content length (longest first) for optimal scheduling:
    // heavy batches run first while all threads are busy, short chunks fill gaps.
    let mut indexed_chunks: Vec<(usize, &CodeChunk)> = chunks.iter().enumerate().collect();
    indexed_chunks.sort_unstable_by(|(_, a), (_, b)| b.content.len().cmp(&a.content.len()));

    let _span = info_span!("embed_chunks", chunk_count = chunks.len(), batch_size).entered();
    profiler.embed_begin(chunks.len());
    let done_counter = AtomicUsize::new(0);
    let bs = batch_size.max(1);

    let mut results: Vec<SearchResult> = indexed_chunks
        .par_chunks(bs)
        .map_init(
            || {
                model
                    .create_session()
                    .expect("failed to create ONNX session")
            },
            |session, batch| {
                let batch_span = info_span!(
                    "batch",
                    size = batch.len(),
                    max_content_len = batch
                        .iter()
                        .map(|(_, c)| c.content.len())
                        .max()
                        .unwrap_or(0),
                );
                let _entered = batch_span.enter();

                // Tokenize entire batch
                let (indices, encodings, max_tokens) = {
                    let _tok_span = info_span!("tokenize_batch").entered();
                    let mut indices = Vec::with_capacity(batch.len());
                    let mut encodings = Vec::with_capacity(batch.len());
                    let mut max_tokens = 0usize;
                    for &(idx, chunk) in batch {
                        if let Ok(enc) = tokenize(&chunk.content, tokenizer) {
                            max_tokens = max_tokens.max(enc.input_ids.len());
                            indices.push(idx);
                            encodings.push(enc);
                        }
                    }
                    (indices, encodings, max_tokens)
                };

                // Run batch inference
                let embeddings = {
                    let _inf_span =
                        info_span!("inference", batch_len = encodings.len(), max_tokens).entered();
                    match crate::model::embed_batch(session, &encodings) {
                        Ok(e) => e,
                        Err(_) => return vec![],
                    }
                };

                let done = done_counter.fetch_add(batch.len(), Ordering::Relaxed) + batch.len();
                profiler.embed_tick(done);

                // Pair embeddings with original chunks and compute similarity
                indices
                    .iter()
                    .zip(embeddings)
                    .map(|(&idx, emb)| {
                        let sim = similarity::dot_product(&query_embedding, &emb);
                        SearchResult {
                            chunk: chunks[idx].clone(),
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
        let _span = info_span!("rank", result_count = results.len()).entered();
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

/// Source text either owned (from `read_to_string`) or mmap'd.
enum SourceText {
    Owned(String),
    Mapped(Mmap),
}

impl std::ops::Deref for SourceText {
    type Target = str;
    fn deref(&self) -> &str {
        match self {
            Self::Owned(s) => s,
            Self::Mapped(m) => {
                // SAFETY: we only construct Mapped from files that passed UTF-8 validation
                #[expect(
                    unsafe_code,
                    reason = "validated UTF-8 before constructing Mapped variant"
                )]
                unsafe {
                    std::str::from_utf8_unchecked(m)
                }
            }
        }
    }
}

/// Read a source file, using mmap for large files and `read_to_string` for small ones.
#[expect(unsafe_code, reason = "mmap of read-only source file")]
fn read_source(path: &Path) -> Option<SourceText> {
    let metadata = std::fs::metadata(path).ok()?;
    if metadata.len() >= MMAP_THRESHOLD {
        let file = std::fs::File::open(path).ok()?;
        // SAFETY: file is read-only, not modified while mapped
        let mmap = unsafe { Mmap::map(&file) }.ok()?;
        // Validate UTF-8 before wrapping
        std::str::from_utf8(&mmap).ok()?;
        Some(SourceText::Mapped(mmap))
    } else {
        std::fs::read_to_string(path).ok().map(SourceText::Owned)
    }
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
