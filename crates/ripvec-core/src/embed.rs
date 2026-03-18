//! Parallel batch embedding pipeline.
//!
//! Three-phase architecture: discover files (I/O-bound via `ignore`),
//! chunk in parallel (CPU-bound via `rayon`), then embed in parallel
//! batches using candle BERT models.
//!
//! # Batch inference
//!
//! Instead of one forward pass per chunk, chunks are grouped into batches
//! of configurable size (default 32). Each batch is tokenized, padded to
//! the longest sequence, and run as a single forward pass with shape
//! `[batch_size, max_seq_len]`. This amortizes per-call overhead and enables
//! SIMD across the batch dimension.
//!
//! # Scheduling
//!
//! Chunks are sorted by content length **descending** (longest first) before
//! batching. This ensures the heaviest batches run first while all threads
//! are busy, and short chunks fill in the gaps at the end — classic
//! longest-job-first scheduling for optimal load balancing.
//!
//! # Parallelism
//!
//! On CPU, each rayon thread gets its own model clone (cheap — `BertModel`
//! uses `Arc`'d weights internally). On GPU, batches run sequentially from
//! Rust while the device parallelizes internally.

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use memmap2::Mmap;
use rayon::prelude::*;
use tracing::{info_span, instrument};

use crate::chunk::{ChunkConfig, CodeChunk};
use crate::model::{EmbeddingModel, Encoding};
use crate::similarity;

/// Files larger than this are memory-mapped instead of read into a String.
/// Mmap avoids heap allocation and lets the OS page cache handle I/O.
const MMAP_THRESHOLD: u64 = 32 * 1024; // 32 KB

/// Default batch size for embedding inference.
pub const DEFAULT_BATCH_SIZE: usize = 32;

/// Chunk scheduling order for the embedding pipeline.
///
/// Controls how chunks are sorted before batching. Sorting longest-first
/// (`Descending`) is the classic longest-job-first heuristic for optimal
/// load balancing across rayon threads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SortOrder {
    /// Longest chunks first (default — best load balance).
    #[default]
    Descending,
    /// Shortest chunks first.
    Ascending,
    /// No sorting — process chunks in file-walk order.
    None,
}

/// Runtime configuration for the search pipeline.
///
/// All tuning parameters that were previously compile-time constants are
/// gathered here so they can be set from CLI arguments without recompiling.
#[derive(Debug, Clone)]
pub struct SearchConfig {
    /// Chunks per inference call. Larger values amortize call overhead
    /// but consume more memory. Default: 32.
    pub batch_size: usize,
    /// Maximum tokens fed to the model per chunk. `0` means no limit.
    /// Capping tokens controls inference cost for minified or dense source.
    /// Default: 256 (CLS pooling makes early tokens most important).
    pub max_tokens: usize,
    /// Chunking parameters forwarded to the chunking phase.
    pub chunk: ChunkConfig,
    /// Sort order for chunks before batching.
    pub sort_order: SortOrder,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            batch_size: DEFAULT_BATCH_SIZE,
            max_tokens: 256,
            chunk: ChunkConfig::default(),
            sort_order: SortOrder::Descending,
        }
    }
}

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
/// All tuning parameters (batch size, token limit, chunk sizing, sort order)
/// are controlled via [`SearchConfig`].
///
/// # Errors
///
/// Returns an error if the query cannot be tokenized or embedded.
///
/// # Panics
///
/// Panics if a per-thread model clone fails during parallel embedding
/// (should not happen if the model loaded successfully).
#[expect(
    clippy::too_many_lines,
    reason = "multi-phase pipeline is inherently sequential"
)]
#[instrument(skip_all, fields(root = %root.display(), top_k, batch_size = cfg.batch_size))]
pub fn search(
    root: &Path,
    query: &str,
    model: &EmbeddingModel,
    tokenizer: &tokenizers::Tokenizer,
    top_k: usize,
    cfg: &SearchConfig,
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
                let Some(lang_config) = crate::languages::config_for_extension(ext) else {
                    return vec![];
                };
                let Some(source) = read_source(path) else {
                    return vec![];
                };
                let chunks = crate::chunk::chunk_file(path, &source, &lang_config, &cfg.chunk);
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
        let enc = tokenize(query, tokenizer, cfg.max_tokens)?;
        let mut results = crate::model::embed_batch(model, &[enc])?;
        results.pop().unwrap_or_default()
    };

    // Phase 4: Embed all chunks in parallel batches.
    // Sort order is configurable: Descending (longest first) is the classic
    // longest-job-first heuristic for optimal load balancing across rayon threads.
    let mut indexed_chunks: Vec<(usize, &CodeChunk)> = chunks.iter().enumerate().collect();
    match cfg.sort_order {
        SortOrder::Descending => {
            indexed_chunks.sort_unstable_by(|(_, a), (_, b)| b.content.len().cmp(&a.content.len()));
        }
        SortOrder::Ascending => {
            indexed_chunks.sort_unstable_by(|(_, a), (_, b)| a.content.len().cmp(&b.content.len()));
        }
        SortOrder::None => {}
    }

    let batch_size = cfg.batch_size;
    let max_tokens_cfg = cfg.max_tokens;
    let _span = info_span!("embed_chunks", chunk_count = chunks.len(), batch_size).entered();
    profiler.embed_begin(chunks.len());
    let done_counter = AtomicUsize::new(0);
    let bs = batch_size.max(1);

    let is_gpu = !matches!(model.device(), candle_core::Device::Cpu);

    let mut results: Vec<SearchResult> = if is_gpu {
        // GPU: sequential from Rust, device parallelizes internally.
        // Use larger batch size to keep the GPU busy.
        let gpu_bs = bs * 4;
        let mut all_results = Vec::new();
        for batch in indexed_chunks.chunks(gpu_bs) {
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

            let (indices, encodings, max_tok) = {
                let _tok_span = info_span!("tokenize_batch").entered();
                let mut indices = Vec::with_capacity(batch.len());
                let mut encodings = Vec::with_capacity(batch.len());
                let mut max_tok = 0usize;
                for &(idx, chunk) in batch {
                    if let Ok(enc) = tokenize(&chunk.content, tokenizer, max_tokens_cfg) {
                        max_tok = max_tok.max(enc.input_ids.len());
                        indices.push(idx);
                        encodings.push(enc);
                    }
                }
                (indices, encodings, max_tok)
            };

            let embeddings = {
                let _inf_span =
                    info_span!("inference", batch_len = encodings.len(), max_tok).entered();
                match crate::model::embed_batch(model, &encodings) {
                    Ok(e) => e,
                    Err(_) => continue,
                }
            };

            let done = done_counter.fetch_add(batch.len(), Ordering::Relaxed) + batch.len();
            profiler.embed_tick(done);

            all_results.extend(indices.iter().zip(embeddings).map(|(&idx, emb)| {
                let sim = similarity::dot_product(&query_embedding, &emb);
                SearchResult {
                    chunk: chunks[idx].clone(),
                    similarity: sim,
                }
            }));
        }
        all_results
    } else {
        // CPU: parallel with model clones per thread (Arc'd weights, cheap clone).
        indexed_chunks
            .par_chunks(bs)
            .map_init(
                || model.clone(),
                |thread_model, batch| {
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
                    let (indices, encodings, max_tok) = {
                        let _tok_span = info_span!("tokenize_batch").entered();
                        let mut indices = Vec::with_capacity(batch.len());
                        let mut encodings = Vec::with_capacity(batch.len());
                        let mut max_tok = 0usize;
                        for &(idx, chunk) in batch {
                            if let Ok(enc) = tokenize(&chunk.content, tokenizer, max_tokens_cfg) {
                                max_tok = max_tok.max(enc.input_ids.len());
                                indices.push(idx);
                                encodings.push(enc);
                            }
                        }
                        (indices, encodings, max_tok)
                    };

                    // Run batch inference
                    let embeddings = {
                        let _inf_span =
                            info_span!("inference", batch_len = encodings.len(), max_tok).entered();
                        match crate::model::embed_batch(thread_model, &encodings) {
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
            .collect()
    };
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

/// Tokenize text into an [`Encoding`] ready for model inference.
///
/// When `max_tokens` is non-zero, truncates to that many tokens to cap
/// inference cost regardless of text density (minified JS can have 3-4x more
/// tokens per byte than formatted code). CLS pooling only uses the first
/// token's representation, so the beginning of a definition carries most
/// semantic weight. Pass `0` for no limit.
fn tokenize(
    text: &str,
    tokenizer: &tokenizers::Tokenizer,
    max_tokens: usize,
) -> crate::Result<Encoding> {
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| crate::Error::Tokenization(e.to_string()))?;

    let full_len = encoding.get_ids().len();
    let len = if max_tokens == 0 {
        full_len
    } else {
        full_len.min(max_tokens)
    };
    Ok(Encoding {
        input_ids: encoding.get_ids()[..len]
            .iter()
            .map(|&x| i64::from(x))
            .collect(),
        attention_mask: encoding.get_attention_mask()[..len]
            .iter()
            .map(|&x| i64::from(x))
            .collect(),
        token_type_ids: encoding.get_type_ids()[..len]
            .iter()
            .map(|&x| i64::from(x))
            .collect(),
    })
}
