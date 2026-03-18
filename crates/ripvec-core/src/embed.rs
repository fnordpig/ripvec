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
    /// BERT attention cost scales linearly with token count, and CLS pooling
    /// means the first token's representation carries most semantic weight.
    /// Default: 128 (7.7× faster than 512, with minimal quality loss).
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
            max_tokens: 0,
            chunk: ChunkConfig::default(),
            sort_order: SortOrder::None,
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

    // Phase 2: Embed query
    let query_embedding = {
        let _span = info_span!("embed_query").entered();
        let _guard = profiler.phase("embed_query");
        let enc = tokenize(query, tokenizer, cfg.max_tokens)?;
        let mut results = crate::model::embed_batch(model, &[enc])?;
        results.pop().unwrap_or_default()
    };

    // Phase 3: Embed all chunks — strategy depends on device
    let bs = cfg.batch_size.max(1);
    let max_tokens_cfg = cfg.max_tokens;
    let _span = info_span!("embed_chunks", chunk_count = chunks.len(), batch_size = bs).entered();
    profiler.embed_begin(chunks.len());

    let is_gpu = !matches!(model.device(), candle_core::Device::Cpu);
    let embeddings: Vec<Vec<f32>> = if is_gpu {
        embed_gpu_pipelined(&chunks, model, tokenizer, max_tokens_cfg, bs, profiler)?
    } else {
        embed_cpu_parallel(&chunks, model, tokenizer, max_tokens_cfg, bs, profiler)?
    };
    profiler.embed_done();

    // Phase 4: Parallel similarity ranking (rayon — just dot products)
    let mut results: Vec<SearchResult> = {
        let _span = info_span!("rank", chunk_count = chunks.len()).entered();
        let guard = profiler.phase("rank");

        let scored: Vec<SearchResult> = embeddings
            .into_par_iter()
            .zip(chunks.par_iter())
            .map(|(emb, chunk)| SearchResult {
                chunk: chunk.clone(),
                similarity: similarity::dot_product(&query_embedding, &emb),
            })
            .collect();

        guard.set_detail(format!(
            "top {} from {}",
            top_k.min(scored.len()),
            scored.len()
        ));
        scored
    };

    results.sort_unstable_by(|a, b| {
        b.similarity
            .partial_cmp(&a.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(top_k);

    Ok(results)
}

/// CPU embedding: rayon-parallel, one model clone per thread.
///
/// Each rayon thread gets a cheap Arc-clone of the model and runs
/// independent BLAS calls. Accelerate/MKL handles per-call SIMD but
/// multiple concurrent calls saturate all cores.
fn embed_cpu_parallel(
    chunks: &[CodeChunk],
    model: &EmbeddingModel,
    tokenizer: &tokenizers::Tokenizer,
    max_tokens: usize,
    batch_size: usize,
    profiler: &crate::profile::Profiler,
) -> crate::Result<Vec<Vec<f32>>> {
    let done_counter = std::sync::atomic::AtomicUsize::new(0);
    let first_error: std::sync::Mutex<Option<crate::Error>> = std::sync::Mutex::new(None);

    let results: Vec<Vec<f32>> = chunks
        .par_chunks(batch_size)
        .map_init(
            || model.clone(),
            |thread_model, batch| {
                // Skip remaining batches if we already hit an error
                if first_error.lock().ok().is_some_and(|e| e.is_some()) {
                    return vec![vec![]; batch.len()];
                }

                let encodings: Vec<Encoding> = batch
                    .iter()
                    .filter_map(|chunk| tokenize(&chunk.content, tokenizer, max_tokens).ok())
                    .collect();

                match crate::model::embed_batch(thread_model, &encodings) {
                    Ok(batch_embeddings) => {
                        let done = done_counter
                            .fetch_add(batch.len(), std::sync::atomic::Ordering::Relaxed)
                            + batch.len();
                        profiler.embed_tick(done);
                        batch_embeddings
                    }
                    Err(e) => {
                        if let Ok(mut guard) = first_error.lock()
                            && guard.is_none()
                        {
                            *guard = Some(e);
                        }
                        vec![vec![]; batch.len()]
                    }
                }
            },
        )
        .flatten()
        .collect();

    // If any batch failed, return the first error
    if let Some(err) = first_error.into_inner().ok().flatten() {
        return Err(err);
    }

    Ok(results)
}

/// GPU embedding with bounded producer-consumer pipeline.
///
/// A producer thread tokenizes batches on CPU (rayon-parallel) and sends
/// them through a bounded channel. The main thread feeds tokenized
/// batches to the GPU. When the buffer is full the producer blocks,
/// bounding peak memory for tokenized data to `O(RING_SIZE * gpu_bs)`.
///
/// For candle Metal/CUDA, each `embed_batch` call does:
///   1. `Tensor::from_vec` — CPU→device transfer
///   2. `model.forward`   — device compute
///   3. `to_vec2`         — device→CPU transfer
fn embed_gpu_pipelined(
    chunks: &[CodeChunk],
    model: &EmbeddingModel,
    tokenizer: &tokenizers::Tokenizer,
    max_tokens: usize,
    batch_size: usize,
    profiler: &crate::profile::Profiler,
) -> crate::Result<Vec<Vec<f32>>> {
    use std::sync::mpsc;

    /// Max tokenized batches buffered ahead of GPU consumption.
    const RING_SIZE: usize = 4;

    type BatchMsg = (Vec<Encoding>, Vec<bool>);

    let gpu_bs = batch_size * 4;
    let (tx, rx) = mpsc::sync_channel::<BatchMsg>(RING_SIZE);

    let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(chunks.len());
    let mut done = 0usize;
    let mut first_error: Option<crate::Error> = None;

    std::thread::scope(|s| {
        s.spawn(|| {
            for chunk_batch in chunks.chunks(gpu_bs) {
                let encodings: Vec<Option<Encoding>> = chunk_batch
                    .par_iter()
                    .map(|chunk| tokenize(&chunk.content, tokenizer, max_tokens).ok())
                    .collect();

                let mut valid = Vec::with_capacity(encodings.len());
                let mut mask = Vec::with_capacity(encodings.len());
                for enc in encodings {
                    if let Some(e) = enc {
                        valid.push(e);
                        mask.push(true);
                    } else {
                        mask.push(false);
                    }
                }

                if tx.send((valid, mask)).is_err() {
                    break;
                }
            }
            drop(tx);
        });

        for (valid, mask) in rx {
            let batch_len = mask.len();

            if valid.is_empty() {
                embeddings.extend(mask.iter().map(|_| vec![]));
                done += batch_len;
                profiler.embed_tick(done);
                continue;
            }

            match crate::model::embed_batch(model, &valid) {
                Ok(batch_embeddings) => {
                    let mut emb_iter = batch_embeddings.into_iter();
                    for ok in &mask {
                        if *ok {
                            embeddings.push(emb_iter.next().unwrap_or_default());
                        } else {
                            embeddings.push(vec![]);
                        }
                    }
                }
                Err(e) => {
                    first_error = Some(e);
                    break;
                }
            }

            done += batch_len;
            profiler.embed_tick(done);
        }
    });

    if let Some(err) = first_error {
        return Err(err);
    }

    Ok(embeddings)
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

/// Hard limit matching bge-small-en-v1.5 `max_position_embeddings`.
/// Sequences at or above this length cause an OOB error in the model.
const MODEL_MAX_TOKENS: usize = 512;

/// Tokenize text into an [`Encoding`] ready for model inference.
///
/// Always truncates to [`MODEL_MAX_TOKENS`] (the model's position embedding
/// limit). When `max_tokens` is non-zero, further truncates to that value.
/// CLS pooling means the first token's representation carries most semantic
/// weight, so truncation has minimal quality impact.
fn tokenize(
    text: &str,
    tokenizer: &tokenizers::Tokenizer,
    max_tokens: usize,
) -> crate::Result<Encoding> {
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| crate::Error::Tokenization(e.to_string()))?;

    let full_len = encoding.get_ids().len();
    let mut len = full_len.min(MODEL_MAX_TOKENS);
    if max_tokens > 0 {
        len = len.min(max_tokens);
    }
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
