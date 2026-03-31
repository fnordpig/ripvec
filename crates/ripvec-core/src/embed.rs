//! Parallel batch embedding pipeline with streaming backpressure.
//!
//! Two pipeline modes:
//!
//! - **Batch mode** (< `STREAMING_THRESHOLD` files): walk, chunk all, tokenize
//!   all, sort by length, embed. Simple and optimal for small corpora.
//!
//! - **Streaming mode** (>= `STREAMING_THRESHOLD` files): three-stage pipeline
//!   with bounded channels. Chunks flow through: rayon chunk workers ->
//!   tokenize+batch collector -> GPU embed consumer. The GPU starts after the
//!   first `batch_size` encodings are ready (~50ms), not after all chunks are
//!   done. Backpressure prevents unbounded memory growth.
//!
//! # Batch inference
//!
//! Instead of one forward pass per chunk, chunks are grouped into batches
//! of configurable size (default 32). Each batch is tokenized, padded to
//! the longest sequence, and run as a single forward pass with shape
//! `[batch_size, max_seq_len]`. This amortizes per-call overhead and enables
//! SIMD across the batch dimension.
//!
//! # Parallelism
//!
//! On CPU, each rayon thread gets its own backend clone (cheap — most
//! backends use `Arc`'d weights internally). On GPU, batches run sequentially
//! from Rust while the device parallelizes internally.

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use rayon::prelude::*;
use tracing::{debug, info_span, instrument, warn};

use crate::backend::{EmbedBackend, Encoding};
use crate::chunk::{ChunkConfig, CodeChunk};

/// Default batch size for embedding inference.
pub const DEFAULT_BATCH_SIZE: usize = 32;

/// File count threshold for switching from batch to streaming pipeline.
///
/// Below this, the batch path (chunk all -> tokenize all -> sort -> embed)
/// is simpler and allows global sort-by-length optimization. Above this,
/// streaming eliminates GPU idle time during chunking/tokenization.
const STREAMING_THRESHOLD: usize = 1000;

/// Number of batch-sized buffers in the embed channel for backpressure.
///
/// Keeps memory bounded: at most `RING_SIZE * batch_size` encodings in flight.
/// Matches the ring-buffer depth documented on [`EmbedBackend`].
const RING_SIZE: usize = 4;

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
    /// Force all files to be chunked as plain text (sliding windows only).
    /// When `false` (default), files with recognized extensions use tree-sitter
    /// semantic chunking, and unrecognized extensions fall back to sliding windows.
    pub text_mode: bool,
    /// MRL cascade pre-filter dimension.
    ///
    /// When set, [`SearchIndex`](crate::index::SearchIndex) stores a truncated
    /// and L2-re-normalized copy of the embedding matrix at this dimension for
    /// fast two-phase cascade search. `None` (default) disables cascade search.
    pub cascade_dim: Option<usize>,
    /// Optional file type filter (e.g. "rust", "python", "js").
    ///
    /// When set, only files matching this type (using ripgrep's built-in type
    /// database) are collected during the walk phase.
    pub file_type: Option<String>,
    /// Search mode: hybrid (default), semantic, or keyword.
    pub mode: crate::hybrid::SearchMode,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            batch_size: DEFAULT_BATCH_SIZE,
            max_tokens: 0,
            chunk: ChunkConfig::default(),
            text_mode: false,
            cascade_dim: None,
            file_type: None,
            mode: crate::hybrid::SearchMode::Hybrid,
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

/// Walk, chunk, and embed all files in a directory.
///
/// Returns the chunks and their corresponding embedding vectors.
/// This is the building block for both one-shot search and interactive mode.
/// The caller handles query embedding and ranking.
///
/// Accepts multiple backends for hybrid scheduling — chunks are distributed
/// across all backends via work-stealing (see [`embed_distributed`]).
///
/// Automatically selects between two pipeline modes:
/// - **Batch** (< `STREAMING_THRESHOLD` files): chunk all, tokenize all, sort
///   by length, embed. Optimal for small corpora.
/// - **Streaming** (>= `STREAMING_THRESHOLD` files): three-stage pipeline with
///   bounded channels. GPU starts after the first batch is ready, not after all
///   chunks are done. Eliminates GPU idle time during chunking/tokenization.
///
/// # Errors
///
/// Returns an error if file walking, chunking, or embedding fails.
#[instrument(skip_all, fields(root = %root.display(), batch_size = cfg.batch_size))]
pub fn embed_all(
    root: &Path,
    backends: &[&dyn EmbedBackend],
    tokenizer: &tokenizers::Tokenizer,
    cfg: &SearchConfig,
    profiler: &crate::profile::Profiler,
) -> crate::Result<(Vec<CodeChunk>, Vec<Vec<f32>>)> {
    if backends.is_empty() {
        return Err(crate::Error::Other(anyhow::anyhow!(
            "no embedding backends provided"
        )));
    }

    // Phase 1: Collect files (respects .gitignore, filters by extension)
    let files = {
        let _span = info_span!("walk").entered();
        let guard = profiler.phase("walk");
        let files = crate::walk::collect_files(root, cfg.file_type.as_deref());
        guard.set_detail(format!("{} files", files.len()));
        files
    };

    if files.len() >= STREAMING_THRESHOLD {
        embed_all_streaming(&files, backends, tokenizer, cfg, profiler)
    } else {
        embed_all_batch(&files, backends, tokenizer, cfg, profiler)
    }
}

/// Batch pipeline: chunk all -> tokenize all -> sort by length -> embed.
///
/// Optimal for small corpora where the global sort-by-length optimization
/// matters more than eliminating GPU idle time.
fn embed_all_batch(
    files: &[std::path::PathBuf],
    backends: &[&dyn EmbedBackend],
    tokenizer: &tokenizers::Tokenizer,
    cfg: &SearchConfig,
    profiler: &crate::profile::Profiler,
) -> crate::Result<(Vec<CodeChunk>, Vec<Vec<f32>>)> {
    // Phase 2: Chunk all files in parallel.
    let chunks: Vec<CodeChunk> = {
        let _span = info_span!("chunk", file_count = files.len()).entered();
        let chunk_start = Instant::now();
        let text_mode = cfg.text_mode;
        let result: Vec<CodeChunk> = files
            .par_iter()
            .flat_map(|path| {
                let Some(source) = read_source(path) else {
                    return vec![];
                };
                let chunks = if text_mode {
                    crate::chunk::chunk_text(path, &source, &cfg.chunk)
                } else {
                    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
                    match crate::languages::config_for_extension(ext) {
                        Some(lang_config) => {
                            crate::chunk::chunk_file(path, &source, &lang_config, &cfg.chunk)
                        }
                        None => crate::chunk::chunk_text(path, &source, &cfg.chunk),
                    }
                };
                profiler.chunk_thread_report(chunks.len());
                chunks
            })
            .collect();
        profiler.chunk_summary(result.len(), files.len(), chunk_start.elapsed());
        result
    };

    // Phase 3: Pre-tokenize all chunks in parallel (CPU-bound, all rayon threads)
    let bs = cfg.batch_size.max(1);
    let max_tokens_cfg = cfg.max_tokens;
    let model_max = backends[0].max_tokens();
    let _span = info_span!("embed_chunks", chunk_count = chunks.len(), batch_size = bs).entered();
    profiler.embed_begin(chunks.len());

    let all_encodings: Vec<Option<Encoding>> = chunks
        .par_iter()
        .map(|chunk| {
            tokenize(
                &chunk.enriched_content,
                tokenizer,
                max_tokens_cfg,
                model_max,
            )
            .inspect_err(|e| {
                warn!(file = %chunk.file_path, err = %e, "tokenization failed, skipping chunk");
            })
            .ok()
        })
        .collect();

    // Sort chunks and their encodings together by descending token count.
    // This groups similar-length sequences into the same batch, minimizing
    // padding waste (short chunks no longer get padded to a long neighbour).
    let mut paired: Vec<(CodeChunk, Option<Encoding>)> =
        chunks.into_iter().zip(all_encodings).collect();
    paired.sort_by(|a, b| {
        let len_a = a.1.as_ref().map_or(0, |e| e.input_ids.len());
        let len_b = b.1.as_ref().map_or(0, |e| e.input_ids.len());
        len_b.cmp(&len_a) // descending — longest first
    });
    let (chunks, sorted_encodings): (Vec<CodeChunk>, Vec<Option<Encoding>>) =
        paired.into_iter().unzip();

    // Phase 4: Distribute pre-tokenized batches across all backends
    let embeddings = embed_distributed(&sorted_encodings, backends, bs, profiler)?;
    profiler.embed_done();

    // Filter out chunks whose tokenization failed (empty embedding vectors).
    let (chunks, embeddings): (Vec<_>, Vec<_>) = chunks
        .into_iter()
        .zip(embeddings)
        .filter(|(_, emb)| !emb.is_empty())
        .unzip();

    Ok((chunks, embeddings))
}

/// Streaming pipeline: chunk -> tokenize -> batch -> embed with backpressure.
///
/// Three concurrent stages connected by bounded channels:
///
/// 1. **Chunk producers** (rayon pool, in a scoped thread): read + parse files,
///    send chunks to channel.
/// 2. **Tokenize + batch collector** (scoped thread): tokenize chunks, sort
///    within batch windows, send full batches to the embed channel.
/// 3. **Embed consumer** (main thread): calls `embed_distributed` on each
///    batch, collects results.
///
/// The bounded channels provide natural backpressure: if the GPU falls behind,
/// the tokenize stage blocks, which blocks chunk producers via the chunk channel.
/// If chunking is fast and the GPU is slow, at most
/// `8 * batch_size + RING_SIZE * batch_size` items are in memory.
///
/// Uses `std::thread::scope` so all threads can borrow the caller's stack
/// (`tokenizer`, `backends`, `profiler`) without `'static` bounds.
#[expect(
    clippy::too_many_lines,
    reason = "streaming pipeline has inherent complexity in thread coordination"
)]
fn embed_all_streaming(
    files: &[std::path::PathBuf],
    backends: &[&dyn EmbedBackend],
    tokenizer: &tokenizers::Tokenizer,
    cfg: &SearchConfig,
    profiler: &crate::profile::Profiler,
) -> crate::Result<(Vec<CodeChunk>, Vec<Vec<f32>>)> {
    use crossbeam_channel::bounded;

    let bs = cfg.batch_size.max(1);
    let max_tokens_cfg = cfg.max_tokens;
    let model_max = backends[0].max_tokens();
    let file_count = files.len();
    let text_mode = cfg.text_mode;
    let chunk_config = cfg.chunk.clone();

    // Bounded channel from chunk producers -> tokenize+batch stage.
    // Factor of 8 gives enough buffering for rayon parallelism without
    // unbounded growth (at most ~8 batches worth of chunks in flight).
    let (chunk_tx, chunk_rx) = bounded::<CodeChunk>(bs * 8);

    // Bounded channel from tokenize+batch stage -> embed consumer.
    // RING_SIZE batches in flight provides enough pipeline depth for GPU
    // to stay busy while the next batch is being tokenized.
    let (batch_tx, batch_rx) = bounded::<Vec<(Encoding, CodeChunk)>>(RING_SIZE);

    // Shared counters for profiling across the streaming pipeline.
    let total_chunks_produced = AtomicUsize::new(0);
    let embed_done_counter = AtomicUsize::new(0);
    let chunk_start = Instant::now();

    // All stages run inside std::thread::scope so they can borrow from the
    // caller's stack (tokenizer, backends, profiler, files, etc.).
    std::thread::scope(|scope| {
        // --- Stage 1: Chunk producers (rayon inside a scoped thread) ---
        //
        // Spawns a scoped thread that drives rayon's par_iter. Each file is
        // chunked independently and chunks are sent into the bounded channel.
        // If the channel is full, rayon workers block, providing backpressure.
        scope.spawn(|| {
            let _span = info_span!("chunk_stream", file_count).entered();
            files.par_iter().for_each(|path| {
                let Some(source) = read_source(path) else {
                    return;
                };
                let chunks = if text_mode {
                    crate::chunk::chunk_text(path, &source, &chunk_config)
                } else {
                    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
                    match crate::languages::config_for_extension(ext) {
                        Some(lang_config) => {
                            crate::chunk::chunk_file(path, &source, &lang_config, &chunk_config)
                        }
                        None => crate::chunk::chunk_text(path, &source, &chunk_config),
                    }
                };
                let n = chunks.len();
                for chunk in chunks {
                    // Channel disconnected means downstream errored; stop.
                    if chunk_tx.send(chunk).is_err() {
                        return;
                    }
                }
                profiler.chunk_thread_report(n);
                total_chunks_produced.fetch_add(n, Ordering::Relaxed);
            });
            // chunk_tx is dropped here, closing the channel — but the borrow
            // of chunk_tx lives until the scoped thread ends. We need to
            // explicitly drop it so the tokenize stage sees the channel close.
            drop(chunk_tx);
        });

        // --- Stage 2: Tokenize + batch collector (scoped thread) ---
        //
        // Receives individual chunks, tokenizes each (HuggingFace tokenizer
        // is Send + Sync), and accumulates into batch-sized buffers. Within
        // each buffer, entries are sorted by descending token count — the same
        // padding-reduction optimization as the batch path, applied locally.
        let tokenize_handle = scope.spawn(move || -> crate::Result<()> {
            let _span = info_span!("tokenize_stream").entered();
            let mut buffer: Vec<(Encoding, CodeChunk)> = Vec::with_capacity(bs);

            for chunk in &chunk_rx {
                match tokenize(
                    &chunk.enriched_content,
                    tokenizer,
                    max_tokens_cfg,
                    model_max,
                ) {
                    Ok(encoding) => {
                        buffer.push((encoding, chunk));
                        if buffer.len() >= bs {
                            // Sort within batch by descending token count.
                            buffer.sort_by(|a, b| b.0.input_ids.len().cmp(&a.0.input_ids.len()));
                            let batch = std::mem::replace(&mut buffer, Vec::with_capacity(bs));
                            if batch_tx.send(batch).is_err() {
                                // Embed consumer dropped; stop tokenizing.
                                return Ok(());
                            }
                        }
                    }
                    Err(e) => {
                        warn!(
                            file = %chunk.file_path, err = %e,
                            "tokenization failed, skipping chunk"
                        );
                    }
                }
            }

            // Flush remaining partial batch.
            if !buffer.is_empty() {
                buffer.sort_by(|a, b| b.0.input_ids.len().cmp(&a.0.input_ids.len()));
                let _ = batch_tx.send(buffer);
            }
            // batch_tx drops here, closing the embed channel.

            Ok(())
        });

        // --- Stage 3: Embed consumer (main thread within scope) ---
        //
        // Receives sorted batches, embeds via the backend(s), collects results.
        // Profiler is driven from here since this thread owns the reference.
        let _span = info_span!("embed_stream").entered();

        // Total isn't known upfront in streaming mode; start at 0 and update.
        profiler.embed_begin(0);

        let mut all_chunks: Vec<CodeChunk> = Vec::new();
        let mut all_embeddings: Vec<Vec<f32>> = Vec::new();
        let mut embed_error: Option<crate::Error> = None;

        for batch in &batch_rx {
            // Update the profiler total as more chunks are produced.
            let current_total = total_chunks_produced.load(Ordering::Relaxed);
            if current_total > 0 {
                profiler.embed_begin_update_total(current_total);
            }

            let (encodings, chunks): (Vec<Encoding>, Vec<CodeChunk>) = batch.into_iter().unzip();

            // Wrap as Option<Encoding> for embed_distributed compatibility.
            let opt_encodings: Vec<Option<Encoding>> = encodings.into_iter().map(Some).collect();

            match embed_distributed(&opt_encodings, backends, bs, profiler) {
                Ok(batch_embeddings) => {
                    let done = embed_done_counter.fetch_add(chunks.len(), Ordering::Relaxed)
                        + chunks.len();
                    profiler.embed_tick(done);

                    for (chunk, emb) in chunks.into_iter().zip(batch_embeddings) {
                        if !emb.is_empty() {
                            all_chunks.push(chunk);
                            all_embeddings.push(emb);
                        }
                    }
                }
                Err(e) => {
                    embed_error = Some(e);
                    // break exits the for loop; batch_rx drops naturally after.
                    break;
                }
            }
        }

        // Report chunk summary now that all stages have completed (or errored).
        let final_total = total_chunks_produced.load(Ordering::Relaxed);
        profiler.chunk_summary(final_total, file_count, chunk_start.elapsed());
        profiler.embed_done();

        // Wait for tokenize thread and check for errors.
        let tokenize_result = tokenize_handle.join();

        // Error priority: embed > tokenize > thread panic.
        if let Some(e) = embed_error {
            return Err(e);
        }
        match tokenize_result {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(e),
            Err(_) => {
                return Err(crate::Error::Other(anyhow::anyhow!(
                    "tokenize thread panicked"
                )));
            }
        }

        Ok((all_chunks, all_embeddings))
    })
}

/// Search a directory for code chunks semantically similar to a query.
///
/// Walks the directory, chunks all supported files, embeds everything
/// in parallel batches, and returns the top-k results ranked by similarity.
///
/// Accepts multiple backends for hybrid scheduling — the first backend
/// (`backends[0]`) is used for query embedding.
///
/// All tuning parameters (batch size, token limit, chunk sizing) are
/// controlled via [`SearchConfig`].
///
/// # Errors
///
/// Returns an error if the query cannot be tokenized or embedded.
///
/// # Panics
///
/// Panics if a per-thread backend clone fails during parallel embedding
/// (should not happen if the backend loaded successfully).
#[instrument(skip_all, fields(root = %root.display(), top_k, batch_size = cfg.batch_size))]
pub fn search(
    root: &Path,
    query: &str,
    backends: &[&dyn EmbedBackend],
    tokenizer: &tokenizers::Tokenizer,
    top_k: usize,
    cfg: &SearchConfig,
    profiler: &crate::profile::Profiler,
) -> crate::Result<Vec<SearchResult>> {
    if backends.is_empty() {
        return Err(crate::Error::Other(anyhow::anyhow!(
            "no embedding backends provided"
        )));
    }

    // Phases 1, 2, 3, 4: walk, chunk, pre-tokenize, embed all files
    let (chunks, embeddings) = embed_all(root, backends, tokenizer, cfg, profiler)?;

    let t_query_start = std::time::Instant::now();

    // Phase 5: Build hybrid index (semantic + BM25)
    let hybrid = {
        let _span = info_span!("build_hybrid_index").entered();
        let _guard = profiler.phase("build_hybrid_index");
        crate::hybrid::HybridIndex::new(chunks, embeddings, cfg.cascade_dim)?
    };

    let mode = cfg.mode;
    let effective_top_k = if top_k > 0 { top_k } else { usize::MAX };

    // Phase 6: Embed query (skip for keyword-only mode)
    let query_embedding = if mode == crate::hybrid::SearchMode::Keyword {
        // Keyword mode: no embedding needed, use zero vector
        let dim = hybrid.semantic.hidden_dim;
        vec![0.0f32; dim]
    } else {
        let _span = info_span!("embed_query").entered();
        let _guard = profiler.phase("embed_query");
        let t_tok = std::time::Instant::now();
        let enc = tokenize(query, tokenizer, cfg.max_tokens, backends[0].max_tokens())?;
        let tok_ms = t_tok.elapsed().as_secs_f64() * 1000.0;
        let t_emb = std::time::Instant::now();
        let mut results = backends[0].embed_batch(&[enc])?;
        let emb_ms = t_emb.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "[search] query: tokenize={tok_ms:.1}ms embed={emb_ms:.1}ms total_since_embed_all={:.1}ms",
            t_query_start.elapsed().as_secs_f64() * 1000.0
        );
        results.pop().ok_or_else(|| {
            crate::Error::Other(anyhow::anyhow!("backend returned no embedding for query"))
        })?
    };

    // Phase 7: Hybrid/semantic/keyword ranking
    let ranked = {
        let _span = info_span!("rank", chunk_count = hybrid.chunks().len()).entered();
        let guard = profiler.phase("rank");
        // Threshold only applies to semantic modes; keyword/hybrid use RRF scores
        let threshold = if mode == crate::hybrid::SearchMode::Semantic {
            0.0 // SearchIndex::rank applies its own threshold
        } else {
            0.0
        };
        let results = hybrid.search(&query_embedding, query, effective_top_k, threshold, mode);
        guard.set_detail(format!(
            "{mode} top {} from {}",
            effective_top_k.min(results.len()),
            hybrid.chunks().len()
        ));
        results
    };

    let results: Vec<SearchResult> = ranked
        .into_iter()
        .map(|(idx, score)| SearchResult {
            chunk: hybrid.chunks()[idx].clone(),
            similarity: score,
        })
        .collect();

    Ok(results)
}

/// Shared state for [`embed_distributed`] workers.
struct DistributedState<'a> {
    tokenized: &'a [Option<Encoding>],
    cursor: std::sync::atomic::AtomicUsize,
    error_flag: std::sync::atomic::AtomicBool,
    first_error: std::sync::Mutex<Option<crate::Error>>,
    done_counter: std::sync::atomic::AtomicUsize,
    batch_size: usize,
    profiler: &'a crate::profile::Profiler,
}

impl DistributedState<'_> {
    /// Worker loop: claim batches from the shared cursor, embed, collect results.
    fn run_worker(&self, backend: &dyn EmbedBackend) -> Vec<(usize, Vec<f32>)> {
        use std::sync::atomic::Ordering;

        let n = self.tokenized.len();
        // GPU backends grab larger batches to amortize per-call overhead.
        // MLX's lazy eval graph optimizer benefits from large matrices.
        // Metal sub-batches internally via MAX_BATCH to limit padding waste.
        let grab_size = if backend.is_gpu() {
            self.batch_size * 4
        } else {
            self.batch_size
        };
        let mut results = Vec::new();

        loop {
            if self.error_flag.load(Ordering::Relaxed) {
                break;
            }

            let start = self.cursor.fetch_add(grab_size, Ordering::Relaxed);
            if start >= n {
                break;
            }
            let end = (start + grab_size).min(n);
            let batch = &self.tokenized[start..end];

            // Separate valid encodings from Nones, tracking which indices succeeded
            let mut valid = Vec::with_capacity(batch.len());
            let mut valid_indices = Vec::with_capacity(batch.len());
            for (i, enc) in batch.iter().enumerate() {
                if let Some(e) = enc {
                    // TODO(perf): cloning 3 Vecs per chunk; consider making
                    // `EmbedBackend::embed_batch` accept `&[&Encoding]` to avoid this.
                    valid.push(e.clone());
                    valid_indices.push(start + i);
                } else {
                    results.push((start + i, vec![]));
                }
            }

            if valid.is_empty() {
                let done =
                    self.done_counter.fetch_add(batch.len(), Ordering::Relaxed) + batch.len();
                self.profiler.embed_tick(done);
                continue;
            }

            match backend.embed_batch(&valid) {
                Ok(batch_embeddings) => {
                    for (idx, emb) in valid_indices.into_iter().zip(batch_embeddings) {
                        results.push((idx, emb));
                    }
                    let done =
                        self.done_counter.fetch_add(batch.len(), Ordering::Relaxed) + batch.len();
                    self.profiler.embed_tick(done);
                }
                Err(e) => {
                    self.error_flag.store(true, Ordering::Relaxed);
                    if let Ok(mut guard) = self.first_error.lock()
                        && guard.is_none()
                    {
                        *guard = Some(e);
                    }
                    break;
                }
            }
        }

        results
    }
}

/// Distribute pre-tokenized chunks across multiple backends using work-stealing.
///
/// Each backend gets a dedicated worker thread. Workers compete on a shared
/// `AtomicUsize` cursor to claim batches of chunks. GPU backends grab larger
/// batches (`batch_size * 4`), CPU backends grab smaller ones (`batch_size`).
/// Results are written by original chunk index — no merge step needed.
///
/// When `backends` has a single entry, no extra threads are spawned.
///
/// # Errors
///
/// Returns the first error from any backend. Other workers exit early
/// when an error is detected.
#[expect(
    unsafe_code,
    reason = "BLAS thread count must be set via env vars before spawning workers"
)]
pub(crate) fn embed_distributed(
    tokenized: &[Option<Encoding>],
    backends: &[&dyn EmbedBackend],
    batch_size: usize,
    profiler: &crate::profile::Profiler,
) -> crate::Result<Vec<Vec<f32>>> {
    let n = tokenized.len();
    let state = DistributedState {
        tokenized,
        cursor: std::sync::atomic::AtomicUsize::new(0),
        error_flag: std::sync::atomic::AtomicBool::new(false),
        first_error: std::sync::Mutex::new(None),
        done_counter: std::sync::atomic::AtomicUsize::new(0),
        batch_size: batch_size.max(1),
        profiler,
    };

    // Collect (index, embedding) pairs from all workers
    let all_pairs: Vec<(usize, Vec<f32>)> =
        if backends.len() == 1 && backends[0].supports_clone() && !backends[0].is_gpu() {
            // Single cloneable CPU backend: spawn N workers with single-threaded BLAS.
            //
            // BLAS libraries (OpenBLAS, MKL) internally spawn threads for each matmul.
            // For small matrices ([1,384]×[384,384]), this thread overhead dominates —
            // profiling shows 80% of time in sched_yield (thread contention).
            //
            // Instead: force BLAS to single-thread per worker, parallelize across
            // independent BERT inferences. Each worker gets its own cloned backend.
            // Force BLAS libraries to single-threaded mode.
            // We parallelize across independent BERT inferences instead.
            // env vars don't always work (OpenBLAS may ignore after init),
            // so also call the runtime API directly.
            unsafe {
                std::env::set_var("OPENBLAS_NUM_THREADS", "1");
                std::env::set_var("MKL_NUM_THREADS", "1");
                std::env::set_var("VECLIB_MAXIMUM_THREADS", "1"); // macOS Accelerate

                // Direct FFI to set BLAS thread count — works even after init
                #[cfg(all(not(target_os = "macos"), feature = "cpu"))]
                {
                    unsafe extern "C" {
                        fn openblas_set_num_threads(num: std::ffi::c_int);
                    }
                    openblas_set_num_threads(1);
                }
            }

            let num_workers = rayon::current_num_threads().max(1);
            std::thread::scope(|s| {
                let handles: Vec<_> = (0..num_workers)
                    .map(|_| {
                        s.spawn(|| {
                            // Per-thread: force single-threaded BLAS (thread-local setting).
                            // On macOS 15+ this calls BLASSetThreading; on Linux openblas_set_num_threads.
                            #[cfg(any(feature = "cpu", feature = "cpu-accelerate"))]
                            crate::backend::driver::cpu::force_single_threaded_blas();
                            let cloned = backends[0].clone_backend();
                            state.run_worker(cloned.as_ref())
                        })
                    })
                    .collect();
                let mut all = Vec::new();
                for handle in handles {
                    if let Ok(pairs) = handle.join() {
                        all.extend(pairs);
                    }
                }
                all
            })
        } else if backends.len() == 1 {
            // Single non-cloneable backend (GPU or CPU ModernBERT): run on the calling thread.
            // GPU backends handle parallelism internally; CPU uses BLAS internal
            // multi-threading (Accelerate/OpenBLAS) for intra-GEMM parallelism.
            state.run_worker(backends[0])
        } else {
            // Multiple backends: one thread per backend via std::thread::scope
            std::thread::scope(|s| {
                let handles: Vec<_> = backends
                    .iter()
                    .map(|&backend| {
                        s.spawn(|| {
                            // CPU backends that support cloning get a thread-local copy
                            if backend.supports_clone() {
                                let cloned = backend.clone_backend();
                                state.run_worker(cloned.as_ref())
                            } else {
                                state.run_worker(backend)
                            }
                        })
                    })
                    .collect();

                let mut all = Vec::new();
                for handle in handles {
                    if let Ok(pairs) = handle.join() {
                        all.extend(pairs);
                    } else {
                        warn!("worker thread panicked");
                        state
                            .error_flag
                            .store(true, std::sync::atomic::Ordering::Relaxed);
                    }
                }
                all
            })
        };

    // Check for errors before assembling results
    if let Some(err) = state.first_error.into_inner().ok().flatten() {
        return Err(err);
    }

    // Scatter results into output vec by original index
    let mut embeddings: Vec<Vec<f32>> = vec![vec![]; n];
    for (idx, emb) in all_pairs {
        embeddings[idx] = emb;
    }

    Ok(embeddings)
}

/// Read a source file into a `String`, skipping binary files.
///
/// Reads the file as raw bytes first, checks for NUL bytes in the first 8 KB
/// to detect binary files, then converts to UTF-8. Returns `None` (with a
/// debug log) when the file cannot be read, is binary, or is not valid UTF-8.
pub(crate) fn read_source(path: &Path) -> Option<String> {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            debug!(path = %path.display(), err = %e, "skipping file: read failed");
            return None;
        }
    };

    // Skip binary files: NUL byte anywhere in the first 8 KB is a reliable signal.
    if memchr::memchr(0, &bytes[..bytes.len().min(8192)]).is_some() {
        debug!(path = %path.display(), "skipping binary file");
        return None;
    }

    match std::str::from_utf8(&bytes) {
        Ok(s) => Some(s.to_string()),
        Err(e) => {
            debug!(path = %path.display(), err = %e, "skipping file: not valid UTF-8");
            None
        }
    }
}

/// Tokenize text into an [`Encoding`] ready for model inference.
///
/// Delegates to [`crate::tokenize::tokenize_query`] for the core encoding,
/// then applies an additional `max_tokens` truncation when non-zero.
/// CLS pooling means the first token's representation carries most semantic
/// weight, so truncation has minimal quality impact.
fn tokenize(
    text: &str,
    tokenizer: &tokenizers::Tokenizer,
    max_tokens: usize,
    model_max_tokens: usize,
) -> crate::Result<Encoding> {
    let mut enc = crate::tokenize::tokenize_query(text, tokenizer, model_max_tokens)?;
    if max_tokens > 0 {
        let len = enc.input_ids.len().min(max_tokens);
        enc.input_ids.truncate(len);
        enc.attention_mask.truncate(len);
        enc.token_type_ids.truncate(len);
    }
    Ok(enc)
}

/// Normalize similarity scores to `[0,1]` and apply a `PageRank` structural boost.
///
/// Each result's similarity is min-max normalized, then a weighted `PageRank`
/// score is added: `final = normalized + alpha * pagerank`. This promotes
/// architecturally important files (many dependents) in search results.
///
/// Called from the MCP search handler which has access to the `RepoGraph`,
/// rather than from [`search`] directly.
pub fn apply_structural_boost<S: ::std::hash::BuildHasher>(
    results: &mut [SearchResult],
    file_ranks: &std::collections::HashMap<String, f32, S>,
    alpha: f32,
) {
    if results.is_empty() || alpha == 0.0 {
        return;
    }

    let min = results
        .iter()
        .map(|r| r.similarity)
        .fold(f32::INFINITY, f32::min);
    let max = results
        .iter()
        .map(|r| r.similarity)
        .fold(f32::NEG_INFINITY, f32::max);
    let range = (max - min).max(1e-12);

    for r in results.iter_mut() {
        let normalized = (r.similarity - min) / range;
        let pr = file_ranks.get(&r.chunk.file_path).copied().unwrap_or(0.0);
        r.similarity = normalized + alpha * pr;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "cpu")]
    #[ignore = "loads model + embeds full source tree; run with `cargo test -- --ignored`"]
    fn search_with_backend_trait() {
        let backend = crate::backend::load_backend(
            crate::backend::BackendKind::Cpu,
            "BAAI/bge-small-en-v1.5",
            crate::backend::DeviceHint::Cpu,
            None,
        )
        .unwrap();
        let tokenizer = crate::tokenize::load_tokenizer("BAAI/bge-small-en-v1.5").unwrap();
        let cfg = SearchConfig::default();
        let profiler = crate::profile::Profiler::noop();
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let results = search(
            &dir,
            "embedding model",
            &[backend.as_ref()],
            &tokenizer,
            1,
            &cfg,
            &profiler,
        );
        assert!(results.is_ok());
        assert!(!results.unwrap().is_empty());
    }

    #[test]
    #[cfg(feature = "cpu")]
    fn embed_distributed_produces_correct_count() {
        let backend = crate::backend::load_backend(
            crate::backend::BackendKind::Cpu,
            "BAAI/bge-small-en-v1.5",
            crate::backend::DeviceHint::Cpu,
            None,
        )
        .unwrap();
        let tokenizer = crate::tokenize::load_tokenizer("BAAI/bge-small-en-v1.5").unwrap();
        let profiler = crate::profile::Profiler::noop();

        // Tokenize a few strings
        let texts = ["fn hello() {}", "class Foo:", "func main() {}"];
        let encoded: Vec<Option<Encoding>> = texts
            .iter()
            .map(|t| super::tokenize(t, &tokenizer, 0, 512).ok())
            .collect();

        let results =
            super::embed_distributed(&encoded, &[backend.as_ref()], 32, &profiler).unwrap();

        assert_eq!(results.len(), 3);
        // All should be 384-dim (bge-small hidden size)
        for (i, emb) in results.iter().enumerate() {
            assert_eq!(emb.len(), 384, "embedding {i} should be 384-dim");
        }
    }

    /// Truncate an embedding to `dims` dimensions and L2-normalize.
    fn truncate_and_normalize(emb: &[f32], dims: usize) -> Vec<f32> {
        let trunc = &emb[..dims];
        let norm: f32 = trunc.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        trunc.iter().map(|x| x / norm).collect()
    }

    /// Rank corpus embeddings against a query, return top-K chunk indices.
    fn rank_topk(query: &[f32], corpus: &[Vec<f32>], k: usize) -> Vec<usize> {
        let mut scored: Vec<(usize, f32)> = corpus
            .iter()
            .enumerate()
            .map(|(i, emb)| {
                let dot: f32 = query.iter().zip(emb).map(|(a, b)| a * b).sum();
                (i, dot)
            })
            .collect();
        scored.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        scored.into_iter().take(k).map(|(i, _)| i).collect()
    }

    /// MRL retrieval recall test: does truncated search retrieve the same results?
    ///
    /// Embeds the ripvec codebase at full dimension, then tests whether
    /// truncating to fewer dimensions retrieves the same top-10 results.
    /// This is the real MRL quality test — per-vector cosine is trivially 1.0
    /// but retrieval recall can degrade if the first N dims don't preserve
    /// relative ordering between different vectors.
    #[test]
    #[ignore = "loads model + embeds; run with --nocapture"]
    #[expect(
        clippy::cast_precision_loss,
        reason = "top_k and overlap are small counts"
    )]
    fn mrl_retrieval_recall() {
        let model = "BAAI/bge-small-en-v1.5";
        let backends = crate::backend::detect_backends(model, None).unwrap();
        let tokenizer = crate::tokenize::load_tokenizer(model).unwrap();
        let cfg = SearchConfig::default();
        let profiler = crate::profile::Profiler::noop();

        // Embed the ripvec source tree
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap();
        eprintln!("Embedding {}", root.display());
        let backend_refs: Vec<&dyn crate::backend::EmbedBackend> =
            backends.iter().map(std::convert::AsRef::as_ref).collect();
        let (chunks, embeddings) =
            embed_all(root, &backend_refs, &tokenizer, &cfg, &profiler).unwrap();
        let full_dim = embeddings[0].len();
        eprintln!(
            "Corpus: {} chunks, {full_dim}-dim embeddings\n",
            chunks.len()
        );

        // Test queries spanning different semantic intents
        let queries = [
            "error handling in the embedding pipeline",
            "tree-sitter chunking and AST parsing",
            "Metal GPU kernel dispatch",
            "file watcher for incremental reindex",
            "cosine similarity ranking",
        ];

        let top_k = 10;
        let mrl_dims: Vec<usize> = [32, 64, 128, 192, 256, full_dim]
            .into_iter()
            .filter(|&d| d <= full_dim)
            .collect();

        eprintln!("=== MRL Retrieval Recall@{top_k} (vs full {full_dim}-dim) ===\n");

        for query in &queries {
            // Embed query at full dim
            let enc = tokenize(query, &tokenizer, 0, backends[0].max_tokens()).unwrap();
            let query_emb = backends[0].embed_batch(&[enc]).unwrap().pop().unwrap();

            // Full-dim reference ranking
            let ref_topk = rank_topk(&query_emb, &embeddings, top_k);

            eprintln!("Query: \"{query}\"");
            eprintln!(
                "  Full-dim top-1: {} ({})",
                chunks[ref_topk[0]].name, chunks[ref_topk[0]].file_path
            );

            for &dims in &mrl_dims {
                // Truncate corpus and query
                let trunc_corpus: Vec<Vec<f32>> = embeddings
                    .iter()
                    .map(|e| truncate_and_normalize(e, dims))
                    .collect();
                let trunc_query = truncate_and_normalize(&query_emb, dims);

                let trunc_topk = rank_topk(&trunc_query, &trunc_corpus, top_k);

                // Recall@K: how many of the full-dim top-K appear in truncated top-K
                let overlap = ref_topk.iter().filter(|i| trunc_topk.contains(i)).count();
                let recall = overlap as f32 / top_k as f32;
                let marker = if dims == full_dim {
                    " (ref)"
                } else if recall >= 0.8 {
                    " ***"
                } else {
                    ""
                };
                eprintln!(
                    "  dims={dims:>3}: Recall@{top_k}={recall:.1} ({overlap}/{top_k}){marker}"
                );
            }
            eprintln!();
        }
    }

    fn make_result(file_path: &str, similarity: f32) -> SearchResult {
        SearchResult {
            chunk: CodeChunk {
                file_path: file_path.to_string(),
                name: "test".to_string(),
                kind: "function".to_string(),
                start_line: 1,
                end_line: 10,
                enriched_content: String::new(),
                content: String::new(),
            },
            similarity,
        }
    }

    #[test]
    fn structural_boost_normalizes_and_applies() {
        let mut results = vec![
            make_result("src/a.rs", 0.8),
            make_result("src/b.rs", 0.4),
            make_result("src/c.rs", 0.6),
        ];
        let mut ranks = std::collections::HashMap::new();
        ranks.insert("src/a.rs".to_string(), 0.5);
        ranks.insert("src/b.rs".to_string(), 1.0);
        ranks.insert("src/c.rs".to_string(), 0.0);

        apply_structural_boost(&mut results, &ranks, 0.2);

        // a: normalized=(0.8-0.4)/0.4=1.0, boost=0.2*0.5=0.1 => 1.1
        assert!((results[0].similarity - 1.1).abs() < 1e-6);
        // b: normalized=(0.4-0.4)/0.4=0.0, boost=0.2*1.0=0.2 => 0.2
        assert!((results[1].similarity - 0.2).abs() < 1e-6);
        // c: normalized=(0.6-0.4)/0.4=0.5, boost=0.2*0.0=0.0 => 0.5
        assert!((results[2].similarity - 0.5).abs() < 1e-6);
    }

    #[test]
    fn structural_boost_noop_on_empty() {
        let mut results: Vec<SearchResult> = vec![];
        let ranks = std::collections::HashMap::new();
        apply_structural_boost(&mut results, &ranks, 0.2);
        assert!(results.is_empty());
    }

    #[test]
    fn structural_boost_noop_on_zero_alpha() {
        let mut results = vec![make_result("src/a.rs", 0.8)];
        let mut ranks = std::collections::HashMap::new();
        ranks.insert("src/a.rs".to_string(), 1.0);
        apply_structural_boost(&mut results, &ranks, 0.0);
        // Should be unchanged
        assert!((results[0].similarity - 0.8).abs() < 1e-6);
    }
}
