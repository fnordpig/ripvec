//! Parallel batch embedding pipeline.
//!
//! Three-phase architecture: discover files (I/O-bound via `ignore`),
//! chunk in parallel (CPU-bound via `rayon`), then embed in parallel
//! batches using any [`EmbedBackend`] implementation.
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
use std::time::Instant;

use memmap2::Mmap;
use rayon::prelude::*;
use tracing::{debug, info_span, instrument, warn};

use crate::backend::{EmbedBackend, Encoding};
use crate::chunk::{ChunkConfig, CodeChunk};
use crate::similarity;

/// Files larger than this are memory-mapped instead of read into a String.
/// Mmap avoids heap allocation and lets the OS page cache handle I/O.
const MMAP_THRESHOLD: u64 = 32 * 1024; // 32 KB

/// Default batch size for embedding inference.
pub const DEFAULT_BATCH_SIZE: usize = 32;

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
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            batch_size: DEFAULT_BATCH_SIZE,
            max_tokens: 0,
            chunk: ChunkConfig::default(),
            text_mode: false,
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
        let files = crate::walk::collect_files(root);
        guard.set_detail(format!("{} files", files.len()));
        files
    };

    // Phase 2: Chunk all files in parallel.
    // Large files are mmap'd to avoid heap allocation; small files use read_to_string.
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
                    // Force plain-text sliding windows for all files
                    crate::chunk::chunk_text(path, &source, &cfg.chunk)
                } else {
                    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
                    match crate::languages::config_for_extension(ext) {
                        Some(lang_config) => {
                            crate::chunk::chunk_file(path, &source, &lang_config, &cfg.chunk)
                        }
                        // Unrecognized extension: fall back to plain-text windows
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
            tokenize(&chunk.content, tokenizer, max_tokens_cfg, model_max)
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

    // Phase 5: Embed query (using the primary backend)
    let query_embedding = {
        let _span = info_span!("embed_query").entered();
        let _guard = profiler.phase("embed_query");
        let enc = tokenize(query, tokenizer, cfg.max_tokens, backends[0].max_tokens())?;
        let mut results = backends[0].embed_batch(&[enc])?;
        results.pop().ok_or_else(|| {
            crate::Error::Other(anyhow::anyhow!("backend returned no embedding for query"))
        })?
    };

    // Phase 5: Parallel similarity ranking (rayon — just dot products)
    let mut results: Vec<SearchResult> = {
        let _span = info_span!("rank", chunk_count = chunks.len()).entered();
        let guard = profiler.phase("rank");

        let scored: Vec<SearchResult> = chunks
            .into_par_iter()
            .zip(embeddings.into_par_iter())
            .map(|(chunk, emb)| SearchResult {
                chunk,
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
    if top_k > 0 {
        results.truncate(top_k);
    }

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
fn embed_distributed(
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
    let all_pairs: Vec<(usize, Vec<f32>)> = if backends.len() == 1 {
        // Single backend: run directly on the main thread, no spawning overhead
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
///
/// Returns `None` (with a debug log) when the file cannot be read or is not valid UTF-8.
#[expect(unsafe_code, reason = "mmap of read-only source file")]
fn read_source(path: &Path) -> Option<SourceText> {
    let metadata = match std::fs::metadata(path) {
        Ok(m) => m,
        Err(e) => {
            debug!(path = %path.display(), err = %e, "skipping file: metadata read failed");
            return None;
        }
    };
    if metadata.len() >= MMAP_THRESHOLD {
        let file = match std::fs::File::open(path) {
            Ok(f) => f,
            Err(e) => {
                debug!(path = %path.display(), err = %e, "skipping file: open failed");
                return None;
            }
        };
        // SAFETY: file is read-only, not modified while mapped
        let mmap = match unsafe { Mmap::map(&file) } {
            Ok(m) => m,
            Err(e) => {
                debug!(path = %path.display(), err = %e, "skipping file: mmap failed");
                return None;
            }
        };
        // Validate UTF-8 before wrapping
        if std::str::from_utf8(&mmap).is_err() {
            debug!(path = %path.display(), "skipping file: not valid UTF-8");
            return None;
        }
        Some(SourceText::Mapped(mmap))
    } else {
        match std::fs::read_to_string(path) {
            Ok(s) => Some(SourceText::Owned(s)),
            Err(e) => {
                debug!(path = %path.display(), err = %e, "skipping file: read failed");
                None
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "loads model + embeds full source tree; run with `cargo test -- --ignored`"]
    fn search_with_backend_trait() {
        let backend = crate::backend::load_backend(
            crate::backend::BackendKind::Candle,
            "BAAI/bge-small-en-v1.5",
            crate::backend::DeviceHint::Cpu,
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
    fn embed_distributed_produces_correct_count() {
        let backend = crate::backend::load_backend(
            crate::backend::BackendKind::Candle,
            "BAAI/bge-small-en-v1.5",
            crate::backend::DeviceHint::Cpu,
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
}
