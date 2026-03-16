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
pub fn search(
    root: &Path,
    query: &str,
    model: &Mutex<EmbeddingModel>,
    tokenizer: &tokenizers::Tokenizer,
    top_k: usize,
) -> crate::Result<Vec<SearchResult>> {
    // Phase 1: Collect files (respects .gitignore, filters by extension)
    let files = crate::walk::collect_files(root);

    // Phase 2: Chunk all files in parallel
    let chunks: Vec<CodeChunk> = files
        .par_iter()
        .flat_map(|path| {
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            let config = match crate::languages::config_for_extension(ext) {
                Some(c) => c,
                None => return vec![],
            };
            let source = match std::fs::read_to_string(path) {
                Ok(s) => s,
                Err(_) => return vec![],
            };
            crate::chunk::chunk_file(path, &source, &config)
        })
        .collect();

    // Phase 3: Embed the query (mutex-guarded mutable borrow)
    let query_embedding = embed_text(query, model, tokenizer)?;

    // Phase 4: Embed each chunk and compute cosine similarity.
    // Inference is serialised through the Mutex; chunking was the
    // parallel bottleneck, so total throughput is still good.
    let mut results: Vec<SearchResult> = chunks
        .iter()
        .filter_map(|chunk| {
            let emb = embed_text(&chunk.content, model, tokenizer).ok()?;
            let sim = similarity::dot_product(&query_embedding, &emb);
            Some(SearchResult {
                chunk: chunk.clone(),
                similarity: sim,
            })
        })
        .collect();

    // Phase 5: Rank by similarity (descending) and keep top-k
    results.sort_unstable_by(|a, b| {
        b.similarity
            .partial_cmp(&a.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(top_k);
    Ok(results)
}

/// Tokenize `text` and run it through the embedding model.
///
/// Acquires the mutex lock for the duration of the ONNX inference call.
fn embed_text(
    text: &str,
    model: &Mutex<EmbeddingModel>,
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

    let mut guard = model
        .lock()
        .map_err(|e| crate::Error::Other(anyhow::anyhow!("model mutex poisoned: {e}")))?;
    guard.embed(&ids, &mask, &type_ids)
}
