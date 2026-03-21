//! Incremental reindex orchestrator.
//!
//! Ties together the manifest, object store, diff, and embedding pipeline
//! to provide a single `incremental_index` function that loads cached
//! embeddings and only re-embeds changed files.

use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::backend::EmbedBackend;
use crate::cache::diff;
use crate::cache::file_cache::FileCache;
use crate::cache::manifest::Manifest;
use crate::cache::store::ObjectStore;
use crate::chunk::CodeChunk;
use crate::embed::SearchConfig;
use crate::index::SearchIndex;
use crate::profile::Profiler;

/// Statistics from an incremental reindex operation.
#[derive(Debug)]
pub struct ReindexStats {
    /// Total chunks in the final index.
    pub chunks_total: usize,
    /// Chunks that were re-embedded (from dirty files).
    pub chunks_reembedded: usize,
    /// Files unchanged (loaded from cache).
    pub files_unchanged: usize,
    /// Files that were new or modified.
    pub files_changed: usize,
    /// Files removed since last index.
    pub files_deleted: usize,
    /// Wall-clock duration of the reindex.
    pub duration_ms: u64,
}

/// Load or incrementally update a persistent index.
///
/// 1. Resolve cache directory
/// 2. If manifest exists and model matches: Merkle diff, re-embed dirty files
/// 3. If no manifest: full embed from scratch
/// 4. Rebuild `SearchIndex` from all cached objects
///
/// # Errors
///
/// Returns an error if embedding fails or the cache directory is inaccessible.
pub fn incremental_index(
    root: &Path,
    backends: &[&dyn EmbedBackend],
    tokenizer: &tokenizers::Tokenizer,
    cfg: &SearchConfig,
    profiler: &Profiler,
    model_repo: &str,
    cache_dir_override: Option<&Path>,
) -> crate::Result<(SearchIndex, ReindexStats)> {
    let start = Instant::now();

    if backends.is_empty() {
        return Err(crate::Error::Other(anyhow::anyhow!(
            "no embedding backends provided"
        )));
    }

    let cache_dir = resolve_cache_dir(root, cache_dir_override);
    let manifest_path = cache_dir.join("manifest.json");
    let objects_dir = cache_dir.join("objects");
    let store = ObjectStore::new(&objects_dir);

    // Try loading existing manifest
    let existing_manifest = Manifest::load(&manifest_path).ok();

    if let Some(manifest) = existing_manifest.filter(|m| m.is_compatible(model_repo)) {
        // Incremental path: diff → re-embed dirty → merge
        incremental_path(
            root, backends, tokenizer, cfg, profiler, model_repo, &cache_dir, &store, manifest,
            start,
        )
    } else {
        // Cold path: full embed
        full_index_path(
            root, backends, tokenizer, cfg, profiler, model_repo, &cache_dir, &store, start,
        )
    }
}

/// Incremental reindex: diff, re-embed dirty files, merge with cached.
#[expect(clippy::too_many_arguments, reason = "pipeline state passed through")]
#[expect(
    clippy::cast_possible_truncation,
    reason = "duration in ms won't exceed u64"
)]
fn incremental_path(
    root: &Path,
    backends: &[&dyn EmbedBackend],
    tokenizer: &tokenizers::Tokenizer,
    cfg: &SearchConfig,
    profiler: &Profiler,
    _model_repo: &str,
    cache_dir: &Path,
    store: &ObjectStore,
    mut manifest: Manifest,
    start: Instant,
) -> crate::Result<(SearchIndex, ReindexStats)> {
    let diff_result = diff::compute_diff(root, &manifest)?;

    let files_changed = diff_result.dirty.len();
    let files_deleted = diff_result.deleted.len();
    let files_unchanged = diff_result.unchanged;

    // Remove deleted files from manifest
    for deleted in &diff_result.deleted {
        manifest.remove_file(deleted);
    }

    // Re-embed dirty files
    let mut new_chunks_count = 0;
    for dirty_path in &diff_result.dirty {
        let relative = dirty_path
            .strip_prefix(root)
            .unwrap_or(dirty_path)
            .to_string_lossy()
            .to_string();

        // Remove old entry if it exists
        manifest.remove_file(&relative);

        // Chunk this file
        let Some(source) = crate::embed::read_source(dirty_path) else {
            continue;
        };

        let ext = dirty_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");
        let chunks = if cfg.text_mode {
            crate::chunk::chunk_text(dirty_path, &source, &cfg.chunk)
        } else {
            match crate::languages::config_for_extension(ext) {
                Some(lang_config) => {
                    crate::chunk::chunk_file(dirty_path, &source, &lang_config, &cfg.chunk)
                }
                None => crate::chunk::chunk_text(dirty_path, &source, &cfg.chunk),
            }
        };

        if chunks.is_empty() {
            continue;
        }

        // Tokenize
        let model_max = backends[0].max_tokens();
        let encodings: Vec<Option<crate::backend::Encoding>> = chunks
            .iter()
            .map(|chunk| crate::tokenize::tokenize_query(&chunk.content, tokenizer, model_max).ok())
            .collect();

        // Embed
        let embeddings =
            crate::embed::embed_distributed(&encodings, backends, cfg.batch_size, profiler)?;

        // Filter out failed tokenizations
        let (good_chunks, good_embeddings): (Vec<_>, Vec<_>) = chunks
            .into_iter()
            .zip(embeddings.into_iter())
            .filter(|(_, emb)| !emb.is_empty())
            .unzip();

        let hidden_dim = good_embeddings.first().map_or(384, Vec::len);

        // Save to object store
        let content_hash = diff::hash_file(dirty_path)?;
        let file_cache = FileCache {
            chunks: good_chunks.clone(),
            embeddings: good_embeddings.iter().flatten().copied().collect(),
            hidden_dim,
        };
        store.write(&content_hash, &file_cache.to_bytes())?;

        // Update manifest
        let mtime = diff::mtime_secs(dirty_path);
        let size = std::fs::metadata(dirty_path).map_or(0, |m| m.len());
        manifest.add_file(&relative, mtime, size, &content_hash, good_chunks.len());
        new_chunks_count += good_chunks.len();
    }

    // Recompute Merkle hashes
    manifest.recompute_hashes();

    // GC unreferenced objects
    let referenced = manifest.referenced_hashes();
    store.gc(&referenced)?;

    // Save manifest
    manifest.save(&cache_dir.join("manifest.json"))?;

    // Rebuild SearchIndex from all cached objects
    let (all_chunks, all_embeddings) = load_all_from_store(store, &manifest)?;
    let chunks_total = all_chunks.len();
    let index = SearchIndex::new(all_chunks, &all_embeddings);

    Ok((
        index,
        ReindexStats {
            chunks_total,
            chunks_reembedded: new_chunks_count,
            files_unchanged,
            files_changed,
            files_deleted,
            duration_ms: start.elapsed().as_millis() as u64,
        },
    ))
}

/// Full index from scratch: embed everything, save to cache.
#[expect(clippy::too_many_arguments, reason = "pipeline state passed through")]
#[expect(
    clippy::cast_possible_truncation,
    reason = "duration in ms won't exceed u64"
)]
fn full_index_path(
    root: &Path,
    backends: &[&dyn EmbedBackend],
    tokenizer: &tokenizers::Tokenizer,
    cfg: &SearchConfig,
    profiler: &Profiler,
    model_repo: &str,
    cache_dir: &Path,
    store: &ObjectStore,
    start: Instant,
) -> crate::Result<(SearchIndex, ReindexStats)> {
    let (chunks, embeddings) = crate::embed::embed_all(root, backends, tokenizer, cfg, profiler)?;

    let hidden_dim = embeddings.first().map_or(384, Vec::len);

    // Group chunks and embeddings by file, save to store
    let mut manifest = Manifest::new(model_repo);
    let mut file_groups: std::collections::BTreeMap<String, (Vec<CodeChunk>, Vec<Vec<f32>>)> =
        std::collections::BTreeMap::new();

    for (chunk, emb) in chunks.iter().zip(embeddings.iter()) {
        file_groups
            .entry(chunk.file_path.clone())
            .or_default()
            .0
            .push(chunk.clone());
        file_groups
            .entry(chunk.file_path.clone())
            .or_default()
            .1
            .push(emb.clone());
    }

    for (file_path, (file_chunks, file_embeddings)) in &file_groups {
        // file_path from CodeChunk is already an absolute or cwd-relative path
        let file_path_buf = PathBuf::from(file_path);

        let content_hash = diff::hash_file(&file_path_buf).unwrap_or_else(|_| {
            // File might not exist (e.g., generated content) — use chunk content hash
            blake3::hash(file_chunks[0].content.as_bytes())
                .to_hex()
                .to_string()
        });

        let flat_emb: Vec<f32> = file_embeddings.iter().flatten().copied().collect();
        let fc = FileCache {
            chunks: file_chunks.clone(),
            embeddings: flat_emb,
            hidden_dim,
        };
        store.write(&content_hash, &fc.to_bytes())?;

        let relative = file_path_buf
            .strip_prefix(root)
            .unwrap_or(&file_path_buf)
            .to_string_lossy()
            .to_string();
        let mtime = diff::mtime_secs(&file_path_buf);
        let size = std::fs::metadata(&file_path_buf).map_or(0, |m| m.len());
        manifest.add_file(&relative, mtime, size, &content_hash, file_chunks.len());
    }

    manifest.recompute_hashes();
    manifest.save(&cache_dir.join("manifest.json"))?;

    let chunks_total = chunks.len();
    let files_changed = file_groups.len();
    let index = SearchIndex::new(chunks, &embeddings);

    Ok((
        index,
        ReindexStats {
            chunks_total,
            chunks_reembedded: chunks_total,
            files_unchanged: 0,
            files_changed,
            files_deleted: 0,
            duration_ms: start.elapsed().as_millis() as u64,
        },
    ))
}

/// Load all cached chunks and embeddings from the object store.
fn load_all_from_store(
    store: &ObjectStore,
    manifest: &Manifest,
) -> crate::Result<(Vec<CodeChunk>, Vec<Vec<f32>>)> {
    let mut all_chunks = Vec::new();
    let mut all_embeddings = Vec::new();

    for entry in manifest.files.values() {
        let bytes = store.read(&entry.content_hash)?;
        let fc = FileCache::from_bytes(&bytes)?;
        let dim = fc.hidden_dim;

        for (i, chunk) in fc.chunks.into_iter().enumerate() {
            let start = i * dim;
            let end = start + dim;
            if end <= fc.embeddings.len() {
                all_embeddings.push(fc.embeddings[start..end].to_vec());
                all_chunks.push(chunk);
            }
        }
    }

    Ok((all_chunks, all_embeddings))
}

/// Resolve the cache directory for a project.
///
/// Priority: override > `RIPVEC_CACHE` env > XDG cache dir.
/// The project hash is blake3 of the canonical absolute path.
#[must_use]
pub fn resolve_cache_dir(root: &Path, override_dir: Option<&Path>) -> PathBuf {
    let project_hash = {
        let canonical = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
        blake3::hash(canonical.to_string_lossy().as_bytes())
            .to_hex()
            .to_string()
    };

    if let Some(dir) = override_dir {
        return dir.join(&project_hash);
    }

    if let Ok(env_dir) = std::env::var("RIPVEC_CACHE") {
        return PathBuf::from(env_dir).join(&project_hash);
    }

    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join("ripvec")
        .join(&project_hash)
}
