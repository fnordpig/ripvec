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
use crate::hybrid::HybridIndex;
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
    repo_level: bool,
) -> crate::Result<(HybridIndex, ReindexStats)> {
    let start = Instant::now();
    tracing::info!(root = %root.display(), model = model_repo, "incremental_index starting");

    if backends.is_empty() {
        return Err(crate::Error::Other(anyhow::anyhow!(
            "no embedding backends provided"
        )));
    }

    // When repo_level is requested, ensure .ripvec/config.toml exists
    // so that resolve_cache_dir will find it and use the repo-local path.
    if repo_level {
        let ripvec_dir = root.join(".ripvec");
        let config_path = ripvec_dir.join("config.toml");
        if !config_path.exists() {
            let config = crate::cache::config::RepoConfig::new(
                model_repo,
                crate::cache::manifest::MANIFEST_VERSION.to_string(),
            );
            config.save(&ripvec_dir)?;
        }
        // Gitignore the manifest — it's rebuilt from objects on first use.
        // Objects are content-addressed and never cause merge conflicts.
        let gitignore_path = ripvec_dir.join(".gitignore");
        if !gitignore_path.exists() {
            let _ = std::fs::write(&gitignore_path, "cache/manifest.json\n");
        }
    }

    let cache_dir = resolve_cache_dir(root, model_repo, cache_dir_override);
    let portable = is_repo_local(&cache_dir);
    let manifest_path = cache_dir.join("manifest.json");
    let objects_dir = cache_dir.join("objects");
    let store = ObjectStore::new(&objects_dir);

    // Try loading existing manifest, or rebuild from objects if missing.
    let existing_manifest = Manifest::load(&manifest_path)
        .ok()
        .or_else(|| rebuild_manifest_from_objects(&cache_dir, root, model_repo));

    if let Some(manifest) = existing_manifest.filter(|m| m.is_compatible(model_repo)) {
        tracing::info!(
            files = manifest.files.len(),
            "manifest loaded, running incremental diff"
        );
        // Incremental path: diff → re-embed dirty → merge
        incremental_path(
            root, backends, tokenizer, cfg, profiler, model_repo, &cache_dir, &store, manifest,
            start, portable,
        )
    } else {
        // Cold path: full embed
        full_index_path(
            root, backends, tokenizer, cfg, profiler, model_repo, &cache_dir, &store, start,
            portable,
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
    portable: bool,
) -> crate::Result<(HybridIndex, ReindexStats)> {
    let diff_result = diff::compute_diff(root, &manifest)?;

    let files_changed = diff_result.dirty.len();
    let files_deleted = diff_result.deleted.len();
    let files_unchanged = diff_result.unchanged;

    tracing::info!(
        changed = files_changed,
        deleted = files_deleted,
        unchanged = files_unchanged,
        "diff complete"
    );

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
            .map(|chunk| {
                crate::tokenize::tokenize_query(&chunk.enriched_content, tokenizer, model_max).ok()
            })
            .collect();

        // Embed
        let embeddings =
            crate::embed::embed_distributed(&encodings, backends, cfg.batch_size, profiler)?;

        // Filter out failed tokenizations
        let (good_chunks, good_embeddings): (Vec<_>, Vec<_>) = chunks
            .into_iter()
            .zip(embeddings)
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
        let bytes = if portable {
            file_cache.to_portable_bytes()
        } else {
            file_cache.to_bytes()
        };
        store.write(&content_hash, &bytes)?;

        // Update manifest
        let mtime = diff::mtime_secs(dirty_path);
        let size = std::fs::metadata(dirty_path).map_or(0, |m| m.len());
        manifest.add_file(&relative, mtime, size, &content_hash, good_chunks.len());
        new_chunks_count += good_chunks.len();
    }

    // Heal stale mtimes (e.g., after git clone where all mtimes are wrong
    // but content hashes match). This ensures the fast-path mtime check
    // works on subsequent runs.
    heal_manifest_mtimes(root, &mut manifest);

    // Recompute Merkle hashes
    manifest.recompute_hashes();

    // GC unreferenced objects
    let referenced = manifest.referenced_hashes();
    store.gc(&referenced)?;

    // Save manifest
    manifest.save(&cache_dir.join("manifest.json"))?;

    // Rebuild HybridIndex (semantic + BM25) from all cached objects
    tracing::info!("loading cached objects from store");
    let (all_chunks, all_embeddings) = load_all_from_store(store, &manifest)?;
    let chunks_total = all_chunks.len();
    tracing::info!(
        chunks = chunks_total,
        "building HybridIndex (BM25 + PolarQuant)"
    );
    let hybrid = HybridIndex::new(all_chunks, &all_embeddings, None)?;
    tracing::info!("HybridIndex ready");

    Ok((
        hybrid,
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
    portable: bool,
) -> crate::Result<(HybridIndex, ReindexStats)> {
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
        let bytes = if portable {
            fc.to_portable_bytes()
        } else {
            fc.to_bytes()
        };
        store.write(&content_hash, &bytes)?;

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
    let hybrid = HybridIndex::new(chunks, &embeddings, None)?;

    Ok((
        hybrid,
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

/// Check if the resolved cache directory is inside a `.ripvec/` directory.
#[must_use]
pub fn is_repo_local(cache_dir: &Path) -> bool {
    cache_dir.components().any(|c| c.as_os_str() == ".ripvec")
}

/// Update manifest file mtimes to match the current filesystem.
///
/// After a git clone, all file mtimes are set to clone time, making the
/// fast-path mtime check miss on every file. This function updates the
/// manifest mtimes so subsequent diffs use the fast path.
pub fn heal_manifest_mtimes(root: &Path, manifest: &mut Manifest) {
    for (relative, entry) in &mut manifest.files {
        let file_path = root.join(relative);
        let mtime = diff::mtime_secs(&file_path);
        if mtime != entry.mtime_secs {
            entry.mtime_secs = mtime;
        }
    }
}

/// Check whether `pull.autoStash` needs to be configured for a repo-local cache.
///
/// Returns `Some(message)` with a human-readable prompt if the setting has not
/// been configured yet. Returns `None` if already configured (in git config or
/// `.ripvec/config.toml`) or if the cache is not repo-local.
#[must_use]
pub fn check_auto_stash(root: &Path) -> Option<String> {
    use std::process::Command;

    let ripvec_dir = root.join(".ripvec");
    let config = crate::cache::config::RepoConfig::load(&ripvec_dir).ok()?;
    if !config.cache.local {
        return None;
    }

    // Already decided via config.toml
    if config.cache.auto_stash.is_some() {
        return None;
    }

    // Already set in git config (by user or previous run)
    let git_check = Command::new("git")
        .args(["config", "--local", "pull.autoStash"])
        .current_dir(root)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;
    if git_check.status.success() {
        // Sync the existing git setting into config.toml so we don't check again
        let val = String::from_utf8_lossy(&git_check.stdout)
            .trim()
            .eq_ignore_ascii_case("true");
        let _ = apply_auto_stash(root, val);
        return None;
    }

    Some(
        "ripvec: Repo-local cache can dirty the worktree and block `git pull`.\n\
         Enable `pull.autoStash` for this repo? (git stashes dirty files before pull, pops after)"
            .to_string(),
    )
}

/// Apply the user's `auto_stash` choice: set git config and save to `config.toml`.
///
/// When `enable` is true, runs `git config --local pull.autoStash true`.
/// The choice is persisted to `.ripvec/config.toml` so the prompt is not repeated.
///
/// # Errors
///
/// Returns an error if `config.toml` cannot be read or written.
pub fn apply_auto_stash(root: &Path, enable: bool) -> crate::Result<()> {
    use std::process::Command;

    let ripvec_dir = root.join(".ripvec");
    let mut config = crate::cache::config::RepoConfig::load(&ripvec_dir)?;
    config.cache.auto_stash = Some(enable);
    config.save(&ripvec_dir)?;

    if enable {
        let _ = Command::new("git")
            .args(["config", "--local", "pull.autoStash", "true"])
            .current_dir(root)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
    }

    Ok(())
}

/// Load a `FileCache` from bytes, auto-detecting the format.
/// Checks for bitcode magic first (portable), then falls back to rkyv.
fn load_file_cache(bytes: &[u8]) -> crate::Result<FileCache> {
    if bytes.len() >= 2 && bytes[..2] == [0x42, 0x43] {
        FileCache::from_portable_bytes(bytes)
    } else {
        FileCache::from_bytes(bytes)
    }
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
        let fc = load_file_cache(&bytes)?;
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

/// Load a pre-built index from the disk cache without re-embedding.
///
/// This is the lightweight read path for processes that don't own the index
/// (e.g., the LSP process reading caches built by the MCP process).
/// Returns `None` if no compatible cache exists for this root.
///
/// Uses an advisory file lock on `manifest.lock` to avoid reading
/// a half-written cache.
#[must_use]
pub fn load_cached_index(root: &Path, model_repo: &str) -> Option<HybridIndex> {
    let cache_dir = resolve_cache_dir(root, model_repo, None);
    let manifest_path = cache_dir.join("manifest.json");
    let objects_dir = cache_dir.join("objects");
    let lock_path = cache_dir.join("manifest.lock");

    // Ensure cache dir exists (it might not if no index has been built)
    if !manifest_path.exists() {
        return None;
    }

    // Acquire a shared (read) lock — blocks if a writer holds the exclusive lock
    let lock_file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .read(true)
        .open(&lock_path)
        .ok()?;
    let lock = fd_lock::RwLock::new(lock_file);
    let _guard = lock.read().ok()?;

    let manifest = Manifest::load(&manifest_path)
        .ok()
        .or_else(|| rebuild_manifest_from_objects(&cache_dir, root, model_repo))?;
    if !manifest.is_compatible(model_repo) {
        return None;
    }

    let store = ObjectStore::new(&objects_dir);
    let (chunks, embeddings) = load_all_from_store(&store, &manifest).ok()?;
    HybridIndex::new(chunks, &embeddings, None).ok()
}

/// Resolve the cache directory for a project + model combination.
///
/// Resolution priority:
/// 1. `override_dir` parameter (highest)
/// 2. `.ripvec/config.toml` in directory tree (repo-local)
/// 3. `RIPVEC_CACHE` environment variable
/// 4. XDG cache dir (`~/.cache/ripvec/`)
///
/// For repo-local, the cache lives at `.ripvec/cache/` directly (no project hash
/// or version subdirectory — the config.toml pins the model and version).
///
/// For user-level cache, layout is `<base>/<project_hash>/v<VERSION>-<model_slug>/`.
#[must_use]
pub fn resolve_cache_dir(root: &Path, model_repo: &str, override_dir: Option<&Path>) -> PathBuf {
    // Priority 1: explicit override
    if let Some(dir) = override_dir {
        let project_hash = hash_project_root(root);
        let version_dir = format_version_dir(model_repo);
        return dir.join(&project_hash).join(version_dir);
    }

    // Priority 2: repo-local .ripvec/config.toml (with model validation)
    if let Some(ripvec_dir) = crate::cache::config::find_repo_config(root)
        && let Ok(config) = crate::cache::config::RepoConfig::load(&ripvec_dir)
    {
        if config.cache.model == model_repo {
            return ripvec_dir.join("cache");
        }
        eprintln!(
            "[ripvec] repo-local index model mismatch: config has '{}', runtime wants '{}' — falling back to user cache",
            config.cache.model, model_repo
        );
    }

    // Priority 3+4: env var or XDG
    let project_hash = hash_project_root(root);
    let version_dir = format_version_dir(model_repo);

    let base = if let Ok(env_dir) = std::env::var("RIPVEC_CACHE") {
        PathBuf::from(env_dir).join(&project_hash)
    } else {
        dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("/tmp"))
            .join("ripvec")
            .join(&project_hash)
    };

    base.join(version_dir)
}

/// Blake3 hash of the canonical project root path.
fn hash_project_root(root: &Path) -> String {
    let canonical = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    blake3::hash(canonical.to_string_lossy().as_bytes())
        .to_hex()
        .to_string()
}

/// Format the version subdirectory name from model repo.
fn format_version_dir(model_repo: &str) -> String {
    let model_slug = model_repo
        .rsplit('/')
        .next()
        .unwrap_or(model_repo)
        .to_lowercase();
    format!("v{}-{model_slug}", crate::cache::manifest::MANIFEST_VERSION)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn heal_stale_mtimes() {
        use crate::cache::diff;
        use crate::cache::manifest::Manifest;
        use std::io::Write;

        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("test.rs");
        let content = "fn main() {}";
        {
            let mut f = std::fs::File::create(&file_path).unwrap();
            f.write_all(content.as_bytes()).unwrap();
        }

        // Create manifest with correct content hash but wrong mtime
        let content_hash = blake3::hash(content.as_bytes()).to_hex().to_string();
        let mut manifest = Manifest::new("test-model");
        manifest.add_file(
            "test.rs",
            9_999_999, // deliberately wrong mtime
            content.len() as u64,
            &content_hash,
            1,
        );

        // After heal, the manifest mtime should match the filesystem
        heal_manifest_mtimes(dir.path(), &mut manifest);
        let actual_mtime = diff::mtime_secs(&file_path);
        assert_eq!(manifest.files["test.rs"].mtime_secs, actual_mtime);
    }

    #[test]
    fn resolve_uses_repo_local_when_present() {
        let dir = TempDir::new().unwrap();
        let cfg = crate::cache::config::RepoConfig::new("nomic-ai/modernbert-embed-base", "3");
        cfg.save(&dir.path().join(".ripvec")).unwrap();

        let result = resolve_cache_dir(dir.path(), "nomic-ai/modernbert-embed-base", None);
        assert!(
            result.starts_with(dir.path().join(".ripvec").join("cache")),
            "expected repo-local cache dir, got: {result:?}"
        );
    }

    #[test]
    fn resolve_falls_back_to_user_cache_when_no_config() {
        let dir = TempDir::new().unwrap();
        let result = resolve_cache_dir(dir.path(), "nomic-ai/modernbert-embed-base", None);
        assert!(
            !result.to_string_lossy().contains(".ripvec"),
            "should not use repo-local without config, got: {result:?}"
        );
    }

    #[test]
    fn resolve_override_takes_priority_over_repo_local() {
        let dir = TempDir::new().unwrap();
        let override_dir = TempDir::new().unwrap();

        let cfg = crate::cache::config::RepoConfig::new("nomic-ai/modernbert-embed-base", "3");
        cfg.save(&dir.path().join(".ripvec")).unwrap();

        let result = resolve_cache_dir(
            dir.path(),
            "nomic-ai/modernbert-embed-base",
            Some(override_dir.path()),
        );
        assert!(
            !result.starts_with(dir.path().join(".ripvec")),
            "override should win over repo-local, got: {result:?}"
        );
    }
}

/// Rebuild a manifest by scanning the object store and deserializing each object.
///
/// Used when `manifest.json` is gitignored and only the objects directory is
/// committed. Scans every object, extracts the file path from the chunks,
/// stats the source file for mtime/size, and constructs a valid manifest.
///
/// Returns `None` if the objects directory doesn't exist or is empty.
#[must_use]
pub fn rebuild_manifest_from_objects(
    cache_dir: &std::path::Path,
    root: &std::path::Path,
    model_repo: &str,
) -> Option<super::manifest::Manifest> {
    use super::file_cache::FileCache;
    use super::manifest::{FileEntry, MANIFEST_VERSION, Manifest};
    use super::store::ObjectStore;
    use std::collections::BTreeMap;

    let store = ObjectStore::new(&cache_dir.join("objects"));
    let hashes = store.list_hashes();
    if hashes.is_empty() {
        return None;
    }

    tracing::info!(
        objects = hashes.len(),
        "rebuilding manifest from object store"
    );

    let mut files = BTreeMap::new();

    for hash in &hashes {
        let Ok(bytes) = store.read(hash) else {
            continue;
        };
        let Ok(fc) =
            FileCache::from_portable_bytes(&bytes).or_else(|_| FileCache::from_bytes(&bytes))
        else {
            continue;
        };
        let Some(first_chunk) = fc.chunks.first() else {
            continue;
        };

        // The chunk's file_path may be absolute or relative.
        // Try to make it relative to root for the manifest key.
        let chunk_path = std::path::Path::new(&first_chunk.file_path);
        let rel_path = chunk_path
            .strip_prefix(root)
            .unwrap_or(chunk_path)
            .to_string_lossy()
            .to_string();

        // Stat the actual file for mtime/size.
        let abs_path = root.join(&rel_path);
        let (mtime_secs, size) = if let Ok(meta) = std::fs::metadata(&abs_path) {
            let mtime = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map_or(0, |d| d.as_secs());
            (mtime, meta.len())
        } else {
            (0, 0) // file may not exist on this machine yet
        };

        files.insert(
            rel_path,
            FileEntry {
                mtime_secs,
                size,
                content_hash: hash.clone(),
                chunk_count: fc.chunks.len(),
            },
        );
    }

    if files.is_empty() {
        return None;
    }

    let manifest = Manifest {
        version: MANIFEST_VERSION,
        model_repo: model_repo.to_string(),
        root_hash: String::new(), // will be recomputed on next incremental_index
        directories: BTreeMap::new(), // will be recomputed on next incremental_index
        files,
    };

    tracing::info!(
        files = manifest.files.len(),
        "manifest rebuilt from objects"
    );

    // Write the rebuilt manifest to disk so subsequent runs use it.
    let manifest_path = cache_dir.join("manifest.json");
    if let Ok(json) = serde_json::to_string_pretty(&manifest) {
        let _ = std::fs::write(&manifest_path, json);
    }

    Some(manifest)
}
