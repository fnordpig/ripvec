# Repo-Local Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Opt-in repo-local index storage in `.ripvec/cache/` with portable bitcode serialization, so pre-built indices can be committed to git and shared across teams/CI.

**Architecture:** Replace rkyv with bitcode for FileCache serialization (portable across architectures). Add `.ripvec/config.toml` discovery to the cache resolution chain. `--repo-level` flag on CLI and `repo_level` param on MCP `reindex` create the local store; subsequent runs auto-detect it.

**Tech Stack:** bitcode (serialization), zstd (compression), blake3 (hashing), toml (config), serde (derive)

---

### Task 1: Add bitcode dependency, derive on CodeChunk and FileCache

**Files:**
- Modify: `Cargo.toml` (workspace) — add `bitcode` to workspace deps
- Modify: `crates/ripvec-core/Cargo.toml` — add bitcode dep
- Modify: `crates/ripvec-core/src/chunk.rs:43` — add bitcode derives to CodeChunk
- Modify: `crates/ripvec-core/src/cache/file_cache.rs:12` — add bitcode derives to FileCache

- [ ] **Step 1: Add bitcode to workspace dependencies**

In `Cargo.toml` (workspace root), add to `[workspace.dependencies]`:

```toml
bitcode = "0.6"
```

- [ ] **Step 2: Add bitcode to ripvec-core dependencies**

In `crates/ripvec-core/Cargo.toml`, add under `[dependencies]`:

```toml
bitcode.workspace = true
```

- [ ] **Step 3: Add bitcode derives to CodeChunk**

In `crates/ripvec-core/src/chunk.rs`, change the derive on `CodeChunk` from:

```rust
#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct CodeChunk {
```

to:

```rust
#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize, bitcode::Encode, bitcode::Decode)]
pub struct CodeChunk {
```

- [ ] **Step 4: Add bitcode derives to FileCache**

In `crates/ripvec-core/src/cache/file_cache.rs`, change the derive on `FileCache` from:

```rust
#[derive(Debug, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct FileCache {
```

to:

```rust
#[derive(Debug, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize, bitcode::Encode, bitcode::Decode)]
pub struct FileCache {
```

- [ ] **Step 5: Verify it compiles**

Run: `cargo check -p ripvec-core`
Expected: compiles with no errors (both rkyv and bitcode derives coexist)

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml crates/ripvec-core/Cargo.toml crates/ripvec-core/src/chunk.rs crates/ripvec-core/src/cache/file_cache.rs
git commit -m "feat: add bitcode derives to CodeChunk and FileCache for portable serialization"
```

---

### Task 2: Add portable bitcode serialization methods to FileCache

**Files:**
- Modify: `crates/ripvec-core/src/cache/file_cache.rs` — add `to_portable_bytes`/`from_portable_bytes`, update tests

- [ ] **Step 1: Write the failing test for portable round-trip**

Add to the `tests` module in `crates/ripvec-core/src/cache/file_cache.rs`:

```rust
#[test]
fn portable_round_trip() {
    let fc = FileCache {
        chunks: vec![CodeChunk {
            file_path: "test.rs".into(),
            name: "foo".into(),
            kind: "function".into(),
            start_line: 1,
            end_line: 10,
            enriched_content: "fn foo() {}".into(),
            content: "fn foo() {}".into(),
        }],
        embeddings: vec![1.0, 2.0, 3.0, 4.0],
        hidden_dim: 4,
    };
    let bytes = fc.to_portable_bytes();
    let loaded = FileCache::from_portable_bytes(&bytes).unwrap();
    assert_eq!(loaded.chunks.len(), 1);
    assert_eq!(loaded.chunks[0].name, "foo");
    assert_eq!(loaded.embeddings.len(), 4);
    assert_eq!(loaded.hidden_dim, 4);
}

#[test]
fn portable_empty_cache() {
    let fc = FileCache {
        chunks: vec![],
        embeddings: vec![],
        hidden_dim: 384,
    };
    let bytes = fc.to_portable_bytes();
    let loaded = FileCache::from_portable_bytes(&bytes).unwrap();
    assert_eq!(loaded.chunks.len(), 0);
    assert_eq!(loaded.embeddings.len(), 0);
    assert_eq!(loaded.hidden_dim, 384);
}

#[test]
fn portable_invalid_bytes_returns_error() {
    let result = FileCache::from_portable_bytes(b"garbage data");
    assert!(result.is_err());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p ripvec-core file_cache::tests::portable_round_trip`
Expected: FAIL — `to_portable_bytes` not found

- [ ] **Step 3: Implement portable serialization methods**

Add to the `impl FileCache` block in `crates/ripvec-core/src/cache/file_cache.rs`:

```rust
/// Magic bytes to identify bitcode-encoded (portable) cache objects.
/// First byte 0x42 ('B') followed by 0x43 ('C') — "BC" for bitcode.
const BITCODE_MAGIC: [u8; 2] = [0x42, 0x43];

/// Serialize to portable zstd-compressed bitcode bytes.
///
/// Unlike `to_bytes` (rkyv, architecture-dependent), this format is safe
/// to share across different CPU architectures (e.g., x86_64 CI → aarch64 Mac).
#[must_use]
pub fn to_portable_bytes(&self) -> Vec<u8> {
    let raw = bitcode::encode(self);
    let compressed = zstd::encode_all(raw.as_slice(), 1)
        .expect("zstd compression should never fail on valid data");
    let mut out = Vec::with_capacity(BITCODE_MAGIC.len() + compressed.len());
    out.extend_from_slice(&BITCODE_MAGIC);
    out.extend_from_slice(&compressed);
    out
}

/// Deserialize from portable zstd-compressed bitcode bytes.
///
/// Expects the `BITCODE_MAGIC` prefix. Returns an error if the bytes
/// are not a valid portable archive.
pub fn from_portable_bytes(bytes: &[u8]) -> crate::Result<Self> {
    if bytes.len() < 2 || bytes[..2] != BITCODE_MAGIC {
        return Err(crate::Error::Other(anyhow::anyhow!(
            "not a portable bitcode cache object (missing magic)"
        )));
    }
    let compressed = &bytes[2..];
    let raw = zstd::decode_all(compressed).map_err(|e| {
        crate::Error::Other(anyhow::anyhow!("zstd decompression failed: {e}"))
    })?;
    bitcode::decode(&raw).map_err(|e| {
        crate::Error::Other(anyhow::anyhow!("bitcode deserialization failed: {e}"))
    })
}
```

Note: Move the `BITCODE_MAGIC` constant inside `impl FileCache` or as a module-level constant next to `ZSTD_MAGIC`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p ripvec-core file_cache::tests::portable`
Expected: all 3 portable tests PASS

- [ ] **Step 5: Run all file_cache tests to confirm no regressions**

Run: `cargo test -p ripvec-core file_cache::tests`
Expected: all tests PASS (both rkyv and bitcode paths)

- [ ] **Step 6: Commit**

```bash
git add crates/ripvec-core/src/cache/file_cache.rs
git commit -m "feat: add portable bitcode serialization to FileCache"
```

---

### Task 3: Add config.toml support and repo-local cache resolution

**Files:**
- Create: `crates/ripvec-core/src/cache/config.rs` — config struct + discovery
- Modify: `crates/ripvec-core/src/cache/mod.rs` — add `pub mod config;`
- Modify: `crates/ripvec-core/Cargo.toml` — add `toml` dep

- [ ] **Step 1: Add toml dependency**

In `Cargo.toml` (workspace root), add to `[workspace.dependencies]`:

```toml
toml = "0.8"
```

In `crates/ripvec-core/Cargo.toml`, add under `[dependencies]`:

```toml
toml.workspace = true
```

- [ ] **Step 2: Write the failing tests**

Create `crates/ripvec-core/src/cache/config.rs`:

```rust
//! Repo-local cache configuration.
//!
//! When a `.ripvec/config.toml` exists in the project directory tree,
//! it signals that this project uses repo-local cache storage.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Repo-local ripvec configuration.
#[derive(Debug, Serialize, Deserialize)]
pub struct RepoConfig {
    /// Cache settings.
    pub cache: CacheConfig,
}

/// Cache section of the repo config.
#[derive(Debug, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Whether repo-local caching is enabled.
    pub local: bool,
    /// The embedding model used to build this index.
    pub model: String,
    /// Cache format version (for compatibility checks).
    pub version: String,
}

impl RepoConfig {
    /// Create a new config for the given model and manifest version.
    #[must_use]
    pub fn new(model: &str, version: u32) -> Self {
        Self {
            cache: CacheConfig {
                local: true,
                model: model.to_string(),
                version: format!("{version}"),
            },
        }
    }

    /// Serialize to TOML string.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    pub fn to_toml(&self) -> crate::Result<String> {
        toml::to_string_pretty(self)
            .map_err(|e| crate::Error::Other(anyhow::anyhow!("toml serialization: {e}")))
    }

    /// Deserialize from TOML string.
    ///
    /// # Errors
    ///
    /// Returns an error if the TOML is invalid.
    pub fn from_toml(s: &str) -> crate::Result<Self> {
        toml::from_str(s)
            .map_err(|e| crate::Error::Other(anyhow::anyhow!("toml deserialization: {e}")))
    }

    /// Save config to a file, creating parent directories as needed.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written.
    pub fn save(&self, path: &Path) -> crate::Result<()> {
        let toml_str = self.to_toml()?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| crate::Error::Io {
                path: parent.display().to_string(),
                source: e,
            })?;
        }
        std::fs::write(path, toml_str).map_err(|e| crate::Error::Io {
            path: path.display().to_string(),
            source: e,
        })
    }

    /// Load config from a file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or parsed.
    pub fn load(path: &Path) -> crate::Result<Self> {
        let s = std::fs::read_to_string(path).map_err(|e| crate::Error::Io {
            path: path.display().to_string(),
            source: e,
        })?;
        Self::from_toml(&s)
    }
}

/// Walk up from `start` looking for `.ripvec/config.toml`.
///
/// Returns the path to `.ripvec/` if found and `cache.local` is true,
/// or `None` if no repo-local config exists in the directory tree.
#[must_use]
pub fn find_repo_config(start: &Path) -> Option<PathBuf> {
    let mut current = start.canonicalize().unwrap_or_else(|_| start.to_path_buf());
    loop {
        let config_path = current.join(".ripvec").join("config.toml");
        if config_path.is_file() {
            if let Ok(cfg) = RepoConfig::load(&config_path) {
                if cfg.cache.local {
                    return Some(current.join(".ripvec"));
                }
            }
        }
        if !current.pop() {
            return None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn round_trip_toml() {
        let cfg = RepoConfig::new("nomic-ai/modernbert-embed-base", 3);
        let toml_str = cfg.to_toml().unwrap();
        let loaded = RepoConfig::from_toml(&toml_str).unwrap();
        assert!(loaded.cache.local);
        assert_eq!(loaded.cache.model, "nomic-ai/modernbert-embed-base");
        assert_eq!(loaded.cache.version, "3");
    }

    #[test]
    fn save_and_load() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join(".ripvec").join("config.toml");
        let cfg = RepoConfig::new("test-model", 3);
        cfg.save(&path).unwrap();
        let loaded = RepoConfig::load(&path).unwrap();
        assert!(loaded.cache.local);
        assert_eq!(loaded.cache.model, "test-model");
    }

    #[test]
    fn find_repo_config_in_current_dir() {
        let dir = TempDir::new().unwrap();
        let cfg = RepoConfig::new("test-model", 3);
        cfg.save(&dir.path().join(".ripvec").join("config.toml"))
            .unwrap();

        let result = find_repo_config(dir.path());
        assert!(result.is_some());
        assert_eq!(result.unwrap(), dir.path().join(".ripvec"));
    }

    #[test]
    fn find_repo_config_in_parent_dir() {
        let dir = TempDir::new().unwrap();
        let cfg = RepoConfig::new("test-model", 3);
        cfg.save(&dir.path().join(".ripvec").join("config.toml"))
            .unwrap();

        // Create a subdirectory and search from there
        let sub = dir.path().join("src").join("deep");
        std::fs::create_dir_all(&sub).unwrap();

        let result = find_repo_config(&sub);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), dir.path().join(".ripvec"));
    }

    #[test]
    fn find_repo_config_not_found() {
        let dir = TempDir::new().unwrap();
        let result = find_repo_config(dir.path());
        assert!(result.is_none());
    }

    #[test]
    fn find_repo_config_ignores_disabled() {
        let dir = TempDir::new().unwrap();
        let toml_str = "[cache]\nlocal = false\nmodel = \"test\"\nversion = \"3\"\n";
        let config_path = dir.path().join(".ripvec").join("config.toml");
        std::fs::create_dir_all(config_path.parent().unwrap()).unwrap();
        std::fs::write(&config_path, toml_str).unwrap();

        let result = find_repo_config(dir.path());
        assert!(result.is_none());
    }
}
```

- [ ] **Step 3: Register the module**

In `crates/ripvec-core/src/cache/mod.rs`, add:

```rust
pub mod config;
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p ripvec-core cache::config::tests`
Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml crates/ripvec-core/Cargo.toml crates/ripvec-core/src/cache/config.rs crates/ripvec-core/src/cache/mod.rs
git commit -m "feat: add .ripvec/config.toml support and repo-local discovery"
```

---

### Task 4: Update resolve_cache_dir to check for repo-local config

**Files:**
- Modify: `crates/ripvec-core/src/cache/reindex.rs:331-370` — update `resolve_cache_dir`

- [ ] **Step 1: Write failing test for repo-local resolution**

Add to the `tests` module in a new test file or in `reindex.rs` (if it has tests). Since `reindex.rs` doesn't have a test module, add one at the bottom:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn resolve_uses_repo_local_when_present() {
        let dir = TempDir::new().unwrap();
        let cfg = crate::cache::config::RepoConfig::new("nomic-ai/modernbert-embed-base", 3);
        cfg.save(&dir.path().join(".ripvec").join("config.toml"))
            .unwrap();

        let result = resolve_cache_dir(dir.path(), "nomic-ai/modernbert-embed-base", None);
        // Should point to .ripvec/cache/ not ~/.cache/ripvec/...
        assert!(result.starts_with(dir.path().join(".ripvec").join("cache")));
    }

    #[test]
    fn resolve_falls_back_to_user_cache_when_no_config() {
        let dir = TempDir::new().unwrap();
        let result = resolve_cache_dir(dir.path(), "nomic-ai/modernbert-embed-base", None);
        // Should NOT contain .ripvec
        assert!(!result.to_string_lossy().contains(".ripvec"));
    }

    #[test]
    fn resolve_override_takes_priority_over_repo_local() {
        let dir = TempDir::new().unwrap();
        let override_dir = TempDir::new().unwrap();

        let cfg = crate::cache::config::RepoConfig::new("nomic-ai/modernbert-embed-base", 3);
        cfg.save(&dir.path().join(".ripvec").join("config.toml"))
            .unwrap();

        let result = resolve_cache_dir(
            dir.path(),
            "nomic-ai/modernbert-embed-base",
            Some(override_dir.path()),
        );
        // Override should win over repo-local
        assert!(!result.starts_with(dir.path().join(".ripvec")));
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p ripvec-core cache::reindex::tests::resolve_uses_repo_local`
Expected: FAIL — current `resolve_cache_dir` doesn't check for `.ripvec/`

- [ ] **Step 3: Update resolve_cache_dir**

Replace the `resolve_cache_dir` function in `crates/ripvec-core/src/cache/reindex.rs`:

```rust
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

    // Priority 2: repo-local .ripvec/config.toml
    if let Some(ripvec_dir) = crate::cache::config::find_repo_config(root) {
        return ripvec_dir.join("cache");
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
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p ripvec-core cache::reindex::tests`
Expected: all 3 tests PASS

- [ ] **Step 5: Run full workspace check**

Run: `cargo check --workspace`
Expected: compiles (callers of `resolve_cache_dir` unchanged — same signature)

- [ ] **Step 6: Commit**

```bash
git add crates/ripvec-core/src/cache/reindex.rs
git commit -m "feat: resolve_cache_dir checks for repo-local .ripvec/config.toml"
```

---

### Task 5: Add repo-local write path to reindex (portable serialization)

**Files:**
- Modify: `crates/ripvec-core/src/cache/reindex.rs` — add `repo_level` parameter, use portable bytes for repo-local, create config on first use

- [ ] **Step 1: Update incremental_index signature**

Add a `repo_level: bool` parameter to `incremental_index` in `crates/ripvec-core/src/cache/reindex.rs`:

```rust
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
```

When `repo_level` is true and no `.ripvec/config.toml` exists yet, create it:

```rust
    // If repo_level requested, ensure .ripvec/config.toml exists
    if repo_level {
        let ripvec_dir = root.join(".ripvec");
        let config_path = ripvec_dir.join("config.toml");
        if !config_path.exists() {
            let config = crate::cache::config::RepoConfig::new(
                model_repo,
                crate::cache::manifest::MANIFEST_VERSION,
            );
            config.save(&config_path)?;
        }
    }

    let cache_dir = resolve_cache_dir(root, model_repo, cache_dir_override);
```

- [ ] **Step 2: Use portable serialization for repo-local stores**

Add a helper to determine if the resolved cache is repo-local:

```rust
/// Check if the resolved cache directory is inside a .ripvec/ directory.
fn is_repo_local(cache_dir: &Path) -> bool {
    cache_dir.components().any(|c| c.as_os_str() == ".ripvec")
}
```

In `incremental_path` and `full_index_path`, change the store write calls to use portable bytes when repo-local. Pass a `portable: bool` parameter through:

In the `store.write` call for saving file caches, replace:
```rust
store.write(&content_hash, &file_cache.to_bytes())?;
```
with:
```rust
let bytes = if portable {
    file_cache.to_portable_bytes()
} else {
    file_cache.to_bytes()
};
store.write(&content_hash, &bytes)?;
```

Similarly in `load_all_from_store`, handle both formats by trying portable first if it has the magic, falling back to rkyv:

```rust
fn load_file_cache(bytes: &[u8]) -> crate::Result<FileCache> {
    // Check for bitcode magic (portable format)
    if bytes.len() >= 2 && bytes[..2] == [0x42, 0x43] {
        FileCache::from_portable_bytes(bytes)
    } else {
        FileCache::from_bytes(bytes)
    }
}
```

Replace the `FileCache::from_bytes(&bytes)?` call in `load_all_from_store` with `load_file_cache(&bytes)?`.

- [ ] **Step 3: Fix all callers of incremental_index**

There are 4 call sites that need the new `repo_level: bool` parameter:

1. `crates/ripvec/src/main.rs` — `run_interactive` (~line 252): add `false` (CLI will handle this in Task 7)
2. `crates/ripvec/src/main.rs` — `run_oneshot` (~line 427): add `false`
3. `crates/ripvec-mcp/src/server.rs` — `run_background_index` (~line 318): add `false`
4. `crates/ripvec-mcp/src/tools.rs` — `ensure_root` (~line 237): add `false`

All initially pass `false`. The CLI adds `--repo-level` in Task 7; MCP adds `repo_level` param in Task 8.

- [ ] **Step 4: Run full workspace check and tests**

Run: `cargo check --workspace && cargo test --workspace`
Expected: compiles and all tests pass

- [ ] **Step 5: Commit**

```bash
git add crates/ripvec-core/src/cache/reindex.rs crates/ripvec-core/src/cache/file_cache.rs crates/ripvec/src/main.rs crates/ripvec-mcp/src/server.rs crates/ripvec-mcp/src/tools.rs
git commit -m "feat: repo-local write path with portable bitcode serialization"
```

---

### Task 6: Bump manifest version to v3

**Files:**
- Modify: `crates/ripvec-core/src/cache/manifest.rs:49` — bump `MANIFEST_VERSION` from 2 to 3

- [ ] **Step 1: Bump the version constant**

In `crates/ripvec-core/src/cache/manifest.rs`, change:

```rust
pub const MANIFEST_VERSION: u32 = 2;
```

to:

```rust
pub const MANIFEST_VERSION: u32 = 3;
```

- [ ] **Step 2: Verify existing caches will be invalidated**

Run: `cargo test -p ripvec-core cache::manifest::tests::is_compatible`
Expected: PASS — `is_compatible` checks version, so v2 manifests will be rejected and rebuilt

- [ ] **Step 3: Run full test suite**

Run: `cargo test --workspace`
Expected: all tests pass

- [ ] **Step 4: Commit**

```bash
git add crates/ripvec-core/src/cache/manifest.rs
git commit -m "feat: bump MANIFEST_VERSION to 3 for bitcode migration"
```

---

### Task 7: Add --repo-level CLI flag

**Files:**
- Modify: `crates/ripvec/src/cli.rs` — add `--repo-level` flag
- Modify: `crates/ripvec/src/main.rs` — pass `repo_level` to `incremental_index`

- [ ] **Step 1: Add the flag to Args**

In `crates/ripvec/src/cli.rs`, add after the `index` field (~line 127):

```rust
    /// Store the index in `.ripvec/cache/` at the project root (repo-local).
    ///
    /// Creates `.ripvec/config.toml` on first use. The index can then be
    /// committed to git so teammates get instant search without re-embedding.
    /// Requires `--index`.
    #[arg(long, requires = "index")]
    pub repo_level: bool,
```

- [ ] **Step 2: Pass repo_level through to incremental_index**

In `crates/ripvec/src/main.rs`, update both `run_interactive` and `run_oneshot` to pass `args.repo_level` instead of `false` to `incremental_index`.

In `run_interactive` (~line 252):
```rust
        let (index, _stats) = ripvec_core::cache::reindex::incremental_index(
            std::path::Path::new(&args.path),
            &backend_refs,
            &tokenizer,
            search_cfg,
            &live_profiler,
            &model_repo,
            args.cache_dir.as_deref().map(std::path::Path::new),
            args.repo_level,
        )
```

In `run_oneshot` (~line 427):
```rust
        let (index, stats) = ripvec_core::cache::reindex::incremental_index(
            std::path::Path::new(&args.path),
            &backend_refs,
            tokenizer,
            search_cfg,
            profiler,
            &model_repo,
            args.cache_dir.as_deref().map(std::path::Path::new),
            args.repo_level,
        )
```

- [ ] **Step 3: Verify it compiles**

Run: `cargo check -p ripvec`
Expected: compiles

- [ ] **Step 4: Test the flag is recognized**

Run: `cargo run -p ripvec -- --help | grep repo-level`
Expected: shows `--repo-level` with description

- [ ] **Step 5: Commit**

```bash
git add crates/ripvec/src/cli.rs crates/ripvec/src/main.rs
git commit -m "feat: add --repo-level CLI flag for repo-local index storage"
```

---

### Task 8: Add repo_level parameter to MCP reindex tool

**Files:**
- Modify: `crates/ripvec-mcp/src/tools.rs` — add `repo_level` to `ReindexParams`, pass through
- Modify: `crates/ripvec-mcp/src/server.rs` — pass `repo_level` in `run_background_index`

- [ ] **Step 1: Update ReindexParams**

In `crates/ripvec-mcp/src/tools.rs`, update `ReindexParams`:

```rust
/// Parameters for the `reindex` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ReindexParams {
    /// Project root directory to reindex. Uses the server's default project root if omitted.
    pub root: Option<String>,
    /// Store the index in `.ripvec/cache/` at the project root (repo-local).
    /// Creates `.ripvec/config.toml` on first use. Commit the `.ripvec/` directory
    /// to git so teammates get instant search without re-embedding.
    #[serde(default)]
    pub repo_level: bool,
}
```

- [ ] **Step 2: Pass repo_level through in the reindex tool handler**

In the `reindex` tool handler in `crates/ripvec-mcp/src/tools.rs`, find where `incremental_index` is called for custom roots and pass `params.repo_level`. For the default root path, pass `params.repo_level` to `run_background_index`.

Update `run_background_index` signature in `crates/ripvec-mcp/src/server.rs` to accept `repo_level: bool`:

```rust
pub async fn run_background_index(server: &RipvecServer, repo_level: bool) {
```

And pass it to `incremental_index`:

```rust
        let (index, stats) = ripvec_core::cache::reindex::incremental_index(
            &root,
            &backend_refs,
            &tokenizer,
            &cfg,
            &profiler,
            model_repo,
            None,
            repo_level,
        )?;
```

- [ ] **Step 3: Fix all callers of run_background_index**

There are 3 call sites:
1. `server.rs` — startup in `main.rs` (~line): pass `false` (startup always uses existing config)
2. `tools.rs` — `reindex` tool handler: pass `params.repo_level`
3. `server.rs` — `run_file_watcher`: pass `false` (watcher preserves existing location)

Also update `ensure_root` in `tools.rs` to pass `false` (on-demand roots don't create repo-local configs).

- [ ] **Step 4: Verify it compiles**

Run: `cargo check --workspace`
Expected: compiles

- [ ] **Step 5: Commit**

```bash
git add crates/ripvec-mcp/src/tools.rs crates/ripvec-mcp/src/server.rs crates/ripvec-mcp/src/main.rs
git commit -m "feat: add repo_level parameter to MCP reindex tool"
```

---

### Task 9: Enhance index_status to report cache location

**Files:**
- Modify: `crates/ripvec-mcp/src/tools.rs` — add `cache_location` field to index_status response

- [ ] **Step 1: Add cache_location to the index_status response**

In the `index_status` handler in `crates/ripvec-mcp/src/tools.rs`, add the resolved cache location to both the custom-root and default-root responses:

For the default root case, resolve and report the cache location:

```rust
            let cache_dir = ripvec_core::cache::reindex::resolve_cache_dir(
                &self.project_root,
                "nomic-ai/modernbert-embed-base",
                None,
            );
            let is_repo_local = cache_dir.components().any(|c| c.as_os_str() == ".ripvec");

            let response = serde_json::json!({
                "ready": ready,
                "indexing": is_indexing,
                "chunks": chunk_count,
                "files": files_count,
                "extensions": ext_counts,
                "project_root": self.project_root.display().to_string(),
                "cache_location": cache_dir.display().to_string(),
                "repo_local": is_repo_local,
            });
```

Apply the same pattern to the custom-root branch.

- [ ] **Step 2: Verify it compiles**

Run: `cargo check -p ripvec-mcp`
Expected: compiles

- [ ] **Step 3: Commit**

```bash
git add crates/ripvec-mcp/src/tools.rs
git commit -m "feat: index_status reports cache location and repo-local status"
```

---

### Task 10: Update manifest mtimes on first clone (mtime self-heal)

**Files:**
- Modify: `crates/ripvec-core/src/cache/reindex.rs` — update mtimes after diff when content matches

- [ ] **Step 1: Write the failing test**

Add to `crates/ripvec-core/src/cache/reindex.rs` tests:

```rust
#[test]
fn diff_heals_stale_mtimes() {
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
        9999999, // deliberately wrong mtime
        content.len() as u64,
        &content_hash,
        1,
    );

    // Diff should find 0 dirty files (content matches)
    let diff_result = diff::compute_diff(dir.path(), &manifest).unwrap();
    assert!(diff_result.dirty.is_empty());
    assert_eq!(diff_result.unchanged, 1);

    // After heal_mtimes, the manifest mtime should match the filesystem
    heal_manifest_mtimes(dir.path(), &mut manifest);
    let actual_mtime = diff::mtime_secs(&file_path);
    assert_eq!(manifest.files["test.rs"].mtime_secs, actual_mtime);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p ripvec-core cache::reindex::tests::diff_heals_stale_mtimes`
Expected: FAIL — `heal_manifest_mtimes` not found

- [ ] **Step 3: Implement heal_manifest_mtimes**

Add to `crates/ripvec-core/src/cache/reindex.rs`:

```rust
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
```

- [ ] **Step 4: Call heal_manifest_mtimes after incremental diff**

In `incremental_path`, after the diff and before saving the manifest, if the diff found zero dirty files but had to hash-verify any (i.e., some mtimes were stale), heal and re-save:

After the `for dirty_path in &diff_result.dirty` loop and before `manifest.recompute_hashes()`, add:

```rust
    // Heal stale mtimes (e.g., after git clone where all mtimes are wrong
    // but content hashes match). This ensures the fast-path mtime check
    // works on subsequent runs.
    heal_manifest_mtimes(root, &mut manifest);
```

- [ ] **Step 5: Run tests**

Run: `cargo test -p ripvec-core cache::reindex::tests`
Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add crates/ripvec-core/src/cache/reindex.rs
git commit -m "feat: heal manifest mtimes after clone for fast-path optimization"
```

---

### Task 11: Add plugin command /ripvec-repo-index

**Files:**
- Create: `../my-claude-plugins/plugins/ripvec/commands/repo-index.md`

- [ ] **Step 1: Create the command**

Create `../my-claude-plugins/plugins/ripvec/commands/repo-index.md`:

```markdown
---
name: repo-index
description: Create a repo-level search index that can be committed to git
---

Call the `reindex` MCP tool with repo-level storage enabled:
```
reindex(repo_level: true)
```

This creates a `.ripvec/` directory at the project root containing:
- `config.toml` — model and version pin
- `cache/` — the search index (manifest + object store)

After indexing completes, commit `.ripvec/` to git so teammates get instant
semantic search without re-embedding.

If the index already exists as repo-local, this re-indexes incrementally
(only changed files are re-embedded).

Report the result: how many chunks indexed, from how many files, and remind
the user to `git add .ripvec/ && git commit` to share the index.
```

- [ ] **Step 2: Commit and push**

```bash
cd ../my-claude-plugins
git add plugins/ripvec/commands/repo-index.md
git commit -m "feat(ripvec): add /ripvec-repo-index command for repo-local indexing"
git push
```

---

### Task 12: Update documentation

**Files:**
- Modify: `README.md` — add repo-level indexing section
- Modify: `CLAUDE.md` — update cache resolution docs

- [ ] **Step 1: Add repo-level section to README.md**

Add a section after the existing indexing documentation in `README.md` (read it first to find the right location):

```markdown
### Repo-Level Indexing

Share pre-built search indices with your team by storing them in the repo:

```bash
ripvec --index --repo-level "your query"
```

This creates `.ripvec/` at the project root — commit it to git:

```bash
git add .ripvec/
git commit -m "chore: add ripvec search index"
```

Teammates who clone the repo get instant semantic search with zero embedding time.
The index auto-validates on first use (content hashes are checked, not file timestamps).

For large repos where the index is too big to commit, add to `.gitignore`:
```
.ripvec/cache/objects/
```
This keeps the config but skips the embedding data — teammates will re-embed on first use
but benefit from incremental updates afterward.
```

- [ ] **Step 2: Update CLAUDE.md cache resolution docs**

In `CLAUDE.md`, update the search modes / caching section to mention the resolution chain:

```
## Cache resolution
1. `--cache-dir` override (highest priority)
2. `.ripvec/config.toml` in directory tree → `.ripvec/cache/` (repo-local)
3. `RIPVEC_CACHE` environment variable
4. `~/.cache/ripvec/` (default)

Use `--repo-level --index` to create a repo-local index. Subsequent runs
auto-detect `.ripvec/` — no flag needed.
```

- [ ] **Step 3: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs: add repo-level indexing section to README and CLAUDE.md"
```

---

### Task 13: Run clippy and full verification

**Files:** None (verification only)

- [ ] **Step 1: Format check**

Run: `cargo fmt --check`
Expected: no formatting issues

- [ ] **Step 2: Clippy**

Run: `cargo clippy --all-targets -- -D warnings`
Expected: zero warnings

- [ ] **Step 3: Full test suite**

Run: `cargo test --workspace`
Expected: all tests pass

- [ ] **Step 4: Fix any issues found, commit fixes**

If clippy or tests fail, fix and commit:

```bash
git commit -m "fix: address clippy/test issues from repo-local cache implementation"
```
