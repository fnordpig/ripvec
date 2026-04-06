//! Repository-local configuration for ripvec cache.
//!
//! Stores and discovers `.ripvec/config.toml` files that pin the embedding
//! model and enable repo-local index storage for a project.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{Error, Result};

/// Top-level structure for `.ripvec/config.toml`.
#[derive(Debug, Serialize, Deserialize)]
pub struct RepoConfig {
    /// Cache settings for this repository.
    pub cache: CacheConfig,
}

/// Cache configuration stored in `.ripvec/config.toml`.
#[derive(Debug, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Whether to use a repo-local cache directory instead of the global one.
    pub local: bool,
    /// The embedding model repo used to build this index (e.g. `"BAAI/bge-small-en-v1.5"`).
    pub model: String,
    /// Manifest format version string (e.g. `"3"`).
    pub version: String,
}

impl RepoConfig {
    /// Create a new config with `local = true` for the given model and version.
    #[must_use]
    pub fn new(model: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            cache: CacheConfig {
                local: true,
                model: model.into(),
                version: version.into(),
            },
        }
    }

    /// Serialize to a TOML string.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Other`] if serialization fails.
    pub fn to_toml(&self) -> Result<String> {
        toml::to_string(self)
            .map_err(|e| Error::Other(anyhow::anyhow!("failed to serialize config: {e}")))
    }

    /// Deserialize from a TOML string.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Other`] if deserialization fails.
    pub fn from_toml(s: &str) -> Result<Self> {
        toml::from_str(s)
            .map_err(|e| Error::Other(anyhow::anyhow!("failed to deserialize config: {e}")))
    }

    /// Write the config to `<path>/config.toml`, creating parent directories as needed.
    ///
    /// `path` should be the `.ripvec/` directory.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] on filesystem errors, or [`Error::Other`] on
    /// serialization failure.
    pub fn save(&self, path: &Path) -> Result<()> {
        std::fs::create_dir_all(path).map_err(|source| Error::Io {
            path: path.display().to_string(),
            source,
        })?;
        let file = path.join("config.toml");
        let contents = self.to_toml()?;
        std::fs::write(&file, contents).map_err(|source| Error::Io {
            path: file.display().to_string(),
            source,
        })
    }

    /// Load config from `<path>/config.toml`.
    ///
    /// `path` should be the `.ripvec/` directory.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] if the file cannot be read, or [`Error::Other`]
    /// on parse failure.
    pub fn load(path: &Path) -> Result<Self> {
        let file = path.join("config.toml");
        let contents = std::fs::read_to_string(&file).map_err(|source| Error::Io {
            path: file.display().to_string(),
            source,
        })?;
        Self::from_toml(&contents)
    }
}

/// Walk up the directory tree from `start` looking for `.ripvec/config.toml`.
///
/// Returns `Some(ripvec_dir)` (the `.ripvec/` directory path) if a config is
/// found **and** `cache.local == true`. Returns `None` if no config is found
/// or if `local` is `false`.
#[must_use]
pub fn find_repo_config(start: &Path) -> Option<PathBuf> {
    let mut current = start.to_path_buf();
    loop {
        let candidate = current.join(".ripvec");
        if candidate.join("config.toml").exists() {
            if let Ok(cfg) = RepoConfig::load(&candidate) {
                if cfg.cache.local {
                    return Some(candidate);
                }
            }
            // Config exists but local == false (or unreadable) — stop searching.
            return None;
        }
        match current.parent() {
            Some(parent) => current = parent.to_path_buf(),
            None => return None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn round_trip_toml() {
        let cfg = RepoConfig::new("BAAI/bge-small-en-v1.5", "3");
        let toml_str = cfg.to_toml().expect("serialize");
        let restored = RepoConfig::from_toml(&toml_str).expect("deserialize");
        assert!(restored.cache.local);
        assert_eq!(restored.cache.model, "BAAI/bge-small-en-v1.5");
        assert_eq!(restored.cache.version, "3");
    }

    #[test]
    fn save_and_load() {
        let dir = TempDir::new().expect("tempdir");
        let ripvec_dir = dir.path().join(".ripvec");
        let cfg = RepoConfig::new("nomic-ai/modernbert-embed-base", "3");
        cfg.save(&ripvec_dir).expect("save");
        assert!(ripvec_dir.join("config.toml").exists());
        let loaded = RepoConfig::load(&ripvec_dir).expect("load");
        assert!(loaded.cache.local);
        assert_eq!(loaded.cache.model, "nomic-ai/modernbert-embed-base");
        assert_eq!(loaded.cache.version, "3");
    }

    #[test]
    fn find_repo_config_in_current_dir() {
        let dir = TempDir::new().expect("tempdir");
        let ripvec_dir = dir.path().join(".ripvec");
        RepoConfig::new("BAAI/bge-small-en-v1.5", "3")
            .save(&ripvec_dir)
            .expect("save");
        let found = find_repo_config(dir.path());
        assert_eq!(found.as_deref(), Some(ripvec_dir.as_path()));
    }

    #[test]
    fn find_repo_config_in_parent_dir() {
        let dir = TempDir::new().expect("tempdir");
        let ripvec_dir = dir.path().join(".ripvec");
        RepoConfig::new("BAAI/bge-small-en-v1.5", "3")
            .save(&ripvec_dir)
            .expect("save");
        let subdir = dir.path().join("src").join("foo");
        std::fs::create_dir_all(&subdir).expect("mkdir");
        let found = find_repo_config(&subdir);
        assert_eq!(found.as_deref(), Some(ripvec_dir.as_path()));
    }

    #[test]
    fn find_repo_config_not_found() {
        let dir = TempDir::new().expect("tempdir");
        // No .ripvec directory — should return None.
        assert!(find_repo_config(dir.path()).is_none());
    }

    #[test]
    fn find_repo_config_ignores_disabled() {
        let dir = TempDir::new().expect("tempdir");
        let ripvec_dir = dir.path().join(".ripvec");
        // Manually write config with local = false.
        std::fs::create_dir_all(&ripvec_dir).expect("mkdir");
        let cfg_str =
            "[cache]\nlocal = false\nmodel = \"BAAI/bge-small-en-v1.5\"\nversion = \"3\"\n";
        std::fs::write(ripvec_dir.join("config.toml"), cfg_str).expect("write");
        assert!(find_repo_config(dir.path()).is_none());
    }
}
