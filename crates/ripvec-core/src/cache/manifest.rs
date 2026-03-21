//! Manifest tracking indexed files with a Merkle tree for fast change detection.
//!
//! The manifest stores per-file metadata (mtime, size, content hash, chunk count)
//! and per-directory metadata (mtime, hash). Directory hashes are computed bottom-up
//! from children's content hashes, enabling top-down pruning during diffs.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

/// Persistent manifest tracking all indexed files and their state.
#[derive(Debug, Serialize, Deserialize)]
pub struct Manifest {
    /// Cache format version (bump to invalidate all caches).
    pub version: u32,
    /// The embedding model used to generate this index.
    pub model_repo: String,
    /// Root Merkle hash (changes when any file changes).
    pub root_hash: String,
    /// Directory entries with mtime and computed Merkle hash.
    pub directories: BTreeMap<String, DirEntry>,
    /// File entries with metadata and content hash.
    pub files: BTreeMap<String, FileEntry>,
}

/// Directory metadata for Merkle tree change detection.
#[derive(Debug, Serialize, Deserialize)]
pub struct DirEntry {
    /// Merkle hash computed from children's content hashes.
    pub hash: String,
    /// Last modification time (seconds since epoch).
    pub mtime_secs: u64,
}

/// File metadata for change detection and cache lookup.
#[derive(Debug, Serialize, Deserialize)]
pub struct FileEntry {
    /// Last modification time (seconds since epoch).
    pub mtime_secs: u64,
    /// File size in bytes.
    pub size: u64,
    /// Blake3 hash of the file content (used as object store key).
    pub content_hash: String,
    /// Number of chunks extracted from this file.
    pub chunk_count: usize,
}

/// Current cache format version.
const MANIFEST_VERSION: u32 = 1;

impl Manifest {
    /// Create a new empty manifest for the given model.
    #[must_use]
    pub fn new(model_repo: &str) -> Self {
        Self {
            version: MANIFEST_VERSION,
            model_repo: model_repo.to_string(),
            root_hash: String::new(),
            directories: BTreeMap::new(),
            files: BTreeMap::new(),
        }
    }

    /// Register a file in the manifest.
    pub fn add_file(
        &mut self,
        relative_path: &str,
        mtime_secs: u64,
        size: u64,
        content_hash: &str,
        chunk_count: usize,
    ) {
        self.files.insert(
            relative_path.to_string(),
            FileEntry {
                mtime_secs,
                size,
                content_hash: content_hash.to_string(),
                chunk_count,
            },
        );
    }

    /// Remove a file from the manifest.
    pub fn remove_file(&mut self, relative_path: &str) {
        self.files.remove(relative_path);
    }

    /// Recompute all directory hashes and the root hash bottom-up.
    ///
    /// For each directory, the hash is blake3 of the sorted children's content
    /// hashes concatenated. The root hash is blake3 of all top-level entries.
    pub fn recompute_hashes(&mut self) {
        // Collect all directory paths from file paths
        let mut dir_children: BTreeMap<String, Vec<String>> = BTreeMap::new();

        for (file_path, entry) in &self.files {
            let path = Path::new(file_path);
            // Register hash with each ancestor directory
            let mut current = path.parent();
            while let Some(dir) = current {
                let dir_str = dir.to_string_lossy().to_string();
                if dir_str.is_empty() {
                    break;
                }
                dir_children
                    .entry(dir_str)
                    .or_default()
                    .push(entry.content_hash.clone());
                current = dir.parent();
            }
            // Top-level files go into root
            dir_children
                .entry(String::new())
                .or_default()
                .push(entry.content_hash.clone());
        }

        // Compute directory hashes
        self.directories.clear();
        for (dir_path, child_hashes) in &mut dir_children {
            if dir_path.is_empty() {
                continue;
            }
            child_hashes.sort();
            let combined = child_hashes.join("");
            let hash = blake3::hash(combined.as_bytes()).to_hex().to_string();
            self.directories.insert(
                dir_path.clone(),
                DirEntry {
                    hash,
                    mtime_secs: 0, // populated during diff, not recompute
                },
            );
        }

        // Root hash from all file content hashes
        let mut all_hashes: Vec<&str> = self
            .files
            .values()
            .map(|e| e.content_hash.as_str())
            .collect();
        all_hashes.sort_unstable();
        let combined = all_hashes.join("");
        self.root_hash = blake3::hash(combined.as_bytes()).to_hex().to_string();
    }

    /// Serialize to JSON string.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    pub fn to_json(&self) -> crate::Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| crate::Error::Other(anyhow::anyhow!("manifest serialization: {e}")))
    }

    /// Deserialize from JSON string.
    ///
    /// # Errors
    ///
    /// Returns an error if the JSON is invalid.
    pub fn from_json(json: &str) -> crate::Result<Self> {
        serde_json::from_str(json)
            .map_err(|e| crate::Error::Other(anyhow::anyhow!("manifest deserialization: {e}")))
    }

    /// Save manifest to a file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written.
    pub fn save(&self, path: &Path) -> crate::Result<()> {
        let json = self.to_json()?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| crate::Error::Io {
                path: parent.display().to_string(),
                source: e,
            })?;
        }
        std::fs::write(path, json).map_err(|e| crate::Error::Io {
            path: path.display().to_string(),
            source: e,
        })
    }

    /// Load manifest from a file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or parsed.
    pub fn load(path: &Path) -> crate::Result<Self> {
        let json = std::fs::read_to_string(path).map_err(|e| crate::Error::Io {
            path: path.display().to_string(),
            source: e,
        })?;
        Self::from_json(&json)
    }

    /// Check whether this manifest is compatible with the given model.
    #[must_use]
    pub fn is_compatible(&self, model_repo: &str) -> bool {
        self.version == MANIFEST_VERSION && self.model_repo == model_repo
    }

    /// Collect all content hashes referenced by files in this manifest.
    #[must_use]
    pub fn referenced_hashes(&self) -> Vec<String> {
        self.files
            .values()
            .map(|e| e.content_hash.clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_json() {
        let mut m = Manifest::new("BAAI/bge-small-en-v1.5");
        m.add_file("src/main.rs", 1000, 4523, "abc123", 8);
        m.add_file("src/lib.rs", 1001, 2000, "def456", 5);
        m.recompute_hashes();

        let json = m.to_json().unwrap();
        let loaded = Manifest::from_json(&json).unwrap();
        assert_eq!(loaded.files.len(), 2);
        assert_eq!(loaded.model_repo, "BAAI/bge-small-en-v1.5");
        assert!(!loaded.root_hash.is_empty());
    }

    #[test]
    fn root_hash_changes_on_file_change() {
        let mut m1 = Manifest::new("model");
        m1.add_file("a.rs", 1000, 100, "hash1", 5);
        m1.recompute_hashes();
        let h1 = m1.root_hash.clone();

        let mut m2 = Manifest::new("model");
        m2.add_file("a.rs", 1001, 100, "hash2", 5);
        m2.recompute_hashes();

        assert_ne!(h1, m2.root_hash);
    }

    #[test]
    fn root_hash_stable_for_same_content() {
        let mut m1 = Manifest::new("model");
        m1.add_file("a.rs", 1000, 100, "hash1", 5);
        m1.add_file("b.rs", 1001, 200, "hash2", 3);
        m1.recompute_hashes();

        let mut m2 = Manifest::new("model");
        m2.add_file("b.rs", 1001, 200, "hash2", 3);
        m2.add_file("a.rs", 1000, 100, "hash1", 5);
        m2.recompute_hashes();

        assert_eq!(m1.root_hash, m2.root_hash);
    }

    #[test]
    fn directory_hashes_computed() {
        let mut m = Manifest::new("model");
        m.add_file("src/main.rs", 1000, 100, "hash1", 5);
        m.add_file("src/lib.rs", 1001, 200, "hash2", 3);
        m.add_file("tests/test.rs", 1002, 300, "hash3", 2);
        m.recompute_hashes();

        assert!(m.directories.contains_key("src"));
        assert!(m.directories.contains_key("tests"));
    }

    #[test]
    fn save_and_load() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("manifest.json");

        let mut m = Manifest::new("test-model");
        m.add_file("foo.rs", 100, 50, "aaa", 1);
        m.recompute_hashes();
        m.save(&path).unwrap();

        let loaded = Manifest::load(&path).unwrap();
        assert_eq!(loaded.files.len(), 1);
        assert_eq!(loaded.root_hash, m.root_hash);
    }

    #[test]
    fn is_compatible() {
        let m = Manifest::new("BAAI/bge-small-en-v1.5");
        assert!(m.is_compatible("BAAI/bge-small-en-v1.5"));
        assert!(!m.is_compatible("nomic-ai/CodeRankEmbed"));
    }

    #[test]
    fn referenced_hashes() {
        let mut m = Manifest::new("model");
        m.add_file("a.rs", 1, 1, "hash_a", 1);
        m.add_file("b.rs", 2, 2, "hash_b", 2);
        let hashes = m.referenced_hashes();
        assert_eq!(hashes.len(), 2);
        assert!(hashes.contains(&"hash_a".to_string()));
        assert!(hashes.contains(&"hash_b".to_string()));
    }
}
