//! Content-addressed object store with git-style sharding.
//!
//! Objects are stored at `<root>/xx/yyyyyyyy...` where `xx` is the first
//! two hex chars of the blake3 hash and `yyyy...` is the remainder. This
//! prevents too many files in a single directory.

use std::path::{Path, PathBuf};

/// Content-addressed object store with git-style `xx/hash` sharding.
pub struct ObjectStore {
    /// Root directory of the store (e.g., `~/.cache/ripvec/<project>/objects/`).
    root: PathBuf,
}

impl ObjectStore {
    /// Create a new store rooted at the given directory.
    ///
    /// The directory is created on first write, not at construction time.
    #[must_use]
    pub fn new(root: &Path) -> Self {
        Self {
            root: root.to_path_buf(),
        }
    }

    /// Write data to the store under the given hash.
    ///
    /// Creates the `xx/` prefix directory if it doesn't exist.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be created or the file
    /// cannot be written.
    pub fn write(&self, hash: &str, data: &[u8]) -> crate::Result<()> {
        let path = self.object_path(hash);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| crate::Error::Io {
                path: parent.display().to_string(),
                source: e,
            })?;
        }
        std::fs::write(&path, data).map_err(|e| crate::Error::Io {
            path: path.display().to_string(),
            source: e,
        })
    }

    /// Read data from the store for the given hash.
    ///
    /// # Errors
    ///
    /// Returns an error if the object file does not exist or cannot be read.
    pub fn read(&self, hash: &str) -> crate::Result<Vec<u8>> {
        let path = self.object_path(hash);
        std::fs::read(&path).map_err(|e| crate::Error::Io {
            path: path.display().to_string(),
            source: e,
        })
    }

    /// Check whether an object exists in the store.
    #[must_use]
    pub fn exists(&self, hash: &str) -> bool {
        self.object_path(hash).exists()
    }

    /// Remove objects not in the `keep` set. Returns the number of removed objects.
    ///
    /// Also removes empty `xx/` prefix directories after cleanup.
    ///
    /// # Errors
    ///
    /// Returns an error if the store directory cannot be read.
    pub fn gc(&self, keep: &[String]) -> crate::Result<usize> {
        let keep_set: std::collections::HashSet<&str> = keep.iter().map(String::as_str).collect();
        let mut removed = 0;

        let Ok(entries) = std::fs::read_dir(&self.root) else {
            return Ok(0); // store doesn't exist yet
        };

        for prefix_entry in entries.flatten() {
            let prefix_path = prefix_entry.path();
            if !prefix_path.is_dir() {
                continue;
            }
            let prefix = prefix_entry.file_name();
            let prefix_str = prefix.to_string_lossy();

            if let Ok(files) = std::fs::read_dir(&prefix_path) {
                for file_entry in files.flatten() {
                    let file_name = file_entry.file_name();
                    let hash = format!("{}{}", prefix_str, file_name.to_string_lossy());
                    if !keep_set.contains(hash.as_str())
                        && std::fs::remove_file(file_entry.path()).is_ok()
                    {
                        removed += 1;
                    }
                }
            }

            // Remove empty prefix directory
            let _ = std::fs::remove_dir(&prefix_path); // fails silently if not empty
        }

        Ok(removed)
    }

    /// Resolve the filesystem path for an object hash.
    fn object_path(&self, hash: &str) -> PathBuf {
        debug_assert!(
            hash.len() >= 3,
            "hash must be at least 3 chars for sharding"
        );
        self.root.join(&hash[..2]).join(&hash[2..])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn write_and_read_round_trip() {
        let dir = TempDir::new().unwrap();
        let store = ObjectStore::new(dir.path());
        let hash = "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890";
        let data = b"hello world";
        store.write(hash, data).unwrap();
        let read = store.read(hash).unwrap();
        assert_eq!(read, data);
    }

    #[test]
    fn git_style_sharding() {
        let dir = TempDir::new().unwrap();
        let store = ObjectStore::new(dir.path());
        let hash = "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890";
        store.write(hash, b"data").unwrap();
        assert!(dir.path().join("ab").join(&hash[2..]).exists());
    }

    #[test]
    fn gc_removes_unreferenced() {
        let dir = TempDir::new().unwrap();
        let store = ObjectStore::new(dir.path());
        let h1 = "aaaa0000111122223333444455556666aaaa0000111122223333444455556666";
        let h2 = "bbbb0000111122223333444455556666bbbb0000111122223333444455556666";
        store.write(h1, b"keep").unwrap();
        store.write(h2, b"delete").unwrap();
        let removed = store.gc(&[h1.to_string()]).unwrap();
        assert_eq!(removed, 1);
        assert!(store.exists(h1));
        assert!(!store.exists(h2));
    }

    #[test]
    fn gc_empty_store_returns_zero() {
        let dir = TempDir::new().unwrap();
        let store = ObjectStore::new(dir.path());
        assert_eq!(store.gc(&[]).unwrap(), 0);
    }

    #[test]
    fn read_nonexistent_returns_error() {
        let dir = TempDir::new().unwrap();
        let store = ObjectStore::new(dir.path());
        assert!(store.read("abc123").is_err());
    }
}
