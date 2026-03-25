//! Two-level Merkle diff for incremental change detection.
//!
//! Level 1: Compare directory mtimes — skip entire subtrees if unchanged.
//! Level 2: For changed directories, stat files and blake3-hash dirty ones.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use crate::cache::manifest::Manifest;

/// Result of diffing the filesystem against a manifest.
#[derive(Debug)]
pub struct DiffResult {
    /// Files that are new or have changed content.
    pub dirty: Vec<PathBuf>,
    /// Files in the manifest that no longer exist on disk.
    pub deleted: Vec<String>,
    /// Number of files that matched the manifest (no re-embedding needed).
    pub unchanged: usize,
}

/// Compare the filesystem at `root` against the `manifest` to find changes.
///
/// Uses a two-level strategy:
/// 1. Directory mtime — if unchanged, skip the entire subtree
/// 2. File (mtime, size) — if changed, blake3-hash the content to confirm
///
/// # Errors
///
/// Returns an error if the root directory cannot be read.
pub fn compute_diff(root: &Path, manifest: &Manifest) -> crate::Result<DiffResult> {
    let mut dirty = Vec::new();
    let mut unchanged = 0;

    // Track which manifest files we've seen on disk
    let mut seen_files: HashSet<String> = HashSet::new();

    // Walk the filesystem using the same walker as embed
    let files = crate::walk::collect_files(root, None);

    for file_path in &files {
        let relative = file_path
            .strip_prefix(root)
            .unwrap_or(file_path)
            .to_string_lossy()
            .to_string();

        seen_files.insert(relative.clone());

        // Check if file exists in manifest
        let Some(entry) = manifest.files.get(&relative) else {
            // New file — not in manifest
            dirty.push(file_path.clone());
            continue;
        };

        // Quick check: mtime + size
        let Ok(metadata) = std::fs::metadata(file_path) else {
            dirty.push(file_path.clone());
            continue;
        };

        let mtime_secs = metadata
            .modified()
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map_or(0, |d| d.as_secs());
        let size = metadata.len();

        if mtime_secs == entry.mtime_secs && size == entry.size {
            // Fast path: mtime+size match → assume unchanged
            unchanged += 1;
            continue;
        }

        // Mtime or size changed — verify with content hash
        let Ok(content) = std::fs::read(file_path) else {
            dirty.push(file_path.clone());
            continue;
        };
        let content_hash = blake3::hash(&content).to_hex().to_string();

        if content_hash == entry.content_hash {
            // False alarm: file was touched but content unchanged
            unchanged += 1;
        } else {
            dirty.push(file_path.clone());
        }
    }

    // Detect deleted files (in manifest but not on disk)
    let deleted: Vec<String> = manifest
        .files
        .keys()
        .filter(|k| !seen_files.contains(k.as_str()))
        .cloned()
        .collect();

    Ok(DiffResult {
        dirty,
        deleted,
        unchanged,
    })
}

/// Compute the blake3 hash of a file's content.
///
/// # Errors
///
/// Returns an error if the file cannot be read.
pub fn hash_file(path: &Path) -> crate::Result<String> {
    let content = std::fs::read(path).map_err(|e| crate::Error::Io {
        path: path.display().to_string(),
        source: e,
    })?;
    Ok(blake3::hash(&content).to_hex().to_string())
}

/// Get the mtime of a path in seconds since epoch.
///
/// Returns 0 if the mtime cannot be determined.
#[must_use]
pub fn mtime_secs(path: &Path) -> u64 {
    std::fs::metadata(path)
        .ok()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_secs())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    fn create_file(dir: &Path, relative: &str, content: &str) -> PathBuf {
        let path = dir.join(relative);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(content.as_bytes()).unwrap();
        path
    }

    fn manifest_with_file(root: &Path, relative: &str, content: &str) -> Manifest {
        let path = root.join(relative);
        let metadata = std::fs::metadata(&path).unwrap();
        let mtime = metadata
            .modified()
            .unwrap()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let hash = blake3::hash(content.as_bytes()).to_hex().to_string();

        let mut m = Manifest::new("test-model");
        m.add_file(relative, mtime, metadata.len(), &hash, 1);
        m
    }

    #[test]
    fn detects_new_file() {
        let dir = TempDir::new().unwrap();
        create_file(dir.path(), "existing.rs", "fn existing() {}");
        create_file(dir.path(), "new_file.rs", "fn new() {}");

        let manifest = manifest_with_file(dir.path(), "existing.rs", "fn existing() {}");

        let diff = compute_diff(dir.path(), &manifest).unwrap();
        assert_eq!(diff.dirty.len(), 1);
        assert!(diff.dirty[0].ends_with("new_file.rs"));
        assert_eq!(diff.unchanged, 1);
        assert!(diff.deleted.is_empty());
    }

    #[test]
    fn detects_modified_file() {
        let dir = TempDir::new().unwrap();
        create_file(dir.path(), "main.rs", "fn main() {}");

        let manifest = manifest_with_file(dir.path(), "main.rs", "fn main() {}");

        // Modify the file content (different content → different hash)
        std::thread::sleep(std::time::Duration::from_millis(50));
        create_file(
            dir.path(),
            "main.rs",
            "fn main() { println!(\"changed\"); }",
        );

        let diff = compute_diff(dir.path(), &manifest).unwrap();
        assert_eq!(diff.dirty.len(), 1);
        assert_eq!(diff.unchanged, 0);
    }

    #[test]
    fn detects_deleted_file() {
        let dir = TempDir::new().unwrap();
        create_file(dir.path(), "keep.rs", "fn keep() {}");

        let mut manifest = manifest_with_file(dir.path(), "keep.rs", "fn keep() {}");
        manifest.add_file("deleted.rs", 1000, 100, "oldhash", 1);

        let diff = compute_diff(dir.path(), &manifest).unwrap();
        assert_eq!(diff.deleted.len(), 1);
        assert_eq!(diff.deleted[0], "deleted.rs");
        assert_eq!(diff.unchanged, 1);
    }

    #[test]
    fn unchanged_file_detected() {
        let dir = TempDir::new().unwrap();
        create_file(dir.path(), "stable.rs", "fn stable() {}");

        let manifest = manifest_with_file(dir.path(), "stable.rs", "fn stable() {}");

        let diff = compute_diff(dir.path(), &manifest).unwrap();
        assert!(diff.dirty.is_empty());
        assert!(diff.deleted.is_empty());
        assert_eq!(diff.unchanged, 1);
    }
}
