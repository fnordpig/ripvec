//! Directory traversal using the `ignore` crate.
//!
//! Respects `.gitignore` rules, skips hidden files, and filters
//! to files with supported source extensions.

use ignore::WalkBuilder;
use std::path::{Path, PathBuf};

/// Walk a directory tree and collect paths to supported source files.
///
/// Respects `.gitignore` rules and skips hidden files and directories.
#[must_use]
pub fn collect_files(root: &Path) -> Vec<PathBuf> {
    WalkBuilder::new(root)
        .hidden(true)
        .git_ignore(true)
        .git_global(true)
        .build()
        .filter_map(std::result::Result::ok)
        .filter(|e| e.file_type().is_some_and(|ft| ft.is_file()))
        .filter(|e| {
            e.path()
                .extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| crate::languages::config_for_extension(ext).is_some())
        })
        .map(ignore::DirEntry::into_path)
        .collect()
}
