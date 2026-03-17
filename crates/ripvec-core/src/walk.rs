//! Parallel directory traversal using the `ignore` crate.
//!
//! Respects `.gitignore` rules, skips hidden files, and filters
//! to files with supported source extensions. Uses `build_parallel()`
//! for multi-threaded file discovery.

use ignore::WalkBuilder;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Walk a directory tree in parallel and collect paths to supported source files.
///
/// Respects `.gitignore` rules and skips hidden files and directories.
/// Uses the `ignore` crate's parallel walker for multi-threaded traversal.
#[must_use]
pub fn collect_files(root: &Path) -> Vec<PathBuf> {
    let files = Mutex::new(Vec::new());

    WalkBuilder::new(root)
        .hidden(true)
        .git_ignore(true)
        .git_global(true)
        .build_parallel()
        .run(|| {
            Box::new(|entry| {
                let Ok(entry) = entry else {
                    return ignore::WalkState::Continue;
                };
                let dominated_file = entry.file_type().is_some_and(|ft| ft.is_file());
                if !dominated_file {
                    return ignore::WalkState::Continue;
                }
                let dominated_ext = entry
                    .path()
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .is_some_and(|ext| crate::languages::config_for_extension(ext).is_some());
                if !dominated_ext {
                    return ignore::WalkState::Continue;
                }
                if let Ok(mut files) = files.lock() {
                    files.push(entry.into_path());
                }
                ignore::WalkState::Continue
            })
        });

    files.into_inner().unwrap_or_default()
}
