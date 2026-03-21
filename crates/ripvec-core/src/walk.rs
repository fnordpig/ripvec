//! Parallel directory traversal using the `ignore` crate.
//!
//! Respects `.gitignore` rules, skips hidden files, and filters
//! to files with supported source extensions. Uses `build_parallel()`
//! for multi-threaded file discovery.

use ignore::WalkBuilder;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Walk a directory tree in parallel and collect file paths.
///
/// Respects `.gitignore` rules and skips hidden files and directories.
/// Collects all files — the chunking phase decides whether to use
/// tree-sitter (known extensions) or sliding-window fallback (unknown).
///
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
                if !entry.file_type().is_some_and(|ft| ft.is_file()) {
                    return ignore::WalkState::Continue;
                }
                // Skip known generated/binary files that add noise to the index
                if let Some(name) = entry.path().file_name().and_then(|n| n.to_str())
                    && matches!(
                        name,
                        "Cargo.lock"
                            | "package-lock.json"
                            | "yarn.lock"
                            | "pnpm-lock.yaml"
                            | "poetry.lock"
                            | "Gemfile.lock"
                            | "go.sum"
                    ) {
                        return ignore::WalkState::Continue;
                    }
                if let Ok(mut files) = files.lock() {
                    files.push(entry.into_path());
                }
                ignore::WalkState::Continue
            })
        });

    let mut files = files.into_inner().unwrap_or_default();
    files.sort();
    files
}
