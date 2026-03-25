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
/// When `file_type` is `Some`, only files matching that type (using
/// ripgrep's built-in type database, e.g. "rust", "python", "js") are
/// collected.
///
/// Uses the `ignore` crate's parallel walker for multi-threaded traversal.
#[must_use]
pub fn collect_files(root: &Path, file_type: Option<&str>) -> Vec<PathBuf> {
    let files = Mutex::new(Vec::new());

    let mut builder = WalkBuilder::new(root);
    builder.hidden(true).git_ignore(true).git_global(true);

    if let Some(ft) = file_type {
        let mut types_builder = ignore::types::TypesBuilder::new();
        types_builder.add_defaults();
        types_builder.select(ft);
        if let Ok(types) = types_builder.build() {
            builder.types(types);
        }
    }

    builder.build_parallel().run(|| {
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
                )
            {
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
