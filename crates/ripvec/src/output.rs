//! Result formatting for different output modes.

use crate::cli::OutputFormat;
use owo_colors::OwoColorize as _;
use ripvec_core::embed::SearchResult;

/// Format and print search results according to the chosen output format.
pub fn print_results(results: &[SearchResult], format: &OutputFormat) {
    match format {
        OutputFormat::Plain => print_plain(results),
        OutputFormat::Json => print_json(results),
        OutputFormat::Color => print_color(results),
    }
}

fn print_plain(results: &[SearchResult]) {
    for (i, r) in results.iter().enumerate() {
        println!(
            "{}. {} ({}:{}-{}) [{:.3}]",
            i + 1,
            r.chunk.name,
            r.chunk.file_path,
            r.chunk.start_line,
            r.chunk.end_line,
            r.similarity,
        );
        println!("{}", r.chunk.content);
        println!();
    }
}

/// Infer a language name from a file path's extension.
fn language_from_path(file_path: &str) -> &'static str {
    let ext = std::path::Path::new(file_path)
        .extension()
        .and_then(|e| e.to_str());
    match ext {
        Some("rs") => "rust",
        Some("py" | "pyi") => "python",
        Some("js" | "mjs" | "cjs") => "javascript",
        Some("ts" | "mts" | "cts") => "typescript",
        Some("tsx") => "tsx",
        Some("jsx") => "jsx",
        Some("go") => "go",
        Some("java") => "java",
        Some("c" | "h") => "c",
        Some("cpp" | "cc" | "cxx" | "hpp" | "hxx") => "cpp",
        Some("rb") => "ruby",
        Some("sql") => "sql",
        Some("toml") => "toml",
        Some("yaml" | "yml") => "yaml",
        Some("json") => "json",
        Some("md" | "markdown") => "markdown",
        Some("sh" | "bash" | "zsh") => "shell",
        Some(_) | None => "text",
    }
}

/// Print results as JSON Lines (one JSON object per line).
fn print_json(results: &[SearchResult]) {
    for r in results {
        let obj = serde_json::json!({
            "file_path": r.chunk.file_path,
            "start_line": r.chunk.start_line,
            "end_line": r.chunk.end_line,
            "name": r.chunk.name,
            "kind": r.chunk.kind,
            "similarity": r.similarity,
            "content": r.chunk.content,
            "language": language_from_path(&r.chunk.file_path),
        });
        // Each line is a compact JSON object (JSON Lines format)
        if let Ok(line) = serde_json::to_string(&obj) {
            println!("{line}");
        }
    }
}

fn print_color(results: &[SearchResult]) {
    for (i, r) in results.iter().enumerate() {
        let index = format!("{}.", i + 1);
        let location = format!(
            "{}:{}-{}",
            r.chunk.file_path, r.chunk.start_line, r.chunk.end_line
        );
        let score = format!("[{:.3}]", r.similarity);
        println!(
            "{} {} {} {}",
            index.green().bold(),
            r.chunk.name.bold(),
            location.cyan(),
            score.yellow(),
        );
        println!("{}", r.chunk.content);
        println!();
    }
}
