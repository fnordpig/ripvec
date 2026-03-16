//! Result formatting for different output modes.

use crate::cli::OutputFormat;
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

fn print_json(results: &[SearchResult]) {
    let items: Vec<serde_json::Value> = results
        .iter()
        .map(|r| {
            serde_json::json!({
                "name": r.chunk.name,
                "file": r.chunk.file_path,
                "start_line": r.chunk.start_line,
                "end_line": r.chunk.end_line,
                "similarity": r.similarity,
                "content": r.chunk.content,
            })
        })
        .collect();
    println!(
        "{}",
        serde_json::to_string_pretty(&items).unwrap_or_default()
    );
}

fn print_color(results: &[SearchResult]) {
    for (i, r) in results.iter().enumerate() {
        println!(
            "\x1b[1;32m{}.\x1b[0m \x1b[1m{}\x1b[0m \x1b[36m{}:{}-{}\x1b[0m \x1b[33m[{:.3}]\x1b[0m",
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
