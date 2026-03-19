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
