//! Dump token length distribution for chunks in a directory.

#![expect(
    clippy::stable_sort_primitive,
    clippy::cast_precision_loss,
    reason = "example utility — clarity over micro-optimization"
)]

use std::path::Path;

fn main() {
    let dir = std::env::args().nth(1).unwrap_or_else(|| ".".into());
    let tokenizer = ripvec_core::tokenize::load_tokenizer("BAAI/bge-small-en-v1.5").unwrap();
    let files = ripvec_core::walk::collect_files(Path::new(&dir));
    let mut token_counts: Vec<usize> = Vec::new();

    for path in &files {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        let Some(lang) = ripvec_core::languages::config_for_extension(ext) else {
            continue;
        };
        let Ok(source) = std::fs::read_to_string(path) else {
            continue;
        };
        let cfg = ripvec_core::chunk::ChunkConfig::default();
        let chunks = ripvec_core::chunk::chunk_file(path, &source, &lang, &cfg);
        for chunk in &chunks {
            if let Ok(enc) = tokenizer.encode(chunk.content.as_str(), true) {
                token_counts.push(enc.get_ids().len());
            }
        }
    }

    token_counts.sort();
    let n = token_counts.len();
    if n == 0 {
        println!("No chunks found");
        return;
    }
    println!("Total chunks: {n}");
    println!("P25: {}", token_counts[n / 4]);
    println!("P50: {}", token_counts[n / 2]);
    println!("P75: {}", token_counts[3 * n / 4]);
    println!("P90: {}", token_counts[9 * n / 10]);
    println!("P95: {}", token_counts[95 * n / 100]);
    println!("P99: {}", token_counts[99 * n / 100]);
    println!("Max: {}", token_counts[n - 1]);
    println!();

    let buckets = [0usize, 16, 32, 64, 128, 256, 512, 1024, usize::MAX];
    for w in buckets.windows(2) {
        let count = token_counts
            .iter()
            .filter(|&&t| t >= w[0] && t < w[1])
            .count();
        let pct = count as f64 / n as f64 * 100.0;
        if w[1] == usize::MAX {
            println!("{:>5}+     : {:>6} ({:.1}%)", w[0], count, pct);
        } else {
            println!("{:>5}-{:<5}: {:>6} ({:.1}%)", w[0], w[1], count, pct);
        }
    }
}
