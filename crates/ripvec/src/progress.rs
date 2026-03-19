//! Progress bar display using indicatif.
//!
//! Provides lightweight wrappers around [`indicatif`] progress bars for
//! the embedding pipeline phases. Used only when stderr is a TTY and
//! `--profile` is not active.

use indicatif::{ProgressBar, ProgressStyle};
use std::time::Duration;

/// Create a spinner for indeterminate phases (model load, walk).
///
/// The spinner auto-ticks at 80 ms intervals and displays `msg` alongside
/// the animation. Call [`ProgressBar::finish_with_message`] or
/// [`ProgressBar::finish`] to stop it.
pub fn spinner(msg: &str) -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::with_template("{spinner:.cyan} {msg}")
            .unwrap()
            .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"),
    );
    pb.set_message(msg.to_string());
    pb.enable_steady_tick(Duration::from_millis(80));
    pb
}

/// Create a determinate progress bar for the embedding phase.
///
/// Displays a filled bar with position/total, throughput, and ETA.
/// `total` is the number of chunks to embed.
#[expect(
    dead_code,
    reason = "reserved for per-chunk progress; wired in a future task"
)]
pub fn embed_bar(total: u64) -> ProgressBar {
    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::with_template("{bar:40.cyan/blue} {pos}/{len} │ {per_sec} │ ETA {eta}")
            .unwrap()
            .progress_chars("█▉▊▋▌▍▎▏  "),
    );
    pb
}
