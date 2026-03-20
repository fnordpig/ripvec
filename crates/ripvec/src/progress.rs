//! Progress bar display using indicatif.
//!
//! Provides lightweight wrappers around [`indicatif`] progress bars for
//! the embedding pipeline phases. Used only when stderr is a TTY and
//! `--profile` is not active.

use indicatif::{ProgressBar, ProgressStyle};
use ripvec_core::profile::EmbedProgress;
use std::collections::VecDeque;
use std::fmt::Write as _;
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

/// Sparkline block characters, from lowest to highest.
const SPARK_CHARS: &[char] = &['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

/// Number of rate samples to keep for the sparkline.
const SPARK_WIDTH: usize = 12;

/// Width of the inference gauge in characters.
const GAUGE_WIDTH: usize = 10;

/// Rich embed progress display with sparkline, inference gauge, and animations.
pub struct EmbedDisplay {
    /// The underlying indicatif progress bar.
    bar: ProgressBar,
    /// Rolling window of recent throughput rates for the sparkline.
    rate_history: VecDeque<f64>,
    /// All-time peak rate (sticky max for sparkline scaling).
    peak_rate: f64,
}

impl EmbedDisplay {
    /// Create a new embed display for `total` chunks.
    pub fn new(total: u64) -> Self {
        let bar = ProgressBar::new(total);
        bar.set_style(
            ProgressStyle::with_template(
                "{spinner:.cyan} {wide_bar:.cyan/blue} {pos}/{len} {msg}  ETA {eta}",
            )
            .unwrap()
            .tick_chars("🧠⚡🔥✨💫🌀🔮🧬")
            .progress_chars("█▶░"),
        );
        bar.set_message("warming up\u{2026}");
        bar.enable_steady_tick(Duration::from_millis(100));
        Self {
            bar,
            rate_history: VecDeque::with_capacity(SPARK_WIDTH),
            peak_rate: 0.0,
        }
    }

    /// Update the display with new progress data.
    pub fn update(&mut self, p: &EmbedProgress) {
        self.bar.set_position(p.done as u64);

        // Update sparkline history and sticky peak
        if p.overall_rate > 0.0 {
            let rate = p.window_rate.max(p.overall_rate * 0.1);
            if rate > self.peak_rate {
                self.peak_rate = rate;
            }
            self.rate_history.push_back(rate);
            if self.rate_history.len() > SPARK_WIDTH {
                self.rate_history.pop_front();
            }
        }

        let sparkline = render_sparkline(self.rate_history.make_contiguous(), self.peak_rate);
        let gauge = render_gauge(p.inference_pct, p.lock_wait_pct);

        self.bar.set_message(format!(
            "\x1b[33m{:.0}/s\x1b[0m {sparkline} \x1b[36m{gauge}\x1b[0m",
            p.overall_rate,
        ));
    }

    /// Finish the display with a summary message.
    pub fn finish(&self, chunks: usize, n_files: usize) {
        self.bar
            .set_style(ProgressStyle::with_template("{msg}").unwrap());
        self.bar.finish_with_message(format!(
            "\x1b[32m\u{2714}\x1b[0m Indexed \x1b[1m{chunks}\x1b[0m chunks from \x1b[1m{n_files}\x1b[0m files",
        ));
    }

    /// Finish and clear the display (for oneshot mode).
    pub fn finish_and_clear(&self) {
        self.bar.finish_and_clear();
    }
}

/// Render a sparkline from a slice of rate values.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "display-only: normalized index is always 0..=7"
)]
fn render_sparkline(rates: &[f64], peak: f64) -> String {
    if rates.is_empty() {
        return " ".repeat(SPARK_WIDTH);
    }
    // Scale against the all-time peak so bars only shrink relative to best performance.
    let max = peak.max(rates.iter().copied().fold(f64::NEG_INFINITY, f64::max));
    let range = max.max(1.0);

    // Pad left if we don't have enough history yet
    let pad = SPARK_WIDTH.saturating_sub(rates.len());
    let mut s = " ".repeat(pad);

    for &rate in rates {
        let normalized = (rate / range * 7.0).round() as usize;
        let idx = normalized.min(SPARK_CHARS.len() - 1);
        // Color: green for high, yellow for mid, red for low
        let color = if normalized >= 5 {
            "\x1b[32m" // green
        } else if normalized >= 2 {
            "\x1b[33m" // yellow
        } else {
            "\x1b[31m" // red
        };
        s.push_str(color);
        s.push(SPARK_CHARS[idx]);
        s.push_str("\x1b[0m");
    }
    s
}

/// Render an inference/lock-wait gauge.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "display-only: gauge width is a small constant"
)]
fn render_gauge(inference_pct: f64, lock_wait_pct: f64) -> String {
    let inf_blocks = (inference_pct * GAUGE_WIDTH as f64).round() as usize;
    let lock_blocks = GAUGE_WIDTH - inf_blocks.min(GAUGE_WIDTH);

    let mut s = String::with_capacity(GAUGE_WIDTH * 4 + 20);

    // Inference portion: bright green filled blocks
    if inf_blocks > 0 {
        s.push_str("\x1b[32m");
        for _ in 0..inf_blocks {
            s.push('\u{2593}'); // ▓
        }
        s.push_str("\x1b[0m");
    }

    // Lock-wait portion: dim red empty blocks
    if lock_blocks > 0 {
        s.push_str("\x1b[31m");
        for _ in 0..lock_blocks {
            s.push('\u{2591}'); // ░
        }
        s.push_str("\x1b[0m");
    }

    // Label
    if lock_wait_pct > 0.05 {
        let _ = write!(s, " \x1b[31m{:.0}% wait\x1b[0m", lock_wait_pct * 100.0);
    }

    s
}
