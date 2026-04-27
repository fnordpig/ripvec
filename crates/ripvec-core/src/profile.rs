//! Pipeline profiling instrumentation.
//!
//! The [`Profiler`] enum provides real-time timing output for each pipeline
//! phase. When disabled ([`Profiler::Noop`]), all methods are no-ops with
//! zero overhead at release optimization levels.

use std::cell::Cell;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Pipeline profiler that prints phase timing to stderr.
///
/// Use [`Profiler::new`] to create an active or noop profiler based on
/// whether `--profile` was passed.
/// Callback for live progress updates (e.g. driving a progress bar).
///
/// Called with a formatted status string whenever the profiler would
/// normally print to stderr. Set via [`Profiler::with_callback`].
type ProgressCallback = Box<dyn Fn(&str) + Send + Sync>;

/// Snapshot of embed-phase progress, passed to the tick callback.
#[derive(Debug, Clone)]
pub struct EmbedProgress {
    /// Chunks completed so far.
    pub done: usize,
    /// Total chunks to embed.
    pub total: usize,
    /// Chunks/s over the most recent reporting window.
    pub window_rate: f64,
    /// Chunks/s since embed phase started.
    pub overall_rate: f64,
    /// Fraction of time spent waiting for the model lock (0.0–1.0).
    pub lock_wait_pct: f64,
    /// Fraction of time spent in inference (0.0–1.0).
    pub inference_pct: f64,
}

/// Callback for per-batch embed progress.
type EmbedTickCallback = Box<dyn Fn(&EmbedProgress) + Send + Sync>;

#[expect(
    clippy::large_enum_variant,
    reason = "Active is the common case; Noop is only for --profile=false"
)]
pub enum Profiler {
    /// Actively collects timing and prints to stderr.
    #[expect(
        private_interfaces,
        reason = "EmbedState is intentionally opaque; callers cannot name it"
    )]
    Active {
        /// Wall-clock start of the entire run.
        start: Instant,
        /// Reporting interval for embed progress.
        interval: Duration,
        /// Mutable embed-phase state (sequential access only, Mutex for Sync).
        embed: Mutex<EmbedState>,
        /// Per-rayon-thread chunk counts (parallel access during chunk phase).
        chunk_counts: Mutex<Vec<usize>>,
        /// Optional callback for live progress updates.
        on_progress: Option<ProgressCallback>,
        /// Optional callback for per-chunk embed progress (drives progress bars).
        on_embed_tick: Option<EmbedTickCallback>,
    },
    /// No-op profiler. All methods are empty.
    Noop,
}

pub(crate) struct EmbedState {
    phase_start: Instant,
    last_report: Instant,
    chunks_at_last_report: usize,
    /// Timestamp of the last `embed_tick` call (for per-batch window rate).
    last_tick: Instant,
    /// Chunks completed at last tick (for per-batch window rate).
    chunks_at_last_tick: usize,
    total_lock_wait: Duration,
    total_inference: Duration,
    total_chunks: usize,
}

impl EmbedState {
    fn new() -> Self {
        let now = Instant::now();
        Self {
            phase_start: now,
            last_report: now,
            chunks_at_last_report: 0,
            last_tick: now,
            chunks_at_last_tick: 0,
            total_lock_wait: Duration::ZERO,
            total_inference: Duration::ZERO,
            total_chunks: 0,
        }
    }
}

impl Profiler {
    /// Create a new profiler. If `enabled` is false, returns `Noop`.
    #[must_use]
    pub fn new(enabled: bool, interval: Duration) -> Self {
        if enabled {
            Self::Active {
                start: Instant::now(),
                interval,
                embed: Mutex::new(EmbedState::new()),
                chunk_counts: Mutex::new(Vec::new()),
                on_progress: None,
                on_embed_tick: None,
            }
        } else {
            Self::Noop
        }
    }

    /// Create an active profiler that drives a progress callback instead of
    /// printing to stderr. The callback receives formatted status messages
    /// at each pipeline phase transition and embed progress tick.
    #[must_use]
    pub fn with_callback(interval: Duration, cb: impl Fn(&str) + Send + Sync + 'static) -> Self {
        Self::Active {
            start: Instant::now(),
            interval,
            embed: Mutex::new(EmbedState::new()),
            chunk_counts: Mutex::new(Vec::new()),
            on_progress: Some(Box::new(cb)),
            on_embed_tick: None,
        }
    }

    /// Set a callback that fires on every embed chunk completion with `(done, total)`.
    ///
    /// Unlike the throttled `on_progress` callback, this fires for every chunk
    /// so progress bars can update smoothly (indicatif handles display rate).
    #[must_use]
    pub fn with_embed_tick(mut self, cb: impl Fn(&EmbedProgress) + Send + Sync + 'static) -> Self {
        if let Self::Active {
            ref mut on_embed_tick,
            ..
        } = self
        {
            *on_embed_tick = Some(Box::new(cb));
        }
        self
    }

    /// Create a no-op profiler.
    #[must_use]
    pub fn noop() -> Self {
        Self::Noop
    }

    /// Send a progress message: calls the callback if set, otherwise prints to stderr.
    fn report(&self, msg: &str) {
        if let Self::Active { on_progress, .. } = self {
            if let Some(cb) = on_progress {
                cb(msg);
            } else {
                eprintln!("{msg}");
            }
        }
    }

    /// Print the system info header line.
    pub fn header(&self, version: &str, model_repo: &str, threads: usize, cores: usize) {
        if let Self::Active { .. } = self {
            self.report(&format!(
                "[profile] ripvec {version} | {cores}-core | rayon: {threads} threads | model: {model_repo}",
            ));
        }
    }

    /// Start timing a named phase. Returns a guard that prints on drop.
    #[must_use]
    pub fn phase(&self, name: &'static str) -> PhaseGuard<'_> {
        if let Self::Active { start, .. } = self {
            self.report(&format!(
                "[{:.3}s] {name}: starting",
                start.elapsed().as_secs_f64(),
            ));
        }
        PhaseGuard {
            profiler: self,
            name,
            start: Instant::now(),
            detail: Cell::new(None),
        }
    }

    /// Record that a rayon thread produced `n` chunks during the chunk phase.
    pub fn chunk_thread_report(&self, n: usize) {
        if let Self::Active { chunk_counts, .. } = self
            && let Ok(mut counts) = chunk_counts.lock()
        {
            let idx = rayon::current_thread_index().unwrap_or(0);
            if counts.len() <= idx {
                counts.resize(idx + 1, 0);
            }
            counts[idx] += n;
        }
    }

    /// Print the chunk phase summary with thread utilization stats.
    pub fn chunk_summary(&self, total_chunks: usize, total_files: usize, elapsed: Duration) {
        if let Self::Active {
            start,
            chunk_counts,
            ..
        } = self
        {
            let wall = start.elapsed();
            if let Ok(counts) = chunk_counts.lock() {
                let active = counts.iter().filter(|&&c| c > 0).count();
                let pool_size = rayon::current_num_threads();
                if active > 0 {
                    let min = counts
                        .iter()
                        .filter(|&&c| c > 0)
                        .min()
                        .copied()
                        .unwrap_or(0);
                    let max = counts.iter().max().copied().unwrap_or(0);
                    self.report(&format!(
                        "[{:.1}s]  chunk: {} chunks from {} files in {:.0?} ({} threads, {} active, skew: {}-{} chunks/thread)",
                        wall.as_secs_f64(),
                        total_chunks,
                        total_files,
                        elapsed,
                        pool_size,
                        active,
                        min,
                        max,
                    ));
                } else {
                    self.report(&format!(
                        "[{:.1}s]  chunk: {} chunks from {} files in {:.0?} ({} threads)",
                        wall.as_secs_f64(),
                        total_chunks,
                        total_files,
                        elapsed,
                        pool_size,
                    ));
                }
            }
        }
    }

    /// Begin the embed phase. Call before the embedding loop.
    pub fn embed_begin(&self, total: usize) {
        if let Self::Active { embed, .. } = self
            && let Ok(mut state) = embed.lock()
        {
            let now = Instant::now();
            state.phase_start = now;
            state.last_report = now;
            state.chunks_at_last_report = 0;
            state.total_lock_wait = Duration::ZERO;
            state.total_inference = Duration::ZERO;
            state.total_chunks = total;
        }
    }

    /// Update the total chunk count for the embed phase.
    ///
    /// Used by the streaming pipeline where the total isn't known at
    /// [`Self::embed_begin`] time. Only updates if the new total is larger
    /// than the current one (monotonic).
    pub fn embed_begin_update_total(&self, total: usize) {
        if let Self::Active { embed, .. } = self
            && let Ok(mut state) = embed.lock()
            && total > state.total_chunks
        {
            state.total_chunks = total;
        }
    }

    /// Called after each chunk is embedded. Prints periodic progress.
    #[expect(
        clippy::cast_precision_loss,
        reason = "display-only rate/percentage calculations; sub-1% precision loss is acceptable"
    )]
    pub fn embed_tick(&self, done: usize) {
        if let Self::Active {
            start,
            interval,
            embed,
            on_embed_tick,
            ..
        } = self
        {
            let Ok(mut state) = embed.lock() else {
                return;
            };
            let now = Instant::now();
            let overall_elapsed = now.duration_since(state.phase_start).as_secs_f64();
            let overall_rate = if overall_elapsed > 0.0 {
                done as f64 / overall_elapsed
            } else {
                0.0
            };
            let total_timing = state.total_lock_wait + state.total_inference;
            let lock_pct = if total_timing.as_nanos() > 0 {
                state.total_lock_wait.as_nanos() as f64 / total_timing.as_nanos() as f64
            } else {
                0.0
            };

            // Per-batch callback: compute a fresh per-batch window rate so the
            // sparkline gets a new data point on every batch, not every interval.
            if let Some(cb) = on_embed_tick {
                let tick_elapsed = now.duration_since(state.last_tick).as_secs_f64();
                let tick_chunks = done - state.chunks_at_last_tick;
                let batch_rate = if tick_elapsed > 0.0 {
                    tick_chunks as f64 / tick_elapsed
                } else {
                    overall_rate
                };
                state.last_tick = now;
                state.chunks_at_last_tick = done;

                cb(&EmbedProgress {
                    done,
                    total: state.total_chunks,
                    window_rate: batch_rate,
                    overall_rate,
                    lock_wait_pct: lock_pct,
                    inference_pct: 1.0 - lock_pct,
                });
            }

            // Throttled text report for --profile / spinner modes.
            if now.duration_since(state.last_report) >= *interval {
                let wall = start.elapsed();
                let report_elapsed = now.duration_since(state.last_report).as_secs_f64();
                let report_chunks = done - state.chunks_at_last_report;
                let window_rate = if report_elapsed > 0.0 {
                    report_chunks as f64 / report_elapsed
                } else {
                    0.0
                };
                self.report(&format!(
                    "[{:.1}s]  embed: {}/{} (last {:.0}s: {:.1}/s, overall: {:.1}/s) lock_wait: {:.0}% inference: {:.0}%",
                    wall.as_secs_f64(),
                    done,
                    state.total_chunks,
                    report_elapsed,
                    window_rate,
                    overall_rate,
                    lock_pct * 100.0,
                    (1.0 - lock_pct) * 100.0,
                ));
                state.last_report = now;
                state.chunks_at_last_report = done;
            }
        }
    }

    /// Byte-based progress for streaming mode.
    ///
    /// Shows `processed_bytes/total_bytes` as MB with chunk rate. The total is
    /// known from the walk phase (file sizes), so the denominator is stable.
    #[expect(
        clippy::cast_precision_loss,
        reason = "display-only: sub-1% precision loss acceptable for MB/rate"
    )]
    pub fn embed_tick_bytes(&self, done_chunks: usize, bytes_processed: u64, total_bytes: u64) {
        if let Self::Active {
            start,
            interval,
            embed,
            ..
        } = self
        {
            let Ok(mut state) = embed.lock() else {
                return;
            };
            let now = Instant::now();
            let overall_elapsed = now.duration_since(state.phase_start).as_secs_f64();
            let overall_rate = if overall_elapsed > 0.0 {
                done_chunks as f64 / overall_elapsed
            } else {
                0.0
            };

            if now.duration_since(state.last_report) >= *interval {
                let wall = start.elapsed();
                let report_elapsed = now.duration_since(state.last_report).as_secs_f64();
                let report_chunks = done_chunks - state.chunks_at_last_report;
                let window_rate = if report_elapsed > 0.0 {
                    report_chunks as f64 / report_elapsed
                } else {
                    0.0
                };
                let mb_done = bytes_processed as f64 / (1024.0 * 1024.0);
                let mb_total = total_bytes as f64 / (1024.0 * 1024.0);
                self.report(&format!(
                    "[{:.1}s]  embed: {:.1}/{:.1} MB (last {:.0}s: {:.1}/s, overall: {:.1}/s)",
                    wall.as_secs_f64(),
                    mb_done,
                    mb_total,
                    report_elapsed,
                    window_rate,
                    overall_rate,
                ));
                state.last_report = now;
                state.chunks_at_last_report = done_chunks;
            }
        }
    }

    /// Accumulate time spent waiting for the model mutex lock.
    pub fn embed_lock_wait(&self, duration: Duration) {
        if let Self::Active { embed, .. } = self
            && let Ok(mut state) = embed.lock()
        {
            state.total_lock_wait += duration;
        }
    }

    /// Accumulate time spent in ONNX inference.
    pub fn embed_inference(&self, duration: Duration) {
        if let Self::Active { embed, .. } = self
            && let Ok(mut state) = embed.lock()
        {
            state.total_inference += duration;
        }
    }

    /// Print the final embed phase summary.
    #[expect(
        clippy::cast_precision_loss,
        reason = "display-only rate/percentage calculations; sub-1% precision loss is acceptable"
    )]
    pub fn embed_done(&self) {
        if let Self::Active { start, embed, .. } = self
            && let Ok(state) = embed.lock()
        {
            let wall = start.elapsed();
            if state.total_chunks == 0 {
                self.report(&format!(
                    "[{:.1}s]  embed: skipped (0 chunks)",
                    wall.as_secs_f64()
                ));
                return;
            }
            let elapsed = Instant::now().duration_since(state.phase_start);
            let rate = if elapsed.as_secs_f64() > 0.0 {
                state.total_chunks as f64 / elapsed.as_secs_f64()
            } else {
                0.0
            };
            let total_timing = state.total_lock_wait + state.total_inference;
            let lock_pct = if total_timing.as_nanos() > 0 {
                state.total_lock_wait.as_nanos() as f64 / total_timing.as_nanos() as f64 * 100.0
            } else {
                0.0
            };
            self.report(&format!(
                "[{:.1}s]  embed: {}/{} done in {:.1}s ({:.1}/s) lock_wait: {:.0}% inference: {:.0}%",
                wall.as_secs_f64(),
                state.total_chunks,
                state.total_chunks,
                elapsed.as_secs_f64(),
                rate,
                lock_pct,
                100.0 - lock_pct,
            ));
        }
    }

    /// Print the total wall-clock time.
    pub fn finish(&self) {
        if let Self::Active { start, .. } = self {
            let elapsed = start.elapsed().as_secs_f64();
            self.report(&format!("[{elapsed:.1}s]  total: {elapsed:.1}s"));
        }
    }
}

/// Guard returned by [`Profiler::phase`]. Prints elapsed time on drop.
pub struct PhaseGuard<'a> {
    profiler: &'a Profiler,
    name: &'static str,
    start: Instant,
    detail: Cell<Option<String>>,
}

impl PhaseGuard<'_> {
    /// Attach a detail string printed alongside the phase timing.
    pub fn set_detail(&self, detail: String) {
        self.detail.set(Some(detail));
    }
}

impl Drop for PhaseGuard<'_> {
    fn drop(&mut self) {
        if let Profiler::Active { start, .. } = self.profiler {
            let elapsed = self.start.elapsed();
            let wall = start.elapsed();
            let msg = if let Some(detail) = self.detail.take() {
                format!(
                    "[{:.3}s] {}: {} in {:.1?}",
                    wall.as_secs_f64(),
                    self.name,
                    detail,
                    elapsed,
                )
            } else {
                format!(
                    "[{:.3}s] {}: {:.1?}",
                    wall.as_secs_f64(),
                    self.name,
                    elapsed,
                )
            };
            self.profiler.report(&msg);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn profiler_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<Profiler>();
    }

    #[test]
    fn noop_profiler_all_methods_are_safe() {
        let p = Profiler::noop();
        p.header("0.1.0", "test/repo", 4, 8);
        {
            let g = p.phase("test_phase");
            g.set_detail("some detail".to_string());
        }
        p.chunk_thread_report(10);
        p.chunk_summary(100, 50, Duration::from_millis(89));
        p.embed_begin(100);
        p.embed_tick(1);
        p.embed_lock_wait(Duration::from_millis(1));
        p.embed_inference(Duration::from_millis(10));
        p.embed_done();
        p.finish();
    }

    #[test]
    fn active_profiler_phase_guard_formats_correctly() {
        let p = Profiler::new(true, Duration::from_secs(10));
        {
            let g = p.phase("test_phase");
            std::thread::sleep(Duration::from_millis(5));
            g.set_detail("42 items".to_string());
        }
        // Just verify it doesn't panic — stderr output is visual
    }

    #[test]
    fn embed_tick_respects_interval() {
        let p = Profiler::new(true, Duration::from_secs(100)); // very long interval
        p.embed_begin(10);
        for i in 1..=10 {
            p.embed_tick(i);
            // With a 100s interval, no periodic output should fire
        }
        p.embed_done();
    }

    #[test]
    fn chunk_summary_with_zero_files() {
        let p = Profiler::new(true, Duration::from_secs(10));
        p.chunk_summary(0, 0, Duration::from_millis(0));
        // Should not panic or divide by zero
    }

    #[test]
    fn embed_done_with_zero_chunks() {
        let p = Profiler::new(true, Duration::from_secs(10));
        p.embed_begin(0);
        p.embed_done();
        // Should print "skipped" line, not divide by zero
    }
}
