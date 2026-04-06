//! Interactive TUI for real-time semantic search.
//!
//! Embeds the codebase once, then drops into a three-pane layout where
//! the user types a query and sees ranked results update live. Uses
//! [`ratatui`] with a [`crossterm`] backend.

pub mod highlight;
pub mod input;
pub mod ui;

use std::io;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use crossterm::event::{self, Event};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::prelude::CrosstermBackend;

use ripvec_core::backend::EmbedBackend;

/// TUI application state.
pub struct App {
    /// Current query string.
    pub query: String,
    /// Index of selected result in the list.
    pub selected: usize,
    /// Scroll offset for the preview pane.
    pub preview_scroll: u16,
    /// Pre-computed hybrid index (semantic + BM25) for re-ranking.
    pub index: ripvec_core::hybrid::HybridIndex,
    /// Current ranked results: `(chunk_index, similarity_score)`.
    pub results: Vec<(usize, f32)>,
    /// Embedding backends (primary first, used for query re-embedding).
    pub backends: Vec<Box<dyn EmbedBackend>>,
    /// Tokenizer for query encoding.
    pub tokenizer: tokenizers::Tokenizer,
    /// Minimum similarity threshold.
    pub threshold: f32,
    /// Duration of the last ranking pass (for status display).
    pub rank_time_ms: f64,
    /// Syntax highlighter for the preview pane.
    pub highlighter: highlight::Highlighter,
    /// Set to `true` to exit the event loop.
    pub should_quit: bool,
    /// If set, suspend TUI and open this `(file_path, line_number)` in `$EDITOR`.
    pub open_editor: Option<(String, usize)>,
    /// Summary of indexed files by extension (e.g. "208 chunks │ 15 rs, 8 py, 6 js").
    pub index_summary: String,
    /// Receiver for file watcher events (only active with `--index -i`).
    pub watcher_rx: Option<std::sync::mpsc::Receiver<()>>,
    /// Keep the watcher alive (dropped when App is dropped).
    pub watcher_handle: Option<notify::RecommendedWatcher>,
    /// Transient status message (e.g. "↻ 3 files updated") with display time.
    pub status_flash: Option<(String, std::time::Instant)>,
    /// Cache config for incremental reindex on watcher events.
    pub cache_config: Option<CacheConfig>,
    /// When set, the next draw will clear the entire terminal first.
    ///
    /// This prevents visual artifacts when the preview content changes
    /// drastically (e.g. navigating results quickly). Set by navigation
    /// keys and Ctrl-L.
    pub force_redraw: bool,
}

/// Configuration for the persistent cache (used by the TUI file watcher).
pub struct CacheConfig {
    /// Project root directory.
    pub root: std::path::PathBuf,
    /// Model repository name.
    pub model_repo: String,
    /// Optional cache directory override.
    pub cache_dir: Option<std::path::PathBuf>,
}

impl App {
    /// Re-rank all chunks against the current query.
    ///
    /// If the query is empty, clears results. Otherwise tokenizes + embeds
    /// the query and computes dot-product similarity against every chunk.
    pub fn rerank(&mut self) {
        if self.query.is_empty() {
            self.results.clear();
            self.rank_time_ms = 0.0;
            self.selected = 0;
            self.preview_scroll = 0;
            return;
        }

        let start = Instant::now();

        // Tokenize + embed the query
        let model_max = self.backends[0].max_tokens();
        let Ok(enc) =
            ripvec_core::tokenize::tokenize_query(&self.query, &self.tokenizer, model_max)
        else {
            self.results.clear();
            self.rank_time_ms = 0.0;
            return;
        };
        let query_emb = match self.backends[0].embed_batch(&[enc]) {
            Ok(mut vecs) => match vecs.pop() {
                Some(v) => v,
                None => return,
            },
            Err(_) => return,
        };

        // BLAS matrix-vector multiply for ranking
        self.results = self.index.search(
            &query_emb,
            &self.query,
            200,
            self.threshold,
            ripvec_core::hybrid::SearchMode::Hybrid,
        );

        self.rank_time_ms = start.elapsed().as_secs_f64() * 1000.0;
        self.selected = 0;
        self.preview_scroll = 0;
    }
}

/// Run the interactive TUI event loop.
///
/// Takes ownership of the [`App`] state and blocks until the user quits
/// (Esc or Ctrl-C). Restores the terminal on exit, even on error.
///
/// # Errors
///
/// Returns an error if terminal setup or rendering fails.
pub fn run(mut app: App) -> Result<()> {
    // Set up terminal
    enable_raw_mode().context("failed to enable raw mode")?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen).context("failed to enter alternate screen")?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).context("failed to create terminal")?;

    let result = event_loop(&mut terminal, &mut app);

    // Restore terminal — always runs, even on error
    disable_raw_mode().ok();
    execute!(terminal.backend_mut(), LeaveAlternateScreen).ok();
    terminal.show_cursor().ok();

    result
}

/// Core event loop: poll for input, handle keys, redraw.
fn event_loop(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>, app: &mut App) -> Result<()> {
    loop {
        // When force_redraw is set, clear the back-buffer so ratatui
        // repaints every cell instead of diffing against stale content.
        // This prevents visual artifacts when the preview changes rapidly.
        if app.force_redraw {
            terminal.clear().context("failed to clear terminal")?;
            app.force_redraw = false;
        }

        terminal
            .draw(|frame| ui::draw(frame, app))
            .context("failed to draw frame")?;

        if app.should_quit {
            return Ok(());
        }

        // Poll at 50 ms for responsive input.
        // Non-key events (resize, mouse, etc.) are consumed and ignored;
        // ratatui re-queries terminal size on every draw.
        if event::poll(Duration::from_millis(50)).context("event poll failed")?
            && let Event::Key(key) = event::read().context("event read failed")?
        {
            let needs_rerank = input::handle_key(app, key);
            if needs_rerank {
                app.rerank();
            }
        }

        // Check for file watcher events (non-blocking)
        if let Some(ref rx) = app.watcher_rx
            && rx.try_recv().is_ok()
        {
            // Drain additional events
            while rx.try_recv().is_ok() {}

            // Re-index incrementally if cache config is set
            if let Some(ref cfg) = app.cache_config {
                let backend_refs: Vec<&dyn EmbedBackend> =
                    app.backends.iter().map(|b| &**b).collect();
                let search_cfg = ripvec_core::embed::SearchConfig::default();
                let profiler = ripvec_core::profile::Profiler::noop();

                match ripvec_core::cache::reindex::incremental_index(
                    &cfg.root,
                    &backend_refs,
                    &app.tokenizer,
                    &search_cfg,
                    &profiler,
                    &cfg.model_repo,
                    cfg.cache_dir.as_deref(),
                    false,
                ) {
                    Ok((new_index, stats)) => {
                        let msg = format!(
                            "\u{21bb} {} files updated ({} chunks re-embedded)",
                            stats.files_changed, stats.chunks_reembedded,
                        );
                        app.index = new_index;
                        app.status_flash = Some((msg, std::time::Instant::now()));
                        app.rerank();
                    }
                    Err(e) => {
                        app.status_flash =
                            Some((format!("reindex error: {e}"), std::time::Instant::now()));
                    }
                }
            }
        }

        // Clear status flash after 3 seconds
        if let Some((_, when)) = &app.status_flash
            && when.elapsed() > Duration::from_secs(3)
        {
            app.status_flash = None;
        }

        // If Enter was pressed on a result, suspend TUI, open editor, resume.
        if let Some((file, line)) = app.open_editor.take() {
            execute!(terminal.backend_mut(), LeaveAlternateScreen)
                .context("failed to leave alternate screen")?;
            disable_raw_mode().context("failed to disable raw mode")?;

            let editor = std::env::var("EDITOR").unwrap_or_else(|_| "vi".to_string());
            std::process::Command::new(&editor)
                .arg(format!("+{line}"))
                .arg(&file)
                .status()
                .with_context(|| format!("failed to launch editor '{editor}'"))?;

            enable_raw_mode().context("failed to re-enable raw mode")?;
            execute!(terminal.backend_mut(), EnterAlternateScreen)
                .context("failed to re-enter alternate screen")?;
            terminal.clear().context("failed to clear terminal")?;
        }
    }
}
