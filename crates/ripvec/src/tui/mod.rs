//! Interactive TUI for real-time semantic search.
//!
//! Embeds the codebase once, then drops into a three-pane layout where
//! the user types a query and sees ranked results update live. Uses
//! [`ratatui`] with a [`crossterm`] backend.

pub mod highlight;
pub mod index;
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

use ripvec_core::backend::{EmbedBackend, Encoding};

use index::SearchIndex;

/// TUI application state.
pub struct App {
    /// Current query string.
    pub query: String,
    /// Index of selected result in the list.
    pub selected: usize,
    /// Scroll offset for the preview pane.
    pub preview_scroll: u16,
    /// Pre-computed search index for BLAS-accelerated re-ranking.
    pub index: SearchIndex,
    /// Current ranked results: `(chunk_index, similarity_score)`.
    pub results: Vec<(usize, f32)>,
    /// Embedding backend for re-embedding queries.
    pub backend: Box<dyn EmbedBackend>,
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
        let Ok(enc) = tokenize_query(&self.query, &self.tokenizer) else {
            self.results.clear();
            self.rank_time_ms = 0.0;
            return;
        };
        let query_emb = match self.backend.embed_batch(&[enc]) {
            Ok(mut vecs) => match vecs.pop() {
                Some(v) => v,
                None => return,
            },
            Err(_) => return,
        };

        // BLAS matrix-vector multiply for ranking
        self.results = self.index.rank(&query_emb, self.threshold);

        self.rank_time_ms = start.elapsed().as_secs_f64() * 1000.0;
        self.selected = 0;
        self.preview_scroll = 0;
    }
}

/// Hard limit matching bge-small-en-v1.5 `max_position_embeddings`.
const MODEL_MAX_TOKENS: usize = 512;

/// Tokenize a query string into an [`Encoding`] for inference.
fn tokenize_query(text: &str, tokenizer: &tokenizers::Tokenizer) -> Result<Encoding> {
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;

    let len = encoding.get_ids().len().min(MODEL_MAX_TOKENS);
    Ok(Encoding {
        input_ids: encoding.get_ids()[..len]
            .iter()
            .map(|&x| i64::from(x))
            .collect(),
        attention_mask: encoding.get_attention_mask()[..len]
            .iter()
            .map(|&x| i64::from(x))
            .collect(),
        token_type_ids: encoding.get_type_ids()[..len]
            .iter()
            .map(|&x| i64::from(x))
            .collect(),
    })
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
        terminal
            .draw(|frame| ui::draw(frame, app))
            .context("failed to draw frame")?;

        if app.should_quit {
            return Ok(());
        }

        // Poll at 50 ms for responsive input
        if event::poll(Duration::from_millis(50)).context("event poll failed")?
            && let Event::Key(key) = event::read().context("event read failed")?
        {
            let needs_rerank = input::handle_key(app, key);
            if needs_rerank {
                app.rerank();
            }
        }
    }
}
