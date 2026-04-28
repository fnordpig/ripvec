//! Full-screen indexing dashboard.
//!
//! This mode is used by `ripvec --index` when no query is supplied. It runs
//! the indexing pipeline in a worker thread and renders live progress, cached
//! tracing events, and profiler status in the alternate screen.

use std::collections::VecDeque;
use std::io;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::layout::{Constraint, Layout, Margin, Rect};
use ratatui::prelude::CrosstermBackend;
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::symbols;
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{
    BarChart, Block, Borders, Gauge, LineGauge, List, ListItem, Paragraph, Scrollbar,
    ScrollbarOrientation, ScrollbarState, Sparkline, Wrap,
};
use ratatui::{Frame, Terminal};
use ripvec_core::backend::EmbedBackend;
use ripvec_core::cache::reindex::ReindexStats;
use ripvec_core::hybrid::HybridIndex;
use ripvec_core::profile::EmbedProgress;

const MAX_ATLAS_POINTS: usize = 20_000;
const ATLAS_MARGIN_RATIO: f64 = 0.08;
const ATLAS_MIN_SPAN: f64 = 0.02;
const ATLAS_LOWER_PERCENTILE: f64 = 0.02;
const ATLAS_UPPER_PERCENTILE: f64 = 0.98;

/// Verbosity level used by the dashboard log pane.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    /// Error events.
    Error,
    /// Warning events.
    Warn,
    /// Informational events.
    Info,
    /// Debug events.
    Debug,
    /// Trace events.
    Trace,
}

impl LogLevel {
    /// Cycle to the next visible verbosity.
    #[must_use]
    pub const fn next(self) -> Self {
        match self {
            Self::Warn => Self::Info,
            Self::Info => Self::Debug,
            Self::Debug => Self::Trace,
            Self::Trace | Self::Error => Self::Warn,
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::Error => "error",
            Self::Warn => "warn",
            Self::Info => "info",
            Self::Debug => "debug",
            Self::Trace => "trace",
        }
    }

    const fn color(self) -> Color {
        match self {
            Self::Error => Color::LightRed,
            Self::Warn => Color::LightYellow,
            Self::Info => Color::LightCyan,
            Self::Debug => Color::LightGreen,
            Self::Trace => Color::Gray,
        }
    }

    const fn from_tracing(level: tracing::Level) -> Self {
        match level {
            tracing::Level::ERROR => Self::Error,
            tracing::Level::WARN => Self::Warn,
            tracing::Level::INFO => Self::Info,
            tracing::Level::DEBUG => Self::Debug,
            tracing::Level::TRACE => Self::Trace,
        }
    }
}

/// One log line retained by the dashboard.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogLine {
    /// Event level.
    pub level: LogLevel,
    /// Renderable message.
    pub message: String,
}

/// Deterministic 2D projection of one high-dimensional embedding.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EmbeddingPoint {
    /// Projected X coordinate in `[0, 1]`.
    pub x: f64,
    /// Projected Y coordinate in `[0, 1]`.
    pub y: f64,
    /// Projected third axis in `[0, 1]`, used as color temperature.
    pub z: f64,
}

impl LogLine {
    /// Create a log line.
    #[must_use]
    pub fn new(level: LogLevel, message: impl Into<String>) -> Self {
        Self {
            level,
            message: message.into(),
        }
    }
}

/// Thread-safe bounded log buffer.
#[derive(Clone)]
pub struct LogBuffer {
    inner: Arc<Mutex<VecDeque<LogLine>>>,
    capacity: usize,
}

impl LogBuffer {
    /// Create a bounded log buffer.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(VecDeque::with_capacity(capacity))),
            capacity,
        }
    }

    /// Push a line, evicting the oldest line when full.
    pub fn push(&self, line: LogLine) {
        if let Ok(mut lines) = self.inner.lock() {
            if lines.len() >= self.capacity {
                lines.pop_front();
            }
            lines.push_back(line);
        }
    }

    /// Snapshot buffered lines without consuming them.
    #[must_use]
    pub fn snapshot(&self) -> Vec<LogLine> {
        self.inner
            .lock()
            .map(|lines| lines.iter().cloned().collect())
            .unwrap_or_default()
    }
}

/// Tracing layer that records events into the dashboard log buffer.
pub struct BufferLayer {
    buffer: LogBuffer,
}

impl BufferLayer {
    /// Create a tracing layer backed by a dashboard log buffer.
    #[must_use]
    pub const fn new(buffer: LogBuffer) -> Self {
        Self { buffer }
    }
}

struct MessageVisitor<'a>(&'a mut String);

impl tracing::field::Visit for MessageVisitor<'_> {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        use std::fmt::Write as _;
        if field.name() == "message" {
            let _ = write!(self.0, "{value:?}");
        } else {
            let _ = write!(self.0, " {field}={value:?}");
        }
    }

    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        use std::fmt::Write as _;
        if field.name() == "message" {
            let _ = write!(self.0, "{value}");
        } else {
            let _ = write!(self.0, " {field}={value}");
        }
    }
}

impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for BufferLayer {
    fn on_event(
        &self,
        event: &tracing::Event<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        let meta = event.metadata();
        let mut message = String::new();
        event.record(&mut MessageVisitor(&mut message));
        if message.is_empty() {
            message = meta.target().to_string();
        }
        self.buffer
            .push(LogLine::new(LogLevel::from_tracing(*meta.level()), message));
    }
}

/// Final indexing output, ready to hand to the search TUI.
pub struct IndexBuild {
    /// Built hybrid index.
    pub index: HybridIndex,
    /// Reindex statistics.
    pub stats: ReindexStats,
    /// Loaded embedding backends.
    pub backends: Vec<Box<dyn EmbedBackend>>,
    /// Loaded tokenizer.
    pub tokenizer: tokenizers::Tokenizer,
    /// Summary shown by the search TUI.
    pub index_summary: String,
    /// Model repo used for cache compatibility and file watching.
    pub model_repo: String,
}

/// Dashboard event sent from the indexing worker.
pub enum DashboardEvent {
    /// Current high-level phase.
    Phase(String),
    /// Per-batch embedding progress.
    EmbedTick(EmbedProgress),
    /// Newly embedded vectors projected into atlas space.
    Embeddings(Vec<EmbeddingPoint>),
    /// Indexing finished or failed.
    Finished(Box<std::result::Result<IndexBuild, String>>),
}

/// Thread-safe sink used by worker callbacks.
#[derive(Clone)]
pub struct EventSink {
    sender: Sender<DashboardEvent>,
    logs: LogBuffer,
}

impl EventSink {
    /// Create a dashboard event channel.
    #[must_use]
    pub fn channel(logs: LogBuffer) -> (Self, Receiver<DashboardEvent>) {
        let (sender, receiver) = mpsc::channel();
        (Self { sender, logs }, receiver)
    }

    /// Send a high-level phase and mirror it into the log pane.
    pub fn phase(&self, phase: impl Into<String>) {
        let phase = phase.into();
        self.logs
            .push(LogLine::new(LogLevel::Info, format!("phase: {phase}")));
        let _ = self.sender.send(DashboardEvent::Phase(phase));
    }

    /// Record profiler text and infer phase transitions when possible.
    pub fn profiler_message(&self, message: &str) {
        self.logs.push(LogLine::new(LogLevel::Info, message));
        if let Some(phase) = phase_from_profile_message(message) {
            let _ = self.sender.send(DashboardEvent::Phase(phase));
        }
    }

    /// Send embedding progress.
    pub fn embed_tick(&self, progress: &EmbedProgress) {
        let _ = self
            .sender
            .send(DashboardEvent::EmbedTick(progress.clone()));
    }

    /// Project and send completed embedding vectors.
    pub fn embeddings(&self, embeddings: &[Vec<f32>]) {
        let points: Vec<_> = embeddings
            .iter()
            .filter(|embedding| !embedding.is_empty())
            .map(|embedding| project_embedding(embedding))
            .collect();
        if !points.is_empty() {
            let _ = self.sender.send(DashboardEvent::Embeddings(points));
        }
    }

    /// Send the terminal indexing result.
    pub fn finish(&self, result: std::result::Result<IndexBuild, String>) {
        let _ = self.sender.send(DashboardEvent::Finished(Box::new(result)));
    }
}

/// Static dashboard configuration.
pub struct DashboardConfig {
    /// Indexed root path.
    pub root: String,
    /// Model repo.
    pub model_repo: String,
    /// Whether Enter should continue into search when indexing completes.
    pub search_after: bool,
}

/// Outcome selected by the user.
pub enum DashboardOutcome {
    /// User quit before indexing completed.
    Quit,
    /// Indexing completed and user left the dashboard.
    Done(IndexBuild),
    /// Indexing completed and user requested search.
    Search(IndexBuild),
}

struct App {
    config: DashboardConfig,
    phase: String,
    started: Instant,
    embed: Option<EmbedProgress>,
    rates: Vec<u64>,
    embedding_points: Vec<EmbeddingPoint>,
    log_level: LogLevel,
    log_scroll: usize,
    log_focus: bool,
    done: Option<IndexBuild>,
    error: Option<String>,
    should_quit: bool,
    search_requested: bool,
    frame: u64,
}

impl App {
    fn new(config: DashboardConfig) -> Self {
        Self {
            config,
            phase: "Starting".to_string(),
            started: Instant::now(),
            embed: None,
            rates: Vec::new(),
            embedding_points: Vec::new(),
            log_level: LogLevel::Info,
            log_scroll: 0,
            log_focus: false,
            done: None,
            error: None,
            should_quit: false,
            search_requested: false,
            frame: 0,
        }
    }

    fn handle_event(&mut self, event: DashboardEvent) {
        match event {
            DashboardEvent::Phase(phase) => self.phase = phase,
            DashboardEvent::EmbedTick(progress) => {
                let rate = rate_to_u64(progress.window_rate);
                self.rates.push(rate);
                if self.rates.len() > 160 {
                    self.rates.remove(0);
                }
                self.embed = Some(progress);
            }
            DashboardEvent::Embeddings(mut points) => {
                self.embedding_points.append(&mut points);
                let excess = self.embedding_points.len().saturating_sub(MAX_ATLAS_POINTS);
                if excess > 0 {
                    self.embedding_points.drain(0..excess);
                }
            }
            DashboardEvent::Finished(result) => match *result {
                Ok(build) => {
                    self.phase = "Index ready".to_string();
                    self.done = Some(build);
                }
                Err(error) => {
                    self.phase = "Index failed".to_string();
                    self.error = Some(error);
                }
            },
        }
    }

    fn handle_key(&mut self, key: KeyEvent) {
        if !key.is_press() {
            return;
        }
        match key.code {
            KeyCode::Esc | KeyCode::Char('q') => self.should_quit = true,
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.should_quit = true;
            }
            KeyCode::Char('v') => self.log_level = self.log_level.next(),
            KeyCode::Char('l') => self.log_focus = !self.log_focus,
            KeyCode::PageUp | KeyCode::Char('k') if self.log_focus => {
                self.log_scroll = self.log_scroll.saturating_add(5);
            }
            KeyCode::PageDown | KeyCode::Char('j') if self.log_focus => {
                self.log_scroll = self.log_scroll.saturating_sub(5);
            }
            KeyCode::Enter if self.config.search_after && self.done.is_some() => {
                self.search_requested = true;
                self.should_quit = true;
            }
            _ => {}
        }
    }

    fn elapsed(&self) -> Duration {
        self.started.elapsed()
    }

    fn progress_ratio(&self) -> f64 {
        self.embed.as_ref().map_or(0.0, |p| {
            if p.total == 0 {
                0.0
            } else {
                (p.done as f64 / p.total as f64).clamp(0.0, 1.0)
            }
        })
    }
}

/// Run the dashboard until the user quits or requests search.
///
/// # Errors
///
/// Returns an error if terminal setup, event polling, or rendering fails.
pub fn run(
    receiver: &Receiver<DashboardEvent>,
    logs: &LogBuffer,
    config: DashboardConfig,
) -> Result<DashboardOutcome> {
    enable_raw_mode().context("failed to enable raw mode")?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen).context("failed to enter alternate screen")?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).context("failed to create terminal")?;

    let mut app = App::new(config);
    let result = event_loop(&mut terminal, &mut app, receiver, logs);

    disable_raw_mode().ok();
    execute!(terminal.backend_mut(), LeaveAlternateScreen).ok();
    terminal.show_cursor().ok();

    result.map(|()| {
        if app.search_requested {
            DashboardOutcome::Search(app.done.expect("search requires completed index"))
        } else if let Some(build) = app.done {
            DashboardOutcome::Done(build)
        } else {
            DashboardOutcome::Quit
        }
    })
}

fn event_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut App,
    receiver: &Receiver<DashboardEvent>,
    logs: &LogBuffer,
) -> Result<()> {
    while !app.should_quit {
        while let Ok(event) = receiver.try_recv() {
            app.handle_event(event);
        }

        app.frame = app.frame.wrapping_add(1);
        terminal
            .draw(|frame| draw(frame, app, logs))
            .context("failed to draw dashboard")?;

        if event::poll(Duration::from_millis(50)).context("event poll failed")?
            && let Event::Key(key) = event::read().context("event read failed")?
        {
            app.handle_key(key);
        }
    }
    Ok(())
}

fn draw(frame: &mut Frame, app: &App, logs: &LogBuffer) {
    let area = frame.area();
    let log_height = (area.height / 3).clamp(7, 16);
    let chunks = Layout::vertical([
        Constraint::Length(3),
        Constraint::Length(6),
        Constraint::Min(8),
        Constraint::Length(log_height),
        Constraint::Length(1),
    ])
    .split(area);

    draw_header(frame, app, chunks[0]);
    draw_metrics(frame, app, chunks[1]);
    draw_body(frame, app, chunks[2]);
    draw_logs(frame, app, logs, chunks[3]);
    draw_footer(frame, app, chunks[4]);
}

fn draw_header(frame: &mut Frame, app: &App, area: Rect) {
    let status = if let Some(error) = &app.error {
        format!("failed: {error}")
    } else if app.done.is_some() {
        "complete".to_string()
    } else {
        app.phase.clone()
    };
    let title = Line::from(vec![
        Span::styled(
            " ripvec index atlas ",
            Style::default()
                .fg(Color::Black)
                .bg(Color::LightCyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("  "),
        Span::styled(&status, Style::default().fg(Color::LightYellow)),
        Span::raw(format!("  {:.1}s", app.elapsed().as_secs_f64())),
    ]);
    let meta = Line::from(vec![
        Span::styled("root ", Style::default().fg(Color::DarkGray)),
        Span::raw(&app.config.root),
        Span::styled("  model ", Style::default().fg(Color::DarkGray)),
        Span::raw(&app.config.model_repo),
    ]);
    frame.render_widget(
        Paragraph::new(vec![title, meta]).block(Block::bordered()),
        area,
    );
}

fn draw_metrics(frame: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::horizontal([
        Constraint::Percentage(36),
        Constraint::Percentage(32),
        Constraint::Percentage(32),
    ])
    .split(area);

    let ratio = app.progress_ratio();
    let label = app.embed.as_ref().map_or_else(
        || app.phase.clone(),
        |p| format!("{}/{} chunks", p.done, p.total),
    );
    let gauge = Gauge::default()
        .block(Block::bordered().title("embedding"))
        .gauge_style(
            Style::default()
                .fg(Color::LightCyan)
                .bg(Color::Black)
                .add_modifier(Modifier::BOLD),
        )
        .use_unicode(true)
        .label(label)
        .ratio(ratio);
    frame.render_widget(gauge, chunks[0]);

    let line = LineGauge::default()
        .block(Block::bordered().title("semantic saturation"))
        .filled_style(Style::default().fg(Color::LightMagenta))
        .filled_symbol(symbols::line::THICK_HORIZONTAL)
        .unfilled_symbol(symbols::line::HORIZONTAL)
        .ratio(ratio);
    frame.render_widget(line, chunks[1]);

    let latest_rate = app.rates.last().copied().unwrap_or_default();
    let spark = Sparkline::default()
        .block(Block::bordered().title(format!("chunk velocity {latest_rate}/s")))
        .style(Style::default().fg(Color::LightGreen))
        .data(&app.rates)
        .bar_set(symbols::bar::NINE_LEVELS);
    frame.render_widget(spark, chunks[2]);
}

fn draw_body(frame: &mut Frame, app: &App, area: Rect) {
    let panes =
        Layout::horizontal([Constraint::Percentage(38), Constraint::Percentage(62)]).split(area);
    draw_stats(frame, app, panes[0]);
    draw_field(frame, app, panes[1]);
}

fn draw_stats(frame: &mut Frame, app: &App, area: Rect) {
    let Some(build) = app.done.as_ref() else {
        let text = vec![
            Line::from("waiting for cache topology...".light_cyan()),
            Line::from(""),
            Line::from(
                "The dashboard is live before model load, cache diff, chunking, embedding, and index finalization.",
            ),
        ];
        frame.render_widget(
            Paragraph::new(text)
                .wrap(Wrap { trim: true })
                .block(Block::bordered().title("corpus")),
            area,
        );
        return;
    };

    let stats = &build.stats;
    let data = [
        ("dirty", stats.files_changed as u64),
        ("cached", stats.files_unchanged as u64),
        ("gone", stats.files_deleted as u64),
        ("chunks", stats.chunks_reembedded as u64),
    ];
    let chart = BarChart::default()
        .block(Block::bordered().title("cache delta"))
        .data(&data)
        .bar_width(7)
        .bar_gap(1)
        .bar_set(symbols::bar::NINE_LEVELS)
        .value_style(Style::default().fg(Color::Black).bg(Color::LightGreen))
        .bar_style(Style::default().fg(Color::LightCyan))
        .label_style(Style::default().fg(Color::Gray));
    frame.render_widget(chart, area);
}

fn draw_field(frame: &mut Frame, app: &App, area: Rect) {
    let title = format!(
        "embedding atlas {} projected chunks | auto-fit",
        app.embedding_points.len()
    );
    let heatmap_area = area.inner(Margin {
        vertical: 1,
        horizontal: 1,
    });
    let text = if app.embedding_points.is_empty() {
        vec![
            Line::from("waiting for first embedding batch...".light_cyan()),
            Line::from(""),
            Line::from("x/y: deterministic random projection"),
            Line::from("x/y view: robust auto-fit as the cloud grows"),
            Line::from("color: third projected axis, Viridis-style"),
            Line::from("brightness: chunk density per cell"),
        ]
    } else {
        heatmap_lines(
            &app.embedding_points,
            heatmap_area.width,
            heatmap_area.height,
        )
    };
    frame.render_widget(
        Paragraph::new(text).block(Block::bordered().title(title)),
        area,
    );
}

fn draw_logs(frame: &mut Frame, app: &App, logs: &LogBuffer, area: Rect) {
    let all = logs.snapshot();
    let visible = visible_log_lines(&all, app.log_level);
    let height = area.height.saturating_sub(2) as usize;
    let max_scroll = visible.len().saturating_sub(height);
    let scroll = app.log_scroll.min(max_scroll);
    let start = visible.len().saturating_sub(height + scroll);
    let end = visible.len().saturating_sub(scroll);

    let items: Vec<ListItem> = visible[start..end]
        .iter()
        .map(|line| {
            let content = Line::from(vec![
                Span::styled(
                    format!("{:<5} ", line.level.label()),
                    Style::default().fg(line.level.color()),
                ),
                Span::raw(&line.message),
            ]);
            ListItem::new(content)
        })
        .collect();

    let title = if app.log_focus {
        format!("logs <= {} [focused]", app.log_level.label())
    } else {
        format!("logs <= {}", app.log_level.label())
    };
    let list = List::new(items).block(
        Block::default()
            .title(title)
            .borders(Borders::ALL)
            .border_style(if app.log_focus {
                Style::default().fg(Color::LightYellow)
            } else {
                Style::default().fg(Color::DarkGray)
            }),
    );
    frame.render_widget(list, area);

    let mut state = ScrollbarState::new(visible.len()).position(scroll);
    frame.render_stateful_widget(
        Scrollbar::new(ScrollbarOrientation::VerticalRight)
            .begin_symbol(Some("↑"))
            .end_symbol(Some("↓")),
        area.inner(Margin {
            vertical: 1,
            horizontal: 0,
        }),
        &mut state,
    );
}

fn draw_footer(frame: &mut Frame, app: &App, area: Rect) {
    let action = if app.done.is_some() && app.config.search_after {
        "ENTER search │ "
    } else if app.done.is_some() {
        "complete │ "
    } else {
        ""
    };
    let hints = format!(
        "{action}v verbosity │ l logs │ q quit │ level {}",
        app.log_level.label()
    );
    frame.render_widget(
        Paragraph::new(Text::from(hints))
            .style(Style::default().bg(Color::DarkGray).fg(Color::White)),
        area,
    );
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "display-only sparkline rates are clamped before integer conversion"
)]
fn rate_to_u64(rate: f64) -> u64 {
    let rounded = rate.max(0.0).round();
    if rounded >= u64::MAX as f64 {
        u64::MAX
    } else {
        rounded as u64
    }
}

/// Project a high-dimensional embedding into a stable `[0, 1]` atlas point.
#[must_use]
pub fn project_embedding(embedding: &[f32]) -> EmbeddingPoint {
    if embedding.is_empty() {
        return EmbeddingPoint {
            x: 0.5,
            y: 0.5,
            z: 0.5,
        };
    }
    let x = project_axis(embedding, 0x9e37_79b9_7f4a_7c15);
    let y = project_axis(embedding, 0xbf58_476d_1ce4_e5b9);
    let z = project_axis(embedding, 0x94d0_49bb_1331_11eb);
    EmbeddingPoint { x, y, z }
}

fn project_axis(embedding: &[f32], seed: u64) -> f64 {
    let scale = (3.0 / embedding.len() as f64).sqrt();
    let dot = embedding
        .iter()
        .enumerate()
        .map(|(i, value)| f64::from(*value) * random_axis_weight(i, seed) * scale)
        .sum::<f64>();
    ((dot * 1.4).tanh() + 1.0) * 0.5
}

fn random_axis_weight(index: usize, seed: u64) -> f64 {
    let mut x = seed ^ index as u64;
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^= x >> 31;
    let unit = (x as f64) / (u64::MAX as f64);
    unit.mul_add(2.0, -1.0)
}

fn heatmap_lines(points: &[EmbeddingPoint], width: u16, height: u16) -> Vec<Line<'static>> {
    let width = usize::from(width).max(1);
    let height = usize::from(height).max(1);
    let mut cells = vec![HeatCell::default(); width * height];
    let bounds = AtlasBounds::from_points(points);

    for point in points {
        let x = coordinate_to_index(bounds.normalize_x(point.x), width);
        // Invert y so larger projected values appear toward the top.
        let y = height - 1 - coordinate_to_index(bounds.normalize_y(point.y), height);
        let cell = &mut cells[y * width + x];
        cell.count = cell.count.saturating_add(1);
        cell.z_sum += point.z;
    }

    (0..height)
        .map(|y| {
            let spans: Vec<_> = (0..width)
                .map(|x| {
                    let cell = &cells[y * width + x];
                    match heatmap_cell_char(cell.count) {
                        Some(symbol) => Span::styled(
                            symbol.to_string(),
                            Style::default().fg(heatmap_color(cell.average_z())),
                        ),
                        None => Span::raw(" "),
                    }
                })
                .collect();
            Line::from(spans)
        })
        .collect()
}

#[derive(Clone, Copy, Debug)]
struct AtlasBounds {
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
}

impl AtlasBounds {
    fn from_points(points: &[EmbeddingPoint]) -> Self {
        let xs = points.iter().map(|point| point.x).collect();
        let ys = points.iter().map(|point| point.y).collect();
        let (x_min, x_max) = axis_bounds(xs);
        let (y_min, y_max) = axis_bounds(ys);
        Self {
            x_min,
            x_max,
            y_min,
            y_max,
        }
    }

    fn normalize_x(self, value: f64) -> f64 {
        normalize_axis(value, self.x_min, self.x_max)
    }

    fn normalize_y(self, value: f64) -> f64 {
        normalize_axis(value, self.y_min, self.y_max)
    }
}

fn axis_bounds(mut values: Vec<f64>) -> (f64, f64) {
    values.retain(|value| value.is_finite());
    if values.is_empty() {
        return (0.0, 1.0);
    }
    values.sort_by(f64::total_cmp);
    let lower = percentile(&values, ATLAS_LOWER_PERCENTILE);
    let upper = percentile(&values, ATLAS_UPPER_PERCENTILE);
    expand_axis_bounds(lower, upper)
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "percentile positions are clamped to valid vector indices before conversion"
)]
fn percentile(sorted: &[f64], percentile: f64) -> f64 {
    let max_index = sorted.len().saturating_sub(1);
    let position = percentile.clamp(0.0, 1.0) * max_index as f64;
    let index = position.round() as usize;
    sorted[index.min(max_index)]
}

fn expand_axis_bounds(lower: f64, upper: f64) -> (f64, f64) {
    let center = (lower + upper) * 0.5;
    let observed_span = (upper - lower).abs();
    let padded_span = observed_span
        .max(ATLAS_MIN_SPAN)
        .mul_add(ATLAS_MARGIN_RATIO * 2.0, observed_span.max(ATLAS_MIN_SPAN));
    let half = padded_span * 0.5;
    let mut min = center - half;
    let mut max = center + half;

    if min < 0.0 {
        max = (max - min).min(1.0);
        min = 0.0;
    }
    if max > 1.0 {
        min = (min - (max - 1.0)).max(0.0);
        max = 1.0;
    }
    if (max - min).abs() <= f64::EPSILON {
        (0.0, 1.0)
    } else {
        (min, max)
    }
}

fn normalize_axis(value: f64, min: f64, max: f64) -> f64 {
    ((value - min) / (max - min)).clamp(0.0, 1.0)
}

#[derive(Clone, Default)]
struct HeatCell {
    count: usize,
    z_sum: f64,
}

impl HeatCell {
    fn average_z(&self) -> f64 {
        if self.count == 0 {
            0.5
        } else {
            self.z_sum / self.count as f64
        }
    }
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "heatmap coordinates are clamped to the target grid before conversion"
)]
fn coordinate_to_index(value: f64, size: usize) -> usize {
    let scaled = value.clamp(0.0, 1.0) * size.saturating_sub(1) as f64;
    scaled.round() as usize
}

fn heatmap_color(z: f64) -> Color {
    let z = z.clamp(0.0, 1.0);
    if z < 0.16 {
        Color::Rgb(68, 1, 84)
    } else if z < 0.33 {
        Color::Rgb(59, 82, 139)
    } else if z < 0.50 {
        Color::Rgb(33, 145, 140)
    } else if z < 0.66 {
        Color::Rgb(94, 201, 98)
    } else if z < 0.83 {
        Color::Rgb(173, 220, 48)
    } else {
        Color::Rgb(253, 231, 37)
    }
}

fn heatmap_cell_char(count: usize) -> Option<char> {
    match count {
        0 => None,
        1..=2 => Some('·'),
        3..=5 => Some('░'),
        6..=10 => Some('▒'),
        11..=20 => Some('▓'),
        _ => Some('█'),
    }
}

/// Filter log lines by selected verbosity.
#[must_use]
pub fn visible_log_lines(lines: &[LogLine], verbosity: LogLevel) -> Vec<LogLine> {
    lines
        .iter()
        .filter(|line| line.level <= verbosity)
        .cloned()
        .collect()
}

/// Infer a user-facing phase label from profiler output.
#[must_use]
pub fn phase_from_profile_message(message: &str) -> Option<String> {
    let phase = message.split(']').nth(1)?.trim().split(':').next()?.trim();
    let label = match phase {
        "model_load" => "Loading embedding model",
        "cache_prepare" => "Preparing persistent cache",
        "cache_manifest" => "Reading cache manifest",
        "cache_diff" => "Diffing repository",
        "reembed_dirty_files" => "Embedding changed files",
        "cache_load_objects" => "Loading cached objects",
        "cache_gc" => "Garbage-collecting cache",
        "cache_manifest_save" => "Saving cache manifest",
        "cache_write_objects" => "Writing cache objects",
        "build_hybrid_index" => "Building hybrid search index",
        _ => return None,
    };
    Some(label.to_string())
}

#[cfg(test)]
mod tests {
    use super::{
        EmbeddingPoint, LogLevel, LogLine, heatmap_cell_char, heatmap_color, heatmap_lines,
        phase_from_profile_message, project_embedding, visible_log_lines,
    };
    use ratatui::style::Color;

    #[test]
    fn verbosity_cycles_through_all_levels() {
        assert_eq!(LogLevel::Warn.next(), LogLevel::Info);
        assert_eq!(LogLevel::Info.next(), LogLevel::Debug);
        assert_eq!(LogLevel::Debug.next(), LogLevel::Trace);
        assert_eq!(LogLevel::Trace.next(), LogLevel::Warn);
    }

    #[test]
    fn visible_logs_respect_selected_verbosity() {
        let lines = vec![
            LogLine::new(LogLevel::Warn, "warn"),
            LogLine::new(LogLevel::Info, "info"),
            LogLine::new(LogLevel::Debug, "debug"),
            LogLine::new(LogLevel::Trace, "trace"),
        ];

        assert_eq!(visible_log_lines(&lines, LogLevel::Warn).len(), 1);
        assert_eq!(visible_log_lines(&lines, LogLevel::Info).len(), 2);
        assert_eq!(visible_log_lines(&lines, LogLevel::Debug).len(), 3);
        assert_eq!(visible_log_lines(&lines, LogLevel::Trace).len(), 4);
    }

    #[test]
    fn profiler_messages_map_to_human_phases() {
        assert_eq!(
            phase_from_profile_message("[0.001s] model_load: starting"),
            Some("Loading embedding model".to_string())
        );
        assert_eq!(
            phase_from_profile_message("[0.123s] build_hybrid_index: 42 chunks in 1.2s"),
            Some("Building hybrid search index".to_string())
        );
        assert_eq!(phase_from_profile_message("plain log line"), None);
    }

    #[test]
    fn embedding_projection_is_deterministic_and_bounded() {
        let embedding = vec![0.25, -0.5, 0.75, -1.0, 0.5, 0.125];
        let a = project_embedding(&embedding);
        let b = project_embedding(&embedding);

        assert_eq!(a, b);
        assert!((0.0..=1.0).contains(&a.x));
        assert!((0.0..=1.0).contains(&a.y));
        assert!((0.0..=1.0).contains(&a.z));
    }

    #[test]
    fn heatmap_cell_intensity_increases_with_density() {
        assert!(heatmap_cell_char(0).is_none());
        assert_eq!(heatmap_cell_char(1), Some('·'));
        assert_eq!(heatmap_cell_char(3), Some('░'));
        assert_eq!(heatmap_cell_char(7), Some('▒'));
        assert_eq!(heatmap_cell_char(15), Some('▓'));
        assert_eq!(heatmap_cell_char(32), Some('█'));
    }

    #[test]
    fn heatmap_auto_fits_tight_clusters_across_available_cells() {
        let points: Vec<_> = (0..24)
            .map(|i| {
                let offset = f64::from(i) * 0.0008;
                EmbeddingPoint {
                    x: 0.49 + offset,
                    y: 0.49 + offset,
                    z: 0.5,
                }
            })
            .collect();

        let lines = heatmap_lines(&points, 30, 10);
        let occupied: Vec<_> = lines
            .iter()
            .enumerate()
            .flat_map(|(row, line)| {
                line.spans
                    .iter()
                    .enumerate()
                    .filter_map(move |(column, span)| {
                        (!span.content.trim().is_empty()).then_some((row, column))
                    })
            })
            .collect();
        let min_column = occupied.iter().map(|(_, column)| *column).min().unwrap();
        let max_column = occupied.iter().map(|(_, column)| *column).max().unwrap();
        let min_row = occupied.iter().map(|(row, _)| *row).min().unwrap();
        let max_row = occupied.iter().map(|(row, _)| *row).max().unwrap();

        assert!(max_column - min_column >= 20);
        assert!(max_row - min_row >= 6);
    }

    #[test]
    fn heatmap_color_scale_avoids_alarm_red() {
        assert_ne!(heatmap_color(0.0), Color::Red);
        assert_ne!(heatmap_color(0.0), Color::LightRed);
        assert_ne!(heatmap_color(0.5), Color::Red);
        assert_ne!(heatmap_color(0.5), Color::LightRed);
        assert_ne!(heatmap_color(1.0), Color::Red);
        assert_ne!(heatmap_color(1.0), Color::LightRed);
    }
}
