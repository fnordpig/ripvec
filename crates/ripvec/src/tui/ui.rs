//! Layout rendering for the three-pane TUI.
//!
//! ```text
//! +-- Query -------------------------------------------------------+
//! | > query_text_                            N matches, X.Xms      |
//! +-- Results ----------------+-- Preview -------------------------+
//! | > 1. [0.82] func_name    |  fn func_name() {                  |
//! |   2. [0.76] other_func   |      // code here                  |
//! |   ...                     |  }                                 |
//! +---------------------------+------------------------------------+
//! |  file.rs:42-51                                      ESC quit   |
//! +----------------------------------------------------------------+
//! ```

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap};

use super::App;

/// Draw the full TUI frame: query bar, results/preview split, status bar.
pub fn draw(frame: &mut Frame, app: &App) {
    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // query bar
            Constraint::Min(5),    // results + preview
            Constraint::Length(1), // status bar
        ])
        .split(frame.area());

    draw_query_bar(frame, app, outer[0]);
    draw_main_panes(frame, app, outer[1]);
    draw_status_bar(frame, app, outer[2]);
}

/// Draw the query input bar with match count and timing.
fn draw_query_bar(frame: &mut Frame, app: &App, area: Rect) {
    let status_text = if app.query.is_empty() {
        String::from("type to search")
    } else {
        format!("{} matches, {:.1}ms", app.results.len(), app.rank_time_ms,)
    };

    let line = Line::from(vec![
        Span::styled(
            "> ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(&app.query),
        Span::styled("_", Style::default().fg(Color::DarkGray)),
        // Right-align the status by padding
        Span::raw("  "),
        Span::styled(status_text, Style::default().fg(Color::DarkGray)),
    ]);

    let block = Block::default()
        .title(" Query ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan));

    let paragraph = Paragraph::new(line).block(block);
    frame.render_widget(paragraph, area);
}

/// Draw the results list and preview pane side by side.
fn draw_main_panes(frame: &mut Frame, app: &App, area: Rect) {
    let panes = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(40), // results
            Constraint::Percentage(60), // preview
        ])
        .split(area);

    draw_results(frame, app, panes[0]);
    draw_preview(frame, app, panes[1]);
}

/// Draw the ranked results list.
fn draw_results(frame: &mut Frame, app: &App, area: Rect) {
    let items: Vec<ListItem> = app
        .results
        .iter()
        .enumerate()
        .map(|(i, (chunk_idx, score))| {
            let chunk = &app.index.chunks[*chunk_idx];
            let marker = if i == app.selected { "\u{25b8} " } else { "  " };
            let text = format!(
                "{marker}{}. [{:.2}] {}",
                i + 1,
                score,
                truncate_name(&chunk.name, 30),
            );
            let style = if i == app.selected {
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            ListItem::new(text).style(style)
        })
        .collect();

    let block = Block::default()
        .title(" Results ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Green));

    let list = List::new(items).block(block).highlight_style(
        Style::default()
            .fg(Color::Yellow)
            .add_modifier(Modifier::BOLD),
    );

    let mut state = ListState::default();
    if !app.results.is_empty() {
        state.select(Some(app.selected));
    }

    frame.render_stateful_widget(list, area, &mut state);
}

/// Draw the preview pane showing the selected chunk's content with syntax highlighting.
fn draw_preview(frame: &mut Frame, app: &App, area: Rect) {
    let block = Block::default()
        .title(" Preview ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Blue));

    let lines: Vec<Line<'_>> = if let Some((chunk_idx, _)) = app.results.get(app.selected) {
        let chunk = &app.index.chunks[*chunk_idx];
        let extension = std::path::Path::new(&chunk.file_path)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");
        app.highlighter.highlight(&chunk.content, extension)
    } else {
        vec![Line::raw("No result selected")]
    };

    let paragraph = Paragraph::new(lines)
        .block(block)
        .wrap(Wrap { trim: false })
        .scroll((app.preview_scroll, 0));

    frame.render_widget(paragraph, area);
}

/// Draw the status bar with file location and help text.
fn draw_status_bar(frame: &mut Frame, app: &App, area: Rect) {
    let location = if let Some((chunk_idx, _)) = app.results.get(app.selected) {
        let chunk = &app.index.chunks[*chunk_idx];
        format!(
            "{}:{}-{}",
            chunk.file_path, chunk.start_line, chunk.end_line
        )
    } else {
        String::new()
    };

    let chunks_info = format!("{} chunks indexed", app.index.chunks.len());

    let line = Line::from(vec![
        Span::styled(format!(" {location}"), Style::default().fg(Color::Cyan)),
        Span::raw("  "),
        Span::styled(chunks_info, Style::default().fg(Color::DarkGray)),
        Span::raw("  "),
        Span::styled(
            "ESC quit  \u{2191}\u{2193} navigate  PgUp/PgDn scroll ",
            Style::default().fg(Color::DarkGray),
        ),
    ]);

    let paragraph =
        Paragraph::new(line).style(Style::default().bg(Color::DarkGray).fg(Color::White));
    frame.render_widget(paragraph, area);
}

/// Truncate a name to fit in the results column.
fn truncate_name(name: &str, max_len: usize) -> &str {
    if name.len() <= max_len {
        name
    } else {
        // Find a safe UTF-8 boundary
        let mut end = max_len;
        while end > 0 && !name.is_char_boundary(end) {
            end -= 1;
        }
        &name[..end]
    }
}
