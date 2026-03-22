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
        app.index_summary.clone()
    } else {
        format!("{} matches, {:.1}ms", app.results.len(), app.rank_time_ms)
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

/// Keybinding hints shown right-aligned in the status bar.
const STATUS_HINTS: &str =
    "ESC quit \u{2502} ENTER open \u{2502} \u{2191}\u{2193} navigate \u{2502} ^L redraw ";

/// Draw the status bar with file location left-aligned and keybinding hints right-aligned.
fn draw_status_bar(frame: &mut Frame, app: &App, area: Rect) {
    let location = if let Some((chunk_idx, _)) = app.results.get(app.selected) {
        let chunk = &app.index.chunks[*chunk_idx];
        format!(
            " {}:{}-{}",
            chunk.file_path, chunk.start_line, chunk.end_line
        )
    } else {
        String::new()
    };

    // Pad the location string with spaces so the hints appear flush-right.
    let width = area.width as usize;
    let hints_len = STATUS_HINTS.chars().count();
    let loc_len = location.chars().count();
    let pad = width.saturating_sub(loc_len + hints_len);

    // Show status flash (e.g. "↻ 3 files updated") if active, otherwise normal location
    let line = if let Some((ref msg, _)) = app.status_flash {
        let msg_len = msg.chars().count();
        let flash_pad = width.saturating_sub(msg_len + hints_len);
        Line::from(vec![
            Span::styled(
                format!(" {msg}"),
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(ratatui::style::Modifier::BOLD),
            ),
            Span::raw(" ".repeat(flash_pad)),
            Span::styled(STATUS_HINTS, Style::default().fg(Color::DarkGray)),
        ])
    } else {
        Line::from(vec![
            Span::styled(&*location, Style::default().fg(Color::Cyan)),
            Span::raw(" ".repeat(pad)),
            Span::styled(STATUS_HINTS, Style::default().fg(Color::DarkGray)),
        ])
    };

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
