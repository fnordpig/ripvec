//! Keyboard input handling for the TUI.
//!
//! Maps key events to application actions. Returns whether the query
//! changed and a re-rank is needed.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use super::App;

/// Handle a key event, mutating application state.
///
/// Returns `true` if the query changed and results should be re-ranked.
pub fn handle_key(app: &mut App, key: KeyEvent) -> bool {
    match key.code {
        // Quit
        KeyCode::Esc => {
            app.should_quit = true;
            false
        }
        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.should_quit = true;
            false
        }

        // Clear query
        KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            if app.query.is_empty() {
                return false;
            }
            app.query.clear();
            true
        }

        // Force full redraw (standard terminal convention)
        KeyCode::Char('l') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.force_redraw = true;
            false
        }

        // Type a character
        KeyCode::Char(c) => {
            app.query.push(c);
            true
        }

        // Delete character
        KeyCode::Backspace => {
            if app.query.pop().is_some() {
                return true;
            }
            false
        }

        // Navigate results
        KeyCode::Up => {
            if app.selected > 0 {
                app.selected -= 1;
                app.preview_scroll = 0;
                app.force_redraw = true;
            }
            false
        }
        KeyCode::Down => {
            if !app.results.is_empty() && app.selected < app.results.len() - 1 {
                app.selected += 1;
                app.preview_scroll = 0;
                app.force_redraw = true;
            }
            false
        }

        // Scroll preview
        KeyCode::PageUp => {
            app.preview_scroll = app.preview_scroll.saturating_sub(10);
            app.force_redraw = true;
            false
        }
        KeyCode::PageDown => {
            app.preview_scroll = app.preview_scroll.saturating_add(10);
            app.force_redraw = true;
            false
        }

        // Jump to first / last result
        KeyCode::Home => {
            if !app.results.is_empty() {
                app.selected = 0;
                app.preview_scroll = 0;
                app.force_redraw = true;
            }
            false
        }
        KeyCode::End => {
            if !app.results.is_empty() {
                app.selected = app.results.len() - 1;
                app.preview_scroll = 0;
                app.force_redraw = true;
            }
            false
        }

        // Open selected result in $EDITOR
        KeyCode::Enter => {
            if let Some(&(chunk_idx, _)) = app.results.get(app.selected) {
                let chunk = &app.index.chunks()[chunk_idx];
                app.open_editor = Some((chunk.file_path.clone(), chunk.start_line));
            }
            false
        }

        // Tab and all other unrecognised keys are ignored.
        // Tab is listed here as a placeholder for future focus switching.
        _ => false,
    }
}
