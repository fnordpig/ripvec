//! Syntax highlighting for the preview pane using syntect.

use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use syntect::easy::HighlightLines;
use syntect::highlighting::{Style as SynStyle, ThemeSet};
use syntect::parsing::SyntaxSet;

/// Cached syntax highlighting state.
///
/// Holds pre-loaded syntax definitions and themes so highlighting is fast
/// after the initial construction cost.
pub struct Highlighter {
    /// Syntax definitions for all supported languages.
    syntax_set: SyntaxSet,
    /// Color themes.
    theme_set: ThemeSet,
}

impl Highlighter {
    /// Create a new highlighter with default syntax definitions and theme.
    pub fn new() -> Self {
        Self {
            syntax_set: SyntaxSet::load_defaults_newlines(),
            theme_set: ThemeSet::load_defaults(),
        }
    }

    /// Highlight source code and convert to ratatui [`Line`]s.
    ///
    /// Uses the file extension to determine the syntax. Falls back to
    /// plain text if the extension is unrecognized.
    pub fn highlight(&self, content: &str, extension: &str) -> Vec<Line<'static>> {
        let syntax = self
            .syntax_set
            .find_syntax_by_extension(extension)
            .unwrap_or_else(|| self.syntax_set.find_syntax_plain_text());

        let theme = &self.theme_set.themes["base16-ocean.dark"];
        let mut h = HighlightLines::new(syntax, theme);

        content
            .lines()
            .map(|line| {
                let spans: Vec<Span<'static>> = match h.highlight_line(line, &self.syntax_set) {
                    Ok(ranges) => ranges
                        .into_iter()
                        .map(|(style, text)| {
                            Span::styled(text.to_string(), syntect_to_ratatui(style))
                        })
                        .collect(),
                    Err(_) => vec![Span::raw(line.to_string())],
                };
                Line::from(spans)
            })
            .collect()
    }
}

/// Convert a syntect style to a ratatui style.
fn syntect_to_ratatui(style: SynStyle) -> Style {
    let fg = style.foreground;
    Style::default().fg(Color::Rgb(fg.r, fg.g, fg.b))
}
