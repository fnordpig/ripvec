# Interactive TUI Semantic Search — Design Spec

## Goal

Add an interactive mode (`ripvec -i ./src`) that embeds the codebase once, then provides a real-time semantic search TUI where results re-rank as the user types — like fzf but for meaning.

## Architecture

Two-phase flow:
1. **Index phase**: Walk → chunk → embed with indicatif progress bar. All embeddings held in memory.
2. **Interactive phase**: ratatui TUI with three-pane layout. Each keystroke triggers an ndarray BLAS dot product against the full embedding matrix (~3ms for 237K vectors). No persistence — embeddings die when the user exits.

## Dependencies

| Crate | Version | Purpose |
|---|---|---|
| `ratatui` | 0.29 | TUI framework |
| `crossterm` | 0.28 | Terminal backend for ratatui |
| `indicatif` | 0.17 | Progress bars during index phase |
| `owo-colors` | 4.1 | Colored non-interactive output (replaces hardcoded ANSI) |
| `syntect` | 5.2 | Syntax highlighting in preview pane |
| `ndarray` | 0.17 | BLAS-accelerated dot product for real-time re-ranking |

## TUI Layout

```
┌─ Query ──────────────────────────────────────────────────────────┐
│ > auth middleware_                              47 matches, 2.1ms │
├─ Results ─────────────────────┬─ Preview ────────────────────────┤
│ ► 1. [0.82] auth_middleware   │  fn auth_middleware(             │
│   2. [0.76] verify_token      │      req: &Request,              │
│   3. [0.71] login_handler     │      next: Next,                 │
│   4. [0.68] session_check     │  ) -> Response {                 │
│   5. [0.65] parse_bearer      │      let token = req.headers()   │
│   6. [0.61] role_guard        │          .get("Authorization")   │
│   ...                         │          .and_then(|v| v.strip_  │
│                               │      verify_token(&token)?;      │
│                               │      next.run(req).await         │
│                               │  }                               │
├───────────────────────────────┴──────────────────────────────────┤
│  src/auth.rs:42-51                                    ESC quit   │
└──────────────────────────────────────────────────────────────────┘
```

### Panes

- **Query bar** (top): Text input with cursor. Right-aligned: match count + ranking time.
- **Results list** (left): Scrollable list of matches. Shows rank, similarity score, function name. Selected item highlighted.
- **Code preview** (right): Syntax-highlighted source of the selected result. Scrollable independently if content exceeds pane height.
- **Status bar** (bottom): File path:lines of selected result. Right-aligned: keybinding hints.

### Keybindings

| Key | Action |
|---|---|
| Any character | Append to query, re-rank results |
| Backspace | Delete character, re-rank |
| Up/Down or Ctrl-P/N | Move selection in results list |
| Enter | Open selected result in `$EDITOR` at line number |
| Esc or Ctrl-C | Quit |
| Tab | Toggle focus between results list and preview scroll |
| Ctrl-U | Clear query |

## Index Phase

Uses indicatif for progress:

```
ripvec -i ./src
⠋ Loading model BAAI/bge-small-en-v1.5...          0.2s
⠋ Walking files...                              2,652 files
⠋ Chunking...                                  25,890 chunks
████████████████████░░░░░░  78% │ 20,194/25,890 │ 168/s │ ETA 34s
```

After embedding completes, terminal clears and TUI renders.

### Data structures held in memory

```rust
struct SearchIndex {
    /// All chunks with metadata (file, name, lines, content)
    chunks: Vec<CodeChunk>,
    /// Embedding matrix [num_chunks, hidden_dim] as contiguous ndarray
    embeddings: ndarray::Array2<f32>,
    /// Tokenizer for potential future use (not needed for re-ranking)
    _tokenizer: tokenizers::Tokenizer,
}
```

The embedding matrix is stored as a contiguous `Array2<f32>` for BLAS-accelerated matrix-vector multiply. Re-ranking is a single `embeddings.dot(&query_vec)` call.

## Re-ranking Flow

On each keystroke:
1. Tokenize the query string (single call, ~12us)
2. Embed the query (single `backend.embed_batch(&[enc])` call, ~10-40ms)
3. Compute similarities: `embeddings.dot(&query_vec)` via BLAS (~3ms for 237K)
4. Sort by similarity, filter by threshold, take top N for display
5. Update TUI

Steps 1-2 are the bottleneck at ~40ms. Steps 3-4 are <5ms. Total: ~45ms per keystroke, well within interactive range. If query embedding becomes a bottleneck, debounce at 100ms.

Note: The query must be re-embedded (not just re-ranked) because the embedding captures semantic meaning. Changing "auth" to "authentication" produces a different embedding vector.

## Non-interactive Output Improvements

Replace hardcoded ANSI escapes in `output.rs` with `owo-colors`:

```rust
// Before: "\x1b[1;32m{}\x1b[0m"
// After:
use owo_colors::OwoColorize;
println!("{}", (i + 1).green().bold());
```

`owo-colors` respects `NO_COLOR` env var and is zero-cost when colors are disabled.

## File Structure

```
crates/ripvec/src/
├── main.rs          # Add -i flag routing
├── cli.rs           # Add --interactive / -i flag
├── output.rs        # Replace ANSI with owo-colors
├── tui/
│   ├── mod.rs       # TUI app state + event loop
│   ├── ui.rs        # ratatui widget layout (3 panes)
│   ├── input.rs     # Keyboard event handling
│   ├── index.rs     # SearchIndex struct + re-ranking
│   └── highlight.rs # syntect integration for preview pane
└── progress.rs      # indicatif progress bars for index phase

crates/ripvec-core/src/
├── similarity.rs    # Replace scalar loop with ndarray BLAS
└── embed.rs         # Add method to return raw embeddings (not just ranked results)
```

## Changes to ripvec-core

### New: `embed_all` function

`search()` currently embeds + ranks + truncates in one call. The TUI needs the raw embeddings for re-ranking. Add:

```rust
/// Embed all chunks and return the raw embedding matrix + chunks.
/// Does not rank or filter — the caller handles that.
pub fn embed_all(
    root: &Path,
    backend: &dyn EmbedBackend,
    tokenizer: &tokenizers::Tokenizer,
    cfg: &SearchConfig,
    profiler: &Profiler,
) -> Result<(Vec<CodeChunk>, Vec<Vec<f32>>)>
```

This extracts phases 1-4 from `search()` without the ranking phase. The existing `search()` can call `embed_all` internally to avoid duplication.

### similarity.rs: ndarray BLAS

Replace the scalar dot product with ndarray for batch ranking:

```rust
/// Rank all chunks against a query embedding using BLAS.
/// Returns similarity scores in the same order as the input embeddings.
pub fn rank_all(
    embeddings: &ndarray::Array2<f32>,
    query: &ndarray::Array1<f32>,
) -> Vec<f32> {
    embeddings.dot(query).to_vec()
}
```

The existing `dot_product` stays for single-pair comparisons. `rank_all` is the batch version for the TUI.

## Editor Integration

On Enter, open the selected file at the correct line:

```rust
fn open_in_editor(file: &str, line: usize) -> std::io::Result<()> {
    let editor = std::env::var("EDITOR").unwrap_or_else(|_| "vi".to_string());
    // Most editors accept +LINE syntax
    Command::new(&editor)
        .arg(format!("+{line}"))
        .arg(file)
        .status()?;
    Ok(())
}
```

The TUI suspends (crossterm `LeaveAlternateScreen`), runs the editor, then resumes on editor exit.

## Error Handling

- Model load failure: show error and exit (not recoverable)
- File walk/chunk errors: log via tracing, continue (some files may be unreadable)
- Embedding errors: propagate via `?` as today
- TUI render errors: exit cleanly, restore terminal
- Editor spawn failure: show error in status bar, don't crash

## Testing

- Unit tests for `SearchIndex` re-ranking (known embeddings → expected order)
- Unit tests for `rank_all` BLAS dot product matches scalar version
- Snapshot tests (insta) for non-interactive output with owo-colors
- Integration test: index small fixture directory, verify TUI state
- Manual testing: interactive mode on code corpus

## Scope Boundaries

**In scope:**
- `-i` interactive mode with ratatui TUI
- indicatif progress bars for index phase
- owo-colors for non-interactive output
- ndarray BLAS for real-time re-ranking
- syntect syntax highlighting in preview
- Editor integration (Enter → $EDITOR)

**Out of scope (future):**
- Persistent index (lance/disk cache)
- File watching / incremental re-index
- Fuzzy matching (only semantic search)
- Multi-query / saved searches
- Custom themes / color schemes
