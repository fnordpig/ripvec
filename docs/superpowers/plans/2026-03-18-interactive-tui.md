# Interactive TUI — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `-i` interactive mode with real-time semantic search TUI, plus indicatif progress bars, owo-colors output, and ndarray BLAS ranking.

**Architecture:** Two-phase flow: index phase (indicatif progress) builds in-memory SearchIndex, then ratatui TUI re-ranks via ndarray BLAS on each keystroke.

**Tech Stack:** ratatui 0.29, crossterm 0.28, indicatif 0.17, owo-colors 4.1, syntect 5.2, ndarray 0.17

---

## Task 1: Add dependencies and ndarray BLAS ranking

**Files:**
- Modify: `Cargo.toml` (workspace deps)
- Modify: `crates/ripvec-core/Cargo.toml` (ndarray dep)
- Modify: `crates/ripvec/Cargo.toml` (ratatui, crossterm, indicatif, owo-colors, syntect)
- Modify: `crates/ripvec-core/src/similarity.rs` (add `rank_all`)

- [ ] Add workspace dependencies: ratatui, crossterm, indicatif, owo-colors, syntect
- [ ] Add ndarray to ripvec-core (already present for ort feature, make non-optional or add separate)
- [ ] Implement `rank_all(embeddings: &Array2<f32>, query: &Array1<f32>) -> Vec<f32>` using BLAS dot product
- [ ] Test: `rank_all` produces same results as scalar `dot_product` loop
- [ ] Benchmark: verify <5ms for 237K × 384 matrix
- [ ] Commit: `feat: add ndarray BLAS ranking for real-time re-ranking`

---

## Task 2: Extract `embed_all` from `search()`

**Files:**
- Modify: `crates/ripvec-core/src/embed.rs`

- [ ] Extract walk + chunk + embed phases into `pub fn embed_all(...)` returning `(Vec<CodeChunk>, Vec<Vec<f32>>)`
- [ ] Refactor `search()` to call `embed_all()` internally, then rank
- [ ] Test: `search()` produces identical results after refactor
- [ ] Commit: `refactor: extract embed_all for reuse by TUI`

---

## Task 3: Replace ANSI escapes with owo-colors

**Files:**
- Modify: `crates/ripvec/src/output.rs`

- [ ] Replace hardcoded `\x1b[...m` sequences with owo-colors methods
- [ ] Verify `NO_COLOR=1` disables colors
- [ ] Test: output matches visually
- [ ] Commit: `refactor: replace ANSI escapes with owo-colors`

---

## Task 4: Add indicatif progress bars for index phase

**Files:**
- Create: `crates/ripvec/src/progress.rs`
- Modify: `crates/ripvec-core/src/embed.rs` (add progress callback)

- [ ] Create `ProgressReporter` that wraps indicatif bars for model_load, walk, chunk, embed phases
- [ ] Add optional progress callback to `embed_all()` (or use the existing `Profiler` trait)
- [ ] Wire into main.rs for both `-i` mode and normal mode (replace text profiler when TTY)
- [ ] Test: verify progress bar renders without panic
- [ ] Commit: `feat: add indicatif progress bars for embedding phase`

---

## Task 5: Build TUI app skeleton with ratatui

**Files:**
- Create: `crates/ripvec/src/tui/mod.rs`
- Create: `crates/ripvec/src/tui/ui.rs`
- Create: `crates/ripvec/src/tui/input.rs`
- Modify: `crates/ripvec/src/cli.rs` (add `-i` flag)
- Modify: `crates/ripvec/src/main.rs` (route to TUI)

- [ ] Add `-i` / `--interactive` flag to CLI
- [ ] Create TUI app state struct: query string, selected index, scroll offset, SearchIndex
- [ ] Implement event loop: crossterm events → state updates → ratatui render
- [ ] Implement three-pane layout (query bar, results list, preview pane, status bar)
- [ ] Wire keystroke → re-embed query → ndarray rank → update results
- [ ] Test: TUI initializes and renders without panic
- [ ] Commit: `feat: interactive TUI skeleton with three-pane layout`

---

## Task 6: Build SearchIndex and real-time re-ranking

**Files:**
- Create: `crates/ripvec/src/tui/index.rs`

- [ ] Implement `SearchIndex` struct holding chunks + ndarray Array2 embeddings
- [ ] Implement `SearchIndex::from_embed_all()` converting Vec<Vec<f32>> to Array2
- [ ] Implement `SearchIndex::query()` that embeds query + ranks via BLAS, returns top matches with timing
- [ ] Handle empty query (show nothing or show all chunks sorted by name)
- [ ] Test: known embeddings → expected ranking order
- [ ] Commit: `feat: SearchIndex with BLAS re-ranking`

---

## Task 7: Add syntect syntax highlighting in preview pane

**Files:**
- Create: `crates/ripvec/src/tui/highlight.rs`

- [ ] Load syntect theme (base16-ocean or similar dark theme)
- [ ] Map file extensions to syntect syntax definitions
- [ ] Implement `highlight_code(content: &str, extension: &str) -> Vec<ratatui::text::Line>`
- [ ] Integrate into preview pane rendering
- [ ] Cache highlighted lines per selected result (avoid re-highlighting on scroll)
- [ ] Commit: `feat: syntect syntax highlighting in preview pane`

---

## Task 8: Editor integration and polish

**Files:**
- Modify: `crates/ripvec/src/tui/input.rs`
- Modify: `crates/ripvec/src/tui/mod.rs`

- [ ] On Enter: suspend TUI, open `$EDITOR +line file`, resume TUI on exit
- [ ] Add Ctrl-U to clear query
- [ ] Add Tab to toggle preview scroll focus
- [ ] Add match count + ranking time to query bar
- [ ] Add file path + line range to status bar
- [ ] Handle terminal resize events
- [ ] Commit: `feat: editor integration and TUI polish`

---

## Task 9: Integration testing and cleanup

**Files:**
- Modify: `crates/ripvec/tests/integration.rs`

- [ ] Test: `-i` flag is recognized by CLI parser
- [ ] Test: non-interactive mode still works identically
- [ ] Run full test suite: `cargo test --workspace`
- [ ] Run clippy: `cargo clippy --all-targets -- -D warnings`
- [ ] Manual test: `ripvec -i tests/corpus/code` on code corpus
- [ ] Manual test: `ripvec -i tests/corpus/gutenberg` on text corpus
- [ ] Commit: `test: integration tests for interactive mode`

---

## Risk Assessment

| Task | Risk | Mitigation |
|---|---|---|
| Task 1 (ndarray) | Low — mature crate | Already used by ORT backend |
| Task 2 (embed_all) | Low — pure refactor | Existing tests cover behavior |
| Task 5 (ratatui TUI) | Medium — UI code is fiddly | Keep layout simple, iterate |
| Task 6 (real-time ranking) | Medium — query re-embedding latency | Debounce if >100ms |
| Task 7 (syntect) | Low — well-documented | Can skip if too slow |
