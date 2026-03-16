# Claude Code Rust Development Setup for ripvec

## Overview

Configuration design for an optimal Claude Code development environment for ripvec — a semantic search CLI that searches code, structured text (SQL, Jinja2, etc.), and plain text using ONNX embeddings, tree-sitter chunking, and cosine similarity.

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Hook style | Lightweight | ort/tree-sitter native deps make clippy-on-edit slow; bacon in split terminal provides continuous feedback |
| MCP servers | Full stack (3 servers) | rust-mcp-server for cargo tooling, crates-mcp for dependency research, rust-analyzer-mcp for code intelligence |
| LSP plugin | Official Anthropic | Simplest install; rust-analyzer-mcp provides fallback if LSP plugin flakes |
| Cargo tools | Essentials + quality | bacon, nextest, insta, machete, audit — covers dev loop + security for native deps |
| Delivery | Version-controlled config + install script | Config travels with repo; script is convenience, not requirement |

## Files

### 1. `CLAUDE.md` — Project conventions

Note: This CLAUDE.md describes the target workspace architecture. Until the
workspace is restructured, per-crate commands (e.g. `cargo test -p ripvec-core`)
won't work. Update this file as the project structure evolves.

```markdown
# ripvec

Semantic search CLI — like ripgrep but for meaning, not text. Searches code,
structured text (SQL, Jinja2, etc.), and plain text using ONNX embeddings,
tree-sitter chunking, and cosine similarity. Rust 2024 edition.

## Commands
- `cargo check --workspace`              # Fast compilation check (prefer over build)
- `cargo test --workspace`               # All tests
- `cargo test -p ripvec-core`            # Core library tests only
- `cargo nextest run`                    # Faster test runner (requires cargo-nextest)
- `cargo clippy --all-targets -- -D warnings`  # Lint (zero warnings)
- `cargo fmt` / `cargo fmt --check`      # Format

## Architecture
Cargo workspace with three crates (target structure):
- `ripvec-core` — shared library (model, chunking, embedding, search)
- `ripvec` — CLI binary (clap)
- `ripvec-mcp` — MCP server binary (rmcp)

## Conventions
- Error handling: `thiserror` in ripvec-core, `anyhow` in binaries
- Never `.unwrap()` except in tests; use `?` everywhere
- `#[expect()]` over `#[allow()]` for lint suppression
- All public items must have doc comments (`///`)
- Prefer `impl Trait` parameters over generic type params where possible

## Chunking architecture
- Tree-sitter grammars are pluggable via a trait/registry — not hardcoded
- Support code (Rust, Python, JS/TS, Go, Java, C/C++), structured text
  (SQL, Jinja2), and plain text (paragraph/sentence splitting)
- `Parser` is not thread-safe — create per-thread via rayon
- `Query` is Send + Sync — share via Arc

## Key invariants
- ONNX model (~50MB) must NOT be committed to git
- `_mmap` field must outlive the ONNX `Session`
- BGE models use CLS pooling; all-MiniLM uses mean pooling — don't mix
- All embeddings are L2-normalized; cosine similarity = dot product

## Before committing
Run: `cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test --workspace`
```

### 2. `.claude/settings.json` — Lightweight hooks

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "file_path=$(jq -r '.tool_input.file_path // empty'); if [ -n \"$file_path\" ] && echo \"$file_path\" | rg -q '\\.rs$'; then cargo fmt --quiet 2>&1; fi",
            "timeout": 30
          }
        ],
      }
    ],
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "cargo check --workspace --quiet 2>&1",
            "timeout": 120
          }
        ]
      }
    ]
  }
}
```

Design notes:
- Hook reads tool input from stdin via `jq`, extracting `tool_input.file_path`
- Uses `rg` instead of `grep` per user's shell alias convention
- 120s timeout on Stop accounts for ort/tree-sitter native compilation
- No clippy-on-edit — bacon in split terminal covers this
- No commit gating — CLAUDE.md documents pre-commit command
- Stop hook has no matcher — the event fires on all stops
- Requires `jq` on PATH (standard on macOS with Homebrew; install if missing)

### 3. `.mcp.json` — Rust MCP servers

```json
{
  "mcpServers": {
    "rust-mcp-server": {
      "command": "rust-mcp-server",
      "args": []
    },
    "rust-analyzer": {
      "command": "rust-analyzer-mcp"
    },
    "crates": {
      "command": "crates-mcp"
    }
  }
}
```

Server capabilities:
- **rust-mcp-server** — 25+ cargo tools including `cargo-check`, `cargo-test`, `cargo-clippy`, `cargo-fmt`, `cargo-add`, `cargo-machete`, `rustc-explain`
- **rust-analyzer-mcp** — code intelligence tools including `definition`, `references`, `hover`, `diagnostics`, `workspace_diagnostics`, `code_actions`
- **crates-mcp** — `search_crates`, `get_crate_info`, `get_crate_versions`, `get_crate_documentation`

context7 is already configured globally and not duplicated here.

### 4. `bacon.toml` — Background compiler

```toml
default_job = "check"

[jobs.check]
command = ["cargo", "check", "--workspace", "--all-targets", "--color", "always"]

[jobs.clippy]
command = ["cargo", "clippy", "--workspace", "--all-targets", "--color", "always", "--", "-D", "warnings"]

[jobs.test]
command = ["cargo", "nextest", "run", "--workspace", "--color", "always"]

[jobs.doc]
command = ["cargo", "doc", "--workspace", "--no-deps", "--color", "always"]
```

### 5. `scripts/dev-setup.sh` — One-time installation

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "=== Rust toolchain components ==="
# rust-src is required by rust-analyzer for std library analysis
rustup component add rust-analyzer rust-src rustfmt clippy

echo "=== Cargo development tools ==="
cargo install --locked bacon cargo-nextest cargo-insta cargo-machete cargo-audit

echo "=== Cargo MCP servers ==="
cargo install --locked rust-mcp-server rust-analyzer-mcp crates-mcp

echo "=== Claude Code LSP plugin ==="
claude plugin install rust-analyzer-lsp

echo "=== Done! ==="
echo "Restart Claude Code to activate LSP and MCP servers."
echo "Run 'bacon' in a split terminal for continuous compilation feedback."
```

Idempotent — safe to re-run. Does not modify global `~/.claude/settings.json`.

## Deliberate Omissions

- **No clippy-on-edit hooks** — bacon covers this without slowing the edit cycle
- **No commit-gating hooks** — CLAUDE.md documents the pre-commit command
- **No `deny.toml`** — add when dependencies exist to audit
- **No CI/CD workflows** — specified in design.md, belong in implementation plan
- **No nightly/miri** — not needed until unsafe code appears
- **No Piebald-AI plugin or tweakcc** — official plugin is simpler; rust-analyzer-mcp is fallback
- **No `ENABLE_LSP_TOOL` env var** — unclear if still required; add if LSP doesn't activate
