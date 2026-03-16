# Claude Code Rust Development Setup — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Configure Claude Code with LSP, MCP servers, hooks, and dev tooling for optimal Rust development on the ripvec project.

**Architecture:** Five version-controlled config files (CLAUDE.md, .claude/settings.json, .mcp.json, bacon.toml, scripts/dev-setup.sh) plus a one-time script execution to install toolchain components, cargo tools, MCP servers, and the LSP plugin.

**Tech Stack:** Rust (stable), rust-analyzer, cargo tooling (bacon, nextest, insta, machete, audit), MCP servers (rust-mcp-server, rust-analyzer-mcp, crates-mcp), Claude Code hooks

**Spec:** `docs/superpowers/specs/2026-03-15-claude-code-rust-setup-design.md`

---

## Chunk 1: Config Files and Installation

### Task 1: Create CLAUDE.md

**Files:**
- Create: `CLAUDE.md`

- [ ] **Step 1: Write CLAUDE.md**

Copy the exact content from the spec (Section 1). The file goes at the project root.

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

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "Add CLAUDE.md with project conventions and commands"
```

---

### Task 2: Create .claude/settings.json (hooks)

**Files:**
- Create: `.claude/settings.json`

- [ ] **Step 1: Verify jq is available**

Run: `command -v jq`
Expected: a path like `/opt/homebrew/bin/jq` or `/usr/local/bin/jq`

If missing: `brew install jq`

- [ ] **Step 2: Write .claude/settings.json**

The `.claude/` directory already exists. Write the hooks config from the spec (Section 2):

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
        ]
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

- [ ] **Step 3: Commit**

```bash
git add .claude/settings.json
git commit -m "Add Claude Code hooks: auto-format on edit, cargo check on stop"
```

---

### Task 3: Create .mcp.json (MCP servers)

**Files:**
- Create: `.mcp.json`

- [ ] **Step 1: Write .mcp.json**

At the project root. Content from spec Section 3:

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

- [ ] **Step 2: Commit**

```bash
git add .mcp.json
git commit -m "Add MCP server config: rust-mcp-server, rust-analyzer-mcp, crates-mcp"
```

---

### Task 4: Create bacon.toml

**Files:**
- Create: `bacon.toml`

- [ ] **Step 1: Write bacon.toml**

At the project root. Content from spec Section 4:

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

- [ ] **Step 2: Commit**

```bash
git add bacon.toml
git commit -m "Add bacon.toml for background compilation jobs"
```

---

### Task 5: Create scripts/dev-setup.sh

**Files:**
- Create: `scripts/dev-setup.sh`

- [ ] **Step 1: Write scripts/dev-setup.sh**

The `scripts/` directory already exists. Content from spec Section 5:

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

- [ ] **Step 2: Commit**

```bash
chmod +x scripts/dev-setup.sh
git add scripts/dev-setup.sh
git commit -m "Add dev-setup.sh for one-time tool and plugin installation"
```

---

### Task 6: Run dev-setup.sh

This task installs all tools and plugins. It will take several minutes due to cargo compilation.

- [ ] **Step 1: Run the setup script**

Run: `./scripts/dev-setup.sh`

Expected output (summarized):
- `rustup component add` reports components as already installed or newly installed
- `cargo install` compiles and installs 8 tools (bacon, cargo-nextest, cargo-insta, cargo-machete, cargo-audit, rust-mcp-server, rust-analyzer-mcp, crates-mcp)
- `claude plugin install` installs and enables the rust-analyzer-lsp plugin

This will take 5-15 minutes depending on network and CPU.

- [ ] **Step 2: Verify toolchain components**

Run: `rustup component list --installed | rg 'rust-analyzer|rust-src|clippy|rustfmt'`

Expected: all four components listed.

- [ ] **Step 3: Verify cargo tools**

Run: `bacon --version && cargo nextest --version && cargo insta --version && cargo machete --version && cargo audit --version`

Expected: version output for each tool (no errors).

- [ ] **Step 4: Verify MCP servers**

Run: `command -v rust-mcp-server && command -v rust-analyzer-mcp && command -v crates-mcp`

Expected: paths for all three binaries (e.g. `/Users/rwaugh/.cargo/bin/rust-mcp-server`).

- [ ] **Step 5: Verify LSP plugin**

Run: `claude plugin list 2>&1 | rg rust-analyzer`

Expected: `rust-analyzer-lsp` listed as enabled.

---

### Task 7: Verify setup end-to-end

- [ ] **Step 1: Verify cargo fmt hook would work**

Run: `cargo fmt --quiet` (on current src/main.rs)

Expected: exit 0, no output (file already formatted).

- [ ] **Step 2: Verify cargo check hook would work**

Run: `cargo check --workspace --quiet`

Expected: exit 0, compiles successfully. Note: first run compiles dependencies.

- [ ] **Step 3: Verify MCP server connectivity**

Run: `claude mcp list`

Expected: `rust-mcp-server`, `rust-analyzer`, and `crates` listed (they may show as connected or pending restart).

- [ ] **Step 4: Push all commits to remote**

```bash
git push origin main
```

- [ ] **Step 5: Print summary**

Confirm these are all working:
- CLAUDE.md at project root
- Hooks in .claude/settings.json (PostToolUse fmt, Stop check)
- MCP servers in .mcp.json (3 servers)
- bacon.toml (4 jobs)
- scripts/dev-setup.sh (idempotent installer)
- All cargo tools installed
- LSP plugin enabled
