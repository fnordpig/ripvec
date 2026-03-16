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
