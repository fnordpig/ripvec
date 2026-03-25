#!/bin/bash
# Check if ripvec-mcp is installed and available in PATH.
# Runs on SessionStart — prints a hint if the binary is missing.

if ! command -v ripvec-mcp &>/dev/null; then
	echo "ripvec-mcp not found. Install with: cargo install --git https://github.com/fnordpig/ripvec ripvec-mcp"
	exit 1
fi
