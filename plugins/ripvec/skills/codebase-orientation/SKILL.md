---
name: codebase-orientation
description: "Use when starting work on unfamiliar code, asked about project structure or architecture, exploring how modules connect, or needing to understand a codebase before making changes. Triggers on: 'how does this project work', 'explain the architecture', 'show me the structure', 'where should I start', 'what are the main modules', 'how are things organized'."
---

# Codebase Orientation

When you need to understand how a project is structured — which files are central, what depends on what, where the key abstractions live — use `get_repo_map` before reading individual files.

## Why this matters

Reading files one by one to understand architecture is slow and expensive. A single `get_repo_map` call returns a PageRank-weighted overview showing:
- Which files are structurally central (imported by many others)
- Key definitions (traits, structs, functions) with signatures
- Dependency flow (who calls whom)

This replaces 10+ sequential Read operations with one tool call.

## How to orient

**Step 1: Get the structural overview**
```
get_repo_map(max_tokens: 2000)
```
The output ranks files by structural importance. The top files are the architectural spine — read these first.

**Step 2: Zoom into the area you're working on**
```
get_repo_map(focus_file: "src/auth/middleware.ts", max_tokens: 1500)
```
Topic-sensitive PageRank concentrates on the focus file's neighborhood — what it depends on and what depends on it.

**Step 3: Read the top-ranked files**
Use LSP `documentSymbol` on the 2-3 highest-ranked files for detailed symbol listings, then `Read` for implementation details.

## Examples across languages

**Rust monorepo**: "How does the backend trait system work?"
→ `get_repo_map` shows `backend/mod.rs` as high-rank with trait definition, then each backend impl file as callees

**Django project**: "Where do I start understanding this app?"
→ `get_repo_map` shows `urls.py` and `models.py` as most-imported, `views.py` as primary caller — read in that order

**React + Express full-stack**: "How does the frontend talk to the backend?"
→ `get_repo_map` reveals the API boundary: `api/routes/index.ts` as the hub, `src/hooks/useApi.ts` as the frontend entry point

**dbt project**: "Which models are most critical?"
→ `get_repo_map` PageRanks the models — the most-referenced staging models rank highest, showing the dependency spine

## When NOT to use this

- You already know the file you need → just `Read` it
- You need an exact string → use `Grep`
- You need a specific symbol definition → use LSP `goToDefinition`

Orientation is for the "I don't know where to start" moment. Once oriented, switch to precise tools.
