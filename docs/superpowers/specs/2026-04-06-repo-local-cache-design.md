# Repo-Local Cache Indices

**Date**: 2026-04-06
**Status**: Approved

## Problem

Indices are stored globally in `~/.cache/ripvec/<project_hash>/...`. This means:
- Team members must re-embed from scratch after cloning
- CI/CD can't prebuild and share indices
- No way to ship a repo with instant semantic search out of the box

## Solution

Opt-in repo-local indices stored in `.ripvec/cache/` at the project root, committed to git. Runtime resolution walks up the directory tree to find a repo-local index before falling back to the user-level cache.

## Storage Layout

```
.ripvec/
├── config.toml                          # Marker + model pin
└── cache/
    ├── manifest.json                    # File entries + Merkle hashes
    └── objects/                          # Content-addressed object store
        └── ab/
            └── <hash>                   # bitcode+zstd compressed FileCache
```

### config.toml

Minimal — only what's needed for cache validation:

```toml
[cache]
local = true
model = "nomic-ai/modernbert-embed-base"
version = "3"
```

## Serialization Migration

Replace rkyv with **bitcode** for `FileCache` serialization in the object store.

- `FileCache`, `CodeChunk`, and their fields get `#[derive(bitcode::Encode, bitcode::Decode)]`
- Pipeline: `bitcode::encode → zstd::compress → write` (and reverse on read)
- Manifest version bumps from `v2` to `v3` — old rkyv caches automatically invalidated and rebuilt
- No migration path: existing caches re-embed on first use. Clean break.

**Unchanged:**
- `manifest.json` remains serde_json (human-readable, small)
- Object store layout (`xx/<hash>`) unchanged
- Merkle tree content hashing unchanged (blake3 of file content — fully portable)
- Diff algorithm unchanged

### Why bitcode

rkyv's zero-copy deserialization is architecture-dependent (endianness, pointer width). An index built on x86_64 Linux CI won't deserialize on aarch64 Mac — the most common real-world cross-platform scenario.

bitcode is the fastest non-zero-copy serialization format in Rust, fully portable, no schema ceremony. `#[derive(bitcode::Encode, bitcode::Decode)]` is drop-in.

## Cache Resolution Chain

In order of priority:

1. `--cache-dir` explicit override (highest — unchanged)
2. Walk up from project root looking for `.ripvec/config.toml` with `local = true` → use that `.ripvec/cache/`
3. `RIPVEC_CACHE` environment variable
4. `dirs::cache_dir()/ripvec/<project_hash>/...` (current default)

On the read path (search, repo map, etc.), resolution is automatic — no flags needed. The presence of `.ripvec/config.toml` in the directory tree is sufficient.

### Config Validation

When a repo-local index is found, validate `config.toml` model + version against the runtime model. On mismatch: warn and fall back to user-level cache. Don't silently use wrong embeddings.

## CLI Interface

New `--repo-level` flag alongside the existing `--index`:

```
ripvec --index --repo-level "search query"
```

- **First run**: creates `.ripvec/config.toml` + `.ripvec/cache/`, embeds into repo-local store, searches
- **Subsequent runs**: `--index` alone detects `.ripvec/` via resolution chain, uses it automatically. `--repo-level` only needed for initial creation.

## MCP Interface

New `repo_level: bool` parameter on `ReindexParams`:

```json
{ "root": "/path/to/project", "repo_level": true }
```

Same behavior: creates `.ripvec/` if absent, indexes into it. Subsequent tool calls (search_code, get_repo_map, etc.) automatically resolve via the chain — no `repo_level` param needed on read paths.

### index_status Enhancement

Report which cache location is active:

```
Index ready: 2383 chunks from 147 files (repo-local: .ripvec/cache/)
```

## Plugin Command

`/ripvec-repo-index` in `my-claude-plugins/plugins/ripvec/commands/`:

- Calls `reindex` with `repo_level: true` and `root` set to current project
- Description: "Create a repo-level search index that can be committed to git"

## First-Clone Experience

When a developer clones a repo with `.ripvec/` committed:

1. **Detection**: MCP/CLI finds `.ripvec/config.toml` via resolution chain — repo-local cache active
2. **Validation**: config.toml model + version checked against runtime model. Mismatch → warn, fall back to user cache
3. **Mtime heal**: First `compute_diff()` hits mtime misses on every file (git doesn't preserve mtimes), falls back to blake3 content hashing. All hashes match → zero files re-embedded. Manifest mtimes updated in-place for future fast-path.
4. **Ready**: Index loaded from pre-built objects, search available immediately after hash verification

**Cost**: One-time O(all files) blake3 hashing — seconds, not the minutes of re-embedding the index replaces.

**Partial staleness**: Developer pulls new commits that modify files. Changed files' content hashes won't match → re-embedded incrementally, new objects written to `.ripvec/cache/`. Developer can commit the updated index.

## Documentation

- **README.md**: Section on repo-level indexing — how to create, what to commit, gitignore options for large repos, first-clone experience, model pinning
- **CLAUDE.md**: Update cache resolution docs to reflect new chain

## Scope Summary

1. bitcode migration — swap rkyv for bitcode in FileCache, bump manifest version
2. Cache resolution chain — walk up for `.ripvec/config.toml`, fall back to user cache
3. `--repo-level` flag — CLI flag + MCP `reindex` param, creates `.ripvec/` on first use
4. Plugin command — `/ripvec-repo-index`
5. `index_status` enhancement — report active cache location
6. Mtime self-heal — update manifest mtimes after first clone validation
7. Documentation — README + CLAUDE.md
