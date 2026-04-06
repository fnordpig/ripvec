# CI Fixes + Multi-Architecture Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all CI failures (tree-sitter ABI mismatch, MSRV toolchain) and expand release builds to 5 targets including ARM Linux and pre-built CUDA binaries.

**Architecture:** Bump tree-sitter to 0.26 for ABI 15 compatibility, add tracing warnings for query compilation failures, fix MSRV toolchain version, expand release.yml matrix with CUDA container builds on both x86_64 and aarch64.

**Tech Stack:** tree-sitter 0.26, GitHub Actions, nvidia/cuda Docker images, cargo features

---

### Task 1: Bump tree-sitter to 0.26 and fix grammar versions

**Files:**
- Modify: `Cargo.toml` (workspace root, lines 23-30) — bump tree-sitter and grammar versions
- Modify: `Cargo.lock` — auto-updated

- [ ] **Step 1: Update tree-sitter version in workspace Cargo.toml**

In `Cargo.toml`, change:

```toml
tree-sitter = "0.24"
```

to:

```toml
tree-sitter = "0.26"
```

- [ ] **Step 2: Check which grammar crates need bumps for tree-sitter 0.26 compatibility**

Run: `cargo check -p ripvec-core 2>&1 | head -30`

If any grammar crates fail to compile against tree-sitter 0.26, bump them to their latest versions. The current versions and likely needed bumps:

| Crate | Current | Action |
|-------|---------|--------|
| `tree-sitter-rust` | `"0.24"` | May need `"0.25"` or later |
| `tree-sitter-python` | `"0.23"` | May need bump |
| `tree-sitter-javascript` | `"0.25"` | Likely compatible |
| `tree-sitter-typescript` | `"0.23"` | May need bump |
| `tree-sitter-go` | `"0.23"` | May need bump |
| `tree-sitter-java` | `"0.23"` | May need bump |
| `tree-sitter-c` | `"0.24"` | May need bump |
| `tree-sitter-cpp` | `"0.23"` | May need bump |

Bump each one that fails to the latest compatible version. Use `cargo search tree-sitter-<lang>` to find latest versions.

- [ ] **Step 3: Check if streaming-iterator API changed**

tree-sitter 0.26 may have changed `QueryCursor::matches()` to return a standard `Iterator` instead of `StreamingIterator`. If so:

In `crates/ripvec-core/src/chunk.rs`, change:
```rust
use streaming_iterator::StreamingIterator;
// ...
while let Some(m) = StreamingIterator::next(&mut matches) {
```
to:
```rust
for m in matches {
```

And the same in `crates/ripvec-core/src/repo_map.rs`.

If `streaming-iterator` is no longer needed, remove it from workspace Cargo.toml and ripvec-core's Cargo.toml.

- [ ] **Step 4: Verify all tests pass**

Run: `cargo test -p ripvec-core --lib 2>&1 | tail -20`

The 13 previously failing tests should now pass:
- `chunk::tests::chunks_rust_functions_and_structs`
- `chunk::tests::empty_file_produces_no_chunks`
- `chunk::tests::enriched_content_has_header`
- `chunk::tests::extract_signature_rust_function`
- `chunk::tests::fallback_large_file_produces_windows`
- `chunk::tests::fallback_small_file_single_chunk`
- `chunk::tests::header_dropped_when_exceeding_max_bytes`
- `chunk::tests::large_definition_is_windowed`
- `chunk::tests::scope_chain_rust_impl_method`
- `languages::tests::all_supported_extensions`
- `languages::tests::rust_extension_resolves`
- `repo_map::tests::test_build_graph_on_fixtures`
- `repo_map::tests::test_extract_imports_rust`

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml Cargo.lock crates/ripvec-core/Cargo.toml crates/ripvec-core/src/chunk.rs crates/ripvec-core/src/repo_map.rs
git commit -m "fix: bump tree-sitter to 0.26 for ABI 15 grammar compatibility"
```

---

### Task 2: Add tracing warnings for query compilation failures

**Files:**
- Modify: `crates/ripvec-core/src/languages.rs:130` — log warning instead of silent `.ok()?`
- Modify: `crates/ripvec-core/src/repo_map.rs:127` — log warning instead of silent `.ok()?`

- [ ] **Step 1: Fix languages.rs**

In `crates/ripvec-core/src/languages.rs`, change line 130 from:

```rust
    let query = Query::new(&lang, query_str).ok()?;
```

to:

```rust
    let query = match Query::new(&lang, query_str) {
        Ok(q) => q,
        Err(e) => {
            tracing::warn!(ext, %e, "tree-sitter query compilation failed — language may be ABI-incompatible");
            return None;
        }
    };
```

Add at the top of the file if not already present:

```rust
use tree_sitter::Query;
```

(Check if `Query` is already imported or used via full path.)

- [ ] **Step 2: Fix repo_map.rs**

In `crates/ripvec-core/src/repo_map.rs`, change line 127 from:

```rust
    let query = Query::new(&lang, query_str).ok()?;
```

to:

```rust
    let query = match Query::new(&lang, query_str) {
        Ok(q) => q,
        Err(e) => {
            tracing::warn!(ext, %e, "import query compilation failed — language may be ABI-incompatible");
            return None;
        }
    };
```

- [ ] **Step 3: Verify it compiles**

Run: `cargo check -p ripvec-core`

- [ ] **Step 4: Commit**

```bash
git add crates/ripvec-core/src/languages.rs crates/ripvec-core/src/repo_map.rs
git commit -m "fix: log warnings for tree-sitter query compilation failures"
```

---

### Task 3: Fix MSRV toolchain version in ci.yml

**Files:**
- Modify: `.github/workflows/ci.yml:67` — fix toolchain version

- [ ] **Step 1: Fix the toolchain version**

In `.github/workflows/ci.yml`, change line 67 from:

```yaml
      - uses: dtolnay/rust-toolchain@1.100.0
```

to:

```yaml
      - uses: dtolnay/rust-toolchain@1.88.0
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "fix: use actual MSRV 1.88.0 in CI (not non-existent 1.100.0)"
```

---

### Task 4: Expand release.yml to 5-target matrix

**Files:**
- Modify: `.github/workflows/release.yml` — rewrite build matrix

- [ ] **Step 1: Read the current release.yml**

Read `.github/workflows/release.yml` to understand the current structure.

- [ ] **Step 2: Replace the build job's matrix**

Replace the entire `build` job in `.github/workflows/release.yml` with this expanded version:

```yaml
  build:
    name: Build (${{ matrix.name }})
    needs: [prepare]
    if: needs.prepare.outputs.should_release == 'true'
    strategy:
      fail-fast: false
      matrix:
        include:
          # CPU-only Linux x86_64
          - name: linux-x86_64
            target: x86_64-unknown-linux-gnu
            os: ubuntu-latest
            features: ""
            suffix: ""
            container: ""
            apt: libopenblas-dev

          # CUDA Linux x86_64
          - name: linux-x86_64-cuda
            target: x86_64-unknown-linux-gnu
            os: ubuntu-latest
            features: "--features cuda"
            suffix: "-cuda"
            container: nvidia/cuda:13.1.0-devel-ubuntu24.04
            apt: ""

          # CPU-only Linux ARM64 (Graviton)
          - name: linux-arm64
            target: aarch64-unknown-linux-gnu
            os: ubuntu-24.04-arm
            features: ""
            suffix: ""
            container: ""
            apt: libopenblas-dev

          # CUDA Linux ARM64
          - name: linux-arm64-cuda
            target: aarch64-unknown-linux-gnu
            os: ubuntu-24.04-arm
            features: "--features cuda"
            suffix: "-cuda"
            container: nvidia/cuda:13.1.0-devel-ubuntu24.04
            apt: ""

          # macOS ARM64 (Apple Silicon — Metal + MLX auto-enabled)
          - name: macos-arm64
            target: aarch64-apple-darwin
            os: macos-latest
            features: ""
            suffix: ""
            container: ""
            apt: ""

    runs-on: ${{ matrix.os }}
    container: ${{ matrix.container || null }}
    steps:
      - uses: actions/checkout@v6
        with:
          ref: ${{ needs.prepare.outputs.version }}

      - uses: dtolnay/rust-toolchain@stable

      - uses: Swatinem/rust-cache@v2
        with:
          key: release-${{ matrix.name }}

      - name: Install system dependencies
        if: matrix.apt != ''
        run: |
          apt-get update && apt-get install -y ${{ matrix.apt }}

      - name: Build release binaries
        run: cargo build --release --workspace ${{ matrix.features }}

      - name: Package binaries
        run: |
          tag="${{ needs.prepare.outputs.version }}"
          staging="ripvec-${tag}-${{ matrix.target }}${{ matrix.suffix }}"
          mkdir -p "$staging"
          cp target/release/ripvec "$staging/" 2>/dev/null || true
          cp target/release/ripvec-mcp "$staging/" 2>/dev/null || true
          tar czf "${staging}.tar.gz" "$staging"
          echo "ASSET=${staging}.tar.gz" >> "$GITHUB_ENV"

      - name: Upload artifact
        uses: actions/upload-artifact@v7
        with:
          name: ${{ matrix.name }}
          path: ${{ env.ASSET }}
```

Key changes from original:
- 5 targets instead of 2
- `container` field for CUDA builds (empty string for non-container builds, handled by `${{ matrix.container || null }}`)
- `features` field for `--features cuda`
- `suffix` field for `-cuda` in archive name
- `name` field for human-readable job names and artifact dedup
- `apt` install uses plain `apt-get` (not `sudo apt-get`) because container jobs run as root
- Removed `sudo` from apt-get — containers run as root, and non-container jobs on GitHub runners have passwordless sudo. Use a conditional: inside containers `apt-get` directly, outside containers `sudo apt-get`.

- [ ] **Step 3: Fix the apt-get sudo issue**

The apt-get step needs to handle both container (root, no sudo) and non-container (needs sudo) contexts. Replace the install step with:

```yaml
      - name: Install system dependencies
        if: matrix.apt != ''
        run: |
          if command -v sudo &>/dev/null; then
            sudo apt-get update && sudo apt-get install -y ${{ matrix.apt }}
          else
            apt-get update && apt-get install -y ${{ matrix.apt }}
          fi
```

- [ ] **Step 4: Verify the YAML is valid**

Run: `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/release.yml'))" && echo "YAML valid"`

(Or just eyeball the indentation carefully.)

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/release.yml
git commit -m "feat: expand release builds to 5 targets (ARM Linux, CUDA x86+ARM, macOS)"
```

---

### Task 5: Verify full CI suite locally

**Files:** None (verification only)

- [ ] **Step 1: Format check**

Run: `cargo fmt --check`
Expected: clean

- [ ] **Step 2: Clippy**

Run: `cargo clippy --all-targets -- -D warnings`
Expected: zero warnings

- [ ] **Step 3: Full test suite**

Run: `cargo test -p ripvec-core --lib 2>&1 | tail -5`
Expected: all tests pass (including the 13 previously failing tree-sitter tests)

Run: `cargo test -p ripvec -p ripvec-mcp 2>&1 | tail -5`
Expected: all pass

- [ ] **Step 4: Commit any fixes**

If any issues found, fix and commit:

```bash
git commit -m "fix: address issues from CI verification"
```
