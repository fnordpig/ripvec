# CI Fixes + Multi-Architecture Release Builds

**Date**: 2026-04-06
**Status**: Approved

## Problem

1. CI is broken on main: MSRV job uses non-existent Rust 1.100.0, Clippy has a lint failure, 13 tree-sitter tests fail due to ABI version mismatch
2. Release workflow only builds 2 targets (x86_64 Linux CPU, aarch64 macOS). Missing ARM Linux (Graviton), CUDA-enabled builds for both x86_64 and aarch64

## Solution

Fix all CI failures and expand the release matrix to 5 targets including pre-built CUDA binaries.

## Part 1: Fix CI Failures

### Tree-sitter ABI mismatch (13 test failures)

`tree-sitter` runtime is `0.24.7` (supports grammar ABI 13-14). Three grammar crates (`tree-sitter-rust` 0.24.2, `tree-sitter-javascript` 0.25.0, `tree-sitter-c` 0.24.1) emit ABI version 15. When `Query::new()` is called with an ABI-15 language, it returns `Err`, which is silently swallowed by `.ok()?` in `languages.rs:130` and `repo_map.rs:127`.

**Fix**: Bump `tree-sitter` from `"0.24"` to `"0.26"` in workspace Cargo.toml (supports ABI 15). May require bumping some grammar crate versions for API compatibility. Change `.ok()?` to log warnings via `tracing::warn!` so ABI mismatches are immediately visible in future.

### MSRV job

`ci.yml` specifies `@1.100.0` which doesn't exist. Change to `@1.88.0` (the actual `rust-version` in Cargo.toml).

### Clippy lint

"unneeded return" error on main. Fix the code (find and remove the redundant return statement).

## Part 2: Multi-Architecture Release Matrix

### Build targets

| Target | Runner | Container | Features | Archive suffix |
|--------|--------|-----------|----------|----------------|
| `x86_64-unknown-linux-gnu` | `ubuntu-latest` | — | CPU (default) | `x86_64-unknown-linux-gnu` |
| `x86_64-unknown-linux-gnu` + CUDA | `ubuntu-latest` | `nvidia/cuda:13.1.0-devel-ubuntu24.04` | `cuda` | `x86_64-unknown-linux-gnu-cuda` |
| `aarch64-unknown-linux-gnu` | `ubuntu-24.04-arm` | — | CPU (default) | `aarch64-unknown-linux-gnu` |
| `aarch64-unknown-linux-gnu` + CUDA | `ubuntu-24.04-arm` | `nvidia/cuda:13.1.0-devel-ubuntu24.04` | `cuda` | `aarch64-unknown-linux-gnu-cuda` |
| `aarch64-apple-darwin` | `macos-latest` | — | Metal+MLX (auto) | `aarch64-apple-darwin` |

### CUDA container strategy

CUDA builds run inside `nvidia/cuda:13.1.0-devel-ubuntu24.04` container on GitHub-hosted runners. The container provides the CUDA toolkit (nvcc, cuBLAS headers) needed at compile time. Users need CUDA 13 runtime on their machines.

The `nvidia/cuda` images are multi-arch (`linux/amd64` + `linux/arm64`), so the same image works on both `ubuntu-latest` (x86) and `ubuntu-24.04-arm`.

CUDA builds use `cargo build --release --workspace --features cuda`.

### CPU Linux builds

Both x86_64 and aarch64 CPU builds need `libopenblas-dev` for the CPU BLAS backend. Install via `apt-get` on the runner (or in the container for CUDA builds, if openblas is also needed there — CUDA builds don't need openblas since they use cuBLAS).

### Archive naming

`ripvec-v{VERSION}-{TARGET}.tar.gz` where TARGET includes the `-cuda` suffix for CUDA builds.

### Unchanged

- `publish` job: publishes source to crates.io from Ubuntu, unchanged
- `gpu.yml`: self-hosted runner tests, separate concern
- `test.yml`: no workflow changes (tree-sitter fix is code, not CI)
- `ci.yml`: only MSRV toolchain version fix

## Scope

1. Bump `tree-sitter` to `"0.26"`, update grammar crate versions if needed
2. Add `tracing::warn!` for query compilation failures in `languages.rs` and `repo_map.rs`
3. Fix MSRV toolchain version in `ci.yml`
4. Fix clippy "unneeded return" in code
5. Expand release.yml matrix to 5 targets (2 Linux CPU, 2 Linux CUDA, 1 macOS)
6. Add CUDA container configuration to release.yml
