---
name: general-debugging
description: This skill should be used when debugging any ripvec issue — incorrect search results, embedding quality problems, model loading failures, cache invalidation, performance regressions, or build failures. Also use when the user mentions "broken", "wrong results", "NaN", "hang", "crash", "regression", or "bisect".
---

# ripvec General Debugging Guide

Systematic approaches to diagnosing issues across all backends and components.

## Triage: What's Broken?

| Symptom | Component | First action |
|---------|-----------|--------------|
| Zero search results | Embeddings or ranking | Check `-T 0.0` (threshold), check `--mode semantic` |
| Wrong search results | Model quality | Compare `--fast` (BGE) vs default (ModernBERT) |
| NaN embeddings | Forward pass | Add NaN check after `embed_all()`, see metal-debugging skill |
| GPU hang | Metal driver | Test fewer layers, see metal-debugging skill |
| Slow throughput | Backend perf | Use bench.py, see general-benchmarking skill |
| Build failure | Compilation | `cargo check --workspace`, check feature flags |
| Cache stale | Index system | `--clear-cache` or check `v2-<model_slug>/` dir |

## Isolation Strategy

Always isolate variables one at a time:

1. **Backend**: `--device cpu` vs default (Metal). If CPU works but Metal doesn't, it's a GPU issue.
2. **Model**: `--fast` (BGE-small 12L) vs default (ModernBERT 22L). If BGE works, it's ModernBERT-specific.
3. **Corpus**: ripvec code (`.`) vs Flask corpus. If small works but large doesn't, it's a scaling issue.
4. **Corpus size**: Use a smaller corpus (e.g., a single file) to isolate embedding vs ranking issues. Compare `--mode semantic` vs `--mode hybrid` to isolate search mode problems.
5. **Batch size**: `-b 1` vs `-b 32`. If batch=1 works, it's a batching/padding issue.
6. **Mode**: `--mode semantic` vs `--mode keyword` vs `--mode hybrid`. Isolates embedding vs BM25 vs fusion.

## Git Bisect for Regressions

```bash
# Find which commit broke ModernBERT
git bisect start
git bisect bad HEAD
git bisect good <known_good_commit>
# For each bisect step:
cargo build --release && ./target/release/ripvec "test" -n 3 . --format json --mode semantic -T 0.0
```

## Embedding Quality Checks

Add after `embed_all()` to diagnose:

```rust
let nan_count = embeddings.iter().filter(|e| e.iter().any(|x| x.is_nan())).count();
let zero_count = embeddings.iter().filter(|e| e.iter().all(|x| *x == 0.0)).count();
let l2_norms: Vec<f32> = embeddings.iter()
    .map(|e| e.iter().map(|x| x * x).sum::<f32>().sqrt())
    .collect();
let avg_l2 = l2_norms.iter().sum::<f32>() / l2_norms.len() as f32;
eprintln!("[DEBUG] {nan_count} NaN, {zero_count} zero, avg L2={avg_l2:.4}");
```

Healthy: 0 NaN, 0 zero, avg L2 ≈ 1.0 (L2-normalized).

## Cache Issues

Cache dir: `~/.cache/ripvec/<project_hash>/v<VERSION>-<model_slug>/`

Version or model mismatch → stale cache. Fix: `--clear-cache` or delete the dir. The `v2-` prefix auto-invalidates when `MANIFEST_VERSION` is bumped.

## Subagent Strategy for Debugging

For complex bugs, dispatch specialized agents:

- **Bisect agent (Sonnet)**: Git bisect across commits to find the regression point
- **Trace agent (Opus)**: Deep analysis with tracemeld of profiling data
- **Correctness agent (Sonnet)**: Unit test at model-scale dimensions (128×768×768)
- **Architecture analysis (Opus)**: Compare code flow in two kernels line-by-line

See subagent-strategies skill for detailed dispatch patterns.

## Common Gotchas

- **`RIPVEC_NO_MPS=1` produces empty results**: The FP32 gemm() dispatch sends float* to a kernel expecting half* — a type mismatch, not a kernel bug.
- **Model download on first run**: ModernBERT weights are ~570MB. First run downloads from HuggingFace. Set `HF_HOME` to control cache location.
- **`prepare_batch_unpadded` vs `prepare_batch`**: Unpadded path concatenates tokens flat (total_tokens < padded_tokens). Ensure all subsequent operations use the correct token count.
- **`ensure_fp16` only runs at weight load time**: If a weight tensor's `.fp16` field is None during inference, the FP16 pre-conversion was skipped. Check the load function.
