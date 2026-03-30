---
name: general-benchmarking
description: This skill should be used when benchmarking ripvec throughput on any backend, running performance comparisons, evaluating recall/quality at different layer counts, using bench.py, or doing regression testing. Also use when the user mentions "bench.py", "throughput", "benchmark", "recall", "regression test", "batch size sweep", or "performance comparison".
---

# ripvec Benchmarking Guide

All benchmarking MUST use `scripts/bench/bench.py`. Never run ad-hoc Bash commands for performance measurements — bench.py handles thermals, cooldown, result tracking, and reproducibility.

## bench.py Core Usage

```bash
# Throughput benchmark (single config)
uv run scripts/bench/bench.py --configs mps --no-build --layers 22

# Compare multiple configurations
uv run scripts/bench/bench.py --configs mps compute cpu --no-build --layers 3 22

# Batch size sweep
uv run scripts/bench/bench.py --configs mps --batch-sizes 32 64 128 --no-build

# Correctness validation (top-3 rankings must match across configs)
uv run scripts/bench/bench.py --validate --configs mps compute --no-build

# Recall evaluation (layer count quality vs 22-layer baseline)
uv run scripts/bench/bench.py --recall --layers 10 14 16 18 22 --no-build

# Thermal check only
uv run scripts/bench/bench.py --thermals
```

## Configuration Names

| Config | Env var | Backend | Notes |
|--------|---------|---------|-------|
| `mps` | (none) | Metal MPS FP16 | Default on macOS, uses Apple's pre-tuned GEMM |
| `compute` | `RIPVEC_NO_MPS=1` | Metal compute | Native simdgroup GEMM, zero MPS transitions |
| `cpu` | `--device cpu` | CPU (Accelerate/OpenBLAS) | BLAS-backed, works everywhere |
| `q8` | `RIPVEC_Q8=1` | Metal INT8 | block_q8_0 quantized weights |

## Thermal Monitoring

bench.py shows live thermals during each run:
```
[mps-22L] 5s speed=100% gpu=99% mem=12.5GB
```

- `speed`: CPU speed limit (100% = full speed, <100% = thermally throttled)
- `gpu`: GPU utilization via ioreg AGXAccelerator
- `mem`: System memory via vm_stat

**Trust numbers only when `speed=100%`.** Thermally throttled runs produce inconsistent results. bench.py automatically cools between runs.

## Corpus Selection

| Corpus | Path | Chunks | Use for |
|--------|------|--------|---------|
| ripvec code | `.` | ~1000 | Quick iteration, traces, development |
| Flask | `tests/corpus/code/flask` | ~2383 | Production benchmarks (default) |
| Custom | `--corpus /path` | varies | Specific testing |

Flask is the standard for commit messages and PR numbers. ripvec code is for fast iteration.

## Regression Testing Protocol

Before committing any backend change:

1. **Record baseline**: `uv run scripts/bench/bench.py --configs mps --no-build --layers 3`
2. Make the change, `cargo build --release`
3. **Verify**: `uv run scripts/bench/bench.py --configs mps --no-build --layers 3`
4. Must match within 5%. If adding new kernels, MPS regression is possible (see metal-debugging skill)

## Recall Evaluation

ModernBERT concentrates semantic quality in the final layers. Layer shedding results:

| Layers | Recall@10 vs 22L |
|--------|------------------|
| 10 | ~11% (garbage) |
| 14 | ~9% (garbage) |
| 22 | 100% (baseline) |

**Cannot cut layers for quality.** All 22 are required.

## Extending bench.py

Need a new parameter? **Extend bench.py directly** — add to `expand_configs()` and wire through `run_benchmark()`. Results auto-save to `scripts/bench/results/*.json` for historical tracking.

## Interpreting Results

Healthy throughput ranges (Flask corpus, M2 Max, 22 layers):

| Backend | Expected | Alarm if below |
|---------|----------|----------------|
| Metal MPS FP16 | 70-75/s | <60/s |
| Metal compute FP16 | 45-50/s | <35/s |
| CPU Accelerate | 70-75/s | <60/s |
| INT8 block_q8_0 | 30-35/s | <25/s |
| BGE-small MPS | 340-360/s | <300/s |
