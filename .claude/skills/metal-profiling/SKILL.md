---
name: metal-profiling
description: This skill should be used when profiling Metal GPU workloads with tracemeld and xctrace, recording Metal System Traces, analyzing GPU compute utilization, diagnosing MPS encoder transition overhead, or comparing MPS vs compute kernel efficiency. Also use when the user mentions "tracemeld", "xctrace", "Metal System Trace", "GPU utilization", "encoder transitions", "GPU idle", "driver overhead", or "profile Metal".
---

# Metal GPU Profiling with tracemeld

Record Metal System Traces with xctrace, import into tracemeld, analyze bottlenecks, and diff across iterations.

## Recording Traces

```bash
xcrun xctrace record \
  --template 'Metal System Trace' \
  --output /tmp/trace.trace \
  --time-limit 15s \
  --no-prompt \
  --target-stdout /dev/null \
  --launch -- ./target/release/ripvec "session" -n 1 . --layers 3 --mode semantic -T 0.0
```

**Critical rules:**
- Use `--env VAR=VALUE` for environment variables — parent shell env does NOT propagate
- Use ripvec's own code (`.`, ~1000 chunks) with layers=3 — keeps traces <500MB
- Flask corpus at 22 layers produces 8GB+ traces that fail to export
- Metal System Trace template requires GPU events to export — compute-only (zero MPS) paths may fail with "Document Missing Template Error" on large captures

## tracemeld Analysis Workflow

```
import_profile(source: "/tmp/trace.trace", format: "xctrace")
profile_summary(group_by: "lane")        # GPU compute %, driver %, idle %
bottleneck(dimension: "wall_ms", top_n: 5) # most impactful spans
hotpaths(dimension: "wall_ms", top_n: 10)  # critical call chains
starvations(min_idle_pct: 3)               # lane idle analysis
save_baseline(name: "before-change", checkpoint: "before", task: "description")
```

After making changes:
```
# Record new trace, import
diff_profile(baseline: "before-change", dimension: "wall_ms")
# Shows: headline % change, regressions, improvements, new/removed stacks
```

## Key Metrics by Lane

| Lane | What it measures | Healthy range |
|------|-----------------|---------------|
| gpu-compute | GPU ALU time (our kernels + MPS internals) | 85-98% |
| driver | Metal driver command processing | 0.5-3% |
| gpu-vertex | Window compositor (not us) | 2-10% (ignore) |
| gpu-fragment | Window compositor (not us) | 2-10% (ignore) |

**GPU idle** = wall_time - gpu_compute - driver. This is encoder transition gaps.

## MPS vs Compute Profile Signatures

**MPS path** (default):
- ~3500 compute spans (many small: element-wise between MPS GEMMs)
- 88 encoder transitions per forward pass (one per MPS GEMM call)
- 7-10% GPU idle from transitions
- 2-3% driver overhead

**Compute path** (RIPVEC_NO_MPS=1):
- ~1600 compute spans (zero MPS transitions, one persistent encoder)
- 0.5-1% driver overhead
- 3-6% GPU idle
- But per-FLOP throughput is lower than MPS

## Diagnosing Specific Issues

### "Is it the kernel or the dispatch?"
No-op kernel test: replace kernel body with `return;`. If still slow, dispatch overhead dominates. If fast, kernel compute is the bottleneck.

### "Is it the loads or the compute?"
Single K-tile test: add `if (loop_k >= 32) return;` in the kernel. If one K-tile takes as long as expected for 1/24th of the work, loads dominate. If it's nearly as slow as the full kernel, the loads ARE the bottleneck.

### "Why does adding a kernel slow everything?"
Check pipeline state regression. Record MPS throughput before and after adding the kernel source. If MPS drops, it's the `device half*` pipeline state bug (see metal-debugging skill).

### "Which simdgroup is wrong?"
For correctness issues, the NL1 bug pattern: if only simdgroups 0+1 produce correct output but 2+3 produce garbage, the cooperative B load doesn't cover all 64 N-rows.

## Baseline Management

Save baselines at key milestones:
```
save_baseline(name: "gemm-mfa-wrapper", checkpoint: "before", task: "native simdgroup migration")
save_baseline(name: "gemm-native-v1", checkpoint: "after", task: "iteration 1")
```

Always diff against the original baseline for total impact. Diff against the previous iteration to isolate each step's contribution.

## Additional Resources

- `references/tracemeld-cheatsheet.md` — Quick reference for all tracemeld MCP tool parameters
