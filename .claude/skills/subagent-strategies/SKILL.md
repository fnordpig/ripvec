---
name: subagent-strategies
description: This skill should be used when planning how to divide work across subagents, choosing between opus/sonnet/haiku for tasks, dispatching parallel agents, writing effective agent prompts, or debugging agent failures. Also use when the user mentions "subagent", "background agent", "parallel agents", "opus vs sonnet", "agent prompt", or "dispatch agent".
---

# Subagent Strategies for ripvec Development

Proven patterns for leveraging opus, sonnet, and haiku agents effectively on GPU kernel development, benchmarking, and code exploration tasks.

## Model Selection Guide

| Task Type | Model | Why |
|-----------|-------|-----|
| **Architecture analysis** | Opus | Needs to reason about type flows, memory layouts, multi-file interactions |
| **Kernel correctness debugging** | Opus | Must trace index math line-by-line, compare two kernels, write unit tests |
| **Code exploration / search** | Explore agent | Built for multi-step grep/read workflows |
| **Mechanical refactoring** | Sonnet | Clear instructions, many files, repetitive changes |
| **Adding a dependency / flag** | Haiku | Simple Cargo.toml edit, CLI flag plumbing |
| **Benchmark comparison** | Sonnet | Run bench.py, parse output, report numbers |
| **llama.cpp research** | Opus | Reading unfamiliar C/Metal code, extracting architectural insights |
| **NomicBert/CodeRankEmbed removal** | Sonnet (parallel) | Two agents: one for CLI/docs, one for backend code |
| **Git bisect** | Sonnet | Mechanical: checkout, build, test, report |

## Prompt Patterns That Work

### Give Complete Context

**Bad**: "Fix the INT8 kernel correctness bug."
**Good**: Specify the symptom, what's been tried, exact files, test commands, and success criteria.

### Specify What NOT to Change

Always include a "DO NOT" section:
```
DO NOT:
- Change the FP16 kernel (gemm_f16w_f32a_kernel)
- Change the MPS dispatch path
- Modify architecture code (modern_bert.rs)
```

This prevents agents from "fixing" things by changing the wrong code.

### Include Verification Commands

```
Build and test:
  cargo build --release
  RIPVEC_Q8=1 ./target/release/ripvec "session" -n 3 tests/corpus/code/flask --format json --mode semantic -T 0.0
  # Top-3 names must match MPS reference
```

### Report Status Format

Ask agents to report: `DONE | DONE_WITH_CONCERNS | BLOCKED | NEEDS_CONTEXT`

## Parallel Agent Patterns

### Wave Pattern (Dependencies)

```
Wave 1 (parallel):
  [Sonnet] Task A: modify file X
  [Sonnet] Task B: modify file Y
  [Haiku]  Task C: add dependency

Wave 2 (after merge):
  [Opus] Task D: wire A+B+C together (touches all files)
```

Used for: Phase 4 Hybrid Search (3 parallel + 2 sequential).

### Worktree Isolation

Use `isolation: "worktree"` for agents that modify code:
- Each agent gets a clean copy of the repo
- No merge conflicts during parallel execution
- Cherry-pick or merge commits after completion
- Watch for different base commits — agents branch from HEAD at dispatch time

### Background Research + Foreground Implementation

Dispatch research agents in background while implementing in foreground:
```
Background: [Opus] "Analyze llama.cpp GEMM kernel architecture"
Foreground: Write the initial kernel based on known patterns
When background completes: Incorporate research findings
```

## Common Agent Failures

### Agent Changes Wrong Files

**Cause**: Prompt didn't specify file boundaries clearly.
**Fix**: List exact files to modify AND files NOT to modify.

### Agent Puts Methods in Wrong impl Block

**Cause**: Rust's `impl Trait for Struct` vs `impl Struct` — agent doesn't know which block is which.
**Fix**: Specify: "Add as inherent methods on MetalDriver (NOT on the Driver trait impl)."

### Agent Uses Wrong Crate

**Cause**: Agent uses `half::f16` but the crate isn't imported.
**Fix**: Tell the agent what crates are available, or say "use only stdlib types."

### Agent's Worktree Diverged

**Cause**: Worktree branched from an old commit; agent's changes don't apply to current main.
**Fix**: Cherry-pick with `--3way` or manual merge. Check `git log worktree-agent-xxx -3` before merging.

### GPU Contention from Multiple Agents

**Cause**: Multiple background agents running ripvec on Metal simultaneously.
**Fix**: Agents that test on GPU should use `--device cpu` or small corpora. Only one Metal workload at a time.

## Cost-Effective Patterns

### Haiku for Plumbing

Adding a CLI flag, Cargo dependency, or trait method signature costs ~$0.02 with Haiku vs ~$0.50 with Opus. Use Haiku for mechanical changes with clear instructions.

### Sonnet for Parallel Mechanical Work

Two Sonnet agents in parallel costs ~$1.00 and completes in ~3 minutes. Same work sequentially in Opus costs ~$2.00 and takes ~6 minutes.

### Opus Only for Reasoning

Reserve Opus for tasks requiring multi-step reasoning: kernel debugging, architecture analysis, cross-file correctness verification. If the task is "apply this pattern to these 4 call sites," use Sonnet.

## When NOT to Use Subagents

- **Quick single-file edits**: Faster to do inline than to write the prompt
- **Exploratory debugging**: The feedback loop is too tight for agent latency
- **Tasks requiring your current context**: Agents start fresh — rebuilding context costs more than doing it yourself
- **GPU-intensive testing in parallel**: Metal can only handle one heavy workload; parallel agents will contend

## bench.py Integration

Agents that benchmark MUST use bench.py:
```
Include in agent prompt:
"Benchmark with: uv run scripts/bench/bench.py --configs mps --no-build --layers 3
Do NOT run ad-hoc Bash benchmarks."
```

This ensures thermal handling, result tracking, and reproducibility.
