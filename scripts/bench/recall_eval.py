#!/usr/bin/env python3
"""Recall@10 evaluation for inference optimizations.

Compares search results against stored baselines to measure recall loss.

Usage:
    # Capture baseline (run once before optimizations):
    uv run scripts/bench/recall_eval.py --capture-baseline

    # Evaluate a config against baseline:
    uv run scripts/bench/recall_eval.py --eval --extra-args "--svd-rank auto"
    uv run scripts/bench/recall_eval.py --eval --extra-args "--skip-layers 6,7,13,14"
    uv run scripts/bench/recall_eval.py --eval --extra-args "--svd-rank auto --prune-ratio 0.3 --skip-layers 6,7,13,14"
"""

from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BINARY = REPO_ROOT / "target" / "release" / "ripvec"
CORPUS = REPO_ROOT / "tests" / "corpus" / "code" / "flask"
BASELINES_DIR = Path(__file__).resolve().parent / "recall_baselines"

QUERIES = [
    "error handling middleware",
    "database connection pooling",
    "request routing dispatch",
    "session management cookies",
    "template rendering jinja",
]


def run_queries(mode: str, extra_args: list[str] | None = None) -> dict[str, list[str]]:
    """Run all queries, return {query: [file_paths]} for top-10."""
    results: dict[str, list[str]] = {}
    for q in QUERIES:
        cmd = [
            str(BINARY),
            q,
            str(CORPUS),
            "-n",
            "10",
            "--format",
            "json",
            "--mode",
            mode,
        ]
        if extra_args:
            cmd.extend(extra_args)
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        paths = []
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                p = obj.get("file_path") or obj.get("path") or obj.get("file") or ""
                if p:
                    paths.append(p)
            except json.JSONDecodeError:
                continue
        results[q] = paths[:10]
    return results


def capture_baseline() -> None:
    """Capture baselines for both modes."""
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    for mode in ("semantic", "hybrid"):
        print(f"Capturing {mode} baseline...")
        results = run_queries(mode)
        path = BASELINES_DIR / f"baseline_{mode}.json"
        path.write_text(json.dumps(results, indent=2))
        print(f"  Saved: {path}")
        for q, paths in results.items():
            print(f"  {q}: {len(paths)} results")


def load_baseline(mode: str) -> dict[str, list[str]]:
    """Load stored baseline."""
    path = BASELINES_DIR / f"baseline_{mode}.json"
    if not path.exists():
        print(f"Error: baseline not found: {path}", file=sys.stderr)
        print("Run with --capture-baseline first.", file=sys.stderr)
        sys.exit(1)
    return json.loads(path.read_text())


def recall_at_k(baseline: list[str], candidate: list[str], k: int = 10) -> float:
    """Compute Recall@k = |intersection| / min(k, |baseline|)."""
    b_set = set(baseline[:k])
    c_set = set(candidate[:k])
    if not b_set:
        return 1.0
    return len(b_set & c_set) / len(b_set)


def evaluate(extra_args_str: str) -> None:
    """Evaluate a config against baselines."""
    extra_args = extra_args_str.split() if extra_args_str else []
    print(f"Config: {extra_args_str or '(baseline)'}")
    print(f"{'=' * 70}")

    for mode in ("semantic", "hybrid"):
        baseline = load_baseline(mode)
        candidate = run_queries(mode, extra_args)

        recalls = []
        print(f"\n  {mode.upper()} Recall@10:")
        for q in QUERIES:
            b = baseline.get(q, [])
            c = candidate.get(q, [])
            r = recall_at_k(b, c)
            recalls.append(r)
            status = "✓" if r >= (0.9 if mode == "semantic" else 0.98) else "✗"
            print(
                f"    {status} {q}: {r:.0%} ({len(set(b[:10]) & set(c[:10]))}/{len(set(b[:10]))})"
            )

        avg = sum(recalls) / len(recalls) if recalls else 0
        target = 90 if mode == "semantic" else 98
        status = "PASS" if avg * 100 >= target else "FAIL"
        print(f"  Average: {avg:.1%} (target: ≥{target}%) [{status}]")


def main() -> int:
    p = argparse.ArgumentParser(description="Recall@10 evaluation")
    p.add_argument(
        "--capture-baseline", action="store_true", help="Capture baseline results"
    )
    p.add_argument("--eval", action="store_true", help="Evaluate against baseline")
    p.add_argument(
        "--extra-args", default="", help="Extra ripvec args (e.g., '--svd-rank auto')"
    )
    args = p.parse_args()

    if args.capture_baseline:
        capture_baseline()
        return 0
    if args.eval:
        evaluate(args.extra_args)
        return 0
    p.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
