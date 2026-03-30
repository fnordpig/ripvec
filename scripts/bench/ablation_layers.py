#!/usr/bin/env python3
"""Per-layer ablation: drop each layer individually, measure Recall@10.

Usage:
    uv run scripts/bench/ablation_layers.py
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

# Reuse recall_eval machinery
sys.path.insert(0, str(Path(__file__).resolve().parent))
from recall_eval import load_baseline, run_queries, recall_at_k, QUERIES

NUM_LAYERS = 22


def main() -> int:
    print("Loading baselines...")
    sem_baseline = load_baseline("semantic")
    hyb_baseline = load_baseline("hybrid")

    results: list[dict] = []

    for layer in range(NUM_LAYERS):
        skip_arg = f"--skip-layers {layer}"
        extra = skip_arg.split()
        print(f"\n--- Ablating layer {layer} ---")

        sem_candidate = run_queries("semantic", extra)
        hyb_candidate = run_queries("hybrid", extra)

        sem_recalls = []
        hyb_recalls = []
        for q in QUERIES:
            sem_recalls.append(recall_at_k(sem_baseline[q], sem_candidate.get(q, [])))
            hyb_recalls.append(recall_at_k(hyb_baseline[q], hyb_candidate.get(q, [])))

        sem_avg = sum(sem_recalls) / len(sem_recalls)
        hyb_avg = sum(hyb_recalls) / len(hyb_recalls)

        results.append(
            {
                "layer": layer,
                "semantic_recall": round(sem_avg * 100, 1),
                "hybrid_recall": round(hyb_avg * 100, 1),
                "sem_loss": round((1 - sem_avg) * 100, 1),
                "hyb_loss": round((1 - hyb_avg) * 100, 1),
            }
        )
        print(f"  Semantic: {sem_avg:.1%}  Hybrid: {hyb_avg:.1%}")

    # Sort by semantic recall (worst first = most important layer)
    results.sort(key=lambda r: r["semantic_recall"])

    print("\n" + "=" * 70)
    print("  Per-Layer Ablation Results (sorted by semantic impact)")
    print("=" * 70)
    print(
        f"{'Layer':>6} {'Sem Recall':>12} {'Sem Loss':>10} {'Hyb Recall':>12} {'Hyb Loss':>10}"
    )
    print("-" * 70)
    for r in results:
        sem_ok = "✓" if r["semantic_recall"] >= 90 else "✗"
        hyb_ok = "✓" if r["hybrid_recall"] >= 98 else "✗"
        print(
            f"  {r['layer']:>4}   {sem_ok} {r['semantic_recall']:>6.1f}%"
            f"    {r['sem_loss']:>6.1f}%"
            f"   {hyb_ok} {r['hybrid_recall']:>6.1f}%"
            f"    {r['hyb_loss']:>6.1f}%"
        )
    print("=" * 70)

    # Save results
    out_path = (
        Path(__file__).resolve().parent / "recall_baselines" / "ablation_layers.json"
    )
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved: {out_path}")

    # Suggest safe-to-skip layers
    safe = [
        r for r in results if r["semantic_recall"] >= 90 and r["hybrid_recall"] >= 98
    ]
    if safe:
        safe_layers = sorted(r["layer"] for r in safe)
        print(f"\nLayers safe to skip (≥90% sem, ≥98% hyb): {safe_layers}")
    else:
        print("\nNo layers can be individually skipped within recall budget.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
