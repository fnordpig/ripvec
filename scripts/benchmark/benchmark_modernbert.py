#!/usr/bin/env -S uv run --with sentence-transformers --with torch --python 3.12
"""
Benchmark modernbert-embed-base and bge-small-en-v1.5 on ripvec's own codebase.

Usage:
    uv run scripts/benchmark-modernbert.py

Requires: uv (https://docs.astral.sh/uv/)
Downloads models on first run (~500MB each).
"""

import os
import sys
import time
from pathlib import Path

from sentence_transformers import SentenceTransformer


def collect_chunks(root: Path, max_files: int = 50) -> list[dict]:
    """Collect code chunks from .rs, .py, .ts, .js, .go files."""
    chunks = []
    extensions = {".rs", ".py", ".ts", ".js", ".go", ".java", ".c", ".cpp", ".h"}
    for path in sorted(root.rglob("*")):
        if path.suffix not in extensions:
            continue
        if any(
            p in path.parts for p in ["target", "node_modules", ".git", "__pycache__"]
        ):
            continue
        try:
            text = path.read_text(errors="replace")
        except Exception:
            continue
        rel = str(path.relative_to(root))
        # Simple chunking: split on double newlines, take chunks > 50 chars
        for i, block in enumerate(text.split("\n\n")):
            block = block.strip()
            if len(block) > 50:
                chunks.append({"path": rel, "idx": i, "content": block[:2000]})
        if len(chunks) > 500:
            break
    return chunks


def benchmark_model(
    model_name: str,
    query_prefix: str,
    doc_prefix: str,
    chunks: list[dict],
    queries: list[str],
    top_k: int = 10,
) -> dict:
    """Embed corpus + queries, compute Recall@K at various MRL dims."""
    print(f"\n{'=' * 60}")
    print(f"Model: {model_name}")
    print(f"{'=' * 60}")

    t0 = time.time()
    model = SentenceTransformer(model_name, trust_remote_code=True)
    load_time = time.time() - t0
    print(f"  Load: {load_time:.1f}s")

    # Embed corpus
    docs = [f"{doc_prefix}{c['content']}" for c in chunks]
    t1 = time.time()
    doc_embeddings = model.encode(docs, show_progress_bar=True, batch_size=64)
    embed_time = time.time() - t1
    throughput = len(docs) / embed_time
    full_dim = doc_embeddings.shape[1]
    print(
        f"  Embed: {len(docs)} chunks in {embed_time:.1f}s ({throughput:.0f}/s), dim={full_dim}"
    )

    # Embed queries
    q_texts = [f"{query_prefix}{q}" for q in queries]
    q_embeddings = model.encode(q_texts)

    # Compute Recall@K at various MRL dims
    import numpy as np

    results = {}
    mrl_dims = [64, 128, 256, 384, 512, full_dim]
    mrl_dims = [d for d in mrl_dims if d <= full_dim]

    # Full-dim reference ranking
    sims_full = q_embeddings @ doc_embeddings.T
    ref_topk = [np.argsort(-sims_full[i])[:top_k].tolist() for i in range(len(queries))]

    print(f"\n  MRL Recall@{top_k} (vs full {full_dim}-dim):")
    for dim in mrl_dims:
        # Truncate + L2 renorm
        d_trunc = doc_embeddings[:, :dim]
        d_norms = np.linalg.norm(d_trunc, axis=1, keepdims=True)
        d_trunc = d_trunc / np.maximum(d_norms, 1e-12)

        q_trunc = q_embeddings[:, :dim]
        q_norms = np.linalg.norm(q_trunc, axis=1, keepdims=True)
        q_trunc = q_trunc / np.maximum(q_norms, 1e-12)

        sims = q_trunc @ d_trunc.T
        recalls = []
        for i in range(len(queries)):
            trunc_topk = np.argsort(-sims[i])[:top_k].tolist()
            overlap = len(set(ref_topk[i]) & set(trunc_topk))
            recalls.append(overlap / top_k)
        avg_recall = sum(recalls) / len(recalls)
        marker = " (ref)" if dim == full_dim else " ***" if avg_recall >= 0.8 else ""
        print(f"    dims={dim:>4}: Recall@{top_k}={avg_recall:.2f}{marker}")
        results[dim] = avg_recall

    # Show top-1 results for each query at full dim
    print(f"\n  Top-1 results (full {full_dim}-dim):")
    for i, q in enumerate(queries):
        top_idx = ref_topk[i][0]
        score = sims_full[i][top_idx]
        chunk = chunks[top_idx]
        print(f'    Q: "{q[:60]}..."')
        print(f"       → {chunk['path']}:{chunk['idx']} (sim={score:.3f})")

    return {
        "model": model_name,
        "dim": full_dim,
        "throughput": throughput,
        "load_time": load_time,
        "mrl_recalls": results,
    }


def main():
    root = Path(__file__).parent.parent
    print(f"Codebase: {root}")

    chunks = collect_chunks(root)
    print(
        f"Collected {len(chunks)} chunks from {len(set(c['path'] for c in chunks))} files"
    )

    queries = [
        "error handling with thiserror and anyhow",
        "tree-sitter chunking and AST parsing",
        "Metal GPU kernel dispatch and command buffer encoding",
        "cosine similarity search and top-K ranking",
        "file watcher for incremental reindex",
        "PageRank structural overview of codebase",
        "GEMM matrix multiplication with simdgroup",
        "attention mask and softmax computation",
    ]

    # Benchmark both models
    results = []

    results.append(
        benchmark_model(
            "nomic-ai/modernbert-embed-base",
            query_prefix="search_query: ",
            doc_prefix="search_document: ",
            chunks=chunks,
            queries=queries,
        )
    )

    results.append(
        benchmark_model(
            "BAAI/bge-small-en-v1.5",
            query_prefix="",
            doc_prefix="",
            chunks=chunks,
            queries=queries,
        )
    )

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for r in results:
        print(f"\n{r['model']}:")
        print(
            f"  Dim: {r['dim']}, Throughput: {r['throughput']:.0f}/s, Load: {r['load_time']:.1f}s"
        )
        for dim, recall in sorted(r["mrl_recalls"].items()):
            print(f"  MRL@{dim}: {recall:.2f}")


if __name__ == "__main__":
    main()
