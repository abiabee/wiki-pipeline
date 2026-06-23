"""Step 3 — Build the similarity graph from embeddings.

Inputs:
  - output/embeddings/vectors.npy   (N x D, L2-normalized, float32)
  - output/embeddings/index.json    leaf_id -> row mapping
  - input/leaves/file-*.json        used only to attach human-readable titles

Strategy:
  - Vectors are unit-norm, so cosine similarity = V @ V.T (a single matmul).
  - For each leaf we keep its top-K nearest neighbors above `--min-score`.
  - We emit two artifacts with different shapes for different consumers:

      edges.json    : undirected, deduplicated edge list. One entry per pair.
                      Downstream steps (rule-based edges, clustering) will
                      consume this.
      neighbors.json: directed top-K per leaf, ranks preserved. Used for
                      "related pages" UI and graph traversal.

  - `--mutual-only` requires both A and B to have each other in their
    top-K before emitting the edge. Off by default; turn on when noise
    becomes a problem.

Run:
  python scripts/build_similarity_graph.py
  python scripts/build_similarity_graph.py --top-k 10 --min-score 0.5
  python scripts/build_similarity_graph.py --mutual-only
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from rich.table import Table

from _common import (
    EDGES_FILE,
    EMBEDDINGS_INDEX,
    EMBEDDINGS_VECTORS,
    GRAPH_DIR,
    LEAVES_DIR,
    NEIGHBORS_FILE,
    ROOT,
    console,
)


DEFAULT_TOP_K = 8
DEFAULT_MIN_SCORE = 0.55


def _load_embeddings() -> Tuple[List[str], np.ndarray, Dict[str, Any]]:
    """Returns (leaf_ids in row order, vectors, raw index dict)."""
    if not EMBEDDINGS_VECTORS.exists() or not EMBEDDINGS_INDEX.exists():
        console.print(
            "[red]Embeddings not found.[/]\n"
            "Run [bold]python scripts/embed_leaves.py[/] first."
        )
        sys.exit(2)

    index = json.loads(EMBEDDINGS_INDEX.read_text(encoding="utf-8"))
    vectors = np.load(EMBEDDINGS_VECTORS)

    # Order leaf_ids by row so vectors[i] corresponds to leaf_ids[i].
    leaves = index.get("leaves", {})
    if len(leaves) != vectors.shape[0]:
        console.print(
            f"[red]Index ({len(leaves)} leaves) and vectors "
            f"({vectors.shape[0]} rows) disagree.[/]"
        )
        sys.exit(2)
    leaf_ids = [None] * vectors.shape[0]
    for leaf_id, entry in leaves.items():
        leaf_ids[entry["row"]] = leaf_id
    if any(lid is None for lid in leaf_ids):
        console.print("[red]Index has gaps in row numbers.[/]")
        sys.exit(2)

    return leaf_ids, vectors.astype(np.float32, copy=False), index


def _load_titles(leaf_ids: List[str]) -> Dict[str, str]:
    """Pull `embedding.title` (fallback: source.name) for each leaf."""
    titles: Dict[str, str] = {}
    for leaf_id in leaf_ids:
        path = LEAVES_DIR / f"{leaf_id}.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            titles[leaf_id] = leaf_id
            continue
        title = (
            (data.get("embedding") or {}).get("title")
            or (data.get("source") or {}).get("name")
            or leaf_id
        )
        titles[leaf_id] = str(title).strip() or leaf_id
    return titles


def _top_k_neighbors(
    sim: np.ndarray,
    top_k: int,
    min_score: float,
) -> List[List[Tuple[int, float]]]:
    """For each row, return list of (col_idx, score) sorted by score desc.

    Self-similarity (diagonal) is excluded. Anything below `min_score` is
    dropped, even if it would otherwise fit in the top-K window.
    """
    n = sim.shape[0]
    masked = sim.copy()
    np.fill_diagonal(masked, -np.inf)

    # argpartition is faster than full sort for large N; we still sort
    # the small top-K window for stable, ranked output.
    k = min(top_k, n - 1)
    # Indices of top-k by score, unordered within the slice:
    part = np.argpartition(-masked, kth=k - 1, axis=1)[:, :k]

    neighbors: List[List[Tuple[int, float]]] = []
    for i in range(n):
        idxs = part[i]
        scores = masked[i, idxs]
        order = np.argsort(-scores)
        ranked = [
            (int(idxs[o]), float(scores[o]))
            for o in order
            if scores[o] >= min_score
        ]
        neighbors.append(ranked)
    return neighbors


def _build_edges(
    leaf_ids: List[str],
    titles: Dict[str, str],
    neighbors: List[List[Tuple[int, float]]],
    mutual_only: bool,
) -> List[Dict[str, Any]]:
    """Convert directed top-K lists into a deduplicated undirected edge list.

    With `mutual_only=True`, an edge is only emitted when each side has the
    other in its top-K. This trims noise in exchange for graph density.
    """
    if mutual_only:
        in_top_k: List[set[int]] = [
            {j for j, _ in row} for row in neighbors
        ]

    seen: set[Tuple[int, int]] = set()
    edges: List[Dict[str, Any]] = []
    for i, row in enumerate(neighbors):
        for j, score in row:
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in seen:
                continue
            if mutual_only and i not in in_top_k[j]:
                continue
            seen.add((a, b))
            from_id = leaf_ids[a]
            to_id = leaf_ids[b]
            edges.append(
                {
                    "from": from_id,
                    "to": to_id,
                    "from_title": titles.get(from_id, from_id),
                    "to_title": titles.get(to_id, to_id),
                    "type": "similarity",
                    "score": round(float(score), 4),
                }
            )
    edges.sort(key=lambda e: e["score"], reverse=True)
    return edges


def _build_neighbors_payload(
    leaf_ids: List[str],
    titles: Dict[str, str],
    neighbors: List[List[Tuple[int, float]]],
) -> Dict[str, List[Dict[str, Any]]]:
    payload: Dict[str, List[Dict[str, Any]]] = {}
    for i, row in enumerate(neighbors):
        leaf_id = leaf_ids[i]
        payload[leaf_id] = [
            {
                "leaf_id": leaf_ids[j],
                "title": titles.get(leaf_ids[j], leaf_ids[j]),
                "score": round(float(score), 4),
                "rank": rank,
            }
            for rank, (j, score) in enumerate(row, start=1)
        ]
    return payload


def _print_summary(
    n: int,
    top_k: int,
    min_score: float,
    mutual_only: bool,
    edges: List[Dict[str, Any]],
    neighbors_payload: Dict[str, List[Dict[str, Any]]],
    titles: Dict[str, str],
) -> None:
    table = Table(title="Similarity graph summary", show_lines=False)
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_row("Leaves", str(n))
    table.add_row("Params", f"top_k={top_k}, min_score={min_score}, mutual_only={mutual_only}")
    table.add_row("Edges", str(len(edges)))
    avg_neighbors = (
        sum(len(v) for v in neighbors_payload.values()) / max(n, 1)
    )
    table.add_row("Avg neighbors / leaf", f"{avg_neighbors:.2f}")
    isolated = [lid for lid, ns in neighbors_payload.items() if not ns]
    table.add_row(
        "Isolated leaves",
        f"{len(isolated)}"
        + (f"  [yellow](first 3: {', '.join(isolated[:3])})[/]" if isolated else ""),
    )
    if edges:
        scores = [e["score"] for e in edges]
        table.add_row(
            "Score range (edges)",
            f"min={min(scores):.3f}  median={float(np.median(scores)):.3f}  max={max(scores):.3f}",
        )
    table.add_row("Edges file", str(EDGES_FILE.relative_to(ROOT)))
    table.add_row("Neighbors file", str(NEIGHBORS_FILE.relative_to(ROOT)))
    console.print(table)

    if edges:
        sample = Table(title="Top 5 edges", show_lines=False)
        sample.add_column("Score", justify="right")
        sample.add_column("From")
        sample.add_column("To")
        for e in edges[:5]:
            sample.add_row(
                f"{e['score']:.3f}",
                e["from_title"],
                e["to_title"],
            )
        console.print(sample)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Max neighbors kept per leaf (default {DEFAULT_TOP_K}).",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=DEFAULT_MIN_SCORE,
        help=f"Floor cosine score for an edge (default {DEFAULT_MIN_SCORE}).",
    )
    parser.add_argument(
        "--mutual-only",
        action="store_true",
        help="Only emit edges where both leaves are in each other's top-K.",
    )
    args = parser.parse_args(argv)

    leaf_ids, vectors, _index = _load_embeddings()
    titles = _load_titles(leaf_ids)

    console.print(
        f"Loaded [bold]{len(leaf_ids)}[/] vectors of dim {vectors.shape[1]}; "
        f"computing cosine similarity..."
    )
    sim = vectors @ vectors.T

    neighbors = _top_k_neighbors(sim, top_k=args.top_k, min_score=args.min_score)
    edges = _build_edges(
        leaf_ids=leaf_ids,
        titles=titles,
        neighbors=neighbors,
        mutual_only=args.mutual_only,
    )
    neighbors_payload = _build_neighbors_payload(leaf_ids, titles, neighbors)

    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    edges_doc = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "params": {
            "metric": "cosine",
            "top_k": args.top_k,
            "min_score": args.min_score,
            "mutual_only": args.mutual_only,
        },
        "edge_count": len(edges),
        "edges": edges,
    }
    EDGES_FILE.write_text(
        json.dumps(edges_doc, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    neighbors_doc = {
        "generated_at": edges_doc["generated_at"],
        "params": edges_doc["params"],
        "neighbors": neighbors_payload,
    }
    NEIGHBORS_FILE.write_text(
        json.dumps(neighbors_doc, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    _print_summary(
        n=len(leaf_ids),
        top_k=args.top_k,
        min_score=args.min_score,
        mutual_only=args.mutual_only,
        edges=edges,
        neighbors_payload=neighbors_payload,
        titles=titles,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
