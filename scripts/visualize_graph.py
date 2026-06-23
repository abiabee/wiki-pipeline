"""Optional — render an interactive 2D map of the similarity graph.

Not part of the production pipeline; this is a sanity-check tool. It loads
the embeddings + edges + leaf metadata, projects vectors to 2D with UMAP,
and writes a self-contained interactive HTML to output/graph.html.

Each dot is a leaf:
  - position : 2D UMAP projection of the embedding (similar leaves cluster)
  - color    : primary business_area (first entry, lowercased)
  - size     : degree in the similarity graph (more connected = bigger)
  - hover    : title + business areas + degree

Each edge is a line whose opacity scales with cosine score.

Run:
  python scripts/visualize_graph.py
  python scripts/visualize_graph.py --min-edge-score 0.65   # cleaner plot
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
from rich.table import Table

from _common import (
    EDGES_FILE,
    EMBEDDINGS_INDEX,
    EMBEDDINGS_VECTORS,
    GRAPH_HTML,
    LEAVES_DIR,
    ROOT,
    console,
)


def _load_inputs() -> Tuple[List[str], np.ndarray, List[Dict[str, Any]]]:
    if not EMBEDDINGS_VECTORS.exists() or not EMBEDDINGS_INDEX.exists():
        console.print(
            "[red]Embeddings missing. Run embed_leaves.py first.[/]"
        )
        sys.exit(2)
    if not EDGES_FILE.exists():
        console.print(
            "[red]Edges file missing. Run build_similarity_graph.py first.[/]"
        )
        sys.exit(2)

    index = json.loads(EMBEDDINGS_INDEX.read_text(encoding="utf-8"))
    vectors = np.load(EMBEDDINGS_VECTORS)
    edges_doc = json.loads(EDGES_FILE.read_text(encoding="utf-8"))

    leaf_ids: List[str] = [""] * vectors.shape[0]
    for lid, entry in index["leaves"].items():
        leaf_ids[entry["row"]] = lid

    return leaf_ids, vectors, edges_doc["edges"]


def _load_leaf_meta(leaf_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Pull the bits we need for hover/color/size from each leaf JSON."""
    meta: Dict[str, Dict[str, Any]] = {}
    for leaf_id in leaf_ids:
        path = LEAVES_DIR / f"{leaf_id}.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            meta[leaf_id] = {"title": leaf_id, "business_area": [], "primary": "unknown"}
            continue
        emb = data.get("embedding") or {}
        cls = data.get("classification") or {}
        src = data.get("source") or {}
        title = (emb.get("title") or src.get("name") or leaf_id).strip()
        business_area = [str(x).lower() for x in (cls.get("business_area") or [])]
        primary = business_area[0] if business_area else "unknown"
        meta[leaf_id] = {
            "title": title,
            "business_area": business_area,
            "primary": primary,
            "doc_status": cls.get("status", "unknown"),
        }
    return meta


def _palette(categories: List[str]) -> Dict[str, str]:
    """Stable, distinct, qualitative colors for up to ~20 categories."""
    base = [
        "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
        "#EECA3B", "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC",
        "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
        "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF",
    ]
    return {cat: base[i % len(base)] for i, cat in enumerate(sorted(categories))}


def _build_html(
    leaf_ids: List[str],
    coords: np.ndarray,
    meta: Dict[str, Dict[str, Any]],
    edges: List[Dict[str, Any]],
    min_edge_score: float,
) -> "Any":
    import plotly.graph_objects as go

    row_of = {lid: i for i, lid in enumerate(leaf_ids)}
    degree: Dict[str, int] = defaultdict(int)
    kept_edges = [e for e in edges if e["score"] >= min_edge_score]
    for e in kept_edges:
        degree[e["from"]] += 1
        degree[e["to"]] += 1

    primaries = [meta[lid]["primary"] for lid in leaf_ids]
    palette = _palette(sorted(set(primaries)))

    edge_traces = []
    if kept_edges:
        edge_x: List[float] = []
        edge_y: List[float] = []
        for e in kept_edges:
            i = row_of[e["from"]]
            j = row_of[e["to"]]
            edge_x.extend([coords[i, 0], coords[j, 0], None])
            edge_y.extend([coords[i, 1], coords[j, 1], None])
        edge_traces.append(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(width=0.5, color="rgba(120,120,120,0.35)"),
                hoverinfo="skip",
                showlegend=False,
                name="similarity edges",
            )
        )

    # One trace per category so the legend gives a usable filter.
    fig = go.Figure(data=edge_traces)
    by_primary: Dict[str, List[int]] = defaultdict(list)
    for i, lid in enumerate(leaf_ids):
        by_primary[meta[lid]["primary"]].append(i)

    for primary, idxs in sorted(by_primary.items()):
        xs = [float(coords[i, 0]) for i in idxs]
        ys = [float(coords[i, 1]) for i in idxs]
        sizes = [
            6 + 2.5 * (degree.get(leaf_ids[i], 0) ** 0.5)
            for i in idxs
        ]
        hover = [
            "<b>{title}</b><br>"
            "primary: {primary}<br>"
            "areas: {areas}<br>"
            "status: {status}<br>"
            "degree: {deg}<br>"
            "<i>{lid}</i>".format(
                title=meta[leaf_ids[i]]["title"],
                primary=meta[leaf_ids[i]]["primary"],
                areas=", ".join(meta[leaf_ids[i]]["business_area"]) or "(none)",
                status=meta[leaf_ids[i]]["doc_status"],
                deg=degree.get(leaf_ids[i], 0),
                lid=leaf_ids[i],
            )
            for i in idxs
        ]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                name=primary,
                marker=dict(
                    size=sizes,
                    color=palette[primary],
                    line=dict(width=0.6, color="white"),
                    opacity=0.9,
                ),
                hovertemplate="%{text}<extra></extra>",
                text=hover,
            )
        )

    fig.update_layout(
        title=dict(
            text=(
                f"Wiki similarity graph — {len(leaf_ids)} leaves, "
                f"{len(kept_edges)} edges (min score {min_edge_score})"
            )
        ),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        hovermode="closest",
        plot_bgcolor="white",
        legend=dict(title="primary business_area"),
        margin=dict(l=10, r=10, t=60, b=10),
        height=820,
    )
    return fig


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-edge-score",
        type=float,
        default=0.55,
        help="Hide edges below this cosine score (default 0.55).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="UMAP random seed (default 42).",
    )
    args = parser.parse_args(argv)

    leaf_ids, vectors, edges = _load_inputs()
    meta = _load_leaf_meta(leaf_ids)

    console.print(
        f"Projecting [bold]{vectors.shape[0]}[/] vectors to 2D with UMAP..."
    )
    import umap  # heavy import; do it after we know inputs are valid

    reducer = umap.UMAP(
        n_components=2,
        random_state=args.seed,
        n_neighbors=min(15, max(2, vectors.shape[0] - 1)),
        metric="cosine",
    )
    coords = reducer.fit_transform(vectors)

    fig = _build_html(
        leaf_ids=leaf_ids,
        coords=coords,
        meta=meta,
        edges=edges,
        min_edge_score=args.min_edge_score,
    )

    GRAPH_HTML.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(GRAPH_HTML), include_plotlyjs="cdn", full_html=True)

    primaries = Counter(m["primary"] for m in meta.values())
    table = Table(title="Visualization summary", show_lines=False)
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_row("Output", str(GRAPH_HTML.relative_to(ROOT)))
    table.add_row("Leaves plotted", str(len(leaf_ids)))
    table.add_row(
        "Edges plotted",
        f"{sum(1 for e in edges if e['score'] >= args.min_edge_score)} "
        f"(of {len(edges)})",
    )
    table.add_row(
        "Primary business areas",
        ", ".join(f"{k}={v}" for k, v in primaries.most_common()),
    )
    console.print(table)
    console.print(
        f"[green]Open[/] [bold]{GRAPH_HTML.relative_to(ROOT)}[/] in your browser."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
