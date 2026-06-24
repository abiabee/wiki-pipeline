"""Optional — render an interactive 2D map of the combined graph.

Not part of the production pipeline; this is a sanity-check tool. It loads
the embeddings + leaf metadata + combined edges (similarity ⨁ rules), projects
vectors to 2D with UMAP, and writes a self-contained interactive HTML to
output/graph.html.

Each dot is a leaf:
  - position : 2D UMAP projection of the embedding (similar leaves cluster)
  - color    : primary business_area (first entry, lowercased)
  - size     : degree in the combined graph (more connected = bigger)
  - hover    : title + business areas + status + degree + rescue count

Edges are split into three classes so the rule-based rescue is visible:
  - similarity_only (gray)    : sim > 0, rule_weight == 0
  - both           (dark)     : sim > 0 AND rule_weight > 0  (reinforced)
  - rescued        (orange)   : sim == 0 AND rule_weight > 0 (rules brought it in)

Run:
  python scripts/visualize_graph.py
  python scripts/visualize_graph.py --min-edge-weight 0.45      # denser view
  python scripts/visualize_graph.py --source similarity         # step 3 only
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
    COMBINED_EDGES_FILE,
    EDGES_FILE,
    EMBEDDINGS_INDEX,
    EMBEDDINGS_VECTORS,
    GRAPH_HTML,
    LEAVES_DIR,
    ROOT,
    console,
)


# Edge visual classes
EDGE_CLASS_SIM_ONLY = "similarity_only"
EDGE_CLASS_BOTH = "both"
EDGE_CLASS_RESCUED = "rescued"

EDGE_CLASS_STYLES: Dict[str, Dict[str, Any]] = {
    EDGE_CLASS_SIM_ONLY: {
        "color": "rgba(150,150,150,0.30)",
        "width": 0.5,
        "label": "similarity only",
    },
    EDGE_CLASS_BOTH: {
        "color": "rgba(60,90,160,0.45)",
        "width": 0.7,
        "label": "similarity + rules",
    },
    EDGE_CLASS_RESCUED: {
        "color": "rgba(245,140,40,0.55)",
        "width": 0.7,
        "label": "rescued (rules only)",
    },
}


def _load_inputs(
    source: str,
) -> Tuple[List[str], np.ndarray, List[Dict[str, Any]]]:
    if not EMBEDDINGS_VECTORS.exists() or not EMBEDDINGS_INDEX.exists():
        console.print(
            "[red]Embeddings missing. Run embed_leaves.py first.[/]"
        )
        sys.exit(2)

    target_file = COMBINED_EDGES_FILE if source == "combined" else EDGES_FILE
    if not target_file.exists():
        if source == "combined":
            console.print(
                "[red]combined_edges.json not found. "
                "Run build_rule_edges.py first, or use --source similarity.[/]"
            )
        else:
            console.print(
                "[red]edges.json not found. Run build_similarity_graph.py first.[/]"
            )
        sys.exit(2)

    index = json.loads(EMBEDDINGS_INDEX.read_text(encoding="utf-8"))
    vectors = np.load(EMBEDDINGS_VECTORS)
    edges_doc = json.loads(target_file.read_text(encoding="utf-8"))

    leaf_ids: List[str] = [""] * vectors.shape[0]
    for lid, entry in index["leaves"].items():
        leaf_ids[entry["row"]] = lid

    edges = edges_doc["edges"]
    normalized = [_normalize_edge(e, source) for e in edges]
    # An edge can come back as None when the source is `similarity` (step 3
    # raw file) and the entry is a rule edge written by step 4 — we only
    # plot similarity rows in that mode.
    return leaf_ids, vectors, [e for e in normalized if e is not None]


def _normalize_edge(edge: Dict[str, Any], source: str) -> Dict[str, Any] | None:
    """Coerce both legacy and combined edge shapes into one format.

    Returns a dict with at least:
      from, to, weight (float), edge_class (str), reason_count (int)
    """
    if source == "combined":
        # combined_edges.json shape — already has everything we need.
        sim = float(edge.get("similarity") or 0.0)
        rule_w = float(edge.get("rule_weight") or 0.0)
        if sim > 0 and rule_w > 0:
            cls = EDGE_CLASS_BOTH
        elif rule_w > 0:
            cls = EDGE_CLASS_RESCUED
        else:
            cls = EDGE_CLASS_SIM_ONLY
        return {
            "from": edge["from"],
            "to": edge["to"],
            "weight": float(edge["combined_weight"]),
            "similarity": sim,
            "rule_weight": rule_w,
            "edge_class": cls,
            "reason_count": int(edge.get("reason_count", 0)),
        }

    # source == "similarity"  → only keep similarity-typed rows from step 3
    if edge.get("type") not in (None, "similarity"):
        return None
    score = edge.get("score")
    if score is None:
        return None
    return {
        "from": edge["from"],
        "to": edge["to"],
        "weight": float(score),
        "similarity": float(score),
        "rule_weight": 0.0,
        "edge_class": EDGE_CLASS_SIM_ONLY,
        "reason_count": 1,
    }


def _load_leaf_meta(leaf_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Pull the bits we need for hover/color/size from each leaf JSON."""
    meta: Dict[str, Dict[str, Any]] = {}
    for leaf_id in leaf_ids:
        path = LEAVES_DIR / f"{leaf_id}.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            meta[leaf_id] = {
                "title": leaf_id,
                "business_area": [],
                "primary": "unknown",
                "doc_status": "unknown",
            }
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
    min_edge_weight: float,
    source: str,
) -> "Any":
    import plotly.graph_objects as go

    row_of = {lid: i for i, lid in enumerate(leaf_ids)}
    kept_edges = [
        e for e in edges
        if e["weight"] >= min_edge_weight
        and e["from"] in row_of
        and e["to"] in row_of
    ]

    degree: Dict[str, int] = defaultdict(int)
    rescued_per_leaf: Dict[str, int] = defaultdict(int)
    for e in kept_edges:
        degree[e["from"]] += 1
        degree[e["to"]] += 1
        if e["edge_class"] == EDGE_CLASS_RESCUED:
            rescued_per_leaf[e["from"]] += 1
            rescued_per_leaf[e["to"]] += 1

    primaries = [meta[lid]["primary"] for lid in leaf_ids]
    palette = _palette(sorted(set(primaries)))

    # One trace per edge class so the legend lets you toggle each kind.
    fig = go.Figure()
    edges_by_class: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in kept_edges:
        edges_by_class[e["edge_class"]].append(e)

    # Plot order: faintest first (sim_only), then both, rescued on top.
    for cls in (EDGE_CLASS_SIM_ONLY, EDGE_CLASS_BOTH, EDGE_CLASS_RESCUED):
        bucket = edges_by_class.get(cls, [])
        if not bucket:
            continue
        style = EDGE_CLASS_STYLES[cls]
        edge_x: List[float] = []
        edge_y: List[float] = []
        for e in bucket:
            i = row_of[e["from"]]
            j = row_of[e["to"]]
            edge_x.extend([float(coords[i, 0]), float(coords[j, 0]), None])
            edge_y.extend([float(coords[i, 1]), float(coords[j, 1]), None])
        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(width=style["width"], color=style["color"]),
                hoverinfo="skip",
                name=f"{style['label']} ({len(bucket)})",
                legendgroup="edges",
                legendgrouptitle=dict(text="edges"),
            )
        )

    # Node traces (one per primary business_area for legend filtering).
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
            "degree: {deg} (rescued: {res})<br>"
            "<i>{lid}</i>".format(
                title=meta[leaf_ids[i]]["title"],
                primary=meta[leaf_ids[i]]["primary"],
                areas=", ".join(meta[leaf_ids[i]]["business_area"]) or "(none)",
                status=meta[leaf_ids[i]]["doc_status"],
                deg=degree.get(leaf_ids[i], 0),
                res=rescued_per_leaf.get(leaf_ids[i], 0),
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
                legendgroup="nodes",
                legendgrouptitle=dict(text="primary business_area"),
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

    title_kind = "combined" if source == "combined" else "similarity-only"
    rescued_count = len(edges_by_class.get(EDGE_CLASS_RESCUED, []))
    fig.update_layout(
        title=dict(
            text=(
                f"Wiki {title_kind} graph — {len(leaf_ids)} leaves, "
                f"{len(kept_edges)} edges (min weight {min_edge_weight}"
                + (f", {rescued_count} rescued" if source == "combined" else "")
                + ")"
            )
        ),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        hovermode="closest",
        plot_bgcolor="white",
        legend=dict(groupclick="toggleitem"),
        margin=dict(l=10, r=10, t=60, b=10),
        height=820,
    )
    return fig


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=("combined", "similarity"),
        default="combined",
        help=(
            "Which graph to plot. 'combined' (default) reads combined_edges.json "
            "and shows similarity, rule, and rescued edges. 'similarity' reads "
            "edges.json filtered to type=similarity (the step-3 view, for comparison)."
        ),
    )
    parser.add_argument(
        "--min-edge-weight",
        type=float,
        default=0.55,
        help=(
            "Hide edges below this weight. For combined: combined_weight. "
            "For similarity: cosine score. Default 0.55."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="UMAP random seed (default 42).",
    )
    args = parser.parse_args(argv)

    leaf_ids, vectors, edges = _load_inputs(args.source)
    meta = _load_leaf_meta(leaf_ids)

    console.print(
        f"Projecting [bold]{vectors.shape[0]}[/] vectors to 2D with UMAP "
        f"(source: {args.source})..."
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
        min_edge_weight=args.min_edge_weight,
        source=args.source,
    )

    GRAPH_HTML.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(GRAPH_HTML), include_plotlyjs="cdn", full_html=True)

    primaries = Counter(m["primary"] for m in meta.values())
    edges_kept = sum(1 for e in edges if e["weight"] >= args.min_edge_weight)
    edge_class_counts = Counter(
        e["edge_class"] for e in edges if e["weight"] >= args.min_edge_weight
    )

    table = Table(title="Visualization summary", show_lines=False)
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_row("Output", str(GRAPH_HTML.relative_to(ROOT)))
    table.add_row("Source", args.source)
    table.add_row("Leaves plotted", str(len(leaf_ids)))
    table.add_row(
        "Edges plotted",
        f"{edges_kept} (of {len(edges)})",
    )
    if args.source == "combined":
        table.add_row(
            "  by class",
            ", ".join(
                f"{cls}={edge_class_counts.get(cls, 0)}"
                for cls in (EDGE_CLASS_SIM_ONLY, EDGE_CLASS_BOTH, EDGE_CLASS_RESCUED)
            ),
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
