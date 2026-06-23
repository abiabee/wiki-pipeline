"""Step 4 — Build deterministic (rule-based) edges and combine with similarity.

Rule edges are derived from leaf metadata, NOT embeddings. Two leaves get a
typed edge for each entity / business_area value they share. The base weight
for each rule type lives in `_common.RULE_BASE_WEIGHTS` and is multiplied by
an IDF factor that crushes broad signals (e.g. "shared_product: Paystand"
across most of the corpus).

This step is idempotent: it reads `edges.json`, keeps the existing
`type == "similarity"` edges from step 3, regenerates rule edges from
scratch, and writes everything back.

Outputs:
  - edges.json              all edges (similarity + rule), multi-typed
  - combined_edges.json     one row per pair, saturating combined weight
  - related_neighbors.json  top-K per leaf by combined_weight
  - entity_index.json       reverse map: entity_type -> slug -> leaves
  - step4_summary.json      health metrics + potential alias collisions

Combination math:
  combined = 1 - (1 - similarity) * Π (1 - wᵢ)   for each rule weight wᵢ
  rescued  = (similarity == 0) and (rule_weight > 0)

Run:
  python scripts/build_rule_edges.py
  python scripts/build_rule_edges.py --top-k 15
  python scripts/build_rule_edges.py --min-rule-weight 0.10
  python scripts/build_rule_edges.py --include-skipped-with-entities
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from rich.table import Table
from slugify import slugify

from _common import (
    COMBINED_EDGES_FILE,
    EDGES_FILE,
    ENTITY_INDEX_FILE,
    GRAPH_DIR,
    LEAVES_DIR,
    NEIGHBORS_FILE,
    OUTPUT_DIR,
    RELATED_NEIGHBORS_FILE,
    ROOT,
    RULE_BASE_WEIGHTS,
    STEP4_SUMMARY_FILE,
    VALIDATION_REPORT,
    console,
)


DEFAULT_TOP_K = 10
DEFAULT_MIN_RULE_WEIGHT = 0.05

EMBEDDABLE_STATUSES = {"ok", "ok_with_warnings"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slug(value: str) -> Optional[str]:
    """Slugify with a fallback. Returns None if the input is empty/garbage."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    out = slugify(text)
    return out or None


def _canonical_pair(a: str, b: str) -> Tuple[str, str]:
    """Order a pair so (from, to) is stable regardless of insertion order."""
    return (a, b) if a < b else (b, a)


def _saturate(weights: Iterable[float]) -> float:
    """Probabilistic OR: 1 - Π (1 - wᵢ). Stays in [0, 1] for any count."""
    p = 1.0
    for w in weights:
        if w <= 0:
            continue
        p *= 1.0 - w
    return 1.0 - p


def _idf(group_size: int, n: int) -> float:
    """Information value of a signal that's shared by `group_size` of `n` leaves.

    - group_size == 1: no signal (only one leaf has it). Returns 0.
    - group_size == n: signal is everywhere (no information). Returns 0.
    - Otherwise: log(n / group_size) / log(n), bounded in (0, 1].
    """
    if group_size <= 1 or group_size >= n or n <= 1:
        return 0.0
    return math.log(n / group_size) / math.log(n)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load_validation_report() -> Dict[str, Any]:
    if not VALIDATION_REPORT.exists():
        console.print(
            "[red]Validation report not found. Run validate_leaves.py first.[/]"
        )
        sys.exit(2)
    return json.loads(VALIDATION_REPORT.read_text(encoding="utf-8"))


def _select_leaves(
    report: Dict[str, Any],
    include_skipped_with_entities: bool,
) -> List[str]:
    """Return the leaf_ids that participate in step 4 (deterministic)."""
    selected: List[str] = []
    for leaf_id, info in report.get("leaves", {}).items():
        status = info.get("status")
        if status in EMBEDDABLE_STATUSES:
            selected.append(leaf_id)
        elif (
            include_skipped_with_entities
            and status == "skipped_not_ready"
        ):
            # Defer the actual entity check to load time; we drop empty leaves
            # there so the candidate list still reads as straightforward here.
            selected.append(leaf_id)
    return sorted(selected)


def _load_leaf(leaf_id: str) -> Optional[Dict[str, Any]]:
    path = LEAVES_DIR / f"{leaf_id}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _has_any_entity_or_area(leaf: Dict[str, Any]) -> bool:
    entities = leaf.get("entities") or {}
    for values in entities.values():
        if values:
            return True
    bas = (leaf.get("classification") or {}).get("business_area") or []
    return bool([b for b in bas if b and b.lower() != "unknown"])


# ---------------------------------------------------------------------------
# Entity index
# ---------------------------------------------------------------------------


# entity_index[entity_type][slug] = {
#     "label": str,                # most-common original label
#     "label_counts": Counter,     # all original labels seen, for alias debug
#     "leaf_ids": set[str],
# }
EntityIndex = Dict[str, Dict[str, Dict[str, Any]]]


def _build_entity_index(
    leaves_data: Dict[str, Dict[str, Any]],
) -> EntityIndex:
    """Bucket every entity / business_area value by (type, slug)."""
    index: EntityIndex = defaultdict(lambda: defaultdict(_new_bucket))

    for leaf_id, leaf in leaves_data.items():
        entities = leaf.get("entities") or {}
        for entity_type, raw_values in entities.items():
            if entity_type not in RULE_BASE_WEIGHTS:
                continue  # rule type intentionally excluded (e.g. document_type)
            for raw in raw_values or []:
                slug = _slug(raw)
                if not slug:
                    continue
                bucket = index[entity_type][slug]
                bucket["label_counts"][str(raw).strip()] += 1
                bucket["leaf_ids"].add(leaf_id)

        if "business_area" in RULE_BASE_WEIGHTS:
            for raw in (leaf.get("classification") or {}).get("business_area") or []:
                # Skip the literal sentinel "unknown" — it shouldn't unify leaves.
                if str(raw).strip().lower() == "unknown":
                    continue
                slug = _slug(raw)
                if not slug:
                    continue
                bucket = index["business_area"][slug]
                bucket["label_counts"][str(raw).strip()] += 1
                bucket["leaf_ids"].add(leaf_id)

    # Resolve "label" = most-common original spelling.
    for entity_type, slugs in index.items():
        for slug, bucket in slugs.items():
            counts: Counter = bucket["label_counts"]
            if counts:
                bucket["label"] = counts.most_common(1)[0][0]
            else:
                bucket["label"] = slug

    return index


def _new_bucket() -> Dict[str, Any]:
    return {
        "label": "",
        "label_counts": Counter(),
        "leaf_ids": set(),
    }


def _serialize_entity_index(index: EntityIndex) -> Dict[str, Any]:
    """Convert sets/Counters to JSON-friendly types and sort everything."""
    out: Dict[str, Any] = {}
    for entity_type in sorted(index.keys()):
        slugs_out: Dict[str, Any] = {}
        for slug in sorted(index[entity_type].keys()):
            bucket = index[entity_type][slug]
            slugs_out[slug] = {
                "label": bucket["label"],
                "size": len(bucket["leaf_ids"]),
                "leaf_ids": sorted(bucket["leaf_ids"]),
                "all_labels": sorted(bucket["label_counts"].keys()),
            }
        out[entity_type] = slugs_out
    return out


def _alias_warnings(index: EntityIndex) -> List[Dict[str, Any]]:
    """Surface (type, slug) buckets with multiple distinct original spellings.

    These are the *most likely* aliases worth folding via a curated map later.
    Each entry includes a sample of what was seen, ordered by frequency.
    """
    warnings: List[Dict[str, Any]] = []
    for entity_type, slugs in index.items():
        for slug, bucket in slugs.items():
            counts: Counter = bucket["label_counts"]
            if len(counts) <= 1:
                continue
            warnings.append(
                {
                    "entity_type": entity_type,
                    "slug": slug,
                    "label_variants": [
                        {"label": label, "count": cnt}
                        for label, cnt in counts.most_common()
                    ],
                    "leaf_count": len(bucket["leaf_ids"]),
                }
            )
    warnings.sort(key=lambda w: (w["entity_type"], w["slug"]))
    return warnings


# ---------------------------------------------------------------------------
# Rule edges
# ---------------------------------------------------------------------------


def _generate_rule_edges(
    index: EntityIndex,
    n_leaves: int,
    min_rule_weight: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Walk the entity index and emit one typed edge per (pair, shared_value).

    Returns the edge list plus a per-type emitted-count map for the summary.
    """
    edges: List[Dict[str, Any]] = []
    counts: Dict[str, int] = defaultdict(int)
    skipped_low_weight: int = 0

    for entity_type, slugs in index.items():
        base = RULE_BASE_WEIGHTS[entity_type]
        for slug, bucket in slugs.items():
            leaf_ids = bucket["leaf_ids"]
            group_size = len(leaf_ids)
            if group_size < 2:
                continue
            weight = base * _idf(group_size, n_leaves)
            if weight < min_rule_weight:
                # Pre-count how many edges we *would* have emitted; this is
                # the noise the IDF + threshold combo is filtering out.
                skipped_low_weight += group_size * (group_size - 1) // 2
                continue
            edge_type = _edge_type_for(entity_type)
            label = bucket["label"]
            for a, b in combinations(sorted(leaf_ids), 2):
                edges.append(
                    {
                        "from": a,
                        "to": b,
                        "type": edge_type,
                        "shared_slug": slug,
                        "shared_label": label,
                        "weight": round(weight, 4),
                        "group_size": group_size,
                    }
                )
                counts[edge_type] += 1
    counts["__skipped_low_weight__"] = skipped_low_weight
    return edges, counts


def _edge_type_for(entity_type: str) -> str:
    """Map plural entity field name to singular edge type."""
    table = {
        "customers": "shared_customer",
        "partners": "shared_partner",
        "erps": "shared_erp",
        "competitors": "shared_competitor",
        "products": "shared_product",
        "policies": "shared_policy",
        "people": "shared_person",
        "features": "shared_feature",
        "business_area": "shared_business_area",
    }
    return table.get(entity_type, f"shared_{entity_type}")


# ---------------------------------------------------------------------------
# Combine similarity + rules
# ---------------------------------------------------------------------------


def _load_existing_similarity_edges() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Pull just the type=='similarity' edges out of edges.json.

    Returns (similarity_edges, original_doc_metadata).
    """
    if not EDGES_FILE.exists():
        console.print(
            "[red]edges.json not found. Run build_similarity_graph.py first.[/]"
        )
        sys.exit(2)

    doc = json.loads(EDGES_FILE.read_text(encoding="utf-8"))
    edges = doc.get("edges", [])
    sim = [e for e in edges if e.get("type") == "similarity"]
    return sim, doc


def _build_combined_edges(
    similarity_edges: List[Dict[str, Any]],
    rule_edges: List[Dict[str, Any]],
    embeddable_set: Set[str],
) -> List[Dict[str, Any]]:
    """Collapse to one entry per (a, b) with saturating combined weight."""
    pair_data: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for e in similarity_edges:
        key = _canonical_pair(e["from"], e["to"])
        slot = pair_data.setdefault(
            key,
            {"sim": 0.0, "rules": [], "from_title": "", "to_title": ""},
        )
        slot["sim"] = float(e.get("score", 0.0))
        slot["from_title"] = e.get("from_title", "")
        slot["to_title"] = e.get("to_title", "")

    for e in rule_edges:
        key = _canonical_pair(e["from"], e["to"])
        slot = pair_data.setdefault(
            key,
            {"sim": 0.0, "rules": [], "from_title": "", "to_title": ""},
        )
        slot["rules"].append(e)

    combined: List[Dict[str, Any]] = []
    for (a, b), info in pair_data.items():
        rule_weights = [r["weight"] for r in info["rules"]]
        rule_weight = _saturate(rule_weights)
        sim = info["sim"]
        combined_weight = _saturate([sim, rule_weight])

        # Only include pairs where at least one endpoint is embeddable AND
        # the other is in our participating set. (We keep it permissive when
        # --include-skipped-with-entities is on; both ends will already be
        # in our index buckets in that mode.)
        if a not in embeddable_set and b not in embeddable_set:
            continue

        reasons: List[Dict[str, Any]] = []
        if sim > 0:
            reasons.append({"type": "similarity", "score": round(sim, 4)})
        for r in info["rules"]:
            reasons.append(
                {
                    "type": r["type"],
                    "shared_label": r["shared_label"],
                    "shared_slug": r["shared_slug"],
                    "weight": r["weight"],
                    "group_size": r["group_size"],
                }
            )

        combined.append(
            {
                "from": a,
                "to": b,
                "from_title": info["from_title"],
                "to_title": info["to_title"],
                "similarity": round(sim, 4),
                "rule_weight": round(rule_weight, 4),
                "combined_weight": round(combined_weight, 4),
                "rescued": sim == 0.0 and rule_weight > 0.0,
                "reason_count": len(reasons),
                "reasons": reasons,
            }
        )

    combined.sort(key=lambda e: e["combined_weight"], reverse=True)
    return combined


# ---------------------------------------------------------------------------
# Related-neighbors (top-K)
# ---------------------------------------------------------------------------


def _build_related_neighbors(
    combined_edges: List[Dict[str, Any]],
    leaf_ids: List[str],
    titles: Dict[str, str],
    top_k: int,
) -> Dict[str, List[Dict[str, Any]]]:
    """Per-leaf top-K outgoing edges, ranked by combined_weight."""
    by_leaf: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in combined_edges:
        by_leaf[e["from"]].append(_neighbor_view(e, "to", titles))
        by_leaf[e["to"]].append(_neighbor_view(e, "from", titles))

    out: Dict[str, List[Dict[str, Any]]] = {}
    for leaf_id in leaf_ids:
        items = sorted(
            by_leaf.get(leaf_id, []),
            key=lambda d: d["combined_weight"],
            reverse=True,
        )[:top_k]
        for rank, item in enumerate(items, start=1):
            item["rank"] = rank
        out[leaf_id] = items
    return out


def _neighbor_view(
    edge: Dict[str, Any], other_side: str, titles: Dict[str, str]
) -> Dict[str, Any]:
    other_id = edge[other_side]
    return {
        "leaf_id": other_id,
        "title": titles.get(other_id, other_id),
        "combined_weight": edge["combined_weight"],
        "similarity": edge["similarity"],
        "rule_weight": edge["rule_weight"],
        "rescued": edge["rescued"],
        "reasons": edge["reasons"],
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _write_edges_json(
    similarity_edges: List[Dict[str, Any]],
    rule_edges: List[Dict[str, Any]],
    sim_doc_meta: Dict[str, Any],
) -> None:
    """Re-write edges.json with similarity + rule edges, sim params preserved."""
    edges = list(similarity_edges) + list(rule_edges)
    edges.sort(
        key=lambda e: (
            0 if e.get("type") == "similarity" else 1,
            -float(e.get("score", e.get("weight", 0.0))),
        )
    )
    doc = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "params": {
            "similarity": sim_doc_meta.get("params", {}),
            # Rule params are mirrored in step4_summary.json; keep edges.json
            # focused on the data, not the run config.
        },
        "edge_count": len(edges),
        "similarity_edge_count": len(similarity_edges),
        "rule_edge_count": len(rule_edges),
        "edges": edges,
    }
    EDGES_FILE.write_text(
        json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _write_combined_edges(combined: List[Dict[str, Any]], params: Dict[str, Any]) -> None:
    doc = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "params": params,
        "edge_count": len(combined),
        "rescued_count": sum(1 for e in combined if e["rescued"]),
        "edges": combined,
    }
    COMBINED_EDGES_FILE.write_text(
        json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _write_related_neighbors(
    related: Dict[str, List[Dict[str, Any]]], params: Dict[str, Any]
) -> None:
    doc = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "params": params,
        "neighbors": related,
    }
    RELATED_NEIGHBORS_FILE.write_text(
        json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _write_entity_index(serialized: Dict[str, Any]) -> None:
    doc = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "entity_types": sorted(serialized.keys()),
        "index": serialized,
    }
    ENTITY_INDEX_FILE.write_text(
        json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _write_summary(summary: Dict[str, Any]) -> None:
    STEP4_SUMMARY_FILE.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------


def _print_summary(
    n_leaves: int,
    rule_edge_counts: Dict[str, int],
    rule_edges: List[Dict[str, Any]],
    similarity_count: int,
    combined: List[Dict[str, Any]],
    alias_warnings: List[Dict[str, Any]],
    top_k: int,
    min_rule_weight: float,
) -> None:
    table = Table(title="Step 4 — rule edges + combined graph", show_lines=False)
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_row("Leaves participating", str(n_leaves))
    table.add_row("Similarity edges (kept)", str(similarity_count))
    table.add_row("Rule edges (emitted)", str(len(rule_edges)))
    table.add_row(
        "Rule edges skipped (weight < threshold)",
        str(rule_edge_counts.get("__skipped_low_weight__", 0)),
    )
    table.add_row("Combined pairs (collapsed)", str(len(combined)))
    table.add_row(
        "Rescued pairs (sim==0, rule>0)",
        str(sum(1 for e in combined if e["rescued"])),
    )
    if combined:
        weights = [e["combined_weight"] for e in combined]
        table.add_row(
            "Combined weight",
            f"min={min(weights):.3f}  median={statistics.median(weights):.3f}  max={max(weights):.3f}",
        )
    table.add_row(
        "Params",
        f"top_k={top_k}, min_rule_weight={min_rule_weight}, "
        f"saturating combination",
    )
    table.add_row("Edges file", str(EDGES_FILE.relative_to(ROOT)))
    table.add_row("Combined edges", str(COMBINED_EDGES_FILE.relative_to(ROOT)))
    table.add_row("Related neighbors", str(RELATED_NEIGHBORS_FILE.relative_to(ROOT)))
    table.add_row("Entity index", str(ENTITY_INDEX_FILE.relative_to(ROOT)))
    table.add_row("Summary", str(STEP4_SUMMARY_FILE.relative_to(ROOT)))
    console.print(table)

    by_type = Counter(
        {k: v for k, v in rule_edge_counts.items() if not k.startswith("__")}
    )
    if by_type:
        bt = Table(title="Rule edges by type", show_lines=False)
        bt.add_column("Type", style="bold")
        bt.add_column("Count", justify="right")
        for edge_type, c in by_type.most_common():
            bt.add_row(edge_type, str(c))
        console.print(bt)

    if alias_warnings:
        ax = Table(
            title=f"Potential aliases (same slug, multiple labels) — {len(alias_warnings)} cases",
            show_lines=False,
        )
        ax.add_column("Type")
        ax.add_column("Slug")
        ax.add_column("Variants seen")
        for w in alias_warnings[:10]:
            variants = ", ".join(
                f"{v['label']!r}×{v['count']}" for v in w["label_variants"]
            )
            ax.add_row(w["entity_type"], w["slug"], variants)
        if len(alias_warnings) > 10:
            ax.add_row("…", "…", f"({len(alias_warnings) - 10} more in step4_summary.json)")
        console.print(ax)

    rescued = [e for e in combined if e["rescued"]]
    if rescued:
        rt = Table(
            title=f"Top rescued connections (sim==0, rule>0) — {len(rescued)} total, showing 5",
            show_lines=False,
        )
        rt.add_column("Combined", justify="right")
        rt.add_column("From")
        rt.add_column("To")
        rt.add_column("Reasons")
        for e in sorted(rescued, key=lambda x: x["combined_weight"], reverse=True)[:5]:
            reasons = "; ".join(
                f"{r['type']}={r.get('shared_label', r.get('score'))}"
                for r in e["reasons"]
            )
            rt.add_row(
                f"{e['combined_weight']:.3f}",
                e.get("from_title") or e["from"],
                e.get("to_title") or e["to"],
                reasons[:80] + ("…" if len(reasons) > 80 else ""),
            )
        console.print(rt)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Top-K per leaf in related_neighbors.json (default {DEFAULT_TOP_K}).",
    )
    parser.add_argument(
        "--min-rule-weight",
        type=float,
        default=DEFAULT_MIN_RULE_WEIGHT,
        help=(
            f"Drop rule edges whose post-IDF weight falls below this "
            f"(default {DEFAULT_MIN_RULE_WEIGHT})."
        ),
    )
    parser.add_argument(
        "--include-skipped-with-entities",
        action="store_true",
        help=(
            "Also include `skipped_not_ready` leaves that have at least one "
            "entity or non-'unknown' business_area. They get no similarity "
            "edges (no embedding) but participate in rule edges and in "
            "entity_index. Off by default."
        ),
    )
    args = parser.parse_args(argv)

    report = _load_validation_report()
    candidate_ids = _select_leaves(
        report,
        include_skipped_with_entities=args.include_skipped_with_entities,
    )

    leaves_data: Dict[str, Dict[str, Any]] = {}
    skipped_empty: List[str] = []
    embeddable_set: Set[str] = set()
    for leaf_id in candidate_ids:
        leaf = _load_leaf(leaf_id)
        if leaf is None:
            continue
        status = report["leaves"][leaf_id]["status"]
        if status in EMBEDDABLE_STATUSES:
            embeddable_set.add(leaf_id)
            leaves_data[leaf_id] = leaf
        elif args.include_skipped_with_entities:
            if _has_any_entity_or_area(leaf):
                leaves_data[leaf_id] = leaf
            else:
                skipped_empty.append(leaf_id)

    if not leaves_data:
        console.print("[red]No participating leaves; aborting.[/]")
        return 2

    n_leaves = len(leaves_data)
    console.print(
        f"[bold]{n_leaves}[/] leaves participating "
        f"(embeddable: {len(embeddable_set)}, "
        f"skipped-included: {n_leaves - len(embeddable_set)}, "
        f"skipped-empty-dropped: {len(skipped_empty)})"
    )

    index = _build_entity_index(leaves_data)
    serialized_index = _serialize_entity_index(index)
    alias_warnings = _alias_warnings(index)

    rule_edges, rule_counts = _generate_rule_edges(
        index, n_leaves=n_leaves, min_rule_weight=args.min_rule_weight
    )

    similarity_edges, sim_doc_meta = _load_existing_similarity_edges()

    combined = _build_combined_edges(
        similarity_edges=similarity_edges,
        rule_edges=rule_edges,
        embeddable_set=embeddable_set | (set(leaves_data.keys()) - embeddable_set),
    )

    titles = _collect_titles(leaves_data, similarity_edges)
    related = _build_related_neighbors(
        combined_edges=combined,
        leaf_ids=sorted(leaves_data.keys()),
        titles=titles,
        top_k=args.top_k,
    )

    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    params = {
        "top_k": args.top_k,
        "min_rule_weight": args.min_rule_weight,
        "include_skipped_with_entities": args.include_skipped_with_entities,
        "combination": "saturating: 1 - (1-sim) * Π (1-w_i)",
        "idf_formula": "log(N / group_size) / log(N)",
        "rule_base_weights": dict(RULE_BASE_WEIGHTS),
    }

    _write_edges_json(similarity_edges, rule_edges, sim_doc_meta)
    _write_combined_edges(combined, params)
    _write_related_neighbors(related, params)
    _write_entity_index(serialized_index)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "leaves": {
            "participating": n_leaves,
            "embeddable": len(embeddable_set),
            "skipped_with_entities": n_leaves - len(embeddable_set),
            "skipped_dropped_empty": skipped_empty,
        },
        "params": params,
        "edge_counts": {
            "similarity": len(similarity_edges),
            "rule_total": len(rule_edges),
            "rule_by_type": {
                k: v for k, v in rule_counts.items() if not k.startswith("__")
            },
            "rule_skipped_low_weight": rule_counts.get("__skipped_low_weight__", 0),
            "combined_pairs": len(combined),
            "rescued_pairs": sum(1 for e in combined if e["rescued"]),
        },
        "entity_specificity": _entity_specificity(serialized_index),
        "potential_alias_collisions": alias_warnings,
    }
    _write_summary(summary)

    _print_summary(
        n_leaves=n_leaves,
        rule_edge_counts=rule_counts,
        rule_edges=rule_edges,
        similarity_count=len(similarity_edges),
        combined=combined,
        alias_warnings=alias_warnings,
        top_k=args.top_k,
        min_rule_weight=args.min_rule_weight,
    )
    return 0


def _collect_titles(
    leaves_data: Dict[str, Dict[str, Any]],
    similarity_edges: List[Dict[str, Any]],
) -> Dict[str, str]:
    titles: Dict[str, str] = {}
    for leaf_id, leaf in leaves_data.items():
        title = (
            (leaf.get("embedding") or {}).get("title")
            or (leaf.get("source") or {}).get("name")
            or leaf_id
        )
        titles[leaf_id] = str(title).strip() or leaf_id
    # Backfill from existing edges (covers any leaf we somehow missed).
    for e in similarity_edges:
        titles.setdefault(e["from"], e.get("from_title", e["from"]))
        titles.setdefault(e["to"], e.get("to_title", e["to"]))
    return titles


def _entity_specificity(serialized_index: Dict[str, Any]) -> Dict[str, Any]:
    """Per-rule-type stats on group sizes, useful for tuning weights."""
    out: Dict[str, Any] = {}
    for entity_type, slugs in serialized_index.items():
        sizes = [info["size"] for info in slugs.values() if info["size"] >= 2]
        if not sizes:
            out[entity_type] = {"groups_with_pairs": 0}
            continue
        out[entity_type] = {
            "groups_with_pairs": len(sizes),
            "min_group": min(sizes),
            "median_group": int(statistics.median(sizes)),
            "max_group": max(sizes),
            "total_unique_values": len(slugs),
        }
    return out


if __name__ == "__main__":
    sys.exit(main())
