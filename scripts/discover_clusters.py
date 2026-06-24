"""Step 6 — Discover the section node tree (cluster discovery + LLM rendering contracts).

Inputs (all from prior pipeline steps):
  output/sections_assignment.json  per-leaf primary section + cross-listings (step 5)
  output/sections_index.json       section -> leaf_ids (step 5)
  output/graph/entity_index.json   entity_type -> slug -> leaves (step 4)
  output/graph/combined_edges.json sim + rule combined weights per pair (step 4)
  output/embeddings/index.json     leaf metadata (titles)
  input/leaves/*.json              raw leaves (audience, document_type, entity labels)

Outputs:
  output/tree/nodes.json           flat dict of every node's rendering contract
  output/tree/section_trees.json   per-section traversal index (root + leaf placements)
  output/tree/step6_summary.json   per-section stats, recursion counts, warnings

The tree shape per section (depth = position below the section root):

  section_root (depth 0)
  +- entity_type_group (depth 1, "ERPs")     -- only when 2+ entities qualify
  |  +- entity_group (depth 2, "NetSuite")
  |     +- (sub_cluster | leaf)+ (depth 3)   -- recursion if leaf_count > 10
  +- entity_group (depth 1)                  -- when only one entity of a type qualifies
  +- graph_cluster (depth 1)                 -- topical cluster on leftover leaves
  +- orphan_bucket (depth 1, "Other")        -- leaves that didn't fit anything

Algorithm per section:
  1. Entity decomposition. For every entity_type in ENTITY_TYPE_SECTION, group
     section leaves by canonical slug; any slug with at least
     MIN_ENTITY_GROUP_SIZE in-section members becomes an entity_group node.
  2. Graph clustering. Take leaves not placed in any entity_group, build a
     pairwise distance matrix from combined_edges (1 - combined_weight, 1.0
     for missing pairs), and run sklearn AgglomerativeClustering with
     average linkage and a precomputed metric.
  3. Orphan bucket. Anything still unplaced lands in "Other".
  4. Recursion. Any node with leaf_count_recursive > RECURSION_LEAF_THRESHOLD
     gets one more pass of graph clustering on its own leaves, capped at
     MAX_NODE_DEPTH.
  5. Cross-listing. A leaf can appear in multiple entity_groups; its
     primary_node_id is the largest entity_group it belongs to (tie-broken
     by section priority).
  6. Decoration. Each node gets featured/supporting leaves, evidence, graph
     stats, key_themes, an llm_brief, and a quality block.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
from rich.table import Table
from sklearn.cluster import AgglomerativeClustering
from slugify import slugify

from _common import (
    COHESION_HIGH_THRESHOLD,
    COHESION_LOW_THRESHOLD,
    COMBINED_EDGES_FILE,
    EMBEDDINGS_INDEX,
    EMBEDDINGS_VECTORS,
    ENTITY_INDEX_FILE,
    ENTITY_TYPE_SECTION,
    GRAPH_CLUSTER_DISTANCE_THRESHOLD,
    LEAVES_DIR,
    MAX_NODE_DEPTH,
    MIN_ENTITY_GROUP_SIZE,
    MIN_GRAPH_CLUSTER_SIZE,
    NODES_FILE,
    OUTPUT_DIR,
    RECURSION_LEAF_THRESHOLD,
    ROOT,
    SECTIONS_ASSIGNMENT_FILE,
    SECTIONS_INDEX_FILE,
    SECTION_LABELS,
    SECTION_ORDER,
    SECTION_TREES_FILE,
    STEP6_SUMMARY_FILE,
    TIE_BREAK_PRIORITY,
    TREE_DIR,
    TREE_TAXONOMY_VERSION,
    console,
)


# ---------------------------------------------------------------------------
# Constants / small helpers
# ---------------------------------------------------------------------------


# Cap how many entity_type buckets we surface per node (keeps `evidence`
# readable and prevents pathological "every leaf shares ten entity types").
EVIDENCE_TOP_PER_TYPE = 5

# How many leaves to call out as featured/hub on every node. Section roots
# get a slightly larger window because they synthesize many subgroups.
FEATURED_LEAF_LIMIT = 3
SECTION_ROOT_FEATURED_LIMIT = 5

# How many edges to include in `graph.strongest_edges`. Mostly for human
# review of the JSON; the LLM uses it to ground "why are these together?".
STRONGEST_EDGES_LIMIT = 5

# Minimum strongest-edges average weight to call a node "tightly knit" in
# the LLM brief. Below this we suppress that descriptor.
TIGHTLY_KNIT_FLOOR = 0.55


def _slug(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return slugify(text) or None


def _ordered_pair(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a < b else (b, a)


def _section_priority(section_id: str) -> int:
    """Earlier in TIE_BREAK_PRIORITY = higher priority. Unknown sections sort last."""
    try:
        return TIE_BREAK_PRIORITY.index(section_id)
    except ValueError:
        return len(TIE_BREAK_PRIORITY) + 1


def _section_slug(section_id: str) -> str:
    """topics/sales -> topics-sales; entities/erps -> entities-erps."""
    return section_id.replace("/", "-")


def _humanize_entity_type(entity_type: str) -> str:
    return {
        "customers": "Customers",
        "competitors": "Competitors",
        "erps": "ERPs",
        "products": "Products",
        "features": "Features",
        "partners": "Partners",
        "people": "People",
        "policies": "Policies",
    }.get(entity_type, entity_type.title())


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load_json(path) -> Dict[str, Any]:
    if not path.exists():
        console.print(f"[red]Required input not found: {path.relative_to(ROOT)}[/]")
        sys.exit(1)
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_assignments() -> Dict[str, Any]:
    payload = _load_json(SECTIONS_ASSIGNMENT_FILE)
    return payload.get("assignments", {})


def _load_sections_index() -> Dict[str, Any]:
    payload = _load_json(SECTIONS_INDEX_FILE)
    return payload.get("sections", {})


def _load_entity_index() -> Dict[str, Dict[str, Dict[str, Any]]]:
    payload = _load_json(ENTITY_INDEX_FILE)
    return payload.get("index", {})


def _load_combined_edges() -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Returns a dict keyed by ordered pair -> edge record."""
    payload = _load_json(COMBINED_EDGES_FILE)
    edges: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for edge in payload.get("edges", []):
        pair = _ordered_pair(edge["from"], edge["to"])
        edges[pair] = edge
    return edges


def _load_embeddings_index() -> Dict[str, Dict[str, Any]]:
    payload = _load_json(EMBEDDINGS_INDEX)
    return payload.get("leaves", {})


def _load_leaf_metadata() -> Dict[str, Dict[str, Any]]:
    """Pulls title, audience, document_type, business_area, status, entity labels.

    We re-read the raw leaves rather than refetch through entity_index because
    we want the *non-slug* labels and the per-leaf `audience` array which step
    4 didn't index.
    """
    metadata: Dict[str, Dict[str, Any]] = {}
    for path in sorted(LEAVES_DIR.glob("*.json")):
        leaf_id = path.stem
        try:
            with path.open("r", encoding="utf-8") as fh:
                doc = json.load(fh)
        except Exception:
            continue
        classification = doc.get("classification", {}) or {}
        entities = doc.get("entities", {}) or {}
        source = doc.get("source", {}) or {}
        # Title lives at `source.name` in the leaf schema; fall back to the
        # leaf id only if that's missing (shouldn't happen post-validation).
        title = source.get("name") or doc.get("title") or leaf_id
        metadata[leaf_id] = {
            "title": title,
            "audience": [str(a) for a in classification.get("audience", []) if a],
            "business_area": classification.get("business_area"),
            "document_type": classification.get("document_type"),
            "status": classification.get("status"),
            "entities": {
                k: [str(v) for v in (entities.get(k) or []) if v]
                for k in (
                    "customers",
                    "competitors",
                    "erps",
                    "products",
                    "features",
                    "partners",
                    "people",
                    "policies",
                )
            },
        }
    return metadata


# ---------------------------------------------------------------------------
# Pair-distance / clustering primitives
# ---------------------------------------------------------------------------


def _build_distance_matrix(
    leaf_ids: List[str],
    edges: Dict[Tuple[str, str], Dict[str, Any]],
) -> np.ndarray:
    """Build a square distance matrix for the given leaf set.

    distance[i, j] = 1 - combined_weight if a (i, j) edge exists, else 1.0.
    Diagonal is 0. The matrix is symmetric and clipped to [0, 1].
    """
    n = len(leaf_ids)
    matrix = np.ones((n, n), dtype=np.float32)
    np.fill_diagonal(matrix, 0.0)
    index_of = {leaf_id: i for i, leaf_id in enumerate(leaf_ids)}
    for i, a in enumerate(leaf_ids):
        for b in leaf_ids[i + 1 :]:
            edge = edges.get(_ordered_pair(a, b))
            if not edge:
                continue
            weight = max(0.0, min(1.0, float(edge.get("combined_weight", 0.0))))
            distance = 1.0 - weight
            j = index_of[b]
            matrix[i, j] = distance
            matrix[j, i] = distance
    return matrix


def _cluster_leaves(
    leaf_ids: List[str],
    edges: Dict[Tuple[str, str], Dict[str, Any]],
    distance_threshold: float = GRAPH_CLUSTER_DISTANCE_THRESHOLD,
) -> List[List[str]]:
    """Run agglomerative clustering and return clusters as lists of leaf_ids.

    We use average linkage because section subgroups tend to be "loose"
    rather than tight cliques. With `metric="precomputed"` sklearn skips
    its own distance computation and consumes our matrix directly.

    Singletons (leaves with no neighbour above the threshold) come back as
    their own cluster. The caller decides whether to keep them or fold
    them into the orphan bucket.
    """
    if not leaf_ids:
        return []
    if len(leaf_ids) == 1:
        return [list(leaf_ids)]
    matrix = _build_distance_matrix(leaf_ids, edges)
    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="precomputed",
        linkage="average",
    )
    labels = model.fit_predict(matrix)
    by_label: Dict[int, List[str]] = defaultdict(list)
    for leaf_id, label in zip(leaf_ids, labels):
        by_label[int(label)].append(leaf_id)
    return [sorted(group) for group in by_label.values()]


# ---------------------------------------------------------------------------
# Node skeleton + ID helpers
# ---------------------------------------------------------------------------


def _make_node_id(parts: List[str]) -> str:
    """Compose a stable node id like `node-topics-sales-erps-netsuite`.

    Each part is slugified independently so dotted/spaced inputs survive.
    """
    safe = []
    for p in parts:
        s = _slug(p) if not p.startswith("entities-") and not p.startswith("topics-") else p
        if s:
            safe.append(s)
    return "node-" + "-".join(safe)


def _new_node(
    *,
    node_id: str,
    slug: str,
    title: str,
    section: str,
    parent_node_id: Optional[str],
    ancestor_path: List[str],
    depth: int,
    kind: str,
    page_role: str,
    anchor: Optional[Dict[str, str]],
) -> Dict[str, Any]:
    """Allocate a node with all expected keys present (filled later)."""
    return {
        "node_id": node_id,
        "slug": slug,
        "title": title,
        "section": section,
        "parent_node_id": parent_node_id,
        "ancestor_path": list(ancestor_path),
        "depth": depth,
        "kind": kind,
        "page_role": page_role,
        "anchor": anchor,
        "children": [],
        "leaf_ids": [],
        "leaf_ids_recursive": [],
        "leaf_count_recursive": 0,
        "primary_leaf_ids": [],
        "featured_leaf_ids": [],
        "supporting_leaf_ids": [],
        "key_themes": [],
        "summary_short": "",
        "evidence": {
            "source_leaf_ids": [],
            "source_count": 0,
            "top_entities": {},
            "audiences": [],
            "document_types": [],
        },
        "graph": {
            "hub_leaf_ids": [],
            "average_edge_weight": 0.0,
            "cohesion": 0.0,
            "edge_count": 0,
            "rescued_edge_count": 0,
            "strongest_edges": [],
        },
        "rendering": {
            "template": "",
            "show_children": True,
            "show_featured_sources": True,
            "show_all_sources": True,
            "show_related_nodes": True,
        },
        "quality": {
            "confidence": "low",
            "warnings": [],
            "needs_human_review": False,
        },
        "llm_brief": {},
        "tree_taxonomy_version": TREE_TAXONOMY_VERSION,
    }


# ---------------------------------------------------------------------------
# Tree assembly
# ---------------------------------------------------------------------------


def _build_section_root(
    section: str,
    section_leaves: List[str],
) -> Dict[str, Any]:
    section_label = SECTION_LABELS.get(section, section)
    section_slug = _section_slug(section)
    return _new_node(
        node_id=_make_node_id([section_slug]),
        slug=section_slug,
        title=section_label,
        section=section,
        parent_node_id=None,
        ancestor_path=[],
        depth=0,
        kind="section_root",
        page_role="section_landing",
        anchor=None,
    )


def _entity_groups_in_section(
    entity_type: str,
    section_leaves: Set[str],
    entity_index: Dict[str, Dict[str, Dict[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    """For one entity_type, return slug -> {label, leaf_ids} restricted to section.

    Only slugs that meet `MIN_ENTITY_GROUP_SIZE` are returned. The caller
    decides whether to also wrap them in an entity_type_group node.
    """
    bucket = entity_index.get(entity_type, {}) or {}
    qualifying: Dict[str, Dict[str, Any]] = {}
    for slug, record in bucket.items():
        in_section = [lid for lid in record.get("leaf_ids", []) if lid in section_leaves]
        if len(in_section) >= MIN_ENTITY_GROUP_SIZE:
            qualifying[slug] = {
                "label": record.get("label") or slug,
                "leaf_ids": sorted(in_section),
            }
    return qualifying


def _build_entity_group_node(
    *,
    section: str,
    entity_type: str,
    slug: str,
    label: str,
    leaf_ids: List[str],
    parent_node_id: str,
    parent_ancestors: List[str],
    depth: int,
) -> Dict[str, Any]:
    section_slug = _section_slug(section)
    node_id = _make_node_id([section_slug, entity_type, slug])
    node = _new_node(
        node_id=node_id,
        slug=slug,
        title=label,
        section=section,
        parent_node_id=parent_node_id,
        ancestor_path=parent_ancestors + [parent_node_id],
        depth=depth,
        kind="entity_group",
        page_role="cluster_landing",
        anchor={"entity_type": entity_type, "slug": slug, "label": label},
    )
    node["leaf_ids"] = sorted(leaf_ids)
    return node


def _build_entity_type_group_node(
    *,
    section: str,
    entity_type: str,
    parent_node_id: str,
    parent_ancestors: List[str],
) -> Dict[str, Any]:
    section_slug = _section_slug(section)
    node_id = _make_node_id([section_slug, entity_type])
    title = f"{_humanize_entity_type(entity_type)} ({SECTION_LABELS.get(section, section)})"
    return _new_node(
        node_id=node_id,
        slug=entity_type,
        title=title,
        section=section,
        parent_node_id=parent_node_id,
        ancestor_path=parent_ancestors + [parent_node_id],
        depth=1,
        kind="entity_type_group",
        page_role="entity_index",
        anchor={"entity_type": entity_type},
    )


def _build_graph_cluster_node(
    *,
    section: str,
    cluster_index: int,
    leaf_ids: List[str],
    parent_node_id: str,
    parent_ancestors: List[str],
    depth: int,
    leaf_metadata: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Topical cluster (not bound to any single entity).

    The node's slug/title is `cluster-N`; the human-readable title gets
    refined in the decoration pass once we know the dominant theme.
    """
    section_slug = _section_slug(section)
    slug = f"cluster-{cluster_index:02d}"
    node_id = _make_node_id([section_slug] + (parent_ancestors[1:] if depth > 1 else []) + [slug])
    node = _new_node(
        node_id=node_id,
        slug=slug,
        title=f"Cluster {cluster_index}",
        section=section,
        parent_node_id=parent_node_id,
        ancestor_path=parent_ancestors + [parent_node_id],
        depth=depth,
        kind="graph_cluster",
        page_role="cluster_landing",
        anchor=None,
    )
    node["leaf_ids"] = sorted(leaf_ids)
    return node


def _build_recursion_cluster_node(
    *,
    parent: Dict[str, Any],
    cluster_index: int,
    leaf_ids: List[str],
    leaf_metadata: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Sub-cluster created when an oversized node was split.

    The slug carries the parent's slug for traceability. Title gets refined
    downstream once themes are computed.
    """
    parent_slug = parent["slug"]
    slug = f"{parent_slug}-cluster-{cluster_index:02d}"
    section_slug = _section_slug(parent["section"])
    # Anchor onto the parent so node ids are unique even across recursion.
    node_id = _make_node_id([section_slug, parent_slug, f"sub-{cluster_index:02d}"])
    node = _new_node(
        node_id=node_id,
        slug=slug,
        title=f"{parent['title']} — group {cluster_index}",
        section=parent["section"],
        parent_node_id=parent["node_id"],
        ancestor_path=parent["ancestor_path"] + [parent["node_id"]],
        depth=parent["depth"] + 1,
        kind="graph_cluster",
        page_role="cluster_landing",
        anchor=None,
    )
    node["leaf_ids"] = sorted(leaf_ids)
    return node


def _build_orphan_bucket_node(
    *,
    section: str,
    leaf_ids: List[str],
    parent_node_id: str,
    parent_ancestors: List[str],
    depth: int,
) -> Dict[str, Any]:
    """Catch-all bucket for leaves that didn't fit any cluster.

    The slug is always "other" so the node id is stable and findable in
    step 7 (e.g. `node-topics-sales-other`).
    """
    section_slug = _section_slug(section)
    slug = "other"
    node_id = _make_node_id([section_slug] + (parent_ancestors[1:] if depth > 1 else []) + [slug])
    node = _new_node(
        node_id=node_id,
        slug=slug,
        title="Other",
        section=section,
        parent_node_id=parent_node_id,
        ancestor_path=parent_ancestors + [parent_node_id],
        depth=depth,
        kind="orphan_bucket",
        page_role="other_listing",
        anchor=None,
    )
    node["leaf_ids"] = sorted(leaf_ids)
    return node


def _maybe_recurse_into(
    parent: Dict[str, Any],
    *,
    edges: Dict[Tuple[str, str], Dict[str, Any]],
    leaf_metadata: Dict[str, Dict[str, Any]],
    nodes_out: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """If `parent` is too big to read, split it into sub-clusters once.

    Returns the new sub-nodes (so the caller can push them into the global
    node registry). Mutates `parent`'s `children` and `leaf_ids` in place
    if a useful split was found; otherwise leaves them alone and returns
    an empty list.
    """
    direct_leaves = list(parent.get("leaf_ids", []))
    if not direct_leaves:
        return []
    if parent["depth"] >= MAX_NODE_DEPTH:
        return []
    if len(direct_leaves) <= RECURSION_LEAF_THRESHOLD:
        return []

    clusters = _cluster_leaves(direct_leaves, edges)
    real_clusters = [c for c in clusters if len(c) >= MIN_GRAPH_CLUSTER_SIZE]
    if len(real_clusters) < 2:
        # Only one big cluster — splitting wouldn't actually help readability.
        return []

    # Sort clusters deterministically by size (desc) then leading leaf id.
    real_clusters.sort(key=lambda c: (-len(c), c[0]))
    leftover = [lid for c in clusters if len(c) < MIN_GRAPH_CLUSTER_SIZE for lid in c]

    new_subnodes: List[Dict[str, Any]] = []
    for idx, members in enumerate(real_clusters, start=1):
        sub = _build_recursion_cluster_node(
            parent=parent,
            cluster_index=idx,
            leaf_ids=members,
            leaf_metadata=leaf_metadata,
        )
        new_subnodes.append(sub)
        nodes_out[sub["node_id"]] = sub

    # The parent now exposes sub-nodes as its primary navigation; keep
    # leftovers as direct leaf children alongside the sub-nodes (mixed
    # children, as discussed in the design).
    parent["children"] = (
        [{"kind": "node", "node_id": s["node_id"], "relationship": "subtopic"} for s in new_subnodes]
        + [{"kind": "leaf", "leaf_id": lid, "relationship": "source"} for lid in leftover]
    )
    parent["leaf_ids"] = sorted(leftover)
    return new_subnodes


def _build_section_tree(
    section: str,
    section_leaves: List[str],
    *,
    entity_index: Dict[str, Dict[str, Dict[str, Any]]],
    edges: Dict[Tuple[str, str], Dict[str, Any]],
    leaf_metadata: Dict[str, Dict[str, Any]],
    nodes_out: Dict[str, Dict[str, Any]],
    section_stats: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the full sub-tree for one section. Returns the section_root node."""
    section_set = set(section_leaves)
    root = _build_section_root(section, section_leaves)
    nodes_out[root["node_id"]] = root

    placed: Set[str] = set()  # leaves placed in any entity_group
    top_level_children: List[Dict[str, Any]] = []
    section_stats["entity_groups"] = 0
    section_stats["entity_type_groups"] = 0
    section_stats["graph_clusters"] = 0
    section_stats["orphan_bucket_size"] = 0
    section_stats["recursive_splits"] = 0
    section_stats["entity_groups_by_type"] = {}

    # 1) Entity decomposition pass, in a stable order.
    for entity_type in ENTITY_TYPE_SECTION:
        qualifying = _entity_groups_in_section(entity_type, section_set, entity_index)
        if not qualifying:
            continue

        section_stats["entity_groups_by_type"][entity_type] = len(qualifying)
        section_stats["entity_groups"] += len(qualifying)

        if len(qualifying) == 1:
            # Single qualifying entity — promote directly to depth 1 instead
            # of wrapping in an entity_type_group with one child (would be
            # awkward UI for "ERPs > NetSuite" if that's the only ERP).
            slug, info = next(iter(qualifying.items()))
            node = _build_entity_group_node(
                section=section,
                entity_type=entity_type,
                slug=slug,
                label=info["label"],
                leaf_ids=info["leaf_ids"],
                parent_node_id=root["node_id"],
                parent_ancestors=[],
                depth=1,
            )
            top_level_children.append(node)
            nodes_out[node["node_id"]] = node
            placed.update(info["leaf_ids"])
            continue

        # 2+ qualifying entities — wrap in an entity_type_group.
        type_node = _build_entity_type_group_node(
            section=section,
            entity_type=entity_type,
            parent_node_id=root["node_id"],
            parent_ancestors=[],
        )
        section_stats["entity_type_groups"] += 1
        nodes_out[type_node["node_id"]] = type_node
        top_level_children.append(type_node)

        # Stable child ordering by size (desc) then label.
        ordered = sorted(
            qualifying.items(),
            key=lambda kv: (-len(kv[1]["leaf_ids"]), kv[1]["label"].lower()),
        )
        type_node_child_refs: List[Dict[str, Any]] = []
        for slug, info in ordered:
            child = _build_entity_group_node(
                section=section,
                entity_type=entity_type,
                slug=slug,
                label=info["label"],
                leaf_ids=info["leaf_ids"],
                parent_node_id=type_node["node_id"],
                parent_ancestors=[root["node_id"]],
                depth=2,
            )
            nodes_out[child["node_id"]] = child
            type_node_child_refs.append(
                {"kind": "node", "node_id": child["node_id"], "relationship": "subtopic"}
            )
            placed.update(info["leaf_ids"])

            # Recurse if the entity_group itself is too large.
            sub_added = _maybe_recurse_into(
                child,
                edges=edges,
                leaf_metadata=leaf_metadata,
                nodes_out=nodes_out,
            )
            if sub_added:
                section_stats["recursive_splits"] += 1

        type_node["children"] = type_node_child_refs

    # 2) Graph clustering pass on whatever's left.
    leftover = [lid for lid in section_leaves if lid not in placed]
    orphan_pool: List[str] = []
    if leftover:
        clusters = _cluster_leaves(leftover, edges)
        clusters.sort(key=lambda c: (-len(c), c[0]))
        cluster_idx = 1
        for members in clusters:
            if len(members) >= MIN_GRAPH_CLUSTER_SIZE:
                node = _build_graph_cluster_node(
                    section=section,
                    cluster_index=cluster_idx,
                    leaf_ids=members,
                    parent_node_id=root["node_id"],
                    parent_ancestors=[],
                    depth=1,
                    leaf_metadata=leaf_metadata,
                )
                nodes_out[node["node_id"]] = node
                top_level_children.append(node)
                section_stats["graph_clusters"] += 1
                placed.update(members)
                cluster_idx += 1

                sub_added = _maybe_recurse_into(
                    node,
                    edges=edges,
                    leaf_metadata=leaf_metadata,
                    nodes_out=nodes_out,
                )
                if sub_added:
                    section_stats["recursive_splits"] += 1
            else:
                orphan_pool.extend(members)

    # 3) Orphan bucket — even one orphan gets a home so nothing is lost.
    if orphan_pool:
        orphan = _build_orphan_bucket_node(
            section=section,
            leaf_ids=sorted(orphan_pool),
            parent_node_id=root["node_id"],
            parent_ancestors=[],
            depth=1,
        )
        nodes_out[orphan["node_id"]] = orphan
        top_level_children.append(orphan)
        section_stats["orphan_bucket_size"] = len(orphan_pool)
        placed.update(orphan_pool)

    # Set root children references in order: entity-led first, topical
    # clusters next, orphan bucket last (matches reading order in the UI).
    def _child_sort_key(n: Dict[str, Any]) -> Tuple[int, int, str]:
        kind_rank = {
            "entity_type_group": 0,
            "entity_group": 1,
            "graph_cluster": 2,
            "orphan_bucket": 3,
        }.get(n["kind"], 4)
        return (kind_rank, -len(n.get("leaf_ids", [])), n["title"].lower())

    top_level_children.sort(key=_child_sort_key)
    root["children"] = [
        {"kind": "node", "node_id": n["node_id"], "relationship": "subtopic"}
        for n in top_level_children
    ]

    # Track which leaves never made it into ANY child (should be empty if
    # the orphan bucket fired correctly).
    section_stats["section_leaf_count"] = len(section_set)
    section_stats["unplaced_leaf_ids"] = sorted(section_set - placed)
    return root


# ---------------------------------------------------------------------------
# Decoration pass (themes, hubs, evidence, graph stats, llm_brief, quality)
# ---------------------------------------------------------------------------


def _resolve_leaf_children(node: Dict[str, Any]) -> List[str]:
    """Return leaf_ids that hang directly off this node (post-recursion).

    For terminal nodes this is everything. For nodes that were split, only
    the leftover loose leaves remain at this level (sub-clusters carry the
    rest).
    """
    return list(node.get("leaf_ids", []))


def _collect_leaves_recursive(
    node: Dict[str, Any],
    nodes: Dict[str, Dict[str, Any]],
) -> List[str]:
    """Union of all leaf_ids reachable through this node, deduped, sorted."""
    seen: Set[str] = set(node.get("leaf_ids", []))
    for child in node.get("children", []):
        if child.get("kind") == "leaf":
            lid = child.get("leaf_id")
            if lid:
                seen.add(lid)
        elif child.get("kind") == "node":
            sub = nodes.get(child["node_id"])
            if not sub:
                continue
            for lid in _collect_leaves_recursive(sub, nodes):
                seen.add(lid)
    return sorted(seen)


def _intra_node_edges(
    leaf_ids: Iterable[str],
    edges: Dict[Tuple[str, str], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """All combined edges whose endpoints are both inside `leaf_ids`."""
    leaf_set = set(leaf_ids)
    out: List[Dict[str, Any]] = []
    for a, b in combinations(sorted(leaf_set), 2):
        edge = edges.get(_ordered_pair(a, b))
        if edge:
            out.append(edge)
    return out


def _weighted_degree(
    leaf_id: str,
    leaves_in_node: Set[str],
    edges: Dict[Tuple[str, str], Dict[str, Any]],
) -> float:
    """Sum of combined_weight to every other in-node leaf. The hub metric."""
    total = 0.0
    for other in leaves_in_node:
        if other == leaf_id:
            continue
        edge = edges.get(_ordered_pair(leaf_id, other))
        if edge:
            total += float(edge.get("combined_weight", 0.0))
    return total


def _pick_hubs(
    leaves_in_node: List[str],
    edges: Dict[Tuple[str, str], Dict[str, Any]],
    leaf_metadata: Dict[str, Dict[str, Any]],
    limit: int,
) -> List[str]:
    """Top `limit` leaves by weighted-degree within the node.

    Tie-break: shorter title first (more "primary"-feeling), then leaf_id.
    Singleton nodes return their lone leaf.
    """
    if not leaves_in_node:
        return []
    if len(leaves_in_node) == 1:
        return list(leaves_in_node)
    leaf_set = set(leaves_in_node)
    scored = []
    for lid in leaves_in_node:
        title = (leaf_metadata.get(lid) or {}).get("title", lid)
        scored.append((_weighted_degree(lid, leaf_set, edges), len(title), lid))
    scored.sort(key=lambda t: (-t[0], t[1], t[2]))
    return [lid for _, _, lid in scored[:limit]]


def _aggregate_entities(
    leaf_ids: Iterable[str],
    leaf_metadata: Dict[str, Dict[str, Any]],
    *,
    skip: Optional[Tuple[str, str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Count entity labels across `leaf_ids` and return top-N per type.

    `skip` lets us exclude the anchor entity from an entity_group's own
    evidence (otherwise NetSuite always lists "NetSuite" as its top ERP,
    which is noise).
    """
    counters: Dict[str, Counter] = defaultdict(Counter)
    for lid in leaf_ids:
        meta = leaf_metadata.get(lid)
        if not meta:
            continue
        for entity_type, labels in meta.get("entities", {}).items():
            for label in labels:
                slug = _slug(label)
                if not slug:
                    continue
                if skip and skip[0] == entity_type and skip[1] == slug:
                    continue
                counters[entity_type][label] += 1

    top_per_type: Dict[str, List[Dict[str, Any]]] = {}
    for entity_type, counter in counters.items():
        top = counter.most_common(EVIDENCE_TOP_PER_TYPE)
        if not top:
            continue
        top_per_type[entity_type] = [
            {"label": label, "count": count} for label, count in top
        ]
    return top_per_type


def _aggregate_audiences(
    leaf_ids: Iterable[str],
    leaf_metadata: Dict[str, Dict[str, Any]],
) -> List[str]:
    counter: Counter = Counter()
    for lid in leaf_ids:
        for aud in (leaf_metadata.get(lid) or {}).get("audience", []):
            slug = _slug(aud)
            if slug:
                counter[slug] += 1
    return [aud for aud, _ in counter.most_common(5)]


def _aggregate_doc_types(
    leaf_ids: Iterable[str],
    leaf_metadata: Dict[str, Dict[str, Any]],
) -> List[str]:
    counter: Counter = Counter()
    for lid in leaf_ids:
        dt = (leaf_metadata.get(lid) or {}).get("document_type")
        slug = _slug(dt)
        if slug:
            counter[slug] += 1
    return [dt for dt, _ in counter.most_common(5)]


def _build_key_themes(
    node: Dict[str, Any],
    top_entities: Dict[str, List[Dict[str, Any]]],
    audiences: List[str],
    doc_types: List[str],
) -> List[str]:
    """Pick up to 4 short, human-readable themes for the LLM brief.

    Priority: anchor entity > top entities (any type) > audience > doc_type.
    """
    themes: List[str] = []

    if node.get("anchor") and node["anchor"].get("label"):
        themes.append(str(node["anchor"]["label"]))

    # Entity types that tend to make good titles, in priority order.
    priority_types = ["customers", "competitors", "erps", "products", "partners", "features"]
    for et in priority_types:
        for entry in top_entities.get(et, [])[:2]:
            label = entry["label"]
            if label not in themes:
                themes.append(label)
            if len(themes) >= 4:
                return themes

    for dt in doc_types:
        pretty = dt.replace("-", " ").title()
        if pretty not in themes:
            themes.append(pretty)
        if len(themes) >= 4:
            return themes

    for aud in audiences:
        pretty = aud.replace("-", " ").title()
        if pretty not in themes:
            themes.append(pretty)
        if len(themes) >= 4:
            return themes

    return themes[:4]


def _refine_cluster_title(
    node: Dict[str, Any],
    themes: List[str],
) -> str:
    """Replace generic 'Cluster N' / 'group N' titles once we know themes."""
    if node["kind"] != "graph_cluster" or not themes:
        return node["title"]
    primary = themes[0]
    section_label = SECTION_LABELS.get(node["section"], node["section"])
    if node["depth"] == 1:
        return f"{primary} ({section_label})"
    return primary


def _build_summary_short(
    node: Dict[str, Any],
    leaf_ids_recursive: List[str],
    themes: List[str],
) -> str:
    """One-line teaser used as the node's tagline.

    The LLM doesn't have to use this verbatim; it's a starting point and a
    sanity check that the node is coherent at all.
    """
    n = len(leaf_ids_recursive)
    section_label = SECTION_LABELS.get(node["section"], node["section"])
    label_word = "document" if n == 1 else "documents"

    if node["kind"] == "section_root":
        return f"{n} {label_word} across {section_label}."
    if node["kind"] == "entity_type_group":
        et = node["anchor"]["entity_type"] if node.get("anchor") else "entities"
        return f"{n} {label_word} grouped by {_humanize_entity_type(et).lower()} in {section_label}."
    if node["kind"] == "orphan_bucket":
        return f"{n} {label_word} that didn't fit a named cluster."

    theme_phrase = ", ".join(themes[:3]) if themes else "this cluster"
    return f"{n} {label_word} about {theme_phrase}."


def _build_llm_brief(
    node: Dict[str, Any],
    *,
    leaves_for_brief: List[str],
    themes: List[str],
    audiences: List[str],
    leaf_metadata: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """The rendering contract: instructions + grounded evidence for the LLM.

    This is the single most important field for the agent that writes
    upper-node HTML. It tells the agent *why* the node exists and *how* to
    write the page; it does NOT pre-write the summary.
    """
    section_label = SECTION_LABELS.get(node["section"], node["section"])
    title = node["title"]

    if node["kind"] == "section_root":
        purpose = (
            f"Create the section landing page for {section_label}. "
            "Orient the reader to the subtopics in this section."
        )
        primary_question = f"What does Paystand publish about {section_label}?"
    elif node["kind"] == "entity_type_group":
        et = _humanize_entity_type(node["anchor"]["entity_type"]) if node.get("anchor") else "this group"
        purpose = (
            f"Create an index page for {et} within {section_label}. "
            "Briefly describe each child entity so readers can pick where to dive."
        )
        primary_question = f"Which {et.lower()} matter to Paystand and where are they used?"
    elif node["kind"] == "entity_group":
        anchor_label = node["anchor"]["label"] if node.get("anchor") else title
        purpose = (
            f"Create a reader-facing landing page for {anchor_label} within {section_label}."
        )
        primary_question = (
            f"What should a Paystand employee understand about {anchor_label}?"
        )
    elif node["kind"] == "orphan_bucket":
        purpose = (
            f"Create a fallback listing for {section_label} documents that didn't "
            "fit a named cluster. Group them by document type or theme if a pattern is obvious."
        )
        primary_question = f"What other {section_label} documents exist that don't have a cluster home?"
    else:  # graph_cluster
        purpose = (
            f"Create a reader-facing landing page that explains the {title} "
            f"theme within {section_label}."
        )
        primary_question = f"What should a Paystand employee understand about {title}?"

    featured_titles = []
    for lid in leaves_for_brief:
        meta = leaf_metadata.get(lid)
        if not meta:
            continue
        featured_titles.append({"leaf_id": lid, "title": meta.get("title", lid)})

    must_include = [
        "What this topic is and why it matters to Paystand",
        "The most important child pages or source files",
        "Common themes or decisions across the source files",
        "A list of source documents at the end",
    ]
    if node["kind"] == "section_root" or node["kind"] == "entity_type_group":
        must_include = [
            "A short intro that explains the section",
            "A scannable list of the major subtopics",
            "Pointers to the most important child pages",
        ]

    suggested_focus = list(themes)
    return {
        "purpose": purpose,
        "audience": audiences if audiences else ["unspecified"],
        "primary_question": primary_question,
        "summary_instruction": (
            "Synthesize across the child leaves. Do not summarize each file "
            "one by one unless needed. Prefer themes, patterns, decisions, "
            "reusable context, and links to the most useful source pages."
        ),
        "source_policy": (
            "Only make claims supported by leaf summaries or source files. "
            "Cite leaf or source titles at the end. If evidence is weak, say so."
        ),
        "must_include": must_include,
        "suggested_focus": suggested_focus,
        "featured_sources": featured_titles,
    }


def _decorate_node(
    node: Dict[str, Any],
    *,
    nodes: Dict[str, Dict[str, Any]],
    edges: Dict[Tuple[str, str], Dict[str, Any]],
    leaf_metadata: Dict[str, Dict[str, Any]],
) -> None:
    """Fill in everything that depends on the assembled tree.

    Bottom-up: the section_root's children are decorated first so we can
    consult sub-node `featured_leaf_ids` when picking section-level hubs.
    The actual ordering is enforced by the caller.
    """
    direct_leaves = _resolve_leaf_children(node)
    leaves_recursive = _collect_leaves_recursive(node, nodes)
    node["leaf_ids_recursive"] = leaves_recursive
    node["leaf_count_recursive"] = len(leaves_recursive)

    # Children list normalisation. Leaf-only terminal nodes get explicit
    # leaf-refs so step 7 doesn't have to look at `leaf_ids` separately.
    has_node_children = any(c.get("kind") == "node" for c in node.get("children", []))
    if not has_node_children and direct_leaves:
        node["children"] = [
            {"kind": "leaf", "leaf_id": lid, "relationship": "source"}
            for lid in direct_leaves
        ]

    # Featured / hub leaves. For non-leaf-bearing nodes (only sub-nodes),
    # we lift hubs from the recursive set so the LLM brief still has
    # concrete sources to cite.
    feature_pool = direct_leaves if direct_leaves else leaves_recursive
    feature_limit = (
        SECTION_ROOT_FEATURED_LIMIT if node["kind"] == "section_root" else FEATURED_LEAF_LIMIT
    )
    featured = _pick_hubs(feature_pool, edges, leaf_metadata, feature_limit)
    node["featured_leaf_ids"] = featured
    node["supporting_leaf_ids"] = [lid for lid in feature_pool if lid not in featured]

    # Evidence — entity rollup, audiences, doc_types. Skip the node's own
    # anchor entity to avoid noise like "NetSuite mentions NetSuite".
    skip = None
    if node.get("anchor") and node["anchor"].get("entity_type") and node["anchor"].get("slug"):
        skip = (node["anchor"]["entity_type"], node["anchor"]["slug"])
    top_entities = _aggregate_entities(leaves_recursive, leaf_metadata, skip=skip)
    audiences = _aggregate_audiences(leaves_recursive, leaf_metadata)
    doc_types = _aggregate_doc_types(leaves_recursive, leaf_metadata)
    node["evidence"] = {
        "source_leaf_ids": leaves_recursive,
        "source_count": len(leaves_recursive),
        "top_entities": top_entities,
        "audiences": audiences,
        "document_types": doc_types,
    }

    # Graph stats — limited to direct intra-cluster edges (sub-cluster
    # cohesion is the property of the sub-cluster, not the parent).
    intra_pool = direct_leaves if direct_leaves else leaves_recursive
    edge_records = _intra_node_edges(intra_pool, edges)
    if edge_records:
        weights = [float(e.get("combined_weight", 0.0)) for e in edge_records]
        avg = sum(weights) / len(weights)
        rescued_count = sum(1 for e in edge_records if e.get("rescued"))
        sorted_edges = sorted(
            edge_records,
            key=lambda e: float(e.get("combined_weight", 0.0)),
            reverse=True,
        )[:STRONGEST_EDGES_LIMIT]
        strongest = []
        for e in sorted_edges:
            top_reason = next(
                (r for r in e.get("reasons", []) if r.get("type") != "similarity"),
                None,
            )
            reason_text = "shared similarity"
            if top_reason:
                rtype = top_reason.get("type", "").replace("shared_", "shared ")
                rlabel = top_reason.get("shared_label") or top_reason.get("label") or ""
                reason_text = f"{rtype}: {rlabel}".strip(": ")
            strongest.append(
                {
                    "from": e["from"],
                    "to": e["to"],
                    "weight": round(float(e.get("combined_weight", 0.0)), 4),
                    "reason": reason_text,
                }
            )
        node["graph"] = {
            "hub_leaf_ids": featured,
            "average_edge_weight": round(avg, 4),
            "cohesion": round(avg, 4),
            "edge_count": len(edge_records),
            "rescued_edge_count": rescued_count,
            "strongest_edges": strongest,
        }
    else:
        node["graph"] = {
            "hub_leaf_ids": featured,
            "average_edge_weight": 0.0,
            "cohesion": 0.0,
            "edge_count": 0,
            "rescued_edge_count": 0,
            "strongest_edges": [],
        }

    # Themes / summary / llm_brief.
    themes = _build_key_themes(node, top_entities, audiences, doc_types)
    node["key_themes"] = themes
    node["title"] = _refine_cluster_title(node, themes)
    node["summary_short"] = _build_summary_short(node, leaves_recursive, themes)

    leaves_for_brief = featured if featured else leaves_recursive[:FEATURED_LEAF_LIMIT]
    node["llm_brief"] = _build_llm_brief(
        node,
        leaves_for_brief=leaves_for_brief,
        themes=themes,
        audiences=audiences,
        leaf_metadata=leaf_metadata,
    )

    # Rendering hints — vary by node kind.
    template_by_kind = {
        "section_root": "section_landing",
        "entity_type_group": "entity_index",
        "entity_group": "cluster_landing",
        "graph_cluster": "cluster_landing",
        "orphan_bucket": "other_listing",
    }
    node["rendering"] = {
        "template": template_by_kind.get(node["kind"], "cluster_landing"),
        "show_children": True,
        "show_featured_sources": bool(featured),
        "show_all_sources": node["kind"] != "section_root",
        "show_related_nodes": node["depth"] > 0,
    }

    # Quality / confidence band.
    cohesion = node["graph"]["cohesion"]
    n = node["leaf_count_recursive"]
    warnings: List[str] = []
    if n < MIN_GRAPH_CLUSTER_SIZE and node["kind"] not in {"section_root", "entity_type_group"}:
        warnings.append("very_small_node")
    if cohesion == 0.0 and n > 1 and node["kind"] not in {"section_root", "entity_type_group"}:
        warnings.append("no_internal_edges")
    if cohesion >= COHESION_HIGH_THRESHOLD and n >= MIN_GRAPH_CLUSTER_SIZE:
        confidence = "high"
    elif cohesion >= COHESION_LOW_THRESHOLD or node["kind"] in {"section_root", "entity_type_group", "entity_group"}:
        confidence = "medium"
    else:
        confidence = "low"
    needs_review = (
        node["kind"] == "graph_cluster" and cohesion < COHESION_LOW_THRESHOLD
    ) or (node["kind"] == "orphan_bucket" and n >= RECURSION_LEAF_THRESHOLD)
    node["quality"] = {
        "confidence": confidence,
        "warnings": warnings,
        "needs_human_review": needs_review,
    }


def _decorate_tree_bottom_up(
    section_root: Dict[str, Any],
    nodes: Dict[str, Dict[str, Any]],
    *,
    edges: Dict[Tuple[str, str], Dict[str, Any]],
    leaf_metadata: Dict[str, Dict[str, Any]],
) -> None:
    """Decorate every node under a section root, deepest first."""
    visited: List[Dict[str, Any]] = []

    def walk(n: Dict[str, Any]) -> None:
        for child in n.get("children", []):
            if child.get("kind") == "node":
                sub = nodes.get(child["node_id"])
                if sub:
                    walk(sub)
        visited.append(n)

    walk(section_root)
    for n in visited:
        _decorate_node(n, nodes=nodes, edges=edges, leaf_metadata=leaf_metadata)


# ---------------------------------------------------------------------------
# Leaf placements (cross-listing primary_node_id)
# ---------------------------------------------------------------------------


def _compute_leaf_placements(
    nodes: Dict[str, Dict[str, Any]],
    leaf_primary_section: Dict[str, str],
) -> Dict[str, Dict[str, Any]]:
    """For every leaf, decide which node is its 'primary' home.

    Rules (in priority order):
      1. Prefer nodes inside the leaf's `primary_section` (chosen by step 5).
         Within that section, pick the largest entity_group / graph_cluster.
      2. If no entity_group / graph_cluster in primary_section claims the
         leaf, fall back to the orphan_bucket of primary_section.
      3. If primary_section has nothing for this leaf (rare — leaf below
         min thresholds everywhere), fall back to the largest entity_group
         / graph_cluster in any other section.
      4. Tie-break: larger leaf_count_recursive, then section priority,
         then node_id alphabetical (determinism).

    Section_root / entity_type_group nodes never count as primary homes;
    they are wrappers, not concrete cluster pages.
    """
    appearances: Dict[str, List[str]] = defaultdict(list)
    primary_pool_in_section: Dict[str, List[str]] = defaultdict(list)
    primary_pool_anywhere: Dict[str, List[str]] = defaultdict(list)
    orphan_in_section: Dict[str, List[str]] = defaultdict(list)
    orphan_anywhere: Dict[str, List[str]] = defaultdict(list)

    for node_id, node in nodes.items():
        if node["kind"] in {"section_root", "entity_type_group"}:
            continue
        for lid in node["leaf_ids_recursive"]:
            appearances[lid].append(node_id)
            in_primary_section = node["section"] == leaf_primary_section.get(lid)
            if node["kind"] == "orphan_bucket":
                if in_primary_section:
                    orphan_in_section[lid].append(node_id)
                else:
                    orphan_anywhere[lid].append(node_id)
            else:
                if in_primary_section:
                    primary_pool_in_section[lid].append(node_id)
                else:
                    primary_pool_anywhere[lid].append(node_id)

    def pick(pool: List[str]) -> str:
        pool.sort(
            key=lambda nid: (
                -nodes[nid]["leaf_count_recursive"],
                _section_priority(nodes[nid]["section"]),
                nid,
            )
        )
        return pool[0]

    placements: Dict[str, Dict[str, Any]] = {}
    all_lids = set(appearances.keys())
    for lid in all_lids:
        chosen: Optional[str] = None
        for pool in (
            primary_pool_in_section.get(lid),
            orphan_in_section.get(lid),
            primary_pool_anywhere.get(lid),
            orphan_anywhere.get(lid),
        ):
            if pool:
                chosen = pick(pool)
                break
        if chosen is None:
            continue
        placements[lid] = {
            "primary_node_id": chosen,
            "primary_section": nodes[chosen]["section"],
            "appears_in_node_ids": sorted(appearances[lid]),
        }

    return placements


# ---------------------------------------------------------------------------
# Tree validation (catch structural bugs before step 7 inherits them)
# ---------------------------------------------------------------------------


def _validate_tree(
    nodes: Dict[str, Dict[str, Any]],
    section_roots: Dict[str, str],
    leaf_metadata: Dict[str, Dict[str, Any]],
    section_to_leaves: Dict[str, List[str]],
) -> List[str]:
    """Return a list of validation warnings (empty = clean run)."""
    warnings: List[str] = []
    seen_node_ids: Set[str] = set()

    for node_id, node in nodes.items():
        if node_id != node["node_id"]:
            warnings.append(f"node_id mismatch: registry key {node_id} != node.node_id {node['node_id']}")
        if node_id in seen_node_ids:
            warnings.append(f"duplicate node_id: {node_id}")
        seen_node_ids.add(node_id)

        if node["depth"] > MAX_NODE_DEPTH:
            warnings.append(f"{node_id}: depth {node['depth']} exceeds MAX_NODE_DEPTH={MAX_NODE_DEPTH}")

        for child in node.get("children", []):
            if child.get("kind") == "node":
                cid = child.get("node_id")
                if cid not in nodes:
                    warnings.append(f"{node_id}: child references missing node {cid}")
                else:
                    child_node = nodes[cid]
                    if child_node["parent_node_id"] != node_id:
                        warnings.append(
                            f"{node_id}: child {cid} has parent_node_id={child_node['parent_node_id']}"
                        )
            elif child.get("kind") == "leaf":
                lid = child.get("leaf_id")
                if lid and lid not in leaf_metadata:
                    warnings.append(f"{node_id}: leaf {lid} not in leaf metadata")

    # Every section's leaf must show up in its section_root's leaf_ids_recursive.
    for section, leaves in section_to_leaves.items():
        root_id = section_roots.get(section)
        if not root_id:
            continue
        root_recursive = set(nodes[root_id]["leaf_ids_recursive"])
        missing = [lid for lid in leaves if lid not in root_recursive]
        if missing:
            warnings.append(
                f"section {section}: {len(missing)} leaves not reachable from root "
                f"(first 3: {missing[:3]})"
            )

    return warnings


# ---------------------------------------------------------------------------
# Section trees (the navigation index)
# ---------------------------------------------------------------------------


def _build_section_tree_view(
    section_root: Dict[str, Any],
    nodes: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """The compact per-section view step 7 walks for navigation."""

    def node_summary(n: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "node_id": n["node_id"],
            "title": n["title"],
            "kind": n["kind"],
            "depth": n["depth"],
            "leaf_count_recursive": n["leaf_count_recursive"],
            "anchor": n.get("anchor"),
        }

    primary_children = []
    for child_ref in section_root.get("children", []):
        if child_ref.get("kind") != "node":
            continue
        sub = nodes.get(child_ref["node_id"])
        if sub:
            primary_children.append(node_summary(sub))

    return {
        "root_node_id": section_root["node_id"],
        "label": SECTION_LABELS.get(section_root["section"], section_root["section"]),
        "primary_children": primary_children,
        "leaf_count_recursive": section_root["leaf_count_recursive"],
    }


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-section breakdown table at the end.",
    )
    args = parser.parse_args()

    console.print("[bold cyan]Step 6 — discover clusters[/]")

    # Load everything up front.
    assignments = _load_assignments()
    sections_index = _load_sections_index()
    entity_index = _load_entity_index()
    edges = _load_combined_edges()
    leaf_metadata = _load_leaf_metadata()
    embeddings_index = _load_embeddings_index()

    if not assignments:
        console.print("[red]No assignments found. Run assign_sections.py first.[/]")
        sys.exit(1)
    if not edges:
        console.print("[yellow]combined_edges.json is empty; clusters will rely on sparse links only.[/]")

    # Map section -> leaf_ids using sections_index (already curated by step 5).
    section_to_leaves: Dict[str, List[str]] = {
        section: list(payload.get("leaf_ids", []))
        for section, payload in sections_index.items()
        if payload.get("leaf_ids")
    }
    if not section_to_leaves:
        console.print("[red]sections_index.json is empty.[/]")
        sys.exit(1)

    console.print(
        f"  loaded {len(assignments)} assignments, {len(section_to_leaves)} non-empty sections, "
        f"{len(edges)} combined edges"
    )

    # Build the trees.
    nodes: Dict[str, Dict[str, Any]] = {}
    section_roots: Dict[str, str] = {}
    section_stats_all: Dict[str, Dict[str, Any]] = {}
    sections_in_canonical_order = [s for s in SECTION_ORDER if s in section_to_leaves]
    extras = [s for s in section_to_leaves if s not in sections_in_canonical_order]
    sections_in_canonical_order.extend(sorted(extras))

    for section in sections_in_canonical_order:
        leaves = sorted(section_to_leaves[section])
        section_stats: Dict[str, Any] = {}
        section_stats_all[section] = section_stats
        root = _build_section_tree(
            section,
            leaves,
            entity_index=entity_index,
            edges=edges,
            leaf_metadata=leaf_metadata,
            nodes_out=nodes,
            section_stats=section_stats,
        )
        section_roots[section] = root["node_id"]

        # Decorate every node under this root before moving to the next section.
        _decorate_tree_bottom_up(
            root,
            nodes,
            edges=edges,
            leaf_metadata=leaf_metadata,
        )

    # Cross-listing. Step 5 already chose a primary_section per leaf; we
    # respect that so a leaf primary-homes inside its own section's tree
    # rather than wherever the corresponding entity_group is biggest.
    leaf_primary_section = {
        lid: rec.get("primary_section")
        for lid, rec in assignments.items()
        if rec.get("primary_section")
    }
    leaf_placements = _compute_leaf_placements(nodes, leaf_primary_section)

    # primary_leaf_ids on each node = leaves whose primary_node_id == this node.
    # Reset first because _new_node pre-allocates the field as an empty list.
    for node in nodes.values():
        node["primary_leaf_ids"] = []
    for lid, placement in leaf_placements.items():
        target = nodes.get(placement["primary_node_id"])
        if target is not None:
            target["primary_leaf_ids"].append(lid)
    for node in nodes.values():
        node["primary_leaf_ids"] = sorted(node["primary_leaf_ids"])

    # Validate.
    warnings = _validate_tree(nodes, section_roots, leaf_metadata, section_to_leaves)
    if warnings:
        console.print(f"[yellow]validation warnings ({len(warnings)}):[/]")
        for w in warnings[:15]:
            console.print(f"  [yellow]- {w}[/]")
        if len(warnings) > 15:
            console.print(f"  [yellow]... and {len(warnings) - 15} more[/]")

    # Build outputs.
    section_trees = {
        section: _build_section_tree_view(nodes[root_id], nodes)
        for section, root_id in section_roots.items()
    }

    TREE_DIR.mkdir(parents=True, exist_ok=True)

    nodes_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tree_taxonomy_version": TREE_TAXONOMY_VERSION,
        "params": {
            "min_entity_group_size": MIN_ENTITY_GROUP_SIZE,
            "min_graph_cluster_size": MIN_GRAPH_CLUSTER_SIZE,
            "recursion_leaf_threshold": RECURSION_LEAF_THRESHOLD,
            "max_node_depth": MAX_NODE_DEPTH,
            "graph_cluster_distance_threshold": GRAPH_CLUSTER_DISTANCE_THRESHOLD,
        },
        "node_count": len(nodes),
        "nodes": dict(sorted(nodes.items())),
    }
    with NODES_FILE.open("w", encoding="utf-8") as fh:
        json.dump(nodes_payload, fh, indent=2)

    section_trees_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tree_taxonomy_version": TREE_TAXONOMY_VERSION,
        "section_count": len(section_trees),
        "section_trees": dict(sorted(section_trees.items())),
        "leaf_placements": dict(sorted(leaf_placements.items())),
    }
    with SECTION_TREES_FILE.open("w", encoding="utf-8") as fh:
        json.dump(section_trees_payload, fh, indent=2)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tree_taxonomy_version": TREE_TAXONOMY_VERSION,
        "totals": {
            "section_count": len(section_roots),
            "node_count": len(nodes),
            "node_count_by_kind": dict(Counter(n["kind"] for n in nodes.values())),
            "leaves_with_placement": len(leaf_placements),
            "embeddable_leaves_in_index": len(embeddings_index),
            "warnings": warnings,
        },
        "by_section": section_stats_all,
        "params": nodes_payload["params"],
    }
    with STEP6_SUMMARY_FILE.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    # Console summary.
    console.print(
        f"[green]wrote[/] [bold]{NODES_FILE.relative_to(ROOT)}[/] "
        f"({len(nodes)} nodes)"
    )
    console.print(
        f"[green]wrote[/] [bold]{SECTION_TREES_FILE.relative_to(ROOT)}[/] "
        f"({len(section_trees)} sections, {len(leaf_placements)} leaf placements)"
    )
    console.print(
        f"[green]wrote[/] [bold]{STEP6_SUMMARY_FILE.relative_to(ROOT)}[/]"
    )

    if args.verbose:
        table = Table(title="Step 6 — per-section breakdown")
        table.add_column("section")
        table.add_column("leaves", justify="right")
        table.add_column("entity_groups", justify="right")
        table.add_column("graph_clusters", justify="right")
        table.add_column("orphan", justify="right")
        table.add_column("recursive_splits", justify="right")
        for section in sections_in_canonical_order:
            stats = section_stats_all.get(section, {})
            table.add_row(
                section,
                str(stats.get("section_leaf_count", 0)),
                str(stats.get("entity_groups", 0)),
                str(stats.get("graph_clusters", 0)),
                str(stats.get("orphan_bucket_size", 0)),
                str(stats.get("recursive_splits", 0)),
            )
        console.print(table)


if __name__ == "__main__":
    main()
