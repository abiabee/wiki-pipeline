"""Shared paths, constants, and helpers used by every pipeline step.

Keeping a single source of truth for filesystem layout means each script can
be run independently as `python scripts/<step>.py` without import friction.
"""

from __future__ import annotations

import sys
from pathlib import Path

from rich.console import Console

ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "input"
LEAVES_DIR = INPUT_DIR / "leaves"

OUTPUT_DIR = ROOT / "output"
EMBEDDINGS_DIR = OUTPUT_DIR / "embeddings"

VALIDATION_REPORT = OUTPUT_DIR / "_validation.json"
EMBEDDINGS_VECTORS = EMBEDDINGS_DIR / "vectors.npy"
EMBEDDINGS_INDEX = EMBEDDINGS_DIR / "index.json"

GRAPH_DIR = OUTPUT_DIR / "graph"
EDGES_FILE = GRAPH_DIR / "edges.json"
NEIGHBORS_FILE = GRAPH_DIR / "neighbors.json"
COMBINED_EDGES_FILE = GRAPH_DIR / "combined_edges.json"
RELATED_NEIGHBORS_FILE = GRAPH_DIR / "related_neighbors.json"
ENTITY_INDEX_FILE = GRAPH_DIR / "entity_index.json"
STEP4_SUMMARY_FILE = GRAPH_DIR / "step4_summary.json"
GRAPH_HTML = OUTPUT_DIR / "graph.html"
SECTION_TREE_HTML = OUTPUT_DIR / "section_tree.html"

# Step 5 outputs (section assignment) — these are classification artifacts,
# not graph artifacts, so they live at the top of output/.
SECTIONS_ASSIGNMENT_FILE = OUTPUT_DIR / "sections_assignment.json"
SECTIONS_INDEX_FILE = OUTPUT_DIR / "sections_index.json"
STEP5_SUMMARY_FILE = OUTPUT_DIR / "step5_summary.json"

# Step 6 outputs (cluster discovery + node tree) — `nodes.json` is the flat
# dictionary of every node's rendering contract, `section_trees.json` is the
# per-section traversal index that step 7 walks to build pages.
TREE_DIR = OUTPUT_DIR / "tree"
NODES_FILE = TREE_DIR / "nodes.json"
SECTION_TREES_FILE = TREE_DIR / "section_trees.json"
STEP6_SUMMARY_FILE = TREE_DIR / "step6_summary.json"


# Base rule-edge weights, before IDF normalization. These are the *ceiling*
# for each rule type; the actual emitted weight is `base * idf(group_size)`.
# `audience` and `document_type` were intentionally excluded for v1 (too
# noisy on this corpus).
RULE_BASE_WEIGHTS: dict[str, float] = {
    "customers": 1.0,
    "partners": 0.9,
    "erps": 0.8,
    "competitors": 0.7,
    "products": 0.7,
    "policies": 0.6,
    "people": 0.5,
    "features": 0.4,
    "business_area": 0.3,
}


console = Console()


def ensure_repo_on_path() -> None:
    """Allow `from scripts._schema import Leaf` when run as a plain script."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


KNOWN_BUSINESS_AREAS = {
    "product",
    "integrations",
    "engineering",
    "sales",
    "marketing",
    "channel",
    "channels",
    "partnerships",
    "operations",
    "finance",
    "hr",
    "legal",
    "compliance",
    "risk",
    "security",
    "payments",
    "reconciliation",
    "customer-success",
    "customer_success",
    "unknown",
}

KNOWN_STATUSES = {"current", "stale", "deprecated", "draft", "archived", "unknown"}

KNOWN_SENSITIVITIES = {
    "public",
    "normal",
    "internal",
    "confidential",
    "pii-risk",
    "restricted",
    "unknown",
}


# ---------------------------------------------------------------------------
# Step 5 — section assignment vocabularies
#
# These constants drive `assign_sections.py`. They're the curated, human-facing
# mapping from leaf metadata to wiki sections. Edit here to retune the wiki.
# ---------------------------------------------------------------------------


# Canonical ordering of all sections that the wiki may present. Used as the
# enumeration order in `sections_index.json` and as the source of truth for
# what counts as a "valid" section. Adding a new section requires updating
# this list, SECTION_LABELS, and (usually) SECTION_ALIAS / TIE_BREAK_PRIORITY.
SECTION_ORDER: list[str] = [
    "topics/product",
    "topics/integrations",
    "topics/engineering",
    "topics/sales",
    "topics/marketing",
    "topics/channel",
    "topics/operations",
    "topics/hr",
    "topics/legal",
    "topics/compliance",
    "topics/payments",
    "topics/reconciliation",
    "topics/customer-success",
    "topics/company",
    "entities/customers",
    "entities/competitors",
    "entities/erps",
    "entities/products",
    "entities/features",
    "entities/partners",
    "entities/people",
    "decisions",
    "meta",
]

SECTION_LABELS: dict[str, str] = {
    "topics/product": "Product",
    "topics/integrations": "Integrations",
    "topics/engineering": "Engineering",
    "topics/sales": "Sales",
    "topics/marketing": "Marketing",
    "topics/channel": "Channel / Partners",
    "topics/operations": "Finance / Operations",
    "topics/hr": "HR / People",
    "topics/legal": "Legal",
    "topics/compliance": "Risk & Compliance",
    "topics/payments": "Payments",
    "topics/reconciliation": "Reconciliation",
    "topics/customer-success": "Customer Success",
    "topics/company": "Company / Leadership",
    "entities/customers": "Customers",
    "entities/competitors": "Competitors",
    "entities/erps": "ERPs",
    "entities/products": "Products",
    "entities/features": "Features",
    "entities/partners": "Partners",
    "entities/people": "People",
    "decisions": "Decisions",
    "meta": "Meta",
}

# Maps slugified `business_area` and `audience` values onto sections.
# Lookup is case-insensitive; values must already be slugified (lowercase,
# hyphens). Keys here are the canonical slug spelling.
SECTION_ALIAS: dict[str, str] = {
    "sales": "topics/sales",
    "marketing": "topics/marketing",
    "product": "topics/product",
    "engineering": "topics/engineering",
    "integrations": "topics/integrations",
    "channel": "topics/channel",
    "channels": "topics/channel",
    "partnerships": "topics/channel",
    "finance": "topics/operations",
    "operations": "topics/operations",
    "security": "topics/compliance",
    "compliance": "topics/compliance",
    "risk": "topics/compliance",
    "legal": "topics/legal",
    "hr": "topics/hr",
    "company-culture": "topics/hr",
    "customer-success": "topics/customer-success",
    "board": "topics/company",
    "ceo": "topics/company",
    "payments": "topics/payments",
    "reconciliation": "topics/reconciliation",
}

# Each entity type (key = entity_index bucket name) triggers a section
# whenever a leaf has any value in that bucket. `policies` is folded into
# topics/compliance per the curated map; there is no entities/policies
# section.
ENTITY_TYPE_SECTION: dict[str, str] = {
    "customers": "entities/customers",
    "competitors": "entities/competitors",
    "erps": "entities/erps",
    "products": "entities/products",
    "features": "entities/features",
    "partners": "entities/partners",
    "people": "entities/people",
    "policies": "topics/compliance",
}

# Score weights per signal source. business_area dominates because it's
# the curator's strongest declarative statement about what a doc is "about".
SECTION_SIGNAL_WEIGHTS: dict[str, int] = {
    "business_area": 5,
    "audience": 3,
    "document_type": 2,
    # `entity_type` is a default; per-type overrides live in ENTITY_TYPE_WEIGHTS.
    "entity_type": 2,
}

# Per-entity-type weight; falls back to SECTION_SIGNAL_WEIGHTS["entity_type"]
# when not listed. `features` and `people` are noisier so they get less.
ENTITY_TYPE_WEIGHTS: dict[str, int] = {
    "customers": 2,
    "competitors": 2,
    "erps": 2,
    "products": 2,
    "partners": 2,
    "policies": 2,
    "features": 1,
    "people": 1,
}

# Match is case-insensitive substring on the slugified document_type.
# Be conservative — skip rather than guess. Multiple matches against the
# same target section count once.
DOC_TYPE_SECTION_HINTS: dict[str, str] = {
    "contract": "topics/legal",
    "addendum": "topics/legal",
    "agreement": "topics/legal",
    "icp": "topics/sales",
    "battlecard": "topics/sales",
    "deck": "topics/sales",
    "sales-deck": "topics/sales",
    "sales-collateral": "topics/sales",
    "playbook": "topics/sales",
    "scoping-document": "topics/compliance",
    "policy": "topics/compliance",
    "decision": "decisions",
    "adr": "decisions",
    "rfc": "decisions",
}

# When sections tie on score, the one earlier in this list wins. Encodes the
# user's "specific department > generic sales > entity reference" preference.
TIE_BREAK_PRIORITY: list[str] = [
    "topics/compliance",
    "topics/legal",
    "topics/hr",
    "topics/customer-success",
    "topics/engineering",
    "topics/operations",
    "topics/payments",
    "topics/reconciliation",
    "topics/integrations",
    "topics/product",
    "topics/marketing",
    "topics/company",
    "topics/sales",
    "topics/channel",
    "entities/customers",
    "entities/erps",
    "entities/partners",
    "entities/competitors",
    "entities/products",
    "entities/features",
    "entities/people",
    "decisions",
    "meta",
]

# A leaf's `sections` array only includes sections whose final score meets
# this floor. Keeps the cross-listing meaningful.
SECTIONS_MIN_SCORE: int = 2

# Stamped into the assignment file so callers can detect taxonomy drift.
SECTIONS_TAXONOMY_VERSION: str = "v1"


# ---------------------------------------------------------------------------
# Step 6 — cluster discovery (entity decomposition + graph clustering)
#
# These thresholds shape the node tree. They are intentionally lenient at
# the current corpus size (85 leaves) and should be raised once the full set
# is loaded.
# ---------------------------------------------------------------------------

# An entity (e.g. "NetSuite") only earns its own entity_group node if at
# least this many leaves in the section reference it. User decision #9.
MIN_ENTITY_GROUP_SIZE: int = 3

# A graph cluster (run on leaves left over after entity decomposition) only
# becomes a node if it has at least this many members. Smaller clusters get
# absorbed into the section's "Other" orphan bucket.
MIN_GRAPH_CLUSTER_SIZE: int = 3

# Once a node's recursive leaf count exceeds this, we try to split it one
# level deeper. User decision #11.
RECURSION_LEAF_THRESHOLD: int = 10

# Hard cap on tree depth below the section root. depth=0 is the section
# root, so MAX_NODE_DEPTH=3 means at most three levels of nodes under it.
MAX_NODE_DEPTH: int = 3

# Distance threshold for sklearn's AgglomerativeClustering. Distance is
# `1 - combined_weight` between two leaves; missing edges are treated as
# distance=1.0. A threshold of 0.5 effectively says "cluster while pairs
# share combined_weight ≥ 0.5". Raise to make clusters tighter.
GRAPH_CLUSTER_DISTANCE_THRESHOLD: float = 0.5

# Cohesion bands feed the node's `quality.confidence`. Cohesion is the mean
# combined_weight across all intra-node edges; nodes with no internal edges
# (singleton or fully-disconnected) get confidence="low".
COHESION_HIGH_THRESHOLD: float = 0.65
COHESION_LOW_THRESHOLD: float = 0.45

# Stamped onto every node so step 7 can detect tree-shape drift.
TREE_TAXONOMY_VERSION: str = "v1"
