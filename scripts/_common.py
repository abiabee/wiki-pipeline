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
