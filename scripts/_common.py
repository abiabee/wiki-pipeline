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
GRAPH_HTML = OUTPUT_DIR / "graph.html"


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
