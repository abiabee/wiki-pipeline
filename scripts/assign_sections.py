"""Step 5 — Assign each leaf to wiki sections (canonical, multi-label).

This is the *human-facing* placement decision: where does each file live in
the wiki sidebar? The result drives the navigation, lets step 7 build entity
landing pages, and is the source of truth that step 6 (clustering) refines
within rather than overrides.

Heavy reuse of step 4:
  - `entity_index.json` provides the leaf-to-bucket mapping for both
    `business_area` and every entity type. We invert it once and never
    re-parse the raw entity arrays in the leaves.
  - Raw leaves are touched only for `source.name`, `classification.audience`,
    and `classification.document_type` — the three fields that step 4
    intentionally didn't index.

Scoring:
  business_area  → +5 per matching value (via SECTION_ALIAS)
  audience       → +3 per matching value (via SECTION_ALIAS)
  document_type  → +2 per matching hint (DOC_TYPE_SECTION_HINTS, deduped per section)
  entity types   → +2 per non-empty type (or +1 for features/people)

Decision rules:
  primary_section = highest-scoring `topics/*`, tie-broken by TIE_BREAK_PRIORITY.
  If no `topics/*` got any score, fall back to highest `entities/*`, then `meta`.
  `sections` array = every section with score >= SECTIONS_MIN_SCORE, ordered.

Outputs:
  output/sections_assignment.json   leaf_id → assignment record
  output/sections_index.json        section_id → {label, leaf_ids, primary_leaf_ids}
  output/step5_summary.json         distributions, unmapped values, low-confidence list

Run:
  python scripts/assign_sections.py
  python scripts/assign_sections.py --include-skipped-with-entities
  python scripts/assign_sections.py --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from rich.table import Table
from slugify import slugify

from _common import (
    DOC_TYPE_SECTION_HINTS,
    ENTITY_INDEX_FILE,
    ENTITY_TYPE_SECTION,
    ENTITY_TYPE_WEIGHTS,
    LEAVES_DIR,
    OUTPUT_DIR,
    ROOT,
    SECTIONS_ASSIGNMENT_FILE,
    SECTIONS_INDEX_FILE,
    SECTIONS_MIN_SCORE,
    SECTIONS_TAXONOMY_VERSION,
    SECTION_ALIAS,
    SECTION_LABELS,
    SECTION_ORDER,
    SECTION_SIGNAL_WEIGHTS,
    STEP5_SUMMARY_FILE,
    TIE_BREAK_PRIORITY,
    VALIDATION_REPORT,
    console,
)


EMBEDDABLE_STATUSES = {"ok", "ok_with_warnings"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slug(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    return slugify(s) or None


def _section_priority(section_id: str) -> int:
    """Earlier in TIE_BREAK_PRIORITY = higher priority. Unknown sections sort last."""
    try:
        return TIE_BREAK_PRIORITY.index(section_id)
    except ValueError:
        return len(TIE_BREAK_PRIORITY) + 1


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


def _load_entity_index() -> Dict[str, Any]:
    if not ENTITY_INDEX_FILE.exists():
        console.print(
            "[red]entity_index.json not found. Run build_rule_edges.py (step 4) first.[/]"
        )
        sys.exit(2)
    return json.loads(ENTITY_INDEX_FILE.read_text(encoding="utf-8"))


def _select_leaves(
    report: Dict[str, Any],
    include_skipped_with_entities: bool,
) -> List[str]:
    """Same leaf-selection policy as step 4."""
    selected: List[str] = []
    for leaf_id, info in report.get("leaves", {}).items():
        status = info.get("status")
        if status in EMBEDDABLE_STATUSES:
            selected.append(leaf_id)
        elif include_skipped_with_entities and status == "skipped_not_ready":
            selected.append(leaf_id)
    return sorted(selected)


def _build_leaf_buckets(
    entity_index: Dict[str, Any],
) -> Dict[str, Dict[str, List[Tuple[str, str]]]]:
    """Invert entity_index: leaf_id -> {entity_type: [(slug, label)]}."""
    leaf_buckets: Dict[str, Dict[str, List[Tuple[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for entity_type, slugs in entity_index.get("index", {}).items():
        for slug, info in slugs.items():
            label = info.get("label") or slug
            for leaf_id in info.get("leaf_ids", []):
                leaf_buckets[leaf_id][entity_type].append((slug, label))
    return leaf_buckets


def _load_leaf_extras(leaf_id: str) -> Dict[str, Any]:
    """Read the few raw-leaf fields we still need. Defaults are tolerant."""
    path = LEAVES_DIR / f"{leaf_id}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"title": leaf_id, "audience": [], "document_type": ""}

    src = data.get("source") or {}
    emb = data.get("embedding") or {}
    cls = data.get("classification") or {}
    title = (
        (emb.get("title") or "").strip()
        or (src.get("name") or "").strip()
        or leaf_id
    )
    return {
        "title": title,
        "audience": list(cls.get("audience") or []),
        "document_type": str(cls.get("document_type") or "").strip(),
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _score_leaf(
    leaf_id: str,
    buckets: Dict[str, List[Tuple[str, str]]],
    extras: Dict[str, Any],
    unmapped: Dict[str, Counter],
) -> Tuple[Dict[str, int], List[str]]:
    """Compute score_by_section + reasons for one leaf.

    `unmapped` accumulates slugs we don't recognize, for later surfacing in
    step5_summary.json. Mutated in place.
    """
    score: Dict[str, int] = defaultdict(int)
    reasons: List[str] = []

    # 1) business_area — strongest signal
    for slug, label in buckets.get("business_area", []):
        section = SECTION_ALIAS.get(slug.lower())
        if section is None:
            unmapped["business_area"][slug] += 1
            continue
        score[section] += SECTION_SIGNAL_WEIGHTS["business_area"]
        reasons.append(f"business_area {label} → {section}")

    # 2) entity types — each non-empty bucket triggers its target section once
    for entity_type, items in buckets.items():
        if entity_type == "business_area" or not items:
            continue
        section = ENTITY_TYPE_SECTION.get(entity_type)
        if section is None:
            continue
        weight = ENTITY_TYPE_WEIGHTS.get(
            entity_type, SECTION_SIGNAL_WEIGHTS["entity_type"]
        )
        score[section] += weight
        # Use the first (alphabetical) label as a representative reason; all
        # values in this bucket point at the same section anyway.
        sample_label = sorted(items, key=lambda t: t[1].lower())[0][1]
        reasons.append(f"entity {sample_label} ({entity_type}) → {section}")

    # 3) audience — uses the same alias map as business_area, lower weight
    for raw in extras.get("audience") or []:
        slug = _slug(raw)
        if not slug:
            continue
        section = SECTION_ALIAS.get(slug)
        if section is None:
            unmapped["audience"][slug] += 1
            continue
        score[section] += SECTION_SIGNAL_WEIGHTS["audience"]
        reasons.append(f"audience {raw} → {section}")

    # 4) document_type — substring match. Each target section counts at most
    # once per leaf so 'contract_addendum' (matches both 'contract' and
    # 'addendum' → topics/legal) doesn't double-score.
    doc_type_raw = extras.get("document_type", "")
    doc_type_slug = (_slug(doc_type_raw) or "").lower()
    if doc_type_slug:
        seen_sections: Set[str] = set()
        for hint, target in DOC_TYPE_SECTION_HINTS.items():
            if hint in doc_type_slug and target not in seen_sections:
                seen_sections.add(target)
                score[target] += SECTION_SIGNAL_WEIGHTS["document_type"]
                reasons.append(f"document_type {doc_type_raw} → {target}")

    return dict(score), reasons


def _pick_primary(score: Dict[str, int]) -> str:
    """Highest-scoring topics/* wins; tie-broken by TIE_BREAK_PRIORITY.

    Falls back to highest entities/*, then to any other section, then `meta`.
    """
    if not score:
        return "meta"
    sorted_items = sorted(
        score.items(),
        key=lambda kv: (-kv[1], _section_priority(kv[0])),
    )
    for section, value in sorted_items:
        if value > 0 and section.startswith("topics/"):
            return section
    for section, value in sorted_items:
        if value > 0 and section.startswith("entities/"):
            return section
    for section, value in sorted_items:
        if value > 0:
            return section
    return "meta"


def _ordered_sections(
    score: Dict[str, int], primary: str, min_score: int
) -> List[str]:
    """Sections >= min_score, ordered by (-score, priority); primary first."""
    eligible = [s for s, v in score.items() if v >= min_score]
    eligible.sort(key=lambda s: (-score[s], _section_priority(s)))
    if primary not in eligible:
        eligible.insert(0, primary)
    elif eligible[0] != primary:
        # Move primary to the front while preserving rest of the order.
        eligible.remove(primary)
        eligible.insert(0, primary)
    return eligible


def _confidence(score: Dict[str, int], primary: str) -> str:
    """Score-and-margin policy.

    high   : primary >= 5 AND margin over runner-up >= 3
    medium : primary >= 3 OR margin >= 2
    low    : non-zero score, neither high nor medium
    low+meta: zero score → primary will already be 'meta'
    """
    primary_score = score.get(primary, 0)
    if primary_score == 0:
        return "low"
    runner_up = max(
        (v for s, v in score.items() if s != primary), default=0
    )
    margin = primary_score - runner_up

    if primary_score >= 5 and margin >= 3:
        return "high"
    if primary_score >= 3 or margin >= 2:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def _assign_leaf(
    leaf_id: str,
    buckets: Dict[str, List[Tuple[str, str]]],
    extras: Dict[str, Any],
    unmapped: Dict[str, Counter],
) -> Dict[str, Any]:
    score, reasons = _score_leaf(leaf_id, buckets, extras, unmapped)
    primary = _pick_primary(score)
    if score.get(primary, 0) == 0:
        # No signal at all → meta with the primary explicitly recorded.
        primary = "meta"
        sections = ["meta"]
        confidence = "low"
        reasons.append("no signals matched → meta (catch-all)")
    else:
        sections = _ordered_sections(score, primary, SECTIONS_MIN_SCORE)
        confidence = _confidence(score, primary)

    return {
        "title": extras["title"],
        "primary_section": primary,
        "sections": sections,
        "confidence": confidence,
        "score_by_section": dict(
            sorted(score.items(), key=lambda kv: (-kv[1], _section_priority(kv[0])))
        ),
        "reasons": reasons,
    }


def _build_sections_index(
    assignments: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """section_id → {label, leaf_count, primary_leaf_count, leaf_ids, primary_leaf_ids}."""
    leaves_by_section: Dict[str, Set[str]] = defaultdict(set)
    primary_by_section: Dict[str, Set[str]] = defaultdict(set)
    for leaf_id, rec in assignments.items():
        primary_by_section[rec["primary_section"]].add(leaf_id)
        for s in rec["sections"]:
            leaves_by_section[s].add(leaf_id)

    out: Dict[str, Any] = {}
    # Walk SECTION_ORDER first so the index is canonical, then append any
    # unexpected sections (shouldn't happen, but defensive).
    seen: Set[str] = set()
    for section in SECTION_ORDER:
        seen.add(section)
        leaves = sorted(leaves_by_section.get(section, set()))
        primaries = sorted(primary_by_section.get(section, set()))
        out[section] = {
            "label": SECTION_LABELS.get(section, section),
            "leaf_count": len(leaves),
            "primary_leaf_count": len(primaries),
            "leaf_ids": leaves,
            "primary_leaf_ids": primaries,
        }
    for section in sorted(set(leaves_by_section) - seen):
        leaves = sorted(leaves_by_section[section])
        primaries = sorted(primary_by_section.get(section, set()))
        out[section] = {
            "label": section,
            "leaf_count": len(leaves),
            "primary_leaf_count": len(primaries),
            "leaf_ids": leaves,
            "primary_leaf_ids": primaries,
        }
    return out


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _write_assignment(
    assignments: Dict[str, Dict[str, Any]],
    params: Dict[str, Any],
) -> None:
    doc = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "section_taxonomy_version": SECTIONS_TAXONOMY_VERSION,
        "params": params,
        "assignment_count": len(assignments),
        "assignments": assignments,
    }
    SECTIONS_ASSIGNMENT_FILE.write_text(
        json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _write_sections_index(index: Dict[str, Any]) -> None:
    doc = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "section_taxonomy_version": SECTIONS_TAXONOMY_VERSION,
        "section_count": len(index),
        "sections": index,
    }
    SECTIONS_INDEX_FILE.write_text(
        json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _write_summary(summary: Dict[str, Any]) -> None:
    STEP5_SUMMARY_FILE.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------


def _print_summary(
    assignments: Dict[str, Dict[str, Any]],
    sections_index: Dict[str, Any],
    unmapped: Dict[str, Counter],
    verbose: bool,
) -> None:
    confidence_dist = Counter(rec["confidence"] for rec in assignments.values())
    primary_dist = Counter(rec["primary_section"] for rec in assignments.values())

    head = Table(title="Step 5 — section assignment", show_lines=False)
    head.add_column("Field", style="bold")
    head.add_column("Value")
    head.add_row("Leaves assigned", str(len(assignments)))
    head.add_row("Sections in use", str(sum(1 for s in sections_index.values() if s["leaf_count"])))
    head.add_row(
        "Confidence",
        ", ".join(
            f"{lvl}={confidence_dist.get(lvl, 0)}"
            for lvl in ("high", "medium", "low")
        ),
    )
    head.add_row("Assignments file", str(SECTIONS_ASSIGNMENT_FILE.relative_to(ROOT)))
    head.add_row("Sections index", str(SECTIONS_INDEX_FILE.relative_to(ROOT)))
    head.add_row("Summary", str(STEP5_SUMMARY_FILE.relative_to(ROOT)))
    console.print(head)

    pt = Table(title="Primary section distribution", show_lines=False)
    pt.add_column("Section", style="bold")
    pt.add_column("Primary", justify="right")
    pt.add_column("Total members", justify="right")
    for section in SECTION_ORDER:
        info = sections_index.get(section)
        if not info or info["leaf_count"] == 0 and primary_dist.get(section, 0) == 0:
            continue
        pt.add_row(
            section,
            str(primary_dist.get(section, 0)),
            str(info["leaf_count"]),
        )
    console.print(pt)

    if any(unmapped.values()):
        ut = Table(
            title="Unmapped slugs (candidates for SECTION_ALIAS)",
            show_lines=False,
        )
        ut.add_column("Source")
        ut.add_column("Slug")
        ut.add_column("Hits", justify="right")
        for source, counter in unmapped.items():
            for slug, count in counter.most_common(10):
                ut.add_row(source, slug, str(count))
        console.print(ut)

    low_conf = [
        (lid, rec) for lid, rec in assignments.items() if rec["confidence"] == "low"
    ]
    if low_conf:
        lt = Table(
            title=f"Low-confidence assignments ({len(low_conf)} total, showing 8)",
            show_lines=False,
        )
        lt.add_column("Leaf")
        lt.add_column("Title")
        lt.add_column("Primary")
        lt.add_column("Score")
        for lid, rec in low_conf[:8]:
            lt.add_row(
                lid[:28],
                rec["title"][:50],
                rec["primary_section"],
                str(rec["score_by_section"].get(rec["primary_section"], 0)),
            )
        console.print(lt)

    if verbose:
        console.print("\n[bold]Per-leaf detail (verbose):[/]")
        for lid, rec in assignments.items():
            console.print(
                f"  [dim]{lid[:30]}[/] [bold]{rec['title'][:55]}[/]"
            )
            console.print(
                f"    primary={rec['primary_section']}  "
                f"sections={rec['sections']}  "
                f"confidence={rec['confidence']}"
            )
            for r in rec["reasons"]:
                console.print(f"      [yellow]·[/] {r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-skipped-with-entities",
        action="store_true",
        help=(
            "Also assign sections to `skipped_not_ready` leaves whose data "
            "made it into entity_index.json. Off by default for parity with "
            "the embeddable cohort."
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print every per-leaf assignment with reasons.",
    )
    args = parser.parse_args(argv)

    report = _load_validation_report()
    entity_index = _load_entity_index()

    candidate_ids = set(
        _select_leaves(
            report, include_skipped_with_entities=args.include_skipped_with_entities
        )
    )
    leaf_buckets = _build_leaf_buckets(entity_index)
    # If we're including skipped leaves, only the ones that actually made it
    # into entity_index will have anything to score on. The rest fall through
    # to a meta assignment (and that's probably noise we don't want to emit).
    leaves_to_assign = sorted(candidate_ids & set(leaf_buckets.keys()))
    if not leaves_to_assign:
        # Fall back: even a leaf with no buckets in the index can be assigned
        # to meta. That mostly matters when we *aren't* including skipped.
        leaves_to_assign = sorted(candidate_ids)

    console.print(
        f"Assigning sections for [bold]{len(leaves_to_assign)}[/] leaves "
        f"(candidates: {len(candidate_ids)}, in entity_index: {len(leaf_buckets)})"
    )

    unmapped: Dict[str, Counter] = {
        "business_area": Counter(),
        "audience": Counter(),
    }

    assignments: Dict[str, Dict[str, Any]] = {}
    for leaf_id in leaves_to_assign:
        buckets = leaf_buckets.get(leaf_id, {})
        extras = _load_leaf_extras(leaf_id)
        assignments[leaf_id] = _assign_leaf(leaf_id, buckets, extras, unmapped)

    sections_index = _build_sections_index(assignments)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    params = {
        "include_skipped_with_entities": args.include_skipped_with_entities,
        "min_section_score": SECTIONS_MIN_SCORE,
        "weights": SECTION_SIGNAL_WEIGHTS,
        "entity_type_weights": ENTITY_TYPE_WEIGHTS,
    }
    _write_assignment(assignments, params)
    _write_sections_index(sections_index)

    confidence_dist = Counter(rec["confidence"] for rec in assignments.values())
    primary_dist = Counter(rec["primary_section"] for rec in assignments.values())
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "section_taxonomy_version": SECTIONS_TAXONOMY_VERSION,
        "params": params,
        "leaves": {
            "assigned": len(assignments),
            "candidate_pool": len(candidate_ids),
        },
        "confidence_distribution": dict(confidence_dist),
        "primary_section_distribution": {
            section: primary_dist.get(section, 0)
            for section in SECTION_ORDER
            if primary_dist.get(section, 0) > 0
        },
        "section_membership_distribution": {
            section: info["leaf_count"]
            for section, info in sections_index.items()
            if info["leaf_count"] > 0
        },
        "unmapped_values": {
            source: [
                {"slug": slug, "hits": count}
                for slug, count in counter.most_common()
            ]
            for source, counter in unmapped.items()
            if counter
        },
        "low_confidence_leaves": [
            {
                "leaf_id": lid,
                "title": rec["title"],
                "primary_section": rec["primary_section"],
                "primary_score": rec["score_by_section"].get(
                    rec["primary_section"], 0
                ),
            }
            for lid, rec in assignments.items()
            if rec["confidence"] == "low"
        ],
    }
    _write_summary(summary)

    _print_summary(assignments, sections_index, unmapped, verbose=args.verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
