"""Step 1 — Validate every leaf JSON in input/leaves/.

For each file we run, in order:

  1. JSON parse
  2. Pydantic structural validation (shape + required fields)
  3. Hard business-rule checks (must pass to enter the pipeline)
  4. Soft sanity checks (warnings only)

Each leaf ends up with one of four statuses:

  ok                  - passes everything, ready for embedding
  ok_with_warnings    - passes hard rules, has soft issues worth surfacing
  skipped_not_ready   - promotion.ready_for_clustering is False (e.g. file
                        not found in Drive); excluded but not a "failure"
  failed              - cannot be safely processed downstream

Output:
  - output/_validation.json (structured report consumed by later steps)
  - Console summary table

Run:
  python scripts/validate_leaves.py
  python scripts/validate_leaves.py --verbose   # print per-leaf issues
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from pydantic import ValidationError
from rich.table import Table

from _common import (
    KNOWN_BUSINESS_AREAS,
    KNOWN_SENSITIVITIES,
    KNOWN_STATUSES,
    LEAVES_DIR,
    OUTPUT_DIR,
    ROOT,
    VALIDATION_REPORT,
    console,
    ensure_repo_on_path,
)

ensure_repo_on_path()
from scripts._schema import Leaf  # noqa: E402  (import after path injection)


MIN_EMBEDDING_TEXT_CHARS = 30


def _validate_business_rules(leaf: Leaf, file_path: Path) -> List[str]:
    """Hard checks. Any failure -> leaf is excluded from the pipeline."""
    errors: List[str] = []

    expected_leaf_id = f"file-{leaf.source.drive_file_id}"
    if leaf.leaf_id != expected_leaf_id:
        errors.append(
            f"leaf_id '{leaf.leaf_id}' does not match "
            f"'file-<source.drive_file_id>' (expected '{expected_leaf_id}')"
        )

    expected_filename = f"{leaf.leaf_id}.json"
    if file_path.name != expected_filename:
        errors.append(
            f"filename '{file_path.name}' does not match leaf_id "
            f"'{leaf.leaf_id}' (expected '{expected_filename}')"
        )

    if not leaf.source.drive_file_id.strip():
        errors.append("source.drive_file_id is empty")

    if not leaf.source.name.strip():
        errors.append("source.name is empty")

    if not leaf.embedding.title.strip():
        errors.append("embedding.title is empty")

    text = leaf.embedding.text.strip()
    if not text:
        errors.append("embedding.text is empty")
    elif len(text) < MIN_EMBEDDING_TEXT_CHARS:
        errors.append(
            f"embedding.text has only {len(text)} chars "
            f"(min {MIN_EMBEDDING_TEXT_CHARS}); too thin to embed reliably"
        )

    return errors


def _validate_soft_rules(leaf: Leaf) -> List[str]:
    """Soft checks. These are informational; they don't block downstream steps."""
    warnings: List[str] = []

    for area in leaf.classification.business_area:
        if area.lower() not in KNOWN_BUSINESS_AREAS:
            warnings.append(f"unknown business_area '{area}'")
        elif area != area.lower():
            warnings.append(
                f"business_area '{area}' is not lowercase; will normalize downstream"
            )

    if leaf.classification.status.lower() not in KNOWN_STATUSES:
        warnings.append(f"unknown status '{leaf.classification.status}'")

    if leaf.classification.sensitivity.lower() not in KNOWN_SENSITIVITIES:
        warnings.append(f"unknown sensitivity '{leaf.classification.sensitivity}'")

    if not leaf.source.url:
        warnings.append("source.url is empty")

    if leaf.source.last_modified is None:
        warnings.append("source.last_modified is null")

    if leaf.source.mime_type in {"", "unknown"}:
        warnings.append(f"source.mime_type is '{leaf.source.mime_type}'")

    if not leaf.embedding.keywords:
        warnings.append("embedding.keywords is empty")

    if not leaf.facts:
        warnings.append("no facts extracted")

    has_any_entity = any(
        getattr(leaf.entities, field) for field in type(leaf.entities).model_fields
    )
    if not has_any_entity:
        warnings.append("no entities extracted across any category")

    if leaf.classification.status.lower() in {"deprecated", "stale"}:
        warnings.append(
            f"classification.status is '{leaf.classification.status}'; "
            "may be filtered or down-weighted later"
        )

    return warnings


def _classify_leaf(
    file_path: Path,
) -> Tuple[str, List[str], List[str], Dict[str, Any]]:
    """Returns (status, errors, warnings, summary_meta)."""
    try:
        raw = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return "failed", [f"invalid JSON: {e}"], [], {}
    except OSError as e:
        return "failed", [f"could not read file: {e}"], [], {}

    try:
        leaf = Leaf.model_validate(raw)
    except ValidationError as e:
        # Compact pydantic errors into a flat list of messages.
        errors = [
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
            for err in e.errors()
        ]
        return "failed", errors, [], {}

    errors = _validate_business_rules(leaf, file_path)
    warnings = _validate_soft_rules(leaf)

    summary_meta = {
        "leaf_id": leaf.leaf_id,
        "name": leaf.source.name,
        "document_type": leaf.classification.document_type,
        "business_area": leaf.classification.business_area,
        "doc_status": leaf.classification.status,
        "sensitivity": leaf.classification.sensitivity,
        "ready_for_clustering": leaf.promotion.ready_for_clustering,
    }

    if errors:
        return "failed", errors, warnings, summary_meta

    if not leaf.promotion.ready_for_clustering:
        return "skipped_not_ready", [], warnings, summary_meta

    return ("ok_with_warnings" if warnings else "ok"), [], warnings, summary_meta


def _print_summary(report: Dict[str, Any], verbose: bool) -> None:
    table = Table(title="Leaf validation summary", show_lines=False)
    table.add_column("Status", style="bold")
    table.add_column("Count", justify="right")

    counts = report["counts"]
    style_for = {
        "ok": "green",
        "ok_with_warnings": "yellow",
        "skipped_not_ready": "cyan",
        "failed": "red",
    }
    for status in ("ok", "ok_with_warnings", "skipped_not_ready", "failed"):
        table.add_row(
            f"[{style_for[status]}]{status}[/]",
            str(counts.get(status, 0)),
        )
    table.add_row("[bold]total[/]", str(report["total"]))
    console.print(table)

    failed = [
        (lid, info)
        for lid, info in report["leaves"].items()
        if info["status"] == "failed"
    ]
    if failed:
        console.print("\n[bold red]Failed leaves:[/]")
        for leaf_id, info in failed:
            console.print(f"  [red]✗[/] {leaf_id}  ({info.get('name', '?')})")
            for err in info["errors"]:
                console.print(f"      [red]- {err}[/]")

    if verbose:
        warned = [
            (lid, info)
            for lid, info in report["leaves"].items()
            if info["warnings"]
        ]
        if warned:
            console.print("\n[bold yellow]Leaves with warnings:[/]")
            for leaf_id, info in warned:
                console.print(
                    f"  [yellow]![/] {leaf_id}  ({info.get('name', '?')})"
                )
                for w in info["warnings"]:
                    console.print(f"      [yellow]- {w}[/]")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print every per-leaf warning to the console.",
    )
    args = parser.parse_args(argv)

    if not LEAVES_DIR.exists():
        console.print(f"[red]Leaves directory not found: {LEAVES_DIR}[/]")
        return 2

    leaf_files = sorted(LEAVES_DIR.glob("file-*.json"))
    if not leaf_files:
        console.print(f"[red]No leaf JSONs found in {LEAVES_DIR}[/]")
        return 2

    console.print(
        f"Validating [bold]{len(leaf_files)}[/] leaves from [dim]{LEAVES_DIR}[/]"
    )

    report: Dict[str, Any] = {
        "validated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "leaves_dir": str(LEAVES_DIR.relative_to(ROOT)),
        "total": len(leaf_files),
        "counts": {
            "ok": 0,
            "ok_with_warnings": 0,
            "skipped_not_ready": 0,
            "failed": 0,
        },
        "leaves": {},
    }

    for file_path in leaf_files:
        status, errors, warnings, meta = _classify_leaf(file_path)
        report["counts"][status] += 1
        leaf_id = meta.get("leaf_id") or file_path.stem
        report["leaves"][leaf_id] = {
            "file": file_path.name,
            "status": status,
            "errors": errors,
            "warnings": warnings,
            **meta,
        }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VALIDATION_REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    _print_summary(report, verbose=args.verbose)
    console.print(
        f"\nReport written to [bold]{VALIDATION_REPORT.relative_to(ROOT)}[/]"
    )

    return 0 if report["counts"]["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
