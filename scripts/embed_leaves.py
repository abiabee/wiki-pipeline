"""Step 2 — Embed every clusterable leaf into a dense vector.

Inputs:
  - input/leaves/file-*.json
  - output/_validation.json   (produced by validate_leaves.py)

Strategy:
  - Only embed leaves whose validation status is `ok` or `ok_with_warnings`.
    `failed` and `skipped_not_ready` leaves are excluded by design.
  - The text we embed is built from the leaf's `embedding.{title,text,keywords}`
    so we lean on the curation that already happened upstream rather than the
    raw Drive content (which we may not have access to).
  - We hash the input text per leaf. If the hash already exists in the
    embeddings index AND the model is unchanged, we reuse the cached vector.
    This makes incremental runs cheap: adding one leaf re-embeds one leaf.

Outputs:
  - output/embeddings/vectors.npy   (N x D float32, L2-normalized)
  - output/embeddings/index.json    metadata + leaf_id -> row mapping

Run:
  python scripts/embed_leaves.py
  python scripts/embed_leaves.py --force      # ignore cache, re-embed everything
  python scripts/embed_leaves.py --model BAAI/bge-base-en-v1.5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from _common import (
    EMBEDDINGS_DIR,
    EMBEDDINGS_INDEX,
    EMBEDDINGS_VECTORS,
    LEAVES_DIR,
    ROOT,
    VALIDATION_REPORT,
    console,
)


DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

# Leaves with these validation statuses are considered embeddable.
EMBEDDABLE_STATUSES = {"ok", "ok_with_warnings"}


def _build_text(leaf: Dict[str, Any]) -> str:
    """Concatenate the curated embedding fields into a single string.

    The shape mirrors what the rest of the pipeline assumes lives in the
    leaf's `embedding` block: a human-written title, a paragraph of context,
    and a flat keyword list. Joining them gives the model a wider lexical
    surface than any single field alone.
    """
    emb = leaf.get("embedding", {}) or {}
    title = (emb.get("title") or "").strip()
    text = (emb.get("text") or "").strip()
    keywords = emb.get("keywords") or []
    keywords_line = ", ".join(k.strip() for k in keywords if k and k.strip())

    parts: List[str] = []
    if title:
        parts.append(title)
    if text:
        parts.append(text)
    if keywords_line:
        parts.append(f"[keywords] {keywords_line}")
    return "\n\n".join(parts)


def _hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _load_validation_report() -> Dict[str, Any]:
    if not VALIDATION_REPORT.exists():
        console.print(
            "[red]Validation report not found.[/]\n"
            f"Run [bold]python scripts/validate_leaves.py[/] first."
        )
        sys.exit(2)
    return json.loads(VALIDATION_REPORT.read_text(encoding="utf-8"))


def _load_existing_index(model_name: str) -> Dict[str, Any]:
    """Return the prior index if the model matches; otherwise an empty stub."""
    if not EMBEDDINGS_INDEX.exists():
        return {}
    try:
        idx = json.loads(EMBEDDINGS_INDEX.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if idx.get("model") != model_name:
        console.print(
            f"[yellow]Existing index uses model '{idx.get('model')}', "
            f"but '{model_name}' was requested. Cache will be invalidated.[/]"
        )
        return {}
    return idx


def _select_candidates(report: Dict[str, Any]) -> List[str]:
    """Pick leaves we should embed, in deterministic order."""
    leaves: List[str] = []
    for leaf_id, info in report.get("leaves", {}).items():
        if info.get("status") in EMBEDDABLE_STATUSES:
            leaves.append(leaf_id)
    return sorted(leaves)


def _load_leaf(leaf_id: str) -> Dict[str, Any]:
    path = LEAVES_DIR / f"{leaf_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _plan(
    candidates: List[str],
    existing_index: Dict[str, Any],
    force: bool,
) -> Tuple[List[Tuple[str, str, str]], Dict[str, Dict[str, Any]]]:
    """Decide which leaves need (re-)embedding and which can reuse the cache.

    Returns:
      to_embed: list of (leaf_id, input_text, hash) for leaves we need to encode.
      reuse:    leaf_id -> existing index entry, used to copy cached rows over.
    """
    prev_leaves: Dict[str, Dict[str, Any]] = (
        existing_index.get("leaves", {}) if existing_index else {}
    )

    to_embed: List[Tuple[str, str, str]] = []
    reuse: Dict[str, Dict[str, Any]] = {}

    for leaf_id in candidates:
        leaf = _load_leaf(leaf_id)
        text = _build_text(leaf)
        text_hash = _hash_text(text)

        prev = prev_leaves.get(leaf_id)
        cache_hit = (
            not force
            and prev is not None
            and prev.get("hash") == text_hash
            and "row" in prev
        )
        if cache_hit:
            reuse[leaf_id] = prev
        else:
            to_embed.append((leaf_id, text, text_hash))

    return to_embed, reuse


def _encode(
    model_name: str, texts: List[str]
) -> "Any":  # numpy.ndarray, kept loose to avoid import at top level
    """Lazy-import heavy ML deps so unrelated commands stay light."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        console.print(
            "[red]sentence-transformers is not installed.[/]\n"
            "Install with: [bold]pip install -r requirements.txt[/]"
        )
        sys.exit(2)

    console.print(f"Loading model [bold]{model_name}[/] (first run downloads weights)...")
    model = SentenceTransformer(model_name)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Encoding leaves", total=len(texts))
        # Encode in batches; show_progress_bar=False because we drive our own.
        vectors = model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        progress.update(task, completed=len(texts))

    return vectors


def _write_outputs(
    model_name: str,
    leaf_ids: List[str],
    vectors: "Any",
    text_hashes: Dict[str, str],
    char_counts: Dict[str, int],
) -> None:
    import numpy as np  # local import keeps top of file dependency-light

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_VECTORS, vectors.astype("float32"))

    index = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model_name,
        "dim": int(vectors.shape[1]),
        "count": int(vectors.shape[0]),
        "normalized": True,
        "vectors_file": str(EMBEDDINGS_VECTORS.relative_to(ROOT)),
        "leaves": {
            leaf_id: {
                "row": i,
                "hash": text_hashes[leaf_id],
                "chars": char_counts[leaf_id],
            }
            for i, leaf_id in enumerate(leaf_ids)
        },
    }
    EMBEDDINGS_INDEX.write_text(
        json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _print_summary(
    model_name: str,
    candidates: List[str],
    to_embed_count: int,
    reused_count: int,
    dim: int,
) -> None:
    table = Table(title="Embedding run summary", show_lines=False)
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_row("Model", model_name)
    table.add_row("Embedding dim", str(dim))
    table.add_row("Candidates", str(len(candidates)))
    table.add_row("[green]Newly embedded[/]", str(to_embed_count))
    table.add_row("[cyan]Reused from cache[/]", str(reused_count))
    table.add_row(
        "Vectors file",
        str(EMBEDDINGS_VECTORS.relative_to(ROOT)),
    )
    table.add_row(
        "Index file",
        str(EMBEDDINGS_INDEX.relative_to(ROOT)),
    )
    console.print(table)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"sentence-transformers model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore the existing cache and re-embed every candidate leaf.",
    )
    args = parser.parse_args(argv)

    report = _load_validation_report()
    candidates = _select_candidates(report)
    if not candidates:
        console.print("[red]No embeddable leaves found in the validation report.[/]")
        return 2

    console.print(
        f"Found [bold]{len(candidates)}[/] embeddable leaves "
        f"(status in {sorted(EMBEDDABLE_STATUSES)})"
    )

    existing_index = _load_existing_index(args.model)
    to_embed, reuse = _plan(candidates, existing_index, force=args.force)

    if not to_embed and reuse:
        console.print("[green]Cache hit for every candidate — no encoding needed.[/]")

    text_hashes: Dict[str, str] = {}
    char_counts: Dict[str, int] = {}
    new_vectors = None

    if to_embed:
        texts = [t for _, t, _ in to_embed]
        new_vectors = _encode(args.model, texts)
        for (leaf_id, text, h), _row in zip(to_embed, range(len(to_embed))):
            text_hashes[leaf_id] = h
            char_counts[leaf_id] = len(text)

    # Stitch new + reused rows into a single (N, D) array, ordered by candidates.
    import numpy as np

    dim: int
    if new_vectors is not None:
        dim = int(new_vectors.shape[1])
    else:
        # Pull dim from the prior index (we know it matches because the model matched).
        dim = int(existing_index.get("dim", 0))
        if dim == 0:
            console.print("[red]Cannot determine embedding dim from cache.[/]")
            return 2

    final = np.zeros((len(candidates), dim), dtype="float32")

    # Map leaf_id -> row in `new_vectors`
    new_row_for: Dict[str, int] = {
        leaf_id: i for i, (leaf_id, _, _) in enumerate(to_embed)
    }

    # Load prior vectors if we're reusing any.
    prior_vectors = None
    if reuse:
        try:
            prior_vectors = np.load(EMBEDDINGS_VECTORS)
        except FileNotFoundError:
            console.print(
                "[yellow]Prior index referenced cached vectors but vectors.npy "
                "is missing — re-embedding everything.[/]"
            )
            return main(["--force", "--model", args.model])

    for i, leaf_id in enumerate(candidates):
        if leaf_id in new_row_for:
            final[i] = new_vectors[new_row_for[leaf_id]]
        else:
            entry = reuse[leaf_id]
            assert prior_vectors is not None
            final[i] = prior_vectors[entry["row"]]
            text_hashes[leaf_id] = entry["hash"]
            char_counts[leaf_id] = entry.get("chars", 0)

    _write_outputs(args.model, candidates, final, text_hashes, char_counts)
    _print_summary(
        model_name=args.model,
        candidates=candidates,
        to_embed_count=len(to_embed),
        reused_count=len(reuse),
        dim=dim,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
