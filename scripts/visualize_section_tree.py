"""Optional — render an interactive HTML view of the section-tree taxonomy.

Not part of the production pipeline; this is a sanity-check tool. It reads
the step 6 outputs and writes a self-contained, dependency-free HTML page
to `output/section_tree.html` so you can explore the section -> node -> leaf
hierarchy in a browser.

Inputs:
  - output/tree/section_trees.json   (required)  — section roots + summary
  - output/tree/nodes.json           (optional)  — full node tree, used to
                                                  drill into every depth and
                                                  list child leaves. If
                                                  missing we fall back to the
                                                  primary_children one-level
                                                  view from section_trees.
  - input/leaves/*.json              (optional)  — used to render leaf titles
                                                  and quick metadata. Pass
                                                  `--no-leaves` to skip.

Run:
  python scripts/visualize_section_tree.py
  python scripts/visualize_section_tree.py --no-leaves
  python scripts/visualize_section_tree.py --output /tmp/tree.html
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from html import escape
from pathlib import Path
from typing import Any, Dict, List

from rich.table import Table

from _common import (
    LEAVES_DIR,
    NODES_FILE,
    ROOT,
    SECTION_TREES_FILE,
    SECTION_TREE_HTML,
    console,
)


# Border colors for each node `kind`. Matches palette in visualize_graph.py.
KIND_COLORS: Dict[str, str] = {
    "section_root": "#4C78A8",
    "entity_type_group": "#54A24B",
    "entity_group": "#F58518",
    "graph_cluster": "#B279A2",
    "orphan_bucket": "#BAB0AC",
}


def _load_leaf_meta() -> Dict[str, Dict[str, Any]]:
    """Pull title/status/business area for every leaf, for tooltips.

    Returns {} silently if `input/leaves/` is missing or empty.
    """
    meta: Dict[str, Dict[str, Any]] = {}
    if not LEAVES_DIR.exists():
        return meta
    for path in sorted(LEAVES_DIR.glob("file-*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        leaf_id = data.get("leaf_id") or path.stem
        cls = data.get("classification") or {}
        src = data.get("source") or {}
        emb = data.get("embedding") or {}
        summary = (data.get("summary") or {}).get("one_sentence", "")
        business_area = [str(x).lower() for x in (cls.get("business_area") or [])]
        meta[leaf_id] = {
            "title": (emb.get("title") or src.get("name") or leaf_id).strip(),
            "status": cls.get("status", "unknown"),
            "business_area": business_area[0] if business_area else "unknown",
            "summary": summary,
            "url": src.get("url", ""),
        }
    return meta


def _build_tree(
    node_id: str,
    nodes: Dict[str, Any],
    leaf_meta: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Recursively project a node from `nodes.json` into the render schema."""
    node = nodes.get(node_id)
    if not node:
        return {
            "id": node_id,
            "title": node_id,
            "kind": "missing",
            "depth": 0,
            "leaf_count": 0,
            "anchor": None,
            "summary_short": "",
            "key_themes": [],
            "children": [],
            "leaves": [],
        }

    children: List[Dict[str, Any]] = []
    child_leaf_ids: List[str] = []
    for child in node.get("children", []) or []:
        if child.get("kind") == "leaf":
            child_leaf_ids.append(child.get("leaf_id"))
        else:
            cid = child.get("node_id")
            if cid:
                children.append(_build_tree(cid, nodes, leaf_meta))

    # Defensive: prefer the explicit child-leaf list, fall back to
    # `leaf_ids` on the node itself (only used when no leaves came in as
    # children, which shouldn't happen for the current schema but is cheap
    # to be tolerant of).
    if not child_leaf_ids:
        child_leaf_ids = list(node.get("leaf_ids") or [])

    leaves = []
    for leaf_id in child_leaf_ids:
        info = leaf_meta.get(leaf_id, {})
        leaves.append({
            "id": leaf_id,
            "title": info.get("title", leaf_id),
            "status": info.get("status", "unknown"),
            "business_area": info.get("business_area", "unknown"),
            "summary": info.get("summary", ""),
            "url": info.get("url", ""),
        })

    return {
        "id": node_id,
        "title": node.get("title", node_id),
        "kind": node.get("kind", "node"),
        "depth": node.get("depth", 0),
        "leaf_count": node.get("leaf_count_recursive", 0),
        "anchor": node.get("anchor"),
        "summary_short": node.get("summary_short", ""),
        "key_themes": node.get("key_themes") or [],
        "children": children,
        "leaves": leaves,
    }


def _build_tree_fallback(section_id: str, section: Dict[str, Any]) -> Dict[str, Any]:
    """One-level-deep tree from `section_trees.primary_children` when
    `nodes.json` is unavailable."""
    return {
        "id": section.get("root_node_id", section_id),
        "title": section.get("label", section_id),
        "kind": "section_root",
        "depth": 0,
        "leaf_count": section.get("leaf_count_recursive", 0),
        "anchor": None,
        "summary_short": "",
        "key_themes": [],
        "leaves": [],
        "children": [
            {
                "id": child.get("node_id"),
                "title": child.get("title", child.get("node_id")),
                "kind": child.get("kind", "node"),
                "depth": child.get("depth", 1),
                "leaf_count": child.get("leaf_count_recursive", 0),
                "anchor": child.get("anchor"),
                "summary_short": "",
                "key_themes": [],
                "children": [],
                "leaves": [],
            }
            for child in section.get("primary_children", []) or []
        ],
    }


def _count_kinds(node: Dict[str, Any], counts: Counter) -> None:
    counts[node["kind"]] += 1
    for child in node["children"]:
        _count_kinds(child, counts)


def _count_leaves(node: Dict[str, Any]) -> int:
    """Total leaf references inside this subtree (with duplicates if a leaf
    appears in multiple nodes — matches `leaf_count_recursive` semantics)."""
    total = len(node["leaves"])
    for child in node["children"]:
        total += _count_leaves(child)
    return total


def _render_html(
    sections: List[Dict[str, Any]],
    kind_counts: Dict[str, int],
    section_bars: List[Dict[str, Any]],
    meta: Dict[str, Any],
    leaf_placements_count: int,
    distinct_leaves: int,
) -> str:
    """Assemble the full self-contained HTML document."""

    def to_json(obj: Any) -> str:
        # Escape any literal "</" inside JSON so a stray "</script>" in a
        # value can't break out of the inline script tag.
        return json.dumps(obj, ensure_ascii=False).replace("</", "<\\/")

    sections_json = to_json(sections)
    kind_counts_json = to_json(kind_counts)
    section_bars_json = to_json(section_bars)
    kind_colors_json = to_json(KIND_COLORS)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Wiki section tree</title>
<style>
  :root {{
    --bg: #f6f7fb;
    --panel: #ffffff;
    --ink: #17213c;
    --muted: #647084;
    --line: #d9deea;
    --accent: #2f5cc0;
    --chip: #eef2ff;
    --leaf-bg: #fffbeb;
    --leaf-line: #f1d488;
  }}
  body {{
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
    color: var(--ink);
    background: var(--bg);
  }}
  header {{
    position: sticky; top: 0; z-index: 10;
    background: var(--panel);
    border-bottom: 1px solid var(--line);
    padding: 12px 18px;
    display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
  }}
  header h1 {{ font-size: 17px; margin: 0 8px 0 0; }}
  input, select, button {{
    padding: 6px 8px;
    border: 1px solid var(--line);
    border-radius: 6px;
    background: white;
    font-size: 13px;
  }}
  input {{ min-width: 220px; }}
  button {{ cursor: pointer; }}
  label {{ font-size: 13px; color: var(--muted); display: flex; gap: 6px; align-items: center; }}

  main {{
    display: grid;
    grid-template-columns: 320px 1fr;
    min-height: calc(100vh - 58px);
  }}
  aside {{
    border-right: 1px solid var(--line);
    background: var(--panel);
    padding: 16px;
    overflow: auto;
  }}
  aside h2 {{ font-size: 14px; margin: 18px 0 8px; }}
  aside h2:first-child {{ margin-top: 0; }}
  #canvas {{ padding: 20px 24px; overflow: auto; }}

  .metric {{
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 8px;
    padding: 6px 0;
    border-bottom: 1px solid #eef0f5;
    font-size: 13px;
  }}
  .metric span:last-child {{ font-weight: 600; }}

  .bar-row {{ margin: 6px 0; font-size: 12px; }}
  .bar-label {{ display: flex; justify-content: space-between; gap: 8px; }}
  .bar {{
    height: 6px;
    background: #e8ecf5;
    border-radius: 4px;
    overflow: hidden;
    margin-top: 3px;
  }}
  .bar > div {{ height: 100%; background: var(--accent); }}

  .tree {{ display: flex; flex-direction: column; gap: 14px; }}

  details.node {{
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 10px;
    box-shadow: 0 1px 4px rgba(20, 30, 60, 0.04);
    border-left-width: 4px;
    border-left-style: solid;
  }}
  details.node > summary {{
    cursor: pointer;
    padding: 10px 14px;
    list-style: none;
    display: flex;
    gap: 10px;
    align-items: center;
    justify-content: space-between;
  }}
  details.node > summary::-webkit-details-marker {{ display: none; }}
  details.node > summary::before {{
    content: "▸";
    color: var(--muted);
    width: 12px;
    display: inline-block;
    transition: transform 0.15s;
  }}
  details.node[open] > summary::before {{ transform: rotate(90deg); }}

  .node-title {{ font-weight: 600; font-size: 14px; flex: 1; }}
  .node-section-meta {{
    color: var(--muted);
    font-weight: normal;
    font-size: 12px;
    white-space: nowrap;
  }}

  .node-body {{ padding: 0 14px 12px 28px; position: relative; }}
  .node-body::before {{
    content: "";
    position: absolute;
    left: 14px; top: 0; bottom: 12px;
    border-left: 2px solid var(--line);
  }}
  .node-summary {{
    font-size: 13px;
    color: var(--muted);
    margin: 0 0 8px 0;
  }}
  .meta-chips {{ display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 8px; }}
  .chip {{
    background: var(--chip);
    color: #243b77;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 12px;
    line-height: 1.6;
  }}
  .chip.kind {{ font-weight: 600; }}

  .children-wrap {{ display: flex; flex-direction: column; gap: 8px; }}

  .leaves {{ display: flex; flex-direction: column; gap: 6px; margin: 6px 0; }}
  .leaf {{
    background: var(--leaf-bg);
    border: 1px solid var(--leaf-line);
    border-radius: 8px;
    padding: 8px 10px;
    font-size: 13px;
  }}
  .leaf-title {{ font-weight: 600; }}
  .leaf-meta {{
    font-size: 11px;
    color: var(--muted);
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 3px;
  }}
  .leaf-summary {{
    font-size: 12px;
    color: #3c4866;
    margin-top: 4px;
  }}
  .leaf a {{ color: inherit; text-decoration: none; }}
  .leaf a:hover {{ text-decoration: underline; }}

  .hidden {{ display: none !important; }}

  code {{
    background: #eef0f6;
    border-radius: 4px;
    padding: 1px 4px;
    font-family: "SF Mono", Menlo, Consolas, monospace;
    font-size: 12px;
  }}

  .legend {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; }}
  .legend .swatch {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
  }}
  .legend .dot {{
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 2px;
  }}
</style>
</head>
<body>
<header>
  <h1>Wiki section tree</h1>
  <label>Search
    <input id="search" placeholder="section, node, leaf, anchor, theme">
  </label>
  <label>Kind
    <select id="kindFilter">
      <option value="">all</option>
    </select>
  </label>
  <button id="expandAll">Expand all</button>
  <button id="collapseAll">Collapse all</button>
  <button id="expandTop">Expand sections only</button>
</header>

<main>
  <aside>
    <h2>Summary</h2>
    <div class="metric"><span>Generated</span><span>{escape(str(meta.get("generated_at", "unknown")))}</span></div>
    <div class="metric"><span>Tree version</span><span>{escape(str(meta.get("tree_taxonomy_version", "unknown")))}</span></div>
    <div class="metric"><span>Sections</span><span>{len(sections)}</span></div>
    <div class="metric"><span>Leaf placements</span><span>{leaf_placements_count}</span></div>
    <div class="metric"><span>Distinct leaves rendered</span><span>{distinct_leaves}</span></div>

    <h2>Node kinds</h2>
    <div id="kindStats"></div>
    <div class="legend" id="kindLegend"></div>

    <h2>Largest sections</h2>
    <div id="sectionBars"></div>

    <p style="font-size:12px;color:var(--muted);line-height:1.4;margin-top:18px;">
      This view drills through the full <code>nodes.json</code> tree when
      available, falling back to <code>section_trees.primary_children</code>.
      Leaves carry their <code>one_sentence</code> summary if
      <code>input/leaves/</code> is on disk.
    </p>
  </aside>

  <section id="canvas">
    <div id="tree" class="tree"></div>
  </section>
</main>

<script>
  const sections = {sections_json};
  const kindCounts = {kind_counts_json};
  const sectionBars = {section_bars_json};
  const kindColors = {kind_colors_json};

  const tree = document.getElementById("tree");
  const searchInput = document.getElementById("search");
  const kindFilter = document.getElementById("kindFilter");

  function esc(s) {{
    return String(s ?? "").replace(/[&<>"']/g, m => ({{
      "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#039;"
    }}[m]));
  }}
  function kindLabel(kind) {{
    return String(kind || "unknown").replace(/_/g, " ");
  }}
  function anchorText(anchor) {{
    if (!anchor) return "";
    const parts = [];
    if (anchor.entity_type) parts.push(anchor.entity_type);
    if (anchor.label) parts.push(anchor.label);
    if (anchor.slug && anchor.slug !== anchor.label) parts.push(anchor.slug);
    return parts.join(" · ");
  }}

  function buildSearchBlob(node) {{
    const parts = [
      node.title, node.id, node.kind, anchorText(node.anchor),
      ...(node.key_themes || []),
      node.summary_short || "",
      ...(node.leaves || []).flatMap(l => [l.title, l.id, l.business_area, l.status, l.summary]),
    ];
    for (const c of (node.children || [])) parts.push(buildSearchBlob(c));
    return parts.join(" ").toLowerCase();
  }}

  function renderLeaf(leaf) {{
    const title = leaf.url
      ? `<a href="${{esc(leaf.url)}}" target="_blank" rel="noopener">${{esc(leaf.title)}}</a>`
      : esc(leaf.title);
    return `
      <div class="leaf" data-kind="leaf"
           data-search="${{esc([leaf.title, leaf.id, leaf.business_area, leaf.status, leaf.summary].join(" ").toLowerCase())}}">
        <div class="leaf-title">${{title}}</div>
        <div class="leaf-meta">
          <span class="chip">${{esc(leaf.business_area || "unknown")}}</span>
          <span class="chip">status: ${{esc(leaf.status || "unknown")}}</span>
          <span class="chip"><code>${{esc(leaf.id)}}</code></span>
        </div>
        ${{leaf.summary ? `<div class="leaf-summary">${{esc(leaf.summary)}}</div>` : ""}}
      </div>
    `;
  }}

  function renderNode(node, isRoot) {{
    const color = kindColors[node.kind] || "#9aa3b8";
    const childCount = (node.children || []).length + (node.leaves || []).length;
    const anchor = anchorText(node.anchor);
    const themes = (node.key_themes || []).slice(0, 4);

    const chips = [
      `<span class="chip kind" style="background:${{color}}22;color:${{color}}">${{esc(kindLabel(node.kind))}}</span>`,
      `<span class="chip">${{node.leaf_count}} leaves</span>`,
      `<span class="chip">depth ${{node.depth}}</span>`,
    ];
    if (anchor) chips.push(`<span class="chip">${{esc(anchor)}}</span>`);
    for (const t of themes) chips.push(`<span class="chip">${{esc(t)}}</span>`);

    const childrenHtml = (node.children || []).map(c => renderNode(c, false)).join("");
    const leavesHtml = (node.leaves || []).length
      ? `<div class="leaves">${{(node.leaves || []).map(renderLeaf).join("")}}</div>`
      : "";

    const searchBlob = buildSearchBlob(node);

    return `
      <details class="node" ${{isRoot ? "open" : ""}}
               data-kind="${{esc(node.kind)}}"
               data-search="${{esc(searchBlob)}}"
               style="border-left-color:${{color}}">
        <summary>
          <span class="node-title">${{esc(node.title)}}</span>
          <span class="node-section-meta">${{node.leaf_count}} leaves · ${{childCount}} children</span>
        </summary>
        <div class="node-body">
          ${{node.summary_short ? `<p class="node-summary">${{esc(node.summary_short)}}</p>` : ""}}
          <div class="meta-chips">${{chips.join("")}}</div>
          <div class="children-wrap">
            ${{leavesHtml}}
            ${{childrenHtml}}
          </div>
        </div>
      </details>
    `;
  }}

  function renderTree() {{
    tree.innerHTML = sections.map(s => renderNode(s, true)).join("");
  }}

  function renderFilters() {{
    Object.keys(kindCounts).sort().forEach(kind => {{
      const opt = document.createElement("option");
      opt.value = kind;
      opt.textContent = `${{kindLabel(kind)}} (${{kindCounts[kind]}})`;
      kindFilter.appendChild(opt);
    }});

    document.getElementById("kindStats").innerHTML = Object.entries(kindCounts)
      .sort((a,b) => b[1] - a[1])
      .map(([kind, count]) => `
        <div class="metric">
          <span><span class="dot" style="background:${{kindColors[kind] || "#9aa3b8"}}"></span>
                ${{esc(kindLabel(kind))}}</span>
          <span>${{count}}</span>
        </div>`)
      .join("");

    document.getElementById("kindLegend").innerHTML = Object.keys(kindColors)
      .map(kind => `<span class="swatch"><span class="dot" style="background:${{kindColors[kind]}}"></span>${{esc(kindLabel(kind))}}</span>`)
      .join("");

    const max = Math.max(...sectionBars.map(s => s.leaf_count), 1);
    document.getElementById("sectionBars").innerHTML = sectionBars.slice(0, 12).map(s => `
      <div class="bar-row">
        <div class="bar-label"><span>${{esc(s.section)}}</span><span>${{s.leaf_count}}</span></div>
        <div class="bar"><div style="width:${{Math.round((s.leaf_count / max) * 100)}}%"></div></div>
      </div>
    `).join("");
  }}

  function applyFilters() {{
    const q = searchInput.value.toLowerCase().trim();
    const kind = kindFilter.value;
    const all = document.querySelectorAll("details.node, .leaf");

    all.forEach(el => {{
      const matchesSearch = !q || (el.dataset.search || "").includes(q);
      const matchesKind = !kind || el.dataset.kind === kind || (kind !== "leaf" && el.tagName === "DETAILS" && [...el.querySelectorAll(`[data-kind="${{kind}}"]`)].length > 0);
      el.classList.toggle("hidden", !(matchesSearch && (kind === "" || matchesKind)));
      if ((q || kind) && el.tagName === "DETAILS") el.open = true;
    }});
  }}

  searchInput.addEventListener("input", applyFilters);
  kindFilter.addEventListener("change", applyFilters);
  document.getElementById("expandAll").addEventListener("click", () => {{
    document.querySelectorAll("details.node").forEach(d => d.open = true);
  }});
  document.getElementById("collapseAll").addEventListener("click", () => {{
    document.querySelectorAll("details.node").forEach(d => d.open = false);
  }});
  document.getElementById("expandTop").addEventListener("click", () => {{
    document.querySelectorAll("details.node").forEach(d => {{
      d.open = (d.parentElement && d.parentElement.id === "tree");
    }});
  }});

  renderTree();
  renderFilters();
</script>
</body>
</html>
"""


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--section-trees",
        default=str(SECTION_TREES_FILE),
        help=(
            "Path to section_trees.json "
            f"(default: {SECTION_TREES_FILE.relative_to(ROOT)})"
        ),
    )
    parser.add_argument(
        "--nodes",
        default=str(NODES_FILE),
        help=(
            "Path to nodes.json. Used to drill into every child + leaf. "
            f"(default: {NODES_FILE.relative_to(ROOT)})"
        ),
    )
    parser.add_argument(
        "--output",
        default=str(SECTION_TREE_HTML),
        help=(
            "Output HTML path "
            f"(default: {SECTION_TREE_HTML.relative_to(ROOT)})"
        ),
    )
    parser.add_argument(
        "--no-leaves",
        action="store_true",
        help="Skip loading per-leaf metadata from input/leaves/.",
    )
    args = parser.parse_args(argv)

    section_trees_path = Path(args.section_trees).resolve()
    nodes_path = Path(args.nodes).resolve()
    output_path = Path(args.output).resolve()

    if not section_trees_path.exists():
        console.print(f"[red]Missing[/] {section_trees_path}")
        console.print("Run [bold]scripts/discover_clusters.py[/] first.")
        return 2

    console.print(f"Reading [bold]{section_trees_path.relative_to(ROOT)}[/]...")
    section_doc = json.loads(section_trees_path.read_text(encoding="utf-8"))
    section_trees = section_doc.get("section_trees", {})
    leaf_placements = section_doc.get("leaf_placements", {})

    leaf_meta: Dict[str, Dict[str, Any]] = {}
    if not args.no_leaves:
        console.print(f"Loading leaf metadata from [bold]{LEAVES_DIR.relative_to(ROOT)}[/]...")
        leaf_meta = _load_leaf_meta()
        console.print(f"  Loaded [bold]{len(leaf_meta)}[/] leaves.")

    nodes_index: Dict[str, Any] = {}
    if nodes_path.exists():
        console.print(f"Reading [bold]{nodes_path.relative_to(ROOT)}[/]...")
        nodes_doc = json.loads(nodes_path.read_text(encoding="utf-8"))
        nodes_index = nodes_doc.get("nodes", {})
        console.print(f"  Loaded [bold]{len(nodes_index)}[/] nodes.")
    else:
        console.print(
            "[yellow]nodes.json not found — falling back to one-level "
            "primary_children view.[/]"
        )

    sections: List[Dict[str, Any]] = []
    for section_id, section in section_trees.items():
        root_node_id = section.get("root_node_id")
        if nodes_index and root_node_id in nodes_index:
            tree = _build_tree(root_node_id, nodes_index, leaf_meta)
        else:
            tree = _build_tree_fallback(section_id, section)
        tree["section_id"] = section_id
        # Prefer the section label from section_trees, which is the curated
        # human-facing name (e.g. "Channel / Partners").
        tree["title"] = section.get("label", tree["title"])
        sections.append(tree)

    sections.sort(key=lambda s: s.get("leaf_count", 0), reverse=True)

    kind_counts: Counter = Counter()
    for s in sections:
        _count_kinds(s, kind_counts)

    section_bars = sorted(
        [
            {
                "section": s["title"],
                "section_id": s.get("section_id", s["id"]),
                "leaf_count": s["leaf_count"],
                "children_count": len(s["children"]) + len(s["leaves"]),
            }
            for s in sections
        ],
        key=lambda x: x["leaf_count"],
        reverse=True,
    )

    distinct_leaves = len({
        l["id"]
        for s in sections
        for l in _all_leaves(s)
    })

    html = _render_html(
        sections=sections,
        kind_counts=dict(kind_counts),
        section_bars=section_bars,
        meta=section_doc,
        leaf_placements_count=len(leaf_placements),
        distinct_leaves=distinct_leaves,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

    table = Table(title="Section tree visualization", show_lines=False)
    table.add_column("Field", style="bold")
    table.add_column("Value")
    try:
        out_rel = output_path.relative_to(ROOT)
    except ValueError:
        out_rel = output_path
    table.add_row("Output", str(out_rel))
    table.add_row("Sections", str(len(sections)))
    table.add_row(
        "Nodes",
        ", ".join(f"{k}={v}" for k, v in kind_counts.most_common()),
    )
    table.add_row("Leaf placements", str(len(leaf_placements)))
    table.add_row("Distinct leaves rendered", str(distinct_leaves))
    table.add_row("Leaves with metadata", str(len(leaf_meta)))
    console.print(table)
    console.print(f"[green]Open[/] [bold]{out_rel}[/] in your browser.")
    return 0


def _all_leaves(node: Dict[str, Any]):
    for leaf in node["leaves"]:
        yield leaf
    for child in node["children"]:
        yield from _all_leaves(child)


if __name__ == "__main__":
    sys.exit(main())
