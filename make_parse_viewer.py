#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build an HTML viewer that shows, for each sentence:
1. sentence text
2. dependency plot (SVG)
3. constituency plot (ASCII pretty-print)
4. raw constituency parse
5. dependency table

Usage:
    python make_parse_viewer.py ALE_cds.conllu ALE_cds.txt output.html
"""

from __future__ import annotations

import argparse
import html
import io
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any

from nltk import Tree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("conllu_file", type=Path, help="Path to CoNLL-U file")
    parser.add_argument(
        "constituency_file",
        type=Path,
        help="Path to file with one bracketed constituency parse per line",
    )
    parser.add_argument("output_html", type=Path, help="Output HTML path")
    return parser.parse_args()


def parse_conllu(text: str) -> List[Tuple[List[str], List[Dict[str, str]]]]:
    """
    Returns a list of (comments, tokens) pairs.
    Tokens are dicts with a subset of CoNLL-U fields.
    """
    sentences: List[Tuple[List[str], List[Dict[str, str]]]] = []
    current_tokens: List[Dict[str, str]] = []
    comments: List[str] = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip()

        if not line.strip():
            if current_tokens or comments:
                sentences.append((comments, current_tokens))
            current_tokens = []
            comments = []
            continue

        if line.startswith("#"):
            comments.append(line)
            continue

        parts = line.split("\t")
        if len(parts) != 10:
            continue

        current_tokens.append(
            {
                "id": parts[0],
                "form": parts[1],
                "lemma": parts[2],
                "upos": parts[3],
                "head": parts[6],
                "deprel": parts[7],
            }
        )

    if current_tokens or comments:
        sentences.append((comments, current_tokens))

    return sentences


def sentence_text(comments: List[str], tokens: List[Dict[str, str]]) -> str:
    for c in comments:
        if c.startswith("# text ="):
            return c.split("=", 1)[1].strip()
    return " ".join(
        t["form"] for t in tokens if "-" not in t["id"] and "." not in t["id"]
    )


def sentence_id(comments: List[str], fallback: int) -> str:
    for c in comments:
        if c.startswith("# sent_id ="):
            return c.split("=", 1)[1].strip()
    return str(fallback)


def constituency_ascii(tree_str: str) -> str:
    try:
        tree = Tree.fromstring(tree_str)
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            tree.pretty_print()
        finally:
            sys.stdout = old_stdout
        return buf.getvalue()
    except Exception as e:
        return f"Parse error: {e}\n{tree_str}"


def dependency_svg(tokens: List[Dict[str, str]]) -> str:
    """
    Simple dependency SVG renderer.
    """
    toks = [t for t in tokens if "-" not in t["id"] and "." not in t["id"]]
    if not toks:
        return "<div>No dependency tokens</div>"

    spacing = 120
    left_margin = 55
    baseline = 190
    word_y = baseline
    upos_y = baseline + 24
    id_y = baseline + 42

    xpos = {tok["id"]: left_margin + i * spacing for i, tok in enumerate(toks)}

    arcs = []
    for tok in toks:
        dep = tok["id"]
        head = tok["head"]

        if head == "0":
            arc_h = 55
            arcs.append(("root", dep, head, tok["deprel"], arc_h))
        elif head in xpos:
            dist = abs(int(dep) - int(head))
            arc_h = 35 + dist * 26
            arcs.append(("arc", dep, head, tok["deprel"], arc_h))

    width = left_margin * 2 + max(0, len(toks) - 1) * spacing + 90
    height = baseline + 60

    parts = [
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img">'
    ]
    parts.append(
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">'
        '<path d="M0,0 L8,4 L0,8 z" fill="#444"/></marker></defs>'
    )

    for kind, dep, head, deprel, arc_h in arcs:
        x_dep = xpos[dep]

        if kind == "root":
            y_top = baseline - arc_h
            parts.append(
                f'<line x1="{x_dep}" y1="{y_top}" x2="{x_dep}" y2="{baseline-8}" '
                f'stroke="#444" stroke-width="1.7" marker-end="url(#arrow)"/>'
            )
            parts.append(
                f'<text x="{x_dep+10}" y="{y_top-6}" text-anchor="start" '
                f'font-size="15" font-family="Arial, sans-serif" fill="#222">{html.escape(deprel)}</text>'
            )
        else:
            x_head = xpos[head]
            path = (
                f"M {x_head} {baseline-8} "
                f"C {x_head} {baseline-arc_h}, {x_dep} {baseline-arc_h}, {x_dep} {baseline-8}"
            )
            parts.append(
                f'<path d="{path}" fill="none" stroke="#444" stroke-width="1.7" marker-end="url(#arrow)"/>'
            )
            label_x = (x_head + x_dep) / 2
            label_y = baseline - arc_h - 6
            parts.append(
                f'<text x="{label_x}" y="{label_y}" text-anchor="middle" '
                f'font-size="15" font-family="Arial, sans-serif" fill="#222">{html.escape(deprel)}</text>'
            )

    for tok in toks:
        x = xpos[tok["id"]]
        parts.append(
            f'<text x="{x}" y="{word_y}" text-anchor="middle" '
            f'font-size="22" font-weight="700" font-family="Arial, sans-serif" fill="#111">{html.escape(tok["form"])}</text>'
        )
        parts.append(
            f'<text x="{x}" y="{upos_y}" text-anchor="middle" '
            f'font-size="15" font-family="Arial, sans-serif" fill="#555">{html.escape(tok["upos"])}</text>'
        )
        parts.append(
            f'<text x="{x}" y="{id_y}" text-anchor="middle" '
            f'font-size="12" font-family="Arial, sans-serif" fill="#777">{html.escape(tok["id"])}</text>'
        )

    parts.append("</svg>")
    return "".join(parts)


def make_html(
    conllu_sents: List[Tuple[List[str], List[Dict[str, str]]]],
    const_lines: List[str],
) -> str:
    n = min(len(conllu_sents), len(const_lines))
    cards: List[str] = []

    for i in range(n):
        comments, tokens = conllu_sents[i]
        sid = sentence_id(comments, i + 1)
        text = sentence_text(comments, tokens)
        raw_const = const_lines[i]
        const_ascii = constituency_ascii(raw_const)

        rows = []
        for tok in tokens:
            if "-" in tok["id"] or "." in tok["id"]:
                continue
            rows.append(
                f"<tr><td>{html.escape(tok['id'])}</td>"
                f"<td>{html.escape(tok['form'])}</td>"
                f"<td>{html.escape(tok['lemma'])}</td>"
                f"<td>{html.escape(tok['upos'])}</td>"
                f"<td>{html.escape(tok['head'])}</td>"
                f"<td>{html.escape(tok['deprel'])}</td></tr>"
            )

        cards.append(
            f"""
<section class="card">
  <h2>Sentence {i+1}</h2>
  <div><strong>sent_id:</strong> {html.escape(sid)}</div>
  <div class="sent"><strong>Text:</strong> {html.escape(text)}</div>

  <h3>Dependency plot</h3>
  <div class="panel svgpanel">{dependency_svg(tokens)}</div>

  <h3>Constituency plot</h3>
  <pre class="panel treepanel">{html.escape(const_ascii)}</pre>

  <h3>Raw constituency parse</h3>
  <pre class="panel raw">{html.escape(raw_const)}</pre>

  <h3>Dependency table</h3>
  <table>
    <thead><tr><th>ID</th><th>FORM</th><th>LEMMA</th><th>UPOS</th><th>HEAD</th><th>DEPREL</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</section>
"""
        )

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Dependencies + Constituency Viewer</title>
<style>
body {{
  font-family: Arial, sans-serif;
  margin: 24px;
  line-height: 1.35;
  background: #fff;
}}
.card {{
  border: 1px solid #ccc;
  border-radius: 10px;
  padding: 16px;
  margin: 0 0 22px 0;
}}
.panel {{
  background: #f7f7f7;
  padding: 10px;
  border-radius: 8px;
  overflow: auto;
}}
pre {{
  white-space: pre-wrap;
  margin: 0;
}}
.treepanel {{
  font-family: monospace;
  font-size: 15px;
  line-height: 1.2;
}}
.svgpanel {{
  overflow-x: auto;
}}
.svgpanel svg {{
  display: block;
  min-width: 760px;
  height: auto;
}}
table {{
  border-collapse: collapse;
  width: 100%;
  margin-top: 8px;
}}
th, td {{
  border: 1px solid #ddd;
  padding: 6px 8px;
  text-align: left;
}}
th {{
  background: #f0f0f0;
}}
.sent {{
  margin: 8px 0 12px 0;
  font-size: 1.05em;
}}
.raw {{
  font-size: 13px;
}}
h3 {{
  margin-top: 18px;
}}
</style>
</head>
<body>
<h1>Dependencies + Constituency Viewer</h1>
<p>Each sentence shows the dependency plot, the constituency plot, the raw constituency parse, and the dependency table.</p>
{''.join(cards)}
</body>
</html>"""


def main() -> int:
    args = parse_args()

    conllu_text = args.conllu_file.read_text(encoding="utf-8")
    const_lines = [
        ln.strip()
        for ln in args.constituency_file.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]

    conllu_sents = parse_conllu(conllu_text)
    html_doc = make_html(conllu_sents, const_lines)

    args.output_html.write_text(html_doc, encoding="utf-8")
    print(f"Wrote {args.output_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())