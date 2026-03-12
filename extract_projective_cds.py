#!/usr/bin/env python3
"""
Extract projective child-speech sentences from an INCEpTION export.

Given the project root (the directory that contains `annotation/`), this script:
- Iterates over subfolders under `annotation/` whose names end with `_cds.conllu`.
- For each folder, prefers `mila.conllu`; if absent, falls back to the first annotator
  file (skipping `INITIAL_CAS.conllu`).
- Detects non-projective sentences (any crossing dependency arcs, heads > 0) and
  FILTERS THEM OUT.
- Writes only projective sentences (no crossing arcs) with their original
  comments/lines to an output folder (overwrites existing files). By default the
  folder is `bg_projective_cds` under the project root, but you can choose a
  different base via --output-root.

Usage:
    python3 extract_projective_cds.py --project-root PATH \
        [--output-dir bg_projective_cds] [--output-root PATH]
"""

from __future__ import annotations

import argparse
import pathlib


def normalize_token_line(line: str) -> str:
    """Normalize common non-CoNLL placeholders to '_' in token lines."""
    if not line or line.startswith("#") or "\t" not in line:
        return line
    fields = line.split("\t")
    # Keep ID and FORM intact; normalize other fields.
    for i in range(2, len(fields)):
        val = fields[i].strip()
        if val == "" or val.lower() in {"none", "null"}:
            fields[i] = "_"
    return "\t".join(fields)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--project-root",
        required=True,
        type=pathlib.Path,
        help="Path to INCEpTION export root (containing annotation/)",
    )
    p.add_argument(
        "--output-dir",
        default="bg_projective_cds",
        type=pathlib.Path,
        help="Output directory name or path; if relative, it is created under --output-root (or project root by default)",
    )
    p.add_argument(
        "--output-root",
        default=None,
        type=pathlib.Path,
        help="Base directory for relative output paths (defaults to project root). Use this to write outside the project.",
    )
    return p.parse_args()


def is_nonprojective(sentence_lines: list[str]) -> bool:
    """Detect non-projectivity based on token/head pairs."""
    arcs = []
    for line in sentence_lines:
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 7:
            continue
        tid = parts[0]
        head = parts[6]
        if "-" in tid or "." in tid:
            continue  # skip MWT/empty nodes
        try:
            tid_i = int(tid)
            head_i = int(head)
        except ValueError:
            continue
        if head_i == 0:
            continue
        start, end = sorted((tid_i, head_i))
        arcs.append((start, end))
    for i in range(len(arcs)):
        a_start, a_end = arcs[i]
        for j in range(i + 1, len(arcs)):
            b_start, b_end = arcs[j]
            # Strict crossing (not nested)
            if (a_start < b_start < a_end < b_end) or (b_start < a_start < b_end < a_end):
                return True
    return False


def sentences_from_file(path: pathlib.Path):
    """Yield (sent_text, lines) where lines include comments and tokens."""
    lines = path.read_text(encoding="utf-8").splitlines()
    buf: list[str] = []
    total = 0
    for line in lines:
        if line.strip() == "":
            if buf:
                yield buf
                total += 1
                buf = []
            continue
        buf.append(line)
    if buf:
        yield buf
        total += 1
    return total


def main():
    args = parse_args()
    project_root = args.project_root.resolve()
    ann_root = project_root / "annotation"
    if not ann_root.is_dir():
        raise SystemExit(f"annotation/ not found under {project_root}")

    base_root = args.output_root.resolve() if args.output_root else project_root
    out_dir = args.output_dir if args.output_dir.is_absolute() else (base_root / args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_written = 0
    for doc_dir in sorted(ann_root.iterdir()):
        if not doc_dir.is_dir():
            continue
        if not doc_dir.name.endswith("_cds.conllu"):
            continue
        preferred_path = doc_dir / "mila.conllu"
        if preferred_path.exists():
            ann_path = preferred_path
            ann_name = "mila"
        else:
            # Fallback: choose the first annotator .conllu (excluding INITIAL_CAS)
            candidates = sorted(
                p for p in doc_dir.glob("*.conllu") if p.name.lower() != "initial_cas.conllu"
            )
            if not candidates:
                print(f"[skip] no annotator files in {doc_dir.name}")
                continue
            ann_path = candidates[0]
            ann_name = ann_path.stem
            print(f"[fallback] using {ann_name} in {doc_dir.name} (mila missing)")

        output_path = out_dir / doc_dir.name
        kept_sentences = []
        total_sentences = 0
        for sent_lines in sentences_from_file(ann_path):
            total_sentences += 1
            normalized = [normalize_token_line(line) for line in sent_lines]
            if not is_nonprojective(normalized):
                kept_sentences.append("\n".join(normalized))

        if kept_sentences:
            output_path.write_text("\n\n".join(kept_sentences) + "\n", encoding="utf-8")
            total_written += len(kept_sentences)
            print(
                f"[write] {output_path}  ({len(kept_sentences)} / {total_sentences} sentences, projective only)"
            )
        else:
            # Ensure old file, if any, is removed to reflect zero findings.
            if output_path.exists():
                output_path.unlink()
            print(f"[none]  {doc_dir.name} (no projective sentences after filtering)")

    print(f"Done. Total projective sentences written: {total_written}")


if __name__ == "__main__":
    main()
