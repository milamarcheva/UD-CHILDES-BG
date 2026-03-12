#!/usr/bin/env python3
"""
Build a tagged lexicon from one or more CoNLL-U files.

Each output line: `1.0 0.1 LHS --> lowercased_form`
  - LHS is chosen by --lhs: `upos` (default) or `deprel`
  - Punctuation tokens (UPOS == PUNCT) are excluded
  - Multi-word token header lines (ID contains '-' or '.') are ignored
"""

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create a tagged lexicon from CoNLL-U files.")
    p.add_argument(
        "--conllu",
        nargs="+",
        required=True,
        help="Input CoNLL-U files.",
    )
    p.add_argument(
        "--lhs",
        choices=["upos", "deprel"],
        default="upos",
        help="Which field to use as LHS (default: upos).",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output lexicon file path.",
    )
    p.add_argument(
        "--sent_id_tsv",
        default=None,
        help="Optional TSV with a column 'id' listing sent_ids to include; if provided, only sentences in this list contribute.",
    )
    return p.parse_args()


def load_allowed_ids(tsv_path: Path) -> set[str]:
    import csv

    allowed = set()
    with tsv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if "id" not in reader.fieldnames:
            raise SystemExit(f"TSV {tsv_path} missing 'id' column; columns: {reader.fieldnames}")
        for row in reader:
            sid = (row.get("id") or "").strip()
            if sid:
                allowed.add(sid)
    return allowed


def iter_tokens(conllu_path: Path, allowed_ids):
    current_id = None
    with conllu_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                current_id = None
                continue
            if line.startswith("# sent_id ="):
                current_id = line.split("=", 1)[1].strip()
                continue
            if line.startswith("#"):
                continue
            if allowed_ids is not None and (current_id is None or current_id not in allowed_ids):
                continue
            cols = line.split("\t")
            if len(cols) < 8:
                continue
            tid = cols[0]
            if not tid.isdigit():
                continue  # skip MWT headers (1-2) and ellipsis/empty nodes (1.1)
            form = cols[1]
            upos = cols[3]
            deprel = cols[7]
            yield form, upos, deprel


def main():
    args = parse_args()

    allowed_ids = None
    if args.sent_id_tsv:
        allowed_ids = load_allowed_ids(Path(args.sent_id_tsv))
        print(f"Loaded {len(allowed_ids)} allowed sent_ids from {args.sent_id_tsv}")

    combos = set()
    for path_str in args.conllu:
        path = Path(path_str)
        if not path.exists():
            raise SystemExit(f"Input not found: {path}")
        for form, upos, deprel in iter_tokens(path, allowed_ids):
            if upos.upper() == "PUNCT":
                continue
            lhs = upos if args.lhs == "upos" else deprel
            if not lhs:
                continue
            form_lc = form.lower()
            combos.add((lhs, form_lc))

    lines = []
    for lhs, form in sorted(combos, key=lambda x: (x[0], x[1])):
        lines.append(f"1.0 0.1 {lhs} --> {form}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} entries to {out_path}")


if __name__ == "__main__":
    main()
