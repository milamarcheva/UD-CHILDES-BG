#!/usr/bin/env python3
"""
Add a `normalised_utterance` column to a TSV (in place by default) using a
token-level normalization dictionary. Regex-based rules can be added later.

Default input column: lowercased_utterance
"""

import argparse
import csv
import tempfile
from pathlib import Path


# Simple token replacement dictionary (extend as needed)
NORMALIZATION_MAP = {
    "ал'оношка": "альоношка",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Normalize Bulgarian utterances in a TSV.")
    p.add_argument("--input", required=True, help="Input TSV file.")
    p.add_argument(
        "--column",
        default="lowercased_utterance",
        help="Column name containing the source utterance (default: lowercased_utterance).",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output TSV path. If omitted, overwrite the input file in place.",
    )
    return p.parse_args()


def normalize_text(text: str) -> str:
    tokens = text.split()
    out_tokens = []
    for tok in tokens:
        repl = NORMALIZATION_MAP.get(tok.lower(), tok)
        out_tokens.append(repl)
    return " ".join(out_tokens)


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Input TSV not found: {in_path}")

    out_path = Path(args.output) if args.output else in_path
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    replaced = 0
    total_rows = 0

    with in_path.open("r", encoding="utf-8", newline="") as fin, \
            tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=str(out_path.parent)) as tmp:
        reader = csv.DictReader(fin, delimiter="\t")
        fieldnames = list(reader.fieldnames or [])
        if args.column not in fieldnames:
            raise SystemExit(f"Column '{args.column}' not found; available: {fieldnames}")
        if "normalised_utterance" not in fieldnames:
            fieldnames.append("normalised_utterance")
        writer = csv.DictWriter(tmp, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for row in reader:
            total_rows += 1
            src = row.get(args.column, "") or ""
            norm = normalize_text(src)
            row["normalised_utterance"] = norm
            if norm != src:
                replaced += 1
            writer.writerow(row)

    Path(tmp.name).replace(out_path)
    print(f"Wrote {out_path}")
    print(f"Rows processed: {total_rows}")
    print(f"Rows changed (column differed): {replaced}")


if __name__ == "__main__":
    main()
