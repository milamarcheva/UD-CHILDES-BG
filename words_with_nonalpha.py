#!/usr/bin/env python3
"""
Extract all unique words that contain at least one non-alphabetic character
from a TSV with one sentence per row.

Defaults:
- input column: lowercased_utterance
- output: stdout (one word per line)
"""

import argparse
import csv
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="List words containing non-alphabetic characters from a TSV column.")
    p.add_argument("--input", required=True, help="Input TSV file (one sentence per row).")
    p.add_argument(
        "--column",
        default="lowercased_utterance",
        help="TSV column containing the sentence text (default: lowercased_utterance).",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Optional output file path (one word per line). If omitted, print to stdout.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Input TSV not found: {in_path}")

    words = set()
    with in_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if args.column not in reader.fieldnames:
            raise SystemExit(f"Column '{args.column}' not found; available: {reader.fieldnames}")
        for row in reader:
            text = row.get(args.column, "") or ""
            for tok in text.split():
                if any(not ch.isalpha() for ch in tok):
                    words.add(tok)

    output_text = "\n".join(sorted(words)) + ("\n" if words else "")
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_text, encoding="utf-8")
        print(f"Wrote {len(words)} words to {out_path}")
    else:
        sys.stdout.write(output_text)


if __name__ == "__main__":
    main()
