#!/usr/bin/env python3
"""
Extract all unique words containing a given character from a TSV.

Default column: lowercased_utterance
Default output: stdout (one word per line)
"""

import argparse
import csv
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="List words containing a character from a TSV column.")
    p.add_argument("--input", required=True, help="Input TSV file.")
    p.add_argument("--char", required=True, help="Character to search for (use a single codepoint).")
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

    if len(args.char) == 0:
        raise SystemExit("Error: --char must be non-empty")
    needle = args.char

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
                if needle in tok:
                    words.add(tok)

    output_lines = "\n".join(sorted(words)) + ("\n" if words else "")
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_lines, encoding="utf-8")
        print(f"Wrote {len(words)} words to {out_path}")
    else:
        sys.stdout.write(output_lines)


if __name__ == "__main__":
    main()
