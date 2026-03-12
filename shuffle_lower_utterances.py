#!/usr/bin/env python3
import argparse
import csv
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    data_dir = root / "conllufiles_sentences_lexicons"

    parser = argparse.ArgumentParser(
        description=(
            "Shuffle bg_lower_utterances.tsv with a fixed seed and write both "
            "the shuffled TSV and a one-sentence-per-line yields file."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=data_dir / "bg_lower_utterances.tsv",
        help="Input TSV path.",
    )
    parser.add_argument(
        "--output-tsv",
        type=Path,
        default=data_dir / "bg_lower_utterances_shuffled.tsv",
        help="Output shuffled TSV path.",
    )
    parser.add_argument(
        "--output-yields",
        type=Path,
        default=data_dir / "bg_yields.txt",
        help="Output yields path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling.",
    )
    parser.add_argument(
        "--text-column",
        default="lowercased_utterance",
        help="TSV column to write to the yields file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with args.input.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise SystemExit(f"No header found in {args.input}")
        if args.text_column not in fieldnames:
            raise SystemExit(
                f"Column {args.text_column!r} not found in {args.input}. "
                f"Available columns: {', '.join(fieldnames)}"
            )
        rows = list(reader)

    random.Random(args.seed).shuffle(rows)

    with args.output_tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    with args.output_yields.open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write((row.get(args.text_column) or "").strip() + "\n")

    print(f"Wrote shuffled TSV: {args.output_tsv}")
    print(f"Wrote yields file: {args.output_yields}")
    print(f"Rows shuffled: {len(rows)}")


if __name__ == "__main__":
    main()
