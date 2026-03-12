#!/usr/bin/env python3
"""
Add deterministic row IDs to every row in a LabLing CSV.

Default behavior:
- Reads LabLing_longitudinal_with_manual.csv
- Writes back in place (atomic replace)
- Sets/overwrites `utterance_id` for all rows
"""

from __future__ import annotations

import argparse
import csv
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Add IDs to all rows in a LabLing CSV.")
    p.add_argument(
        "--input_csv",
        default="/Users/milamarcheva/Desktop/morphemic_tokenisation/data/labling_dfs/LabLing_longitudinal_with_manual.csv",
        help="Input CSV path.",
    )
    p.add_argument(
        "--output_csv",
        default=None,
        help="Output CSV path (default: overwrite input CSV in place).",
    )
    p.add_argument(
        "--id_col",
        default="utterance_id",
        help="Name of ID column.",
    )
    p.add_argument(
        "--id_prefix",
        default="LLM_",
        help="ID prefix.",
    )
    p.add_argument(
        "--start_id",
        type=int,
        default=1,
        help="Start integer for generated IDs.",
    )
    p.add_argument(
        "--zero_pad",
        type=int,
        default=6,
        help="Zero padding width for numeric part.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise SystemExit(f"Input CSV not found: {input_csv}")

    output_csv = Path(args.output_csv) if args.output_csv else input_csv

    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit(f"No header found in {input_csv}")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    if args.id_col not in fieldnames:
        fieldnames = [args.id_col] + fieldnames

    for i, row in enumerate(rows, start=args.start_id):
        row[args.id_col] = f"{args.id_prefix}{i:0{args.zero_pad}d}"

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", delete=False, dir=str(output_csv.parent)
    ) as tmp:
        writer = csv.DictWriter(tmp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    Path(tmp.name).replace(output_csv)
    print(f"Updated CSV: {output_csv}")
    print(f"Rows with IDs: {len(rows)}")


if __name__ == "__main__":
    main()
