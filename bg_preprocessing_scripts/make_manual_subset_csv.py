#!/usr/bin/env python3
"""
Create a CSV subset where Manually_corrected is non-empty.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Subset LabLing CSV to rows where Manually_corrected is non-empty."
    )
    p.add_argument(
        "--input_csv",
        default="/Users/milamarcheva/Desktop/morphemic_tokenisation/data/labling_dfs/LabLing_longitudinal_with_manual.csv",
        help="Input CSV path.",
    )
    p.add_argument(
        "--output_csv",
        default="/Users/milamarcheva/Desktop/morphemic_tokenisation/data/labling_dfs/LabLing_stratified_with_manual_with_ids.csv",
        help="Output CSV path.",
    )
    p.add_argument(
        "--manual_col",
        default="Manually_corrected",
        help="Column used for filtering.",
    )
    return p.parse_args()


def is_non_empty(value: str | None) -> bool:
    if value is None:
        return False
    s = str(value).strip()
    if not s:
        return False
    if s.lower() in {"nan", "none", "null"}:
        return False
    return True


def main() -> None:
    args = parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)

    if not input_csv.exists():
        raise SystemExit(f"Input CSV not found: {input_csv}")

    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit(f"No header found in {input_csv}")
        fieldnames = list(reader.fieldnames)
        if args.manual_col not in fieldnames:
            raise SystemExit(f"Column '{args.manual_col}' not found in {input_csv}")
        rows = list(reader)

    filtered = [row for row in rows if is_non_empty(row.get(args.manual_col))]
    random.Random(42).shuffle(filtered)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered)

    print(f"Input rows: {len(rows)}")
    print(f"Filtered rows: {len(filtered)}")
    print(f"Wrote: {output_csv}")


if __name__ == "__main__":
    main()
