#!/usr/bin/env python3
"""
Merge manually verified parses into a larger automatically parsed file.

For each parse in the automatic file:
  - compute its yield (terminal tokens), lowercased
  - if that yield exists in the manual file (also lowercased yields), use the manual parse
  - otherwise keep the automatic parse

The output preserves the order and line count of the automatic file.
"""

import argparse
import re
from pathlib import Path

# Capture tokens from structures like (TAG token)
TOKEN_RE = re.compile(r"\((?:[^()\s]+)\s+([^()\s]+)\)")


def yield_from_parse(parse_line: str) -> str:
    tokens = TOKEN_RE.findall(parse_line)
    return " ".join(tokens).lower()


def build_manual_map(manual_path: Path) -> dict[str, str]:
    mapping = {}
    with manual_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            yld = yield_from_parse(line)
            # keep first occurrence
            mapping.setdefault(yld, line)
    return mapping


def merge_parses(auto_path: Path, manual_map: dict[str, str], out_path: Path) -> tuple[int, int]:
    out_lines = []
    used_manual = 0
    used_auto = 0
    with auto_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                out_lines.append(line)
                continue
            yld = yield_from_parse(line)
            if yld in manual_map:
                out_lines.append(manual_map[yld])
                used_manual += 1
            else:
                out_lines.append(line)
                used_auto += 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return used_manual, used_auto


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge manual parses into automatic parses by yield match.")
    p.add_argument("--auto_parses", required=True, help="Path to automatic parses file (one parse per line).")
    p.add_argument("--manual_parses", required=True, help="Path to manually verified parses file.")
    p.add_argument("--output", required=True, help="Path to merged output file.")
    return p.parse_args()


def main():
    args = parse_args()
    auto_path = Path(args.auto_parses)
    manual_path = Path(args.manual_parses)
    out_path = Path(args.output)

    manual_map = build_manual_map(manual_path)
    used_manual, used_auto = merge_parses(auto_path, manual_map, out_path)
    total = used_manual + used_auto
    print(f"Merged parses written to {out_path}")
    print(f"Manual parses used: {used_manual} ({used_manual/total*100:.2f}%)")
    print(f"Auto parses used:   {used_auto} ({used_auto/total*100:.2f}%)")


if __name__ == "__main__":
    main()
