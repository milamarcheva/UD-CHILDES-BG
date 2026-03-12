#!/usr/bin/env python3
"""
Batch-run Bulgarian Stanza dependency parsing for all files in LabLing_inception.

- Uses bg_stanza_deps_with_charts.py in TXT mode (one sentence per line)
- Does NOT write charts (no --output_charts passed)
- Writes .conllu outputs (default: next to each input file)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch-parse all files in LabLing_inception to CoNLL-U (no charts)."
    )
    p.add_argument(
        "--input-dir",
        default="/Users/milamarcheva/Desktop/morphemic_tokenisation/data/labling_dfs/LabLing_inception",
        help="Input directory with sentence files.",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for .conllu files. Default: same as input-dir.",
    )
    p.add_argument(
        "--parser-script",
        default="/Users/milamarcheva/Desktop/morphemic_tokenisation/bg_stanza_deps_with_charts.py",
        help="Path to bg_stanza_deps_with_charts.py.",
    )
    p.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to run the parser script.",
    )
    p.add_argument("--use-gpu", action="store_true", help="Pass --use_gpu to parser script.")
    p.add_argument("--limit", type=int, default=None, help="Optional sentence limit per file.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    parser_script = Path(args.parser_script)

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")
    if not parser_script.is_file():
        raise SystemExit(f"Parser script not found: {parser_script}")
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in input_dir.rglob("*") if p.is_file() and p.suffix != ".conllu")
    if not files:
        print(f"No input files found in: {input_dir}")
        return

    print(f"Found {len(files)} files")
    print(f"Input dir : {input_dir}")
    print(f"Output dir: {output_dir}")

    for i, in_file in enumerate(files, start=1):
        rel = in_file.relative_to(input_dir)
        out_file = (output_dir / rel).with_name(rel.name + ".conllu")
        out_file.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            args.python,
            str(parser_script),
            "--input_txt",
            str(in_file),
            "--output_conllu",
            str(out_file),
        ]
        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])
        if args.use_gpu:
            cmd.append("--use_gpu")

        print(f"[{i}/{len(files)}] Parsing {in_file.name} -> {out_file.name}")
        subprocess.run(cmd, check=True)

    print(f"Done. Wrote CoNLL-U files to: {output_dir}")


if __name__ == "__main__":
    main()
