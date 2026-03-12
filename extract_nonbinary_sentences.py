#!/usr/bin/env python3
"""
Extract sentences whose constituency parse contains at least one non-binary rule.

Input can be:
- A directory containing parse files (e.g. flat-POS/*.txt), or
- A single parse file.

Assumes one PTB-style tree per line.
Writes one sentence (tree yield) per line to the output file.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Directory or file with PTB trees (one tree per line).",
    )
    p.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output file path (one matching sentence per line).",
    )
    p.add_argument(
        "--lowercase",
        action="store_true",
        help="Lowercase sentence tokens in the output.",
    )
    return p.parse_args()


def iter_tree_files(path: Path):
    if path.is_file():
        yield path
        return
    if not path.is_dir():
        raise SystemExit(f"Input path does not exist: {path}")
    for f in sorted(path.glob("*.txt")):
        if f.name.endswith(".tokens.txt"):
            continue
        if f.name.endswith(".tokens"):
            continue
        yield f


def has_nonbinary_rule(tree) -> bool:
    from nltk import Tree  # type: ignore

    for subtree in tree.subtrees():
        if not isinstance(subtree, Tree):
            continue
        # Non-binary production: more than 2 children on RHS.
        if len(subtree) > 2:
            return True
    return False


def main():
    args = parse_args()
    try:
        from nltk import Tree  # type: ignore
    except ImportError as e:
        raise SystemExit("nltk is required. Install with: pip install nltk") from e

    tree_files = list(iter_tree_files(args.input))
    if not tree_files:
        raise SystemExit(f"No parse files found under: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    total_lines = 0
    kept_lines = 0
    bad_lines = 0

    with args.output.open("w", encoding="utf-8") as out:
        for path in tree_files:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    total_lines += 1
                    try:
                        tree = Tree.fromstring(line)
                    except Exception:
                        bad_lines += 1
                        continue
                    if has_nonbinary_rule(tree):
                        toks = tree.leaves()
                        if args.lowercase:
                            toks = [t.lower() for t in toks]
                        out.write(" ".join(toks) + "\n")
                        kept_lines += 1

    print(f"Read {total_lines} trees from {len(tree_files)} file(s).")
    print(f"Wrote {kept_lines} sentences with at least one non-binary rule to: {args.output}")
    if bad_lines:
        print(f"Skipped {bad_lines} unparsable lines.")


if __name__ == "__main__":
    main()
