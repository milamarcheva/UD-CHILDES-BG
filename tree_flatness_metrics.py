#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute flatness / hierarchy metrics for constituency trees.

Input:
- a folder containing .txt files
- each .txt file should contain one bracketed parse per line

For each file, the script computes:
1. mean nonterminal branching factor
2. proportion of nonterminals with 3+ children
3. proportion of nonterminals with 4+ children
4. mean tree depth
5. mean leaf depth
6. a custom flatness score:
      sum(max(children - 2, 0)) / num_nonterminals

It also prints an overall score across all files.

Usage:
    python tree_flatness_metrics.py /path/to/folder

Optional:
    python tree_flatness_metrics.py /path/to/folder --output_csv metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Tuple

from nltk import Tree


@dataclass
class FileMetrics:
    filename: str
    num_trees: int
    num_failed: int
    num_nonterminals: int
    mean_branching_factor: float
    prop_nodes_3plus: float
    prop_nodes_4plus: float
    mean_tree_depth: float
    mean_leaf_depth: float
    flatness_score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder", type=Path, help="Folder with .txt parse files")
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help="Optional path to save per-file metrics as CSV",
    )
    return parser.parse_args()


def is_preterminal(t: Tree) -> bool:
    return isinstance(t, Tree) and len(t) == 1 and isinstance(t[0], str)


def is_nonterminal(t: Tree) -> bool:
    return isinstance(t, Tree) and not is_preterminal(t)


def iter_nonterminals(t: Tree) -> Iterable[Tree]:
    if is_nonterminal(t):
        yield t
        for child in t:
            if isinstance(child, Tree):
                yield from iter_nonterminals(child)


def tree_depth_edges(t: Tree) -> int:
    """
    Max root-to-leaf depth in edges.
    Preterminal with one terminal child has depth 1.
    """
    if is_preterminal(t):
        return 1
    child_depths = []
    for child in t:
        if isinstance(child, Tree):
            child_depths.append(tree_depth_edges(child))
        else:
            child_depths.append(0)
    return 1 + max(child_depths, default=0)


def leaf_depths_edges(t: Tree, current_depth: int = 0) -> List[int]:
    """
    Return depths of all terminal leaves in edges from root.
    """
    depths: List[int] = []
    for child in t:
        if isinstance(child, Tree):
            depths.extend(leaf_depths_edges(child, current_depth + 1))
        else:
            depths.append(current_depth + 1)
    return depths


def branching_factor(node: Tree) -> int:
    """
    Count all children, including preterminals and nonterminals.
    """
    return len(node)


def compute_metrics_for_trees(trees: List[Tree], filename: str, num_failed: int) -> FileMetrics:
    num_nonterminals = 0
    branching_sum = 0
    nodes_3plus = 0
    nodes_4plus = 0
    flatness_sum = 0
    tree_depth_sum = 0
    leaf_depth_sum = 0
    total_leaf_count = 0

    for tree in trees:
        nts = list(iter_nonterminals(tree))
        num_nonterminals += len(nts)

        for node in nts:
            bf = branching_factor(node)
            branching_sum += bf
            if bf >= 3:
                nodes_3plus += 1
            if bf >= 4:
                nodes_4plus += 1
            flatness_sum += max(bf - 2, 0)

        tree_depth_sum += tree_depth_edges(tree)
        ldepths = leaf_depths_edges(tree)
        leaf_depth_sum += sum(ldepths)
        total_leaf_count += len(ldepths)

    def safe_div(num: float, den: float) -> float:
        return num / den if den else float("nan")

    return FileMetrics(
        filename=filename,
        num_trees=len(trees),
        num_failed=num_failed,
        num_nonterminals=num_nonterminals,
        mean_branching_factor=safe_div(branching_sum, num_nonterminals),
        prop_nodes_3plus=safe_div(nodes_3plus, num_nonterminals),
        prop_nodes_4plus=safe_div(nodes_4plus, num_nonterminals),
        mean_tree_depth=safe_div(tree_depth_sum, len(trees)),
        mean_leaf_depth=safe_div(leaf_depth_sum, total_leaf_count),
        flatness_score=safe_div(flatness_sum, num_nonterminals),
    )


def read_trees_from_file(path: Path) -> Tuple[List[Tree], int]:
    trees: List[Tree] = []
    failed = 0

    with path.open("r", encoding="utf-8") as f:
        for line_num, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                trees.append(Tree.fromstring(line))
            except Exception:
                failed += 1

    return trees, failed


def format_float(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "NA"
    return f"{x:.4f}"


def print_table(results: List[FileMetrics], overall: FileMetrics) -> None:
    headers = [
        "file",
        "trees",
        "failed",
        "mean_branch",
        "prop_3plus",
        "prop_4plus",
        "mean_depth",
        "mean_leaf_depth",
        "flatness",
    ]
    rows = []
    for r in results:
        rows.append([
            r.filename,
            str(r.num_trees),
            str(r.num_failed),
            format_float(r.mean_branching_factor),
            format_float(r.prop_nodes_3plus),
            format_float(r.prop_nodes_4plus),
            format_float(r.mean_tree_depth),
            format_float(r.mean_leaf_depth),
            format_float(r.flatness_score),
        ])

    rows.append([
        "OVERALL",
        str(overall.num_trees),
        str(overall.num_failed),
        format_float(overall.mean_branching_factor),
        format_float(overall.prop_nodes_3plus),
        format_float(overall.prop_nodes_4plus),
        format_float(overall.mean_tree_depth),
        format_float(overall.mean_leaf_depth),
        format_float(overall.flatness_score),
    ])

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def print_row(row: List[str]) -> None:
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))

    print_row(headers)
    print_row(["-" * w for w in widths])
    for row in rows:
        print_row(row)


def save_csv(results: List[FileMetrics], overall: FileMetrics, out_path: Path) -> None:
    fieldnames = list(asdict(overall).keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
        overall_row = asdict(overall)
        overall_row["filename"] = "OVERALL"
        writer.writerow(overall_row)


def combine_all_trees(results: List[Tuple[Path, List[Tree], int]]) -> FileMetrics:
    all_trees: List[Tree] = []
    total_failed = 0
    for _, trees, failed in results:
        all_trees.extend(trees)
        total_failed += failed
    return compute_metrics_for_trees(all_trees, "OVERALL", total_failed)


def main() -> int:
    print("Tree flatness metrics started: ")
    args = parse_args()
    input_folder: Path = args.input_folder

    if not input_folder.exists() or not input_folder.is_dir():
        raise SystemExit(f"Input folder does not exist or is not a directory: {input_folder}")

    txt_files = sorted(input_folder.glob("*.txt"))
    if not txt_files:
        raise SystemExit(f"No .txt files found in {input_folder}")

    parsed_results: List[Tuple[Path, List[Tree], int]] = []
    file_metrics: List[FileMetrics] = []

    for path in txt_files:
        trees, failed = read_trees_from_file(path)
        parsed_results.append((path, trees, failed))
        file_metrics.append(compute_metrics_for_trees(trees, path.name, failed))

    overall = combine_all_trees(parsed_results)
    print_table(file_metrics, overall)

    if args.output_csv is not None:
        save_csv(file_metrics, overall, args.output_csv)
        print(f"\nSaved CSV to: {args.output_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())