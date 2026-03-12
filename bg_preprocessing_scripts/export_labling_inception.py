#!/usr/bin/env python3
"""
Export LabLing sentences into per-child CS/CDS files for INCEpTION.

Rules:
- CS:  Name == Participant
- CDS: Name != Participant

Each file contains one sentence per line from UtterancesCyrillicCleanedNormalised.
Output filenames follow:
- [childname]_cs
- [childname]_cds
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

PUNCT_RE = re.compile(r"([.,;:!?])")
# Quote characters commonly seen in CHAT exports / copied text
QUOTE_RE = re.compile(r'["„“«»]')
ATTACHED_PUNCT_RE = re.compile(r"([^\W\d_])[.,;:!?]|[.,;:!?]([^\W\d_])", flags=re.UNICODE)
START_PUNCT_RE = re.compile(r"^[\s.,;:!?]+")


def _safe_child_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        return "unknown"
    # Keep filesystem-safe names while preserving letters/digits/underscore/dot/dash.
    cleaned = re.sub(r"[^\w.-]+", "_", name, flags=re.UNICODE)
    cleaned = cleaned.strip("_.")
    return cleaned or "unknown"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Split LabLing CSV into per-child CS/CDS sentence files."
    )
    p.add_argument(
        "--input_csv",
        default="/Users/milamarcheva/Desktop/morphemic_tokenisation/data/labling_dfs/LabLing_sample_stratified_corrected.csv",
        help="Input CSV path.",
    )
    p.add_argument(
        "--output_dir",
        default=None,
        help="Output directory (default: <input_csv_dir>/LabLing_inception).",
    )
    p.add_argument("--name_col", default="Name", help="Child name column.")
    p.add_argument("--participant_col", default="Participant", help="Participant column.")
    p.add_argument(
        "--text_col",
        default="UtterancesCyrillicCleanedNormalised",
        help="Sentence text column.",
    )
    p.add_argument(
        "--strip_quotes",
        action="store_true",
        help='Remove quote characters ["„“«»] from output lines.',
    )
    p.add_argument(
        "--fail_on_quotes",
        action="store_true",
        help="Exit with non-zero status if output still contains quotes.",
    )
    return p.parse_args()


def _capitalize_first_letter(text: str) -> str:
    chars = list(text)
    for i, ch in enumerate(chars):
        if ch.isalpha():
            if ch.islower():
                chars[i] = ch.upper()
            break
    return "".join(chars)


def _normalize_sentence(text: str, strip_quotes: bool) -> str:
    sent = (text or "").strip()
    if strip_quotes:
        sent = QUOTE_RE.sub("", sent)
    # Tokenize sentence punctuation into standalone tokens.
    sent = PUNCT_RE.sub(r" \1 ", sent)
    # Ensure no line starts with punctuation tokens.
    sent = START_PUNCT_RE.sub("", sent)
    # Collapse repeated whitespace and trim.
    sent = re.sub(r"\s{2,}", " ", sent).strip()
    # Ensure first alphabetic character is capitalized.
    sent = _capitalize_first_letter(sent)
    return sent


def _starts_with_lowercase_alpha(text: str) -> bool:
    for ch in text:
        if ch.isalpha():
            return ch.islower()
    return False


def _collect_groups(
    df: pd.DataFrame,
    name_col: str,
    participant_col: str,
    text_col: str,
    strip_quotes: bool,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    cs_map: dict[str, list[str]] = defaultdict(list)
    cds_map: dict[str, list[str]] = defaultdict(list)

    names = df[name_col].fillna("").astype(str)
    participants = df[participant_col].fillna("").astype(str)
    texts = df[text_col].fillna("").astype(str)

    for name, participant, text in zip(names, participants, texts):
        sent = _normalize_sentence(text, strip_quotes=strip_quotes)
        if not sent:
            continue
        child = _safe_child_name(name)
        if name.strip() == participant.strip():
            cs_map[child].append(sent)
        else:
            cds_map[child].append(sent)
    return cs_map, cds_map


def _write_grouped(out_dir: Path, grouped: dict[str, list[str]], suffix: str) -> int:
    n_files = 0
    for child in sorted(grouped.keys()):
        out_path = out_dir / f"{child}_{suffix}"
        with out_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(grouped[child]))
            f.write("\n")
        n_files += 1
    return n_files


def _collect_quality_stats(grouped: Dict[str, List[str]]) -> Dict[str, int]:
    stats = {
        "lines": 0,
        "double_spaces": 0,
        "lowercase_start": 0,
        "attached_punct": 0,
        "starts_with_punct": 0,
        "lines_with_quotes": 0,
    }
    for lines in grouped.values():
        for line in lines:
            stats["lines"] += 1
            if "  " in line:
                stats["double_spaces"] += 1
            if _starts_with_lowercase_alpha(line):
                stats["lowercase_start"] += 1
            if ATTACHED_PUNCT_RE.search(line):
                stats["attached_punct"] += 1
            if re.match(r"^[.,;:!?]", line):
                stats["starts_with_punct"] += 1
            if QUOTE_RE.search(line):
                stats["lines_with_quotes"] += 1
    return stats


def main() -> None:
    args = parse_args()

    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise SystemExit(f"Input CSV not found: {input_csv}")

    out_dir = Path(args.output_dir) if args.output_dir else input_csv.parent / "LabLing_inception"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    for col in (args.name_col, args.participant_col, args.text_col):
        if col not in df.columns:
            raise SystemExit(f"Missing required column '{col}' in {input_csv}")

    cs_map, cds_map = _collect_groups(
        df,
        args.name_col,
        args.participant_col,
        args.text_col,
        strip_quotes=bool(args.strip_quotes),
    )
    n_cs = _write_grouped(out_dir, cs_map, "cs")
    n_cds = _write_grouped(out_dir, cds_map, "cds")

    cs_stats = _collect_quality_stats(cs_map)
    cds_stats = _collect_quality_stats(cds_map)
    total_lines = cs_stats["lines"] + cds_stats["lines"]
    total_double = cs_stats["double_spaces"] + cds_stats["double_spaces"]
    total_lower = cs_stats["lowercase_start"] + cds_stats["lowercase_start"]
    total_attached = cs_stats["attached_punct"] + cds_stats["attached_punct"]
    total_start_punct = cs_stats["starts_with_punct"] + cds_stats["starts_with_punct"]
    total_quotes = cs_stats["lines_with_quotes"] + cds_stats["lines_with_quotes"]

    print(f"Wrote CS files: {n_cs}")
    print(f"Wrote CDS files: {n_cds}")
    print(f"Total lines written: {total_lines}")
    print(f"Check double spaces: {total_double}")
    print(f"Check lowercase start: {total_lower}")
    print(f"Check attached punctuation: {total_attached}")
    print(f"Check starts with punctuation: {total_start_punct}")
    print(f"Check lines with quotes: {total_quotes}")
    if args.strip_quotes:
        print("Quote handling: stripped")
    else:
        print("Quote handling: kept")
    print(f"Output directory: {out_dir}")

    if args.fail_on_quotes and total_quotes > 0:
        raise SystemExit("Quotes detected in output and --fail_on_quotes was set.")


if __name__ == "__main__":
    main()
