#!/usr/bin/env python3
"""
Filter a CSV to keep only rows whose utterance tokens are all in a vocab.

Usage:
  python filter_utterances_by_vocab.py \
    --input data/dfs/child_utterances.csv \
    --output data/dfs/child_utterances_vocab_filtered.csv \
    --vocab data/vocabs/filtered_ctb_sents_morphtok.vocab \
    --column utt_cleaned
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
import pandas as pd

WORD_RE = re.compile(r"\b\w+\b")


def load_vocab(vocab_path: str | Path) -> set[str]:
    vocab_path = Path(vocab_path)
    with vocab_path.open("r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def utterance_in_vocab(utt: str, vocab: set[str]) -> bool:
    tokens = WORD_RE.findall(utt or "")
    if not tokens:
        return False
    return all(tok in vocab for tok in tokens)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input CSV path.")
    p.add_argument("--output", required=True, help="Output CSV path.")
    p.add_argument("--vocab", required=True, help="Vocab file path (one token per line).")
    p.add_argument(
        "--column",
        default="utt_cleaned",
        help="CSV column to check against vocab (default: utt_cleaned).",
    )
    args = p.parse_args()

    vocab = load_vocab(args.vocab)
    df = pd.read_csv(args.input)

    if args.column not in df.columns:
        raise SystemExit(f"Column '{args.column}' not found in {args.input}.")

    keep_mask = df[args.column].astype(str).apply(lambda s: utterance_in_vocab(s, vocab))
    filtered = df[keep_mask].copy()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(out_path, index=False, encoding="utf-8")

    print(f"Kept {len(filtered):,} / {len(df):,} rows -> {out_path}")


if __name__ == "__main__":
    main()
