#!/usr/bin/env python3
"""
Filter LabLing CSVs and add a cleaned Cyrillic utterance column.

Example:
  python filter_labling_cds.py \
    --input_csv data/LabLing_longitudinal_all_utterances.csv \
    --output_csv data/LabLing_longitudinal_all_utterances_cds_cleaned.csv \
    -cds
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


# --- Cleaning rules (mirrors aggregate_chat_cds.py) ---
CHAT_TIMECODE_RE = re.compile(r"\u0015\d+_\d+\u0015")
PAREN_PUNCT_RE = re.compile(r"\(\s*[.?!,-]+\s*\)")
ANGLE_BRACKET_RE = re.compile(r"<[^>]*>")
SQUARE_BRACKET_RE = re.compile(r"\[[^\]]*\]")
SQBRACKET_REPL_RE = re.compile(r"\[\s*:\s*([^\]]+?)\s*\]")
BRACKET_REPAIRS = (
    ("[//]", " "),
    ("[/]", " "),
)
NON_ALPHA_KEEP_PUNCT_ASCII = re.compile(r"[^A-Za-z\s.,;:!?\"'-]")
NON_ALPHA_KEEP_PUNCT_UNI = re.compile(r"[^\w\s.,;:!?\"'-]", flags=re.UNICODE)

ALLOW_UNICODE_ALPHA = True

VARIANT_TO_STANDARD = {
    # Cliticizations
}


def clean_utterance_text(text: str) -> str:
    if not text:
        return ""

    cleaned = text
    cleaned = CHAT_TIMECODE_RE.sub(" ", cleaned)
    cleaned = PAREN_PUNCT_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\b0\S+\b", " ", cleaned)
    cleaned = re.sub(r"&\S*", " ", cleaned)

    cleaned = re.sub(r"@[\w-]+", "", cleaned)
    cleaned = cleaned.replace("+", "")

    cleaned = ANGLE_BRACKET_RE.sub(" ", cleaned)
    for bad, repl in BRACKET_REPAIRS:
        cleaned = cleaned.replace(bad, repl)
    cleaned = SQUARE_BRACKET_RE.sub(" ", cleaned)

    cleaned = cleaned.replace("(", "").replace(")", "")

    cleaned = re.sub(r"(?<=\w):(?=\w)", "", cleaned)
    cleaned = re.sub(r":(?=[?.!,;])", "", cleaned)

    cleaned = re.sub(r"\d+", " ", cleaned)
    regex = NON_ALPHA_KEEP_PUNCT_UNI if ALLOW_UNICODE_ALPHA else NON_ALPHA_KEEP_PUNCT_ASCII
    cleaned = regex.sub(" ", cleaned)

    # Separate basic punctuation from words (tokenized punctuation).
    cleaned = re.sub(r"([.,;:!?])", r" \1 ", cleaned)

    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def replace_variants(text: str) -> str:
    if not text:
        return text
    tokens = text.split()
    replaced = [VARIANT_TO_STANDARD.get(tok, tok) for tok in tokens]
    return " ".join(replaced)


def replacement_sqbrackets(text: str) -> str:
    """
    Replace CHAT square-bracket repairs like:
      Де [: две]  ->  две
    If the bracket contains multiple words, replace that many preceding words.
    """
    if not text:
        return text

    tokens = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    out_tokens: list[str] = []
    t_idx = 0

    for m in SQBRACKET_REPL_RE.finditer(text):
        # Add tokens before bracket span
        while t_idx < len(tokens) and tokens[t_idx][2] <= m.start():
            out_tokens.append(tokens[t_idx][0])
            t_idx += 1
        # Skip tokens inside bracket span
        while t_idx < len(tokens) and tokens[t_idx][1] < m.end():
            t_idx += 1

        repl_words = m.group(1).strip().split()
        n = len(repl_words)
        if n > 0:
            if n <= len(out_tokens):
                out_tokens = out_tokens[:-n] + repl_words
            else:
                out_tokens = repl_words[:]

    # Append remaining tokens after last bracket
    while t_idx < len(tokens):
        out_tokens.append(tokens[t_idx][0])
        t_idx += 1

    return " ".join(out_tokens)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Filter LabLing CSV rows and add UtterancesCyrillicCleaned."
    )
    p.add_argument("--input_csv", required=True, help="Input CSV path.")
    p.add_argument("--output_csv", required=True, help="Output CSV path.")
    p.add_argument(
        "--text_column",
        default="UtterancesCyrillic",
        help="Column to clean (default: UtterancesCyrillic).",
    )
    p.add_argument(
        "--clean_column",
        default="UtterancesCyrillicCleaned",
        help="New column name for cleaned text.",
    )
    p.add_argument(
        "--clean_norm_column",
        default="UtterancesCyrillicCleanedNormalised",
        help="New column name for cleaned + [:] normalized text.",
    )
    group = p.add_mutually_exclusive_group()
    group.add_argument(
        "-cds",
        "--cds",
        action="store_true",
        help="Keep only CDS rows (Name != Participant).",
    )
    group.add_argument(
        "-childspeech",
        "--childspeech",
        "-childspeeh",
        "--childspeeh",
        action="store_true",
        help="Keep only child speech rows (Name == Participant).",
    )
    p.add_argument(
        "--ascii",
        action="store_true",
        help="Use ASCII-only cleaning (strip non-ASCII letters).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    global ALLOW_UNICODE_ALPHA
    ALLOW_UNICODE_ALPHA = not args.ascii

    df = pd.read_csv(args.input_csv)

    if args.cds or args.childspeech:
        for col in ("Name", "Participant"):
            if col not in df.columns:
                raise SystemExit(f"Required column '{col}' not found in {args.input_csv}.")
        name = df["Name"].fillna("").astype(str).str.strip()
        participant = df["Participant"].fillna("").astype(str).str.strip()
        if args.cds:
            df = df[name != participant].copy()
        else:
            df = df[name == participant].copy()

    if args.text_column not in df.columns:
        raise SystemExit(f"Column '{args.text_column}' not found in {args.input_csv}.")

    df[args.clean_column] = (
        df[args.text_column]
        .fillna("")
        .astype(str)
        .apply(lambda s: replace_variants(clean_utterance_text(s)))
    )
    df[args.clean_norm_column] = (
        df[args.text_column]
        .fillna("")
        .astype(str)
        .apply(lambda s: replace_variants(clean_utterance_text(replacement_sqbrackets(s))))
    )

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"Wrote {len(df):,} rows -> {out_path}")


if __name__ == "__main__":
    main()
