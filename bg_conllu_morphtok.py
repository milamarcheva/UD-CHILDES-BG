#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Conservative Bulgarian suffix segmenter for CoNLL-U.

Reads CoNLL-U from stdin or a file and appends an MSEG annotation
into MISC for tokens where suffix segmentation is clear.

Focus:
- suffixes only
- high precision
- verbs / nouns / adjectives
- uses FORM, LEMMA, UPOS, FEATS

Example:
    python bg_suffix_segmenter.py input.conllu > output.conllu
"""

from __future__ import annotations
import sys
from typing import Dict, List, Optional, Tuple
import argparse


# -----------------------------
# CoNLL-U helpers
# -----------------------------

FIELDS = [
    "ID", "FORM", "LEMMA", "UPOS", "XPOS",
    "FEATS", "HEAD", "DEPREL", "DEPS", "MISC"
]


def parse_feats(feats_str: str) -> Dict[str, str]:
    if not feats_str or feats_str == "_":
        return {}
    feats = {}
    for part in feats_str.split("|"):
        if "=" in part:
            k, v = part.split("=", 1)
            feats[k] = v
    return feats


def parse_misc(misc_str: str) -> Dict[str, str]:
    if not misc_str or misc_str == "_":
        return {}
    misc = {}
    for part in misc_str.split("|"):
        if "=" in part:
            k, v = part.split("=", 1)
            misc[k] = v
        else:
            misc[part] = ""
    return misc


def serialize_misc(misc: Dict[str, str]) -> str:
    if not misc:
        return "_"
    parts = []
    for k, v in misc.items():
        if v == "":
            parts.append(k)
        else:
            parts.append(f"{k}={v}")
    return "|".join(parts)


def is_word_token(tok_id: str) -> bool:
    # skip multiword tokens like 3-4 and empty nodes like 5.1
    return "-" not in tok_id and "." not in tok_id


def parse_conllu_line(line: str) -> Optional[Dict[str, str]]:
    line = line.rstrip("\n")
    if not line or line.startswith("#"):
        return None
    cols = line.split("\t")
    if len(cols) != 10:
        return None
    return dict(zip(FIELDS, cols))


def serialize_token(tok: Dict[str, str]) -> str:
    return "\t".join(tok[f] for f in FIELDS)


# -----------------------------
# Conservative segmentation
# -----------------------------

VOWELS = set("аеиоуъюяАЕИОУЪЮЯ")


def safe_suffix_split(form: str, suffix: str) -> Optional[Tuple[str, str]]:
    """
    Returns (base, suffix) only if form clearly ends in suffix
    and base is non-empty.
    """
    if len(form) <= len(suffix):
        return None
    if not form.endswith(suffix):
        return None
    base = form[:-len(suffix)]
    if not base:
        return None
    return base, suffix


def near_match(a: str, b: str) -> bool:
    """
    Very conservative similarity for form-base vs lemma:
    exact match or one-char difference at the end.
    """
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return a[:-1] == b[:-1]
    if len(a) + 1 == len(b):
        return a == b[:-1]
    if len(b) + 1 == len(a):
        return b == a[:-1]
    return False


def choose_if_lemma_supported(
    form: str,
    lemma: str,
    candidates: List[Tuple[str, str]]
) -> Optional[Tuple[str, str]]:
    """
    Prefer candidates where base == lemma, then near-match to lemma.
    """
    exact = []
    near = []

    for base, suffix in candidates:
        if base == lemma:
            exact.append((base, suffix))
        elif near_match(base, lemma):
            near.append((base, suffix))

    if exact:
        # Prefer longer suffix if multiple exact candidates exist
        exact.sort(key=lambda x: len(x[1]), reverse=True)
        return exact[0]

    if near:
        near.sort(key=lambda x: len(x[1]), reverse=True)
        return near[0]

    return None


def segment_verb(form: str, lemma: str, feats: Dict[str, str]) -> Optional[Tuple[str, str]]:
    """
    Conservative verb suffixes.

    Intended for clear cases like:
    - заравяла -> заравя + ла
    - заравяло -> заравя + ло
    - заравяли -> заравя + ли
    - заравял  -> заравя + л
    - заравяха -> заравя + ха
    - заравяше -> заравя + ше
    """
    candidates: List[Tuple[str, str]] = []

    # Past active l-participle-like endings
    if feats.get("VerbForm") == "Part" or "Gender" in feats or "Number" in feats:
        for suf in ["лите", "лите", "лите", "л", "ла", "ло", "ли"]:
            sp = safe_suffix_split(form, suf)
            if sp:
                candidates.append(sp)

    # Common finite past/imperfect-like endings
    if feats.get("VerbForm") == "Fin" or feats.get("Tense") in {"Past", "Imp"}:
        for suf in ["хме", "хте", "ха", "ше", "х"]:
            sp = safe_suffix_split(form, suf)
            if sp:
                candidates.append(sp)

    # Very conservative: only keep if lemma supports the base
    return choose_if_lemma_supported(form, lemma, candidates)


def segment_noun_adj(form: str, lemma: str, feats: Dict[str, str]) -> Optional[Tuple[str, str]]:
    """
    Conservative nominal/adjectival suffixes.

    Intended for clear cases like:
    - книгата  -> книга + та
    - книгите -> книги + те
    - хубавите -> хубави + те

    This does NOT try to discover all plural/definite combinations.
    It only handles a small safe subset.
    """
    candidates: List[Tuple[str, str]] = []

    definite = feats.get("Definite") == "Def"
    number = feats.get("Number")
    upos_like_adj = feats.get("Degree") is not None  # just a tiny hint if needed

    if definite:
        # Most reliable article-like endings for a conservative first pass
        suffixes = ["та", "те", "ът", "ят"]
        for suf in suffixes:
            sp = safe_suffix_split(form, suf)
            if sp:
                candidates.append(sp)

        # Slightly less strict: common plural definite surface endings,
        # but keep only if base is lemma-supported.
        for suf in ["ите"]:
            sp = safe_suffix_split(form, suf)
            if sp:
                candidates.append(sp)

    # Prefer exact/near lemma support
    chosen = choose_if_lemma_supported(form, lemma, candidates)
    if chosen:
        return chosen

    # Special handling: книги + те vs книга + ите
    # For forms ending in -ите, if lemma + "те" == form and lemma ends in "и"
    # we prefer [lemma] + те
    if definite and form.endswith("ите"):
        if lemma and form == lemma + "те":
            return lemma, "те"

    return None


def segment_token(tok: Dict[str, str]) -> Optional[Tuple[str, str]]:
    form = tok["FORM"]
    lemma = tok["LEMMA"]
    upos = tok["UPOS"]
    feats = parse_feats(tok["FEATS"])

    if not form or form == "_" or not lemma or lemma == "_":
        return None

    # Leave punctuation, symbols, adpositions, etc. untouched
    if upos in {"PUNCT", "SYM", "X", "ADP", "ADV", "CCONJ", "SCONJ", "PART", "INTJ", "NUM", "PROPN"}:
        return None

    if upos == "VERB" or upos == "AUX":
        return segment_verb(form, lemma, feats)

    if upos in {"NOUN", "ADJ", "DET", "PRON"}:
        return segment_noun_adj(form, lemma, feats)

    return None


def add_mseg_to_token(tok: Dict[str, str], seg: Tuple[str, str]) -> Dict[str, str]:
    base, suffix = seg
    misc = parse_misc(tok["MISC"])
    misc["MSEG"] = f"{base}+{suffix}"
    tok["MISC"] = serialize_misc(misc)
    return tok


# -----------------------------
# Main processing
# -----------------------------

def process_lines(lines: List[str]) -> List[str]:
    output = []

    for raw in lines:
        line = raw.rstrip("\n")

        # preserve comments and blank lines
        if not line or line.startswith("#"):
            output.append(line)
            continue

        cols = line.split("\t")
        if len(cols) != 10:
            output.append(line)
            continue

        tok = dict(zip(FIELDS, cols))

        if not is_word_token(tok["ID"]):
            output.append(line)
            continue

        seg = segment_token(tok)
        if seg is not None:
            tok = add_mseg_to_token(tok, seg)

        output.append(serialize_token(tok))

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Add conservative Bulgarian suffix segmentation (MSEG) to CoNLL-U."
    )
    parser.add_argument(
        "-i", "--input", help="Input CoNLL-U file (default: stdin)", default=None
    )
    parser.add_argument(
        "-o", "--output", help="Output CoNLL-U file (default: stdout)", default=None
    )
    args = parser.parse_args()

    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            lines = f.readlines()
    else:
        lines = sys.stdin.readlines()

    out_lines = process_lines(lines)
    out_text = "\n".join(out_lines) + "\n"

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fout:
            fout.write(out_text)
    else:
        sys.stdout.write(out_text)


if __name__ == "__main__":
    main()
