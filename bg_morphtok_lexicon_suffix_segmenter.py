#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import sys
from collections import defaultdict

# -----------------------------
# Config: small, conservative suffix lists
# -----------------------------

SUFFIXES_BY_POS = {
    "NOUN": [
        "ите", "овете", "евете", "ът", "ят", "та", "то", "те",
        "ове", "еве", "а", "я", "и", "е", "о", "ю"
    ],
    "ADJ": [
        "ите", "ият", "ият", "ата", "ото", "ите",
        "ят", "та", "то", "те",
        "а", "о", "и", "я", "ът"
    ],
    "VERB": [
        # very conservative inflectional endings only
        "хме", "хте", "аха", "яха", "еше", "еше", "иха",
        "ха", "ше", "ла", "ло", "ли", "л",
        "м", "ш", "ме", "те", "т"
    ],
}

MIN_STEM_LEN = 3
MAX_SUFFIX_LEN = 5

# -----------------------------
# Parsing
# -----------------------------

LEX_RE = re.compile(r'^\s*[^ ]+\s+[^ ]+\s+([A-Z]+)\s+-->\s+(.+?)\s*$')

def parse_lexicon(path):
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            m = LEX_RE.match(line)
            if not m:
                continue
            pos = m.group(1)
            word = m.group(2).strip()
            entries.append((pos, word))
    return entries

# -----------------------------
# Segmenter
# -----------------------------

def build_pos_lexicons(entries):
    pos_to_words = defaultdict(set)
    for pos, word in entries:
        pos_to_words[pos].add(word)
    return pos_to_words

def try_segment(word, pos, pos_to_words):
    """
    Return (stem, suffix) or None.
    Only segment if stem exists in same POS lexicon.
    Prefer longest suffix.
    """
    suffixes = SUFFIXES_BY_POS.get(pos, [])
    lex = pos_to_words[pos]

    candidates = []

    for suf in sorted(suffixes, key=len, reverse=True):
        if len(suf) > MAX_SUFFIX_LEN:
            continue
        if not word.endswith(suf):
            continue
        if len(word) <= len(suf):
            continue

        stem = word[:-len(suf)]
        if len(stem) < MIN_STEM_LEN:
            continue

        # strict version: base must exist in same POS lexicon
        if stem in lex:
            candidates.append((stem, suf))

        # small extra heuristic:
        # allow plural/article type split if stem+short ending exists
        # e.g. известните -> известни + те
        elif pos in {"NOUN", "ADJ"}:
            for alt in ["а", "я", "и", "о", "е"]:
                if stem + alt in lex:
                    candidates.append((stem + alt, word[len(stem + alt):]))
                    break

    if not candidates:
        return None

    # choose longest stem, then longest suffix
    candidates.sort(key=lambda x: (len(x[0]), len(x[1])), reverse=True)
    stem, suffix = candidates[0]

    if not suffix:
        return None

    return stem, suffix

# -----------------------------
# Main
# -----------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python bg_lexicon_suffix_segmenter.py bg_lexicon_filtered.txt [output.tsv]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "segmented_lexicon.tsv"

    entries = parse_lexicon(input_path)
    pos_to_words = build_pos_lexicons(entries)

    original_vocab = set()
    segmented_vocab = set()

    rows = []

    for pos, word in entries:
        original_vocab.add(word)

        seg = try_segment(word, pos, pos_to_words)
        if seg is None:
            segmented = word
        else:
            stem, suffix = seg
            segmented = f"{stem}+{suffix}"

        segmented_vocab.add(segmented)
        rows.append((pos, word, segmented))

    with open(output_path, "w", encoding="utf-8") as out:
        out.write("POS\tWORD\tSEGMENTED\n")
        for pos, word, segmented in rows:
            out.write(f"{pos}\t{word}\t{segmented}\n")

    print(f"Entries: {len(entries)}")
    print(f"Original word types: {len(original_vocab)}")
    print(f"Segmented word types: {len(segmented_vocab)}")
    print(f"Wrote: {output_path}")

if __name__ == "__main__":
    main()