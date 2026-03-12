#!/usr/bin/env python3
"""
Extract morphtok-segmented sentences from a CoNLL-U file.
Each sentence is one line; tokens use MSEG if present, otherwise FORM.
MSEG separator '+' is replaced with space, e.g., 'живее+ше' -> 'живее ше'.
"""

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract morphtok sentences line-per-line from CoNLL-U.")
    p.add_argument("-i", "--input", required=True, help="Input CoNLL-U file (with MSEG in MISC).")
    p.add_argument("-o", "--output", required=True, help="Output text file (one sentence per line).")
    return p.parse_args()


def parse_misc(misc: str) -> dict:
    if not misc or misc == "_":
        return {}
    out = {}
    for part in misc.split("|"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k] = v
    return out


def main() -> None:
    args = parse_args()
    inp = Path(args.input)
    outp = Path(args.output)

    sentences = []
    current = []

    with inp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if current:
                    sentences.append(" ".join(current))
                current = []
                continue
            if line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) != 10:
                continue
            misc = parse_misc(cols[9])
            if "MSEG" in misc:
                current.append(misc["MSEG"].replace("+", " "))
            else:
                current.append(cols[1])
    # flush last sentence
    if current:
        sentences.append(" ".join(current))

    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text("\n".join(sentences) + "\n", encoding="utf-8")
    print(f"Wrote {len(sentences)} sentences to {outp}")


if __name__ == "__main__":
    main()
