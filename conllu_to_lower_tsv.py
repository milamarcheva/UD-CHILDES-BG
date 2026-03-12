#!/usr/bin/env python3
"""
Extract sent_id and lowercased sentence text from one or more CoNLL-U files.

Output: TSV with columns `id` and `lowercased_utterance`.
"""

import argparse
from pathlib import Path
import re


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CoNLL-U to TSV (sent_id, lowercased text).")
    p.add_argument(
        "--conllu",
        nargs="+",
        required=True,
        help="Input CoNLL-U file(s).",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output TSV path.",
    )
    return p.parse_args()


NUMBER_RE = re.compile(r"(?:№\s*\d+)|[0-9０-９]+|[0-9]+[×xX*][0-9]+|[0-9]+[–-][0-9]+")
LATIN_RE = re.compile(r"[A-Za-z]")
FORBIDDEN_CHARS = set("…„“_'")


def _is_cyrillic_letter(ch: str) -> bool:
    return "\u0400" <= ch <= "\u04FF"


def token_has_disallowed_chars(token: str) -> bool:
    for ch in token:
        if not _is_cyrillic_letter(ch):
            return True
    return False


def should_drop(sent_id: str, text: str, tokens: list[str]) -> bool:
    if not (sent_id.startswith("book_") or sent_id.startswith("books_")):
        return False
    if NUMBER_RE.search(text):
        return True
    if LATIN_RE.search(text):
        return True
    if any(ch in text for ch in FORBIDDEN_CHARS):
        return True
    for tok in tokens:
        if tok.startswith("по-"):
            body = tok.split("-", 1)[1] if "-" in tok else tok[3:]
            if not body or token_has_disallowed_chars(body):
                return True
            continue
        if tok.startswith("най-"):
            body = tok.split("-", 1)[1] if "-" in tok else tok[4:]
            if not body or token_has_disallowed_chars(body):
                return True
            continue
        if len(tok) > 64:
            return True
        # non-prefix tokens: must be pure Cyrillic letters, no hyphen
        if "-" in tok or token_has_disallowed_chars(tok):
            return True
    return False


def emit_entries(conllu_path: Path):
    sent_id = None
    tokens = []
    with conllu_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("# sent_id ="):
                sent_id = line.split("=", 1)[1].strip()
                tokens = []
            elif not line or line.startswith("#"):
                # sentence boundary or other comment
                if line == "":
                    if sent_id is not None:
                        if tokens:
                            joined = " ".join(tokens)
                            if not should_drop(sent_id, joined, tokens):
                                yield sent_id, joined
                        sent_id = None
                        tokens = []
                continue
            else:
                cols = line.split("\t")
                if len(cols) < 4:
                    continue
                tid = cols[0]
                if "-" in tid or "." in tid:
                    continue  # skip MWT headers and empty nodes
                form = cols[1]
                upos = cols[3]
                if upos == "PUNCT":
                    continue
                tokens.append(form.lower())
        # flush last sentence if file doesn't end with blank line
        if sent_id is not None and tokens:
            joined = " ".join(tokens)
            if not should_drop(sent_id, joined, tokens):
                yield sent_id, joined


def main():
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    entries = []
    for path_str in args.conllu:
        path = Path(path_str)
        if not path.exists():
            raise SystemExit(f"Input not found: {path}")
        entries.extend(list(emit_entries(path)))

    with out_path.open("w", encoding="utf-8") as out:
        out.write("id\tlowercased_utterance\n")
        for sid, txt in entries:
            out.write(f"{sid}\t{txt}\n")

    print(f"Wrote {len(entries)} rows to {out_path}")


if __name__ == "__main__":
    main()
