#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import contextlib

import stanza
from tqdm import tqdm

# ASCII dependency charts
import deplacy


_DEPLACY_RENDER_MODE = None


def _render_depchart(sentence):
    global _DEPLACY_RENDER_MODE
    if _DEPLACY_RENDER_MODE is None:
        # Probe available render signature once.
        for mode in ("format", "plain", "out", "stdout"):
            try:
                if mode == "format":
                    text = deplacy.render(sentence, format="text")
                elif mode == "plain":
                    text = deplacy.render(sentence)
                elif mode == "out":
                    text = deplacy.render(sentence, out=None)
                else:
                    import io
                    import contextlib

                    buf = io.StringIO()
                    with contextlib.redirect_stdout(buf):
                        deplacy.render(sentence)
                    text = buf.getvalue()
                if text:
                    _DEPLACY_RENDER_MODE = mode
                    return text
            except TypeError:
                continue
        _DEPLACY_RENDER_MODE = "plain"
        return deplacy.render(sentence)

    if _DEPLACY_RENDER_MODE == "format":
        return deplacy.render(sentence, format="text")
    if _DEPLACY_RENDER_MODE == "plain":
        return deplacy.render(sentence)
    if _DEPLACY_RENDER_MODE == "out":
        return deplacy.render(sentence, out=None)

    import io
    import contextlib

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        deplacy.render(sentence)
    return buf.getvalue()


def _to_conll_str(val):
    if val is None:
        return "_"
    # Stanza sometimes uses dicts for FEATS/DEPS
    if isinstance(val, dict):
        if not val:
            return "_"
        parts = []
        for k in sorted(val.keys()):
            v = val[k]
            if isinstance(v, (list, tuple)):
                v = ",".join(map(str, v))
            parts.append(f"{k}={v}")
        return "|".join(parts)
    if isinstance(val, (list, tuple)):
        if not val:
            return "_"
        return "|".join(map(str, val))
    s = str(val)
    return s if s else "_"


def _word_to_conll_line(word):
    wid = getattr(word, "id", None)
    if isinstance(wid, (list, tuple)):
        # Word IDs should be ints; take the first if nested.
        wid = wid[0] if wid else None
    fields = [
        _to_conll_str(wid),
        _to_conll_str(getattr(word, "text", None)),
        _to_conll_str(getattr(word, "lemma", None)),
        _to_conll_str(getattr(word, "upos", None)),
        _to_conll_str(getattr(word, "xpos", None)),
        _to_conll_str(getattr(word, "feats", None)),
        _to_conll_str(getattr(word, "head", None)),
        _to_conll_str(getattr(word, "deprel", None)),
        _to_conll_str(getattr(word, "deps", None)),
        _to_conll_str(getattr(word, "misc", None)),
    ]
    return "\t".join(fields)


def _sentence_to_conll(sentence):
    lines = []
    # Emit multi-word token lines if present
    tokens = getattr(sentence, "tokens", None)
    if tokens:
        for tok in tokens:
            tok_id = getattr(tok, "id", None)
            if isinstance(tok_id, (list, tuple)) and len(tok_id) == 2:
                # MWT line: ID (e.g., 1-2), FORM, rest "_"
                mwt_id = f"{tok_id[0]}-{tok_id[1]}"
                lines.append(
                    "\t".join([mwt_id, _to_conll_str(getattr(tok, "text", None))] + ["_"] * 8)
                )
            for w in getattr(tok, "words", []) or []:
                lines.append(_word_to_conll_line(w))
    else:
        for w in getattr(sentence, "words", []) or []:
            lines.append(_word_to_conll_line(w))
    return "\n".join(lines)


def _doc_to_conll(doc):
    return "\n\n".join(_sentence_to_conll(s) for s in doc.sentences)


def parse_args():
    p = argparse.ArgumentParser(
        description="Parse Bulgarian sentences from CSV column or TXT (one per line) with Stanza."
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input_csv", help="Path to input CSV")
    src.add_argument("--input_txt", help="Path to input TXT (one sentence per line)")
    p.add_argument("--text_column", default=None, help="Column name containing sentences (required with --input_csv)")
    p.add_argument("--output_conllu", required=True, help="Output CoNLL-U file")
    p.add_argument(
        "--output_charts",
        default=None,
        help="Optional: output TXT file with dependency charts",
    )
    p.add_argument("--limit", type=int, default=None, help="Optional: limit number of sentences processed")
    p.add_argument("--use_gpu", action="store_true", help="Use GPU if available")
    return p.parse_args()


def main():
    args = parse_args()

    sentences = []
    if args.input_csv:
        input_csv = Path(args.input_csv)
        if not input_csv.exists():
            print(f"ERROR: input_csv not found: {input_csv}", file=sys.stderr)
            sys.exit(1)
        if not args.text_column:
            print("ERROR: --text_column is required with --input_csv", file=sys.stderr)
            sys.exit(1)

        print("Loading CSV...")
        try:
            import pandas as pd
        except ImportError:
            print("ERROR: pandas is required for --input_csv mode.", file=sys.stderr)
            sys.exit(1)

        df = pd.read_csv(input_csv)
        if args.text_column not in df.columns:
            print(f"ERROR: column not found: {args.text_column}", file=sys.stderr)
            print(f"Available columns: {list(df.columns)}", file=sys.stderr)
            sys.exit(1)
        sentences = df[args.text_column].dropna().astype(str).tolist()
        print(f"Loaded {len(sentences)} sentences from column '{args.text_column}'.")
    else:
        input_txt = Path(args.input_txt)
        if not input_txt.exists():
            print(f"ERROR: input_txt not found: {input_txt}", file=sys.stderr)
            sys.exit(1)
        print("Loading TXT...")
        with input_txt.open("r", encoding="utf-8") as f:
            sentences = [line.rstrip("\n") for line in f]
        print(f"Loaded {len(sentences)} lines from '{input_txt}'.")

    if args.limit is not None:
        sentences = sentences[: args.limit]

    print("Downloading/initializing Stanza Bulgarian models...")
    stanza.download("bg", verbose=False)

    nlp = stanza.Pipeline(
        lang="bg",
        processors="tokenize,pos,lemma,depparse",
        tokenize_pretokenized=True,
        use_gpu=args.use_gpu,
        verbose=False,
    )

    out_conllu = Path(args.output_conllu)
    out_charts = Path(args.output_charts) if args.output_charts else None

    print(f"Writing CoNLL-U to: {out_conllu}")
    if out_charts:
        print(f"Writing dependency charts to: {out_charts}")
    else:
        print("Dependency charts disabled (no --output_charts).")

    charts_ctx = (
        out_charts.open("w", encoding="utf-8") if out_charts else contextlib.nullcontext()
    )
    with out_conllu.open("w", encoding="utf-8") as f_conllu, charts_ctx as f_charts:
        for idx, sent_text in enumerate(tqdm(sentences, desc="Parsing"), start=1):
            sent_text = sent_text.strip()
            if not sent_text:
                continue

            doc = nlp(sent_text)

            # --- CoNLL-U output ---
            f_conllu.write(f"# sent_id = {idx}\n")
            f_conllu.write(f"# text = {sent_text}\n")
            if hasattr(doc, "to_conll"):
                conll_text = doc.to_conll()
            else:
                # Older stanza versions don't expose Document.to_conll()
                conll_text = _doc_to_conll(doc)
            f_conllu.write(conll_text.strip() + "\n\n")

            if f_charts:
                # --- ASCII dependency chart output ---
                f_charts.write("=" * 80 + "\n")
                f_charts.write(f"sent_id = {idx}\n")
                f_charts.write(f"text = {sent_text}\n\n")

                # deplacy expects a stanza Document; it will render each sentence
                # In case stanza splits into multiple sentences, render them all.
                for s_i, s in enumerate(doc.sentences, start=1):
                    f_charts.write(f"[stanza_sentence_{s_i}]\n")
                    # Render returns a string; write it.
                    f_charts.write(_render_depchart(s))
                    f_charts.write("\n\n")

    print("Done.")


if __name__ == "__main__":
    main()
