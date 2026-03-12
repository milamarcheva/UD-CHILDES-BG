#!/usr/bin/env python3
"""Annotator agreement metrics for UD-style CoNLL-U files.

Given a project folder exported from INCEpTION (the directory that contains
`annotation/`), this script computes pairwise agreement between annotators over a
specified sentence range for one or more documents.

Metrics per annotator pair:
- UPOS accuracy
- UPOS Cohen's kappa
- Lemma accuracy
- UAS (unlabeled attachment score)
- LAS (labeled attachment score)
- F1/precision/recall for the five most frequent relations across the pair

Usage example (runs on the current project export by default):
    python eval_agreement.py \
        --doc ALE_cds.conllu:mila,yasena:1-90 \
        --doc BOG_cds.conllu:mila,yoana:1-148

Paths are resolved relative to the project export root unless absolute. Each
`--doc` flag takes `DOC_DIR:ann1,ann2[,ann3...]:START-END` where `DOC_DIR` is the
folder under `annotation/` containing per-annotator `.conllu` files.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import pathlib
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple


class Token(NamedTuple):
    tid: int
    form: str
    lemma: str
    upos: str
    head: Optional[int]
    deprel: str


class DocSpec(NamedTuple):
    doc_dir: str
    annotators: List[str]
    start: int
    end: int


def parse_conllu(path: pathlib.Path) -> List[List[Token]]:
    sentences: List[List[Token]] = []
    current: List[Token] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                if current:
                    sentences.append(current)
                    current = []
                continue
            if line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 8:
                raise ValueError(f"Unexpected line format in {path}: {line}")
            token_id = parts[0]
            if "-" in token_id or "." in token_id:
                # Skip multi-word tokens or empty nodes; they should not be
                # compared for attachment/labels.
                continue
            head_val = parts[6]
            try:
                head = int(head_val)
            except ValueError:
                head = None
            token = Token(
                tid=int(token_id),
                form=parts[1],
                lemma=parts[2],
                upos=parts[3],
                head=head,
                deprel=parts[7],
            )
            current.append(token)
    if current:
        sentences.append(current)
    return sentences


def slice_sentences(sentences: Sequence[List[Token]], start: int, end: int) -> List[List[Token]]:
    if start < 1 or end < start:
        raise ValueError(f"Invalid range {start}-{end}")
    # sent_id comments are 1-based; mirror that.
    return list(sentences[start - 1 : end])


def ensure_alignment(sents_a: Sequence[List[Token]], sents_b: Sequence[List[Token]], label: str) -> None:
    if len(sents_a) != len(sents_b):
        raise ValueError(f"Sentence count mismatch for {label}: {len(sents_a)} vs {len(sents_b)}")
    for idx, (sa, sb) in enumerate(zip(sents_a, sents_b), start=1):
        if len(sa) != len(sb):
            raise ValueError(
                f"Token count mismatch in sentence {idx} for {label}: {len(sa)} vs {len(sb)}"
            )
        for ta, tb in zip(sa, sb):
            if ta.tid != tb.tid:
                raise ValueError(
                    f"Token ID mismatch in sentence {idx} for {label}: {ta.tid} vs {tb.tid}"
                )


def cohen_kappa(labels_a: Sequence[str], labels_b: Sequence[str]) -> float:
    if len(labels_a) != len(labels_b):
        raise ValueError("Label sequences must be the same length for kappa")
    n = len(labels_a)
    if n == 0:
        return 0.0
    counts_a = collections.Counter(labels_a)
    counts_b = collections.Counter(labels_b)
    p_o = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    p_e = sum((counts_a[l] / n) * (counts_b[l] / n) for l in set(counts_a) | set(counts_b))
    denom = 1 - p_e
    if denom == 0:
        return 1.0 if p_o == 1 else 0.0
    return (p_o - p_e) / denom


def relation_f1(tokens_a: Sequence[Token], tokens_b: Sequence[Token], top_k: int = 10):
    labels = [t.deprel for t in tokens_a] + [t.deprel for t in tokens_b]
    most_common = [rel for rel, _ in collections.Counter(labels).most_common(top_k)]
    results = []
    for rel in most_common:
        count_a = sum(1 for t in tokens_a if t.deprel == rel)
        count_b = sum(1 for t in tokens_b if t.deprel == rel)
        matches = sum(
            1 for ta, tb in zip(tokens_a, tokens_b) if ta.deprel == tb.deprel == rel
        )
        precision = matches / count_a if count_a else 0.0
        recall = matches / count_b if count_b else 0.0
        denom = count_a + count_b
        f1 = 2 * matches / denom if denom else 0.0
        results.append(
            {
                "relation": rel,
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "matches": matches,
                "count_a": count_a,
                "count_b": count_b,
            }
        )
    return results


def compute_metrics(
    sents_a: Sequence[List[Token]], sents_b: Sequence[List[Token]], exclude_upos: Optional[set[str]] = None
):
    exclude_upos = exclude_upos or set()
    ensure_alignment(sents_a, sents_b, label="pair")
    flat_a = [tok for sent in sents_a for tok in sent]
    flat_b = [tok for sent in sents_b for tok in sent]

    filtered_pairs = [
        (a, b)
        for a, b in zip(flat_a, flat_b)
        if not (a.upos in exclude_upos or b.upos in exclude_upos)
    ]
    total = len(filtered_pairs)
    if total == 0:
        return {
            "tokens": 0,
            "upos_acc": 0.0,
            "upos_kappa": 0.0,
            "lemma_acc": 0.0,
            "uas": 0.0,
            "las": 0.0,
            "relations": [],
            "tokens_a": [],
            "tokens_b": [],
        }

    tokens_a = [a for a, _ in filtered_pairs]
    tokens_b = [b for _, b in filtered_pairs]

    upos_matches = sum(1 for a, b in filtered_pairs if a.upos == b.upos)
    lemma_matches = sum(1 for a, b in filtered_pairs if a.lemma == b.lemma)
    head_matches = sum(1 for a, b in filtered_pairs if a.head == b.head and a.head is not None and b.head is not None)
    head_label_matches = sum(
        1
        for a, b in filtered_pairs
        if a.head is not None and b.head is not None and a.head == b.head and a.deprel == b.deprel
    )
    return {
        "tokens": total,
        "upos_acc": upos_matches / total,
        "upos_kappa": cohen_kappa([t.upos for t in tokens_a], [t.upos for t in tokens_b]),
        "lemma_acc": lemma_matches / total,
        "uas": head_matches / total,
        "las": head_label_matches / total,
        "relations": relation_f1(tokens_a, tokens_b),
        "tokens_a": tokens_a,
        "tokens_b": tokens_b,
    }


def load_spec(annotation_root: pathlib.Path, spec: DocSpec) -> Dict[str, List[List[Token]]]:
    data = {}
    for annot in spec.annotators:
        path = (annotation_root / spec.doc_dir / f"{annot}.conllu").resolve()
        if not path.exists():
            raise FileNotFoundError(f"Missing file for annotator {annot}: {path}")
        data[annot] = parse_conllu(path)
    return data


def parse_doc_arg(arg: str) -> DocSpec:
    try:
        doc_part, annot_part, span_part = arg.split(":", 2)
        annotators = annot_part.split(",")
        start_str, end_str = span_part.split("-", 1)
        return DocSpec(doc_dir=doc_part, annotators=annotators, start=int(start_str), end=int(end_str))
    except Exception as exc:  # noqa: BLE001 - user input parsing
        raise argparse.ArgumentTypeError(
            "--doc expects DOC_DIR:ann1,ann2[,ann3...]:START-END"
        ) from exc


def main():
    default_root = pathlib.Path(__file__).parent / "inception_annotations" / "bg_childes_tb_project_2026-02-27_1021"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=pathlib.Path,
        default=default_root,
        help="Path to INCEpTION export root (the directory containing annotation/)",
    )
    parser.add_argument(
        "--doc",
        action="append",
        required=True,
        help="DOC_DIR:ann1,ann2[,ann3...]:START-END (relative to annotation/)",
        dest="doc_specs",
    )
    args = parser.parse_args()

    annotation_root = args.project_root / "annotation"
    if not annotation_root.is_dir():
        raise SystemExit(f"Annotation directory not found: {annotation_root}")

    specs = [parse_doc_arg(d) for d in args.doc_specs]

    # overall accumulators (punct excluded)
    all_tokens_a: List[Token] = []
    all_tokens_b: List[Token] = []

    for spec in specs:
        annot_data = load_spec(annotation_root, spec)
        sentence_slices = {
            annot: slice_sentences(sents, spec.start, spec.end) for annot, sents in annot_data.items()
        }
        pairs = list(itertools.combinations(spec.annotators, 2)) if len(spec.annotators) > 1 else []
        if not pairs:
            raise SystemExit("At least two annotators are required per document")
        print(f"\nDocument: {spec.doc_dir}  sentences {spec.start}-{spec.end}")
        for a1, a2 in pairs:
            sents_a = sentence_slices[a1]
            sents_b = sentence_slices[a2]
            metrics = compute_metrics(sents_a, sents_b, exclude_upos={"PUNCT"})
            print(f"  Pair: {a1} vs {a2}")
            print(f"    Tokens compared: {metrics['tokens']}")
            print(
                f"    UPOS acc={metrics['upos_acc']:.4f}  kappa={metrics['upos_kappa']:.4f}  "
                f"Lemma acc={metrics['lemma_acc']:.4f}"
            )
            print(f"    UAS={metrics['uas']:.4f}  LAS={metrics['las']:.4f}")
            print("    Top relations (by frequency):")
            for rel in metrics["relations"]:
                print(
                    "      {r:<12s} F1={f1:.4f}  P={p:.4f}  R={rcl:.4f}  "
                    "matches={m}  countA={ca}  countB={cb}".format(
                        r=rel["relation"],
                        f1=rel["f1"],
                        p=rel["precision"],
                        rcl=rel["recall"],
                        m=rel["matches"],
                        ca=rel["count_a"],
                        cb=rel["count_b"],
                    )
                )
            all_tokens_a.extend(metrics["tokens_a"])
            all_tokens_b.extend(metrics["tokens_b"])

    # Combined totals across all requested docs/pairs
    if all_tokens_a:
        combined_metrics = compute_metrics(
            [all_tokens_a], [all_tokens_b], exclude_upos=set()  # already excluded punct earlier
        )
        print("\nCombined totals (all docs/pairs, punct excluded):")
        print(f"  Tokens compared: {combined_metrics['tokens']}")
        print(
            f"  UPOS acc={combined_metrics['upos_acc']:.4f}  kappa={combined_metrics['upos_kappa']:.4f}  "
            f"Lemma acc={combined_metrics['lemma_acc']:.4f}"
        )
        print(f"  UAS={combined_metrics['uas']:.4f}  LAS={combined_metrics['las']:.4f}")
        print("  Top relations (by frequency):")
        for rel in combined_metrics["relations"]:
            print(
                "    {r:<12s} F1={f1:.4f}  P={p:.4f}  R={rcl:.4f}  matches={m}  countA={ca}  countB={cb}".format(
                    r=rel["relation"],
                    f1=rel["f1"],
                    p=rel["precision"],
                    rcl=rel["recall"],
                    m=rel["matches"],
                    ca=rel["count_a"],
                    cb=rel["count_b"],
                )
            )


if __name__ == "__main__":
    main()
