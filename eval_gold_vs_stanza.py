#!/usr/bin/env python3
"""
Evaluate stitched gold parses against automatic parses from an INCEpTION export.

Gold sentences can come from:
- annotation/<DOC>/<annotator>.conllu
- curation/<DOC>/CURATION_USER.conllu via the special source name `curation`

Automatic parses are loaded from:
- source/<DOC>

Each `--gold` argument takes:
    DOC:SRC:START-END

Examples:
    python3 eval_gold_vs_stanza.py \
      --project-root inception_annotations/bg_childes_tb_project_2026-03-13_1754 \
      --gold ALE_cds.conllu:curation:1-90 \
      --gold ALE_cds.conllu:mila:91-215 \
      --gold ALE_cs.conllu:yasena:1-65 \
      --gold ALE_cs.conllu:mila:66-245
"""

from __future__ import annotations

import argparse
import collections
import pathlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from eval_agreement import Token, compute_metrics, parse_conllu, slice_sentences


@dataclass(frozen=True)
class GoldSpec:
    doc: str
    source: str
    start: int
    end: int


def parse_gold_arg(arg: str) -> GoldSpec:
    try:
        doc, source, span = arg.split(":", 2)
        start_str, end_str = span.split("-", 1)
        return GoldSpec(doc=doc, source=source, start=int(start_str), end=int(end_str))
    except Exception as exc:  # noqa: BLE001
        raise argparse.ArgumentTypeError(
            "--gold expects DOC:SRC:START-END"
        ) from exc


def load_doc_conllu(path: pathlib.Path) -> List[List[Token]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing CoNLL-U file: {path}")
    return parse_conllu(path)


def load_annotator_sentences(project_root: pathlib.Path, doc: str, source: str) -> List[List[Token]]:
    source_l = source.lower()
    if source_l == "curation":
        path = project_root / "curation" / doc / "CURATION_USER.conllu"
        return load_doc_conllu(path)

    ann_dir = project_root / "annotation" / doc
    if not ann_dir.is_dir():
        raise FileNotFoundError(f"Missing annotation dir: {ann_dir}")
    for ann_file in ann_dir.glob("*.conllu"):
        if ann_file.stem.lower() == source_l:
            return load_doc_conllu(ann_file)
    raise FileNotFoundError(f"Missing annotator '{source}' in {ann_dir}")


def ensure_token_alignment(
    gold_sents: Sequence[List[Token]],
    auto_sents: Sequence[List[Token]],
    label: str,
) -> None:
    if len(gold_sents) != len(auto_sents):
        raise ValueError(f"Sentence count mismatch for {label}: {len(gold_sents)} vs {len(auto_sents)}")
    for sent_idx, (gold_sent, auto_sent) in enumerate(zip(gold_sents, auto_sents), start=1):
        if len(gold_sent) != len(auto_sent):
            raise ValueError(
                f"Token count mismatch in {label} sentence {sent_idx}: {len(gold_sent)} vs {len(auto_sent)}"
            )
        for gold_tok, auto_tok in zip(gold_sent, auto_sent):
            if gold_tok.tid != auto_tok.tid:
                raise ValueError(
                    f"Token id mismatch in {label} sentence {sent_idx}: {gold_tok.tid} vs {auto_tok.tid}"
                )
            if gold_tok.form != auto_tok.form:
                raise ValueError(
                    f"Token form mismatch in {label} sentence {sent_idx}: '{gold_tok.form}' vs '{auto_tok.form}'"
                )


def assemble_doc_sets(
    project_root: pathlib.Path, specs: Sequence[GoldSpec]
) -> Tuple[Dict[str, List[List[Token]]], Dict[str, List[List[Token]]]]:
    grouped: Dict[str, List[GoldSpec]] = collections.defaultdict(list)
    for spec in specs:
        grouped[spec.doc].append(spec)

    gold_docs: Dict[str, List[List[Token]]] = {}
    auto_docs: Dict[str, List[List[Token]]] = {}

    for doc, doc_specs in grouped.items():
        doc_specs = sorted(doc_specs, key=lambda s: (s.start, s.end, s.source.lower()))
        auto_all = load_doc_conllu(project_root / "source" / doc)

        stitched_gold: List[List[Token]] = []
        stitched_auto: List[List[Token]] = []
        last_end = 0
        for spec in doc_specs:
            if spec.start <= last_end:
                raise ValueError(f"Overlapping ranges in {doc}: {spec.start}-{spec.end}")
            gold_all = load_annotator_sentences(project_root, doc, spec.source)
            gold_slice = slice_sentences(gold_all, spec.start, spec.end)
            auto_slice = slice_sentences(auto_all, spec.start, spec.end)
            ensure_token_alignment(gold_slice, auto_slice, f"{doc}:{spec.source}:{spec.start}-{spec.end}")
            stitched_gold.extend(gold_slice)
            stitched_auto.extend(auto_slice)
            last_end = spec.end

        gold_docs[doc] = stitched_gold
        auto_docs[doc] = stitched_auto

    return gold_docs, auto_docs


def print_metric_block(label: str, metrics: dict, top_k_relations: int) -> None:
    print(label)
    print(f"  Tokens compared: {metrics['tokens']}")
    print(
        f"  UPOS acc={metrics['upos_acc']:.4f}  kappa={metrics['upos_kappa']:.4f}  "
        f"Lemma acc={metrics['lemma_acc']:.4f}"
    )
    print(f"  UAS={metrics['uas']:.4f}  LAS={metrics['las']:.4f}")
    if top_k_relations <= 0:
        print("  Relations (all, by frequency):")
    else:
        print(f"  Top {top_k_relations} relations (by frequency):")
    for rel in metrics["relations"]:
        print(
            "    {r:<12s} F1={f1:.4f}  P={p:.4f}  R={rcl:.4f}  matches={m}  countGold={cg}  countAuto={ca}".format(
                r=rel["relation"],
                f1=rel["f1"],
                p=rel["precision"],
                rcl=rel["recall"],
                m=rel["matches"],
                cg=rel["count_a"],
                ca=rel["count_b"],
            )
        )


def flatten(sentences: Iterable[List[Token]]) -> List[Token]:
    return [token for sent in sentences for token in sent]


def main() -> None:
    default_root = pathlib.Path(__file__).parent / "inception_annotations" / "bg_childes_tb_project_2026-03-13_1754"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=pathlib.Path,
        default=default_root,
        help="Path to INCEpTION export root containing annotation/, curation/, and source/",
    )
    parser.add_argument(
        "--gold",
        action="append",
        required=True,
        dest="gold_specs",
        help="DOC:SRC:START-END where SRC is an annotator name or 'curation'",
    )
    parser.add_argument(
        "--top-k-relations",
        type=int,
        default=20,
        help="How many relations to print in the per-relation F1 table; use 0 for all.",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    specs = [parse_gold_arg(arg) for arg in args.gold_specs]
    gold_docs, auto_docs = assemble_doc_sets(project_root, specs)

    all_gold: List[Token] = []
    all_auto: List[Token] = []
    cds_gold: List[Token] = []
    cds_auto: List[Token] = []
    cs_gold: List[Token] = []
    cs_auto: List[Token] = []

    for doc in sorted(gold_docs):
        metrics = compute_metrics(
            gold_docs[doc],
            auto_docs[doc],
            exclude_upos={"PUNCT"},
            top_k_relations=args.top_k_relations,
        )
        print_metric_block(
            f"\nDocument: {doc}  gold vs source (punct excluded)",
            metrics,
            args.top_k_relations,
        )

        doc_gold_flat = flatten(gold_docs[doc])
        doc_auto_flat = flatten(auto_docs[doc])
        all_gold.extend(doc_gold_flat)
        all_auto.extend(doc_auto_flat)
        if doc.endswith("_cds.conllu"):
            cds_gold.extend(doc_gold_flat)
            cds_auto.extend(doc_auto_flat)
        elif doc.endswith("_cs.conllu"):
            cs_gold.extend(doc_gold_flat)
            cs_auto.extend(doc_auto_flat)

    if cds_gold:
        cds_metrics = compute_metrics(
            [cds_gold],
            [cds_auto],
            exclude_upos={"PUNCT"},
            top_k_relations=args.top_k_relations,
        )
        print_metric_block("\nCombined CDS totals (punct excluded)", cds_metrics, args.top_k_relations)
    if cs_gold:
        cs_metrics = compute_metrics(
            [cs_gold],
            [cs_auto],
            exclude_upos={"PUNCT"},
            top_k_relations=args.top_k_relations,
        )
        print_metric_block("\nCombined CS totals (punct excluded)", cs_metrics, args.top_k_relations)
    if all_gold:
        all_metrics = compute_metrics(
            [all_gold],
            [all_auto],
            exclude_upos={"PUNCT"},
            top_k_relations=args.top_k_relations,
        )
        print_metric_block("\nCombined overall totals (punct excluded)", all_metrics, args.top_k_relations)


if __name__ == "__main__":
    main()
