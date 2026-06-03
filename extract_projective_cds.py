#!/usr/bin/env python3
"""
Extract projective child-speech sentences from an INCEpTION export.

Given the project root (the directory that contains `annotation/`), this script:
- In v1 mode, iterates over subfolders under `annotation/` whose names end with
  `_cds.conllu`.
- In `--v2` mode, uses an expanded slice table covering the current `_cds`
  selection plus additional `_cs` and later `_cds` ranges.
- For each selected folder, prefers the configured annotator slice and falls back
  to `mila` or the first available annotator if that source is missing.
- Detects non-projective sentences (any crossing dependency arcs, heads > 0) and
  FILTERS THEM OUT.
- Writes only projective sentences (no crossing arcs) with their original
  comments/lines to an output folder (overwrites existing files).

Usage:
    python3 extract_projective_cds.py --project-root PATH \
        [--output-dir DIR] [--output-root PATH] [--v2]
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import re
from typing import Dict, List, Tuple, Optional


PUNCT_RE = re.compile(r"([.,;:!?])")
QUOTE_RE = re.compile(r'["„“«»]')
START_PUNCT_RE = re.compile(r"^[\s.,;:!?]+")


def normalize_token_line(line: str) -> str:
    """Normalize common non-CoNLL placeholders to '_' in token lines."""
    if not line or line.startswith("#") or "\t" not in line:
        return line
    fields = line.split("\t")
    # Keep ID and FORM intact; normalize other fields.
    for i in range(2, len(fields)):
        val = fields[i].strip()
        if val == "" or val.lower() in {"none", "null"}:
            fields[i] = "_"
    return "\t".join(fields)


def safe_child_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        return "unknown"
    cleaned = re.sub(r"[^\w.-]+", "_", name, flags=re.UNICODE)
    cleaned = cleaned.strip("_.")
    return cleaned or "unknown"


def capitalize_first_letter(text: str) -> str:
    chars = list(text)
    for i, ch in enumerate(chars):
        if ch.isalpha():
            if ch.islower():
                chars[i] = ch.upper()
            break
    return "".join(chars)


def normalize_export_sentence(text: str, strip_quotes: bool = True) -> str:
    sent = (text or "").strip()
    if strip_quotes:
        sent = QUOTE_RE.sub("", sent)
    sent = PUNCT_RE.sub(r" \1 ", sent)
    sent = START_PUNCT_RE.sub("", sent)
    sent = re.sub(r"\s{2,}", " ", sent).strip()
    return capitalize_first_letter(sent)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--project-root",
        required=True,
        type=pathlib.Path,
        help="Path to INCEpTION export root (containing annotation/)",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        type=pathlib.Path,
        help="Output directory name or path; if relative, it is created under --output-root (or project root by default)",
    )
    p.add_argument(
        "--output-root",
        default=None,
        type=pathlib.Path,
        help="Base directory for relative output paths (defaults to project root). Use this to write outside the project.",
    )
    p.add_argument(
        "--v2",
        action="store_true",
        help="Use the expanded v2 slice table instead of the current v1 selection.",
    )
    p.add_argument(
        "--all-annotated",
        action="store_true",
        help="Write all selected manually annotated sentences, not just projective ones.",
    )
    p.add_argument(
        "--sent-id-csv",
        default=None,
        type=pathlib.Path,
        help=(
            "Optional LabLing CSV with utterance_id values. When provided, utterance_id "
            "is aligned to the INCEpTION source order and written out as # sent_id."
        ),
    )
    return p.parse_args()


def is_nonprojective(sentence_lines: list[str]) -> bool:
    """Detect non-projectivity based on token/head pairs."""
    arcs = []
    for line in sentence_lines:
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 7:
            continue
        tid = parts[0]
        head = parts[6]
        if "-" in tid or "." in tid:
            continue  # skip MWT/empty nodes
        try:
            tid_i = int(tid)
            head_i = int(head)
        except ValueError:
            continue
        if head_i == 0:
            continue
        start, end = sorted((tid_i, head_i))
        arcs.append((start, end))
    for i in range(len(arcs)):
        a_start, a_end = arcs[i]
        for j in range(i + 1, len(arcs)):
            b_start, b_end = arcs[j]
            # Strict crossing (not nested)
            if (a_start < b_start < a_end < b_end) or (b_start < a_start < b_end < a_end):
                return True
    return False


def has_invalid_heads(sentence_lines: list[str]) -> bool:
    """Return True if any token line has non-numeric HEAD or missing DEPREL."""
    for line in sentence_lines:
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 8:
            continue
        tid = parts[0]
        head = parts[6]
        deprel = parts[7]
        if "-" in tid or "." in tid:
            continue
        if not head.isdigit():
            return True
        if not deprel or deprel == "_":
            return True
    return False


def sentences_from_file(path: pathlib.Path) -> List[List[str]]:
    """Return list of sentences (each is list of lines)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    buf: list[str] = []
    sents: List[List[str]] = []
    for line in lines:
        if line.strip() == "":
            if buf:
                sents.append(buf)
                buf = []
            continue
        buf.append(line)
    if buf:
        sents.append(buf)
    return sents


def sentence_text(sentence_lines: List[str]) -> Optional[str]:
    for line in sentence_lines:
        if line.startswith("# text = "):
            return line.removeprefix("# text = ")
    return None


def sanitize_comment_value(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def split_sentence_lines(sentence_lines: List[str]) -> Tuple[List[str], List[str]]:
    comments: List[str] = []
    tokens: List[str] = []
    in_tokens = False
    for line in sentence_lines:
        if not in_tokens and line.startswith("#"):
            comments.append(line)
        else:
            in_tokens = True
            tokens.append(line)
    return comments, tokens


def set_sentence_metadata(
    sentence_lines: List[str],
    metadata: Dict[str, str],
) -> List[str]:
    """Replace or insert selected sentence-level metadata comments."""
    comments, tokens = split_sentence_lines(sentence_lines)
    managed_keys = ("sent_id", "child_age", "participant_role", "original_utterance")
    preserved_comments = [
        line
        for line in comments
        if not any(line.startswith(f"# {key} = ") for key in managed_keys)
    ]

    ordered_comments = list(preserved_comments)
    if "sent_id" in metadata:
        ordered_comments.insert(0, f"# sent_id = {sanitize_comment_value(metadata['sent_id'])}")

    insert_after = 0
    for idx, line in enumerate(ordered_comments):
        if line.startswith("# sent_id = "):
            insert_after = idx + 1
            break

    extras: List[str] = []
    for key in ("child_age", "participant_role", "original_utterance"):
        value = metadata.get(key, "")
        if value:
            extras.append(f"# {key} = {sanitize_comment_value(value)}")
    ordered_comments[insert_after:insert_after] = extras
    return [*ordered_comments, *tokens]


def ensure_sentence_has_sent_id(sentence_lines: List[str], sent_idx: int) -> List[str]:
    """Insert a deterministic sent_id when the source sentence lacks one."""
    if any(line.startswith("# sent_id = ") for line in sentence_lines):
        return sentence_lines
    return [f"# sent_id = {sent_idx}", *sentence_lines]


def aligned_sentence_metadata_from_csv(
    project_root: pathlib.Path,
    csv_path: pathlib.Path,
) -> Dict[str, List[Dict[str, str]]]:
    """Align sentence metadata from the LabLing CSV to INCEpTION source docs."""
    if not csv_path.exists():
        raise SystemExit(f"sent-id CSV not found: {csv_path}")
    source_root = project_root / "source"
    if not source_root.is_dir():
        raise SystemExit(
            f"source/ not found under {project_root}; required for sent-id alignment"
        )

    required_columns = {
        "utterance_id",
        "Name",
        "Participant",
        "Age",
        "Utterance",
        "Manually_corrected",
    }
    doc_rows: Dict[str, List[Tuple[Dict[str, str], str]]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit(f"No header found in {csv_path}")
        missing = required_columns - set(reader.fieldnames)
        if missing:
            raise SystemExit(
                f"Missing required columns in {csv_path}: {', '.join(sorted(missing))}"
            )
        for row_num, row in enumerate(reader, start=2):
            utterance_id = (row.get("utterance_id") or "").strip()
            if not utterance_id:
                raise SystemExit(f"Empty utterance_id at {csv_path}:{row_num}")
            name = (row.get("Name") or "").strip()
            participant = (row.get("Participant") or "").strip()
            text = normalize_export_sentence(row.get("Manually_corrected") or "")
            if not text:
                continue
            suffix = "cs" if name == participant else "cds"
            doc_name = f"{safe_child_name(name)}_{suffix}.conllu"
            metadata = {
                "sent_id": utterance_id,
                "child_age": row.get("Age") or "",
                "participant_role": participant,
                "original_utterance": row.get("Utterance") or "",
            }
            doc_rows.setdefault(doc_name, []).append((metadata, text))

    aligned_metadata: Dict[str, List[Dict[str, str]]] = {}
    for source_file in sorted(source_root.glob("*.conllu")):
        doc_name = source_file.name
        if doc_name not in doc_rows:
            continue
        expected_rows = doc_rows[doc_name]
        source_sents = sentences_from_file(source_file)
        if len(expected_rows) != len(source_sents):
            raise SystemExit(
                f"sent-id alignment count mismatch for {doc_name}: "
                f"CSV has {len(expected_rows)} sentences, source has {len(source_sents)}"
            )
        aligned_doc_metadata: List[Dict[str, str]] = []
        for idx, ((metadata, expected_text), source_sent) in enumerate(
            zip(expected_rows, source_sents),
            start=1,
        ):
            source_text = sentence_text(source_sent)
            if source_text != expected_text:
                raise SystemExit(
                    f"sent-id alignment text mismatch for {doc_name} sentence {idx}: "
                    f"CSV={expected_text!r} source={source_text!r}"
                )
            aligned_doc_metadata.append(metadata)
        aligned_metadata[doc_name] = aligned_doc_metadata
    return aligned_metadata


def aligned_sent_ids_from_csv(
    project_root: pathlib.Path,
    csv_path: pathlib.Path,
) -> Dict[str, List[str]]:
    """Backwards-compatible wrapper returning only aligned sent_id values."""
    metadata_by_doc = aligned_sentence_metadata_from_csv(project_root, csv_path)
    return {
        doc_name: [metadata["sent_id"] for metadata in doc_metadata]
        for doc_name, doc_metadata in metadata_by_doc.items()
    }


def resolved_sentence_metadata(
    metadata_by_doc: Optional[Dict[str, List[Dict[str, str]]]],
    doc_name: str,
    sent_idx: int,
) -> Dict[str, str]:
    if metadata_by_doc is None:
        return {"sent_id": str(sent_idx)}
    doc_metadata = metadata_by_doc.get(doc_name)
    if doc_metadata is None:
        return {"sent_id": str(sent_idx)}
    if sent_idx < 1 or sent_idx > len(doc_metadata):
        raise SystemExit(
            f"sent-id alignment index out of range for {doc_name}: "
            f"sentence {sent_idx} but only {len(doc_metadata)} aligned rows"
        )
    return doc_metadata[sent_idx - 1]


def load_sentences_with_stem(doc_dir: pathlib.Path, stem: str) -> Optional[List[List[str]]]:
    """Load annotator file by stem (case-insensitive) from a doc directory."""
    stem_l = stem.lower()
    for ann_file in doc_dir.glob("*.conllu"):
        if ann_file.stem.lower() == stem_l:
            return sentences_from_file(ann_file)
    return None


# Manual annotator ranges per document (1-based inclusive)
# doc_name (folder) -> list of (start_idx, end_idx, annotator_stem)
ANNOTATOR_RANGES_V1: Dict[str, List[Tuple[int, int, str]]] = {
    "ale_cds.conllu": [
        (1, 90, "curated"),
        (91, 215, "mila"),
    ],
    "bog_cds.conllu": [
        (1, 148, "curated"),
    ],
    "eli_cds.conllu": [
        (1, 160, "tsvetelina"),
        (161, 385, "mila"),
    ],
    "sim_cds.conllu": [
        (1, 160, "tsvetina"),
    ],
    "tef_cds.conllu": [
        (1, 90, "ivelina"),
    ],
}


# Expanded slice table derived from the current selection plus the additional
# spans in the v2 spreadsheet. Overlaps are resolved here into a single,
# non-overlapping precedence order so the output stays deterministic.
ANNOTATOR_RANGES_V2: Dict[str, List[Tuple[int, int, str]]] = {
    "ale_cs.conllu": [
        (1, 65, "yasena"),
        (66, 245, "mila"),
        (246, 320, "yasena"),
        (321, 379, "mila"),
    ],
    "ale_cds.conllu": [
        (1, 90, "curated"),
        (91, 215, "mila"),
        (216, 320, "yasena"),
        (321, 430, "mila"),
    ],
    "bog_cs.conllu": [
        (1, 97, "curated"),
    ],
    "bog_cds.conllu": [
        (1, 148, "curated"),
    ],
    "eli_cs.conllu": [
        (1, 88, "curated"),
        (89, 121, "tsvetelina"),
    ],
    "eli_cds.conllu": [
        (1, 160, "tsvetelina"),
        (161, 385, "mila"),
        (386, 560, "tsvetelina"),
        (561, 706, "mila"),
    ],
    "sim_cs.conllu": [
        (1, 100, "tsvetina"),
        (101, 400, "mila"),
        (401, 504, "tsvetina"),
        (505, 515, "curated"),
        (516, 924, "mila"),
    ],
    "sim_cds.conllu": [
        (1, 160, "tsvetina"),
        (161, 335, "curated"),
        (336, 805, "mila"),
    ],
    "tef_cs.conllu": [
        (1, 180, "ivelina"),
        (181, 275, "tsvetelina"),
        (276, 336, "yoana"),
    ],
    "tef_cds.conllu": [
        (1, 265, "ivelina"),
        (266, 392, "yoana"),
    ],
}


def resolve_annotator(annot_cache: Dict[str, List[List[str]]], stem: str) -> Optional[List[List[str]]]:
    """Return sentences list for annotator stem with fallbacks."""
    stem_l = stem.lower()
    if stem_l in annot_cache:
        return annot_cache[stem_l]
    # general fallback to mila then any other
    if "mila" in annot_cache:
        return annot_cache["mila"]
    if annot_cache:
        return next(iter(annot_cache.values()))
    return None


def selected_ranges(use_v2: bool) -> Dict[str, List[Tuple[int, int, str]]]:
    return ANNOTATOR_RANGES_V2 if use_v2 else ANNOTATOR_RANGES_V1


def should_process_doc(doc_name: str, use_v2: bool, ranges_by_doc: Dict[str, List[Tuple[int, int, str]]]) -> bool:
    doc_key = doc_name.lower()
    if use_v2:
        return doc_key in ranges_by_doc
    return doc_name.endswith("_cds.conllu")


def doc_bucket(doc_name: str) -> Optional[str]:
    doc_name_l = doc_name.lower()
    if doc_name_l.endswith("_cs.conllu"):
        return "cs"
    if doc_name_l.endswith("_cds.conllu"):
        return "cds"
    return None


def keep_sentence(sentence_lines: List[str], all_annotated: bool) -> bool:
    if all_annotated:
        return True
    if has_invalid_heads(sentence_lines):
        return False
    return not is_nonprojective(sentence_lines)


def kept_sentences_from_ranges(
    doc_dir: pathlib.Path,
    cur_root: pathlib.Path,
    ranges: List[Tuple[int, int, str]],
    all_annotated: bool,
    metadata_by_doc: Optional[Dict[str, List[Dict[str, str]]]] = None,
) -> Tuple[List[str], int]:
    """Resolve the configured slice table for one document and return kept sentences."""
    annot_cache: Dict[str, List[List[str]]] = {}
    for ann_file in doc_dir.glob("*.conllu"):
        annot_cache[ann_file.stem.lower()] = sentences_from_file(ann_file)
    if cur_root.is_dir():
        cur_dir = cur_root / doc_dir.name
        cur_file = cur_dir / "CURATION_USER.conllu"
        if cur_file.exists():
            annot_cache["curation_user"] = sentences_from_file(cur_file)

    kept_sentences: List[str] = []
    total_sentences = max(r[1] for r in ranges)
    assigned_indices: set[int] = set()
    for start, end, ann_stem in ranges:
        ann_sents = load_sentences_with_stem(doc_dir, ann_stem) if ann_stem.lower() != "curated" else None
        if ann_stem.lower() == "curated" and cur_root.is_dir():
            cur_dir = cur_root / doc_dir.name
            cur_file = cur_dir / "CURATION_USER.conllu"
            if cur_file.exists():
                ann_sents = sentences_from_file(cur_file)
        if ann_sents is None:
            ann_sents = resolve_annotator(annot_cache, ann_stem)
        if ann_sents is None:
            print(f"[warn] missing annotator {ann_stem} for {doc_dir.name}")
            continue
        for idx in range(start, min(end, len(ann_sents)) + 1):
            if idx in assigned_indices:
                continue
            sent_lines = ann_sents[idx - 1]
            normalized = [normalize_token_line(line) for line in sent_lines]
            normalized = set_sentence_metadata(
                normalized,
                resolved_sentence_metadata(metadata_by_doc, doc_dir.name, idx),
            )
            assigned_indices.add(idx)
            if keep_sentence(normalized, all_annotated):
                kept_sentences.append("\n".join(normalized))
    return kept_sentences, total_sentences


def main():
    args = parse_args()
    project_root = args.project_root.resolve()
    ann_root = project_root / "annotation"
    cur_root = project_root / "curation"
    if not ann_root.is_dir():
        raise SystemExit(f"annotation/ not found under {project_root}")

    ranges_by_doc = selected_ranges(args.v2)
    default_output_dir = pathlib.Path("manually annotated") if args.all_annotated else pathlib.Path("bg_projective_manuallyverified")
    output_dir = args.output_dir if args.output_dir is not None else default_output_dir
    base_root = args.output_root.resolve() if args.output_root else project_root
    out_dir = output_dir if output_dir.is_absolute() else (base_root / output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_by_doc = None
    if args.sent_id_csv is not None:
        metadata_by_doc = aligned_sentence_metadata_from_csv(
            project_root,
            args.sent_id_csv.resolve(),
        )

    total_written = 0
    mode_label = "annotated" if args.all_annotated else "projective"
    totals_by_bucket = {
        "cs": {"written": 0, "total": 0},
        "cds": {"written": 0, "total": 0},
    }
    for doc_dir in sorted(ann_root.iterdir()):
        if not doc_dir.is_dir():
            continue
        if not should_process_doc(doc_dir.name, args.v2, ranges_by_doc):
            continue
        output_path = out_dir / doc_dir.name

        doc_key = doc_dir.name.lower()
        ranges = ranges_by_doc.get(doc_key)
        if ranges:
            kept_sentences, total_sentences = kept_sentences_from_ranges(
                doc_dir,
                cur_root,
                ranges,
                args.all_annotated,
                metadata_by_doc=metadata_by_doc,
            )
        else:
            preferred_path = doc_dir / "mila.conllu"
            if preferred_path.exists():
                ann_path = preferred_path
                ann_name = "mila"
            else:
                candidates = sorted(
                    p for p in doc_dir.glob("*.conllu") if p.name.lower() != "initial_cas.conllu"
                )
                if not candidates:
                    print(f"[skip] no annotator files in {doc_dir.name}")
                    continue
                ann_path = candidates[0]
                ann_name = ann_path.stem
                print(f"[fallback] using {ann_name} in {doc_dir.name} (mila missing)")

            kept_sentences = []
            total_sentences = 0
            for sent_lines in sentences_from_file(ann_path):
                total_sentences += 1
                normalized = [normalize_token_line(line) for line in sent_lines]
                normalized = set_sentence_metadata(
                    normalized,
                    resolved_sentence_metadata(
                        metadata_by_doc,
                        doc_dir.name,
                        total_sentences,
                    ),
                )
                if keep_sentence(normalized, args.all_annotated):
                    kept_sentences.append("\n".join(normalized))

        if kept_sentences:
            output_path.write_text("\n\n".join(kept_sentences) + "\n", encoding="utf-8")
            total_written += len(kept_sentences)
            bucket = doc_bucket(doc_dir.name)
            if bucket is not None:
                totals_by_bucket[bucket]["written"] += len(kept_sentences)
                totals_by_bucket[bucket]["total"] += total_sentences
            print(f"[write] {output_path}  ({len(kept_sentences)} / {total_sentences} sentences {mode_label})")
        else:
            if output_path.exists():
                output_path.unlink()
            bucket = doc_bucket(doc_dir.name)
            if bucket is not None:
                totals_by_bucket[bucket]["total"] += total_sentences
            if args.all_annotated:
                print(f"[none]  {doc_dir.name} (no annotated sentences selected)")
            else:
                print(f"[none]  {doc_dir.name} (no projective sentences after filtering)")

    if totals_by_bucket["cs"]["total"] > 0:
        print(
            f"[total cs] ({totals_by_bucket['cs']['written']} / "
            f"{totals_by_bucket['cs']['total']} sentences {mode_label})"
        )
    if totals_by_bucket["cds"]["total"] > 0:
        print(
            f"[total cds] ({totals_by_bucket['cds']['written']} / "
            f"{totals_by_bucket['cds']['total']} sentences {mode_label})"
        )
    print(f"Done. Total {mode_label} sentences written: {total_written}")


if __name__ == "__main__":
    main()
