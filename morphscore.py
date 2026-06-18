from __future__ import annotations

import argparse
import csv
import importlib
import math
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev
from urllib.request import urlopen

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MORPHTOK_MODULE = "bg_morphtok"
MORPHTOK_MODULE_CHOICES = ("bg_morphtok", "bg_morphtok_verbtypes")
MORPHTOK_MODULE_ALIASES = {"bg_morphtok_verbtypes": "bg_morphtok"}
DEFAULT_DATA_URL_V1 = (
    "https://github.com/catherinearnett/morphscore/raw/v1/data/"
    "bulgarian_morph_data.csv"
)
DEFAULT_DATA_URL_V2 = (
    "https://huggingface.co/datasets/catherinearnett/morphscore/"
    "resolve/main/bulgarian_data.csv?download=true"
)
DEFAULT_WRONG_OUTPUT_V1 = SCRIPT_DIR / "morphtok_wrong.csv"
DEFAULT_WRONG_OUTPUT_V2 = SCRIPT_DIR / "morphtok_wrong_v2.csv"
DEFAULT_ALL_OUTPUT_V1 = SCRIPT_DIR / "morphtok_all.csv"
DEFAULT_ALL_OUTPUT_V2 = SCRIPT_DIR / "morphtok_all_v2.csv"
ADJ_ARTICLE_SUFFIXES = {"та", "то", "те", "я", "ят", "ия", "ият", "ът"}
ADJ_GENDER_SUFFIXES = {"а", "я", "о", "е"}
CANONICAL_ADJ_BOUNDARY_SHIFTS = [
    ("те", "ите"),
    ("та", "ата"),
    ("то", "ото"),
]
SPECIAL_DETAIL_LABELS = {
    "adj_def_boundary",
    "adj_def_split",
    "adj_gender",
    "plu1person_boundary",
}
FINITE_VERBAL_ENDINGS = {
    "м",
    "ш",
    "а",
    "я",
    "е",
    "и",
    "ме",
    "те",
    "т",
    "ем",
    "им",
    "ат",
    "ят",
    "ете",
    "ите",
    "х",
    "ха",
    "хме",
    "хте",
    "ше",
    "й",
    "йте",
}


def configure_morphtok(module_name: str):
    global bg_morphtok
    global DEFAULT_LEXICON_PATH
    global PosLexicon
    global TokenAnalysis
    global analyses_from_doc
    global build_pipeline
    global build_stemmer
    global normalize
    global tokenize_sentence

    resolved_module_name = MORPHTOK_MODULE_ALIASES.get(module_name, module_name)
    bg_morphtok = importlib.import_module(resolved_module_name)
    DEFAULT_LEXICON_PATH = bg_morphtok.DEFAULT_LEXICON_PATH
    PosLexicon = bg_morphtok.PosLexicon
    TokenAnalysis = bg_morphtok.TokenAnalysis
    analyses_from_doc = bg_morphtok.analyses_from_doc
    build_pipeline = bg_morphtok.build_pipeline
    build_stemmer = bg_morphtok.build_stemmer
    normalize = bg_morphtok.normalize
    tokenize_sentence = bg_morphtok.tokenize_sentence
    return bg_morphtok


configure_morphtok(DEFAULT_MORPHTOK_MODULE)


def load_rows(data_file: Path | None, data_url: str) -> list[dict[str, str]]:
    if data_file is not None:
        with data_file.open("r", encoding="utf-8", newline="") as infile:
            return list(csv.DictReader(infile))

    with urlopen(data_url) as response:
        text = response.read().decode("utf-8")
    return list(csv.DictReader(text.splitlines()))


def normalize_gold_text(value: object) -> str | None:
    if is_nan_like(value):
        return None
    return normalize(str(value))


def format_feats(feats: dict[str, str]) -> str:
    if not feats:
        return "_"
    return "|".join(f"{key}={value}" for key, value in feats.items())


def analysis_metadata(analysis: TokenAnalysis | None) -> dict[str, str]:
    if analysis is None:
        return {"UPOS": "", "feats": ""}
    return {
        "UPOS": analysis.upos,
        "feats": format_feats(analysis.feats),
    }


# ----------------------------
# v1 evaluation
# ----------------------------
def morph_eval_v1(morphemes: list[str], tokens: list[str]) -> int:
    if len(tokens) == 1:
        return 0

    for split_idx in range(len(tokens) - 1):
        pt1 = "".join(tokens[: split_idx + 1])
        rest = "".join(tokens[split_idx + 1 :])
        if [pt1, rest] == morphemes:
            return 1

    return -1


def morph_eval_label_v1(result: int) -> str:
    if result == 1:
        return "correct"
    if result == 0:
        return "skipped_single_token"
    return "wrong"


def is_adj_like_analysis(analysis: TokenAnalysis | None) -> bool:
    if analysis is None:
        return False
    if analysis.upos in {"ADJ", "DET", "PRON"}:
        return True
    return analysis.upos == "VERB" and analysis.feats.get("VerbForm") == "Part"


def is_adj_gender_like_analysis(analysis: TokenAnalysis | None) -> bool:
    if analysis is None:
        return False
    if analysis.upos in {"ADJ", "DET", "PRON", "ADV"}:
        return True
    return analysis.upos == "VERB" and analysis.feats.get("VerbForm") == "Part"


def is_plu1person_verb_analysis(analysis: TokenAnalysis | None) -> bool:
    if analysis is None:
        return False
    if analysis.upos not in {"VERB", "AUX"}:
        return False
    return (
        analysis.feats.get("VerbForm") == "Fin"
        and analysis.feats.get("Person") == "1"
        and analysis.feats.get("Number") == "Plur"
    )


def is_finite_verb_analysis(analysis: TokenAnalysis | None) -> bool:
    if analysis is None:
        return False
    if analysis.upos not in {"VERB", "AUX"}:
        return False
    return analysis.feats.get("VerbForm") == "Fin"


def has_adj_def_boundary_shift(
    gold_left: str,
    gold_right: str,
    pred_tokens: list[str],
) -> bool:
    if len(pred_tokens) != 2:
        return False

    pred_left, pred_right = pred_tokens
    if gold_left + gold_right != pred_left + pred_right:
        return False

    moved_from_gold_right = (
        pred_right in ADJ_ARTICLE_SUFFIXES
        and gold_right.endswith(pred_right)
        and len(gold_right) > len(pred_right)
        and pred_left == gold_left + gold_right[: -len(pred_right)]
    )
    moved_from_pred_right = (
        gold_right in ADJ_ARTICLE_SUFFIXES
        and pred_right.endswith(gold_right)
        and len(pred_right) > len(gold_right)
        and gold_left == pred_left + pred_right[: -len(gold_right)]
    )
    return moved_from_gold_right or moved_from_pred_right


def has_plu1person_boundary_shift(
    gold_left: str,
    gold_right: str,
    pred_tokens: list[str],
) -> bool:
    if len(pred_tokens) != 2:
        return False

    pred_left, pred_right = pred_tokens
    if gold_left + gold_right != pred_left + pred_right:
        return False

    return (
        gold_right == "е"
        and pred_right == "ме"
        and gold_left == pred_left + "м"
    )


def has_verbal_ending_boundary_shift(
    gold_left: str,
    gold_right: str,
    pred_tokens: list[str],
    analysis: TokenAnalysis | None,
) -> bool:
    if not is_finite_verb_analysis(analysis):
        return False
    if len(pred_tokens) != 2:
        return False

    pred_left, pred_right = pred_tokens
    if gold_left + gold_right != pred_left + pred_right:
        return False
    if gold_right == pred_right:
        return False
    if gold_right not in FINITE_VERBAL_ENDINGS:
        return False
    if pred_right not in FINITE_VERBAL_ENDINGS:
        return False

    moved_from_gold_right = (
        pred_right.endswith(gold_right)
        and len(pred_right) > len(gold_right)
        and pred_left + pred_right[: -len(gold_right)] == gold_left
    )
    moved_from_pred_right = (
        gold_right.endswith(pred_right)
        and len(gold_right) > len(pred_right)
        and gold_left + gold_right[: -len(pred_right)] == pred_left
    )
    return moved_from_gold_right or moved_from_pred_right


def has_adj_def_split(morphemes: list[str], tokens: list[str]) -> bool:
    if "".join(morphemes) != "".join(tokens):
        return False

    if len(morphemes) == 2 and len(tokens) == 3:
        return (
            tokens[0] == morphemes[0]
            and tokens[1] + tokens[2] == morphemes[1]
            and tokens[2] in ADJ_ARTICLE_SUFFIXES
        )

    if len(morphemes) == 3 and len(tokens) == 4:
        return (
            tokens[:2] == morphemes[:2]
            and tokens[2] + tokens[3] == morphemes[2]
            and tokens[3] in ADJ_ARTICLE_SUFFIXES
        )

    return False


def has_adj_def_boundary_shift_v2(
    morphemes: list[str],
    tokens: list[str],
) -> bool:
    if len(morphemes) == 2 and len(tokens) == 2:
        return has_adj_def_boundary_shift(morphemes[0], morphemes[1], tokens)

    if len(morphemes) == 3 and len(tokens) == 3 and morphemes[0] == tokens[0]:
        return has_adj_def_boundary_shift(morphemes[1], morphemes[2], tokens[1:])

    return False


def classify_v2_detail(
    morphemes: list[str],
    tokens: list[str],
    analysis: TokenAnalysis | None,
) -> str:
    if tokens == morphemes:
        return ""
    if "".join(morphemes) != "".join(tokens):
        return ""

    if is_adj_gender_like_analysis(analysis):
        if len(tokens) == 1 and morphemes and morphemes[-1] in ADJ_GENDER_SUFFIXES:
            return "adj_gender"

    if is_adj_like_analysis(analysis):
        if has_adj_def_boundary_shift_v2(morphemes, tokens):
            return "adj_def_boundary"
        if has_adj_def_split(morphemes, tokens):
            return "adj_def_split"

    if is_plu1person_verb_analysis(analysis):
        if len(morphemes) == 2 and has_plu1person_boundary_shift(
            morphemes[0],
            morphemes[1],
            tokens,
        ):
            return "plu1person_boundary"

    return ""


def classify_v1_mismatch(
    row: dict[str, str],
    tokens: list[str],
    result: int,
    analysis: TokenAnalysis | None,
) -> str:
    gold_left = normalize(row["pt1"])
    gold_right = normalize(row["rest"])

    if is_adj_gender_like_analysis(analysis):
        if result == 0 and len(tokens) == 1 and gold_right in ADJ_GENDER_SUFFIXES:
            return "adj_gender"
    if is_adj_like_analysis(analysis):
        if result == -1 and has_adj_def_boundary_shift(gold_left, gold_right, tokens):
            return "adj_def_boundary"
    if is_plu1person_verb_analysis(analysis):
        if result == -1 and has_plu1person_boundary_shift(gold_left, gold_right, tokens):
            return "plu1person_boundary"
    if result == -1 and has_verbal_ending_boundary_shift(
        gold_left,
        gold_right,
        tokens,
        analysis,
    ):
        return "verb_ending_boundary"

    return morph_eval_label_v1(result)


def get_morphscore_v1(
    rows: list[dict[str, str]],
    tokenizer,
    analyzer=None,
    *,
    include_all_rows: bool = False,
) -> tuple[float, int, int, int, int, int, list[dict[str, str]]]:
    scored_points: list[int] = []
    attempted = 0
    correct = 0
    wrong = 0
    skipped_single_token = 0
    wrong_rows: list[dict[str, str]] = []

    for row in rows:
        morphemes = [
            normalize(row["pt1"]),
            normalize(row["rest"]),
        ]
        word = get_word_v1(row)
        analysis = analyzer(word) if analyzer is not None else None
        tokens = tokenizer(word)
        expected_morphtok = " ".join(morphemes)
        predicted_morphtok = " ".join(tokens)

        result = morph_eval_v1(morphemes, tokens)
        label = classify_v1_mismatch(
            row,
            tokens,
            result,
            analysis,
        )
        if include_all_rows or tokens != morphemes:
            wrong_rows.append(
                {
                    "full_word": word,
                    "pt1": morphemes[0],
                    "rest": morphemes[1],
                    "expected_morphtok": expected_morphtok,
                    "predicted_morphtok": predicted_morphtok,
                    "morphscore_result": label,
                    **analysis_metadata(analysis),
                }
            )

        if result == 0:
            skipped_single_token += 1
            continue

        attempted += 1
        if result == 1:
            correct += 1
            scored_points.append(1)
        else:
            wrong += 1
            scored_points.append(0)

    # This is the original MorphScore reported in Arnett & Bergen (2025),
    # Table 3: binary boundary accuracy over scored items only.
    score = mean(scored_points) if scored_points else 0.0
    total_assessed = len(rows)
    return (
        score,
        attempted,
        correct,
        wrong,
        skipped_single_token,
        total_assessed,
        wrong_rows,
    )


# ----------------------------
# v2 evaluation
# ----------------------------
def is_nan_like(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    text = str(value).strip()
    return text == "" or text.lower() == "nan"


def clean_part(value: object) -> str | None:
    return normalize_gold_text(value)


def get_word_v1(row: dict[str, str]) -> str:
    return normalize(row["full_word"])


def get_word_v2(row: dict[str, str]) -> str:
    return normalize(row["wordform"])


def build_gold_morphemes_v2(row: dict[str, str]) -> list[str]:
    prefix = clean_part(row.get("preceding_part"))
    stem = clean_part(row.get("stem"))
    suffix = clean_part(row.get("following_part"))

    if stem is None:
        return []

    morphemes: list[str] = []
    if prefix is not None:
        morphemes.append(prefix)
    morphemes.append(stem)
    if suffix is not None:
        morphemes.append(suffix)
    return morphemes


def get_predicted_boundaries(tokens: list[str]) -> list[int]:
    boundaries: list[int] = []
    idx = 0
    for tok in tokens:
        idx += len(tok)
        boundaries.append(idx)
    return boundaries


def morph_eval_v2(
    morphemes: list[str],
    tokens: list[str],
    *,
    exclude_single_tok: bool,
    exclude_single_morpheme: bool,
    single_tok_point: float,
    correct_point: float,
    partial_point: float,
) -> tuple[float, float]:
    if len(tokens) == 1:
        if exclude_single_tok:
            return (math.nan, math.nan)
        return (single_tok_point, single_tok_point)

    pred_boundaries = get_predicted_boundaries(tokens)

    if len(morphemes) == 2:
        gold_boundary = len(morphemes[0])
        if gold_boundary in pred_boundaries:
            return (correct_point, 1.0 / len(pred_boundaries))
        return (0.0, 0.0)

    if len(morphemes) == 3:
        b1 = len(morphemes[0])
        b2 = len(morphemes[0]) + len(morphemes[1])
        matched = sum(int(b in pred_boundaries) for b in (b1, b2))

        if matched == 2:
            return (correct_point, 2.0 / len(pred_boundaries))
        if matched == 1:
            return (partial_point, 1.0 / len(pred_boundaries))
        return (0.0, 0.0)

    if len(morphemes) == 1:
        if exclude_single_morpheme:
            return (math.nan, math.nan)
        return (
            (single_tok_point, single_tok_point)
            if morphemes == tokens
            else (0.0, 0.0)
        )

    return (math.nan, math.nan)


def filter_v2_rows(
    rows: list[dict[str, str]],
    *,
    unique_only: bool,
    stem_eq_lemma: bool,
    exclude_numbers: bool,
) -> list[dict[str, str]]:
    filtered: list[dict[str, str]] = []

    for row in rows:
        word = get_word_v2(row)

        if unique_only and row.get("unique") != "unique":
            continue
        if (
            stem_eq_lemma
            and normalize_gold_text(row.get("stem")) != normalize_gold_text(row.get("lemma"))
        ):
            continue
        if exclude_numbers and re.search(r"\d", word):
            continue

        filtered.append(row)

    return filtered


def get_float(row: dict[str, str], key: str, default: float) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def mean_or_zero(values: list[float]) -> float:
    return mean(values) if values else 0.0


def std_or_zero(values: list[float]) -> float:
    return pstdev(values) if len(values) > 1 else 0.0


def f1_or_zero(precision: float, recall: float) -> float:
    denom = precision + recall
    if denom == 0.0:
        return 0.0
    return 2.0 * precision * recall / denom


def get_morphscore_v2(
    rows: list[dict[str, str]],
    tokenizer,
    analyzer=None,
    *,
    include_all_rows: bool = False,
    freq_scale: bool,
    exclude_single_tok: bool,
    exclude_single_morpheme: bool,
    single_tok_point: float,
    correct_point: float,
    partial_point: float,
) -> tuple[dict[str, float], list[dict[str, str]]]:
    recall_points: list[float] = []
    precision_points: list[float] = []
    recall_points_unweighted: list[float] = []
    precision_points_unweighted: list[float] = []
    weights: list[float] = []
    token_char_ratios: list[float] = []

    wrong_rows: list[dict[str, str]] = []

    correct_full = 0
    partial = 0
    wrong = 0
    skipped = 0

    for row in rows:
        word = get_word_v2(row)
        analysis = analyzer(word) if analyzer is not None else None
        morphemes = build_gold_morphemes_v2(row)
        if not morphemes:
            skipped += 1
            continue

        tokens = tokenizer(word)
        expected = " ".join(morphemes)
        predicted = " ".join(tokens)

        recall_pt, precision_pt = morph_eval_v2(
            morphemes,
            tokens,
            exclude_single_tok=exclude_single_tok,
            exclude_single_morpheme=exclude_single_morpheme,
            single_tok_point=single_tok_point,
            correct_point=correct_point,
            partial_point=partial_point,
        )

        if len(word) > 0:
            token_char_ratios.append(len(tokens) / len(word))

        if math.isnan(recall_pt) or math.isnan(precision_pt):
            skipped += 1
            detail_label = classify_v2_detail(morphemes, tokens, analysis)
            if include_all_rows or tokens != morphemes:
                wrong_rows.append(
                    {
                        "wordform": word,
                        "lemma": normalize_gold_text(row.get("lemma")) or "",
                        "stem": normalize_gold_text(row.get("stem")) or "",
                        "preceding_part": normalize_gold_text(row.get("preceding_part")) or "",
                        "following_part": normalize_gold_text(row.get("following_part")) or "",
                        "expected_morphtok": expected,
                        "predicted_morphtok": predicted,
                        "morphscore_result": "skipped",
                        "morphscore_detail": detail_label,
                        **analysis_metadata(analysis),
                    }
                )
            continue

        weight = get_float(row, "word_freq_norm", 1.0) if freq_scale else 1.0
        weights.append(weight)
        recall_points.append(recall_pt * weight)
        precision_points.append(precision_pt * weight)
        recall_points_unweighted.append(recall_pt)
        precision_points_unweighted.append(precision_pt)

        if recall_pt == correct_point:
            correct_full += 1
            label = "correct"
        elif recall_pt == partial_point:
            partial += 1
            label = "partial"
        else:
            wrong += 1
            label = "wrong"

        detail_label = classify_v2_detail(morphemes, tokens, analysis)

        if include_all_rows or tokens != morphemes:
            wrong_rows.append(
                {
                    "wordform": word,
                    "lemma": normalize_gold_text(row.get("lemma")) or "",
                    "stem": normalize_gold_text(row.get("stem")) or "",
                    "preceding_part": normalize_gold_text(row.get("preceding_part")) or "",
                    "following_part": normalize_gold_text(row.get("following_part")) or "",
                    "expected_morphtok": expected,
                    "predicted_morphtok": predicted,
                    "morphscore_result": label,
                    "morphscore_detail": detail_label,
                    **analysis_metadata(analysis),
                }
            )

    total_weight = sum(weights)
    morphscore_recall = sum(recall_points) / total_weight if total_weight else 0.0
    morphscore_precision = sum(precision_points) / total_weight if total_weight else 0.0
    morphscore_recall_unweighted = mean_or_zero(recall_points_unweighted)
    morphscore_precision_unweighted = mean_or_zero(precision_points_unweighted)

    results = {
        "morphscore_recall": morphscore_recall,
        "morphscore_precision": morphscore_precision,
        "morphscore_f1": f1_or_zero(morphscore_precision, morphscore_recall),
        "morphscore_recall_unweighted": morphscore_recall_unweighted,
        "morphscore_precision_unweighted": morphscore_precision_unweighted,
        "morphscore_f1_unweighted": f1_or_zero(
            morphscore_precision_unweighted,
            morphscore_recall_unweighted,
        ),
        "morphscore_recall_std": std_or_zero(recall_points_unweighted),
        "morphscore_precision_std": std_or_zero(precision_points_unweighted),
        "num_samples": float(len(weights)),
        "mean_token_char_ratio": mean_or_zero(token_char_ratios),
        "correct": float(correct_full),
        "partial": float(partial),
        "wrong": float(wrong),
        "skipped": float(skipped),
    }
    return results, wrong_rows


def collect_v1_special_label_counts(rows: list[dict[str, str]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        label = row.get("morphscore_result", "")
        if label in {
            "adj_def_boundary",
            "adj_gender",
            "plu1person_boundary",
            "verb_ending_boundary",
        }:
            counts[label] += 1
    return counts


def print_v1_special_label_summary(rows: list[dict[str, str]]) -> None:
    counts = collect_v1_special_label_counts(rows)
    print("V1 special disagreement labels:")
    print(f"  adj_def_boundary: {counts.get('adj_def_boundary', 0)}")
    print(f"  adj_gender: {counts.get('adj_gender', 0)}")
    print(f"  verb: plu1person_boundary: {counts.get('plu1person_boundary', 0)}")
    print(f"  verb: verb_ending_boundary: {counts.get('verb_ending_boundary', 0)}")


def collect_v2_detail_counts(rows: list[dict[str, str]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        label = row.get("morphscore_detail", "")
        if label in SPECIAL_DETAIL_LABELS:
            counts[label] += 1
    return counts


def print_v2_detail_summary(rows: list[dict[str, str]]) -> None:
    counts = collect_v2_detail_counts(rows)
    print("V2 detailed disagreement labels:")
    print(f"  adj_def_boundary: {counts.get('adj_def_boundary', 0)}")
    print(f"  adj_def_split: {counts.get('adj_def_split', 0)}")
    print(f"  adj_gender: {counts.get('adj_gender', 0)}")
    print(f"  verb: plu1person_boundary: {counts.get('plu1person_boundary', 0)}")


def is_adj_like_row_metadata(row: dict[str, str]) -> bool:
    upos = row.get("UPOS", "")
    feats = row.get("feats", "")
    if upos in {"ADJ", "DET", "PRON"}:
        return True
    return upos == "VERB" and "VerbForm=Part" in feats


def canonical_adj_boundary_shift(
    gold_suffix: str,
    pred_suffix: str,
) -> tuple[str, str] | None:
    if pred_suffix == "ите" and gold_suffix.endswith("те"):
        return ("те", "ите")
    if gold_suffix == "ата" and pred_suffix == "та":
        return ("та", "ата")
    if gold_suffix == "ото" and pred_suffix == "то":
        return ("то", "ото")
    return None


def collect_adj_boundary_shift_counts(
    rows: list[dict[str, str]],
) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()

    for row in rows:
        if row.get("morphscore_result") == "correct":
            continue
        if not is_adj_like_row_metadata(row):
            continue

        expected = row.get("expected_morphtok", "").split()
        predicted = row.get("predicted_morphtok", "").split()
        if len(expected) != 2 or len(predicted) != 2:
            continue
        if "".join(expected) != "".join(predicted):
            continue

        gold_suffix = expected[1]
        pred_suffix = predicted[1]
        if gold_suffix == pred_suffix:
            continue
        canonical = canonical_adj_boundary_shift(gold_suffix, pred_suffix)
        if canonical is not None:
            counts[canonical] += 1

    return counts


def print_adj_boundary_shift_summary(rows: list[dict[str, str]]) -> None:
    counts = collect_adj_boundary_shift_counts(rows)
    print("Adjectival boundary shifts:")
    for gold_suffix, pred_suffix in CANONICAL_ADJ_BOUNDARY_SHIFTS:
        count = counts.get((gold_suffix, pred_suffix), 0)
        print(f"  {gold_suffix} -> {pred_suffix}: {count}")


# ----------------------------
# output helpers
# ----------------------------
def default_data_url(version: str) -> str:
    return DEFAULT_DATA_URL_V2 if version == "v2" else DEFAULT_DATA_URL_V1


def default_wrong_output(version: str) -> Path:
    return DEFAULT_WRONG_OUTPUT_V2 if version == "v2" else DEFAULT_WRONG_OUTPUT_V1


def default_all_output(version: str) -> Path:
    return DEFAULT_ALL_OUTPUT_V2 if version == "v2" else DEFAULT_ALL_OUTPUT_V1


def write_wrong_rows_v1(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "full_word",
        "pt1",
        "rest",
        "expected_morphtok",
        "predicted_morphtok",
        "morphscore_result",
        "UPOS",
        "feats",
    ]
    with path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_wrong_rows_v2(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "wordform",
        "lemma",
        "stem",
        "preceding_part",
        "following_part",
        "expected_morphtok",
        "predicted_morphtok",
        "morphscore_result",
        "morphscore_detail",
        "UPOS",
        "feats",
    ]
    with path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute Bulgarian MorphScore using bg_morphtok.tokenize_sentence(). "
            "Use --morphscore-version v1 for the original pt1/rest evaluation "
            "(the Table 3 MorphScore from Arnett & Bergen, 2025) or "
            "--morphscore-version v2 for the newer stem/preceding_part/"
            "following_part evaluation."
        )
    )
    parser.add_argument(
        "--morphscore-version",
        choices=("v1", "v2"),
        default="v1",
        help="MorphScore evaluation version to use. Default: v1.",
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=None,
        help="Optional local copy of the MorphScore CSV.",
    )
    parser.add_argument(
        "--data-url",
        default=None,
        help=(
            "URL for the MorphScore CSV if --data-file is not provided. "
            "Defaults to the v1 Bulgarian CSV for --morphscore-version v1 and "
            "the v2 Bulgarian CSV for --morphscore-version v2."
        ),
    )
    parser.add_argument(
        "--morphtok-module",
        choices=MORPHTOK_MODULE_CHOICES,
        default=DEFAULT_MORPHTOK_MODULE,
        help=(
            "Tokenizer module used for scoring tokenisation predictions. "
            "Default: bg_morphtok."
        ),
    )
    parser.add_argument(
        "--lexicon",
        type=Path,
        default=None,
        help="Optional CHILDES-derived lexicon used for conservative base checks.",
    )
    parser.add_argument(
        "--bulstem-rules",
        default="stem-context-2",
        help="BulStem rule set or path to a BulStem rules file.",
    )
    parser.add_argument(
        "--bulstem-min-freq",
        type=int,
        default=2,
        help="BulStem minimum rule frequency.",
    )
    parser.add_argument(
        "--bulstem-left-context",
        type=int,
        default=2,
        help="BulStem left context size.",
    )
    parser.add_argument(
        "--download-models",
        action="store_true",
        help="Run classla.download('bg') before building the pipeline.",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Allow CLASSLA to use a GPU if available.",
    )
    parser.add_argument(
        "--show-sample",
        type=int,
        default=0,
        help="Print the first N MorphScore words and their bg_morphtok tokenisations.",
    )
    parser.add_argument(
        "--output-all",
        action="store_true",
        help=(
            "Write all evaluated rows to CSV, including rows labeled correct. "
            "By default only rows whose predicted morphtok differs from the gold "
            "are written."
        ),
    )
    parser.add_argument(
        "--wrong-output",
        type=Path,
        default=None,
        help=(
            "CSV output path. By default this writes only rows whose predicted "
            "morphtok differs from the MorphScore CSV; with --output-all it writes "
            "all evaluated rows. Defaults to morphtok_wrong*.csv without "
            "--output-all and morphtok_all*.csv with --output-all."
        ),
    )

    # v2 scoring/filtering flags. They do not affect v1 mode.
    parser.add_argument(
        "--unique-only",
        dest="unique_only",
        action="store_true",
        default=True,
        help="v2 only: keep only rows marked unique. Default: enabled.",
    )
    parser.add_argument(
        "--no-unique-only",
        dest="unique_only",
        action="store_false",
        help="v2 only: do not require unique rows.",
    )
    parser.add_argument(
        "--stem-eq-lemma",
        dest="stem_eq_lemma",
        action="store_true",
        default=True,
        help="v2 only: keep only rows where stem == lemma. Default: enabled.",
    )
    parser.add_argument(
        "--no-stem-eq-lemma",
        dest="stem_eq_lemma",
        action="store_false",
        help="v2 only: do not require stem == lemma.",
    )
    parser.add_argument(
        "--exclude-numbers",
        dest="exclude_numbers",
        action="store_true",
        default=True,
        help="v2 only: exclude rows whose wordform contains a digit. Default: enabled.",
    )
    parser.add_argument(
        "--no-exclude-numbers",
        dest="exclude_numbers",
        action="store_false",
        help="v2 only: keep rows whose wordform contains digits.",
    )
    parser.add_argument(
        "--freq-scale",
        dest="freq_scale",
        action="store_true",
        default=True,
        help="v2 only: weight scores by word_freq_norm. Default: enabled.",
    )
    parser.add_argument(
        "--no-freq-scale",
        dest="freq_scale",
        action="store_false",
        help="v2 only: disable frequency weighting.",
    )
    parser.add_argument(
        "--exclude-single-tok",
        action="store_true",
        default=False,
        help="v2 only: exclude single-token predictions from scoring. Default: disabled.",
    )
    parser.add_argument(
        "--exclude-single-morpheme",
        action="store_true",
        default=True,
        help="v2 only: exclude one-morpheme gold items from scoring. Default: enabled.",
    )
    parser.add_argument(
        "--include-single-morpheme",
        dest="exclude_single_morpheme",
        action="store_false",
        help="v2 only: include one-morpheme gold items in scoring.",
    )
    parser.add_argument(
        "--single-tok-point",
        type=float,
        default=1.0,
        help="v2 only: score assigned to included single-token items. Default: 1.0.",
    )
    parser.add_argument(
        "--correct-point",
        type=float,
        default=1.0,
        help="v2 only: recall score when all gold boundaries are found. Default: 1.0.",
    )
    parser.add_argument(
        "--partial-point",
        type=float,
        default=0.5,
        help="v2 only: recall score when exactly one of two gold boundaries is found. Default: 0.5.",
    )

    args = parser.parse_args(argv)
    if args.data_url is None:
        args.data_url = default_data_url(args.morphscore_version)
    if args.wrong_output is None:
        if args.output_all:
            args.wrong_output = default_all_output(args.morphscore_version)
        else:
            args.wrong_output = default_wrong_output(args.morphscore_version)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_morphtok(args.morphtok_module)
    if args.lexicon is None:
        args.lexicon = DEFAULT_LEXICON_PATH

    try:
        stemmer = build_stemmer(
            rules=args.bulstem_rules,
            min_freq=args.bulstem_min_freq,
            left_context=args.bulstem_left_context,
        )
    except Exception as exc:
        print(f"Could not initialize BulStem: {exc}", file=sys.stderr)
        return 1

    try:
        pipeline = build_pipeline(
            download_models=args.download_models,
            use_gpu=args.use_gpu,
        )
    except Exception as exc:
        print(f"Could not initialize CLASSLA: {exc}", file=sys.stderr)
        return 1

    lexicon = PosLexicon.from_file(args.lexicon)

    try:
        rows = load_rows(args.data_file, args.data_url)
    except Exception as exc:
        print(f"Could not load MorphScore data: {exc}", file=sys.stderr)
        return 1

    cache: dict[str, list[str]] = {}
    analysis_cache: dict[str, TokenAnalysis | None] = {}

    def my_tokenizer(word: str) -> list[str]:
        cached = cache.get(word)
        if cached is not None:
            return cached

        tokenized = tokenize_sentence(word, pipeline, stemmer, lexicon).split()
        cache[word] = tokenized or [word.lower()]
        return cache[word]

    def analyze_word(word: str) -> TokenAnalysis | None:
        cached = analysis_cache.get(word)
        if word in analysis_cache:
            return cached

        analyses = analyses_from_doc(pipeline(word))
        analysis = analyses[0] if len(analyses) == 1 else None
        analysis_cache[word] = analysis
        return analysis

    if args.morphscore_version == "v1":
        (
            score,
            attempted,
            correct,
            wrong,
            skipped_single_token,
            total_assessed,
            wrong_rows,
        ) = get_morphscore_v1(
            rows,
            my_tokenizer,
            analyzer=analyze_word,
            include_all_rows=args.output_all,
        )
        write_wrong_rows_v1(args.wrong_output, wrong_rows)
        total_accuracy = correct / total_assessed if total_assessed else 0.0
        csv_label = "CSV rows" if args.output_all else "CSV mismatches"

        print(f"V1 MorphScore (Table 3; boundary accuracy, not F1): {score:.6f}")
        print("V1 MorphScore formula: correct / (correct + wrong)")
        print(
            "V1 total accuracy (includes skipped single-token predictions): "
            f"{total_accuracy:.6f}"
        )
        print("V1 total-accuracy formula: correct / total_assessed")
        print(f"Words tokenized: {len(cache)}")
        print(f"Total assessed words: {total_assessed}")
        print(f"Scored words: {attempted}")
        print(f"Correct: {correct}")
        print(f"Wrong: {wrong}")
        print(f"Skipped single-token predictions: {skipped_single_token}")
        print(f"{csv_label}: {len(wrong_rows)}")
        print_v1_special_label_summary(wrong_rows)
        print_adj_boundary_shift_summary(wrong_rows)
        print(f"Wrote: {args.wrong_output}")
    else:
        rows = filter_v2_rows(
            rows,
            unique_only=args.unique_only,
            stem_eq_lemma=args.stem_eq_lemma,
            exclude_numbers=args.exclude_numbers,
        )

        results, wrong_rows = get_morphscore_v2(
            rows,
            my_tokenizer,
            analyzer=analyze_word,
            include_all_rows=args.output_all,
            freq_scale=args.freq_scale,
            exclude_single_tok=args.exclude_single_tok,
            exclude_single_morpheme=args.exclude_single_morpheme,
            single_tok_point=args.single_tok_point,
            correct_point=args.correct_point,
            partial_point=args.partial_point,
        )
        write_wrong_rows_v2(args.wrong_output, wrong_rows)
        csv_label = "CSV rows" if args.output_all else "CSV mismatches"

        print(f"MorphScore2 recall (weighted): {results['morphscore_recall']:.6f}")
        print(f"MorphScore2 precision (weighted): {results['morphscore_precision']:.6f}")
        print(f"MorphScore2 F1 (weighted): {results['morphscore_f1']:.6f}")
        print(
            f"MorphScore2 recall (unweighted): "
            f"{results['morphscore_recall_unweighted']:.6f}"
        )
        print(
            f"MorphScore2 precision (unweighted): "
            f"{results['morphscore_precision_unweighted']:.6f}"
        )
        print(
            f"MorphScore2 F1 (unweighted): "
            f"{results['morphscore_f1_unweighted']:.6f}"
        )
        print(f"MorphScore2 recall std: {results['morphscore_recall_std']:.6f}")
        print(f"MorphScore2 precision std: {results['morphscore_precision_std']:.6f}")
        print(f"Words tokenized: {len(cache)}")
        print(f"Scored words: {int(results['num_samples'])}")
        print(f"Correct: {int(results['correct'])}")
        print(f"Partial: {int(results['partial'])}")
        print(f"Wrong: {int(results['wrong'])}")
        print(f"Skipped: {int(results['skipped'])}")
        print(f"Mean token/char ratio: {results['mean_token_char_ratio']:.6f}")
        print(f"{csv_label}: {len(wrong_rows)}")
        print_v2_detail_summary(wrong_rows)
        print_adj_boundary_shift_summary(wrong_rows)
        print(f"Wrote: {args.wrong_output}")

    if args.show_sample > 0:
        print("\nSample tokenizations:")
        seen_words: set[str] = set()
        shown = 0
        for row in rows:
            word = get_word_v1(row) if args.morphscore_version == "v1" else get_word_v2(row)
            if word in seen_words:
                continue
            seen_words.add(word)
            print(f"{word}\t{' '.join(my_tokenizer(word))}")
            shown += 1
            if shown >= args.show_sample:
                break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
