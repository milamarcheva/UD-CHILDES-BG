from __future__ import annotations

import importlib.util
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "morphscore.py"


def load_morphscore_module():
    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    spec = importlib.util.spec_from_file_location("morphscore", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_get_morphscore_v1_lowercases_word_and_gold():
    morphscore = load_morphscore_module()
    seen_words: list[str] = []

    def tokenizer(word: str) -> list[str]:
        seen_words.append(word)
        return ["годинк", "и"]

    score, attempted, correct, wrong, skipped, total, wrong_rows = morphscore.get_morphscore_v1(
        [
            {
                "full_word": "ГОДИНКИ",
                "pt1": "ГОДИНК",
                "rest": "И",
            }
        ],
        tokenizer,
    )

    assert seen_words == ["годинки"]
    assert score == 1.0
    assert attempted == 1
    assert correct == 1
    assert wrong == 0
    assert skipped == 0
    assert total == 1
    assert wrong_rows == []


def test_get_morphscore_v1_can_include_correct_rows_in_output():
    morphscore = load_morphscore_module()

    score, attempted, correct, wrong, skipped, total, output_rows = morphscore.get_morphscore_v1(
        [
            {
                "full_word": "ГОДИНКИ",
                "pt1": "ГОДИНК",
                "rest": "И",
            }
        ],
        lambda word: ["годинк", "и"],
        include_all_rows=True,
    )

    assert score == 1.0
    assert attempted == 1
    assert correct == 1
    assert wrong == 0
    assert skipped == 0
    assert total == 1
    assert output_rows == [
        {
            "full_word": "годинки",
            "pt1": "годинк",
            "rest": "и",
            "expected_morphtok": "годинк и",
            "predicted_morphtok": "годинк и",
            "morphscore_result": "correct",
            "UPOS": "",
            "feats": "",
        }
    ]


def test_build_gold_morphemes_v2_lowercases_gold_parts():
    morphscore = load_morphscore_module()

    actual = morphscore.build_gold_morphemes_v2(
        {
            "preceding_part": "НАЙ-",
            "stem": "ХУБАВ",
            "following_part": "ИТЕ",
        }
    )

    assert actual == ["най-", "хубав", "ите"]


def test_get_morphscore_v2_lowercases_word_and_wrong_row_gold():
    morphscore = load_morphscore_module()
    seen_words: list[str] = []

    def tokenizer(word: str) -> list[str]:
        seen_words.append(word)
        return ["сладк", "ите"]

    results, wrong_rows = morphscore.get_morphscore_v2(
        [
            {
                "wordform": "СЛАДКИТЕ",
                "lemma": "СЛАДЪК",
                "stem": "СЛАДК",
                "preceding_part": "",
                "following_part": "ИТЕ",
                "word_freq_norm": "1.0",
            }
        ],
        tokenizer,
        freq_scale=True,
        exclude_single_tok=False,
        exclude_single_morpheme=False,
        single_tok_point=1.0,
        correct_point=1.0,
        partial_point=0.5,
    )

    assert seen_words == ["сладките"]
    assert results["wrong"] == 0.0
    assert results["correct"] == 1.0
    assert wrong_rows == []


def test_get_morphscore_v2_lowercases_wrong_row_gold_fields():
    morphscore = load_morphscore_module()
    analysis = morphscore.TokenAnalysis(
        text="сладките",
        lemma="сладък",
        upos="ADJ",
        feats={"Definite": "Def", "Number": "Plur"},
    )

    results, wrong_rows = morphscore.get_morphscore_v2(
        [
            {
                "wordform": "СЛАДКИТЕ",
                "lemma": "СЛАДЪК",
                "stem": "СЛАДК",
                "preceding_part": "",
                "following_part": "ИТЕ",
                "word_freq_norm": "1.0",
            }
        ],
        lambda word: ["сладк", "и", "те"],
        analyzer=lambda word: analysis,
        freq_scale=True,
        exclude_single_tok=False,
        exclude_single_morpheme=False,
        single_tok_point=1.0,
        correct_point=1.0,
        partial_point=0.5,
    )

    assert results["correct"] == 1.0
    assert wrong_rows == [
        {
            "wordform": "сладките",
            "lemma": "сладък",
            "stem": "сладк",
            "preceding_part": "",
            "following_part": "ите",
            "expected_morphtok": "сладк ите",
            "predicted_morphtok": "сладк и те",
            "morphscore_result": "correct",
            "morphscore_detail": "adj_def_split",
            "UPOS": "ADJ",
            "feats": "Definite=Def|Number=Plur",
        }
    ]


def test_get_morphscore_v2_can_include_correct_rows_in_output():
    morphscore = load_morphscore_module()
    analysis = morphscore.TokenAnalysis(
        text="сладките",
        lemma="сладък",
        upos="ADJ",
        feats={"Definite": "Def", "Number": "Plur"},
    )

    results, output_rows = morphscore.get_morphscore_v2(
        [
            {
                "wordform": "СЛАДКИТЕ",
                "lemma": "СЛАДЪК",
                "stem": "СЛАДК",
                "preceding_part": "",
                "following_part": "ИТЕ",
                "word_freq_norm": "1.0",
            }
        ],
        lambda word: ["сладк", "ите"],
        analyzer=lambda word: analysis,
        include_all_rows=True,
        freq_scale=True,
        exclude_single_tok=False,
        exclude_single_morpheme=False,
        single_tok_point=1.0,
        correct_point=1.0,
        partial_point=0.5,
    )

    assert results["correct"] == 1.0
    assert output_rows == [
        {
            "wordform": "сладките",
            "lemma": "сладък",
            "stem": "сладк",
            "preceding_part": "",
            "following_part": "ите",
            "expected_morphtok": "сладк ите",
            "predicted_morphtok": "сладк ите",
            "morphscore_result": "correct",
            "morphscore_detail": "",
            "UPOS": "ADJ",
            "feats": "Definite=Def|Number=Plur",
        }
    ]


def test_filter_v2_rows_compares_stem_and_lemma_case_insensitively():
    morphscore = load_morphscore_module()

    actual = morphscore.filter_v2_rows(
        [
            {
                "wordform": "СЛАДКИТЕ",
                "stem": "СЛАДЪК",
                "lemma": "сладък",
                "unique": "unique",
            }
        ],
        unique_only=True,
        stem_eq_lemma=True,
        exclude_numbers=True,
    )

    assert len(actual) == 1


def test_parse_args_uses_all_output_default_path_when_requested():
    morphscore = load_morphscore_module()

    args = morphscore.parse_args(["--morphscore-version", "v2", "--output-all"])

    assert args.output_all is True
    assert args.wrong_output == morphscore.DEFAULT_ALL_OUTPUT_V2


def test_get_morphscore_v2_labels_adj_def_boundary():
    morphscore = load_morphscore_module()
    analysis = morphscore.TokenAnalysis(
        text="правителственото",
        lemma="правителствен",
        upos="ADJ",
        feats={"Definite": "Def", "Gender": "Neut", "Number": "Sing"},
    )

    results, output_rows = morphscore.get_morphscore_v2(
        [
            {
                "wordform": "правителственото",
                "lemma": "правителствен",
                "stem": "правителствен",
                "preceding_part": "",
                "following_part": "ото",
                "word_freq_norm": "1.0",
            }
        ],
        lambda word: ["правителствено", "то"],
        analyzer=lambda word: analysis,
        include_all_rows=True,
        freq_scale=True,
        exclude_single_tok=False,
        exclude_single_morpheme=False,
        single_tok_point=1.0,
        correct_point=1.0,
        partial_point=0.5,
    )

    assert results["wrong"] == 1.0
    assert output_rows == [
        {
            "wordform": "правителственото",
            "lemma": "правителствен",
            "stem": "правителствен",
            "preceding_part": "",
            "following_part": "ото",
            "expected_morphtok": "правителствен ото",
            "predicted_morphtok": "правителствено то",
            "morphscore_result": "wrong",
            "morphscore_detail": "adj_def_boundary",
            "UPOS": "ADJ",
            "feats": "Definite=Def|Gender=Neut|Number=Sing",
        }
    ]


def test_get_morphscore_v2_labels_plu1person_boundary():
    morphscore = load_morphscore_module()
    analysis = morphscore.TokenAnalysis(
        text="избягаме",
        lemma="избягам",
        upos="VERB",
        feats={
            "Mood": "Ind",
            "Number": "Plur",
            "Person": "1",
            "Tense": "Pres",
            "VerbForm": "Fin",
        },
    )

    results, output_rows = morphscore.get_morphscore_v2(
        [
            {
                "wordform": "избягаме",
                "lemma": "избягам",
                "stem": "избягам",
                "preceding_part": "",
                "following_part": "е",
                "word_freq_norm": "1.0",
            }
        ],
        lambda word: ["избяга", "ме"],
        analyzer=lambda word: analysis,
        include_all_rows=True,
        freq_scale=True,
        exclude_single_tok=False,
        exclude_single_morpheme=False,
        single_tok_point=1.0,
        correct_point=1.0,
        partial_point=0.5,
    )

    assert results["wrong"] == 1.0
    assert output_rows == [
        {
            "wordform": "избягаме",
            "lemma": "избягам",
            "stem": "избягам",
            "preceding_part": "",
            "following_part": "е",
            "expected_morphtok": "избягам е",
            "predicted_morphtok": "избяга ме",
            "morphscore_result": "wrong",
            "morphscore_detail": "plu1person_boundary",
            "UPOS": "VERB",
            "feats": "Mood=Ind|Number=Plur|Person=1|Tense=Pres|VerbForm=Fin",
        }
    ]


def test_collect_v2_detail_counts_and_print_summary():
    morphscore = load_morphscore_module()
    rows = [
        {"morphscore_result": "wrong", "morphscore_detail": "adj_def_boundary"},
        {"morphscore_result": "correct", "morphscore_detail": "adj_def_split"},
        {"morphscore_result": "correct", "morphscore_detail": "adj_def_split"},
        {"morphscore_result": "correct", "morphscore_detail": "adj_gender"},
        {"morphscore_result": "wrong", "morphscore_detail": "plu1person_boundary"},
        {"morphscore_result": "wrong", "morphscore_detail": ""},
    ]

    counts = morphscore.collect_v2_detail_counts(rows)
    assert dict(counts) == {
        "adj_def_boundary": 1,
        "adj_def_split": 2,
        "adj_gender": 1,
        "plu1person_boundary": 1,
    }

    buffer = StringIO()
    with redirect_stdout(buffer):
        morphscore.print_v2_detail_summary(rows)

    assert buffer.getvalue().strip().splitlines() == [
        "V2 detailed disagreement labels:",
        "  adj_def_boundary: 1",
        "  adj_def_split: 2",
        "  adj_gender: 1",
        "  verb: plu1person_boundary: 1",
    ]


def test_get_morphscore_v1_labels_adj_def_boundary():
    morphscore = load_morphscore_module()

    analysis = morphscore.TokenAnalysis(
        text="правителственото",
        lemma="правителствен",
        upos="ADJ",
        feats={"Definite": "Def", "Gender": "Neut", "Number": "Sing"},
    )
    score, attempted, correct, wrong, skipped, total, wrong_rows = morphscore.get_morphscore_v1(
        [
            {
                "full_word": "правителственото",
                "pt1": "правителствен",
                "rest": "ото",
            }
        ],
        lambda word: ["правителствено", "то"],
        analyzer=lambda word: analysis,
    )

    assert score == 0.0
    assert attempted == 1
    assert correct == 0
    assert wrong == 1
    assert skipped == 0
    assert total == 1
    assert wrong_rows == [
        {
            "full_word": "правителственото",
            "pt1": "правителствен",
            "rest": "ото",
            "expected_morphtok": "правителствен ото",
            "predicted_morphtok": "правителствено то",
            "morphscore_result": "adj_def_boundary",
            "UPOS": "ADJ",
            "feats": "Definite=Def|Gender=Neut|Number=Sing",
        }
    ]


def test_get_morphscore_v1_labels_adj_gender():
    morphscore = load_morphscore_module()

    analysis = morphscore.TokenAnalysis(
        text="стара",
        lemma="стар",
        upos="ADJ",
        feats={"Gender": "Fem", "Number": "Sing"},
    )
    score, attempted, correct, wrong, skipped, total, wrong_rows = morphscore.get_morphscore_v1(
        [
            {
                "full_word": "стара",
                "pt1": "стар",
                "rest": "а",
            }
        ],
        lambda word: ["стара"],
        analyzer=lambda word: analysis,
    )

    assert score == 0.0
    assert attempted == 0
    assert correct == 0
    assert wrong == 0
    assert skipped == 1
    assert total == 1
    assert wrong_rows == [
        {
            "full_word": "стара",
            "pt1": "стар",
            "rest": "а",
            "expected_morphtok": "стар а",
            "predicted_morphtok": "стара",
            "morphscore_result": "adj_gender",
            "UPOS": "ADJ",
            "feats": "Gender=Fem|Number=Sing",
        }
    ]


def test_get_morphscore_v1_labels_adj_gender_for_adv_o_case():
    morphscore = load_morphscore_module()

    analysis = morphscore.TokenAnalysis(
        text="изпълнимо",
        lemma="изпълнимо",
        upos="ADV",
        feats={"Degree": "Pos"},
    )
    score, attempted, correct, wrong, skipped, total, wrong_rows = morphscore.get_morphscore_v1(
        [
            {
                "full_word": "изпълнимо",
                "pt1": "изпълним",
                "rest": "о",
            }
        ],
        lambda word: ["изпълнимо"],
        analyzer=lambda word: analysis,
    )

    assert score == 0.0
    assert attempted == 0
    assert correct == 0
    assert wrong == 0
    assert skipped == 1
    assert total == 1
    assert wrong_rows == [
        {
            "full_word": "изпълнимо",
            "pt1": "изпълним",
            "rest": "о",
            "expected_morphtok": "изпълним о",
            "predicted_morphtok": "изпълнимо",
            "morphscore_result": "adj_gender",
            "UPOS": "ADV",
            "feats": "Degree=Pos",
        }
    ]


def test_get_morphscore_v1_labels_plu1person_boundary():
    morphscore = load_morphscore_module()

    analysis = morphscore.TokenAnalysis(
        text="избягаме",
        lemma="избягам",
        upos="VERB",
        feats={
            "Mood": "Ind",
            "Number": "Plur",
            "Person": "1",
            "Tense": "Pres",
            "VerbForm": "Fin",
        },
    )
    score, attempted, correct, wrong, skipped, total, wrong_rows = morphscore.get_morphscore_v1(
        [
            {
                "full_word": "избягаме",
                "pt1": "избягам",
                "rest": "е",
            }
        ],
        lambda word: ["избяга", "ме"],
        analyzer=lambda word: analysis,
    )

    assert score == 0.0
    assert attempted == 1
    assert correct == 0
    assert wrong == 1
    assert skipped == 0
    assert total == 1
    assert wrong_rows == [
        {
            "full_word": "избягаме",
            "pt1": "избягам",
            "rest": "е",
            "expected_morphtok": "избягам е",
            "predicted_morphtok": "избяга ме",
            "morphscore_result": "plu1person_boundary",
            "UPOS": "VERB",
            "feats": "Mood=Ind|Number=Plur|Person=1|Tense=Pres|VerbForm=Fin",
        }
    ]


def test_collect_v1_special_label_counts_and_print_summary():
    morphscore = load_morphscore_module()
    wrong_rows = [
        {"morphscore_result": "adj_def_boundary"},
        {"morphscore_result": "adj_def_boundary"},
        {"morphscore_result": "adj_gender"},
        {"morphscore_result": "plu1person_boundary"},
        {"morphscore_result": "wrong"},
    ]

    counts = morphscore.collect_v1_special_label_counts(wrong_rows)
    assert dict(counts) == {
        "adj_def_boundary": 2,
        "adj_gender": 1,
        "plu1person_boundary": 1,
    }

    buffer = StringIO()
    with redirect_stdout(buffer):
        morphscore.print_v1_special_label_summary(wrong_rows)

    assert buffer.getvalue().strip().splitlines() == [
        "V1 special disagreement labels:",
        "  adj_def_boundary: 2",
        "  adj_gender: 1",
        "  verb: plu1person_boundary: 1",
    ]


def test_collect_adj_boundary_shift_counts_and_print_summary():
    morphscore = load_morphscore_module()

    wrong_rows = [
        {
            "expected_morphtok": "селски те",
            "predicted_morphtok": "селск ите",
            "morphscore_result": "wrong",
            "UPOS": "ADJ",
            "feats": "Definite=Def|Number=Plur",
        },
        {
            "expected_morphtok": "цял ата",
            "predicted_morphtok": "цяла та",
            "morphscore_result": "wrong",
            "UPOS": "ADJ",
            "feats": "Definite=Def|Gender=Fem|Number=Sing",
        },
        {
            "expected_morphtok": "широк ото",
            "predicted_morphtok": "широко то",
            "morphscore_result": "wrong",
            "UPOS": "ADJ",
            "feats": "Definite=Def|Gender=Neut|Number=Sing",
        },
        {
            "expected_morphtok": "заглъхна лите",
            "predicted_morphtok": "заглъхнал ите",
            "morphscore_result": "wrong",
            "UPOS": "VERB",
            "feats": "Definite=Def|Number=Plur|VerbForm=Part",
        },
        {
            "expected_morphtok": "син ята",
            "predicted_morphtok": "синя та",
            "morphscore_result": "wrong",
            "UPOS": "ADJ",
            "feats": "Definite=Def|Gender=Fem|Number=Sing",
        },
        {
            "expected_morphtok": "дишам е",
            "predicted_morphtok": "диша ме",
            "morphscore_result": "wrong",
            "UPOS": "VERB",
            "feats": "Mood=Ind|Number=Plur|Person=1|Tense=Pres|VerbForm=Fin",
        },
    ]

    counts = morphscore.collect_adj_boundary_shift_counts(wrong_rows)
    assert dict(counts) == {
        ("те", "ите"): 2,
        ("ата", "та"): 2,
        ("ото", "то"): 1,
    }

    buffer = StringIO()
    with redirect_stdout(buffer):
        morphscore.print_adj_boundary_shift_summary(wrong_rows)

    assert buffer.getvalue().strip().splitlines() == [
        "Adjectival boundary shifts:",
        "  те -> ите: 2",
        "  ата -> та: 2",
        "  ото -> то: 1",
    ]
