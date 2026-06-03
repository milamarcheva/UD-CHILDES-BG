from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "bg_morphtok.py"
CASES_PATH = Path(__file__).resolve().parent / "data" / "bg_morphtok_cases.csv"


def load_bg_morphtok_module():
    spec = importlib.util.spec_from_file_location("bg_morphtok", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_cases() -> list[dict[str, str]]:
    with CASES_PATH.open("r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile, skipinitialspace=True)
        return [row for row in reader if row.get("source")]


CASES = load_cases()


class DummyStemmer:
    def __init__(self, mapping: dict[str, str]):
        self.mapping = mapping

    def stem(self, text: str) -> str:
        return self.mapping.get(text, text)


def case_id(row: dict[str, str]) -> str:
    return row["source"]


def build_analysis(bg_morphtok_module, row: dict[str, str]):
    return bg_morphtok_module.TokenAnalysis(
        text=row["source"],
        lemma=row["lemma"],
        upos=row["upos"],
        feats=bg_morphtok_module.parse_feats(row["feats"]),
    )


def build_case_stemmer(row: dict[str, str]) -> DummyStemmer:
    source = row["source"]
    stem = row["stem"]
    mapping = {source: stem}
    for prefix in ("по-", "най-"):
        if source.startswith(prefix):
            source = source[len(prefix):]
            mapping[source] = stem
            break

    suffix = suffix_parts(row)
    for index in range(1, len(suffix)):
        stripped_suffix = "".join(suffix[-index:])
        mapping[source[: -len(stripped_suffix)]] = stem
    return DummyStemmer(mapping)


def expected_base(row: dict[str, str], *, base_form: str) -> str:
    if (
        base_form == "lemma-nominalonly"
        and row["upos"] in {"ADJ", "NOUN", "PROPN", "NUM"}
        and (row.get("prefix", "").strip() or row.get("suffix", "").strip())
    ):
        return row["lemma"]
    return row["stem"]


def suffix_parts(row: dict[str, str]) -> list[str]:
    suffix = row.get("suffix", "").strip()
    if not suffix:
        return []
    return [part for part in suffix.split("|||") if part]


def expected_output(
    bg_morphtok_module,
    row: dict[str, str],
    *,
    base_form: str = "stem",
    separate_morphemes: bool = False,
) -> str:
    parts: list[str] = []
    prefix = row.get("prefix", "").strip()
    if prefix:
        parts.append(prefix)

    parts.append(expected_base(row, base_form=base_form))
    suffix = suffix_parts(row)
    if separate_morphemes:
        parts.extend(suffix)
    elif suffix:
        parts.append("".join(suffix))

    return " ".join(bg_morphtok_module.normalize(part) for part in parts if part)


def segment_case(
    bg_morphtok_module,
    row: dict[str, str],
    lexicon,
    *,
    base_form: str = "stem",
    separate_morphemes: bool = False,
) -> str:
    analysis = build_analysis(bg_morphtok_module, row)
    stemmer = build_case_stemmer(row)
    parts = bg_morphtok_module.segment_analysis(
        analysis,
        stemmer,
        lexicon,
        base_form=base_form,
        separate_morphemes=separate_morphemes,
    )
    return " ".join(bg_morphtok_module.normalize(part) for part in parts if part)


@pytest.fixture(scope="session")
def bg_morphtok_module():
    return load_bg_morphtok_module()


@pytest.fixture(scope="session")
def default_lexicon(bg_morphtok_module):
    return bg_morphtok_module.PosLexicon.from_file(
        bg_morphtok_module.DEFAULT_LEXICON_PATH
    )


@pytest.mark.parametrize("row", CASES, ids=case_id)
def test_bg_morphtok_cases_standard(bg_morphtok_module, default_lexicon, row: dict[str, str]):
    actual = segment_case(bg_morphtok_module, row, default_lexicon)
    assert actual == expected_output(bg_morphtok_module, row)


@pytest.mark.parametrize("row", CASES, ids=case_id)
def test_bg_morphtok_cases_lemma_nominalonly(
    bg_morphtok_module,
    default_lexicon,
    row: dict[str, str],
):
    actual = segment_case(
        bg_morphtok_module,
        row,
        default_lexicon,
        base_form="lemma-nominalonly",
    )
    assert actual == expected_output(
        bg_morphtok_module,
        row,
        base_form="lemma-nominalonly",
    )


@pytest.mark.parametrize("row", CASES, ids=case_id)
def test_bg_morphtok_cases_separate_morphemes(
    bg_morphtok_module,
    default_lexicon,
    row: dict[str, str],
):
    actual = segment_case(
        bg_morphtok_module,
        row,
        default_lexicon,
        separate_morphemes=True,
    )
    assert actual == expected_output(
        bg_morphtok_module,
        row,
        separate_morphemes=True,
    )


def test_segment_analysis_uses_surface_stem_by_default(bg_morphtok_module):
    analysis = bg_morphtok_module.TokenAnalysis(
        text="годинки",
        lemma="годинка",
        upos="NOUN",
        feats={"Number": "Plur"},
    )
    stemmer = DummyStemmer({"годинки": "годинк"})

    actual = bg_morphtok_module.segment_analysis(
        analysis,
        stemmer,
        bg_morphtok_module.PosLexicon(),
    )

    assert actual == ["годинк", "и"]


def test_segment_analysis_can_replace_stem_with_lemma(bg_morphtok_module):
    analysis = bg_morphtok_module.TokenAnalysis(
        text="годинки",
        lemma="годинка",
        upos="NOUN",
        feats={"Number": "Plur"},
    )
    stemmer = DummyStemmer({"годинки": "годинк"})

    actual = bg_morphtok_module.segment_analysis(
        analysis,
        stemmer,
        bg_morphtok_module.PosLexicon(),
        base_form="lemma",
    )

    assert actual == ["годинка", "и"]


def test_process_conllu_lines_uses_conllu_lemma(bg_morphtok_module):
    conllu_lines = [
        "# sent_id = 1",
        "# text = годинки",
        "1\tгодинки\tгодинка\tNOUN\t_\tNumber=Plur\t0\troot\t_\t_",
        "",
    ]
    stemmer = DummyStemmer({"годинки": "годинк"})

    actual = bg_morphtok_module.process_conllu_lines(
        conllu_lines,
        stemmer,
        bg_morphtok_module.PosLexicon(),
        base_form="lemma",
    )

    assert actual == ["годинка и"]


def test_nominal_only_mode_keeps_verb_base_as_stem(bg_morphtok_module):
    analysis = bg_morphtok_module.TokenAnalysis(
        text="правиш",
        lemma="правя",
        upos="VERB",
        feats={"VerbForm": "Fin", "Person": "2", "Number": "Sing", "Mood": "Ind"},
    )
    stemmer = DummyStemmer({"правиш": "прави"})

    actual = bg_morphtok_module.segment_analysis(
        analysis,
        stemmer,
        bg_morphtok_module.PosLexicon(),
        base_form="lemma-nominalonly",
    )

    assert actual == ["прави", "ш"]


def test_replace_base_with_lemma_allows_propn_in_nominal_only_mode(bg_morphtok_module):
    analysis = bg_morphtok_module.TokenAnalysis(
        text="Сашето",
        lemma="Саше",
        upos="PROPN",
        feats={"Definite": "Def"},
    )

    actual = bg_morphtok_module.replace_base_with_lemma(
        analysis,
        ("Саше", "то"),
        base_form="lemma-nominalonly",
    )

    assert actual == ["Саше", "то"]


def test_segment_analysis_splits_definite_propn_article(bg_morphtok_module):
    analysis = bg_morphtok_module.TokenAnalysis(
        text="Сашето",
        lemma="Саше",
        upos="PROPN",
        feats={"Definite": "Def"},
    )
    stemmer = DummyStemmer({"Сашето": "саш"})

    actual = bg_morphtok_module.segment_analysis(
        analysis,
        stemmer,
        bg_morphtok_module.PosLexicon(),
        base_form="lemma-nominalonly",
    )

    assert actual == ["Саше", "то"]


def test_segment_analysis_keeps_nondefinite_propn_unsplit(bg_morphtok_module):
    analysis = bg_morphtok_module.TokenAnalysis(
        text="Саше",
        lemma="Саше",
        upos="PROPN",
        feats={},
    )
    stemmer = DummyStemmer({"Саше": "саш"})

    actual = bg_morphtok_module.segment_analysis(
        analysis,
        stemmer,
        bg_morphtok_module.PosLexicon(),
        base_form="lemma-nominalonly",
    )

    assert actual == ["Саше"]


def test_replace_base_with_lemma_allows_num_in_nominal_only_mode(bg_morphtok_module):
    analysis = bg_morphtok_module.TokenAnalysis(
        text="двете",
        lemma="два",
        upos="NUM",
        feats={"Definite": "Def"},
    )

    actual = bg_morphtok_module.replace_base_with_lemma(
        analysis,
        ("две", "те"),
        base_form="lemma-nominalonly",
    )

    assert actual == ["два", "те"]


def test_segment_analysis_splits_definite_num_article(bg_morphtok_module):
    analysis = bg_morphtok_module.TokenAnalysis(
        text="двете",
        lemma="два",
        upos="NUM",
        feats={"Definite": "Def"},
    )
    stemmer = DummyStemmer({"двете": "две"})

    actual = bg_morphtok_module.segment_analysis(
        analysis,
        stemmer,
        bg_morphtok_module.PosLexicon(),
        base_form="lemma-nominalonly",
    )

    assert actual == ["два", "те"]


def test_segment_analysis_keeps_nondefinite_num_unsplit(bg_morphtok_module):
    analysis = bg_morphtok_module.TokenAnalysis(
        text="две",
        lemma="два",
        upos="NUM",
        feats={},
    )
    stemmer = DummyStemmer({"две": "две"})

    actual = bg_morphtok_module.segment_analysis(
        analysis,
        stemmer,
        bg_morphtok_module.PosLexicon(),
        base_form="lemma-nominalonly",
    )

    assert actual == ["две"]


def test_segment_analysis_splits_count_noun_suffix(bg_morphtok_module):
    analysis = bg_morphtok_module.TokenAnalysis(
        text="шанса",
        lemma="шанс",
        upos="NOUN",
        feats={"Number": "Count"},
    )
    stemmer = DummyStemmer({"шанса": "шанс"})

    actual = bg_morphtok_module.segment_analysis(
        analysis,
        stemmer,
        bg_morphtok_module.PosLexicon(),
    )

    assert actual == ["шанс", "а"]


@pytest.mark.parametrize(
    ("text", "lemma", "stem"),
    [
        ("деца", "дете", "дет"),
        ("очи", "око", "оч"),
        ("ръце", "ръка", "рък"),
        ("дървета", "дърво", "дърв"),
        ("цветя", "цвете", "цвет"),
    ],
)
def test_segment_analysis_keeps_irregular_plural_nouns_unsplit(
    bg_morphtok_module,
    text: str,
    lemma: str,
    stem: str,
):
    analysis = bg_morphtok_module.TokenAnalysis(
        text=text,
        lemma=lemma,
        upos="NOUN",
        feats={"Number": "Plur"},
    )
    stemmer = DummyStemmer({text: stem})

    actual = bg_morphtok_module.segment_analysis(
        analysis,
        stemmer,
        bg_morphtok_module.PosLexicon(),
    )

    assert actual == [text]


@pytest.mark.parametrize(
    ("text", "lemma", "stem", "article", "expected_base"),
    [
        ("децата", "дете", "дет", "та", "деца"),
        ("очите", "око", "оч", "те", "очи"),
        ("ръцете", "ръка", "рък", "те", "ръце"),
        ("дърветата", "дърво", "дърв", "та", "дървета"),
        ("цветята", "цвете", "цвет", "та", "цветя"),
    ],
)
@pytest.mark.parametrize("separate_morphemes", [False, True])
def test_segment_analysis_splits_definite_irregular_plural_articles(
    bg_morphtok_module,
    text: str,
    lemma: str,
    stem: str,
    article: str,
    expected_base: str,
    separate_morphemes: bool,
):
    analysis = bg_morphtok_module.TokenAnalysis(
        text=text,
        lemma=lemma,
        upos="NOUN",
        feats={"Definite": "Def", "Number": "Plur"},
    )
    stemmer = DummyStemmer({text: stem, expected_base: stem})

    actual = bg_morphtok_module.segment_analysis(
        analysis,
        stemmer,
        bg_morphtok_module.PosLexicon(),
        separate_morphemes=separate_morphemes,
    )

    assert actual == [expected_base, article]


def test_separate_morphemes_splits_plural_and_article_layers(bg_morphtok_module):
    analysis = bg_morphtok_module.TokenAnalysis(
        text="хубавите",
        lemma="хубав",
        upos="ADJ",
        feats={"Definite": "Def", "Number": "Plur"},
    )
    stemmer = DummyStemmer({"хубавите": "хубав"})

    actual = bg_morphtok_module.segment_analysis(
        analysis,
        stemmer,
        bg_morphtok_module.PosLexicon(),
        separate_morphemes=True,
    )

    assert actual == ["хубав", "и", "те"]


def test_separate_morphemes_keeps_long_definite_article_together(bg_morphtok_module):
    analysis = bg_morphtok_module.TokenAnalysis(
        text="ученият",
        lemma="учен",
        upos="ADJ",
        feats={"Definite": "Def", "Gender": "Masc", "Number": "Sing"},
    )
    stemmer = DummyStemmer({"ученият": "учен"})

    actual = bg_morphtok_module.segment_analysis(
        analysis,
        stemmer,
        bg_morphtok_module.PosLexicon(),
        separate_morphemes=True,
    )

    assert actual == ["учен", "ият"]


def test_separate_morphemes_keeps_ochite_as_ochi_te(bg_morphtok_module):
    analysis = bg_morphtok_module.TokenAnalysis(
        text="очите",
        lemma="око",
        upos="NOUN",
        feats={"Definite": "Def", "Number": "Plur"},
    )
    stemmer = DummyStemmer({"очите": "оч", "очи": "оч"})

    actual = bg_morphtok_module.segment_analysis(
        analysis,
        stemmer,
        bg_morphtok_module.PosLexicon(),
        separate_morphemes=True,
    )

    assert actual == ["очи", "те"]


def test_separate_morphemes_does_not_change_verbs(bg_morphtok_module):
    analysis = bg_morphtok_module.TokenAnalysis(
        text="правиш",
        lemma="правя",
        upos="VERB",
        feats={"VerbForm": "Fin", "Person": "2", "Number": "Sing", "Mood": "Ind"},
    )
    stemmer = DummyStemmer({"правиш": "прави"})

    actual = bg_morphtok_module.segment_analysis(
        analysis,
        stemmer,
        bg_morphtok_module.PosLexicon(),
        separate_morphemes=True,
    )

    assert actual == ["прави", "ш"]


@pytest.mark.parametrize(
    ("text", "lemma", "stem"),
    [
        ("направят", "направя", "направя"),
        ("похарчат", "похарча", "похарча"),
        ("въведат", "въведа", "въведа"),
    ],
)
def test_segment_analysis_allows_bare_t_for_3pl_finite_verbs(
    bg_morphtok_module,
    text: str,
    lemma: str,
    stem: str,
):
    analysis = bg_morphtok_module.TokenAnalysis(
        text=text,
        lemma=lemma,
        upos="VERB",
        feats={
            "Mood": "Ind",
            "Number": "Plur",
            "Person": "3",
            "Tense": "Pres",
            "VerbForm": "Fin",
        },
    )
    stemmer = DummyStemmer({text: stem})

    actual = bg_morphtok_module.segment_analysis(
        analysis,
        stemmer,
        bg_morphtok_module.PosLexicon(),
    )

    assert actual == [stem, "т"]
