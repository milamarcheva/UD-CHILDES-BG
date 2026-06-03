from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "morphemic_constituency_bg.py"


def load_module():
    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    spec = importlib.util.spec_from_file_location(
        "morphemic_constituency_bg",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DummyStemmer:
    def __init__(self, mapping: dict[str, str]):
        self.mapping = mapping

    def stem(self, text: str) -> str:
        return self.mapping.get(text, text)


def analysis(module, text: str, lemma: str, upos: str, feats: dict[str, str]):
    return module.bg_morphtok.TokenAnalysis(
        text=text,
        lemma=lemma,
        upos=upos,
        feats=feats,
    )


def test_rewrite_constituency_tree_handles_aux_verb_and_plural_definite_noun():
    module = load_module()
    tree = (
        "(ROOT (SQ (NOUNP (NOUN Баба)) "
        "(VERBP (VERBP (WHNP (DET какво)) (VERBP (AUX ще) (VERB направи))) "
        "(ADPP (ADP от) (NOUNP (NOUN гъбките))))))"
    )
    analyses = [
        analysis(module, "Баба", "баба", "NOUN", {"Number": "Sing"}),
        analysis(module, "какво", "какво", "DET", {"PronType": "Int"}),
        analysis(module, "ще", "ще", "AUX", {"VerbForm": "Fin"}),
        analysis(
            module,
            "направи",
            "направя",
            "VERB",
            {"Mood": "Ind", "Number": "Sing", "Person": "3", "Tense": "Past", "VerbForm": "Fin"},
        ),
        analysis(module, "от", "от", "ADP", {}),
        analysis(
            module,
            "гъбките",
            "гъбка",
            "NOUN",
            {"Definite": "Def", "Number": "Plur"},
        ),
    ]
    stemmer = DummyStemmer(
        {
            "Баба": "баба",
            "какво": "какво",
            "ще": "ще",
            "направи": "направ",
            "от": "от",
            "гъбките": "гъбк",
            "гъбки": "гъбк",
        }
    )

    actual = module.rewrite_constituency_tree(
        tree,
        analyses,
        stemmer,
        module.bg_morphtok.PosLexicon(),
    )

    expected = (
        "(ROOT (SQ (NOUNP (NOUN баба)) "
        "(VERBP (VERBP (WHNP (DET какво)) "
        "(VERBP (AUX ще) (VERBP (VERB (VB направ) (AGR и))))) "
        "(ADPP (ADP от) (NOUNP (NNDEF (NNS (NOUN гъбк) (DIV и)) (DET те)))))))"
    )
    assert actual == expected
    assert module.tree_yield(module.parse_bracket_tree(actual)) == [
        "баба",
        "какво",
        "ще",
        "направ",
        "и",
        "от",
        "гъбк",
        "и",
        "те",
    ]


def test_rewrite_constituency_tree_handles_verb_agreement_and_definite_noun():
    module = load_module()
    tree = "(ROOT (SQ (VERBP (VERBP (VERB Искаш) (PART ли)) (NOUNP (NOUN книжката)))))"
    analyses = [
        analysis(
            module,
            "Искаш",
            "иска",
            "VERB",
            {"Mood": "Ind", "Number": "Sing", "Person": "2", "Tense": "Pres", "VerbForm": "Fin"},
        ),
        analysis(module, "ли", "ли", "PART", {}),
        analysis(
            module,
            "книжката",
            "книжка",
            "NOUN",
            {"Definite": "Def", "Gender": "Fem", "Number": "Sing"},
        ),
    ]
    stemmer = DummyStemmer(
        {
            "Искаш": "иска",
            "ли": "ли",
            "книжката": "книжка",
        }
    )

    actual = module.rewrite_constituency_tree(
        tree,
        analyses,
        stemmer,
        module.bg_morphtok.PosLexicon(),
    )

    expected = (
        "(ROOT (SQ (VERBP (VERBP (VERB (VB иска) (AGR ш)) (PART ли)) "
        "(NOUNP (NNDEF (NOUN книжка) (DET та))))))"
    )
    assert actual == expected


def test_rewrite_constituency_tree_handles_irregular_plural_noun():
    module = load_module()
    tree = (
        "(ROOT (SQ (VERBP (WHNP (DET Какво)) (VERB правят)) "
        "(NOUNP (NOUN децата))))"
    )
    analyses = [
        analysis(module, "Какво", "какво", "DET", {"PronType": "Int"}),
        analysis(
            module,
            "правят",
            "правя",
            "VERB",
            {"Mood": "Ind", "Number": "Plur", "Person": "3", "Tense": "Pres", "VerbForm": "Fin"},
        ),
        analysis(
            module,
            "децата",
            "дете",
            "NOUN",
            {"Definite": "Def", "Number": "Plur"},
        ),
    ]
    stemmer = DummyStemmer(
        {
            "Какво": "какво",
            "правят": "прав",
            "децата": "дет",
            "деца": "дет",
        }
    )

    actual = module.rewrite_constituency_tree(
        tree,
        analyses,
        stemmer,
        module.bg_morphtok.PosLexicon(),
    )

    expected = (
        "(ROOT (SQ (VERBP (WHNP (DET какво)) (VERB (VB правя) (AGR т))) "
        "(NOUNP (NNDEF (NNS деца) (DET та)))))"
    )
    assert actual == expected


def test_rewrite_constituency_tree_preserves_root_interjection_sister_structure():
    module = load_module()
    tree = "(ROOT (INTJP (INTJ Ква-ква)) (NOUNP (NOUN патката)))"
    analyses = [
        analysis(module, "Ква-ква", "ква-ква", "INTJ", {}),
        analysis(
            module,
            "патката",
            "патка",
            "NOUN",
            {"Definite": "Def", "Gender": "Fem", "Number": "Sing"},
        ),
    ]
    stemmer = DummyStemmer(
        {
            "Ква-ква": "ква-ква",
            "патката": "патка",
        }
    )

    actual = module.rewrite_constituency_tree(
        tree,
        analyses,
        stemmer,
        module.bg_morphtok.PosLexicon(),
    )

    expected = "(ROOT (INTJP (INTJ ква-ква)) (NOUNP (NNDEF (NOUN патка) (DET та))))"
    assert actual == expected
