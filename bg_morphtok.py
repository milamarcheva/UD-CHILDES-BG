from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_BULSTEM_REPO = SCRIPT_DIR.parent / "bulstem-py"
DEFAULT_LEXICON_PATH = (
    SCRIPT_DIR / "conllufiles_sentences_lexicons" / "bg_lexicon_filtered.txt"
)

LEXICON_RE = re.compile(r"^\s*[^ ]+\s+[^ ]+\s+([A-Z]+)\s+-->\s+(.+?)\s*$")
CYRILLIC_WORD_RE = re.compile(r"^[А-Яа-яЁёЍѝ]+(?:-[А-Яа-яЁёЍѝ]+)*$")
VOWELS = set("аеиоуъюяАЕИОУЪЮЯ")
HYPHEN_PREFIX_RE = re.compile(r"^(по-|най-)([А-Яа-яЁёЍѝ]+(?:-[А-Яа-яЁёЍѝ]+)*)$")
CYRILLIC_CHAR_RE = re.compile(r"[А-Яа-яЁёЍѝ]")

NO_SEGMENT_UPOS = {
    "ADP",
    "ADV",
    "CCONJ",
    "INTJ",
    "PART",
    "PUNCT",
    "SCONJ",
    "SYM",
    "X",
}
ARTICLE_SUFFIXES = ("ите", "ото", "ата", "ият", "ия", "ът", "ят", "та", "то", "те", "а", "я")
REDUCED_ARTICLE_SUFFIXES = ("та", "то", "те")
ADJ_ENDING_SUFFIXES = ("а", "я", "о", "е", "и")
PLURAL_NOUN_SUFFIXES = ("ове", "та", "и", "е")
COUNT_NOUN_SUFFIXES = ("а", "я")
ARTICLE_POS = {"ADJ", "DET", "NOUN", "PRON"}
VERB_POS = {"VERB", "AUX"}
NOMINAL_POS = {"ADJ", "NOUN", "PROPN", "NUM"}
LEMMA_BASE_UPOS = {"PROPN", "NOUN", "ADJ", "NUM"}
IRREGULAR_NOUN_PLURALS = {
    # These plurals use a suppletive or stem-changing plural base. Treat the
    # plural surface as the lexical noun base so definite forms split only the
    # article, e.g. деца + та rather than дет + цата or дърве + тата.
    "брат": {"братя"},
    "дете": {"деца"},
    "дърво": {"дървета"},
    "око": {"очи"},
    "ръка": {"ръце"},
    "ухо": {"уши"},
    "цвете": {"цветя"},
    "човек": {"хора"},
}
CYRILLIC_CONFUSABLE_TRANSLATION = str.maketrans(
    {
        "a": "а",
        "c": "с",
        "e": "е",
        "k": "к",
        "m": "м",
        "o": "о",
        "p": "р",
        "t": "т",
        "x": "х",
        "y": "у",
    }
)


@dataclass(frozen=True)
class TokenAnalysis:
    text: str
    lemma: str
    upos: str
    feats: dict[str, str]


def is_conllu_word_id(token_id: str) -> bool:
    return "-" not in token_id and "." not in token_id


class PosLexicon:
    def __init__(self, by_pos: dict[str, set[str]] | None = None):
        self._by_pos = by_pos or {}

    @classmethod
    def from_file(cls, path: Path | None) -> "PosLexicon":
        if path is None or not path.is_file():
            return cls()

        by_pos: dict[str, set[str]] = {}
        with path.open("r", encoding="utf-8") as infile:
            for raw_line in infile:
                match = LEXICON_RE.match(raw_line.rstrip("\n"))
                if not match:
                    continue

                pos, word = match.groups()
                by_pos.setdefault(pos, set()).add(word.lower())

        return cls(by_pos)

    def contains(self, upos: str, word: str) -> bool:
        return word.lower() in self._by_pos.get(upos, set())

    def is_available(self) -> bool:
        return bool(self._by_pos)


def parse_feats(feats_str: str | None) -> dict[str, str]:
    if not feats_str or feats_str == "_":
        return {}

    feats: dict[str, str] = {}
    for part in feats_str.split("|"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        # A few handwritten test rows use the shortened plural value `Plu`;
        # normalize it so the segmentation rules do not depend on that typo.
        if key == "Number" and value == "Plu":
            value = "Plur"
        feats[key] = value
    return feats


def normalize(text: str) -> str:
    lowered = text.lower()
    # Some manually edited CSV rows mix Latin lookalikes into otherwise
    # Bulgarian strings (for example Latin `o` inside `малко`). Normalize
    # those confusables so comparisons stay script-stable.
    if CYRILLIC_CHAR_RE.search(lowered):
        return lowered.translate(CYRILLIC_CONFUSABLE_TRANSLATION)
    return lowered


def is_cyrillic_word(text: str) -> bool:
    return bool(CYRILLIC_WORD_RE.fullmatch(text))


def has_vowel(text: str) -> bool:
    return any(char in VOWELS for char in text)


def is_titlecase_token(text: str) -> bool:
    return bool(text) and text[0].isupper()


def is_lemma_form(analysis: TokenAnalysis) -> bool:
    lemma = analysis.lemma or analysis.text
    return normalize(analysis.text) == normalize(lemma)


def is_definite_propn(analysis: TokenAnalysis) -> bool:
    return analysis.upos == "PROPN" and analysis.feats.get("Definite") == "Def"


def is_definite_num(analysis: TokenAnalysis) -> bool:
    return analysis.upos == "NUM" and analysis.feats.get("Definite") == "Def"


def is_definite_exception_token(analysis: TokenAnalysis) -> bool:
    return is_definite_propn(analysis) or is_definite_num(analysis)


def split_hyphen_prefix(text: str) -> tuple[str, str] | None:
    match = HYPHEN_PREFIX_RE.fullmatch(text)
    if not match:
        return None
    return match.group(1), match.group(2)


def import_bulstemmer():
    import_error = None

    try:
        from bulstem.stem import BulStemmer

        return BulStemmer
    except ImportError as exc:
        import_error = exc

    if LOCAL_BULSTEM_REPO.is_dir():
        local_repo = str(LOCAL_BULSTEM_REPO)
        if local_repo not in sys.path:
            sys.path.insert(0, local_repo)

        try:
            from bulstem.stem import BulStemmer

            return BulStemmer
        except ImportError as exc:
            import_error = exc

    raise ImportError(
        "Could not import 'bulstem'. Install it in the active environment or keep "
        f"the local checkout at {LOCAL_BULSTEM_REPO} available."
    ) from import_error


def build_stemmer(rules: str, min_freq: int, left_context: int):
    BulStemmer = import_bulstemmer()
    return BulStemmer.from_file(
        rules,
        min_freq=min_freq,
        left_context=left_context,
    )


def import_classla():
    try:
        import classla

        return classla
    except ImportError as exc:
        raise ImportError(
            "Could not import 'classla'. Install it first with `pip install classla`, "
            "then download the Bulgarian models with `classla.download('bg')`."
        ) from exc


def build_pipeline(download_models: bool = False, use_gpu: bool = False):
    classla = import_classla()
    if download_models:
        classla.download("bg")

    try:
        return classla.Pipeline(
            "bg",
            processors="tokenize,pos,lemma",
            use_gpu=use_gpu,
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not initialize the CLASSLA Bulgarian pipeline. Make sure "
            "`classla.download('bg')` has been run for the same environment."
        ) from exc


def analyses_from_doc(doc) -> list[TokenAnalysis]:
    analyses: list[TokenAnalysis] = []

    for sentence in getattr(doc, "sentences", []):
        for word in getattr(sentence, "words", []):
            text = getattr(word, "text", "") or ""
            lemma = getattr(word, "lemma", "") or text
            upos = getattr(word, "upos", "") or "X"
            feats = parse_feats(getattr(word, "feats", None))
            analyses.append(TokenAnalysis(text=text, lemma=lemma, upos=upos, feats=feats))

    return analyses


def analyses_from_conllu_lines(lines: Iterable[str]) -> list[list[TokenAnalysis]]:
    sentences: list[list[TokenAnalysis]] = []
    current_sentence: list[TokenAnalysis] = []

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        if not line:
            if current_sentence:
                sentences.append(current_sentence)
                current_sentence = []
            continue
        if line.startswith("#"):
            continue

        columns = line.split("\t")
        if len(columns) != 10:
            continue

        token_id, form, lemma, upos, _xpos, feats, _head, _deprel, _deps, _misc = columns
        if not is_conllu_word_id(token_id):
            continue

        current_sentence.append(
            TokenAnalysis(
                text=form or "",
                lemma=(lemma if lemma and lemma != "_" else form) or "",
                upos=upos or "X",
                feats=parse_feats(feats),
            )
        )

    if current_sentence:
        sentences.append(current_sentence)

    return sentences


def lexical_base_ok(
    analysis: TokenAnalysis,
    base: str,
    lexicon: PosLexicon,
    bulstem_stem: str,
) -> bool:
    base_lower = normalize(base)
    lemma_lower = normalize(analysis.lemma or analysis.text)

    if base_lower == lemma_lower:
        return True
    if is_irregular_plural_base(analysis, base):
        return True
    if base_lower == bulstem_stem:
        return True
    if lexicon.contains(analysis.upos, base_lower):
        return True

    # DT/PRON forms are often absent from the local lexicon; allow them if the
    # base still looks like a real Bulgarian word.
    if analysis.upos in {"DET", "PRON"} and has_vowel(base) and len(base) >= 2:
        return True

    return False


def looks_like_reduced_definite_form(
    analysis: TokenAnalysis,
    lexicon: PosLexicon,
) -> bool:
    if analysis.upos not in {"ADJ", "DET", "PRON"}:
        return False

    surface = analysis.text
    surface_lower = normalize(surface)

    for suffix in REDUCED_ARTICLE_SUFFIXES:
        if not surface_lower.endswith(suffix):
            continue

        base = surface[: len(surface) - len(suffix)]
        if len(base) < 2 or not has_vowel(base):
            continue
        if lexicon.contains(analysis.upos, base):
            return True

    return False


def split_article_suffix(
    analysis: TokenAnalysis,
    lexicon: PosLexicon,
    bulstem_stem: str,
) -> tuple[str, str] | None:
    explicit_definite = analysis.feats.get("Definite") == "Def"
    reduced_definite = looks_like_reduced_definite_form(analysis, lexicon)

    if not explicit_definite and not reduced_definite:
        return None

    if analysis.upos not in ARTICLE_POS and not (
        analysis.upos == "VERB" and analysis.feats.get("VerbForm") == "Part"
    ) and not is_definite_exception_token(analysis):
        return None

    surface = analysis.text
    surface_lower = normalize(surface)
    candidates: list[tuple[int, int, str, str]] = []

    for suffix in ARTICLE_SUFFIXES:
        if not surface_lower.endswith(suffix):
            continue
        if not explicit_definite and suffix not in REDUCED_ARTICLE_SUFFIXES:
            continue

        base = surface[: len(surface) - len(suffix)]
        if len(base) < 2 or not has_vowel(base):
            continue
        if not lexical_base_ok(analysis, base, lexicon, bulstem_stem):
            continue

        score = 0
        base_lower = normalize(base)
        lemma_lower = normalize(analysis.lemma or analysis.text)
        if base_lower == lemma_lower:
            score = 3
        elif base_lower == bulstem_stem:
            score = 2
        elif lexicon.contains(analysis.upos, base):
            score = 1
        elif analysis.upos in {"DET", "PRON"} and has_vowel(base):
            score = 1

        if score <= 0:
            continue
        candidates.append((score, len(suffix), base, surface[-len(suffix) :]))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    _, _, base, suffix = candidates[0]
    return base, suffix


def article_suffix_candidates(
    analysis: TokenAnalysis,
    lexicon: PosLexicon,
) -> list[tuple[str, str]]:
    explicit_definite = analysis.feats.get("Definite") == "Def"
    reduced_definite = looks_like_reduced_definite_form(analysis, lexicon)

    if not explicit_definite and not reduced_definite:
        return []

    if analysis.upos not in ARTICLE_POS and not (
        analysis.upos == "VERB" and analysis.feats.get("VerbForm") == "Part"
    ) and not is_definite_exception_token(analysis):
        return []

    surface = analysis.text
    surface_lower = normalize(surface)
    candidates: list[tuple[str, str]] = []

    for suffix in ARTICLE_SUFFIXES:
        if not surface_lower.endswith(suffix):
            continue
        if not explicit_definite and suffix not in REDUCED_ARTICLE_SUFFIXES:
            continue

        base = surface[: len(surface) - len(suffix)]
        if len(base) < 2 or not has_vowel(base):
            continue
        candidates.append((base, surface[-len(suffix) :]))

    return candidates


def split_adjectival_ending(
    analysis: TokenAnalysis,
    lexicon: PosLexicon,
    bulstem_stem: str,
) -> tuple[str, str] | None:
    if analysis.upos != "ADJ":
        return None
    if analysis.feats.get("Definite") == "Def":
        return None
    if is_titlecase_token(analysis.text):
        return None

    surface = analysis.text
    surface_lower = normalize(surface)
    number = analysis.feats.get("Number")
    if number != "Plur":
        return None

    suffix = "и"
    if surface_lower.endswith(suffix):
        base = surface[: len(surface) - len(suffix)]
        if len(base) >= 2 and has_vowel(base) and lexical_base_ok(analysis, base, lexicon, bulstem_stem):
            return base, surface[-len(suffix) :]

    return None


def noun_base_score(
    analysis: TokenAnalysis,
    base: str,
    bulstem_stem: str,
    lexicon: PosLexicon,
) -> int:
    base_lower = normalize(base)
    lemma_lower = normalize(analysis.lemma or analysis.text)

    if base_lower == lemma_lower:
        return 3
    if is_irregular_plural_base(analysis, base):
        return 3
    if base_lower == bulstem_stem:
        return 2
    if lexicon.contains("NOUN", base):
        return 1
    return 0


def split_plural_noun(
    analysis: TokenAnalysis,
    lexicon: PosLexicon,
    bulstem_stem: str,
) -> tuple[str, str] | None:
    if analysis.upos != "NOUN":
        return None
    number = analysis.feats.get("Number")
    if number not in {"Plur", "Count"}:
        return None
    if analysis.feats.get("Definite") == "Def":
        return None

    surface = analysis.text
    surface_lower = normalize(surface)
    if is_irregular_plural_base(analysis, surface):
        return None
    candidates: list[tuple[int, int, str, str]] = []
    suffixes = COUNT_NOUN_SUFFIXES if number == "Count" else PLURAL_NOUN_SUFFIXES

    # Count forms like шанс-а should only be split when the morphology marks
    # them as Number=Count; the same surface suffix can also be a definite
    # article under Definite=Def and is handled by the article rule above.
    for suffix in suffixes:
        if not surface_lower.endswith(suffix):
            continue

        base = surface[: len(surface) - len(suffix)]
        if len(base) < 2 or not has_vowel(base):
            continue

        score = noun_base_score(analysis, base, bulstem_stem, lexicon)
        if score <= 0:
            continue

        candidates.append((score, len(suffix), base, surface[-len(suffix) :]))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    _, _, base, suffix = candidates[0]
    return base, suffix


def nominal_candidate_score(
    analysis: TokenAnalysis,
    base: str,
    lexicon: PosLexicon,
    bulstem_stem: str,
) -> int:
    base_lower = normalize(base)
    lemma_lower = normalize(analysis.lemma or analysis.text)

    if base_lower == lemma_lower:
        return 3
    if is_irregular_plural_base(analysis, base):
        return 3
    if base_lower == bulstem_stem:
        return 2
    if lexicon.contains(analysis.upos, base):
        return 1
    if analysis.upos in {"PROPN", "NUM"} and len(base) >= 2 and has_vowel(base):
        return 1
    return 0


def is_irregular_plural_base(analysis: TokenAnalysis, base: str) -> bool:
    if analysis.upos != "NOUN":
        return False
    if analysis.feats.get("Number") != "Plur":
        return False
    lemma_lower = normalize(analysis.lemma or analysis.text)
    return normalize(base) in IRREGULAR_NOUN_PLURALS.get(lemma_lower, set())


def strip_definite_feat(feats: dict[str, str]) -> dict[str, str]:
    remaining = dict(feats)
    remaining.pop("Definite", None)
    return remaining


def nominal_surface_form_ok(analysis: TokenAnalysis) -> bool:
    surface = analysis.text
    if len(surface) < 2 or not has_vowel(surface):
        return False

    if analysis.upos == "ADJ":
        number = analysis.feats.get("Number")
        gender = analysis.feats.get("Gender")
        surface_lower = normalize(surface)
        if number == "Plur":
            return surface_lower.endswith("и")
        if gender == "Fem":
            return surface_lower.endswith(("а", "я"))
        if gender == "Neut":
            return surface_lower.endswith(("о", "е"))
        return True

    if analysis.upos == "NOUN":
        number = analysis.feats.get("Number")
        if number == "Plur":
            surface_lower = normalize(surface)
            return surface_lower.endswith(PLURAL_NOUN_SUFFIXES)
        if number == "Count":
            surface_lower = normalize(surface)
            return surface_lower.endswith(COUNT_NOUN_SUFFIXES)
        return True

    if analysis.upos in {"PROPN", "NUM"}:
        return True

    return False


def allowed_verb_suffixes(analysis: TokenAnalysis) -> set[str]:
    feats = analysis.feats
    suffixes: set[str] = set()
    verb_form = feats.get("VerbForm")
    person = feats.get("Person")
    number = feats.get("Number")
    mood = feats.get("Mood")
    tense = feats.get("Tense")

    if verb_form == "Fin":
        if person == "2" and number == "Sing":
            suffixes.update({"ш", "и"})
        if person == "1" and number == "Plur":
            suffixes.update({"ме", "ем", "им"})
        if person == "2" and number == "Plur":
            suffixes.update({"те", "ете", "ите"})
        if person == "3" and number == "Plur":
            # Some present 3pl forms surface with the thematic vowel already
            # included in the base expected by the gold morpheme split, e.g.
            # направя + т instead of направ + ят.
            suffixes.update({"ат", "ят", "т"})
        if tense in {"Past", "Imp"} or mood == "Ind":
            suffixes.update({"х", "ха", "хме", "хте", "ше"})
        if tense == "Past":
            suffixes.add("и")
        if mood == "Imp":
            suffixes.update({"й", "йте", "и"})

    if verb_form == "Part":
        suffixes.update(
            {
                "л",
                "ла",
                "ло",
                "ли",
                "ал",
                "ала",
                "ало",
                "али",
                "ел",
                "ела",
                "ело",
                "ели",
                "ил",
                "ила",
                "ило",
                "или",
                "ял",
                "яла",
                "яло",
                "яли",
            }
        )

    if verb_form == "Ger":
        suffixes.add("йки")

    return suffixes


def verb_base_score(
    analysis: TokenAnalysis,
    base: str,
    bulstem_stem: str,
    lexicon: PosLexicon,
) -> int:
    base_lower = normalize(base)
    lemma_lower = normalize(analysis.lemma or analysis.text)

    if base_lower == lemma_lower:
        return 3
    if base_lower == bulstem_stem:
        return 2
    if lexicon.contains("VERB", base):
        return 1
    return 0


def split_verb_suffix(
    analysis: TokenAnalysis,
    bulstem_stem: str,
    lexicon: PosLexicon,
) -> tuple[str, str] | None:
    if analysis.upos not in VERB_POS:
        return None

    surface = analysis.text
    surface_lower = normalize(surface)
    if bulstem_stem == surface_lower and normalize(analysis.lemma or analysis.text) == surface_lower:
        return None

    candidates: list[tuple[int, int, str, str]] = []
    for suffix in sorted(allowed_verb_suffixes(analysis), key=len, reverse=True):
        if not surface_lower.endswith(suffix):
            continue

        base = surface[: len(surface) - len(suffix)]
        if len(base) < 2 or not has_vowel(base):
            continue

        score = verb_base_score(analysis, base, bulstem_stem, lexicon)
        if score <= 0:
            continue

        candidates.append((score, len(suffix), base, surface[-len(suffix) :]))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    _, _, base, suffix = candidates[0]
    return base, suffix


def replace_base_with_lemma(
    analysis: TokenAnalysis,
    split_parts: tuple[str, str],
    base_form: str,
) -> list[str]:
    return replace_first_segment_with_lemma(analysis, list(split_parts), base_form)


def replace_first_segment_with_lemma(
    analysis: TokenAnalysis,
    segments: list[str],
    base_form: str,
) -> list[str]:
    # Keep the existing surface-derived stem by default. In lemma mode, only
    # nominal/adjectival segmentations swap the leftmost lexical base for the
    # analysis lemma; verbs continue to keep the stem-like base.
    if len(segments) < 2:
        return segments
    if base_form not in {"lemma", "lemma-nominalonly"}:
        return segments
    if analysis.upos not in LEMMA_BASE_UPOS:
        return segments

    lemma = analysis.lemma or analysis.text
    if not lemma or lemma == "_":
        return segments

    return [lemma, *segments[1:]]


def surface_stem(stemmer, text: str) -> str:
    bulstem_stem = normalize(stemmer.stem(text))
    if bulstem_stem:
        return bulstem_stem
    return normalize(text)


def segment_nominal_parts(
    analysis: TokenAnalysis,
    stemmer,
    lexicon: PosLexicon,
    base_form: str,
    separate_morphemes: bool,
) -> list[str] | None:
    if analysis.upos not in NOMINAL_POS:
        return None

    bulstem_stem = surface_stem(stemmer, analysis.text)

    layered_article = split_article_layers(
        analysis,
        stemmer,
        lexicon,
        base_form,
        bulstem_stem,
    )
    if layered_article is not None:
        if separate_morphemes:
            return layered_article
        return [layered_article[0], "".join(layered_article[1:])]

    article_split = split_article_suffix(analysis, lexicon, bulstem_stem)
    if article_split is not None:
        return replace_base_with_lemma(analysis, article_split, base_form)

    plural_noun_split = split_plural_noun(analysis, lexicon, bulstem_stem)
    if plural_noun_split is not None:
        return replace_base_with_lemma(analysis, plural_noun_split, base_form)

    adjectival_split = split_adjectival_ending(analysis, lexicon, bulstem_stem)
    if adjectival_split is not None:
        return replace_base_with_lemma(analysis, adjectival_split, base_form)

    return None


def split_article_layers(
    analysis: TokenAnalysis,
    stemmer,
    lexicon: PosLexicon,
    base_form: str,
    bulstem_stem: str,
) -> list[str] | None:
    # In separate-morphemes mode, prefer article splits that expose an
    # additional nominal suffix layer, e.g. хубав + и + те over хубав + ите.
    best_parts: list[str] | None = None
    best_rank: tuple[int, int, int, int] | None = None

    for base, suffix in article_suffix_candidates(analysis, lexicon):
        residual_analysis = TokenAnalysis(
            text=base,
            lemma=analysis.lemma,
            upos=analysis.upos,
            feats=strip_definite_feat(analysis.feats),
        )
        residual_parts = split_residual_nominal_parts(
            residual_analysis,
            stemmer,
            lexicon,
            base_form,
        )

        if len(residual_parts) == 1:
            candidate_score = nominal_candidate_score(
                analysis, base, lexicon, bulstem_stem)
            if candidate_score <= 0 and not nominal_surface_form_ok(residual_analysis):
                continue
        else:
            candidate_score = nominal_candidate_score(
                analysis, base, lexicon, bulstem_stem)

        # Prefer analyses that expose more suffix layers, and for nominal and
        # adjectival forms prefer a residual base that still looks like a valid
        # standalone surface form such as клета/голяма/широко over a flatter
        # split like клет + ата or широк + ото.
        surface_bonus = 1 if nominal_surface_form_ok(residual_analysis) else 0
        rank = (len(residual_parts), surface_bonus, candidate_score, -len(suffix))
        candidate_parts = [*residual_parts, suffix]
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_parts = candidate_parts

    if best_parts is None:
        return None
    return replace_first_segment_with_lemma(analysis, best_parts, base_form)


def split_residual_nominal_parts(
    analysis: TokenAnalysis,
    stemmer,
    lexicon: PosLexicon,
    base_form: str,
) -> list[str]:
    bulstem_stem = surface_stem(stemmer, analysis.text)

    plural_noun_split = split_plural_noun(analysis, lexicon, bulstem_stem)
    if plural_noun_split is not None:
        return replace_base_with_lemma(analysis, plural_noun_split, base_form)

    adjectival_split = split_adjectival_ending(analysis, lexicon, bulstem_stem)
    if adjectival_split is not None:
        return replace_base_with_lemma(analysis, adjectival_split, base_form)

    return [analysis.text]


def segment_analysis(
    analysis: TokenAnalysis,
    stemmer,
    lexicon: PosLexicon,
    base_form: str = "stem",
    separate_morphemes: bool = False,
) -> list[str]:
    text = analysis.text
    if not text:
        return [text]
    hyphen_prefix_split = split_hyphen_prefix(text)
    if hyphen_prefix_split is not None:
        prefix, remainder = hyphen_prefix_split
        remainder_analysis = TokenAnalysis(
            text=remainder,
            lemma=analysis.lemma,
            upos=analysis.upos,
            feats=analysis.feats,
        )
        return [prefix, *segment_analysis(
            remainder_analysis,
            stemmer,
            lexicon,
            base_form=base_form,
            separate_morphemes=separate_morphemes,
        )]
    # Proper nouns and numerals stay unsplit by default. The only exception is
    # when they are explicitly marked as definite, in which case they can take
    # the article split rule.
    if analysis.upos in NO_SEGMENT_UPOS and not is_definite_exception_token(analysis):
        return [text]
    if not is_cyrillic_word(text):
        return [text]
    if (
        is_lemma_form(analysis)
        and analysis.feats.get("Definite") != "Def"
        and not looks_like_reduced_definite_form(analysis, lexicon)
    ):
        return [text]
    if (
        analysis.feats.get("Definite") != "Def"
        and analysis.upos in {"ADJ", "DET", "NOUN", "PRON"}
        and lexicon.contains("PROPN", text)
    ):
        return [text]

    nominal_segments = segment_nominal_parts(
        analysis,
        stemmer,
        lexicon,
        base_form=base_form,
        separate_morphemes=separate_morphemes,
    )
    if nominal_segments is not None:
        return nominal_segments

    bulstem_stem = surface_stem(stemmer, text)

    article_split = split_article_suffix(analysis, lexicon, bulstem_stem)
    if article_split is not None:
        return replace_base_with_lemma(analysis, article_split, base_form)

    plural_noun_split = split_plural_noun(analysis, lexicon, bulstem_stem)
    if plural_noun_split is not None:
        return replace_base_with_lemma(analysis, plural_noun_split, base_form)

    adjectival_split = split_adjectival_ending(analysis, lexicon, bulstem_stem)
    if adjectival_split is not None:
        return replace_base_with_lemma(analysis, adjectival_split, base_form)

    verb_split = split_verb_suffix(analysis, bulstem_stem, lexicon)
    if verb_split is not None:
        return replace_base_with_lemma(analysis, verb_split, base_form)

    return [text]


def should_emit_token(analysis: TokenAnalysis) -> bool:
    return analysis.upos != "PUNCT"


def tokenize_analyses(
    analyses: list[TokenAnalysis],
    stemmer,
    lexicon: PosLexicon,
    base_form: str = "stem",
    separate_morphemes: bool = False,
) -> str:
    output_tokens: list[str] = []
    for analysis in analyses:
        if not should_emit_token(analysis):
            continue
        output_tokens.extend(
            normalize(token)
            for token in segment_analysis(
                analysis,
                stemmer,
                lexicon,
                base_form=base_form,
                separate_morphemes=separate_morphemes,
            )
        )

    return " ".join(token for token in output_tokens if token)


def tokenize_doc(
    doc,
    stemmer,
    lexicon: PosLexicon,
    base_form: str = "stem",
    separate_morphemes: bool = False,
) -> str:
    analyses = analyses_from_doc(doc)
    return tokenize_analyses(
        analyses,
        stemmer,
        lexicon,
        base_form=base_form,
        separate_morphemes=separate_morphemes,
    )


def tokenize_sentence(
    text: str,
    pipeline,
    stemmer,
    lexicon: PosLexicon,
    base_form: str = "stem",
    separate_morphemes: bool = False,
) -> str:
    doc = pipeline(text)
    return tokenize_doc(
        doc,
        stemmer,
        lexicon,
        base_form=base_form,
        separate_morphemes=separate_morphemes,
    )


def process_conllu_lines(
    lines: Iterable[str],
    stemmer,
    lexicon: PosLexicon,
    base_form: str = "stem",
    separate_morphemes: bool = False,
) -> list[str]:
    output_lines: list[str] = []

    for analyses in analyses_from_conllu_lines(lines):
        output_lines.append(
            tokenize_analyses(
                analyses,
                stemmer,
                lexicon,
                base_form=base_form,
                separate_morphemes=separate_morphemes,
            )
        )

    return output_lines


def process_lines(
    lines: Iterable[str],
    pipeline,
    stemmer,
    lexicon: PosLexicon,
    base_form: str = "stem",
    separate_morphemes: bool = False,
) -> list[str]:
    output_lines: list[str] = []

    for raw_line in lines:
        content = raw_line.rstrip("\r\n")
        if not content.strip():
            output_lines.append("")
            continue
        output_lines.append(
            tokenize_sentence(
                content,
                pipeline,
                stemmer,
                lexicon,
                base_form=base_form,
                separate_morphemes=separate_morphemes,
            )
        )

    return output_lines


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bulgarian morphemic tokeniser driven by CLASSLA analyses and "
            "BulStem stems."
        )
    )
    parser.add_argument(
        "sentence",
        nargs="*",
        help="Sentence to morphemically tokenise. If omitted, read from --input-file or stdin.",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="UTF-8 text file with one sentence per line.",
    )
    parser.add_argument(
        "--input-conllu",
        type=Path,
        default=None,
        help="Optional CoNLL-U file. If set, use FORM/LEMMA/UPOS/FEATS from the dependency parse instead of running CLASSLA.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Optional UTF-8 output file.",
    )
    parser.add_argument(
        "--lexicon",
        type=Path,
        default=DEFAULT_LEXICON_PATH,
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
        "--base-form",
        choices=("stem", "lemma", "lemma-nominalonly"),
        default="stem",
        help="Stem keeps the surface-minus-suffix base; lemma and lemma-nominalonly replace that base with the analysis lemma only for segmented PROPN/NOUN/ADJ/NUM tokens.",
    )
    parser.add_argument(
        "--separate-morphemes",
        action="store_true",
        help="For nominals (ADJ/NOUN/PROPN/NUM), split stacked functional morphemes such as plural plus definiteness into separate tokens when the morphology supports it.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="For a single input sentence, print both the morphtok output and the CLASSLA CoNLL-U analysis.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    try:
        stemmer = build_stemmer(
            rules=args.bulstem_rules,
            min_freq=args.bulstem_min_freq,
            left_context=args.bulstem_left_context,
        )
    except Exception as exc:
        print(f"Could not initialize BulStem: {exc}", file=sys.stderr)
        return 1

    lexicon = PosLexicon.from_file(args.lexicon)

    pipeline = None
    needs_pipeline = args.input_conllu is None
    if needs_pipeline:
        try:
            pipeline = build_pipeline(
                download_models=args.download_models,
                use_gpu=args.use_gpu,
            )
        except Exception as exc:
            print(f"Could not initialize CLASSLA: {exc}", file=sys.stderr)
            return 1

    if args.input_conllu is not None:
        lines = args.input_conllu.read_text(encoding="utf-8").splitlines()
        output_lines = process_conllu_lines(
            lines,
            stemmer,
            lexicon,
            base_form=args.base_form,
            separate_morphemes=args.separate_morphemes,
        )
        output_text = "\n".join(output_lines) + "\n"
        if args.output_file is not None:
            args.output_file.parent.mkdir(parents=True, exist_ok=True)
            args.output_file.write_text(output_text, encoding="utf-8")
        else:
            sys.stdout.write(output_text)
        return 0

    if args.input_file is not None:
        lines = args.input_file.read_text(encoding="utf-8").splitlines()
        output_lines = process_lines(
            lines,
            pipeline,
            stemmer,
            lexicon,
            base_form=args.base_form,
            separate_morphemes=args.separate_morphemes,
        )
        output_text = "\n".join(output_lines) + "\n"
        if args.output_file is not None:
            args.output_file.parent.mkdir(parents=True, exist_ok=True)
            args.output_file.write_text(output_text, encoding="utf-8")
        else:
            sys.stdout.write(output_text)
        return 0

    if args.sentence:
        sentence = " ".join(args.sentence)
        doc = pipeline(sentence)
        output = tokenize_doc(
            doc,
            stemmer,
            lexicon,
            base_form=args.base_form,
            separate_morphemes=args.separate_morphemes,
        )
        if args.debug:
            debug_output = f"morphtok: {output}\n\nconllu:\n{doc.to_conll()}"
            if args.output_file is not None:
                args.output_file.parent.mkdir(parents=True, exist_ok=True)
                args.output_file.write_text(debug_output, encoding="utf-8")
            else:
                print(debug_output)
            return 0
        if args.output_file is not None:
            args.output_file.parent.mkdir(parents=True, exist_ok=True)
            args.output_file.write_text(output + "\n", encoding="utf-8")
        else:
            print(output)
        return 0

    stdin_lines = sys.stdin.read().splitlines()
    output_lines = process_lines(
        stdin_lines,
        pipeline,
        stemmer,
        lexicon,
        base_form=args.base_form,
        separate_morphemes=args.separate_morphemes,
    )
    output_text = "\n".join(output_lines) + "\n"
    if args.output_file is not None:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text(output_text, encoding="utf-8")
    else:
        sys.stdout.write(output_text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
