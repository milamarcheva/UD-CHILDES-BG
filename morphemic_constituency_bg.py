from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import bg_morphtok


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DEP_TO_CONST_ROOT = SCRIPT_DIR.parent / "dep-to-const"
TREE_TOKEN_RE = re.compile(r"\(|\)|[^()\s]+")
MORPHEMIC_NOMINAL_UPOS = bg_morphtok.NOMINAL_POS | {"DET", "PRON"}


@dataclass
class BracketTree:
    label: str
    children: list["BracketTree | str"]

    def to_string(self) -> str:
        parts: list[str] = []
        for child in self.children:
            if isinstance(child, BracketTree):
                parts.append(child.to_string())
            else:
                parts.append(child)
        return f"({self.label} {' '.join(parts)})"


def parse_bracket_tree(tree_str: str) -> BracketTree:
    tokens = TREE_TOKEN_RE.findall(tree_str)
    index = 0

    def parse_node() -> BracketTree:
        nonlocal index
        if index >= len(tokens) or tokens[index] != "(":
            raise ValueError("Expected '(' while parsing tree.")
        index += 1
        if index >= len(tokens):
            raise ValueError("Unexpected end of tree after '('.")

        label = tokens[index]
        index += 1
        children: list[BracketTree | str] = []
        while index < len(tokens) and tokens[index] != ")":
            if tokens[index] == "(":
                children.append(parse_node())
            else:
                children.append(tokens[index])
                index += 1

        if index >= len(tokens) or tokens[index] != ")":
            raise ValueError("Expected ')' while parsing tree.")
        index += 1
        return BracketTree(label, children)

    tree = parse_node()
    if index != len(tokens):
        raise ValueError("Unexpected trailing tokens after tree parse.")
    return tree


def tree_yield(tree: BracketTree | str) -> list[str]:
    if isinstance(tree, str):
        return [tree]

    leaves: list[str] = []
    for child in tree.children:
        leaves.extend(tree_yield(child))
    return leaves


def is_preterminal(tree: BracketTree) -> bool:
    return len(tree.children) == 1 and isinstance(tree.children[0], str)


def sanitize_surface(text: str) -> str:
    return (
        text.replace("(", "-LRB-")
        .replace(")", "-RRB-")
        .replace("（", "-LRB-")
        .replace("）", "-RRB-")
        .replace(" ", "")
    )


def normalize_segment(segment: str) -> str:
    return sanitize_surface(bg_morphtok.normalize(segment))


def filtered_analyses(
    analyses: list[bg_morphtok.TokenAnalysis],
    *,
    exclude_punct: bool,
) -> list[bg_morphtok.TokenAnalysis]:
    if not exclude_punct:
        return analyses
    return [analysis for analysis in analyses if analysis.upos != "PUNCT"]


def segments_for_analysis(
    analysis: bg_morphtok.TokenAnalysis,
    stemmer,
    lexicon: bg_morphtok.PosLexicon,
    *,
    base_form: str,
) -> list[str]:
    return [
        normalize_segment(segment)
        for segment in bg_morphtok.segment_analysis(
            analysis,
            stemmer,
            lexicon,
            base_form=base_form,
            separate_morphemes=True,
        )
    ]


def morpheme_tokens_for_analyses(
    analyses: list[bg_morphtok.TokenAnalysis],
    stemmer,
    lexicon: bg_morphtok.PosLexicon,
    *,
    base_form: str,
) -> list[str]:
    tokens: list[str] = []
    for analysis in analyses:
        tokens.extend(
            segments_for_analysis(
                analysis,
                stemmer,
                lexicon,
                base_form=base_form,
            )
        )
    return tokens


def prefix_label(prefix: str) -> str:
    if prefix == "най-":
        return "SUP"
    if prefix == "по-":
        return "CMP"
    return "PFX"


def nominal_definite_label(upos: str) -> str:
    if upos == "ADJ":
        return "JJDEF"
    if upos in {"NOUN", "PROPN", "NUM"}:
        return "NNDEF"
    return f"{upos}DEF"


def nominal_plural_label(upos: str) -> str:
    if upos == "ADJ":
        return "JJS"
    if upos in {"NOUN", "PROPN", "NUM"}:
        return "NNS"
    return f"{upos}PL"


def is_article_segment(segment: str) -> bool:
    return segment in {normalize_segment(suffix) for suffix in bg_morphtok.ARTICLE_SUFFIXES}


def nominal_plural_suffix_label(
    analysis: bg_morphtok.TokenAnalysis,
    segment: str,
) -> str:
    if analysis.feats.get("Number") == "Count":
        return "COUNT"
    return "DIV"


def lexical_child(preterminal_label: str, segment: str) -> BracketTree:
    return BracketTree(preterminal_label, [segment])


def build_default_split_tree(preterminal_label: str, segments: list[str]) -> BracketTree:
    if len(segments) == 1:
        return BracketTree(preterminal_label, [segments[0]])

    children: list[BracketTree | str] = [lexical_child(preterminal_label, segments[0])]
    children.extend(BracketTree("AFF", [segment]) for segment in segments[1:])
    return BracketTree(preterminal_label, children)


def build_verb_tree(preterminal_label: str, segments: list[str]) -> BracketTree:
    if len(segments) == 1:
        return BracketTree(preterminal_label, [segments[0]])

    # Verbal morphemes are represented as lexical base plus agreement or other
    # inflectional material, e.g. VERB -> VB + AGR.
    children: list[BracketTree | str] = [BracketTree("VB", [segments[0]])]
    children.extend(BracketTree("AGR", [segment]) for segment in segments[1:])
    return BracketTree(preterminal_label, children)


def build_plural_nominal_tree(
    preterminal_label: str,
    analysis: bg_morphtok.TokenAnalysis,
    core_segments: list[str],
) -> BracketTree:
    plural_label = nominal_plural_label(analysis.upos)
    if len(core_segments) == 1:
        # Irregular plurals like деца/очи are already plural lexical bases.
        if analysis.upos == "NOUN" and bg_morphtok.is_irregular_plural_base(
            analysis, core_segments[0]
        ):
            return BracketTree(plural_label, [core_segments[0]])
        return BracketTree(plural_label, [lexical_child(preterminal_label, core_segments[0])])

    children: list[BracketTree | str] = [lexical_child(preterminal_label, core_segments[0])]
    for segment in core_segments[1:]:
        children.append(
            BracketTree(nominal_plural_suffix_label(analysis, segment), [segment])
        )
    return BracketTree(plural_label, children)


def build_nominal_core_tree(
    preterminal_label: str,
    analysis: bg_morphtok.TokenAnalysis,
    core_segments: list[str],
) -> BracketTree:
    if not core_segments:
        return BracketTree(preterminal_label, [""])

    if analysis.feats.get("Number") in {"Plur", "Count"}:
        return build_plural_nominal_tree(preterminal_label, analysis, core_segments)

    return build_default_split_tree(preterminal_label, core_segments)


def build_nominal_tree(
    preterminal_label: str,
    analysis: bg_morphtok.TokenAnalysis,
    segments: list[str],
) -> BracketTree:
    article_segment = None
    core_segments = list(segments)

    # Definite nominals project a dedicated outer layer with the article as its
    # right-hand child. The inner core may itself be plural or count-marked.
    if (
        analysis.feats.get("Definite") == "Def"
        and len(core_segments) >= 2
        and is_article_segment(core_segments[-1])
    ):
        article_segment = core_segments.pop()

    core_tree = build_nominal_core_tree(preterminal_label, analysis, core_segments)
    if article_segment is None:
        return core_tree

    return BracketTree(
        nominal_definite_label(analysis.upos),
        [core_tree, BracketTree("DET", [article_segment])],
    )


def build_prefixed_tree(
    preterminal_label: str,
    prefixes: list[str],
    core_tree: BracketTree,
) -> BracketTree:
    children: list[BracketTree | str] = [
        BracketTree(prefix_label(prefix), [prefix]) for prefix in prefixes
    ]
    children.append(core_tree)
    return BracketTree(preterminal_label, children)


def build_morphemic_preterminal(
    preterminal_label: str,
    analysis: bg_morphtok.TokenAnalysis,
    segments: list[str],
) -> BracketTree:
    if not segments:
        return BracketTree(preterminal_label, [""])

    prefixes: list[str] = []
    core_segments = list(segments)
    while core_segments and core_segments[0] in {"по-", "най-"}:
        prefixes.append(core_segments.pop(0))

    if not core_segments:
        core_tree = BracketTree(preterminal_label, prefixes)
    elif analysis.upos in bg_morphtok.VERB_POS:
        core_tree = build_verb_tree(preterminal_label, core_segments)
    elif analysis.upos in MORPHEMIC_NOMINAL_UPOS:
        core_tree = build_nominal_tree(preterminal_label, analysis, core_segments)
    else:
        core_tree = build_default_split_tree(preterminal_label, core_segments)

    if not prefixes:
        return core_tree
    return build_prefixed_tree(preterminal_label, prefixes, core_tree)


def is_split_verbal_preterminal(tree: BracketTree) -> bool:
    if tree.label not in {"VERB", "AUX"}:
        return False
    return any(isinstance(child, BracketTree) for child in tree.children)


def postprocess_verbal_wrappers(tree: BracketTree) -> BracketTree:
    rewritten_children: list[BracketTree | str] = []
    for child in tree.children:
        if isinstance(child, BracketTree):
            rewritten_children.append(postprocess_verbal_wrappers(child))
        else:
            rewritten_children.append(child)

    tree = BracketTree(tree.label, rewritten_children)
    if tree.label != "VERBP":
        return tree

    has_aux_child = any(
        isinstance(child, BracketTree)
        and child.label == "AUX"
        and is_preterminal(child)
        for child in tree.children
    )
    if not has_aux_child:
        return tree

    adjusted_children: list[BracketTree | str] = []
    for child in tree.children:
        if isinstance(child, BracketTree) and child.label == "VERB" and is_split_verbal_preterminal(child):
            adjusted_children.append(BracketTree("VERBP", [child]))
        else:
            adjusted_children.append(child)
    return BracketTree(tree.label, adjusted_children)


def rewrite_tree_with_morphemes(
    tree: BracketTree,
    analyses: list[bg_morphtok.TokenAnalysis],
    stemmer,
    lexicon: bg_morphtok.PosLexicon,
    *,
    base_form: str,
    state: dict[str, int] | None = None,
) -> BracketTree:
    if state is None:
        state = {"index": 0}

    if is_preterminal(tree):
        if state["index"] >= len(analyses):
            raise ValueError("Preterminal/analysis alignment ran out of analyses.")
        analysis = analyses[state["index"]]
        state["index"] += 1
        segments = segments_for_analysis(
            analysis,
            stemmer,
            lexicon,
            base_form=base_form,
        )
        return build_morphemic_preterminal(tree.label, analysis, segments)

    rewritten_children: list[BracketTree | str] = []
    for child in tree.children:
        if isinstance(child, BracketTree):
            rewritten_children.append(
                rewrite_tree_with_morphemes(
                    child,
                    analyses,
                    stemmer,
                    lexicon,
                    base_form=base_form,
                    state=state,
                )
            )
        else:
            rewritten_children.append(child)
    return BracketTree(tree.label, rewritten_children)


def rewrite_constituency_tree(
    tree_str: str,
    analyses: list[bg_morphtok.TokenAnalysis],
    stemmer,
    lexicon: bg_morphtok.PosLexicon,
    *,
    base_form: str = "stem",
) -> str:
    tree = parse_bracket_tree(tree_str)
    rewritten = rewrite_tree_with_morphemes(
        tree,
        analyses,
        stemmer,
        lexicon,
        base_form=base_form,
    )
    rewritten = postprocess_verbal_wrappers(rewritten)
    expected_yield = morpheme_tokens_for_analyses(
        analyses,
        stemmer,
        lexicon,
        base_form=base_form,
    )
    actual_yield = tree_yield(rewritten)
    assert actual_yield == expected_yield, (
        f"Morphemic yield mismatch: expected {expected_yield}, got {actual_yield}"
    )
    return rewritten.to_string()


def load_dep_to_const_converter(dep_to_const_root: Path):
    converter_path = dep_to_const_root / "src" / "converter.py"
    if not converter_path.is_file():
        raise FileNotFoundError(
            f"Could not find dep-to-const converter at {converter_path}"
        )

    spec = importlib.util.spec_from_file_location(
        "dep_to_const_converter",
        converter_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_structured_pos_tree(
    sentence,
    converter_module,
    *,
    exclude_punct: bool,
    add_root: bool,
) -> str:
    tree_str = converter_module.general_converter(
        converter_module.structured_converter,
        sentence,
        converter_module.get_pos_nt,
        exclude_punct=exclude_punct,
    )
    kept_tokens = converter_module.sentence_tokens(sentence, exclude_punct)
    if len(kept_tokens) == 1:
        tree_str = f"({converter_module.get_pos_nt(kept_tokens[0])} {tree_str})"
    if add_root:
        tree_str = f"(ROOT {tree_str})"
    return tree_str


def sentence_analyses_from_pyconll(sentence) -> list[bg_morphtok.TokenAnalysis]:
    def token_feats(token) -> dict[str, str]:
        raw_feats = getattr(token, "feats", None)
        if not raw_feats:
            return {}
        if isinstance(raw_feats, str):
            return bg_morphtok.parse_feats(raw_feats)

        try:
            items = raw_feats.items()
        except AttributeError:
            return bg_morphtok.parse_feats(str(raw_feats))

        feats: dict[str, str] = {}
        for key, values in items:
            if not values:
                continue
            if isinstance(values, (set, list, tuple)):
                value = ",".join(sorted(str(value) for value in values))
            else:
                value = str(values)
            if key == "Number" and value == "Plu":
                value = "Plur"
            feats[key] = value
        return feats

    analyses: list[bg_morphtok.TokenAnalysis] = []
    for token in sentence:
        if token.is_multiword():
            continue
        analyses.append(
            bg_morphtok.TokenAnalysis(
                text=token.form or "",
                lemma=(token.lemma if token.lemma and token.lemma != "_" else token.form) or "",
                upos=token.upos or "X",
                feats=token_feats(token),
            )
        )
    return analyses


def sentence_to_morphemic_constituency(
    sentence,
    converter_module,
    stemmer,
    lexicon: bg_morphtok.PosLexicon,
    *,
    exclude_punct: bool,
    add_root: bool,
    base_form: str,
) -> tuple[str, str]:
    analyses = filtered_analyses(
        sentence_analyses_from_pyconll(sentence),
        exclude_punct=exclude_punct,
    )
    structured_tree = build_structured_pos_tree(
        sentence,
        converter_module,
        exclude_punct=exclude_punct,
        add_root=add_root,
    )
    morphemic_tree = rewrite_constituency_tree(
        structured_tree,
        analyses,
        stemmer,
        lexicon,
        base_form=base_form,
    )
    morpheme_tokens = " ".join(
        morpheme_tokens_for_analyses(
            analyses,
            stemmer,
            lexicon,
            base_form=base_form,
        )
    )
    return morphemic_tree, morpheme_tokens


def find_conllu_files(source_path: Path) -> list[Path]:
    if source_path.is_file() and source_path.suffix == ".conllu":
        return [source_path]
    return sorted(source_path.glob("**/*.conllu"))


def output_base_path(source_path: Path, conllu_file: Path, output_root: Path) -> Path:
    if source_path.is_file():
        relative = Path(conllu_file.stem)
    else:
        relative = conllu_file.relative_to(source_path).with_suffix("")
    return output_root / relative


def convert_conllu_files(args) -> None:
    try:
        import pyconll
    except ImportError as exc:
        raise ImportError(
            "Could not import 'pyconll'. Install it in the active environment."
        ) from exc

    converter_module = load_dep_to_const_converter(args.dep_to_const_root)
    lexicon = bg_morphtok.PosLexicon.from_file(args.lexicon)
    stemmer = bg_morphtok.build_stemmer(
        rules=args.bulstem_rules,
        min_freq=args.bulstem_min_freq,
        left_context=args.bulstem_left_context,
    )

    source_path = Path(args.source_path)
    output_root = Path(args.output_path)

    for conllu_file in find_conllu_files(source_path):
        corpus = pyconll.load_from_file(str(conllu_file))
        base_path = output_base_path(source_path, conllu_file, output_root)
        base_path.parent.mkdir(parents=True, exist_ok=True)

        parse_path = base_path.with_suffix(".txt")
        tokens_path = Path(str(base_path) + ".tokens")

        with parse_path.open("w", encoding="utf-8") as parse_out, tokens_path.open(
            "w", encoding="utf-8"
        ) as token_out:
            for sentence in corpus:
                morphemic_tree, morpheme_tokens = sentence_to_morphemic_constituency(
                    sentence,
                    converter_module,
                    stemmer,
                    lexicon,
                    exclude_punct=not args.include_punct,
                    add_root=args.add_root,
                    base_form=args.base_form,
                )
                parse_out.write(morphemic_tree)
                parse_out.write("\n")
                token_out.write(morpheme_tokens)
                token_out.write("\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Bulgarian CoNLL-U parses into morphemically tokenized "
            "constituency parses by combining dep-to-const structured POS trees "
            "with bg_morphtok segmentation."
        )
    )
    parser.add_argument("--source_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument(
        "--dep-to-const-root",
        type=Path,
        default=DEFAULT_DEP_TO_CONST_ROOT,
        help="Path to the dep-to-const repository root.",
    )
    parser.add_argument(
        "--lexicon",
        type=Path,
        default=bg_morphtok.DEFAULT_LEXICON_PATH,
        help="Optional CHILDES-derived lexicon for bg_morphtok.",
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
        "--base-form",
        choices=("stem", "lemma", "lemma-nominalonly"),
        default="stem",
        help="Passed through to bg_morphtok for the lexical base choice.",
    )
    parser.add_argument(
        "--include-punct",
        action="store_true",
        help="Keep punctuation in the structured parse and morpheme yield.",
    )
    parser.add_argument(
        "--add-root",
        dest="add_root",
        action="store_true",
        default=True,
        help="Wrap each sentence in a top-level ROOT node. Default: enabled.",
    )
    parser.add_argument(
        "--no-add-root",
        dest="add_root",
        action="store_false",
        help="Do not wrap each sentence in ROOT.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    convert_conllu_files(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
