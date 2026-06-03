from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


ELLIPSIS_RE = re.compile(r"\.\s+\.\s+\.")
TEXT_PREFIX = "# text = "

TOKEN_PROFILE_FIXES = {
    "бе": {
        "lemma": "бе",
        "upos": "INTJ",
        "feats": "_",
    },
    "хайде": {
        "lemma": "хайде",
        "upos": "INTJ",
        "feats": "_",
    },
    "айде": {
        "lemma": "хайде",
        "upos": "INTJ",
        "feats": "_",
    },
    "де": {
        "lemma": "де",
        "upos": "INTJ",
        "feats": "_",
    },
    "ще": {
        "lemma": "ще",
        "upos": "AUX",
        "feats": "_",
    },
    "мамо": {
        "lemma": "мама",
        "upos": "NOUN",
        "feats": "Definite=Ind|Gender=Fem|Number=Sing|Case=Voc",
    },
    "бабо": {
        "lemma": "баба",
        "upos": "NOUN",
        "feats": "Definite=Ind|Gender=Fem|Number=Sing|Case=Voc",
    },
    "тате": {
        "lemma": "тате",
        "upos": "NOUN",
        "feats": "Definite=Ind|Gender=Masc|Number=Sing",
    },
    "тати": {
        "lemma": "тате",
        "upos": "NOUN",
        "feats": "Definite=Ind|Gender=Masc|Number=Sing",
    },
}
TARGET_INTJ = {"браво", "мерси"}
FORM_SURFACE_FIXES = {
    "втоето": "твоето",
    "каъв": "какъв",
}
FORM_LEMMA_FIXES = {
    "виж": "виждам-(се)",
    "дай": "дам-(се)",
    "какъв": "какъв",
    "твоето": "твой",
}
SOFTSIGN_JU_FIXES = {
    "ьу": "ю",
    "ьй": "ю",
    "Ьу": "Ю",
    "Ьй": "Ю",
    "ЬУ": "Ю",
    "ЬЙ": "Ю",
}


@dataclass
class Token:
    id: int
    form: str
    lemma: str
    upos: str
    xpos: str
    feats: str
    head: str
    deprel: str
    deps: str
    misc: str

    @classmethod
    def from_line(cls, line: str) -> "Token":
        cols = line.split("\t")
        if len(cols) != 10:
            raise ValueError(f"Expected 10 CoNLL-U columns, got {len(cols)}: {line}")
        return cls(
            id=int(cols[0]),
            form=cols[1],
            lemma=cols[2],
            upos=cols[3],
            xpos=cols[4],
            feats=cols[5],
            head=cols[6],
            deprel=cols[7],
            deps=cols[8],
            misc=cols[9],
        )

    def to_line(self) -> str:
        return "\t".join(
            [
                str(self.id),
                self.form,
                self.lemma,
                self.upos,
                self.xpos,
                self.feats,
                self.head,
                self.deprel,
                self.deps,
                self.misc,
            ]
        )


def normalize_match(text: str) -> str:
    return text.lower().replace("x", "х")


def replace_softsign_ju(text: str) -> str:
    updated = text
    for source, target in SOFTSIGN_JU_FIXES.items():
        updated = updated.replace(source, target)
    return updated


def match_case(source: str, target: str) -> str:
    if source.isupper():
        return target.upper()
    if source[:1].isupper() and source[1:].islower():
        return target.capitalize()
    return target


def replace_surface_forms_in_text(text: str) -> str:
    updated = text
    for source, target in FORM_SURFACE_FIXES.items():
        pattern = re.compile(rf"\b{re.escape(source)}\b", re.IGNORECASE)
        updated = pattern.sub(lambda match: match_case(match.group(0), target), updated)
    return updated


def fix_text_comment(line: str, counts: Counter[str]) -> str:
    if not line.startswith(TEXT_PREFIX):
        return line

    text = line[len(TEXT_PREFIX) :]
    updated = replace_softsign_ju(ELLIPSIS_RE.sub("...", text))
    updated = replace_surface_forms_in_text(updated)
    if updated != text:
        counts["text_comments_fixed"] += 1
    return f"{TEXT_PREFIX}{updated}"


def sentence_root_id(tokens: list[Token]) -> int | None:
    roots = [token.id for token in tokens if token.head == "0"]
    if not roots:
        return None
    return roots[0]


def merge_split_ellipsis(tokens: list[Token], counts: Counter[str]) -> list[Token]:
    updated = list(tokens)
    index = 0
    while index <= len(updated) - 3:
        window = updated[index : index + 3]
        if all(token.form == "." and token.upos == "PUNCT" for token in window):
            root_id = sentence_root_id(updated)
            merged = Token(
                id=window[0].id,
                form="...",
                lemma="...",
                upos="PUNCT",
                xpos="punct",
                feats="_",
                head=str(root_id) if root_id is not None else window[0].head,
                deprel="punct",
                deps="_",
                misc="_",
            )
            removed_ids = {token.id for token in window[1:]}
            updated = updated[:index] + [merged] + updated[index + 3 :]
            for token in updated[index + 1 :]:
                if token.head.isdigit() and int(token.head) in removed_ids:
                    token.head = str(merged.id)
            counts["ellipsis_merged"] += 1
        else:
            index += 1
    return updated


def renumber_tokens(tokens: list[Token]) -> None:
    id_map = {token.id: index + 1 for index, token in enumerate(tokens)}
    for index, token in enumerate(tokens, start=1):
        token.id = index
    for token in tokens:
        if token.head == "0" or token.head == "_":
            continue
        if token.head.isdigit():
            head_id = int(token.head)
            if head_id in id_map:
                token.head = str(id_map[head_id])


def should_skip_profile_fix(token: Token, norm_form: str) -> bool:
    # Preserve lexical 'ще' ('does not want') when it is already analyzed as VERB/ща.
    return norm_form == "ще" and normalize_match(token.lemma) == "ща" and token.upos == "VERB"


def apply_token_profile_fix(token: Token, norm_form: str, counts: Counter[str]) -> None:
    target = TOKEN_PROFILE_FIXES[norm_form]
    if token.lemma != target["lemma"]:
        token.lemma = target["lemma"]
        counts[f"lemma_fixed:{norm_form}"] += 1
    if token.upos != target["upos"]:
        token.upos = target["upos"]
        counts[f"upos_fixed:{norm_form}"] += 1
        if target["upos"] == "INTJ":
            counts["upos_intj_fixed"] += 1
    if token.feats != target["feats"]:
        token.feats = target["feats"]
        counts[f"feats_fixed:{norm_form}"] += 1


def fix_token(token: Token, counts: Counter[str]) -> None:
    fixed_form = replace_softsign_ju(token.form)
    if fixed_form != token.form:
        token.form = fixed_form
        counts["token_forms_fixed"] += 1

    norm_form = normalize_match(token.form)
    if norm_form in FORM_SURFACE_FIXES:
        replacement = match_case(token.form, FORM_SURFACE_FIXES[norm_form])
        if replacement != token.form:
            token.form = replacement
            counts[f"surface_fixed:{norm_form}"] += 1

    fixed_lemma = replace_softsign_ju(token.lemma)
    if fixed_lemma != token.lemma:
        token.lemma = fixed_lemma
        counts["token_lemmas_softsign_fixed"] += 1

    norm_form = normalize_match(token.form)
    if norm_form in TOKEN_PROFILE_FIXES and not should_skip_profile_fix(token, norm_form):
        apply_token_profile_fix(token, norm_form, counts)

    if norm_form in TARGET_INTJ and token.upos != "INTJ":
        token.upos = "INTJ"
        counts["upos_intj_fixed"] += 1

    if norm_form in FORM_LEMMA_FIXES and token.lemma != FORM_LEMMA_FIXES[norm_form]:
        token.lemma = FORM_LEMMA_FIXES[norm_form]
        counts[f"lemma_fixed:{norm_form}"] += 1

    if normalize_match(token.form).startswith("хубав") and token.lemma != "хубав":
        token.lemma = "хубав"
        counts["lemma_fixed:хубав"] += 1


def fix_final_punct(tokens: list[Token], counts: Counter[str]) -> None:
    if not tokens:
        return
    last = tokens[-1]
    if last.upos != "PUNCT":
        return
    root_id = sentence_root_id(tokens)
    if root_id is None:
        return
    desired_head = str(root_id)
    if last.head != desired_head or last.deprel != "punct":
        last.head = desired_head
        last.deprel = "punct"
        counts["final_punct_attached_to_root"] += 1


def fix_sentence(sentence_lines: list[str]) -> tuple[list[str], Counter[str]]:
    counts: Counter[str] = Counter()
    comment_lines: list[str] = []
    tokens: list[Token] = []

    for line in sentence_lines:
        if line.startswith("#"):
            comment_lines.append(fix_text_comment(line, counts))
        elif line.strip():
            tokens.append(Token.from_line(line))

    original_lines = [*sentence_lines]
    tokens = merge_split_ellipsis(tokens, counts)
    for token in tokens:
        fix_token(token, counts)
    fix_final_punct(tokens, counts)
    renumber_tokens(tokens)

    updated_lines = [*comment_lines, *(token.to_line() for token in tokens)]
    if updated_lines != original_lines:
        counts["sentences_changed"] += 1
    return updated_lines, counts


def split_sentences(text: str) -> list[list[str]]:
    sentences: list[list[str]] = []
    current: list[str] = []
    for line in text.splitlines():
        if line:
            current.append(line)
            continue
        sentences.append(current)
        current = []
    if current:
        sentences.append(current)
    return sentences


def fix_conllu_text(text: str) -> tuple[str, Counter[str]]:
    counts: Counter[str] = Counter()
    fixed_sentences: list[str] = []
    for sentence_lines in split_sentences(text):
        updated_lines, sentence_counts = fix_sentence(sentence_lines)
        counts.update(sentence_counts)
        fixed_sentences.append("\n".join(updated_lines))
    fixed_text = "\n\n".join(fixed_sentences)
    if text.endswith("\n"):
        fixed_text += "\n"
    return fixed_text, counts


def iter_conllu_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix == ".conllu":
            files.append(path)
            continue
        if path.is_dir():
            files.extend(sorted(path.glob("*.conllu")))
    return sorted(files)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply requested UD-CHILDES-BG_cs_and_cds cleanup checks."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[Path("UD-CHILDES-BG_cs_and_cds")],
        help="CoNLL-U file(s) or directories to process.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write the fixes back to disk. Without this flag, run in dry-run mode.",
    )
    args = parser.parse_args()

    files = iter_conllu_files(args.paths)
    if not files:
        raise SystemExit("No .conllu files found.")

    total_counts: Counter[str] = Counter()
    changed_files = 0

    for path in files:
        original = path.read_text(encoding="utf-8")
        fixed, counts = fix_conllu_text(original)
        total_counts.update(counts)
        if fixed != original:
            changed_files += 1
            if args.write:
                path.write_text(fixed, encoding="utf-8")

    mode = "write" if args.write else "dry-run"
    print(f"Mode: {mode}")
    print(f"Files scanned: {len(files)}")
    print(f"Files changed: {changed_files}")
    for key in sorted(total_counts):
        print(f"{key}: {total_counts[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
