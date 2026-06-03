from __future__ import annotations

import argparse
import sys
from pathlib import Path
import re

SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_BULSTEM_REPO = SCRIPT_DIR.parent / "bulstem-py"
WORD_PATTERN = re.compile(r"[^\W\d_]+", re.UNICODE)


def import_bulstemmer():
    import_error = None

    try:
        from bulstem.stem import BulStemmer

        return BulStemmer, "installed package"
    except ImportError as exc:
        import_error = exc

    if LOCAL_BULSTEM_REPO.is_dir():
        local_repo = str(LOCAL_BULSTEM_REPO)
        if local_repo not in sys.path:
            sys.path.insert(0, local_repo)

        try:
            from bulstem.stem import BulStemmer

            return BulStemmer, f"local checkout at {LOCAL_BULSTEM_REPO}"
        except ImportError as exc:
            import_error = exc

    raise ImportError(
        "Could not import 'bulstem' from the active Python environment or the "
        f"local checkout at {LOCAL_BULSTEM_REPO}."
    ) from import_error


def build_stemmer(rules: str, min_freq: int, left_context: int, encoding: str):
    BulStemmer, source = import_bulstemmer()
    stemmer = BulStemmer.from_file(
        rules,
        encoding=encoding,
        min_freq=min_freq,
        left_context=left_context,
    )
    return stemmer, source


def stem_line(line: str, stemmer) -> str:
    return WORD_PATTERN.sub(lambda match: stemmer.stem(match.group(0)), line)


def stem_file(input_path: Path, output_path: Path, stemmer, encoding: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding=encoding, newline="") as infile, output_path.open(
        "w", encoding=encoding, newline=""
    ) as outfile:
        for raw_line in infile:
            content = raw_line.rstrip("\r\n")
            line_ending = raw_line[len(content) :]
            outfile.write(stem_line(content, stemmer) + line_ending)


def run_demo(stemmer, source: str) -> int:
    test_words = [
        "книгите",
        "книга",
        "кучето",
        "кучета",
        "момчето",
        "момчетата",
        "хубав",
        "хубава",
        "хубавите",
        "учител",
        "учители",
        "играя",
        "играеха",
        "децата",
    ]

    print("Imported bulstem successfully.")
    print(f"Loaded stemmer from: {source}")
    print("-" * 60)
    print("Testing stemming:")
    for word in test_words:
        print(f"{word} -> {stemmer.stem(word)}")

    print("-" * 60)
    print(
        "Batch usage: python3 test_bulstem.py input.txt output.txt "
        "[--rules stem-context-2 --min-freq 2 --left-context 2]"
    )
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test bulstem and optionally stem a UTF-8 text file with one "
            "sentence per line."
        )
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        type=Path,
        help="Input text file with one sentence per line.",
    )
    parser.add_argument(
        "output_file",
        nargs="?",
        type=Path,
        help="Output text file for the stemmed sentences.",
    )
    parser.add_argument(
        "--rules",
        default="stem-context-2",
        help="Predefined rule set name or a path to a BulStem rules file.",
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=2,
        help="Minimum rule frequency to keep.",
    )
    parser.add_argument(
        "--left-context",
        type=int,
        default=2,
        help="Number of leading characters that should not be stemmed.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Input and output text encoding.",
    )
    args = parser.parse_args(argv)

    if (args.input_file is None) != (args.output_file is None):
        parser.error("Provide both input_file and output_file, or neither.")

    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    try:
        stemmer, source = build_stemmer(
            rules=args.rules,
            min_freq=args.min_freq,
            left_context=args.left_context,
            encoding=args.encoding,
        )
    except Exception as exc:
        print("Could not initialize bulstem.", file=sys.stderr)
        print(f"Python executable: {sys.executable}", file=sys.stderr)
        print(f"Import/setup error: {exc}", file=sys.stderr)
        return 1

    if args.input_file is None:
        return run_demo(stemmer, source)

    try:
        stem_file(args.input_file, args.output_file, stemmer, args.encoding)
    except Exception as exc:
        print("Could not stem the input file.", file=sys.stderr)
        print(f"Input file: {args.input_file}", file=sys.stderr)
        print(f"Output file: {args.output_file}", file=sys.stderr)
        print(f"File processing error: {exc}", file=sys.stderr)
        return 1

    print(f"Loaded stemmer from: {source}")
    print(f"Stemmed sentences written to: {args.output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
