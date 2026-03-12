import argparse
from pathlib import Path

import pandas as pd

# Keep keys to fields that should stay unchanged between files.
# Do NOT include corrected Bulgarian columns in the key.
MERGE_COLS = ["Name", "Age", "Participant", "Utterance", "UttLen"]


def non_empty_mask(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().ne("")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge manually corrected rows into the full LabLing dataset "
            "without many-to-many row explosion."
        )
    )
    parser.add_argument(
        "--full_csv",
        default=(
            "/Users/milamarcheva/Desktop/morphemic_tokenisation/data/labling_dfs/"
            "LabLing_longitudinal_all_utterances.csv"
        ),
        help="Path to full dataset CSV.",
    )
    parser.add_argument(
        "--corrected_csv",
        default=(
            "/Users/milamarcheva/Desktop/morphemic_tokenisation/data/labling_dfs/"
            "LabLing_sample_stratified_corrected.csv"
        ),
        help="Path to corrected sample CSV.",
    )
    parser.add_argument(
        "--output_csv",
        default=(
            "/Users/milamarcheva/Desktop/morphemic_tokenisation/data/labling_dfs/"
            "LabLing_longitudinal_with_manual.csv"
        ),
        help="Path to merged output CSV.",
    )
    parser.add_argument(
        "--fill_empty_manual_from_cleaned",
        action="store_true",
        help=(
            "If UtterancesCyrillicCleanedNormalised is empty in corrected CSV, "
            "fill Manually_corrected from UtterancesCyrillicCleaned."
        ),
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    full_path = Path(args.full_csv)
    corrected_path = Path(args.corrected_csv)
    out_path = Path(args.output_csv)

    print("Loading full dataset...")
    df_full = pd.read_csv(full_path)
    print(f"Full dataset loaded: {len(df_full)} rows")

    print("\nLoading corrected sample...")
    df_corr = pd.read_csv(corrected_path)
    print(f"Corrected dataset loaded: {len(df_corr)} rows")

    required_corr_cols = set(MERGE_COLS)
    missing_corr = required_corr_cols - set(df_corr.columns)
    if missing_corr:
        raise ValueError(f"Missing required columns in corrected CSV: {sorted(missing_corr)}")

    missing_full = set(MERGE_COLS) - set(df_full.columns)
    if missing_full:
        raise ValueError(f"Missing required columns in full CSV: {sorted(missing_full)}")

    # Prepare one manual value per corrected row.
    # Prefer explicit manual corrections when present.
    df_corr = df_corr.copy()
    if "Manually_corrected" in df_corr.columns:
        df_corr["Manually_corrected"] = df_corr["Manually_corrected"]
    elif "UtterancesCyrillicCleanedNormalised" in df_corr.columns:
        df_corr["Manually_corrected"] = df_corr["UtterancesCyrillicCleanedNormalised"]
    else:
        raise ValueError(
            "Corrected CSV must contain either 'Manually_corrected' or "
            "'UtterancesCyrillicCleanedNormalised'."
        )
    if args.fill_empty_manual_from_cleaned:
        if "UtterancesCyrillicCleaned" not in df_corr.columns:
            raise ValueError(
                "--fill_empty_manual_from_cleaned requires UtterancesCyrillicCleaned in corrected CSV."
            )
        empty_manual = ~non_empty_mask(df_corr["Manually_corrected"])
        df_corr.loc[empty_manual, "Manually_corrected"] = df_corr.loc[
            empty_manual, "UtterancesCyrillicCleaned"
        ]

    # Add duplicate rank to force one-to-one mapping for repeated identical keys.
    df_full = df_full.copy()
    df_corr = df_corr.copy()
    df_full["_dup_rank"] = df_full.groupby(MERGE_COLS, dropna=False).cumcount()
    df_corr["_dup_rank"] = df_corr.groupby(MERGE_COLS, dropna=False).cumcount()

    corr_for_merge = df_corr[MERGE_COLS + ["_dup_rank", "Manually_corrected"]]

    print("\nStarting one-to-one merge...")
    merged = df_full.merge(
        corr_for_merge,
        on=MERGE_COLS + ["_dup_rank"],
        how="left",
        validate="one_to_one",
    ).drop(columns=["_dup_rank"])

    # Stats
    total_rows = len(merged)
    sample_rows = len(df_corr)
    corrected_rows_non_empty = non_empty_mask(merged["Manually_corrected"]).sum()
    sample_manual_non_empty = non_empty_mask(df_corr["Manually_corrected"]).sum()

    print("\n====== MERGE REPORT ======")
    print("Output rows:", total_rows)
    print("Full CSV rows:", len(df_full))
    print("Corrected sample rows:", sample_rows)
    print("Corrected sample rows with non-empty manual text:", sample_manual_non_empty)
    print("Output rows with non-empty Manually_corrected:", corrected_rows_non_empty)
    print("Coverage:", round(corrected_rows_non_empty / total_rows * 100, 3), "%")

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print("\nSaving output CSV...")
    merged.to_csv(out_path, index=False)

    print("\nDONE.")
    print("Saved to:", out_path)


if __name__ == "__main__":
    main()
