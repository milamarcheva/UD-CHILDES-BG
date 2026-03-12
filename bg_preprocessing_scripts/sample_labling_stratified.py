import pandas as pd

# Paths
csv_path = "/Users/milamarcheva/Desktop/morphemic_tokenisation/data/labling_dfs/LabLing_longitudinal_with_manual.csv"
out_path = "/Users/milamarcheva/Desktop/morphemic_tokenisation/data/labling_dfs/LabLing_sample_stratified.csv"

# Load
df = pd.read_csv(csv_path)

print("Original size:", len(df))

# --------------------------------------------------
# Remove utterances that are only "xxx" (any case / punctuation)
# --------------------------------------------------

df = df[
    ~df["Utterance"]
    .astype(str)
    .str.lower()
    .str.replace(r"[^\w]", "", regex=True)
    .eq("xxx")
]

print("After removing xxx:", len(df))

# --------------------------------------------------
# Split: Child vs CDS
# --------------------------------------------------

df_child = df[df["Name"] == df["Participant"]]
df_cds = df[df["Name"] != df["Participant"]]

print("Child:", len(df_child))
print("CDS:", len(df_cds))

# --------------------------------------------------
# Take 5% from each
# --------------------------------------------------

def stratified_sample(df, frac=0.05):
    return (
        df
        .groupby(["Name", "Age"], group_keys=False)
        .apply(lambda x: x.sample(max(1, int(len(x) * frac)), random_state=42))
    )


def fill_manual_from_cleaned_match(
    df,
    cleaned_col="UtterancesCyrillicCleaned",
    normalised_col="UtterancesCyrillicCleanedNormalised",
    manual_col="Manually_corrected",
    todo_value="TODO",
):
    if cleaned_col not in df.columns or normalised_col not in df.columns:
        missing = [c for c in [cleaned_col, normalised_col] if c not in df.columns]
        raise KeyError(f"Missing required column(s): {missing}")

    if manual_col not in df.columns:
        df[manual_col] = ""

    cleaned = df[cleaned_col].fillna("").astype(str).str.strip()
    normalised = df[normalised_col].fillna("").astype(str).str.strip()
    manual = df[manual_col].fillna("").astype(str).str.strip()

    manual_empty = manual.eq("") | manual.str.lower().isin({"nan", "none", "null"})
    equal_mask = cleaned.eq(normalised)

    copy_mask = manual_empty & equal_mask
    todo_mask = manual_empty & ~equal_mask

    df.loc[copy_mask, manual_col] = df.loc[copy_mask, normalised_col]
    df.loc[todo_mask, manual_col] = todo_value

    copied_count = int(copy_mask.sum())
    todo_count = int(todo_mask.sum())
    kept_count = int((~manual_empty).sum())
    print(
        f"Updated '{manual_col}': kept existing {kept_count}, "
        f"copied {copied_count} from '{normalised_col}', set TODO in {todo_count}."
    )
    return df

sample_child = stratified_sample(df_child, frac=0.105)
sample_cds = stratified_sample(df_cds, frac=0.105)

sample = pd.concat([sample_child, sample_cds]).copy()

# child_n = int(len(df_child) * 0.05)
# cds_n = int(len(df_cds) * 0.05)

# print("Sampling child:", child_n)
# print("Sampling CDS:", cds_n)

# sample_child = df_child.sample(n=child_n, random_state=42)
# sample_cds = df_cds.sample(n=cds_n, random_state=42)

# # --------------------------------------------------
# # Combine (no shuffle — keeps blocks)
# # --------------------------------------------------

# sample = pd.concat([sample_child, sample_cds])

print("Total sample:", len(sample))

# --------------------------------------------------
# Drop unwanted columns
# --------------------------------------------------

cols_to_drop = [
    "TokenisedUtt",
    "TokenisedUttLower",
    "TokenisedUttNoPunct",
    "TokenisedUttCyr",
    "berkeley_parse"
]

existing = [c for c in cols_to_drop if c in sample.columns]
sample.drop(columns=existing, inplace=True)

print("Dropped columns:", existing)

# --------------------------------------------------
# Set manual correction based on cleaned/normalised equality
# --------------------------------------------------

sample = fill_manual_from_cleaned_match(sample)

# --------------------------------------------------
# Save
# --------------------------------------------------

sample.to_csv(out_path, index=False)

print("Wrote:", out_path)
