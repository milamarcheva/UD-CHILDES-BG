import pandas as pd

csv_path = "/Users/milamarcheva/Desktop/morphemic_tokenisation/data/LabLing_longitudinal_all_utterances_with_berkeley.csv"

df = pd.read_csv(csv_path)

# Split
df_child = df[df["Name"] == df["Participant"]]
df_cds = df[df["Name"] != df["Participant"]]


# Sample
sample_child = df_child.sample(n=50, random_state=42)
sample_cds = df_cds.sample(n=50, random_state=42)

# Combine
sample = pd.concat([sample_cds, sample_child])


cols_to_drop = [
    "TokenisedUtt",
    "TokenisedUttLower",
    "TokenisedUttNoPunct",
    "TokenisedUttCyr"
]

# Only drop columns that actually exist (avoids crashes)
existing = [c for c in cols_to_drop if c in sample.columns]

print("Dropping:", existing)

sample.drop(columns=existing, inplace=True)


# Write
out_path = "/Users/milamarcheva/Desktop/morphemic_tokenisation/data/LabLing_sample.csv"
sample.to_csv(out_path, index=False)

print("Wrote:", out_path)