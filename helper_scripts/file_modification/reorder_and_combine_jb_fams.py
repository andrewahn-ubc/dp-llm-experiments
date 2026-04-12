import pandas as pd

# Load datasets
train = pd.read_csv("train.csv")  # contains: goal, target
gcg = pd.read_csv("gcg_train.csv")  # contains: goal, perturbed goal
pair = pd.read_csv("pair_train.csv")
autodan = pd.read_csv("autodan_train.csv")

# Rename perturbed columns for clarity
gcg = gcg.rename(columns={"perturbed goal": "GCG Variant"})
pair = pair.rename(columns={"perturbed goal": "PAIR Variant"})
autodan = autodan.rename(columns={"perturbed goal": "AutoDAN Variant"})

# Keep only relevant columns
gcg = gcg[["goal", "GCG Variant"]]
pair = pair[["goal", "PAIR Variant"]]
autodan = autodan[["goal", "AutoDAN Variant"]]

# Merge step by step
merged = train[["goal"]].copy()

merged = merged.merge(gcg, on="goal", how="left")
merged = merged.merge(autodan, on="goal", how="left")
merged = merged.merge(pair, on="goal", how="left")

# Save result
merged.to_csv("combined_dataset.csv", index=False)

print("Done. Saved to combined_dataset.csv")
print(merged.head())