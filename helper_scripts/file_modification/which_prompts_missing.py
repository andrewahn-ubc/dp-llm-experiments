import pandas as pd

# Load CSVs
train_df = pd.read_csv("train.csv")
autodan_df = pd.read_csv("autodan_train.csv")

# Extract goal columns
train_goals = train_df["goal"].astype(str)
autodan_goals = autodan_df["goal"].astype(str)

# Convert to sets for comparison
train_set = set(train_goals)
autodan_set = set(autodan_goals)

# Compute overlaps and differences
common = train_set.intersection(autodan_set)
only_in_train = train_set - autodan_set
only_in_autodan = autodan_set - train_set

# Print stats
print(f"Total in train.csv: {len(train_set)}")
print(f"Total in autodan_train.csv: {len(autodan_set)}")
print(f"Common prompts: {len(common)}")
print(f"Only in train.csv (missing from autodan): {len(only_in_train)}")
print(f"Only in autodan_train.csv (extra or mismatched): {len(only_in_autodan)}")

# Optionally save missing ones to files
pd.DataFrame({"missing_from_autodan": list(only_in_train)}).to_csv("missing_from_autodan.csv", index=False)
pd.DataFrame({"missing_from_train": list(only_in_autodan)}).to_csv("missing_from_train.csv", index=False)

print("\nSaved missing prompts to:")
print("- missing_from_autodan.csv")
print("- missing_from_train.csv")