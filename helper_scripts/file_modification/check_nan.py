import pandas as pd

# Load the CSV
df = pd.read_csv("train.csv")

# 1. Check if ANY NaNs exist in the entire dataset
if df.isna().any().any():
    print("There are NaN values in the dataset.\n")
else:
    print("No NaN values found.\n")

# 2. Count NaNs per column
print("NaN count per column:")
print(df.isna().sum())
print()

# 3. Show rows that contain at least one NaN
nan_rows = df[df.isna().any(axis=1)]

print(f"Number of rows with NaNs: {len(nan_rows)}\n")

# Optional: save those rows for inspection
nan_rows.to_csv("rows_with_nans.csv", index=False)

# Optional: print first few problematic rows
print("Sample rows with NaNs:")
print(nan_rows.head())