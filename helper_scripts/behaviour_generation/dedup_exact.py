import pandas as pd

data = "./combined.csv"
output = "./combined_deduped.csv"

def deduplicate_csv(input_path, output_path):
    df = pd.read_csv(input_path, usecols=["goal"])

    # Mark duplicates (True = will be dropped)
    dup_mask = df.duplicated(subset="goal", keep="first")

    # Print dropped rows
    dropped = df[dup_mask]
    print("Dropped rows:")
    for row in dropped["goal"]:
        print(row)

    # Keep only unique rows
    df = df[~dup_mask]

    df.to_csv(output_path, index=False)

    print(f"\nDone. {len(df)} unique rows saved.")

deduplicate_csv(data, output)
