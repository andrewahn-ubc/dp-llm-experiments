import sys, pandas as pd

f1, f2 = "./harmbench_training_trimmed.csv", "harmbench_trimmed.csv"
out = "harmbench.csv"

df1 = pd.read_csv(f1)
df2 = pd.read_csv(f2)

df = pd.concat([df1, df2], ignore_index=True)
df = df[df["goal"].notna()]
df["goal"] = df["goal"].astype(str).str.strip()
df = df[df["goal"] != ""]

df.to_csv(out, index=False)
print(f"Saved {len(df)} rows to {out}")