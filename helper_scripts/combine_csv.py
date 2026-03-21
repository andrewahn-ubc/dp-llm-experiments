import sys, pandas as pd

f1, f2 = sys.argv[1], sys.argv[2]
out = sys.argv[3] if len(sys.argv) > 3 else "combined.csv"

df1 = pd.read_csv(f1)
df2 = pd.read_csv(f2)

df = pd.concat([df1, df2], ignore_index=True)
df = df[df["goal"].notna()]
df["goal"] = df["goal"].astype(str).str.strip()
df = df[df["goal"] != ""]

df.to_csv(out, index=False)
print(f"Saved {len(df)} rows to {out}")