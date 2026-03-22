import sys, pandas as pd

f1, f2, f3 = "./eval_advbench.csv", "eval_harmbench.csv", "eval_jbb.csv"
out = "combined.csv"

df1 = pd.read_csv(f1)
df2 = pd.read_csv(f2)
df3 = pd.read_csv(f3)

df1["dataset"] = "advbench"
df2["dataset"] = "harmbench"
df3["dataset"] = "jailbreakbench"

df = pd.concat([df1, df2, df3], ignore_index=True)
df = df[df["goal"].notna()]
df["goal"] = df["goal"].astype(str).str.strip()
df = df[df["goal"] != ""]

df.to_csv(out, index=False)
print(f"Saved {len(df)} rows to {out}")