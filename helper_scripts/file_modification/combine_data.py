import pandas as pd
import glob

files = glob.glob("./data/gcg*.csv")

dfs = []
for f in files:
    dfs.append(pd.read_csv(f, engine="python"))

df = pd.concat(dfs, ignore_index=True)

df.to_csv(
    "./data/gcg_output_total.csv",
    index=False,
    quoting=1  # csv.QUOTE_ALL
)
