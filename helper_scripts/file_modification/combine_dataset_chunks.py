import pandas as pd
import glob

files = glob.glob("./missing_autodan_*.csv")

dfs = []
for f in files:
    dfs.append(pd.read_csv(f, engine="python"))

df = pd.concat(dfs, ignore_index=True)

df.to_csv(
    "./missing_autodan_complete.csv",
    index=False,
    quoting=1  
)
