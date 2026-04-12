import pandas as pd
import glob

files = glob.glob("./train_pair_output_*.csv")

dfs = []
for f in files:
    dfs.append(pd.read_csv(f, engine="python"))

df = pd.concat(dfs, ignore_index=True)

df.to_csv(
    "./pair_train.csv",
    index=False,
    quoting=1  
)
