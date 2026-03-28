import pandas as pd
import glob

files = glob.glob("../train_dataset*.csv")

dfs = []
for f in files:
    dfs.append(pd.read_csv(f, engine="python"))

df = pd.concat(dfs, ignore_index=True)

df.to_csv(
    "./autodan_train_dataset.csv",
    index=False,
    quoting=1  
)
