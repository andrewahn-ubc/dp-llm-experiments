import pandas as pd
import glob

files = glob.glob("./test_dataset_*.csv")

dfs = []
for f in files:
    dfs.append(pd.read_csv(f, engine="python"))

df = pd.concat(dfs, ignore_index=True)

df.to_csv(
    "./a_autodan_test.csv",
    index=False,
    quoting=1  
)
