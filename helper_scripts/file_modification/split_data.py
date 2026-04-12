import pandas as pd
import math

input = "./missing_from_autodan.csv"

df = pd.read_csv(input)
n = len(df)
chunk_size = 5
splits = math.ceil(n / chunk_size)

for i in range(splits):
    new_df = df.iloc[i*chunk_size : (i + 1)*chunk_size]

    if i < 10:
        j = "0" + str(i)
    else:
        j = str(i)

    new_df.to_csv(f"missing_autodan_{j}.csv", index=False)