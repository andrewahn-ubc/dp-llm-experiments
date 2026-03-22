import pandas as pd

df = pd.read_csv("data/train.csv")
df2 = pd.read_csv("data/train2.csv")

total_df = pd.concat([df, df2])

total_df.to_csv("data/train_new.csv", index=False)

