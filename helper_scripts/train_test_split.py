import pandas as pd

df = pd.read_csv("data/train.csv")
train_df = df.iloc[:200]

second_train_df = df.iloc[200:]

train_df.to_csv("data/train.csv", index=False)
second_train_df.to_csv("data/train2.csv", index=False)

