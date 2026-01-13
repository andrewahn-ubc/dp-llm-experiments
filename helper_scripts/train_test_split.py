import pandas as pd

df = pd.read_csv("./data/gcg_output_total.csv")

train_df = df.iloc[:400]

test_df = df.iloc[400:]

train_df.to_csv("./data/train.csv", index=False)
test_df.to_csv("./data/test.csv", index=False)

