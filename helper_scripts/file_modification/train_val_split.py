import pandas as pd

df = pd.read_csv("./all_jb_fams_train_response_full.csv")

train_df = df[df["goal_cluster"] == 0]
val_df = df[df["goal_cluster"] != 0]

train_df.to_csv("train.csv", index=False)
val_df.to_csv("validation.csv", index=False)