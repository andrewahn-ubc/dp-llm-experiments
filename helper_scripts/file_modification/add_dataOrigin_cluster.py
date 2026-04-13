import pandas as pd

train_df = pd.read_csv("train.csv")
jb_df = pd.read_csv("all_jb_fams_train_response.csv")

df_new = pd.concat([jb_df, train_df.drop(columns=["goal", "target"])], axis=1)
df_new.to_csv("all_jb_fams_train_response_full.csv", index=False)