import pandas as pd
import numpy as np

df = pd.read_csv("./harmbench_behaviors_text_all.csv")

df["Behavior"] = df["ContextString"].fillna("").astype(str) + np.where(df["ContextString"].notna(), "\n\n", "") + df["Behavior"].fillna("").astype(str)

df = df.drop(columns=["FunctionalCategory", "SemanticCategory", "Tags", "ContextString", "BehaviorID"])
df = df.rename(columns={"Behavior": "goal"})
df.to_csv("harmbench_trimmed.csv", index=False)