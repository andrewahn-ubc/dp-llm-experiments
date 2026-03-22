from datasets import load_dataset

# ds_t = load_dataset(
#     "allenai/wildjailbreak",
#     "train",
#     delimiter="\t",
#     keep_default_na=False,
# )

# ds_e = load_dataset(
#     "allenai/wildjailbreak",
#     "eval",
#     delimiter="\t",
#     keep_default_na=False,
# )

# ds_t["train"].to_csv("wildjailbreak_train.csv", index=False)
# ds_e["train"].to_csv("wildjailbreak_eval.csv", index=False)

import pandas as pd

splits = {'harmful': 'data/harmful-behaviors.csv', 'benign': 'data/benign-behaviors.csv'}
df = pd.read_csv("hf://datasets/JailbreakBench/JBB-Behaviors/" + splits["harmful"])
df.to_csv("jbb_behaviors.csv", index=False)