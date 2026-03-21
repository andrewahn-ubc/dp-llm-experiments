import pandas as pd

input = "./wildjailbreak_train.csv"
output = "./wildjailbreak_train_1k.csv"

df = pd.read_csv(input)
df = df[df["data_type"] == "vanilla_harmful"].head(1000)
df = df.drop(columns=["data_type", "completion", "adversarial"])
df = df.rename(columns={"vanilla" : "goal"})
df.to_csv(output, index = False)