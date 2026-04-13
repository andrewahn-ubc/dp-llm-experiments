import pandas as pd

df = pd.read_csv("./wildjailbreak_eval.csv")
rows = []

vanilla_benign_count = 250
adversarial_benign_count = 250

for idx, row in df.iterrows():
    if (vanilla_benign_count == 0 and adversarial_benign_count == 0):
        break

    if row["data_type"] == "vanilla_benign" and vanilla_benign_count > 0:
        rows.append(row)
        vanilla_benign_count = vanilla_benign_count - 1

    if row["data_type"] == "adversarial_benign" and adversarial_benign_count > 0:
        rows.append(row)
        adversarial_benign_count = adversarial_benign_count - 1

out = pd.DataFrame(rows)
out.to_csv("frr_test.csv", index=False)