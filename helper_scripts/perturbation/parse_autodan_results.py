import json
import csv
import pandas as pd

csv_data = {
    "Original Prompt": [],
    "Original Response": [],
    "Perturbed Prompt": [],
    "Perturbed Response": [],
    "Jailbreak Success": []
}

with open("autodan_first_10.json", "r") as f:
    data = json.load(f)
    
    for i in range(len(data)):
        initial_prompt = data[f"{i}"]["goal"]
        csv_data["Original Prompt"].append(initial_prompt)
        csv_data["Original Response"].append(data[f"{i}"]["log"]["respond"][0])
        csv_data["Perturbed Prompt"].append(initial_prompt + data[f"{i}"]["final_suffix"])
        csv_data["Perturbed Response"].append(data[f"{i}"]["log"]["respond"][-1])
        csv_data["Jailbreak Success"].append(data[f"{i}"]["is_success"])
    
df = pd.DataFrame(csv_data)
df.to_csv("autodan_first_10.csv", index=False)