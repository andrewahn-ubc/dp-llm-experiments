import json
import csv
import pandas as pd

num_chunks = 25

for i in range(num_chunks):
    # if i in {19, 20}:
    #     continue
    dd = f"{i:02d}"
    input_path = f"./llama2_0_missing_autodan_train_{dd}.json"
    output_path = f"./missing_autodan_{dd}.csv"

    csv_data = {
        "goal": [],
        "original response": [],
        "perturbed goal": [],
        "perturbed response": [],
        "jailbreak success": []
    }

    with open(input_path, "r") as f:
        data = json.load(f)
        
        for i in range(len(data)):
            initial_prompt = data[f"{i}"]["goal"]
            csv_data["goal"].append(initial_prompt)
            csv_data["original response"].append(data[f"{i}"]["log"]["respond"][0])
            csv_data["perturbed goal"].append(initial_prompt + data[f"{i}"]["final_suffix"])
            csv_data["perturbed response"].append(data[f"{i}"]["log"]["respond"][-1])
            csv_data["jailbreak success"].append(data[f"{i}"]["is_success"])
        
    df = pd.DataFrame(csv_data)
    df.to_csv(output_path, index=False)