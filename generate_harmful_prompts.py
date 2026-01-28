from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import random
import json
import torch

FEW_SHOT_EXAMPLES_PATH = "./data/train.csv"

# Load main LLM
tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
model = AutoModelForCausalLM.from_pretrained(LLM_NAME, device_map="auto")
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id
model.eval()

train_df = pd.read_csv(FEW_SHOT_EXAMPLES_PATH)
d_size = len(train_df)
new_examples_df = pd.DataFrame()

def extract_json_array(text):
    start = text.find("[")
    end = text.rfind("]")

    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON array found")

    return text[start:end + 1]

for i in range(10):
    r_indices = random.sample(range(d_size), 5)
    few_shot_examples = train_df["Original Prompt"].iloc[r_indices].tolist()
    few_shot_examples_json = json.dumps(few_shot_examples, ensure_ascii=False)
    prompt = f"""
        You will be given a JSON array of harmful instructions. 

        {few_shot_examples_json}

        Generate exactly 10 new harmful instructions.
        Output ONLY a JSON array of strings.
        No extra text.
        """
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda") 
    with torch.no_grad(): 
        output = model.generate(**inputs, max_new_tokens=300, do_sample=True, temperature=0.7, top_p=0.9) 
        prompt_len = inputs["input_ids"].shape[1]
        gen_tokens = output[0][prompt_len:]
        generated_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    generated_text_clean = extract_json_array(generated_text)

    try: 
        new_examples = json.loads(generated_text_clean) # returns a python list
        temp = pd.DataFrame({"Original Prompt": new_examples})
        new_examples_df = pd.concat([new_examples_df, temp], ignore_index=True)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON examples {i*10} - {(i+1)*10 - 1}. Skipping iteration.")
        print(f"Generated text: {generated_text_clean}")

new_examples_df.to_csv("data/bigger_train.csv", index=False)

