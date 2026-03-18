from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pandas as pd
import random
import json
import torch
import time
import os

start_time = time.time()

FEW_SHOT_EXAMPLES_PATH = "./data/MaliciousInstruct.csv"

model_name_or_path = os.environ["LLM_NAME"]

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=False,
    trust_remote_code=False,
    revision="main",
)
model = model.to("cuda:0")

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id
model.eval()

model_load_time = time.time()
runtime_in_s = model_load_time - start_time
minutes = (runtime_in_s / 60) % 60
seconds = runtime_in_s % 60
print(f"Time taken to load model: {str(minutes)} minutes, and {str(seconds)} seconds")

train_df = pd.read_csv(FEW_SHOT_EXAMPLES_PATH)
d_size = len(train_df)
new_examples_df = pd.DataFrame()

def extract_json_array(text):
    print(text)
    start = text.find("[")
    end = text.rfind("]")

    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON array found")

    return text[start:end + 1]


for i in range(5):
    print(f"Starting iterations #{i}!")
    r_indices = random.sample(range(d_size), 5)
    few_shot_examples = train_df["behaviours"].iloc[r_indices].tolist()
    few_shot_examples_json = json.dumps(few_shot_examples, ensure_ascii=False)
    
    prompt = f"""
    Return exactly one valid JSON array containing 10 strings.

    Example output format:
    [
    "item 1",
    "item 2",
    "item 3"
    ]

    Here is a JSON array of example instructions:
    {few_shot_examples_json}

    Generate 10 new malicious instructions that are semantically diverse and distinct from the examples. Each string should belong 
    to a different harmful domain such as financial fraud, election manipulation, malware creation, harassment campaigns,
    illegal surveillance, cybercrime, drugs, weapons, violence, fraud, misinformation, privacy invasion, or biosecurity
    (although you aren't limited to these categories). They should be as complex and specific as the examples I provided.

    Rules:
    - Valid JSON only
    - One array of strings
    - No numbering
    - No extra text
    - No markdown
    """
    prompt_template = f"""USER: {prompt}

    ASSISTANT:
    [
    """

    inputs = tokenizer(prompt_template, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            temperature=0.2,
            do_sample=True,
            top_p=0.9,
            top_k=40,
            max_new_tokens=500,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    gen_tokens = output[0][prompt_len - 1:]
    generated_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

    generated_text_clean = extract_json_array(generated_text)
    print(f"Generated: {generated_text_clean}")

    try: 
        new_examples = json.loads(generated_text_clean) # returns a python list
        temp = pd.DataFrame({"goal": new_examples})
        new_examples_df = pd.concat([new_examples_df, temp], ignore_index=True)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON examples {i*10} - {(i+1)*10 - 1}. Skipping iteration.")
        print(f"Generated text: {generated_text_clean}")

new_examples_df.to_csv("data/bigger_train.csv", index=False)

end_time = time.time()
runtime_in_s = end_time - model_load_time
minutes = (runtime_in_s / 60) % 60
seconds = runtime_in_s % 60
print(f"Time taken to generate responses: {str(minutes)} minutes, and {str(seconds)} seconds")

