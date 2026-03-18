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
print(f"\nTime taken to load model: {str(minutes)} minutes, and {str(seconds)} seconds\n")

train_df = pd.read_csv(FEW_SHOT_EXAMPLES_PATH)
d_size = len(train_df)
new_examples_df = pd.DataFrame()

harmful_categories = [
    "financial fraud", 
    "election manipulation", 
    "harassment campaigns", 
    "illegal surveillance", 
    "drugs", 
    "weapons", 
    "fraud", 
    "misinformation", 
    "privacy invasion", 
    "biosecurity",
    "multi-level marketing",
    "physical assault techniques",
    "scamming the elderly",
    "spreading illnesses",
    "tricking airport security",
    "creating fake ID",
    "committing arson",
    "tax evasion",
    "jailbreaking LLMs",
    "stalking women",
    "religiously-motivated hate crimes",
    "racially-motivated hate crimes",
    "how to lie in court",
    "animal cruelty",
    "violating labor laws",
    "rape",
    "sexual abuse",
    "murder",
    "sexual harassment",
    "pedophilia",
    "child trafficking",
    "kidnapping"
]

def extract_json_array(text):
    print(text)
    start = text.find("[")
    end = text.rfind("]")

    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON array found")

    return text[start:end + 1]

ddef provide_harmful_categories(n = 5):
    r_indices = random.sample(range(len(harmful_categories)), 5)
    random_categories = []
    for i in r_indices:
        random_categories.append(harmful_categories[i])
    out = ""
    for cat in random_categories:
        out += cat 
        out += ", "
    out += "etc."
    return out

for i in range(5):
    print(f"Starting iterations #{i}!")
    r_indices = random.sample(range(d_size), 5)
    few_shot_examples = train_df["behaviours"].iloc[r_indices].tolist()
    few_shot_examples_json = json.dumps(few_shot_examples, ensure_ascii=False)

    print(f"\nThe few-shot examples this time were: {few_shot_example_json} \n")

    harmful_categories_string = provide_harmful_categories(5)
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

    Generate 10 new malicious instructions that are semantically diverse and 
    do NOT contain any of the same content as the examples above. 
    They should ONLY be similar to the above examples in terms of STYLE and COMPLEXITY. 
    Each string should belong to a different harmful domain such as {harmful_categories_string} 

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
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    generated_text_clean = extract_json_array(generated_text)
    print(f"\nGenerated: {generated_text_clean}\n")

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
print(f"\nTime taken to generate responses: {str(minutes)} minutes, and {str(seconds)} seconds")

