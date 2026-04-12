from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import PeftModel
import pandas as pd
import torch
import time

start_time = time.time()

DEVICE = "cuda"
DTYPE = torch.float16
LLM_NAME = "/home/taegyoem/scratch/llama2_7b_chat_hf" 
INPUT_DATASET = "/home/taegyoem/scratch/official_data/all_jb_fams_train.csv"
OUTPUT_DATASET = "/home/taegyoem/scratch/official_data/all_jb_fams_train_response.csv"

# Load main LLM
tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
model = AutoModelForCausalLM.from_pretrained(LLM_NAME, torch_dtype=DTYPE, device_map="auto")

# Load data
df = pd.read_csv(INPUT_DATASET)

# Generate 30 tokens for each regular prompt using the LLM
for index, row in df.iterrows():
    # Original Prompt + Response
    prompt = row["Original Prompt"]
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},  
    ]
    prompt_formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_formatted, return_tensors="pt").to("cuda") 
    with torch.no_grad(): 
        output = model.generate(**inputs, max_new_tokens=30, temperature=None, top_p=None, do_sample=False) 
    input_length = inputs["input_ids"].shape[1]
    new_tokens = output[0][input_length:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Update output 
    df.loc[index, "Original Response"] = answer

# save results in a new dataframe and save it as a new csv file
df.to_csv(OUTPUT_DATASET, index=False)

end_time = time.time()
runtime_in_s = end_time - start_time
print(f"Time taken: {str(runtime_in_s / (60*60))} hours, {str(runtime_in_s / 60)} minutes, and {str(runtime_in_s)} seconds")