import nanogcg 
import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer 
import pandas as pd 
import os
import math
import argparse
import time

start_time = time.time()

# Set up input/output files 
MODEL_PATH = "/home/taegyoem/scratch/llama2_7b_chat_hf" 

# Get input csv path and output file name
parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str)
parser.add_argument("--save_suffix", type=str, default="default")
args = parser.parse_args()
data_path = args.input_file
TASK_ID = args.save_suffix

# Create input/output dataframes
input_df = pd.read_csv(data_path)
output_df = pd.DataFrame(columns = ["goal", "original response", "perturbed goal", "perturbed response"]) 

# Load model 
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH) 
model = AutoModelForCausalLM.from_pretrained( MODEL_PATH, torch_dtype=torch.float16, trust_remote_code=True, local_files_only=True ).to("cuda") 
model.eval() 

model_load_time = time.time()
runtime_in_s = model_load_time - start_time
minutes = (runtime_in_s / 60) % 60
seconds = runtime_in_s % 60
print(f"\nTime taken to load model: {str(int(minutes))} minutes, and {str(int(seconds))} seconds\n")

time_since_last_prompt = model_load_time

for index, (_,row) in enumerate(input_df.iterrows()): 
    # Original Prompt + Response
    prompt = row["goal"]
    messages = [{"role": "user", "content": prompt}]
    prompt_formatted = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt_formatted, return_tensors="pt").to("cuda") 
    with torch.no_grad(): 
        output = model.generate(**inputs, max_new_tokens=30, temperature=0.0, do_sample=False) 
    decoded = tokenizer.decode(output[0], skip_special_tokens=True) 
    prompt_for_slicing = prompt_formatted.replace("<s>", "").replace("</s>", "")
    answer = decoded[len(prompt_for_slicing):]

    # Create Perturbed Prompt + Response 
    target = row["target"] 
    result = nanogcg.run(model, tokenizer, prompt, target) 
    jb_prompt = prompt + result.best_string
    messages = [{"role": "user", "content": jb_prompt}]
    jb_prompt_formatted = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    jb_inputs = tokenizer(jb_prompt_formatted, return_tensors="pt").to("cuda") 
    with torch.no_grad(): 
        jb_output = model.generate(**jb_inputs, max_new_tokens=30, temperature=0.0, do_sample=False) 
    jb_decoded = tokenizer.decode(jb_output[0], skip_special_tokens=True) 
    jb_prompt_for_slicing = jb_prompt_formatted.replace("<s>", "").replace("</s>", "")
    jb_answer = jb_decoded[len(jb_prompt_for_slicing):]

    # Update output 
    output_df.loc[index] = [prompt, answer, jb_prompt, jb_answer] 

    curr_time = time.time()
    runtime_in_s = curr_time - time_since_last_prompt
    minutes = (runtime_in_s / 60) % 60
    seconds = runtime_in_s % 60
    print(f"\nTime taken for this prompt: {str(int(minutes))} minutes, and {str(int(seconds))} seconds")

# Convert output dataframe into CSV format 
output_df.to_csv(f"/home/taegyoem/scratch/dp-llm-experiments/official_data/gcg_output_{TASK_ID}.csv", index=False)

end_time = time.time()
runtime_in_s = int(end_time - start_time)
hours = runtime_in_s // 3600
minutes = (runtime_in_s / 60) % 60
seconds = runtime_in_s % 60
print(f"\nTotal time taken: {str(hours)} hours, {str(int(minutes))} minutes, and {str(int(seconds))} seconds")
