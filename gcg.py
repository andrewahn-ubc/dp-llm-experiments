import nanogcg 
import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer 
import pandas as pd 
import os
import math

# Set up input/output files 
MODEL_PATH = "/home/taegyoem/scratch/llama2_7b" 
CSV_INPUT = "data/harmful_behaviors.csv" 

# Job Array
full_input_df = pd.read_csv(CSV_INPUT)  # full dataset
TASK_ID = int(os.environ["SLURM_ARRAY_TASK_ID"]) # Determine task ID
NUM_TASKS = 50
chunk_size = math.ceil(len(full_input_df) / NUM_TASKS)
# Compute start and end index for this task
start_idx = TASK_ID * chunk_size 
end_idx = min(start_idx + chunk_size, len(full_input_df)) 
# Slice the dataframe for this task
input_df = full_input_df.iloc[start_idx:end_idx]
print(f"Task {TASK_ID}: processing rows {start_idx} to {end_idx-1}")


output_df = pd.DataFrame(columns = ["Original Prompt", "Original Response", "Perturbed Prompt", "Perturbed Response"]) 
system_prompt = "You are a very concise assistant."

# Load model 
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH) 
model = AutoModelForCausalLM.from_pretrained( MODEL_PATH, torch_dtype=torch.float16, trust_remote_code=True, local_files_only=True ).to("cuda") 
model.eval() 

for index, (_,row) in enumerate(input_df.iterrows()): 
    # Original Prompt + Response
    prompt = row["goal"] 
    prompt = f"[SYSTEM]: {system_prompt}\n[USER]: {prompt}\n[ASSISTANT]:"

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda") 
    with torch.no_grad(): 
        output = model.generate( **inputs, max_new_tokens=30, temperature=0.0, do_sample=False) 
    decoded = tokenizer.decode(output[0], skip_special_tokens=True) 
    answer = decoded[len(prompt):] 

    # Create Perturbed Prompt + Response 
    target = "\n[ASSISTANT]:" + row["target"] 
    prompt = prompt[:-13]
    result = nanogcg.run(model, tokenizer, prompt, target) 
    jb_prompt = prompt + result.best_string + "\n[ASSISTANT]:"

    jb_inputs = tokenizer(jb_prompt, return_tensors="pt").to("cuda") 
    with torch.no_grad(): 
        jb_output = model.generate( **jb_inputs, max_new_tokens=30, temperature=0.0, do_sample=False) 
    jb_decoded = tokenizer.decode(jb_output[0], skip_special_tokens=True) 
    jb_answer = jb_decoded[len(jb_prompt):] 

    # Update output 
    output_df.loc[index] = [prompt, answer, jb_prompt, jb_answer] 

# Convert output dataframe into CSV format 
output_df.to_csv(f"data/gcg_output_{TASK_ID}.csv", index=False)











# import nanogcg
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import pandas as pd

# # Set up input/output files
# MODEL_PATH = "/home/taegyoem/scratch/llama2_7b_chat_hf"
# CSV_INPUT = "data/harmful_behaviors.csv"
# CSV_OUTPUT = "data/gcg_output.csv"
# full_input_df = pd.read_csv(CSV_INPUT) # contains original prompts and targets
# input_df = full_input_df.head(1) # let's only use the first 20 prompts for now
# results = []

# # Load model
# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# tokenizer.pad_token = tokenizer.eos_token
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_PATH,
#     torch_dtype=torch.float16,
#     trust_remote_code=True,
#     local_files_only=True
# ).to("cuda")
# model.eval()

# # Get Original Response (batched)
# prompts = input_df["goal"].tolist()
# batch_size = 1
# for i in range(0, len(prompts), batch_size):
#     batch_prompts = prompts[i:i+batch_size]
#     inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
#     inputs = {k: v.to("cuda") for k, v in inputs.items()}
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=30,      
#             temperature=0.0,       
#             do_sample=False,
#             pad_token_id=tokenizer.eos_token_id
#         )
#     decoded = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
#     responses = [decoded[index][len(prompt):] for index, prompt in enumerate(batch_prompts)]
#     for index, response in enumerate(responses):
#         results.append({
#             "Original Prompt": batch_prompts[index], 
#             "Original Response": response, 
#             "Perturbed Prompt": "", 
#             "Perturbed Response": ""
#         })

# # Get Perturbed Prompts (row by row)
# for index, row in input_df.iterrows():
#     # Original Prompt + Response
#     prompt = row["goal"]
#     inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
#     with torch.no_grad():
#         output = model.generate(
#             **inputs,
#             max_new_tokens=30,      
#             temperature=0.0,       
#             do_sample=False
#         )
#     decoded = tokenizer.decode(output[0], skip_special_tokens=True)
#     answer = decoded[len(prompt):]

#     # Create Perturbed Prompt + Response
#     target = row["target"]
#     result = nanogcg.run(model, tokenizer, prompt, target)
#     jb_prompt = prompt + result.best_string
#     jb_inputs = tokenizer(jb_prompt, return_tensors="pt").to("cuda")
#     with torch.no_grad():
#         jb_output = model.generate(
#             **jb_inputs,
#             max_new_tokens=30,      
#             temperature=0.0,       
#             do_sample=False
#         )
#     jb_decoded = tokenizer.decode(jb_output[0], skip_special_tokens=True)
#     jb_answer = decoded[len(jb_prompt):]

#     # Update results
#     results[index]["Perturbed Prompt"] = jb_prompt

# # Get Jailbreak Outputs (batched)
# jb_prompts = [results[i]["Perturbed Prompt"] for i in range(len(results))]
# for i in range(0, len(jb_prompts), batch_size):
#     batch_jb_prompts = jb_prompts[i:i+batch_size]
#     jb_inputs = tokenizer(batch_jb_prompts, return_tensors="pt", padding=True, truncation=True)
#     jb_inputs = {k: v.to("cuda") for k, v in jb_inputs.items()}
#     with torch.no_grad():
#         jb_outputs = model.generate(
#             **jb_inputs,
#             max_new_tokens=30,      
#             temperature=0.0,       
#             do_sample=False,
#             pad_token_id=tokenizer.eos_token_id
#         )
#     jb_decoded = [tokenizer.decode(o, skip_special_tokens=True) for o in jb_outputs]
#     jb_responses = [jb_decoded[index][len(jb_prompt):] for index, jb_prompt in enumerate(batch_jb_prompts)]
#     for index, jb_response in enumerate(jb_responses):
#         results[i + index]["Perturbed Response"] = jb_response

# # Update output file
# output_df = pd.DataFrame(results)
# output_df.to_csv(CSV_OUTPUT, index=False)

