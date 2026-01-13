import nanogcg 
import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer 
import pandas as pd 
import os
import math

# Set up input/output files 
MODEL_PATH = "/home/taegyoem/scratch/llama2_7b_chat_hf" 
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

# Load model 
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH) 
model = AutoModelForCausalLM.from_pretrained( MODEL_PATH, torch_dtype=torch.float16, trust_remote_code=True, local_files_only=True ).to("cuda") 
model.eval() 

for index, (_,row) in enumerate(input_df.iterrows()): 
    # Original Prompt + Response
    prompt = row["goal"]
    messages = [{"role": "user", "content": prompt}]
    prompt_formatted = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    # prompt_formatted = f"[INST] User: {prompt} [/INST] Agent: "

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
    # jb_prompt_formatted = f"<s>[INST] {jb_prompt} [/INST]"


    jb_inputs = tokenizer(jb_prompt_formatted, return_tensors="pt").to("cuda") 
    with torch.no_grad(): 
        jb_output = model.generate(**jb_inputs, max_new_tokens=30, temperature=0.0, do_sample=False) 
    jb_decoded = tokenizer.decode(jb_output[0], skip_special_tokens=True) 
    jb_prompt_for_slicing = jb_prompt_formatted.replace("<s>", "").replace("</s>", "")
    jb_answer = jb_decoded[len(jb_prompt_for_slicing):]

    # Update output 
    output_df.loc[index] = [prompt, answer, jb_prompt, jb_answer] 

# Convert output dataframe into CSV format 
output_df.to_csv(f"data/gcg_output_{TASK_ID}.csv", index=False)