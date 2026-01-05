import nanogcg
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

# Set up input/output files
MODEL_PATH = "/home/taegyoem/scratch/llama2_7b_chat_hf"
CSV_INPUT = "data/harmful_behaviors.csv"
CSV_OUTPUT = "data/gcg_output.csv"
full_input_df = pd.read_csv(CSV_INPUT) # contains original prompts and targets
input_df = full_input_df.head(1) # let's only use the first 20 prompts for now
results = []

# Load model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    local_files_only=True
).to("cuda")
model.eval()

# Get Original Response (batched)
prompts = input_df["goal"].tolist()
batch_size = 1
for i in range(0, len(prompts), batch_size):
    batch_prompts = prompts[i:i+batch_size]
    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,      
            temperature=0.0,       
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    decoded = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
    responses = [decoded[index][len(prompt):] for index, prompt in enumerate(batch_prompts)]
    for index, response in enumerate(responses):
        results.append({
            "Original Prompt": batch_prompts[index], 
            "Original Response": response, 
            "Perturbed Prompt": "", 
            "Perturbed Response": ""
        })

# Get Perturbed Prompts (row by row)
for index, row in input_df.iterrows():
    # Original Prompt + Response
    prompt = row["goal"]

    # Create Perturbed Prompt + Response
    target = row["target"]
    result = nanogcg.run(model, tokenizer, prompt, target)
    jb_prompt = prompt + result.best_string

    # Update results
    results[index]["Perturbed Prompt"] = jb_prompt

# Get Jailbreak Outputs (batched)
jb_prompts = [results[i]["Perturbed Prompt"] for i in range(len(results))]
for i in range(0, len(jb_prompts), batch_size):
    batch_jb_prompts = jb_prompts[i:i+batch_size]
    jb_inputs = tokenizer(batch_jb_prompts, return_tensors="pt", padding=True, truncation=True)
    jb_inputs = {k: v.to("cuda") for k, v in jb_inputs.items()}
    with torch.no_grad():
        jb_outputs = model.generate(
            **jb_inputs,
            max_new_tokens=30,      
            temperature=0.0,       
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    jb_decoded = [tokenizer.decode(o, skip_special_tokens=True) for o in jb_outputs]
    jb_responses = [jb_decoded[index][len(jb_prompt):] for index, jb_prompt in enumerate(batch_jb_prompts)]
    for index, jb_response in enumerate(jb_responses):
        results[i + index]["Perturbed Response"] = jb_response

# Update output file
output_df = pd.DataFrame(results)
output_df.to_csv(CSV_OUTPUT, index=False)

