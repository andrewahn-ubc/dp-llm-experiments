# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch
# import os
# from huggingface_hub import snapshot_download

# MODEL_PATH = "/home/taegyoem/scratch/llama2_7b"  

# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_PATH,
#     torch_dtype=torch.float16,
#     device_map="auto",
#     low_cpu_mem_usage=True,
#     trust_remote_code=True,
#     local_files_only=True
# )

# prompt = "Who is the best basketball player ever?"

# inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# with torch.no_grad():
#     output = model.generate(
#         **inputs,
#         max_new_tokens=150,
#         temperature=0.7,
#     )

# result = tokenizer.decode(output[0], skip_special_tokens=True)

# with open("/home/taegyoem/scratch/llama2_7b_output.txt", "w") as f:
#     f.write(result)
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

MODEL_PATH = "/home/taegyoem/scratch/llama2_7b"
OUTPUT_FILE = "/home/taegyoem/scratch/llama2_7b_output.txt"
PROMPT_FILE = "prompts.json"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    local_files_only=True
)
model.eval()

def run_one_word(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=5,      
            temperature=0.0,       
            do_sample=False
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    answer = decoded[len(prompt):]
    # answer = decoded[len(prompt):].strip()

    # clean answer to one word; strip punctuation
    # answer = answer.split()[0] if answer else ""
    # answer = answer.strip(".,!?;:()[]{}\"'")

    return answer


with open(PROMPT_FILE, "r") as f:
    data = json.load(f)

results = []
num_boundary_cross = 0
num_total = len(data)


for key in data:
    entry = data[key]
    original_prompt = entry["original"]
    perturbed_prompt = entry["perturbed"]

    original_ans = run_one_word(original_prompt)
    perturbed_ans = run_one_word(perturbed_prompt)

    crossed = (original_ans.lower() == "no" and perturbed_ans.lower() == "yes")
    if crossed:
        num_boundary_cross += 1

    output_curr = {
        "index": key,
        "original_prompt": original_prompt,
        "original_answer": original_ans,
        "perturbed_prompt": perturbed_prompt,
        "perturbed_answer": perturbed_ans,
        "boundary_crossed": crossed
    }

    results.append(output_curr)

    with open(OUTPUT_FILE, "a") as f:
        f.write(f"Index {output_curr['index']}:\n")
        f.write(f"  Original:   {output_curr['original_prompt']}\n")
        f.write(f"  Answer:   {output_curr['original_answer']}\n")
        f.write(f"  Perturbed:  {output_curr['perturbed_prompt']}\n")
        f.write(f"  Answer:   {output_curr['perturbed_answer']}\n")
        # f.write(f"  Boundary crossed? {output_curr['boundary_crossed']}\n\n")

# with open(OUTPUT_FILE, "w") as f:
#     f.write("LLaMA-2-7B Boundary Crossing Evaluation\n\n")
#     f.write(f"Total prompts: {num_total}\n")
#     f.write(f"Successful boundary crossings: {num_boundary_cross}\n")
#     f.write(f"Success rate: {num_boundary_cross / num_total:.2%}\n\n")
#     f.write("Detailed Results\n\n")

#     for r in results:
#         f.write(f"Index {r['index']}:\n")
#         f.write(f"  Original:   {r['original_prompt']}\n")
#         f.write(f"  Answer:   {r['original_answer']}\n")
#         f.write(f"  Perturbed:  {r['perturbed_prompt']}\n")
#         f.write(f"  Answer:   {r['perturbed_answer']}\n")
#         f.write(f"  Boundary crossed? {r['boundary_crossed']}\n\n")

