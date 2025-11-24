from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from huggingface_hub import snapshot_download

MODEL_PATH = "/home/taegyoem/scratch/llama2_7b"  
# MODEL_PATH = snapshot_download(
#     "meta-llama/Llama-2-7b-hf",
#     local_dir="/home/taegyoem/scratch/llama2_7b",
#     resume_download=True
# )

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    local_files_only=True
)

prompt = "Who is the best basketball player ever?"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7,
    )

result = tokenizer.decode(output[0], skip_special_tokens=True)

with open("/home/taegyoem/scratch/llama2_7b_output.txt", "w") as f:
    f.write(result)

# import json

# results = []

# prompts = [
#     "Explain why the sky is blue in one short paragraph.",
#     "Summarize quantum mechanics in two sentences."
# ]

# for prompt in prompts:
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#     with torch.no_grad():
#         output = model.generate(**inputs, max_new_tokens=150, temperature=0.7)
#     text = tokenizer.decode(output[0], skip_special_tokens=True)
#     results.append({"prompt": prompt, "output": text})

# with open("/home/$USER/scratch/llama2_7b_outputs.json", "w") as f:
#     json.dump(results, f, indent=2)
