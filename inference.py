from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

MODEL_PATH = "/home/taegyoem/scratch/llama2_7b"  
print("Files in model directory:", os.listdir(MODEL_PATH))

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"   
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
