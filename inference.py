from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_PATH = "/home/taegyoem/scratch/llama2_7b"  

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

print(tokenizer.decode(output[0], skip_special_tokens=True))
