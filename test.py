from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import PeftModel
import pandas as pd
import torch

LLM_NAME = "/home/taegyoem/scratch/llama2_7b_chat_hf" 
TESTING_DATA = "./data/test.csv"
FINETUNED_LLM_PATH = "/home/taegyoem/scratch/finetuned_llm"

tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
base_model = AutoModelForCausalLM.from_pretrained(LLM_NAME, device_map="auto")
finetuned_model = PeftModel.from_pretrained(base_model, FINETUNED_LLM_PATH)

df = pd.read_csv(TESTING_DATA)

# generate 30 tokens for each regular and perturbed prompt using the finetuned LLM
# Original Prompt + Response
for index, (_,row) in enumerate(df.iterrows()):
    # Original Prompt + Response
    prompt = row["Original Prompt"]
    # messages = [{"role": "user", "content": prompt}]          # commented out because this format wasn't used during finetuning
    # prompt_formatted = tokenizer.apply_chat_template(
    #     messages, 
    #     tokenize=False, 
    #     add_generation_prompt=True
    # )
    prompt_formatted = f"[INST] User: {prompt} [/INST] Agent: "

    inputs = tokenizer(prompt_formatted, return_tensors="pt").to("cuda") 
    with torch.no_grad(): 
        output = finetuned_model.generate(**inputs, max_new_tokens=30, temperature=0.0, do_sample=False) 
    decoded = tokenizer.decode(output[0], skip_special_tokens=True) 
    prompt_for_slicing = prompt_formatted.replace("<s>", "").replace("</s>", "")
    answer = decoded[len(prompt_for_slicing):]

    # Perturbed Prompt + Response 
    jb_prompt = row["Perturbed Prompt"] 
    # messages = [{"role": "user", "content": pert_prompt}]
    # jb_prompt_formatted = tokenizer.apply_chat_template(
    #     messages, 
    #     tokenize=False, 
    #     add_generation_prompt=True
    # )
    jb_prompt_formatted = f"[INST] User: {jb_prompt} [/INST] Agent: "

    jb_inputs = tokenizer(jb_prompt_formatted, return_tensors="pt").to("cuda") 
    with torch.no_grad(): 
        jb_output = finetuned_model.generate(**jb_inputs, max_new_tokens=30, temperature=0.0, do_sample=False) 
    jb_decoded = tokenizer.decode(jb_output[0], skip_special_tokens=True) 
    jb_prompt_for_slicing = jb_prompt_formatted.replace("<s>", "").replace("</s>", "")
    jb_answer = jb_decoded[len(jb_prompt_for_slicing):]

    # Update output 
    df.loc[index] = [prompt, answer, jb_prompt, jb_answer] 

# save results in a new dataframe and save it as a new csv file
df.to_csv("data/test_finetuned_output.csv", index=False)
