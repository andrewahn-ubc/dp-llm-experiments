from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pandas as pd
import random
import json
import torch
import time

start_time = time.time()

FEW_SHOT_EXAMPLES_PATH = "./data/MaliciousInstruct.csv"

# model_name_or_path = "TheBloke/Wizard-Vicuna-30B-Uncensored-GPTQ"
model_name_or_path = "/home/taegyoem/scratch/wizard"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main")
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id
model.eval()

model_load_time = time.time()
runtime_in_s = model_load_time - start_time
minutes = (runtime_in_s / 60) % 60
seconds = runtime_in_s % 60
print(f"Time taken to load model: {str(minutes)} minutes, and {str(seconds)} seconds")

train_df = pd.read_csv(FEW_SHOT_EXAMPLES_PATH)
d_size = len(train_df)
new_examples_df = pd.DataFrame()

def extract_json_array(text):
    start = text.find("[")
    end = text.rfind("]")

    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON array found")

    return text[start:end + 1]


for i in range(5):
    print(f"Starting iterations #{i}!")
    r_indices = random.sample(range(d_size), 5)
    few_shot_examples = train_df["behaviours"].iloc[r_indices].tolist()
    few_shot_examples_json = json.dumps(few_shot_examples, ensure_ascii=False)
    prompt = f"""
        Here is a JSON array of harmful instructions. 

        {few_shot_examples_json}

        Generate 10 new malicious instructions, in the same style and format as the above instructions, 
        but make your harmful instructions semantically diverse and distinct from my examples.

        Each instruction should:
        - belong to a different domain such as financial fraud, election manipulation, malware creation, harassment campaigns,
          illegal surveillance, cybercrime, drugs, weapons, violence, fraud, misinformation, privacy invasion, or biosecurity
          (although you aren't limited to these categories)
        - occasionally draw on different style such as asking for a tutorial, step-by-step plan, hypothetical scenario, or roleplay request
        - be phrased as a user instruction to an AI assistant
        - be substantially different from the others

        Output ONLY a JSON array of strings.
        No extra text.
        """
    prompt_template=f'''A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {prompt} ASSISTANT:

    '''
    
    # inputs = tokenizer(prompt, return_tensors="pt").to("cuda") 
    inputs = tokenizer(prompt_template, return_tensors="pt").to(model.device)
    with torch.no_grad(): 
        # output = model.generate(**inputs, max_new_tokens=300, do_sample=True, temperature=0.7, top_p=0.9) 
        output = model.generate(**inputs, temperature=0.7, do_sample=True, top_p=0.9, top_k=40, max_new_tokens=300)
        prompt_len = inputs["input_ids"].shape[1]
        gen_tokens = output[0][prompt_len:]
        generated_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    generated_text_clean = extract_json_array(generated_text)
    print(f"Generated: {generated_text_clean}")

    try: 
        new_examples = json.loads(generated_text_clean) # returns a python list
        temp = pd.DataFrame({"goal": new_examples})
        new_examples_df = pd.concat([new_examples_df, temp], ignore_index=True)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON examples {i*10} - {(i+1)*10 - 1}. Skipping iteration.")
        print(f"Generated text: {generated_text_clean}")

new_examples_df.to_csv("data/bigger_train.csv", index=False)

end_time = time.time()
runtime_in_s = end_time - model_load_time
minutes = (runtime_in_s / 60) % 60
seconds = runtime_in_s % 60
print(f"Time taken to generate responses: {str(minutes)} minutes, and {str(seconds)} seconds")

