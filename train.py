import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import LoraConfig, get_peft_model
import pandas as pd

DEVICE = "cuda"
DTYPE = torch.float16

LLM_NAME = "/home/taegyoem/scratch/llama2_7b_chat_hf" 
GUARD_NAME = "/home/taegyoem/scratch/llama_guard_7b"
TOTAL_DATA_PATH = "./data/gcg_output_total.csv"
TRAIN_DATA_PATH = "./data/gcg_output_train.csv"
TEST_DATA_PATH = "./data/gcg_output_test.csv"

LAMBDA = 1.0
EPSILON = 1.0
MAX_GEN_TOKENS = 30

# Load Main LLM with LoRA
tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    LLM_NAME,
    torch_dtype=DTYPE,
    device_map="auto"
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(base_model, lora_config)
model.train()

# Load Guard LLM
guard_tokenizer = AutoTokenizer.from_pretrained(GUARD_NAME)
guard_model = AutoModelForCausalLM.from_pretrained(
    GUARD_NAME,
    torch_dtype=DTYPE,
    device_map="auto"
)

for p in guard_model.parameters():
    p.requires_grad = False

guard_model.eval()

# Safety Scoring Function (differentiable wrt generated tokens)
def safety_score(prompt, response):
    # Returns P(unsafe | prompt, response)
    text = f"[INST] {prompt} [/INST]\n{response}"
    inputs = guard_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(DEVICE)

    outputs = guard_model(**inputs)
    logits = outputs.logits[:, -1]

    unsafe_token_id = guard_tokenizer.convert_tokens_to_ids("unsafe")
    safe_token_id = guard_tokenizer.convert_tokens_to_ids("safe")

    unsafe_logit = logits[:, unsafe_token_id]
    safe_logit = logits[:, safe_token_id]

    return torch.sigmoid(unsafe_logit - safe_logit)

# Generate Responses
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_GEN_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Likelihood Loss
def lm_loss(prompt, target):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    labels = tokenizer(target, return_tensors="pt").input_ids.to(DEVICE)

    outputs = model(**inputs, labels=labels)
    return outputs.loss

# Full Training Step
def training_step(batch):
    h, r, h_p, r_p = batch

    # LM loss
    lm_loss = lm_loss(h, r)

    # Safety scores
    # TODO: run h and h_p through the main LLM and then pass THAT into the safety score function (then update safety_score to take in tokens)
    C_clean = safety_score(h, r)
    C_pert = safety_score(h_p, r_p)

    # Stability hinge
    stability = torch.clamp(
        C_clean - torch.exp(torch.tensor(EPSILON)) * C_pert,
        min=0
    )

    # total_loss = loss_clean + loss_pert + LAMBDA * stability.mean()
    total_loss = lm_loss + LAMBDA * stability.mean()
    return total_loss

# Training Loop
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

df = pd.read_csv(TRAIN_DATA_PATH)

for epoch in range(3):
    for _, row in df.iterrows():
        batch = (
            row["h"],
            row["r"],
            row["h_prime"],
            row["r_prime"],
        )

        loss = training_step(batch)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    print(f"Epoch {epoch} complete")




