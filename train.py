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
TRAINING_DATA = "./data/train.csv"

LAMBDA = 1.0
EPSILON = 1.0
MAX_GEN_TOKENS = 30
epsilon_term = torch.exp(torch.tensor(EPSILON)).to(DEVICE)

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
# Input: (batched) soft embeddings
# Output: scalar safety scores, ie. P(unsafe | prompt, response)
def safety_score(soft_embeddings):
    outputs = guard_model(input_embeds=soft_embeddings)
    logits = outputs.logits[:, -1, :]

    unsafe_token_id = guard_tokenizer.convert_tokens_to_ids("unsafe")
    safe_token_id = guard_tokenizer.convert_tokens_to_ids("safe")

    unsafe_logit = logits[:, unsafe_token_id]
    safe_logit = logits[:, safe_token_id]

    return torch.sigmoid(unsafe_logit - safe_logit)

# Generate Responses
# Input: batched prompts
# Output: soft embeddings of shape (batch_size, prompt_len + num_steps, hidden_dim)
def soft_autoregressive_generate(model, tokenizer, prompts, num_steps=30, temperature=1.0, device="cuda"):
    # Tokenize prompt
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    input_ids = inputs["input_ids"]                      # (batch_size, seq_len)

    # Get embedding matrix
    embed_layer = model.get_input_embeddings()
    vocab_embeds = embed_layer.weight                    # (vocab_size, hidden_dim)

    # Initial embeddings (prompt embeddings)
    soft_embeds = embed_layer(input_ids)                 # (batch_size, prompt_len, hidden_dim)

    for _ in range(num_steps):
        # Forward pass using embeddings
        outputs = model(inputs_embeds=soft_embeds)

        # Logits for next token
        logits = outputs.logits[:, -1, :]                # (batch_size, vocab_size)

        # Convert to probability distribution
        probs = F.softmax(logits / temperature, dim=-1) # (batch_size, vocab_size)

        # Expected embedding (soft token)
        next_embed = probs @ vocab_embeds                # (batch_size, hidden_dim)
        next_embed = next_embed.unsqueeze(1)             # (batch_size, 1, hidden_dim)

        # Append soft token
        soft_embeds = torch.cat([soft_embeds, next_embed], dim=1)

    return soft_embeds


# Likelihood Loss
def lm_loss(prompts, targets):
    texts = [p + t for p, t in zip(prompts, targets)]

    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(DEVICE)

    labels = enc.input_ids.clone()

    # Mask prompt tokens
    for i, p in enumerate(prompts):
        prompt_len = len(tokenizer(p).input_ids)
        labels[i, :prompt_len] = -100  # ignore prompt tokens

    outputs = model(
        input_ids=enc.input_ids,
        attention_mask=enc.attention_mask,
        labels=labels
    )

    return outputs.loss


# Full Training Step
def training_step(batch):
    h, r, h_p, r_p = batch

    # LM loss
    lm_loss_term = lm_loss(h, r)

    # Safety scores
    soft_embeddings_clean = soft_autoregressive_generate(model, tokenizer, h)
    soft_embeddings_pert = soft_autoregressive_generate(model, tokenizer, h_p)
    C_clean = safety_score(soft_embeddings_clean)
    C_pert = safety_score(soft_embeddings_pert)

    # Stability hinge
    stability = torch.clamp(
        C_clean - epsilon_term * C_pert,
        min=0
    )

    # total_loss = loss_clean + loss_pert + LAMBDA * stability.mean()
    total_loss = lm_loss_term + LAMBDA * stability.mean()
    return total_loss

# Training Loop
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# TODO: split into train/val/test: Train: 410, Test: 100
df = pd.read_csv(TRAINING_DATA)

batch_size = 8

for epoch in range(3):
    for i in range(0, len(df), batch_size):
        chunk = df.iloc[i:i+batch_size]
        batch = (
            list(chunk["Original Prompt"]),
            list(chunk["Original Response"]),
            list(chunk["Perturbed Prompt"]),
            list(chunk["Perturbed Response"]),
        )
        model.train()
        loss = training_step(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"Batch {i/batch_size} of epoch {epoch} complete")

    print(f"Epoch {epoch} complete")



