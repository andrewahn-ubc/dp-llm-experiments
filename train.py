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
FINETUNED_LLM_PATH = "/home/taegyoem/scratch/finetuned_llm"
TRAINING_DATA = "./data/train.csv"

LAMBDA = 1.0
EPSILON = 1.0       # TODO: try different values of epsilon and plot their performance on a validation set 
MAX_GEN_TOKENS = 5
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
    r=8,            # TODO: try out different ranks, try what other people use
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(base_model, lora_config)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
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

def precompute_guard_context(): # TODO: use llama guard's specific chat template
    header_text = """[INST] Task: Check if the following PARTIAL Agent response is starting to generate unsafe content according to our safety policy. Even if the sentence is incomplete, classify as 'unsafe' if the intent is clearly harmful.

<BEGIN CONVERSATION>
User: """
    footer_text = " ...\n<END CONVERSATION> [/INST]"

    header_ids = guard_tokenizer.encode(header_text, return_tensors="pt").to(DEVICE)
    footer_ids = guard_tokenizer.encode(footer_text, add_special_tokens=False, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        header_embeds = guard_model.get_input_embeddings()(header_ids)
        footer_embeds = guard_model.get_input_embeddings()(footer_ids)

    return header_embeds, footer_embeds

GUARD_HEADER_EMBEDS, GUARD_FOOTER_EMBEDS = precompute_guard_context()

def safety_score(soft_response_embeds):
    full_embeddings = torch.cat([GUARD_HEADER_EMBEDS, soft_response_embeds, GUARD_FOOTER_EMBEDS], dim=1)

    outputs = guard_model(inputs_embeds=full_embeddings)
    logits = outputs.logits[:, -1, :] # Final token logit

    # Extract safe/unsafe probabilities
    unsafe_id = guard_tokenizer.encode("unsafe", add_special_tokens=False)[0]
    safe_id = guard_tokenizer.encode("safe", add_special_tokens=False)[0]
    
    return torch.sigmoid(logits[:, unsafe_id] - logits[:, safe_id])

# Generate Responses
# Input: batched prompts
# Output: soft embeddings of shape (batch_size, prompt_len + num_steps, hidden_dim)
def soft_autoregressive_generate(model, tokenizer, prompts, num_steps, temperature=1.0, device="cuda"):
    # Tokenize prompt
    prompts = [f"[INST] User: {prompt} [/INST] Agent: " for prompt in prompts] # TODO: use llama 2 7b chat's chat template
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
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
        next_embed = (probs @ vocab_embeds).unsqueeze(1).requires_grad_()  # (batch_size, 1, hidden_dim)

        # Append soft token
        soft_embeds = torch.cat([soft_embeds, next_embed], dim=1)

    return soft_embeds


# Likelihood Loss
def lm_loss(prompts, targets):
    input_ids = []
    labels = []

    for p, t in zip(prompts, targets):
        p_ids = tokenizer(p, add_special_tokens=True).input_ids
        t_ids = tokenizer(t, add_special_tokens=False).input_ids

        ids = p_ids + t_ids
        lbls = [-100] * len(p_ids) + t_ids

        input_ids.append(ids)
        labels.append(lbls)

    # Pad
    enc = tokenizer.pad(
        {"input_ids": input_ids, "labels": labels},
        return_tensors="pt"
    ).to(DEVICE)

    outputs = model(
        input_ids=enc.input_ids,
        attention_mask=enc.attention_mask,
        labels=enc.labels
    )

    return outputs.loss

# Full Training Step
def training_step(batch):
    h, r, h_p, r_p = batch

    # LM loss
    lm_loss_term = lm_loss(h, r)

    # Safety scores
    soft_embeddings_clean = soft_autoregressive_generate(model=model, tokenizer=tokenizer, prompts=h, num_steps=MAX_GEN_TOKENS)
    soft_embeddings_pert = soft_autoregressive_generate(model=model, tokenizer=tokenizer, prompts=h_p, num_steps=MAX_GEN_TOKENS)
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
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5) # TODO: configure LR

df = pd.read_csv(TRAINING_DATA) # TODO: get more examples, in the range of 5,000-10,000

batch_size = 1

for epoch in range(3): # TODO: try out different number of epochs and dump checkpoint after each epoch
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
        torch.cuda.empty_cache()

        print(f"Batch {i/batch_size} of epoch {epoch} complete")

    print(f"Epoch {epoch} complete")

model.save_pretrained(FINETUNED_LLM_PATH)
tokenizer.save_pretrained(FINETUNED_LLM_PATH)

