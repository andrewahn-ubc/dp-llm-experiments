import math
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import LoraConfig, get_peft_model, PeftModel
import pandas as pd
import time
import random
import argparse
import os
import sys

# train.py lives in train/; sibling package eval/ is at repo root. Slurm runs
# `python .../train/train.py`, so sys.path[0] is train/ unless we add the root.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from eval.eval_helpers import generate_all_jb_responses, classify_all_jb_safety, generate_original_responses, classify_refusal
from train.model_profiles import (
    DEFAULT_MODEL_PROFILE,
    MODEL_PROFILE_CHOICES,
    MISTRAL_SELF_HINGE_PROMPT,
    resolve_profile,
)
import psutil, torch
import gc

def _wandb_enabled():
    return bool(os.environ.get("WANDB_PROJECT", "").strip())


def _wandb_init(args, *, hinge_style: str):
    if not _wandb_enabled():
        return False
    mode = os.environ.get("WANDB_MODE", "online")
    # Offline runs do not need wandb's sidecar service; on Lustre/NFS the service-port
    # handshake often hits ServicePollForTokenError (~30s) before init_timeout applies.
    if str(mode).lower() == "offline":
        os.environ.setdefault("WANDB_DISABLE_SERVICE", "true")
    try:
        import wandb
    except ImportError:
        print("WANDB_PROJECT is set but wandb is not installed; skipping Weights & Biases logging.")
        return False
    run_id = os.environ.get("WANDB_RUN_ID")
    run_name = os.environ.get("WANDB_RUN_NAME")
    init_kwargs = {
        "project": os.environ["WANDB_PROJECT"],
        "mode": mode,
        "dir": os.environ.get("WANDB_DIR", "."),
        "config": {
            "lr": args.lr,
            "lambda_val": args.lambda_val,
            "epsilon": args.epsilon,
            "lora_rank": args.lora_rank,
            "start_epoch": args.start_epoch,
            "total_epochs": args.total_epochs,
            "eval_mode": args.eval_mode,
            "unseen_family": args.unseen_family,
            "system_prompt_mode": getattr(args, "system_prompt_mode", "empty"),
            "lm_loss_input": getattr(args, "lm_loss_input", "clean"),
            "model_profile": getattr(args, "model_profile", DEFAULT_MODEL_PROFILE),
            "hinge_style": hinge_style,
            "shuffle_training_rows": getattr(args, "shuffle_training_rows", True),
            "training_shuffle_seed": getattr(args, "training_shuffle_seed", None),
            "train_data_frac_start": float(getattr(args, "train_data_frac_start", 0.0)),
            "train_data_frac_end": float(getattr(args, "train_data_frac_end", 1.0)),
        },
    }
    if run_name:
        init_kwargs["name"] = run_name
    if run_id:
        init_kwargs["id"] = run_id
    init_kwargs["resume"] = "must" if args.start_epoch > 1 else "allow"
    # Default init_timeout is 90 s; raise it for slow shared filesystems (e.g. Narval
    # Lustre) where the wandb service handshake occasionally exceeds the default.
    # WANDB_DIR is expected to point at $SLURM_TMPDIR/wandb (set in sbatch script);
    # this is a belt-and-suspenders bump on top of that.
    init_kwargs["settings"] = wandb.Settings(init_timeout=300)
    wandb.init(**init_kwargs)
    return True


def _wandb_finish(wandb_on):
    if not wandb_on:
        return
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except ImportError:
        pass


print(f"CPU RAM available: {psutil.virtual_memory().available / 1e9:.1f}G")
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}G")

DEVICE = "cuda"
DTYPE = torch.float16
MAX_GEN_TOKENS = 5
batch_size = 1
neighbour_names = {
    "gcg": ["GCG Variant", "GCG Response"],
    "autodan": ["AutoDAN Variant", "AutoDAN Response"],
    "pair": ["PAIR Variant", "PAIR Response"]
}

# SYSTEM_PROMPT is set in main() from --system-prompt-mode and the model profile.
# None omits the system role in chat templates (no defensive scaffolding).
SYSTEM_PROMPT = None

def load_model(LLM_NAME, lora_rank, resume_from=None):
    # Load Main LLM with LoRA
    tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        LLM_NAME,
        torch_dtype=DTYPE,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    if resume_from is not None:
        model = PeftModel.from_pretrained(base_model, resume_from, is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=lora_rank,            
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
    return model, tokenizer

def unload_model(model, tokenizer=None, extra_tensors=None):
    # Move model weights off GPU first if it is a normal single-device model
    try:
        model.to("cpu")
    except Exception:
        pass

    # Delete any extra tensors that may still hold GPU memory
    if extra_tensors is not None:
        for x in extra_tensors:
            try:
                del x
            except Exception:
                pass

    del model
    if tokenizer is not None:
        del tokenizer

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()  # optional, sometimes helps

def load_guard(GUARD_NAME):
    guard_tokenizer = AutoTokenizer.from_pretrained(GUARD_NAME)
    guard_tokenizer.pad_token = guard_tokenizer.eos_token
    
    guard_model = AutoModelForCausalLM.from_pretrained(
        GUARD_NAME,
        torch_dtype=DTYPE,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    for p in guard_model.parameters():
        p.requires_grad = False

    guard_model.eval()

    return guard_model, guard_tokenizer

def format_prompt(tokenizer, prompt):
    # SYSTEM_PROMPT is set in main() from --system-prompt-mode. None means
    # omit the system role entirely so the chat template emits
    # "[INST] <user> [/INST]" with no <<SYS>> block.
    if SYSTEM_PROMPT is None:
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

# Safety Scoring Function (differentiable wrt generated tokens)
# Input: (batched) soft embeddings
# Output: scalar safety scores, ie. P(unsafe | prompt, response)
def safety_score(guard_model, guard_tokenizer, soft_response_embeds, GUARD_HEADER_EMBEDS, GUARD_FOOTER_EMBEDS, keep_grad=True):
    full_embeddings = torch.cat([GUARD_HEADER_EMBEDS, soft_response_embeds, GUARD_FOOTER_EMBEDS], dim=1)

    outputs = guard_model(inputs_embeds=full_embeddings)
    logits = outputs.logits[:, -1, :] # Final token logit

    # Extract safe/unsafe probabilities
    unsafe_id = guard_tokenizer.encode("unsafe", add_special_tokens=False)[0]
    safe_id = guard_tokenizer.encode("safe", add_special_tokens=False)[0]
    
    return torch.sigmoid(logits[:, unsafe_id] - logits[:, safe_id])

def precompute_hinge_context(guard_model, guard_tokenizer, hinge_style: str):
    """Prefix/suffix embeddings for the hinge guard (Llama-Guard-style or Mistral self-twin)."""
    dev = next(guard_model.parameters()).device
    if hinge_style == "llama_guard":
        header_text = """[INST] Task: Check if the following PARTIAL Agent response is starting to generate unsafe content according to our safety policy. Even if the sentence is incomplete, classify as 'unsafe' if the intent is clearly not safe.

<BEGIN CONVERSATION>
"""
    elif hinge_style == "mistral_self_twin":
        header_text = MISTRAL_SELF_HINGE_PROMPT
    else:
        raise ValueError(f"Unknown hinge_style: {hinge_style!r}")

    footer_text = " ...\n<END CONVERSATION> [/INST]"

    header_ids = guard_tokenizer.encode(header_text, return_tensors="pt").to(dev)
    footer_ids = guard_tokenizer.encode(footer_text, add_special_tokens=False, return_tensors="pt").to(dev)

    with torch.no_grad():
        header_embeds = guard_model.get_input_embeddings()(header_ids)
        footer_embeds = guard_model.get_input_embeddings()(footer_ids)

    return header_embeds, footer_embeds

# Generate Responses
# Input: batched prompts
# Output: soft embeddings of shape (batch_size, prompt_len + num_steps, hidden_dim)
def soft_autoregressive_generate(model, tokenizer, prompts, num_steps, temperature=1.0, device="cuda", keep_grad=True):
    context = torch.enable_grad() if keep_grad else torch.no_grad()
    with context:
        # Tokenize prompt
        prompts = [format_prompt(tokenizer, prompt) for prompt in prompts]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        input_ids = inputs["input_ids"]                      # (batch_size, seq_len)

        # Get embedding matrix
        embed_layer = model.get_input_embeddings()
        vocab_embeds = embed_layer.weight                    # (vocab_size, hidden_dim)

        # Initial embeddings (prompt embeddings)
        soft_embeds_of_conversation = embed_layer(input_ids)                 # (batch_size, prompt_len, hidden_dim)

        for _ in range(num_steps):
            # Forward pass using embeddings
            outputs = model(inputs_embeds=soft_embeds_of_conversation)

            # Logits for next token
            logits = outputs.logits[:, -1, :]                # (batch_size, vocab_size)

            # Convert to probability distribution
            probs = F.softmax(logits / temperature, dim=-1) # (batch_size, vocab_size)

            # Expected embedding (soft token)
            next_embed = (probs @ vocab_embeds).unsqueeze(1).requires_grad_() # (batch_size, 1, hidden_dim)

            # Append soft token
            soft_embeds_of_conversation = torch.cat([soft_embeds_of_conversation, next_embed], dim=1)

    return soft_embeds_of_conversation

def hard_response_embeddings_for_guard(model, tokenizer, prompts, responses):
    conversations = [
        format_prompt(tokenizer, prompt) + response
        for prompt, response in zip(prompts, responses)
    ]

    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding=True,
        truncation=True
    )["input_ids"].to(DEVICE)

    with torch.no_grad():
        embeds = model.get_input_embeddings()(input_ids)

    return embeds

# Likelihood Loss
def lm_loss(model, tokenizer, prompts, targets):
    input_ids = []
    labels = []

    for p, t in zip(prompts, targets):
        prompt_text = format_prompt(tokenizer, p)

        p_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
        t_ids = tokenizer(t, add_special_tokens=False).input_ids

        ids = p_ids + t_ids
        lbls = [-100] * len(p_ids) + t_ids

        input_ids.append(ids)
        labels.append(lbls)

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
def training_step(model, tokenizer, guard_model, guard_tokenizer, batch, GUARD_HEADER_EMBEDS, GUARD_FOOTER_EMBEDS, epsilon_term, LAMBDA, lm_loss_input="clean"):
    h, r, h_p = batch

    # LM loss
    # "clean": condition CE on (clean prompt -> safe response). Refusal behavior on
    #         attacks is transferred only via the stability regularizer below.
    # "perturbed": condition CE on (perturbed prompt -> safe response) — the canonical
    #         adversarial-SFT recipe (a la R2D2). With LAMBDA=0 this is pure adv-SFT;
    #         with LAMBDA>0 it is "adv-SFT + stability regularizer" (Variant D).
    if lm_loss_input == "perturbed":
        lm_loss_term = lm_loss(model, tokenizer, h_p, r)
    else:
        lm_loss_term = lm_loss(model, tokenizer, h, r)

    # Safety scores
    hard_embeddings_clean_of_conversation = hard_response_embeddings_for_guard(model=model, tokenizer=tokenizer, prompts=h, responses=r)
    soft_embeddings_pert_of_conversation = soft_autoregressive_generate(model=model, tokenizer=tokenizer, prompts=h_p, num_steps=MAX_GEN_TOKENS, keep_grad=True)
    C_clean = safety_score(guard_model, guard_tokenizer, hard_embeddings_clean_of_conversation, GUARD_HEADER_EMBEDS, GUARD_FOOTER_EMBEDS, keep_grad=False)
    C_pert = safety_score(guard_model, guard_tokenizer, soft_embeddings_pert_of_conversation, GUARD_HEADER_EMBEDS, GUARD_FOOTER_EMBEDS, keep_grad=True)

    # Stability hinge
    stability = torch.clamp(
        C_pert - epsilon_term * C_clean,
        min=0
    )

    # The full loss function - contains likelihood ratio minimization
    total_loss = lm_loss_term + LAMBDA * stability.mean()

    return total_loss, lm_loss_term.detach(), stability.mean().detach()

# REQUIREMENT: unseen_family must be one of {"gcg", "pair", and "autodan"}
def main(args):
    global SYSTEM_PROMPT
    profile = resolve_profile(getattr(args, "model_profile", DEFAULT_MODEL_PROFILE))
    hinge_style = args.hinge_style or profile.hinge_style

    if args.system_prompt_mode == "default":
        SYSTEM_PROMPT = profile.default_system_prompt
    else:
        SYSTEM_PROMPT = None
    print(
        f"[train] model_profile={profile.key!r} hinge_style={hinge_style!r} "
        f"base_llm={os.path.expandvars(args.base_llm or profile.base_llm)!r}",
        flush=True,
    )
    print(
        f"[train] system_prompt_mode={args.system_prompt_mode!r} "
        f"(SYSTEM_PROMPT={'set' if SYSTEM_PROMPT else 'None / no system role'})",
        flush=True,
    )
    print(
        f"[train] lm_loss_input={args.lm_loss_input!r} "
        f"(CE conditions on {'PERTURBED prompt (adv-SFT-style)' if args.lm_loss_input == 'perturbed' else 'CLEAN prompt'}; "
        f"lambda={args.lambda_val})",
        flush=True,
    )
    wandb_on = _wandb_init(args, hinge_style=hinge_style)
    try:
        _main_train(args, wandb_on)
    finally:
        _wandb_finish(wandb_on)


def _main_train(args, wandb_on):
    # Load the trainable model before allocating CUDA tensors for the loss. On some
    # Slurm + wandb.offline setups the first bare ``.to("cuda")`` before ``load_model``
    # intermittently raises cudaErrorDevicesUnavailable; anchoring on the model device
    # after weights are mapped avoids that ordering hazard.
    profile = resolve_profile(getattr(args, "model_profile", DEFAULT_MODEL_PROFILE))
    base_llm = os.path.expandvars(os.path.expanduser(args.base_llm or profile.base_llm))
    hinge_path = os.path.expandvars(os.path.expanduser(args.hinge_guard_path or profile.hinge_guard_path))
    hinge_style = args.hinge_style or profile.hinge_style

    model, tokenizer = load_model(
        base_llm,
        args.lora_rank,
        resume_from=args.resume_from,
    )
    train_dev = next(model.parameters()).device
    epsilon_term = torch.exp(torch.tensor(args.epsilon, dtype=torch.float32, device=train_dev))

    guard_model, guard_tokenizer = load_guard(hinge_path)
    GUARD_HEADER_EMBEDS, GUARD_FOOTER_EMBEDS = precompute_hinge_context(
        guard_model, guard_tokenizer, hinge_style
    )

    # Load data
    df = pd.read_csv(args.training_data) 
    # Drop rows with missing values in any column used during training
    required_cols = ["Original Prompt", "Original Response"] + \
                    [v[0] for v in neighbour_names.values()]
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    if getattr(args, "shuffle_training_rows", True):
        df = df.sample(frac=1.0, random_state=args.training_shuffle_seed).reset_index(
            drop=True
        )
        print(
            f"[train] Shuffled {len(df)} training rows after NaN filter "
            f"(seed={args.training_shuffle_seed!r})",
            flush=True,
        )

    fs = float(getattr(args, "train_data_frac_start", 0.0))
    fe = float(getattr(args, "train_data_frac_end", 1.0))
    if not (0.0 <= fs < fe <= 1.0):
        raise ValueError(
            f"require 0 <= train_data_frac_start < train_data_frac_end <= 1; got {fs=}, {fe=}"
        )
    n_all = len(df)
    if n_all > 0:
        i0 = max(0, min(n_all, math.floor(fs * n_all + 1e-12)))
        i1 = max(0, min(n_all, math.floor(fe * n_all + 1e-12)))
        if i1 <= i0:
            i1 = min(n_all, i0 + 1)
        df = df.iloc[i0:i1].reset_index(drop=True)
        print(
            f"[train] Using rows [{i0}:{i1}) of {n_all} after shuffle "
            f"(train_data_frac_start={fs}, train_data_frac_end={fe})",
            flush=True,
        )

    print(
        f"[train] Training on {len(df)} rows this run (after NaN filter, optional shuffle, "
        f"and optional frac slice)",
        flush=True,
    )
    print(
        f"Epoch {args.start_epoch}/{args.total_epochs} — checkpoint will be saved under "
        f"{args.finetuned_llm_path}_epoch{args.start_epoch}",
        flush=True,
    )

    # Training Loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if args.resume_from is not None:
        optimizer_path = os.path.join(args.resume_from, "optimizer.pt")
        if os.path.exists(optimizer_path):
            optimizer.load_state_dict(torch.load(optimizer_path, map_location=DEVICE))
            print(f"Loaded optimizer state from {optimizer_path}")
        else:
            print(f"Warning: optimizer state not found at {optimizer_path}. Starting with fresh optimizer.")

    
    for i in range(0, len(df), batch_size):
        chunk = df.iloc[i:i+batch_size]

        if (args.eval_mode == "unseen-family" and args.unseen_family is not None):                
            temp = neighbour_names.copy()
            temp.pop(args.unseen_family)
            neighbour_names_list = list(temp.values())
            perturbed_neighbour = random.choice(neighbour_names_list)
            perturbed_prompt = perturbed_neighbour[0]
            perturbed_response = perturbed_neighbour[1]
        else:
            neighbour_names_list = list(neighbour_names.values())
            perturbed_neighbour = random.choice(neighbour_names_list)
            perturbed_prompt = perturbed_neighbour[0]
            perturbed_response = perturbed_neighbour[1]

        batch = (
            list(chunk["Original Prompt"]),
            list(chunk["Original Response"]),
            list(chunk[perturbed_prompt])
        )

        model.train()
        loss, lm_loss_val, stability_val = training_step(model, tokenizer, guard_model, guard_tokenizer, batch,
                                           GUARD_HEADER_EMBEDS, GUARD_FOOTER_EMBEDS, epsilon_term, args.lambda_val,
                                           lm_loss_input=args.lm_loss_input)
        if i % 10 == 0:
            print(f"Step {i}: total={loss.item():.4f}, lm={lm_loss_val.item():.4f}, stab={stability_val.item():.4f}", flush=True)
            if wandb_on:
                import wandb
                wandb.log(
                    {
                        "train/total_loss": loss.item(),
                        "train/lm_loss": lm_loss_val.item(),
                        "train/stability": stability_val.item(),
                    },
                    step=i,
                )
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.empty_cache()

    checkpoint_path = f"{args.finetuned_llm_path}_epoch{args.start_epoch}"
    model.save_pretrained(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)
    torch.save(optimizer.state_dict(), os.path.join(checkpoint_path, "optimizer.pt"))
    print(f"Saved optimizer state to {os.path.join(checkpoint_path, 'optimizer.pt')}")

    if wandb_on:
        import wandb
        wandb.log(
            {
                "checkpoint/path": checkpoint_path,
                "train/epoch_done": args.start_epoch,
            },
        )

    #NOTE: will skip eval for now, and just evaluate the saved models after all training runs
    # Compute ASR on harmful prompts
    # val_df = pd.read_csv(args.validation_data)
    # end_of_epoch_asr_path = f"{args.harmful_output_file}_epoch{args.start_epoch}.csv"
    # end_of_epoch_frr_path = f"{args.benign_output_file}_epoch{args.start_epoch}.csv"
    # df_with_jb_responses =  generate_all_jb_responses(val_df, 
    #                         batch_size=8, 
    #                         finetuned_model=model, 
    #                         tokenizer=tokenizer, 
    #                         testing_mode=args.eval_mode, 
    #                         unseen_family=args.unseen_family)
    # classify_all_jb_safety(df_with_jb_responses, 
    #                     batch_size = 8, 
    #                     guard_model=guard_model, 
    #                     guard_tokenizer=guard_tokenizer, 
    #                     testing_mode=args.eval_mode,  
    #                     unseen_family=args.unseen_family,
    #                     output_file=end_of_epoch_asr_path) # gonna write the result in-place
    
    # # Compute FRR on benign prompts
    # frr_val_df = pd.read_csv(args.benign_validation_data)
    # # "original" as in finetuned LLM responding to the original prompts
    # df_with_regular_responses = generate_original_responses(frr_val_df, 
    #                             batch_size=8, 
    #                             finetuned_model=model, 
    #                             tokenizer=tokenizer)
    # model.eval()
    # gc.collect()
    # torch.cuda.empty_cache()
    # unload_model(model, tokenizer=tokenizer)
    # refusal_judge = AutoModelForCausalLM.from_pretrained(
    #     args.refusal_judge_path,
    #     torch_dtype=torch.float16,
    #     device_map="auto",
    # )
    # refusal_judge_tokenizer = AutoTokenizer.from_pretrained(args.refusal_judge_path)
    # refusal_judge_tokenizer.pad_token = refusal_judge_tokenizer.eos_token
    # classify_refusal(df_with_regular_responses, 
    #                         batch_size = 8, 
    #                         refusal_model=refusal_judge, 
    #                         refusal_tokenizer=refusal_judge_tokenizer, 
    #                         output_file=end_of_epoch_frr_path) # gonna write the result in-place

if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()

    # Experiment parameters
    parser.add_argument(
        "--eval-mode",
        default = "seen-family",
        help = "Testing mode.",
        choices=["seen-family", "unseen-family", "adaptive"]
    )
    parser.add_argument(
        "--unseen-family",
        default = None,
        help = "Name of unseen family. Only used for the unseen family testing mode.",
        choices=["gcg", "autodan", "pair"]
    )
    parser.add_argument(
        "--model-profile",
        default=os.environ.get("MODEL_PROFILE", DEFAULT_MODEL_PROFILE),
        choices=list(MODEL_PROFILE_CHOICES),
        help="Which base LLM + hinge guard pairing to train (see train/model_profiles.py).",
    )
    parser.add_argument(
        "--base-llm",
        default="",
        help="Override profile base HF directory (default: use profile's base_llm).",
    )
    parser.add_argument(
        "--hinge-guard-path",
        default="",
        help="Override profile hinge guard snapshot (default: profile.hinge_guard_path).",
    )
    parser.add_argument(
        "--hinge-style",
        default="",
        choices=["", "llama_guard", "mistral_self_twin"],
        help="Override hinge prompt family (default: profile.hinge_style).",
    )
    parser.add_argument(
        "--finetuned-llm-path",
        default="/home/taegyoem/scratch/finetuned_llm",
        help="path prefix for finetuned main LLM checkpoints (see submit_wandb_sweep FINETUNED_BASE).",
    )
    parser.add_argument(
        "--refusal-judge-path",
        default = "/home/taegyoem/scratch/llama_3_8b",
        help = "path to refusal LLM"
    )
    parser.add_argument(
        "--training-data",
        default = "/home/taegyoem/scratch/dp-llm-experiments/official_data/train.csv",
        help = "path to csv containing training data"
    )
    parser.add_argument(
        "--shuffle-training-rows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After loading --training-data (and dropping NaNs), shuffle row order before "
            "the epoch loop. Recommended when using <1 full epoch so the first steps are "
            "not always the same prompts. Disable with --no-shuffle-training-rows for a "
            "fixed on-disk order."
        ),
    )
    parser.add_argument(
        "--training-shuffle-seed",
        type=int,
        default=None,
        help=(
            "RNG seed for --shuffle-training-rows (default: None = different order each "
            "process unless NumPy's global seed is set). Ignored when --no-shuffle-training-rows."
        ),
    )
    parser.add_argument(
        "--train-data-frac-start",
        type=float,
        default=0.0,
        help=(
            "After shuffling, keep rows from this fraction of the index range [0, N) "
            "(inclusive lower bound). Use 0 and 0.5 for the first half of a split pass."
        ),
    )
    parser.add_argument(
        "--train-data-frac-end",
        type=float,
        default=1.0,
        help=(
            "Exclusive-style upper bound on the same [0, N) scale as --train-data-frac-start "
            "(end is floored to an integer row index). Use 0.5 and 1.0 for the second half."
        ),
    )
    parser.add_argument(
        "--validation-data",
        default = "/home/taegyoem/scratch/dp-llm-experiments/official_data/validation.csv",
        help = "path to csv containing validation data"
    )
    parser.add_argument(
        "--benign-validation-data",
        default = "/home/taegyoem/scratch/dp-llm-experiments/official_data/frr_validation.csv",
        help = "path to csv containing benign validation data for FRR metric"
    )
    parser.add_argument(
        "--harmful-output-file",
        default = "harmful_output",
        help = "path to csv output that looks the same as validation.csv but with jb responses and safety labels"
    )
    parser.add_argument(
        "--benign-output-file",
        default = "benign_output",
        help = "path to csv output that looks the same as frr_validation.csv but with responses and safety labels"
    )
    parser.add_argument(
        "--lr",
        default = 2e-5,
        type=float,
        help = "learning rate"
    )
    parser.add_argument(
        "--lambda-val",
        default = 1.0,
        type=float,
        help = "regularization strength"
    )
    parser.add_argument(
        "--epsilon",
        default = 0.0,
        type=float,
        help = "how forgiving our regularizer is"
    )
    parser.add_argument(
        "--lora-rank",
        default = 8,
        type=int,
        help = "size of LoRA component"
    )
    parser.add_argument(
        "--total-epochs",
        default = 5,
        type=int,
        help = "Number of outer epochs (each is a separate process that saves its own adapter folder).",
    )
    parser.add_argument(
        "--system-prompt-mode",
        default="empty",
        choices=["default", "empty"],
        help=(
            "Chat-template system role. 'empty' omits the system message. "
            "'default' uses the profile's default_system_prompt (all None in "
            "this repo — same as empty unless you customize model_profiles)."
        ),
    )
    parser.add_argument(
        "--lm-loss-input",
        default="clean",
        choices=["clean", "perturbed"],
        help=(
            "Which prompt to condition the language-modelling cross-entropy on. "
            "'clean' (default) uses the unperturbed Original Prompt; refusal "
            "behavior on attacks is then transferred only via the stability "
            "regularizer (the original method as written). 'perturbed' uses the "
            "GCG/AutoDAN/PAIR-perturbed prompt (a.k.a. R2D2-style adversarial "
            "SFT). Set --lambda-val 0 with --lm-loss-input perturbed for a pure "
            "adversarial-SFT baseline; set --lambda-val >0 for adversarial SFT "
            "+ stability regularizer (the strict-superset variant)."
        ),
    )
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--start-epoch", default=1, type=int)

    args = parser.parse_args()

    main(args)

    end_time = time.time()
    runtime_in_s = end_time - start_time
    print(f"Time taken: {str(runtime_in_s / (60*60))} hours, {str(runtime_in_s / 60)} minutes, and {str(runtime_in_s)} seconds")
