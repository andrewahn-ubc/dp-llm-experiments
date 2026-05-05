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
import sys
from pathlib import Path

import pandas as pd
import time
import random

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval_helpers import (
    generate_all_jb_responses,
    classify_all_jb_safety,
    generate_original_responses,
    classify_refusal,
)
from train.model_profiles import (  # noqa: E402
    DEFAULT_MODEL_PROFILE,
    MODEL_PROFILE_CHOICES,
    resolve_profile,
)
import os
import gc
import psutil, torch

print(f"CPU RAM available: {psutil.virtual_memory().available / 1e9:.1f}G")
try:
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}G")
except Exception:
    pass

DEVICE = "cuda"
DTYPE = torch.float16
MAX_GEN_TOKENS = 5
batch_size = 1
neighbour_names = {
    "gcg": ["GCG Variant", "GCG Response"],
    "autodan": ["AutoDAN Variant", "AutoDAN Response"],
    "pair": ["PAIR Variant", "PAIR Response"]
}

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

def load_model(LLM_NAME, resume_from=None):
    # Load Main LLM with LoRA
    tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        LLM_NAME,
        torch_dtype=DTYPE,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    if resume_from is None:
        return base_model, tokenizer
    else:
        model = PeftModel.from_pretrained(base_model, resume_from, is_trainable=True)

        return model, tokenizer

def load_jailbreak_classifier(model_path: str):
    """HarmBench Mistral val classifier (HF snapshot) for GCG/AutoDAN/PAIR ASR labels."""
    resolved = os.path.expandvars(os.path.expanduser(model_path))
    guard_tokenizer = AutoTokenizer.from_pretrained(resolved)
    guard_tokenizer.pad_token = guard_tokenizer.eos_token
    hb_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else DTYPE
    )
    guard_model = AutoModelForCausalLM.from_pretrained(
        resolved,
        torch_dtype=hb_dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    for p in guard_model.parameters():
        p.requires_grad = False

    guard_model.eval()

    return guard_model, guard_tokenizer


# REQUIREMENT: unseen_family must be one of {"gcg", "pair", and "autodan"}
def main(args):
    profile = resolve_profile(getattr(args, "model_profile", DEFAULT_MODEL_PROFILE))
    base_llm = os.path.expandvars(os.path.expanduser(args.base_llm or profile.base_llm))
    jb_path = os.path.expandvars(
        os.path.expanduser(args.jailbreak_classifier_path or profile.eval_jailbreak_classifier_path)
    )
    ref_path = os.path.expandvars(os.path.expanduser(args.refusal_judge_path or profile.eval_refusal_judge_path))

    print(
        f"[eval] model_profile={profile.key!r} base_llm={base_llm!r} "
        f"jailbreak_cls={jb_path!r} refusal_judge={ref_path!r}",
        flush=True,
    )

    model, tokenizer = load_model(base_llm, resume_from=args.resume_from)
    guard_model, guard_tokenizer = load_jailbreak_classifier(jb_path)

    system_prompt = (
        profile.default_system_prompt
        if args.system_prompt_mode == "default"
        else None
    )
    print(
        f"[eval] system_prompt_mode={args.system_prompt_mode!r} "
        f"(system_prompt={'custom' if system_prompt else 'None / no system role'})"
    )

    # Compute ASR on harmful prompts
    val_df = pd.read_csv(args.validation_data)
    end_of_epoch_asr_path = f"{args.harmful_output_file}.csv"
    end_of_epoch_frr_path = f"{args.benign_output_file}.csv"
    df_with_jb_responses =  generate_all_jb_responses(val_df, 
                            batch_size=8, 
                            finetuned_model=model, 
                            tokenizer=tokenizer, 
                            testing_mode=args.eval_mode, 
                            unseen_family=args.unseen_family,
                            system_prompt=system_prompt)
    classify_all_jb_safety(df_with_jb_responses, 
                        batch_size = 8, 
                        guard_model=guard_model, 
                        guard_tokenizer=guard_tokenizer, 
                        testing_mode=args.eval_mode,  
                        unseen_family=args.unseen_family,
                        output_file=end_of_epoch_asr_path) # gonna write the result in-place
    unload_model(guard_model, tokenizer=guard_tokenizer)

    # Compute FRR on benign prompts
    frr_val_df = pd.read_csv(args.benign_validation_data)
    df_with_regular_responses = generate_original_responses(frr_val_df, 
                                batch_size=8, 
                                finetuned_model=model, 
                                tokenizer=tokenizer,
                                system_prompt=system_prompt)
    model.eval()
    gc.collect()
    torch.cuda.empty_cache()
    unload_model(model, tokenizer=tokenizer)
    refusal_judge = AutoModelForCausalLM.from_pretrained(
        ref_path,
        torch_dtype=DTYPE,
        device_map="auto",
    )
    refusal_judge_tokenizer = AutoTokenizer.from_pretrained(ref_path)
    refusal_judge_tokenizer.pad_token = refusal_judge_tokenizer.eos_token
    classify_refusal(df_with_regular_responses, 
                            batch_size = 8, 
                            refusal_model=refusal_judge, 
                            refusal_tokenizer=refusal_judge_tokenizer, 
                            output_file=end_of_epoch_frr_path) # gonna write the result in-place

if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser()

    # Experiment 
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
        "--refusal-judge-path",
        default="",
        help="Override profile refusal judge (default: profile.eval_refusal_judge_path).",
    )
    parser.add_argument(
        "--jailbreak-classifier-path",
        default="",
        help="Override profile HarmBench classifier path.",
    )
    parser.add_argument(
        "--model-profile",
        default=os.environ.get("MODEL_PROFILE", DEFAULT_MODEL_PROFILE),
        choices=list(MODEL_PROFILE_CHOICES),
        help="Selects base LLM + default eval judges (train/model_profiles.py).",
    )
    parser.add_argument(
        "--base-llm",
        default="",
        help="Override profile base HF directory for the LoRA target.",
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
        "--system-prompt-mode",
        default="empty",
        choices=["default", "empty"],
        help=(
            "'empty' omits the system role. 'default' uses profile "
            "default_system_prompt (None for all built-in profiles)."
        ),
    )
    parser.add_argument("--resume-from", default=None) # this is the model you load at the start

    args = parser.parse_args()
    main(args)

    end_time = time.time()
    runtime_in_s = end_time - start_time
    print(f"Time taken: {str(runtime_in_s / (60*60))} hours, {str(runtime_in_s / 60)} minutes, and {str(runtime_in_s)} seconds")
