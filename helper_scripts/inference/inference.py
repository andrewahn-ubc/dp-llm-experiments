"""
Generate base-model "Original Response" completions for a set of prompts, then merge
them into the relevant dataset CSVs (adds ``Original Prompt`` + ``Original Response``).

This is what makes the Llama 3 training CSVs usable by train.py, which requires
``Original Prompt`` and ``Original Response`` columns (plus the ``* Variant`` columns).

Workflow:
  1. Collect the unique prompts (``--prompt-col``, default ``goal``) across every
     ``--merge-into`` CSV.
  2. Generate one greedy response per unique prompt with the chosen model
     (``--model-profile``, default llama_3_8b_instruct), using no system prompt.
  3. Save a standalone lookup CSV (``--responses-out``: prompt, Original Response).
  4. Merge the responses back into each ``--merge-into`` CSV in place: adds
     ``Original Prompt`` (= prompt-col) and ``Original Response`` while preserving the
     existing columns (e.g. GCG/AutoDAN/PAIR Variant).

Defaults target the cluster layout ``$SCRATCH/dp-llm-experiments/official_data/``.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from train.model_profiles import (  # noqa: E402
    DEFAULT_MODEL_PROFILE,
    MODEL_PROFILE_CHOICES,
    resolve_profile,
)

DEVICE = "cuda"
DTYPE = torch.float16

_DEFAULT_DATA_DIR = "$SCRATCH/dp-llm-experiments/official_data"
_DEFAULT_MERGE_INTO = ",".join(
    f"{_DEFAULT_DATA_DIR}/{name}"
    for name in ("llama3_train.csv", "llama3_validation.csv", "llama3_train_plus_validation.csv")
)


def expand(p: str) -> str:
    return os.path.expandvars(os.path.expanduser(p))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--model-profile",
        default=os.environ.get("MODEL_PROFILE", "llama_3_8b_instruct"),
        choices=list(MODEL_PROFILE_CHOICES),
        help="Model whose base_llm generates the responses (train/model_profiles.py).",
    )
    p.add_argument(
        "--model-path",
        default="",
        help="Override the base LLM path (default: the profile's base_llm).",
    )
    p.add_argument(
        "--merge-into",
        default=_DEFAULT_MERGE_INTO,
        help="Comma-separated CSVs to add Original Prompt + Original Response to (in place).",
    )
    p.add_argument(
        "--prompt-col",
        default="goal",
        help="Column holding the clean prompt in the merge-into CSVs (default: goal).",
    )
    p.add_argument(
        "--response-col",
        default="Original Response",
        help="Name of the response column to write (default: 'Original Response').",
    )
    p.add_argument(
        "--responses-out",
        default=f"{_DEFAULT_DATA_DIR}/llama3_original_responses.csv",
        help="Standalone lookup CSV (prompt, Original Response) saved for reuse.",
    )
    p.add_argument("--max-new-tokens", type=int, default=50)
    return p.parse_args()


def build_prompt(tokenizer, prompt: str) -> str:
    # No system role (Protocol-Undefended), matching the training/eval protocol.
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main() -> int:
    start = time.time()
    args = parse_args()

    profile = resolve_profile(args.model_profile)
    model_path = expand(args.model_path) if args.model_path else expand(profile.base_llm)
    merge_paths = [expand(x.strip()) for x in args.merge_into.split(",") if x.strip()]
    responses_out = expand(args.responses_out)

    print(f"[inference] model_profile={profile.key} base_llm={model_path}", flush=True)
    print(f"[inference] merge-into: {merge_paths}", flush=True)

    # 1. Collect unique prompts across all target CSVs.
    unique_prompts: list[str] = []
    seen: set[str] = set()
    loaded: dict[str, pd.DataFrame] = {}
    for path in merge_paths:
        df = pd.read_csv(path)
        if args.prompt_col not in df.columns:
            raise ValueError(f"{path} is missing prompt column {args.prompt_col!r} (has {list(df.columns)})")
        loaded[path] = df
        for val in df[args.prompt_col].astype(str):
            key = val.strip()
            if key and key not in seen:
                seen.add(key)
                unique_prompts.append(val)
    print(f"[inference] {len(unique_prompts)} unique prompts to generate", flush=True)

    # 2. Load model and generate one response per unique prompt.
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=DTYPE, device_map="auto")
    model.eval()

    responses: dict[str, str] = {}
    for i, prompt in enumerate(unique_prompts):
        formatted = build_prompt(tokenizer, prompt)
        inputs = tokenizer(formatted, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=None,
                top_p=None,
                do_sample=False,
            )
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        responses[prompt.strip()] = tokenizer.decode(new_tokens, skip_special_tokens=True)
        if i % 50 == 0:
            print(f"[inference] {i}/{len(unique_prompts)}", flush=True)

    # 3. Standalone lookup CSV.
    os.makedirs(os.path.dirname(responses_out) or ".", exist_ok=True)
    pd.DataFrame(
        {args.prompt_col: list(responses.keys()), args.response_col: list(responses.values())}
    ).to_csv(responses_out, index=False)
    print(f"[inference] wrote {responses_out} ({len(responses)} rows)", flush=True)

    # 4. Merge into each target CSV in place (add Original Prompt + Original Response).
    for path, df in loaded.items():
        keys = df[args.prompt_col].astype(str).str.strip()
        df["Original Prompt"] = df[args.prompt_col]
        df[args.response_col] = keys.map(responses)
        n_missing = int(df[args.response_col].isna().sum())
        df.to_csv(path, index=False)
        print(
            f"[inference] merged into {path} ({len(df)} rows, {n_missing} without a response)",
            flush=True,
        )

    mins = (time.time() - start) / 60
    print(f"[inference] done in {mins:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
