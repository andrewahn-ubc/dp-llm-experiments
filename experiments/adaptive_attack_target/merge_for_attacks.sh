#!/bin/bash
#SBATCH --job-name=merge_adapt_tgt
#SBATCH --account=def-mijungp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=output/merge_adaptive_target_%j.out

# Merge LoRA adaptive-attack target → full HF dir for GCG/AutoDAN.
#
#   sbatch experiments/adaptive_attack_target/merge_for_attacks.sh

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$SCRATCH/dp-llm-experiments}"
cd "$REPO_ROOT"
mkdir -p output

module load StdEnv/2023 python/3.11
# shellcheck disable=SC1090
source "${VENV_ACTIVATE:-$SCRATCH/venv/nanogcg/bin/activate}"

export TRANSFORMERS_CACHE="${SLURM_TMPDIR:-/tmp}/hf_cache"
export HF_HOME="${SLURM_TMPDIR:-/tmp}/hf_home"
mkdir -p "$TRANSFORMERS_CACHE" "$HF_HOME"

BASE="${BASE:-$SCRATCH/llama2_7b_chat_hf}"
ADAPTER="${ADAPTER:-$SCRATCH/adaptive_attack_target/llama2_7b_chat_lam0.1_eps-0.5_finetuned_llm_epoch5}"
OUT="${OUT:-$SCRATCH/merged_adaptive_l2_lam0.1_eps-0.5_ep5}"

echo "base=$BASE"
echo "adapter=$ADAPTER"
echo "out=$OUT"

python - <<PY
import os
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = os.path.expandvars("$BASE")
adapter = os.path.expandvars("$ADAPTER")
out = os.path.expandvars("$OUT")

assert Path(base).is_dir(), base
assert Path(adapter).is_dir(), adapter

tok = AutoTokenizer.from_pretrained(base)
m = AutoModelForCausalLM.from_pretrained(
    base, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
)
m = PeftModel.from_pretrained(m, adapter).merge_and_unload()
Path(out).mkdir(parents=True, exist_ok=True)
m.save_pretrained(out)
tok.save_pretrained(out)

# Prefer base tokenizer files (avoids AutoDAN/sentencepiece quirks from merge).
for name in ("tokenizer.model", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.json"):
    src = Path(base) / name
    dst = Path(out) / name
    if src.is_file():
        shutil.copy2(src, dst)
        print(f"copied tokenizer file {name}")

print("merged ->", out)
PY
