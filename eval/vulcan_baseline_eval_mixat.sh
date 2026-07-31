#!/bin/bash
#SBATCH --job-name=baseline_mixat
#SBATCH --account=aip-mijungp
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=output/baseline_mixat_%j.out
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=ammany01@cs.ubc.ca
#
# Rebuttal baseline: an apples-to-apples Llama-3 comparison against our DCL model,
# using INSAIT-Institute/Llama3-8B-MixAT (a LoRA adapter over meta-llama/Meta-Llama-3-8B-Instruct
# -- MixAT is only ever distributed as an adapter, never merged, on HF), run through
# the same ASR/FRR pipeline (eval.py) as our fine-tuned model, in adaptive eval mode.
# Runs on Vulcan.
#
# Matches the llama_3_8b_instruct profile's base model exactly (Llama-3, not 3.1),
# so this is a clean like-for-like comparison, not confounded by base-model version.
#
# Requires meta-llama/Meta-Llama-3-8B-Instruct downloaded locally as BASE_LLM below
# (or override BASE_LLM to point at wherever it lives) and the MixAT adapter downloaded
# to BASE_LORA_PATH:
#   hf download meta-llama/Meta-Llama-3-8B-Instruct --local-dir $SCRATCH/hf_models/Meta-Llama-3-8B-Instruct
#   hf download INSAIT-Institute/Llama3-8B-MixAT --local-dir $SCRATCH/hf_models/Llama3-8B-MixAT
#
# From repo root (e.g. $HOME/repos/dp-llm-experiments):
#   mkdir -p output && sbatch eval/vulcan_baseline_eval_mixat.sh

set -euo pipefail

export MODEL_PROFILE=llama_3_8b_instruct
export BASE_LLM="${BASE_LLM:-${SCRATCH}/hf_models/Meta-Llama-3-8B-Instruct}"
export BASE_LORA_PATH="${BASE_LORA_PATH:-${SCRATCH}/hf_models/Llama3-8B-MixAT}"
export BASELINE_TAG="mixat"
export EVAL_MODE="${EVAL_MODE:-adaptive}"

REPO_ROOT="${REPO_ROOT:-${HOME}/repos/dp-llm-experiments}"
exec bash "${REPO_ROOT}/eval/base_model_test_eval_common.sh"
