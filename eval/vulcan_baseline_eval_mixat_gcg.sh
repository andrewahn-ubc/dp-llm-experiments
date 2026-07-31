#!/bin/bash
#SBATCH --job-name=baseline_mixat_gcg
#SBATCH --account=aip-mijungp
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=output/baseline_mixat_gcg_%j.out
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=ammany01@cs.ubc.ca
#
# Rebuttal baseline: the MixAT+GCG variant, INSAIT-Institute/Llama3-8B-MixAT-GCG
# (a LoRA adapter over meta-llama/Meta-Llama-3-8B-Instruct, additionally trained with
# discrete GCG-style adversarial examples mixed in -- unlike plain MixAT). Same base
# model as vulcan_baseline_eval_mixat.sh, so this reuses the same BASE_LLM download
# and only needs a separate adapter. Run through the same ASR/FRR pipeline (eval.py)
# as our fine-tuned model, in adaptive eval mode. Runs on Vulcan.
#
# Matches the llama_3_8b_instruct profile's base model exactly (Llama-3, not 3.1).
#
# Requires meta-llama/Meta-Llama-3-8B-Instruct downloaded locally as BASE_LLM below
# (shared with vulcan_baseline_eval_mixat.sh -- skip if already downloaded) and the
# MixAT-GCG adapter downloaded to BASE_LORA_PATH:
#   hf download meta-llama/Meta-Llama-3-8B-Instruct --local-dir $SCRATCH/hf_models/Meta-Llama-3-8B-Instruct
#   hf download INSAIT-Institute/Llama3-8B-MixAT-GCG --local-dir $SCRATCH/hf_models/Llama3-8B-MixAT-GCG
#
# From repo root (e.g. $HOME/repos/dp-llm-experiments):
#   mkdir -p output && sbatch eval/vulcan_baseline_eval_mixat_gcg.sh

set -euo pipefail

export MODEL_PROFILE=llama_3_8b_instruct
export BASE_LLM="${BASE_LLM:-${SCRATCH}/hf_models/Meta-Llama-3-8B-Instruct}"
export BASE_LORA_PATH="${BASE_LORA_PATH:-${SCRATCH}/hf_models/Llama3-8B-MixAT-GCG}"
export BASELINE_TAG="mixat_gcg"
export EVAL_MODE="${EVAL_MODE:-adaptive}"

REPO_ROOT="${REPO_ROOT:-${HOME}/repos/dp-llm-experiments}"
exec bash "${REPO_ROOT}/eval/base_model_test_eval_common.sh"
