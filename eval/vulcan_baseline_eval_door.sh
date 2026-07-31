#!/bin/bash
#SBATCH --job-name=baseline_door
#SBATCH --account=aip-mijungp
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=output/baseline_door_%j.out
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=ammany01@cs.ubc.ca
#
# Rebuttal baseline: an apples-to-apples Llama-3 comparison against our DCL model,
# using wicai24/Llama-3-8B-Instruct-W-DOOR-exponential -- a merged full checkpoint
# (unlike MixAT, no separate adapter/base-loading needed here), matching our
# llama_3_8b_instruct profile's base model exactly. Runs through the same ASR/FRR
# pipeline (eval.py) as our fine-tuned model, in adaptive eval mode. Runs on Vulcan.
# If BASE_LLM is left as the HF repo ID (default), eval.py/transformers will download
# it at runtime; set BASE_LLM to a local dir if you've already downloaded it (as with
# wicai24/Llama-3-8B-Instruct-W-DOOR-exponential downloaded earlier to $SCRATCH/hf_models/).
#
# From repo root (e.g. $HOME/repos/dp-llm-experiments):
#   mkdir -p output && sbatch eval/vulcan_baseline_eval_door.sh

set -euo pipefail

export MODEL_PROFILE=llama_3_8b_instruct
export BASE_LLM="${BASE_LLM:-${SCRATCH}/hf_models/Llama-3-8B-Instruct-W-DOOR-exponential}"
export BASELINE_TAG="door_exponential"
export EVAL_MODE="${EVAL_MODE:-adaptive}"

REPO_ROOT="${REPO_ROOT:-${HOME}/repos/dp-llm-experiments}"
exec bash "${REPO_ROOT}/eval/base_model_test_eval_common.sh"
