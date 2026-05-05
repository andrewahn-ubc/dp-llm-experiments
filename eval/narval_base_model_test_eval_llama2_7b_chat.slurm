#!/bin/bash
#SBATCH --job-name=base_test_l2
#SBATCH --account=rrg-mijungp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=output/narval_base_model_test_llama2_7b_chat_%j.out
#
# Base Llama-2-7B-Chat on official test CSVs; HarmBench cls + Mistral refusal judge (eval.py).
#
# From repo root (recommended):
#   mkdir -p output && sbatch eval/narval_base_model_test_eval_llama2_7b_chat.slurm
#
# If the repo lives elsewhere:
#   export REPO_ROOT=/lustre07/scratch/$USER/dp-llm-experiments
#   cd "$REPO_ROOT" && mkdir -p output && sbatch eval/narval_base_model_test_eval_llama2_7b_chat.slurm
#
# If CSVs are under official/ instead of official_data/:
#   export OFFICIAL_DATA_DIR=$REPO_ROOT/official

set -euo pipefail

export MODEL_PROFILE=llama_2_7b_chat
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${SCRATCH}/dp-llm-experiments}}"
exec bash "${REPO_ROOT}/eval/base_model_test_eval_common.sh"
