#!/bin/bash
#SBATCH --job-name=base_test_m7
#SBATCH --account=rrg-mijungp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=output/narval_base_model_test_mistral_7b_instruct_%j.out
#
# Base Mistral-7B-Instruct on official test CSVs; HarmBench cls + Mistral refusal judge (eval.py).
#
# From repo root:
#   mkdir -p output && sbatch eval/narval_base_model_test_eval_mistral_7b_instruct.slurm

set -euo pipefail

export MODEL_PROFILE=mistral_7b_instruct
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${SCRATCH}/dp-llm-experiments}}"
exec bash "${REPO_ROOT}/eval/base_model_test_eval_common.sh"
