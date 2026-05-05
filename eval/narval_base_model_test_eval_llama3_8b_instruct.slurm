#!/bin/bash
#SBATCH --job-name=base_test_l3
#SBATCH --account=rrg-mijungp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=output/narval_base_model_test_llama3_8b_instruct_%j.out
#
# Base Llama-3-8B-Instruct on official test CSVs; HarmBench cls + Mistral refusal judge (eval.py).
#
# From repo root:
#   mkdir -p output && sbatch eval/narval_base_model_test_eval_llama3_8b_instruct.slurm

set -euo pipefail

export MODEL_PROFILE=llama_3_8b_instruct
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${SCRATCH}/dp-llm-experiments}}"
exec bash "${REPO_ROOT}/eval/base_model_test_eval_common.sh"
