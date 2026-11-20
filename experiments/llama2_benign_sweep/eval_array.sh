#!/bin/bash
#SBATCH --job-name=l2b_eval
#SBATCH --account=def-mijungp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:55:00
#SBATCH --array=0-0
#SBATCH --output=output/l2b_eval_%A_%a.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SCRATCH}/dp-llm-experiments}"
VENV_ACTIVATE="${VENV_ACTIVATE:-${SCRATCH}/venv/nanogcg/bin/activate}"
: "${MANIFEST:?MANIFEST must point to the evaluation TSV}"
TASK_OFFSET="${TASK_OFFSET:-0}"

cd "$REPO_ROOT"
mkdir -p output
module load StdEnv/2023 python/3.11
source "$VENV_ACTIVATE"

export TRANSFORMERS_CACHE="$SLURM_TMPDIR/hf_cache"
export HF_HOME="$SLURM_TMPDIR/hf_home"
mkdir -p "$TRANSFORMERS_CACHE" "$HF_HOME"

python -m experiments.llama2_benign_sweep.worker eval \
    --manifest "$MANIFEST" \
    --task-id "$((SLURM_ARRAY_TASK_ID + TASK_OFFSET))"
