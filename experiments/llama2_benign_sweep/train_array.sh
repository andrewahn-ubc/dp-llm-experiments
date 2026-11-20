#!/bin/bash
#SBATCH --job-name=l2b_train
#SBATCH --account=def-mijungp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:55:00
#SBATCH --array=0-0
#SBATCH --output=output/l2b_train_%A_%a.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SCRATCH}/dp-llm-experiments}"
VENV_ACTIVATE="${VENV_ACTIVATE:-${SCRATCH}/venv/nanogcg/bin/activate}"
: "${MANIFEST:?MANIFEST must point to the training TSV}"
: "${EPOCH:?EPOCH must be set}"
: "${EPOCHS:?EPOCHS must be set}"
: "${STAGE:?STAGE must be set}"
: "${STAGES:?STAGES must be set}"
TASK_OFFSET="${TASK_OFFSET:-0}"

cd "$REPO_ROOT"
mkdir -p output
module load StdEnv/2023 python/3.11
source "$VENV_ACTIVATE"

export TRANSFORMERS_CACHE="$SLURM_TMPDIR/hf_cache"
export HF_HOME="$SLURM_TMPDIR/hf_home"
mkdir -p "$TRANSFORMERS_CACHE" "$HF_HOME"
unset WANDB_PROJECT || true

EXTRA_ARGS=()
if [[ "${RETAIN_INTERMEDIATE_CHECKPOINTS:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--retain-intermediate-checkpoints)
fi

python -m experiments.llama2_benign_sweep.worker train \
    --manifest "$MANIFEST" \
    --task-id "$((SLURM_ARRAY_TASK_ID + TASK_OFFSET))" \
    --epoch "$EPOCH" \
    --epochs "$EPOCHS" \
    --stage "$STAGE" \
    --stages "$STAGES" \
    "${EXTRA_ARGS[@]}"
