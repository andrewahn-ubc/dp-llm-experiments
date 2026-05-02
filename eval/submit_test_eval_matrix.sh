#!/bin/bash
#SBATCH --job-name=test_eval_mx
#SBATCH --account=rrg-mijungp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --array=0-83
#SBATCH --output=output/test_eval_matrix_%A_%a.out
#
# Submit from repo root (where eval/ lives):
#   sbatch eval/submit_test_eval_matrix.sh
#
# One array task = one (mode, λ, ε) from eval/test_eval_matrix.py, running
#   1× seen-family eval + 3× unseen-family eval (gcg, autodan, pair).
# Default task_count=84 = 6×7 clean_reg + 6×7 pert_reg (same λ×ε grid as submit_wandb_sweep).
#   pert_reg with λ=0 is adversarial SFT—train with --lm-loss-input perturbed and
#   the same (λ,ε) in submit_wandb_sweep (λ=0 is in LAMBDAS there).
#
# Prerequisites:
#   • Seen checkpoints under CHECKPOINT_ROOT (same as submit_wandb_sweep):
#       {slug}_finetuned_llm_epoch${EPOCH}
#   • Held-out checkpoints, same slug convention:
#       heldout_{family}_{slug}_finetuned_llm_epoch${EPOCH}
#
# If you only have seen-family adapters, set before sbatch:
#   export EXTRA_ARGS="--seen-only"
#
# Time / memory: increase if a single task OOMs or times out (4 full eval runs).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p output

module load StdEnv/2023 python/3.11
source "${SCRATCH}/venv/nanogcg/bin/activate"

export TRANSFORMERS_CACHE="${SLURM_TMPDIR}/hf_cache"
export HF_HOME="${SLURM_TMPDIR}/hf_home"
mkdir -p "$TRANSFORMERS_CACHE" "$HF_HOME"

REPO_ROOT="${REPO_ROOT:-${SCRATCH}/dp-llm-experiments}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SCRATCH}/dp-llm-sweep}"

python "${REPO_ROOT}/eval/test_eval_matrix.py" \
    --repo-root "${REPO_ROOT}" \
    --checkpoint-root "${CHECKPOINT_ROOT}" \
    --epoch "${EPOCH:-5}" \
    --lr "${LR:-2e-5}" \
    ${EXTRA_ARGS:-}
