#!/bin/bash
#SBATCH --job-name=test_eval_mx
#SBATCH --account=rrg-mijungp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=2:15:00
#SBATCH --array=0-31
#SBATCH --output=output/test_eval_matrix_%A_%a.out
#
# ---------------------------------------------------------------------------
# How to submit (no extra wrappers — one Slurm job file only)
# ---------------------------------------------------------------------------
#
# Slurm resolves #SBATCH --output relative to the directory you were in when you ran
# sbatch. Always run sbatch from your repo root so logs land in repo output/:
#
#   cd /path/to/dp-llm-experiments
#   mkdir -p output
#   sbatch eval/submit_test_eval_matrix.sh
#
# Rerun only some array tasks (CLI --array replaces the default 0-31 above), e.g.:
#
#   cd /path/to/dp-llm-experiments && mkdir -p output && sbatch --array=7,12,18,30-31 eval/submit_test_eval_matrix.sh
#
# If your repo is not ${SCRATCH}/dp-llm-experiments, set REPO_ROOT before sbatch:
#
#   export REPO_ROOT=/lustre07/scratch/you/dp-llm-experiments
#   cd "$REPO_ROOT" && mkdir -p output && sbatch --array=7,12,18,30-31 eval/submit_test_eval_matrix.sh
#
# Heatmaps after this eval array finishes (use the Job ID sbatch prints):
#   sbatch --dependency=afterok:<JOBID> eval/submit_plot_heatmaps.sh
#
# ---------------------------------------------------------------------------
#
# One array task = one (mode, λ, ε) cell: full clean grid + one perturbed LM run at λ=0
# (default --perturbed-reg-subset lambda0_only), running 1× seen-family + 3× unseen eval.
# Default task_count=32 = 31 clean + 1 perturbed@λ=0 (same representative ε as clean λ=0).
# Legacy 62-task matrix: sbatch --array=0-61 ... and pass --perturbed-reg-subset full (edit python line).
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
# Time / memory: raise --time or --mem if a single task OOMs or times out (4 full eval runs).

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SCRATCH}/dp-llm-experiments}"
if [[ ! -d "${REPO_ROOT}" ]]; then
  echo "ERROR: REPO_ROOT is not a directory: ${REPO_ROOT}" >&2
  echo "Fix: export REPO_ROOT=/absolute/path/to/dp-llm-experiments" >&2
  exit 2
fi
cd "${REPO_ROOT}"
mkdir -p output

module load StdEnv/2023 python/3.11
source "${SCRATCH}/venv/nanogcg/bin/activate"

export TRANSFORMERS_CACHE="${SLURM_TMPDIR}/hf_cache"
export HF_HOME="${SLURM_TMPDIR}/hf_home"
mkdir -p "$TRANSFORMERS_CACHE" "$HF_HOME"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SCRATCH}/dp-llm-sweep}"
MODEL_PROFILE="${MODEL_PROFILE:-llama_2_7b_chat}"

python "${REPO_ROOT}/eval/test_eval_matrix.py" \
    --repo-root "${REPO_ROOT}" \
    --checkpoint-root "${CHECKPOINT_ROOT}" \
    --model-profile "${MODEL_PROFILE}" \
    --epoch "${EPOCH:-5}" \
    --lr "${LR:-2e-5}" \
    --perturbed-reg-subset lambda0_only \
    ${EXTRA_ARGS:-}
