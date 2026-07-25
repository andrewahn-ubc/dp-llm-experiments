#!/bin/bash
#SBATCH --job-name=test_eval_mx
#SBATCH --account=rrg-mijungp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=2:15:00
#SBATCH --array=0-0
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
# Rerun only some array tasks (CLI --array replaces the default range above), e.g.:
#
#   cd /path/to/dp-llm-experiments && mkdir -p output && sbatch --array=0,2 eval/submit_test_eval_matrix.sh
#
# If your repo is not ${SCRATCH}/dp-llm-experiments, set REPO_ROOT before sbatch:
#
#   export REPO_ROOT=/lustre07/scratch/you/dp-llm-experiments
#   cd "$REPO_ROOT" && mkdir -p output && sbatch --array=0,2 eval/submit_test_eval_matrix.sh
#
# Heatmaps after this eval array finishes (use the Job ID sbatch prints):
#   sbatch --dependency=afterok:<JOBID> eval/submit_plot_heatmaps.sh
#
# ---------------------------------------------------------------------------
#
# One array task = one (λ, ε) cell of the clean grid (λ=0 baseline + λ>0 regularized),
# running 1× seen-family + 3× unseen eval. Default --perturbed-reg-subset none (no separate
# perturbed-LM task; clean/perturbed is purely λ=0 vs λ>0).
# Set --array to len(tasks)-1 from: python eval/test_eval_matrix.py --list-tasks
# (run_final_pipeline.py sets --array automatically to match the sweep grid).
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

# Optional overrides (used e.g. by run_final_pipeline.py for validation-set eval):
#   LAMBDAS / EPSILONS          λ×ε grid (must match the training sweep)
#   HARMFUL_TEST / BENIGN_TEST  harmful (ASR) / benign (FRR) CSVs
#   OUT_DIR                     metrics + per-example CSV output dir
#   SYSTEM_PROMPT_MODE          system-prompt policy for harmful eval (default: empty)
#   BENIGN_SYSTEM_PROMPT_MODE   system-prompt policy for FRR eval (e.g. empty = no sys prompt)
# clean/perturbed is λ-based now: the clean grid (λ=0 baseline + λ>0 regularized) covers
# everything, so there is no separate perturbed-LM task. Override with
# PERTURBED_REG_SUBSET=lambda0_only|full only for the legacy R2D2 perturbed-LM sweep.
PY_ARGS=(
    --repo-root "${REPO_ROOT}"
    --checkpoint-root "${CHECKPOINT_ROOT}"
    --model-profile "${MODEL_PROFILE}"
    --epoch "${EPOCH:-5}"
    --lr "${LR:-2e-5}"
    --perturbed-reg-subset "${PERTURBED_REG_SUBSET:-none}"
)
# Use --opt=value form: EPSILONS can start with '-' (e.g. -1,-0.5,...), which argparse
# would otherwise misread as a flag.
[[ -n "${LAMBDAS:-}" ]] && PY_ARGS+=(--lambdas="${LAMBDAS}")
[[ -n "${EPSILONS:-}" ]] && PY_ARGS+=(--epsilons="${EPSILONS}")
[[ -n "${HARMFUL_TEST:-}" ]] && PY_ARGS+=(--harmful-test="${HARMFUL_TEST}")
[[ -n "${BENIGN_TEST:-}" ]] && PY_ARGS+=(--benign-test="${BENIGN_TEST}")
[[ -n "${OUT_DIR:-}" ]] && PY_ARGS+=(--out-dir="${OUT_DIR}")
[[ -n "${SYSTEM_PROMPT_MODE:-}" ]] && PY_ARGS+=(--system-prompt-mode="${SYSTEM_PROMPT_MODE}")
[[ -n "${BENIGN_SYSTEM_PROMPT_MODE:-}" ]] && PY_ARGS+=(--benign-system-prompt-mode="${BENIGN_SYSTEM_PROMPT_MODE}")

python "${REPO_ROOT}/eval/test_eval_matrix.py" "${PY_ARGS[@]}" ${EXTRA_ARGS:-}
