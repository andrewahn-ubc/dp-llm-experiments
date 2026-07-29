#!/bin/bash
#SBATCH --job-name=final_test_ev
#SBATCH --account=rrg-mijungp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --array=0-5
#SBATCH --output=output/final_test_eval_%A_%a.out
#
# Final test-mode eval array: one task = one (λ, ε) cell.
# Each task runs seen + held-out gcg/autodan/pair for that cell on the test set.
#
#   cd $SCRATCH/dp-llm-experiments && mkdir -p output
#   sbatch eval/submit_final_test_eval.sh
#
# Task map (see eval/run_final_test_eval.py --list-tasks):
#   0: λ=1,ε=-0.5   1: λ=3,ε=0   2: λ=3,ε=1
#   3: λ=1,ε=-1     4: λ=30,ε=0  5: λ=30,ε=1
#
# Env overrides (also set by run_final_pipeline.py --mode test):
#   REPO_ROOT, CHECKPOINT_ROOT, MODEL_PROFILE, EPOCH, LR
#   HARMFUL_TEST, BENIGN_TEST, LABELS_CSV, OUT_DIR
#   EXTRA_ARGS

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SCRATCH}/dp-llm-experiments}"
if [[ ! -d "${REPO_ROOT}" ]]; then
  echo "ERROR: REPO_ROOT is not a directory: ${REPO_ROOT}" >&2
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
MODEL_PROFILE="${MODEL_PROFILE:-llama_3_8b_instruct}"

PY_ARGS=(
    --repo-root "${REPO_ROOT}"
    --checkpoint-root "${CHECKPOINT_ROOT}"
    --model-profile "${MODEL_PROFILE}"
    --epoch "${EPOCH:-2}"
    --lr "${LR:-2e-5}"
    --task-id "${SLURM_ARRAY_TASK_ID}"
    --system-prompt-mode "${SYSTEM_PROMPT_MODE:-empty}"
    --benign-system-prompt-mode "${BENIGN_SYSTEM_PROMPT_MODE:-empty}"
)
[[ -n "${HARMFUL_TEST:-}" ]] && PY_ARGS+=(--harmful-test="${HARMFUL_TEST}")
[[ -n "${BENIGN_TEST:-}" ]] && PY_ARGS+=(--benign-test="${BENIGN_TEST}")
[[ -n "${LABELS_CSV:-}" ]] && PY_ARGS+=(--labels-csv="${LABELS_CSV}")
[[ -n "${OUT_DIR:-}" ]] && PY_ARGS+=(--out-dir="${OUT_DIR}")

python "${REPO_ROOT}/eval/run_final_test_eval.py" "${PY_ARGS[@]}" ${EXTRA_ARGS:-}
