#!/bin/bash
# Submits a 5-epoch chained training run (up to 4h/epoch on an L40S, aip-mijungp)
# for one rebuttal core-design ablation (Section 2 items 2.1-2.4), mirroring
# submit_all_train.sh's epoch chaining but parameterized by ABLATION_TAG via
# train/ablation_epoch.sh. Targets Vulcan; see that script's header for the
# one-time venv setup (train/setup_ablation_venv.sh).
#
# After each epoch's training job, also submits a SEPARATE seen-family ASR/FRR
# eval job (train/ablation_eval_epoch.sh) depending only on that epoch --
# eval(N) and epoch(N+1) both depend on epoch(N) and run independently of each
# other (whichever gets a free GPU first), so eval results trickle in as
# progress signal without gating the training chain's wall-clock.
#
# Usage (from repo root on Vulcan, e.g. $HOME/repos/dp-llm-experiments):
#   bash train/submit_ablation.sh clean_adv_sft
#   bash train/submit_ablation.sh symmetric
#   bash train/submit_ablation.sh direct_unsafe
#   bash train/submit_ablation.sh no_clean_sft
#
# Or submit all four:
#   for t in clean_adv_sft symmetric direct_unsafe no_clean_sft; do
#       bash train/submit_ablation.sh "$t"
#   done
#
# Skip the per-epoch eval jobs with SKIP_EVAL=1 (training-only, as before):
#   SKIP_EVAL=1 bash train/submit_ablation.sh clean_adv_sft

set -euo pipefail

ABLATION_TAG="${1:?Usage: submit_ablation.sh <clean_adv_sft|symmetric|direct_unsafe|no_clean_sft>}"
export ABLATION_TAG

TOTAL_EPOCHS="${TOTAL_EPOCHS:-5}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

FINETUNED_BASE="${SCRATCH}/ablation_${ABLATION_TAG}_finetuned_llm"

PREV_JOB=""
PREV_CKPT=""
for ((epoch=1; epoch<=TOTAL_EPOCHS; epoch++)); do
  export START_EPOCH="${epoch}"
  if [[ -n "${PREV_CKPT}" ]]; then
    export RESUME_FROM="${PREV_CKPT}"
  else
    unset RESUME_FROM || true
  fi

  # --export=ALL,... explicitly forwards our loop variables; sbatch's default
  # (implicit ALL) would also work on Narval, but this makes the dependency
  # on env-var inheritance visible rather than incidental.
  EXPORT_VARS="ALL,ABLATION_TAG,START_EPOCH"
  if [[ -n "${RESUME_FROM:-}" ]]; then
    EXPORT_VARS="${EXPORT_VARS},RESUME_FROM"
  fi

  if [[ -z "${PREV_JOB}" ]]; then
    JOB=$(sbatch --parsable --export="${EXPORT_VARS}" "${SCRIPT_DIR}/ablation_epoch.sh")
  else
    JOB=$(sbatch --parsable --export="${EXPORT_VARS}" --dependency=afterok:"${PREV_JOB}" "${SCRIPT_DIR}/ablation_epoch.sh")
  fi
  echo "Submitted ${ABLATION_TAG} epoch ${epoch}: ${JOB} (depends on ${PREV_JOB:-<none>})"

  if [[ "${SKIP_EVAL}" != "1" ]]; then
    EVAL_JOB=$(sbatch --parsable --export="ALL,ABLATION_TAG,EPOCH=${epoch}" \
      --dependency=afterok:"${JOB}" "${SCRIPT_DIR}/ablation_eval_epoch.sh")
    echo "Submitted ${ABLATION_TAG} epoch ${epoch} eval: ${EVAL_JOB} (depends on ${JOB}; independent of epoch $((epoch + 1)))"
  fi

  PREV_JOB="${JOB}"
  PREV_CKPT="${FINETUNED_BASE}_epoch${epoch}"
done

echo "Final adapter (after job ${PREV_JOB} completes): ${PREV_CKPT}"
