#!/bin/bash
# Submit train array then seen-eval array (eval waits for train) on Rorqual.
#
# Usage (from repo root on Rorqual):
#   bash experiments/rollout_T_sensitivity/submit_all.sh
#
# Optional env overrides:
#   ACCOUNT, REPO_ROOT, VENV_ACTIVATE, TRAINING_DATA, HARMFUL_DATA, BENIGN_DATA,
#   CHECKPOINT_ROOT, OUT_DIR

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$SCRATCH/dp-llm-experiments}"
cd "$REPO_ROOT"
mkdir -p output

ACCOUNT="${ACCOUNT:-def-mijungp}"
TRAIN_SH="$REPO_ROOT/experiments/rollout_T_sensitivity/train_array.sh"
EVAL_SH="$REPO_ROOT/experiments/rollout_T_sensitivity/eval_seen_array.sh"

echo "Submitting train array (T=1,3,5,10) ..."
TRAIN_MSG=$(sbatch --account="$ACCOUNT" "$TRAIN_SH")
echo "$TRAIN_MSG"
TRAIN_ID=$(echo "$TRAIN_MSG" | awk '{print $NF}')

echo "Submitting seen-eval array after train job $TRAIN_ID ..."
EVAL_MSG=$(sbatch --account="$ACCOUNT" --dependency=afterok:"$TRAIN_ID" "$EVAL_SH")
echo "$EVAL_MSG"

echo
echo "Metrics will land under:"
echo "  \${CHECKPOINT_ROOT:-$SCRATCH/dp-llm-sweep}/rollout_T_sensitivity/seen_run_lr2e-05_lam0.1_eps-0.5_T{T}_ep1_metrics.tsv"
