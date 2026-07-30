#!/bin/bash
# Submit 5 consecutive 3h Narval jobs (epochs 1→5) with afterok dependencies.
#
# Usage (from repo root on Narval):
#   bash experiments/adaptive_attack_target/submit_narval_5ep.sh
#
# Optional: ACCOUNT=def-XXXX bash experiments/adaptive_attack_target/submit_narval_5ep.sh

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$SCRATCH/dp-llm-experiments}"
cd "$REPO_ROOT"
mkdir -p output

ACCOUNT="${ACCOUNT:-def-mijungp}"
EPOCH_SH="$REPO_ROOT/experiments/adaptive_attack_target/train_narval_epoch.sh"
TOTAL_EPOCHS=5

PREV_ID=""
for ep in $(seq 1 "$TOTAL_EPOCHS"); do
  EXPORT="ALL,START_EPOCH=${ep},TOTAL_EPOCHS=${TOTAL_EPOCHS},REPO_ROOT=${REPO_ROOT}"
  SBATCH_ARGS=(--account="$ACCOUNT" --job-name="adapt_tgt_l2_ep${ep}" --export="$EXPORT")
  if [[ -n "$PREV_ID" ]]; then
    SBATCH_ARGS+=(--dependency="afterok:${PREV_ID}")
  fi
  MSG=$(sbatch "${SBATCH_ARGS[@]}" "$EPOCH_SH")
  echo "$MSG  (epoch $ep)"
  PREV_ID=$(echo "$MSG" | awk '{print $NF}')
done

echo
echo "Chained epochs 1→${TOTAL_EPOCHS} (each 3:00:00 wall)."
echo "Final adapter:"
echo "  \$SCRATCH/adaptive_attack_target/llama2_7b_chat_lam0.1_eps-0.5_finetuned_llm_epoch5"
