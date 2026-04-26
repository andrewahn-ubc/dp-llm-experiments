#!/bin/bash
#SBATCH --job-name=reclassify_frr
#SBATCH --account=rrg-mijungp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=1:00:00
#SBATCH --output=output/reclassify_frr_%j.out

# Re-judges the `Original Safety` column on every *_benign.csv(.csv) file in
# RESULTS_DIR using Mistral-7B-Instruct-v0.2. Does NOT regenerate any model
# responses - it only re-runs the refusal classifier on the saved
# `Original Response` column.
#
# Override defaults via env vars, e.g.:
#   sbatch \
#     --export=ALL,RESULTS_DIR=$SCRATCH/dp-llm-eval/final_eval,JUDGE_PATH=$SCRATCH/mistral_7b_instruct \
#     eval/reclassify_frr.sh
#
# Or run a dry-run (prints new FRR but doesn't overwrite the CSVs) with:
#   sbatch --export=ALL,DRY_RUN=1 eval/reclassify_frr.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p output

# Load environment (matches seen_family_eval.sh / heldout_family_eval.sh).
module load StdEnv/2023 python/3.11
source "$SCRATCH/venv/nanogcg/bin/activate"

# Use node-local storage for HF caches.
export TRANSFORMERS_CACHE="${SLURM_TMPDIR:-/tmp}/hf_cache"
export HF_HOME="${SLURM_TMPDIR:-/tmp}/hf_home"
mkdir -p "$TRANSFORMERS_CACHE" "$HF_HOME"

# ---------------------------------------------------------------------------
# Config (override via env)
# ---------------------------------------------------------------------------
RESULTS_DIR="${RESULTS_DIR:-$SCRATCH/dp-llm-eval/final_eval}"
JUDGE_PATH="${JUDGE_PATH:-$SCRATCH/mistral_7b_instruct}"
BATCH_SIZE="${BATCH_SIZE:-8}"
DRY_RUN="${DRY_RUN:-0}"

# Resolve script location so this works from any CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/reclassify_frr.py"

if [[ ! -d "$RESULTS_DIR" ]]; then
    echo "RESULTS_DIR does not exist: $RESULTS_DIR" >&2
    exit 1
fi
if [[ ! -f "$PY_SCRIPT" ]]; then
    echo "reclassify_frr.py not found at: $PY_SCRIPT" >&2
    exit 1
fi

echo "RESULTS_DIR = $RESULTS_DIR"
echo "JUDGE_PATH  = $JUDGE_PATH"
echo "BATCH_SIZE  = $BATCH_SIZE"
echo "DRY_RUN     = $DRY_RUN"
echo
echo "Benign CSVs to re-judge:"
ls -1 "$RESULTS_DIR"/*_benign.csv* 2>/dev/null || {
    echo "  (none found)" >&2
    exit 1
}
echo

EXTRA_ARGS=()
if [[ "$DRY_RUN" == "1" ]]; then
    EXTRA_ARGS+=(--dry-run)
fi

python "$PY_SCRIPT" \
    --results-dir "$RESULTS_DIR" \
    --judge-path "$JUDGE_PATH" \
    --batch-size "$BATCH_SIZE" \
    "${EXTRA_ARGS[@]}"

echo
echo "Re-classification complete."
echo "Re-aggregate corrected metrics with:"
echo "  python official/aggregate_final_results.py \\"
echo "    --results-dir $RESULTS_DIR \\"
echo "    --out-csv $RESULTS_DIR/aggregated_summary.csv"
