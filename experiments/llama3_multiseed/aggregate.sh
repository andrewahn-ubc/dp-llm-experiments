#!/bin/bash
#SBATCH --job-name=l3_ms_agg
#SBATCH --account=def-mijungp
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --output=output/l3_multiseed_aggregate_%j.out
#
# CPU job: merge seed0 image cells + seed1/2 points into mean±std tables.
#
#   EVAL_ID=...
#   sbatch --dependency=afterok:${EVAL_ID} experiments/llama3_multiseed/aggregate.sh

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$SCRATCH/dp-llm-experiments}"
cd "$REPO_ROOT"
mkdir -p output
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

module load StdEnv/2023 python/3.11
# shellcheck disable=SC1090
source "${VENV_ACTIVATE:-$SCRATCH/venv/nanogcg/bin/activate}"

EVAL_OUT_DIR="${EVAL_OUT_DIR:-$SCRATCH/dp-llm-eval/llama3_multiseed}"
OUT_DIR="${OUT_DIR:-${EVAL_OUT_DIR}/aggregate}"

python experiments/llama3_multiseed/aggregate_multiseed_tables.py \
  --seed0-csv "$REPO_ROOT/experiments/llama3_multiseed/seed0_from_image.csv" \
  --eval-root "$EVAL_OUT_DIR" \
  --out-dir "$OUT_DIR"
