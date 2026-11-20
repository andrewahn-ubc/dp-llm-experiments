#!/bin/bash
#SBATCH --job-name=l2b_report
#SBATCH --account=def-mijungp
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=output/l2b_report_%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SCRATCH}/dp-llm-experiments}"
VENV_ACTIVATE="${VENV_ACTIVATE:-${SCRATCH}/venv/nanogcg/bin/activate}"
: "${MANIFEST:?MANIFEST must point to the evaluation TSV}"
: "${REPORT_DIR:?REPORT_DIR must be set}"

cd "$REPO_ROOT"
mkdir -p output
module load StdEnv/2023 python/3.11
source "$VENV_ACTIVATE"

python -m experiments.llama2_benign_sweep.aggregate_results \
    --manifest "$MANIFEST" \
    --output-dir "$REPORT_DIR"
