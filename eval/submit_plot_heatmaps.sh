#!/bin/bash
#SBATCH --job-name=plot_hm
#SBATCH --account=def-mijungp
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --output=output/plot_heatmaps_%j.out
#
# Default account is RAS (def-mijungp). If you use RAC instead, submit with:
#   sbatch --account=rrg-mijungp eval/submit_plot_heatmaps.sh
#
# CPU-only: aggregate test_eval_matrix metrics TSVs into 8 λ×ε heatmaps (PNG).
# Submit from repo root after eval finishes, or chain via run_final_pipeline.py.
#
# Environment (same defaults as submit_test_eval_matrix.sh):
#   REPO_ROOT   default: ${SCRATCH}/dp-llm-experiments
#   CHECKPOINT_ROOT  default: ${SCRATCH}/dp-llm-sweep
# Metrics are read from ${CHECKPOINT_ROOT}/test_eval_outputs by default (override METRICS_DIR).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p output

module load StdEnv/2023 python/3.11
source "${SCRATCH}/venv/nanogcg/bin/activate"

REPO_ROOT="${REPO_ROOT:-${SCRATCH}/dp-llm-experiments}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SCRATCH}/dp-llm-sweep}"
METRICS_DIR="${METRICS_DIR:-${CHECKPOINT_ROOT}/test_eval_outputs}"
OUT_DIR="${HEATMAP_OUT_DIR:-${METRICS_DIR}/heatmaps}"

python "${REPO_ROOT}/eval/plot_hyperparameter_heatmaps.py" \
    --metrics-dir "${METRICS_DIR}" \
    --output-dir "${OUT_DIR}"
