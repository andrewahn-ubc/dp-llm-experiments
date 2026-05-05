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
# CPU-only: reads test_eval_matrix ``*_metrics.tsv`` and writes ``plot_hyperparameter_heatmaps.py`` outputs:
#   ${OUT_DIR}/aggregate/   — clean-reg LM only; seen + held-out ASR/FRR incl. per-jailbreak-family (13 PNGs)
#   ${OUT_DIR}/by_dataset/ — stratified ASR per ``dataset`` column + duplicated global FRR panels
# Submit from repo root after eval finishes, or chain via run_final_pipeline.py.
#
# Environment (same defaults as submit_test_eval_matrix.sh):
#   REPO_ROOT    default: ${SCRATCH}/dp-llm-experiments
#   CHECKPOINT_ROOT   default: ${SCRATCH}/dp-llm-sweep
#   METRICS_DIR  default: ${CHECKPOINT_ROOT}/test_eval_outputs
#   LABELS_CSV   default: ${REPO_ROOT}/official/combined_test_dataset.csv (must include goal, target, dataset)
#   HEATMAP_OUT_DIR  default: ${METRICS_DIR}/heatmaps

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p output

module load StdEnv/2023 python/3.11
source "${SCRATCH}/venv/nanogcg/bin/activate"

REPO_ROOT="${REPO_ROOT:-${SCRATCH}/dp-llm-experiments}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SCRATCH}/dp-llm-sweep}"
METRICS_DIR="${METRICS_DIR:-${CHECKPOINT_ROOT}/test_eval_outputs}"
LABELS_CSV="${LABELS_CSV:-${REPO_ROOT}/official/combined_test_dataset.csv}"
OUT_DIR="${HEATMAP_OUT_DIR:-${METRICS_DIR}/heatmaps}"

python "${REPO_ROOT}/eval/plot_hyperparameter_heatmaps.py" \
    --metrics-dir "${METRICS_DIR}" \
    --output-dir "${OUT_DIR}" \
    --labels "${LABELS_CSV}"
