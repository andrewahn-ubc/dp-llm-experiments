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
# CPU-only: reads test_eval_matrix ``*_metrics.tsv`` and runs ``plot_hyperparameter_heatmaps.py``.
# Default output: ``<METRICS_DIR>/heatmaps_<model>_lr<rate>/{aggregate,by_dataset}/`` — the Python
# script infers ``<model>`` and ``<rate>`` from the first ``clean_reg`` metrics TSV (no exports needed).
# Override the root only with HEATMAP_OUT_DIR (passed through as ``--output-dir``).
#
# Environment (defaults match submit_test_eval_matrix.sh):
#   REPO_ROOT       default: ${SCRATCH}/dp-llm-experiments
#   CHECKPOINT_ROOT default: ${SCRATCH}/dp-llm-sweep
#   METRICS_DIR     default: ${CHECKPOINT_ROOT}/test_eval_outputs
#   LABELS_CSV      default: ${REPO_ROOT}/official_data/combined_test_dataset.csv
#   HEATMAP_OUT_DIR optional: fixed output root (otherwise Python picks the heatmaps_* subdir)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p output

module load StdEnv/2023 python/3.11
source "${SCRATCH}/venv/nanogcg/bin/activate"

REPO_ROOT="${REPO_ROOT:-${SCRATCH}/dp-llm-experiments}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SCRATCH}/dp-llm-sweep}"
METRICS_DIR="${METRICS_DIR:-${CHECKPOINT_ROOT}/test_eval_outputs}"
LABELS_CSV="${LABELS_CSV:-${REPO_ROOT}/official_data/combined_test_dataset.csv}"

PY_ARGS=(--metrics-dir "${METRICS_DIR}" --labels "${LABELS_CSV}")
if [[ -n "${HEATMAP_OUT_DIR:-}" ]]; then
  PY_ARGS+=(--output-dir "${HEATMAP_OUT_DIR}")
fi

python "${REPO_ROOT}/eval/plot_hyperparameter_heatmaps.py" "${PY_ARGS[@]}"
