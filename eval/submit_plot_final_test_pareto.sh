#!/bin/bash
#SBATCH --job-name=final_pareto
#SBATCH --account=def-mijungp
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --output=output/final_test_pareto_%j.out
#
# CPU job: build 32 ASR-vs-FRR Pareto PNGs from final-test points_*.csv.
# Chain after the eval array, e.g.:
#   sbatch --dependency=afterok:<EVAL_ARRAY_JOBID> eval/submit_plot_final_test_pareto.sh
#
# Env: REPO_ROOT, POINTS_DIR (or OUT_DIR / METRICS_DIR), PARETO_OUT_DIR

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SCRATCH}/dp-llm-experiments}"
cd "${REPO_ROOT}"
mkdir -p output

module load StdEnv/2023 python/3.11
source "${SCRATCH}/venv/nanogcg/bin/activate"

POINTS_DIR="${POINTS_DIR:-${OUT_DIR:-${METRICS_DIR:-${SCRATCH}/dp-llm-sweep/final_test_outputs/lr2e-5}}}"
PARETO_OUT_DIR="${PARETO_OUT_DIR:-${POINTS_DIR}/pareto_charts}"

python "${REPO_ROOT}/eval/plot_final_test_pareto.py" \
  --points-dir "${POINTS_DIR}" \
  --out-dir "${PARETO_OUT_DIR}"
