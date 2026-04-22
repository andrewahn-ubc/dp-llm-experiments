#!/bin/bash
#SBATCH --job-name=fever-eval
#SBATCH --account=aip-mijungp
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=output/fever_eval_%x_%j.out
#SBATCH --mail-user=ammany01@cs.ubc.ca
#SBATCH --mail-type=FAIL,TIME_LIMIT

# =============================================================================
# Submit eval_consistency.py as a SLURM job for long-running evaluations.
#
# Usage (after sourcing env.sh fact-check):
#   sbatch fact_check/eval_consistency.sh
#
# Override the filter at submission time:
#   EVAL_FILTER="loo_" sbatch fact_check/eval_consistency.sh
#   EVAL_FILTER="20260420_161433" sbatch fact_check/eval_consistency.sh
#
# Override batch size, device, or output path:
#   EVAL_BATCH_SIZE=64 sbatch fact_check/eval_consistency.sh
#   EVAL_OUTPUT=/path/to/my_results.csv sbatch fact_check/eval_consistency.sh
# =============================================================================

set -euo pipefail

# ── Paths — must be set by sourcing env.sh fact-check before sbatch ───────────
: "${FC_REPO_ROOT:?FC_REPO_ROOT is not set. Run: source env.sh fact-check}"
: "${FC_DATA_DIR:?FC_DATA_DIR is not set. Run: source env.sh fact-check}"
: "${FC_OUTPUT_ROOT:?FC_OUTPUT_ROOT is not set. Run: source env.sh fact-check}"
: "${FC_VENV:?FC_VENV is not set. Run: source env.sh fact-check}"

REPO_ROOT="$FC_REPO_ROOT"
DATA_DIR="$FC_DATA_DIR"
OUTPUT_ROOT="$FC_OUTPUT_ROOT"
VENV="$FC_VENV"
SLURM_LOG_DIR="${REPO_ROOT}/output"

# ── Eval options ──────────────────────────────────────────────────────────────
EVAL_FILTER="${EVAL_FILTER:-}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
EVAL_OUTPUT="${EVAL_OUTPUT:-${OUTPUT_ROOT}/consistency_eval.csv}"

echo "==================================================================="
echo "Job:          ${SLURM_JOB_ID}"
echo "Node:         $(hostname)"
echo "Models dir:   ${OUTPUT_ROOT}"
echo "Filter:       ${EVAL_FILTER:-<all runs>}"
echo "Batch size:   ${EVAL_BATCH_SIZE}"
echo "Output CSV:   ${EVAL_OUTPUT}"
echo "==================================================================="

mkdir -p "${SLURM_LOG_DIR}"

# ── Stage repo → $SLURM_TMPDIR ───────────────────────────────────────────────
TMP_REPO="${SLURM_TMPDIR}/repo"
echo "[stage] rsyncing repo ${REPO_ROOT} → ${TMP_REPO}"
rsync -a --exclude=".git" --exclude="__pycache__" --exclude="*.pyc" \
    "${REPO_ROOT}/" "${TMP_REPO}/"
echo "[stage] repo ready"

# ── Stage data → $SLURM_TMPDIR ───────────────────────────────────────────────
TMP_DATA="${SLURM_TMPDIR}/data"
echo "[stage] rsyncing data ${DATA_DIR} → ${TMP_DATA}"
mkdir -p "${TMP_DATA}"
rsync -a "${DATA_DIR}/" "${TMP_DATA}/"
echo "[stage] data ready"

# ── Python environment ────────────────────────────────────────────────────────
module purge
module load StdEnv/2023 gcc python/3.11 arrow/21.0.0

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

TMP_VENV="${SLURM_TMPDIR}/venv"
echo "[env] rsyncing venv ${VENV} → ${TMP_VENV}"
rsync -a "${VENV}/" "${TMP_VENV}/"
source "${TMP_VENV}/bin/activate"

# ── Disable all HuggingFace network calls ────────────────────────────────────
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

export HF_HOME="${SLURM_TMPDIR}/hf_home"
export TRANSFORMERS_CACHE="${SLURM_TMPDIR}/hf_cache"
export HF_DATASETS_CACHE="${SLURM_TMPDIR}/hf_datasets"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
echo "[env] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
python -c "import torch; print('[env] GPU:', torch.cuda.get_device_name(0))"

# ── Build optional filter flag ────────────────────────────────────────────────
FILTER_FLAG=""
[ -n "${EVAL_FILTER}" ] && FILTER_FLAG="--filter ${EVAL_FILTER}"

# ── Run ───────────────────────────────────────────────────────────────────────
cd "${TMP_REPO}"

python -m fact_check.eval_consistency \
    --models-dir  "${OUTPUT_ROOT}"             \
    --val-csv     "${TMP_DATA}/fever_val.csv"  \
    --sym-val-csv "${TMP_DATA}/fever_symmetric.csv" \
    --batch-size  "${EVAL_BATCH_SIZE}"         \
    --device      cuda                         \
    --output      "${EVAL_OUTPUT}"             \
    ${FILTER_FLAG}

echo ""
echo "==================================================================="
echo "Eval complete."
echo "  Results: ${EVAL_OUTPUT}"
echo "==================================================================="
