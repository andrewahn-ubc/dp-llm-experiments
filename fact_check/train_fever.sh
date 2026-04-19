#!/bin/bash
#SBATCH --job-name=fever-train
#SBATCH --account=aip-mijungp
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=output/fever_%x_%j.out
#SBATCH --signal=B:SIGUSR1@120

# =============================================================================
# All paths and sweep defaults come from fact_check/config.yaml.
# The only things injected here by sweep_fever.py are the per-job
# hyperparameter env vars:
#   LAMBDA, LR, EPSILON, LORA_RANK, EPOCHS, BATCH_SIZE, CKPT_EVERY_STEPS
# and optionally RESUME_FROM for resuming an interrupted run.
#
# To submit a single run manually:
#   LAMBDA=1.0 LR=2e-5 sbatch fact_check/train_fever.sh
# =============================================================================

set -euo pipefail

# ── Paths — must be set by sourcing env.sh fact-check before sbatch ───────────
: "${FC_REPO_ROOT:?FC_REPO_ROOT is not set. Run: source env.sh fact-check}"
: "${FC_DATA_DIR:?FC_DATA_DIR is not set. Run: source env.sh fact-check}"
: "${FC_MODEL_ROOT:?FC_MODEL_ROOT is not set. Run: source env.sh fact-check}"
: "${FC_OUTPUT_ROOT:?FC_OUTPUT_ROOT is not set. Run: source env.sh fact-check}"
: "${FC_VENV:?FC_VENV is not set. Run: source env.sh fact-check}"

REPO_ROOT="$FC_REPO_ROOT"
DATA_DIR="$FC_DATA_DIR"
MODEL_ROOT="$FC_MODEL_ROOT"
OUTPUT_ROOT="$FC_OUTPUT_ROOT"
VENV="$FC_VENV"
SLURM_LOG_DIR="${REPO_ROOT}/output"

# ── Hyperparameters — must be injected by sweep_fever.py or set manually ─────
: "${LAMBDA:?LAMBDA is not set}"
: "${LR:?LR is not set}"
: "${EPSILON:?EPSILON is not set}"
: "${BATCH_SIZE:?BATCH_SIZE is not set}"
: "${EPOCHS:?EPOCHS is not set}"
: "${LORA_RANK:?LORA_RANK is not set}"
: "${CKPT_EVERY_STEPS:?CKPT_EVERY_STEPS is not set}"
: "${MODEL_NAME:?MODEL_NAME is not set}"
RESUME_FROM="${RESUME_FROM:-}"

RUN_ID="${MODEL_NAME}_lam${LAMBDA}_eps${EPSILON}_lr${LR}_rank${LORA_RANK}"
OUTPUT_PATH="${OUTPUT_ROOT}/${RUN_ID}"
mkdir -p "${SLURM_LOG_DIR}" "${OUTPUT_PATH}"

echo "==================================================================="
echo "Job:      ${SLURM_JOB_ID}"
echo "Node:     $(hostname)"
echo "Run ID:   ${RUN_ID}"
echo "Output:   ${OUTPUT_PATH}"
echo "λ=${LAMBDA}  ε=${EPSILON}  lr=${LR}  batch=${BATCH_SIZE}  epochs=${EPOCHS}"
echo "Checkpoint every ${CKPT_EVERY_STEPS} steps"
[ -n "${RESUME_FROM}" ] && echo "Resuming from: ${RESUME_FROM}"
echo "==================================================================="

# ── SIGUSR1 handler — fires 120s before wall time ─────────────────────────────
_resume_hint() {
    echo ""
    echo "==================================================================="
    echo "WALL TIME APPROACHING — printing resume command."
    echo ""
    echo "Latest checkpoint: ${OUTPUT_PATH}/checkpoints/"
    echo "  ls -td ${OUTPUT_PATH}/checkpoints/*/ | head -1"
    echo ""
    echo "Re-submit with:"
    echo "  RESUME_FROM=\"\$(ls -td ${OUTPUT_PATH}/checkpoints/*/ | head -1)\" \\"
    echo "  LAMBDA=${LAMBDA} LR=${LR} EPSILON=${EPSILON} EPOCHS=${EPOCHS} \\"
    echo "  BATCH_SIZE=${BATCH_SIZE} LORA_RANK=${LORA_RANK} \\"
    echo "  sbatch fact_check/train_fever.sh"
    echo "==================================================================="
}
trap '_resume_hint' SIGUSR1

# ── Stage repo → $SLURM_TMPDIR ───────────────────────────────────────────────
TMP_REPO="${SLURM_TMPDIR}/repo"
echo "[stage] rsyncing repo ${REPO_ROOT} → ${TMP_REPO}"
rsync -a --exclude=".git" --exclude="__pycache__" --exclude="*.pyc" \
    "${REPO_ROOT}/" "${TMP_REPO}/"
echo "[stage] repo ready"

# ── Stage model weights → $SLURM_TMPDIR ──────────────────────────────────────
SRC_MODEL="${MODEL_ROOT}/${MODEL_NAME}"
TMP_MODEL="${SLURM_TMPDIR}/models/${MODEL_NAME}"
echo "[stage] rsyncing model ${SRC_MODEL} → ${TMP_MODEL}"
mkdir -p "${TMP_MODEL}"
rsync -a "${SRC_MODEL}/" "${TMP_MODEL}/"
echo "[stage] model ready"

# ── Stage data → $SLURM_TMPDIR ───────────────────────────────────────────────
TMP_DATA="${SLURM_TMPDIR}/data"
echo "[stage] rsyncing data ${DATA_DIR} → ${TMP_DATA}"
mkdir -p "${TMP_DATA}"
rsync -a "${DATA_DIR}/" "${TMP_DATA}/"
echo "[stage] data ready"

# ── Python environment ────────────────────────────────────────────────────────
module purge
module load StdEnv/2023 python/3.11

TMP_VENV="${SLURM_TMPDIR}/venv"
echo "[env] creating venv at ${TMP_VENV} (seeding from ${VENV})"
python -m venv --copies "${TMP_VENV}"
source "${TMP_VENV}/bin/activate"

# ── Disable all HuggingFace network calls ────────────────────────────────────
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Point HF caches at SLURM_TMPDIR so any incidental cache writes stay local.
export HF_HOME="${SLURM_TMPDIR}/hf_home"
export TRANSFORMERS_CACHE="${SLURM_TMPDIR}/hf_cache"
export HF_DATASETS_CACHE="${SLURM_TMPDIR}/hf_datasets"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}"

# ── CUDA ─────────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
echo "[env] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
python -c "import torch; print('[env] GPU:', torch.cuda.get_device_name(0))"

# ── Build resume flag ─────────────────────────────────────────────────────────
RESUME_FLAG=""
[ -n "${RESUME_FROM}" ] && RESUME_FLAG="--resume-from-checkpoint ${RESUME_FROM}"

# ── Run ───────────────────────────────────────────────────────────────────────
cd "${TMP_REPO}"

python -m fact_check.train_fever \
    --model-path          "${TMP_MODEL}"         \
    --train-csv           "${TMP_DATA}/fever_train_perturbed.csv" \
    --val-csv             "${TMP_DATA}/fever_val.csv"             \
    --sym-val-csv         "${TMP_DATA}/fever_symmetric.csv"       \
    --output-path         "${OUTPUT_PATH}"        \
    --run-id              "${RUN_ID}"             \
    --lambda-val          "${LAMBDA}"             \
    --epsilon             "${EPSILON}"            \
    --lr                  "${LR}"                \
    --batch-size          "${BATCH_SIZE}"         \
    --epochs              "${EPOCHS}"             \
    --lora-rank           "${LORA_RANK}"          \
    --ckpt-every-steps    "${CKPT_EVERY_STEPS}"   \
    ${RESUME_FLAG}

echo ""
echo "==================================================================="
echo "Job complete."
echo "  Output:  ${OUTPUT_PATH}"
echo "  Log:     ${OUTPUT_PATH}/${RUN_ID}.log"
echo "  Results: ${OUTPUT_PATH}/training_log.json"
echo "==================================================================="
