#!/bin/bash
#SBATCH --job-name=rollout_T_train
#SBATCH --account=def-mijungp
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --array=0-3
#SBATCH --output=output/rollout_T_train_%A_%a.out

# Train Llama-2-7B-Chat seen-family adapters for soft-rollout lengths T∈{1,3,5,10}.
# Fixed: λ=1, ε=0.5, lr=2e-5, 1 epoch, lm_loss=clean.
#
# Checkpoints:
#   $CHECKPOINT_ROOT/run_lr2e-05_lam1_eps0.5_T{T}_finetuned_llm_epoch1

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$SCRATCH/dp-llm-experiments}"
cd "$REPO_ROOT"
mkdir -p output
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

module load StdEnv/2023 python/3.11
# shellcheck disable=SC1090
source "${VENV_ACTIVATE:-$SCRATCH/venv/dp-llm-rorqual/bin/activate}"

export TRANSFORMERS_CACHE="${SLURM_TMPDIR}/hf_cache"
export HF_HOME="${SLURM_TMPDIR}/hf_home"
mkdir -p "$TRANSFORMERS_CACHE" "$HF_HOME"

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DISABLE_SERVICE=true
export WANDB_PROJECT="${WANDB_PROJECT:-dp-llm-rollout-T}"
export WANDB_DIR="${SLURM_TMPDIR}/wandb"
export WANDB_DIR_PERSISTENT="${WANDB_DIR_PERSISTENT:-$SCRATCH/wandb_offline}"
mkdir -p "$WANDB_DIR" "$WANDB_DIR_PERSISTENT"
trap 'cp -r "$WANDB_DIR"/* "$WANDB_DIR_PERSISTENT/" 2>/dev/null || true' EXIT

TS=(1 3 5 10)
T="${TS[$SLURM_ARRAY_TASK_ID]}"

LR="${LR:-2e-5}"
LAM="${LAM:-1}"
EPS="${EPS:-0.5}"
EPOCHS="${EPOCHS:-1}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$SCRATCH/dp-llm-sweep}"
TRAINING_DATA="${TRAINING_DATA:-$REPO_ROOT/official_data/train.csv}"
MODEL_PROFILE="${MODEL_PROFILE:-llama_2_7b_chat}"

# Match train.model_profiles.make_run_slug formatting (lr uses :g → 2e-05).
SLUG=$(
  python - <<PY
from train.model_profiles import make_run_slug
print(make_run_slug(float("$LR"), float("$LAM"), float("$EPS"), "clean",
                    model_profile="$MODEL_PROFILE", rollout_length=int("$T")))
PY
)
FINETUNED_BASE="${CHECKPOINT_ROOT}/${SLUG}_finetuned_llm"
mkdir -p "$(dirname "$FINETUNED_BASE")"

export SLUG
export WANDB_RUN_NAME="$SLUG"
export WANDB_RUN_ID
WANDB_RUN_ID=$(python -c "import zlib,os; print(f'{zlib.crc32(os.environ[\"SLUG\"].encode()) & 0xffffffff:08x}')")

echo "=== rollout-T train: T=$T slug=$SLUG ==="
echo "training_data=$TRAINING_DATA"
echo "finetuned_base=$FINETUNED_BASE"

python "$REPO_ROOT/train/train.py" \
  --eval-mode seen-family \
  --system-prompt-mode empty \
  --lm-loss-input clean \
  --model-profile "$MODEL_PROFILE" \
  --finetuned-llm-path "$FINETUNED_BASE" \
  --training-data "$TRAINING_DATA" \
  --lr "$LR" \
  --lambda-val "$LAM" \
  --epsilon "$EPS" \
  --lora-rank 8 \
  --rollout-length "$T" \
  --total-epochs "$EPOCHS" \
  --start-epoch 1

echo "Done. Checkpoint: ${FINETUNED_BASE}_epoch${EPOCHS}"
