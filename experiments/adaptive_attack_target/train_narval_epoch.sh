#!/bin/bash
#SBATCH --job-name=adapt_tgt_l2_ep
#SBATCH --account=def-mijungp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=3:00:00
#SBATCH --output=output/adaptive_attack_target_l2_ep_%j.out

# One epoch of the adaptive-attack target fine-tune (Narval).
# Set START_EPOCH via --export when submitting (see submit_narval_5ep.sh).
#
# Checkpoint:
#   $SCRATCH/adaptive_attack_target/llama2_7b_chat_lam0.1_eps-0.5_finetuned_llm_epoch${START_EPOCH}

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$SCRATCH/dp-llm-experiments}"
cd "$REPO_ROOT"
mkdir -p output
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

START_EPOCH="${START_EPOCH:?Set START_EPOCH (1..5) via sbatch --export}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-5}"

module load StdEnv/2023 python/3.11
# shellcheck disable=SC1090
source "${VENV_ACTIVATE:-$SCRATCH/venv/nanogcg/bin/activate}"

export TRANSFORMERS_CACHE="${SLURM_TMPDIR}/hf_cache"
export HF_HOME="${SLURM_TMPDIR}/hf_home"
mkdir -p "$TRANSFORMERS_CACHE" "$HF_HOME"

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DISABLE_SERVICE=true
export WANDB_PROJECT="${WANDB_PROJECT:-dp-llm-adaptive-target}"
export WANDB_DIR="${SLURM_TMPDIR}/wandb"
export WANDB_DIR_PERSISTENT="${WANDB_DIR_PERSISTENT:-$SCRATCH/wandb_offline}"
mkdir -p "$WANDB_DIR" "$WANDB_DIR_PERSISTENT"
trap 'cp -r "$WANDB_DIR"/* "$WANDB_DIR_PERSISTENT/" 2>/dev/null || true' EXIT

LR="${LR:-2e-5}"
LAM="${LAM:-0.1}"
EPS="${EPS:--0.5}"
MODEL_PROFILE="${MODEL_PROFILE:-llama_2_7b_chat}"
TRAINING_DATA="${TRAINING_DATA:-$REPO_ROOT/official_data/train_plus_validation.csv}"
TARGET_ROOT="${TARGET_ROOT:-$SCRATCH/adaptive_attack_target}"
FINETUNED_BASE="${TARGET_ROOT}/llama2_7b_chat_lam0.1_eps-0.5_finetuned_llm"

SLUG=$(
  python - <<PY
from train.model_profiles import make_run_slug
print(make_run_slug(float("$LR"), float("$LAM"), float("$EPS"), "clean",
                    model_profile="$MODEL_PROFILE"))
PY
)
mkdir -p "$TARGET_ROOT"
echo "$SLUG" > "${TARGET_ROOT}/slug.txt"

export SLUG
export WANDB_RUN_NAME="adaptive_target_${SLUG}_ep${START_EPOCH}"
export WANDB_RUN_ID
WANDB_RUN_ID=$(python -c "import zlib,os; s=os.environ['SLUG']+':'+os.environ['START_EPOCH']; print(f'{zlib.crc32(s.encode()) & 0xffffffff:08x}')")

echo "=== adaptive attack target — epoch ${START_EPOCH}/${TOTAL_EPOCHS} ==="
echo "slug=$SLUG"
echo "finetuned_base=$FINETUNED_BASE"
echo "training_data=$TRAINING_DATA"
echo "λ=$LAM ε=$EPS lr=$LR"

RESUME_ARGS=()
if [[ "$START_EPOCH" -gt 1 ]]; then
  PREV="${FINETUNED_BASE}_epoch$((START_EPOCH - 1))"
  if [[ ! -d "$PREV" ]]; then
    echo "ERROR: missing previous checkpoint $PREV" >&2
    exit 2
  fi
  RESUME_ARGS=(--resume-from "$PREV")
fi

python -m train.train \
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
  --total-epochs "$TOTAL_EPOCHS" \
  --start-epoch "$START_EPOCH" \
  "${RESUME_ARGS[@]}"

OUT="${FINETUNED_BASE}_epoch${START_EPOCH}"
echo "Saved: $OUT"

if [[ "$START_EPOCH" -eq "$TOTAL_EPOCHS" ]]; then
  ln -sfn "$OUT" "${TARGET_ROOT}/LATEST_epoch5"
  cat > "${TARGET_ROOT}/README_PATHS.txt" <<EOF
slug=${SLUG}
finetuned_base=${FINETUNED_BASE}
final_epoch5=${OUT}
symlink=${TARGET_ROOT}/LATEST_epoch5
suggested_merge_out=\$SCRATCH/merged_adaptive_l2_lam0.1_eps-0.5_ep5
EOF
  echo "Final symlink: ${TARGET_ROOT}/LATEST_epoch5"
fi
