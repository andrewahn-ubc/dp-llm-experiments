#!/bin/bash
#SBATCH --job-name=l3_ms_train
#SBATCH --account=def-mijungp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --array=0-19
#SBATCH --output=output/l3_multiseed_train_%A_%a.out
#
# Train Llama-3 seeds {1,2} × 10 table configs (seen 3 + heldout 7).
# Protocol matches the image runs: --total-epochs 1 with both data halves → *_epoch2.
#
#   cd $SCRATCH/dp-llm-experiments && mkdir -p output
#   sbatch experiments/llama3_multiseed/train_array.sh
#
# Checkpoints: $SCRATCH/dp-llm-sweep/multiseed_l3/{slug}_finetuned_llm_epoch2

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$SCRATCH/dp-llm-experiments}"
cd "$REPO_ROOT"
mkdir -p output
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

module load StdEnv/2023 python/3.11
# shellcheck disable=SC1090
source "${VENV_ACTIVATE:-$SCRATCH/venv/nanogcg/bin/activate}"

export TRANSFORMERS_CACHE="${SLURM_TMPDIR}/hf_cache"
export HF_HOME="${SLURM_TMPDIR}/hf_home"
mkdir -p "$TRANSFORMERS_CACHE" "$HF_HOME"

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DISABLE_SERVICE=true
export WANDB_PROJECT="${WANDB_PROJECT:-dp-llm-l3-multiseed}"
export WANDB_DIR="${SLURM_TMPDIR}/wandb"
export WANDB_DIR_PERSISTENT="${WANDB_DIR_PERSISTENT:-$SCRATCH/wandb_offline}"
mkdir -p "$WANDB_DIR" "$WANDB_DIR_PERSISTENT"
trap 'cp -r "$WANDB_DIR"/* "$WANDB_DIR_PERSISTENT/" 2>/dev/null || true' EXIT

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$SCRATCH/dp-llm-sweep/multiseed_l3}"
TRAINING_DATA="${TRAINING_DATA:-$REPO_ROOT/official_data/llama3_train_plus_validation.csv}"
MODEL_PROFILE="${MODEL_PROFILE:-llama_3_8b_instruct}"
LR="${LR:-2e-5}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?}"

# Resolve seed + config from array index.
eval "$(
  python - "$TASK_ID" "$CHECKPOINT_ROOT" <<'PY'
import sys
from experiments.llama3_multiseed.configs import array_index_to_seed_config

task_id = int(sys.argv[1])
ck_root = sys.argv[2]
seed, cfg = array_index_to_seed_config(task_id)
slug = cfg.run_slug(seed)
base = cfg.finetuned_base(ck_root, seed)
def q(s: str) -> str:
    return "'" + str(s).replace("'", "'\"'\"'") + "'"

print(f"SEED={seed}")
print(f"CONFIG_ID={q(cfg.config_id)}")
print(f"ROLE={q(cfg.role)}")
print(f"LAM={cfg.lam:g}")
print(f"EPS={cfg.eps:g}")
print(f"LM_LOSS={q(cfg.lm_loss_input)}")
print(f"HELDOUT_FAM={q(cfg.heldout_family or '')}")
print(f"SLUG={q(slug)}")
print(f"FINETUNED_BASE={q(base)}")
PY
)"

export WANDB_RUN_NAME="$SLUG"
WANDB_RUN_ID=$(python -c "import zlib,os; print(f'{zlib.crc32(os.environ[\"SLUG\"].encode()) & 0xffffffff:08x}')")
export WANDB_RUN_ID

mkdir -p "$(dirname "$FINETUNED_BASE")"

echo "=== l3 multiseed train task=$TASK_ID seed=$SEED config=$CONFIG_ID ==="
echo "slug=$SLUG"
echo "finetuned_base=$FINETUNED_BASE"
echo "training_data=$TRAINING_DATA"

EVAL_MODE="seen-family"
UNSEEN_ARGS=()
if [[ "$ROLE" == "heldout" ]]; then
  EVAL_MODE="unseen-family"
  UNSEEN_ARGS=(--unseen-family "$HELDOUT_FAM")
fi

# Half 1 → *_epoch1
python -m train.train \
  --eval-mode "$EVAL_MODE" \
  "${UNSEEN_ARGS[@]}" \
  --system-prompt-mode empty \
  --lm-loss-input "$LM_LOSS" \
  --model-profile "$MODEL_PROFILE" \
  --finetuned-llm-path "$FINETUNED_BASE" \
  --training-data "$TRAINING_DATA" \
  --lr "$LR" \
  --lambda-val "$LAM" \
  --epsilon "$EPS" \
  --lora-rank 8 \
  --total-epochs 2 \
  --start-epoch 1 \
  --train-data-frac-start 0.0 \
  --train-data-frac-end 0.5 \
  --seed "$SEED" \
  --training-shuffle-seed "$SEED"

# Half 2 → *_epoch2
python -m train.train \
  --eval-mode "$EVAL_MODE" \
  "${UNSEEN_ARGS[@]}" \
  --system-prompt-mode empty \
  --lm-loss-input "$LM_LOSS" \
  --model-profile "$MODEL_PROFILE" \
  --finetuned-llm-path "$FINETUNED_BASE" \
  --training-data "$TRAINING_DATA" \
  --lr "$LR" \
  --lambda-val "$LAM" \
  --epsilon "$EPS" \
  --lora-rank 8 \
  --total-epochs 2 \
  --start-epoch 2 \
  --resume-from "${FINETUNED_BASE}_epoch1" \
  --train-data-frac-start 0.5 \
  --train-data-frac-end 1.0 \
  --seed "$SEED" \
  --training-shuffle-seed "$SEED"

echo "Done. Checkpoint: ${FINETUNED_BASE}_epoch2"
