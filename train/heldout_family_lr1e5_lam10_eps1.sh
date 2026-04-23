#!/bin/bash
#SBATCH --job-name=heldout_fam_lr1e5_lam10_eps1
#SBATCH --account=rrg-mijungp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=4:00:00
#SBATCH --array=0-2
#SBATCH --output=output/heldout_fam_lr1e5_lam10_eps1_%A_%a.out

mkdir -p output

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

# Load environment
module load StdEnv/2023 python/3.11

# Activate virtual environment
source "$SCRATCH/venv/nanogcg/bin/activate"

# Use node-local storage for Hugging Face
export TRANSFORMERS_CACHE=$SLURM_TMPDIR/hf_cache
export HF_HOME=$SLURM_TMPDIR/hf_home
mkdir -p "$TRANSFORMERS_CACHE"
mkdir -p "$HF_HOME"

# Select which family is held out based on the array task id.
# Task 0: gcg held out; Task 1: autodan held out; Task 2: pair held out.
FAMILIES=(gcg autodan pair)
UNSEEN_FAMILY="${FAMILIES[$SLURM_ARRAY_TASK_ID]}"

# Hyperparameters
LR=1e-5
LAMBDA=10
EPSILON=1
LORA_RANK=8
TOTAL_EPOCHS=1
START_EPOCH=1

IDX="heldout_${UNSEEN_FAMILY}_lr1e5_lam10_eps1"
FINETUNED_BASE="$SCRATCH/${IDX}_finetuned_llm"
mkdir -p "$(dirname "$FINETUNED_BASE")"

# Training data: train.csv + validation.csv combined into a single CSV.
TRAINING_DATA="$SCRATCH/dp-llm-experiments/official_data/train_plus_validation.csv"

# Run training (unseen-family / held-out family mode)
python "$SCRATCH/dp-llm-experiments/train/train.py" \
    --eval-mode "unseen-family" \
    --unseen-family "$UNSEEN_FAMILY" \
    --finetuned-llm-path "$FINETUNED_BASE" \
    --training-data "$TRAINING_DATA" \
    --lr "$LR" \
    --lambda-val "$LAMBDA" \
    --epsilon "$EPSILON" \
    --lora-rank "$LORA_RANK" \
    --total-epochs "$TOTAL_EPOCHS" \
    --start-epoch "$START_EPOCH"

echo "Held-out family: ${UNSEEN_FAMILY}"
echo "Final adapter: ${FINETUNED_BASE}_epoch${TOTAL_EPOCHS}"
