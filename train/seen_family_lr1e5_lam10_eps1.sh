#!/bin/bash
#SBATCH --job-name=seen_fam_lr1e5_lam10_eps1
#SBATCH --account=rrg-mijungp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=4:00:00
#SBATCH --output=output/seen_fam_lr1e5_lam10_eps1_%j.out

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

# Hyperparameters
LR=1e-5
LAMBDA=10
EPSILON=1
LORA_RANK=8
TOTAL_EPOCHS=1
START_EPOCH=1

IDX="seen_fam_lr1e5_lam10_eps1"
FINETUNED_BASE="$SCRATCH/${IDX}_finetuned_llm"
mkdir -p "$(dirname "$FINETUNED_BASE")"

# Training data: train.csv + validation.csv combined into a single CSV.
TRAINING_DATA="$SCRATCH/dp-llm-experiments/official_data/train_plus_validation.csv"

# Run training (seen-family mode — all three perturbations used)
python "$SCRATCH/dp-llm-experiments/train/train.py" \
    --eval-mode "seen-family" \
    --finetuned-llm-path "$FINETUNED_BASE" \
    --training-data "$TRAINING_DATA" \
    --lr "$LR" \
    --lambda-val "$LAMBDA" \
    --epsilon "$EPSILON" \
    --lora-rank "$LORA_RANK" \
    --total-epochs "$TOTAL_EPOCHS" \
    --start-epoch "$START_EPOCH"

echo "Final adapter: ${FINETUNED_BASE}_epoch${TOTAL_EPOCHS}"
