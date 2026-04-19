#!/bin/bash
#SBATCH --job-name=train
#SBATCH --account=rrg-mijungp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=3:00:00
#SBATCH --output=output/lr5_train_epoch_5_%j.out

mkdir -p output

set -euo pipefail

# Load environment
module load StdEnv/2023 python/3.11

# Activate virtual environment
source $SCRATCH/venv/nanogcg/bin/activate

# Use node-local storage for Hugging Face
export TRANSFORMERS_CACHE=$SLURM_TMPDIR/hf_cache
export HF_HOME=$SLURM_TMPDIR/hf_home

mkdir -p $TRANSFORMERS_CACHE
mkdir -p $HF_HOME

IDX="lr5"

# Run training (extra dataset paths for parity with previous epoch3.sh; eval in train.py is commented out)
python $SCRATCH/dp-llm-experiments/train/train.py \
    --eval-mode "seen-family" \
    --finetuned-llm-path "$SCRATCH/${IDX}_finetuned_llm" \
    --training-data "$SCRATCH/dp-llm-experiments/official_data/train.csv" \
    --validation-data "$SCRATCH/dp-llm-experiments/official_data/validation.csv" \
    --benign-validation-data "$SCRATCH/dp-llm-experiments/official_data/frr_validation.csv" \
    --harmful-output-file "$SCRATCH/dp-llm-experiments/official_data/${IDX}_val_output" \
    --benign-output-file "$SCRATCH/dp-llm-experiments/official_data/${IDX}_frr_output" \
    --lr 5e-5 \
    --lambda-val 1.0 \
    --epsilon 0.0 \
    --lora-rank 8 \
    --total-epochs 5 \
    --resume-from "$SCRATCH/${IDX}_finetuned_llm_epoch4" \
    --start-epoch 5
