#!/bin/bash
#SBATCH --job-name=pair_adapt_l2
#SBATCH --account=def-mijungp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=logs/pair_adapt_l2_%j.out

# PAIR against merged adaptive-attack target.
# The real driver is $SCRATCH/pair/pair.py (not the stub in this repo).
#
# Before sbatch, edit $SCRATCH/pair/pair.py so that:
#   1. CSV input = adaptive_test chunks (goal,target), e.g.
#      $SCRATCH/dp-llm-experiments/official_data/adaptive_test/test_XX.csv
#   2. main.py gets:
#      --local-llama-path $SCRATCH/merged_adaptive_l2_lam0.1_eps-0.5_ep5
#
# Then:
#   cd $SCRATCH/dp-llm-experiments
#   mkdir -p helper_scripts/perturbation/logs
#   sbatch helper_scripts/perturbation/pair.sh

module purge
module load StdEnv/2023 python/3.11 cuda

source "$SCRATCH/venv/pair/bin/activate"

cd "$SCRATCH/pair"

mkdir -p logs results

MERGED="${MODEL_PATH:-$SCRATCH/merged_adaptive_l2_lam0.1_eps-0.5_ep5}"
echo "PAIR target should be: $MERGED"
echo "(set via --local-llama-path inside \$SCRATCH/pair/pair.py)"

python3 "$SCRATCH/pair/pair.py"
