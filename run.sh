#!/bin/bash
#SBATCH --job-name=train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=output/train.out   

# Load Python
module load StdEnv/2023 python/3.11

# Activate virtual environment 
source $SCRATCH/venv/nanogcg/bin/activate

# Run training script
python ~/scratch/dp-llm-experiments/train.py