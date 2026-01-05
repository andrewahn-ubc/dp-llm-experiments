#!/bin/bash
#SBATCH --job-name=gcg
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:60:00
#SBATCH --output=gcg_%j.out

# Load Python
module load StdEnv/2023 python/3.12.4

# Activate virtual environment 
source ~/envs/llama/bin/activate

# Run inference
python ~/scratch/dp-llm-experiments/gcg.py