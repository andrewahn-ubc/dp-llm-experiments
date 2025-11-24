#!/bin/bash
#SBATCH --job-name=llama_infer
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=llama_infer_%j.out

# Load Python
module load StdEnv/2023 python/3.12.4

# Activate your virtual environment (recommended)
source ~/envs/llama/bin/activate

# Run inference
python ~/scratch/dp-llm-experiments/inference.py