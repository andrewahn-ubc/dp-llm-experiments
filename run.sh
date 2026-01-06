#!/bin/bash
#SBATCH --job-name=gcg
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --array=0-49             # 50 tasks
#SBATCH --output=output/gcg_%A_%a.out   # %A = array job ID, %a = task ID

# Load Python
module load StdEnv/2023 python/3.12.4

# Activate virtual environment 
# source ~/envs/llama/bin/activate
source $SCRATCH/venvs/llama/bin/activate

# Run gcg
# python ~/scratch/dp-llm-experiments/gcg.py
python ~/scratch/dp-llm-experiments/combine_data.py