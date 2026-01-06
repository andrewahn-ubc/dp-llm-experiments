#!/bin/bash
#SBATCH --job-name=gcg
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=output/combine%a.out

# Load Python
module load StdEnv/2023 python/3.12.4

# Activate virtual environment 
source $SCRATCH/venvs/llama/bin/activate

# Combine files
python ~/scratch/dp-llm-experiments/combine_data.py