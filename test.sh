#!/bin/bash
#SBATCH --job-name=train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=output/eval.out

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

# Copy models to local SSD (critical for Narval)
cp -r $SCRATCH/llama2_7b_chat_hf $SLURM_TMPDIR/

export LLM_NAME=$SLURM_TMPDIR/llama2_7b_chat_hf

# Run training
python $SCRATCH/dp-llm-experiments/test.py
