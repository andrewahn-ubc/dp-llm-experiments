#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --account=rrg-mijungp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=output/inference.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p output

# Load environment
module load StdEnv/2023 python/3.11

# Activate virtual environment
source $SCRATCH/venv/nanogcg/bin/activate

# Use node-local storage for Hugging Face
export TRANSFORMERS_CACHE=$SLURM_TMPDIR/hf_cache
export HF_HOME=$SLURM_TMPDIR/hf_home

mkdir -p $TRANSFORMERS_CACHE
mkdir -p $HF_HOME

# Generate base-model Original Response completions and merge them into the relevant
# dataset CSVs. Override the model via MODEL_PROFILE (default: llama_3_8b_instruct).
MODEL_PROFILE="${MODEL_PROFILE:-llama_3_8b_instruct}"

python $SCRATCH/dp-llm-experiments/helper_scripts/inference/inference.py \
    --model-profile "${MODEL_PROFILE}"
