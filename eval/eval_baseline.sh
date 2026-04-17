#!/bin/bash
#SBATCH --job-name=lr1-eval
#SBATCH --account=rrg-mijungp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=6:00:00
#SBATCH --output=output/lr1_test%j.out

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

# Run evaluation
python $SCRATCH/dp-llm-experiments/eval/eval.py \
    --harmful-output-file "baseline_harmful_eval_0416_1924h" \
    --benign-output-file "baseline_benign_eval_0416_1924h" 
