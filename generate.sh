#!/bin/bash
#SBATCH --job-name=generate
#SBATCH --gres=a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=output/generate%j.out

set -euo pipefail

# Load environment
module purge
module load StdEnv/2023 python/3.11 gcc arrow cuda

# Activate virtual environment
source $SCRATCH/venv/wizard/bin/activate

# Use node-local storage for Hugging Face
export TRANSFORMERS_CACHE=$SLURM_TMPDIR/hf_cache
export HF_HOME=$SLURM_TMPDIR/hf_home

mkdir -p $TRANSFORMERS_CACHE
mkdir -p $HF_HOME

# Copy models to local SSD (critical for Narval)
cp -r $SCRATCH/wizard $SLURM_TMPDIR/

export LLM_NAME=$SLURM_TMPDIR/wizard

cd $SCRATCH/dp-llm-experiments

python - <<'PY'
import torch
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

# Run training
python generate_harmful_prompts.py


