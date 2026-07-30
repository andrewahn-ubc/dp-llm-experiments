#!/bin/bash
#SBATCH --job-name=pair_l2_merged
#SBATCH --account=def-mijungp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=logs/pair_l2_merged_%j.out

# PAIR against merged Llama-2 LoRA. Configure $SCRATCH/pair/pair.py to use:
#   --local-llama-path $SCRATCH/merged_run_lr2e-05_lam1_eps0.5_ep5
# (or equivalent target-model path).

module purge
module load StdEnv/2023 python/3.11 cuda

source "$SCRATCH/venv/pair/bin/activate"

cd "$SCRATCH/pair"

mkdir -p logs results

echo "Ensure PAIR local llama path = \$SCRATCH/merged_run_lr2e-05_lam1_eps0.5_ep5"

python3 "$SCRATCH/pair/pair.py"
