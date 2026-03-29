#!/bin/bash
#SBATCH --job-name=pair_eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=logs/pair_eval_%j.out

module purge
module load StdEnv/2023 python/3.11 cuda

source $SCRATCH/venv/pair/bin/activate

cd $SCRATCH/pair

mkdir -p logs results

python3 pair.py