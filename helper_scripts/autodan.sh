#!/bin/bash
#SBATCH --job-name=autodan
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=autodan_%j.out

module load python
module load cuda

source ~/venv/autodan/bin/activate

cd ~/AutoDAN

python autodan_hga_eval.py \
  --dataset_path /home/taegyoem/scratch/dp-llm-experiments/data/harmful_behaviors.csv