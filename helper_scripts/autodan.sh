#!/bin/bash
#SBATCH --job-name=autodan
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=autodan.out

module load python
module load cuda

source $SCRATCH/venv/autodan/bin/activate

export NLTK_DATA=/scratch/taegyoem/nltk_data

cd $SCRATCH/AutoDAN

python autodan_hga_eval.py \
  --dataset_path /home/taegyoem/scratch/dp-llm-experiments/data/first_10.csv
