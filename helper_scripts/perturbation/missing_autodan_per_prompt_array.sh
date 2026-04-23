#!/bin/bash
#SBATCH --job-name=missing_autodan_per_prompt
#SBATCH --account=rrg-mijungp
#SBATCH --array=0-36%8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=output/autodan/missing_autodan_per_prompt_%A_%a.out

module load python cuda

source $SCRATCH/venv/autodan/bin/activate

export NLTK_DATA=/scratch/taegyoem/nltk_data

IDX=$(printf "%02d" ${SLURM_ARRAY_TASK_ID})
DATA_PATH="/home/taegyoem/scratch/dp-llm-experiments/official_data/missing_autodan_${IDX}.csv"

echo "Running on file: $DATA_PATH"

cd $SCRATCH/AutoDAN

python autodan_hga_eval.py \
    --dataset_path "$DATA_PATH" \
    --save_suffix "missing_autodan_per_prompt_${IDX}" \
    --batch_size 8
