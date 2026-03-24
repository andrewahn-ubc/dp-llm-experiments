#!/bin/bash
#SBATCH --job-name=autodan_array
#SBATCH --array=0-21
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=09:00:00
#SBATCH --output=output/autodan/autodan_testing_%A_%a.out

module load python cuda

source $SCRATCH/venv/autodan/bin/activate

IDX=$(printf "%02d" ${SLURM_ARRAY_TASK_ID})
DATA_PATH="/home/taegyoem/scratch/dp-llm-experiments/official_data/test_${IDX}.csv"

echo "Running on file: $DATA_PATH"

cd $SCRATCH/AutoDAN

python autodan_hga_eval.py \
    --dataset_path "$DATA_PATH" \
    --save_suffix "testing_dataset_$IDX"
