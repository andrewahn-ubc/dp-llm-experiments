#!/bin/bash
#SBATCH --job-name=redo_autodan_array_19
#SBATCH --array=0-24%5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=output/autodan/redo_autodan_testing_%A_%a.out

module load python cuda

source $SCRATCH/venv/autodan/bin/activate

IDX=$(printf "%02d" ${SLURM_ARRAY_TASK_ID})
DATA_PATH="/home/taegyoem/scratch/dp-llm-experiments/official_data/redo_autodan_test_19_part_${IDX}.csv"

echo "Running on file: $DATA_PATH"

cd $SCRATCH/AutoDAN

python autodan_hga_eval.py \
    --dataset_path "$DATA_PATH" \
    --save_suffix "testing_dataset_19_part_$IDX" \
    --batch_size 32
