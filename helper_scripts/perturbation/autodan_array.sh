#!/bin/bash
#SBATCH --job-name=autodan_array
#SBATCH --array=0-59
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=09:00:00
#SBATCH --output=output/autodan/autodan_training_%A_%a.out

module load python cuda

source $SCRATCH/venv/autodan/bin/activate

IDX=$(printf "%02d" ${SLURM_ARRAY_TASK_ID})
DATA_PATH="/home/taegyoem/scratch/dp-llm-experiments/official_data/train_${IDX}.csv"

echo "Running on file: $DATA_PATH"

cd $SCRATCH/AutoDAN

python autodan_hga_eval.py \
    --input_file "$DATA_PATH" \
    --save_suffix "training_dataset_$IDX"
