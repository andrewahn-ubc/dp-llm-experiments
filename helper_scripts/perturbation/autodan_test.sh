#!/bin/bash
#SBATCH --job-name=autodan_l2_merged_test
#SBATCH --account=def-mijungp
#SBATCH --array=0-14
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=output/autodan/autodan_l2_merged_test_%A_%a.out

# AutoDAN against merged Llama-2 LoRA. Point the model path inside
# $SCRATCH/AutoDAN at:
#   $SCRATCH/merged_run_lr2e-05_lam1_eps0.5_ep5
# (this wrapper does not pass --model_path; the AutoDAN driver hardcodes it.)
#
# Array tasks map to official/splits/test/test_100.csv … test_114.csv (5 rows each).

module load python cuda

source "$SCRATCH/venv/autodan/bin/activate"

IDX=$(printf "%03d" $((SLURM_ARRAY_TASK_ID + 100)))
DATA_PATH="${DATA_PATH:-/home/taegyoem/scratch/dp-llm-experiments/official_data/test_${IDX}.csv}"

echo "Running on file: $DATA_PATH"
echo "Ensure AutoDAN model path = \$SCRATCH/merged_run_lr2e-05_lam1_eps0.5_ep5"

cd "$SCRATCH/AutoDAN"

python autodan_hga_eval.py \
    --dataset_path "$DATA_PATH" \
    --save_suffix "autodan_l2_merged_test_$IDX" \
    --batch_size 8
