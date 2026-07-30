#!/bin/bash
#SBATCH --job-name=autodan_l2_merged_adaptive
#SBATCH --account=def-mijungp
#SBATCH --array=0-49
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=2:30:00
#SBATCH --output=output/autodan/autodan_l2_merged_adaptive_%A_%a.out

# AutoDAN on adaptive_test (test_00 … test_49, 5 prompts each).
# Point the model path inside $SCRATCH/AutoDAN at:
#   $SCRATCH/merged_run_lr2e-05_lam1_eps0.5_ep5
#
# Copy first:
#   rsync -av official/splits/adaptive_test/ \
#     $SCRATCH/dp-llm-experiments/official_data/adaptive_test/

module load python cuda

# protobuf>=4 breaks sentencepiece's bundled *_pb2.py (AutoDAN tokenizer load).
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

source "$SCRATCH/venv/autodan/bin/activate"

IDX=$(printf "%02d" "$SLURM_ARRAY_TASK_ID")

REPO_ROOT="${REPO_ROOT:-$SCRATCH/dp-llm-experiments}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/official_data/adaptive_test}"
DATA_PATH="${DATA_PATH:-${DATA_ROOT}/test_${IDX}.csv}"

echo "Running on file: $DATA_PATH"
echo "Ensure AutoDAN model path = \$SCRATCH/merged_run_lr2e-05_lam1_eps0.5_ep5"

cd "$SCRATCH/AutoDAN"

python autodan_hga_eval.py \
    --dataset_path "$DATA_PATH" \
    --save_suffix "autodan_l2_merged_adaptive_$IDX" \
    --batch_size 8
