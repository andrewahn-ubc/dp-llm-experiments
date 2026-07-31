#!/bin/bash
#SBATCH --job-name=autodan_adapt_l2
#SBATCH --account=def-mijungp
#SBATCH --array=0-49
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=2:30:00
#SBATCH --output=output/autodan/autodan_adapt_l2_%A_%a.out

# AutoDAN on adaptive_test against merged adaptive-attack target.
# Default:
#   MODEL_PATH=$SCRATCH/merged_adaptive_l2_lam0.1_eps-0.5_ep5
#
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
MODEL_PATH="${MODEL_PATH:-$SCRATCH/merged_adaptive_l2_lam0.1_eps-0.5_ep5}"
SAVE_TAG="${SAVE_TAG:-autodan_l2_adapt_lam0.1_eps-0.5_ep5}"

echo "Running on file: $DATA_PATH"
echo "Model: $MODEL_PATH"

cd "$SCRATCH/AutoDAN"

python autodan_hga_eval.py \
    --dataset_path "$DATA_PATH" \
    --model_path "$MODEL_PATH" \
    --tokenizer_path "$SCRATCH/llama2_7b_chat_hf" \
    --save_suffix "${SAVE_TAG}_$IDX" \
    --batch_size 8
