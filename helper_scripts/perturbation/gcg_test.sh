#!/bin/bash
#SBATCH --job-name=gcg_adapt_l2
#SBATCH --account=def-mijungp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=2:30:00
#SBATCH --array=0-49
#SBATCH --output=output/gcg_adapt_l2_%A_%a.out

# GCG against merged adaptive-attack target on adaptive_test (50×5 prompts).
#
# Default merged model:
#   $SCRATCH/merged_adaptive_l2_lam0.1_eps-0.5_ep5
#
#   rsync -av official/splits/adaptive_test/ \
#     $SCRATCH/dp-llm-experiments/official_data/adaptive_test/
#   MODEL_PATH=... SAVE_TAG=... sbatch helper_scripts/perturbation/gcg_test.sh

module load StdEnv/2023 python/3.11

python - <<'PY'
import time
print("\n start time: " + str(time.time()))
PY

IDX=$(printf "%02d" "$SLURM_ARRAY_TASK_ID")

REPO_ROOT="${REPO_ROOT:-$SCRATCH/dp-llm-experiments}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/official_data/adaptive_test}"
DATA_PATH="${DATA_PATH:-${DATA_ROOT}/test_${IDX}.csv}"
MODEL_PATH="${MODEL_PATH:-$SCRATCH/merged_adaptive_l2_lam0.1_eps-0.5_ep5}"
SAVE_TAG="${SAVE_TAG:-l2_adapt_lam0.1_eps-0.5_ep5}"

echo "Running on file: $DATA_PATH"
echo "Model: $MODEL_PATH"

source "$SCRATCH/venv/nanogcg/bin/activate"

python "${REPO_ROOT}/helper_scripts/perturbation/gcg.py" \
    --input_file "$DATA_PATH" \
    --model_path "$MODEL_PATH" \
    --save_suffix "${SAVE_TAG}_${IDX}"

python - <<'PY'
import time
print("\n end time: " + str(time.time()))
PY
