#!/bin/bash
#SBATCH --job-name=gcg_l2_merged_adaptive
#SBATCH --account=def-mijungp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=2:30:00
#SBATCH --array=0-49
#SBATCH --output=output/gcg_l2_merged_adaptive_%A_%a.out

# GCG against merged Llama-2 LoRA on the adaptive test subset
# (100 advbench + 100 harmbench + 50 jailbreakbench, 5 prompts/chunk).
#
# Chunks: official/splits/adaptive_test/test_00.csv … test_49.csv
# Copy to Narval first, e.g.:
#   rsync -av official/splits/adaptive_test/ \
#     $SCRATCH/dp-llm-experiments/official_data/adaptive_test/
#
# Override: MODEL_PATH=/path DATA_ROOT=/path sbatch ...

module load StdEnv/2023 python/3.11

python - <<'PY'
import time
print("\n start time: " + str(time.time()))
PY

IDX=$(printf "%02d" "$SLURM_ARRAY_TASK_ID")

REPO_ROOT="${REPO_ROOT:-$SCRATCH/dp-llm-experiments}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/official_data/adaptive_test}"
DATA_PATH="${DATA_PATH:-${DATA_ROOT}/test_${IDX}.csv}"
MODEL_PATH="${MODEL_PATH:-$SCRATCH/merged_run_lr2e-05_lam1_eps0.5_ep5}"

echo "Running on file: $DATA_PATH"
echo "Model: $MODEL_PATH"

source "$SCRATCH/venv/nanogcg/bin/activate"

python "${REPO_ROOT}/helper_scripts/perturbation/gcg.py" \
    --input_file "$DATA_PATH" \
    --model_path "$MODEL_PATH" \
    --save_suffix "l2_merged_adaptive_${IDX}"

python - <<'PY'
import time
print("\n end time: " + str(time.time()))
PY
