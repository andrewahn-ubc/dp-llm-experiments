#!/bin/bash
#SBATCH --job-name=gcg_l2_merged_train
#SBATCH --account=def-mijungp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=14:00:00
#SBATCH --array=0-299
#SBATCH --output=output/gcg_l2_merged_train_%A_%a.out

# GCG against merged Llama-2 LoRA:
#   $SCRATCH/merged_run_lr2e-05_lam1_eps0.5_ep5
# Override with: MODEL_PATH=/path sbatch ...

module load StdEnv/2023 python/3.11

python - <<'PY'
import time
print("\n start time: " + str(time.time()))
PY

IDX=$(printf "%02d" "$SLURM_ARRAY_TASK_ID")

DATA_PATH="${DATA_PATH:-/home/taegyoem/scratch/dp-llm-experiments/official_data/train_${IDX}.csv}"
MODEL_PATH="${MODEL_PATH:-$SCRATCH/merged_run_lr2e-05_lam1_eps0.5_ep5}"
REPO_ROOT="${REPO_ROOT:-$SCRATCH/dp-llm-experiments}"

echo "Running on file: $DATA_PATH"
echo "Model: $MODEL_PATH"

source "$SCRATCH/venv/nanogcg/bin/activate"

python "${REPO_ROOT}/helper_scripts/perturbation/gcg.py" \
    --input_file "$DATA_PATH" \
    --model_path "$MODEL_PATH" \
    --save_suffix "l2_merged_train_${IDX}"

python - <<'PY'
import time
print("\n end time: " + str(time.time()))
PY
