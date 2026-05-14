#!/bin/bash
#SBATCH --job-name=gcg_train
#SBATCH --account=rrg-mijungp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=14:00:00
#SBATCH --array=0-299             
#SBATCH --output=output/gcg_l3_train_%A_%a.out   # %A = array job ID, %a = task ID

# Load Python
module load StdEnv/2023 python/3.11

python - <<'PY'
import time
print("\n start time: " + str(time.time()))
PY

IDX=$(printf "%02d" "$SLURM_ARRAY_TASK_ID")

DATA_PATH="/home/taegyoem/scratch/dp-llm-experiments/official_data/train_${IDX}.csv"
MODEL_PATH="$SCRATCH/llama_3_8b_instruct"

echo "Running on file: $DATA_PATH"
echo "Model: $MODEL_PATH"

# Activate virtual environment 
# source ~/envs/llama/bin/activate
source $SCRATCH/venv/nanogcg/bin/activate

# Run gcg
python ~/scratch/dp-llm-experiments/helper_scripts/perturbation/gcg.py \
    --input_file "$DATA_PATH" \
    --model_path "$MODEL_PATH" \
    --save_suffix "train_dataset_$IDX"

python - <<'PY'
import time
print("\n end time: " + str(time.time()))
PY
