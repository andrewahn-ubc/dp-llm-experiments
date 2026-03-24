#!/bin/bash
#SBATCH --job-name=gcg
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --array=0-59             
#SBATCH --output=output/gcg_%A_%a.out   # %A = array job ID, %a = task ID

# Load Python
module load StdEnv/2023 python/3.11

python - <<'PY'
import time
print("\n start time: " + time.time())
PY

IDX=$(printf "%02d" ${SLURM_ARRAY_TASK_ID})
DATA_PATH="/home/taegyoem/scratch/dp-llm-experiments/official_data/train_${IDX}.csv"

echo "Running on file: $DATA_PATH"

# Activate virtual environment 
# source ~/envs/llama/bin/activate
source $SCRATCH/venv/nanogcg/bin/activate

# Run gcg
python ~/scratch/dp-llm-experiments/helper_scripts/perturbation/gcg.py \
    --input_file "$DATA_PATH" \
    --save_suffix "train_dataset_$IDX"

python - <<'PY'
import time
print("\n end time: " + time.time())
PY