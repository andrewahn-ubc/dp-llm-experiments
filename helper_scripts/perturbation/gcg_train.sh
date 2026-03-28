#!/bin/bash
#SBATCH --job-name=gcg_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=14:00:00
#SBATCH --array=0-44             
#SBATCH --output=output/gcg_redo_train_%A_%a.out   # %A = array job ID, %a = task ID

# Load Python
module load StdEnv/2023 python/3.11

python - <<'PY'
import time
print("\n start time: " + str(time.time()))
PY

# Missing dataset indices
MISSING_INDICES=(3
31
43
44
47
49
53
66
71
79
162
166
167
169
170
173
175
177
179
180
181
185
186
187
188
191
192
194
195
196
197
198
199
200
201
202
203
204
205
206
208
212
252
265)

# Map array task ID -> actual missing index
REAL_IDX=${MISSING_INDICES[$SLURM_ARRAY_TASK_ID]}

IDX=$(printf "%02d" "$REAL_IDX")

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
print("\n end time: " + str(time.time()))
PY