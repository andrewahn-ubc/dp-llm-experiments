#!/bin/bash
#SBATCH --job-name=pair_test_rest
#SBATCH --account=def-mijungp
#SBATCH --array=0-154
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=4:00:00
#SBATCH --output=logs/pair_test_rest_%A_%a.out

# PAIR on adaptive_test_remainder chunks (already on Rorqual):
#   $SCRATCH/official_data/adaptive_test_remainder/   (or …/links/scratch/…)
#
# Before sbatch, edit $SCRATCH/pair/pair.py so that:
#   CSV = $SCRATCH/official_data/adaptive_test_remainder/<chunk matching SLURM_ARRAY_TASK_ID>
#   --local-llama-path = your merged target (same as adaptive PAIR run)
#
# Prefer reading:
#   os.environ["PAIR_DATASET_PATH"]
#
#   cd $SCRATCH/dp-llm-experiments   # or wherever you submit from
#   mkdir -p logs
#   # set array to match #chunks-1, e.g. if chunk_00..chunk_154:
#   sbatch helper_scripts/perturbation/pair_test_rest.sh
#
#   DATA_ROOT=$SCRATCH/official_data/adaptive_test_remainder \
#   MODEL_PATH=$SCRATCH/merged_adaptive_l2_lam0.1_eps-0.5_ep5 \
#   sbatch --array=0-154 helper_scripts/perturbation/pair_test_rest.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$SCRATCH/dp-llm-experiments}"
mkdir -p logs

module purge
module load StdEnv/2023 python/3.11 cuda

source "$SCRATCH/venv/pair/bin/activate"

IDX=$(printf "%02d" "${SLURM_ARRAY_TASK_ID}")
# Rorqual: /home/taegyoem/links/scratch → $SCRATCH
DATA_ROOT="${DATA_ROOT:-$SCRATCH/official_data/adaptive_test_remainder}"
MODEL_PATH="${MODEL_PATH:-$SCRATCH/merged_adaptive_l2_lam0.1_eps-0.5_ep5}"

# Accept chunk_XX.csv or test_XX.csv
if [[ -f "${DATA_ROOT}/chunk_${IDX}.csv" ]]; then
  DATA_PATH="${DATA_ROOT}/chunk_${IDX}.csv"
elif [[ -f "${DATA_ROOT}/test_${IDX}.csv" ]]; then
  DATA_PATH="${DATA_ROOT}/test_${IDX}.csv"
else
  echo "ERROR: no chunk_${IDX}.csv or test_${IDX}.csv under $DATA_ROOT" >&2
  ls "$DATA_ROOT" | head -20 >&2
  exit 2
fi

export PAIR_DATASET_PATH="$DATA_PATH"
export PAIR_MODEL_PATH="$MODEL_PATH"
export PAIR_SAVE_SUFFIX="${SAVE_TAG:-pair_test_rest}_${IDX}"

echo "dataset=$DATA_PATH"
echo "model=$MODEL_PATH (set --local-llama-path in pair.py to this)"
echo "save_suffix=$PAIR_SAVE_SUFFIX"

cd "$SCRATCH/pair"
python3 "$SCRATCH/pair/pair.py"
