#!/bin/bash
#SBATCH --job-name=gcg_adapt_rest
#SBATCH --account=def-mijungp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=2:30:00
#SBATCH --array=0-164
#SBATCH --output=output/gcg_adapt_rest_%A_%a.out

# GCG on adaptive_test_remainder (full test − adaptive subset).
# Default data: $SCRATCH/dp-llm-experiments/official_data/adaptive_test_remainder/
# Default model: $SCRATCH/merged_adaptive_l2_lam0.1_eps-0.5_ep5
#
#   mkdir -p output
#   N=$(ls $SCRATCH/dp-llm-experiments/official_data/adaptive_test_remainder/*.csv | wc -l)
#   sbatch --array=0-$((N-1)) helper_scripts/perturbation/gcg_test_rest.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$SCRATCH/dp-llm-experiments}"
mkdir -p output

module load StdEnv/2023 python/3.11

ID2=$(printf "%02d" "${SLURM_ARRAY_TASK_ID}")
ID3=$(printf "%03d" "${SLURM_ARRAY_TASK_ID}")
REPO_ROOT="${REPO_ROOT:-$SCRATCH/dp-llm-experiments}"
# Do not derive from REPO_ROOT — a wrong exported REPO_ROOT broke earlier runs.
DATA_ROOT="${DATA_ROOT:-$SCRATCH/dp-llm-experiments/official_data/adaptive_test_remainder}"
MODEL_PATH="${MODEL_PATH:-$SCRATCH/merged_adaptive_l2_lam0.1_eps-0.5_ep5}"
SAVE_TAG="${SAVE_TAG:-l2_adapt_rest_lam0.1_eps-0.5_ep5}"

echo "DATA_ROOT=$DATA_ROOT"
if [[ ! -d "$DATA_ROOT" ]]; then
  echo "ERROR: DATA_ROOT missing: $DATA_ROOT" >&2
  exit 2
fi

DATA_PATH=""
for cand in \
  "${DATA_ROOT}/test_${ID3}.csv" \
  "${DATA_ROOT}/test_${ID2}.csv" \
  "${DATA_ROOT}/chunk_${ID3}.csv" \
  "${DATA_ROOT}/chunk_${ID2}.csv"
do
  if [[ -f "$cand" ]]; then DATA_PATH="$cand"; break; fi
done
if [[ -z "$DATA_PATH" ]]; then
  echo "ERROR: no test/chunk csv for task ${SLURM_ARRAY_TASK_ID} in $DATA_ROOT" >&2
  ls "$DATA_ROOT" | head -20 >&2
  exit 2
fi
IDX="$ID2"

echo "Running on file: $DATA_PATH"
echo "Model: $MODEL_PATH"

source "$SCRATCH/venv/nanogcg/bin/activate"

python "${REPO_ROOT}/helper_scripts/perturbation/gcg.py" \
    --input_file "$DATA_PATH" \
    --model_path "$MODEL_PATH" \
    --save_suffix "${SAVE_TAG}_${IDX}"
