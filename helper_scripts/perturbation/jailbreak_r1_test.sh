#!/bin/bash
#SBATCH --job-name=jb_r1_test
#SBATCH --account=def-mijungp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --array=0-51
#SBATCH --output=output/jailbreak_r1_test_%A_%a.out

# Generate Jailbreak-R1 variants for combined_test_dataset goals (chunked).
#
# Prerequisites (once, on a login node with HF access):
#   huggingface-cli login
#   huggingface-cli download yukiyounai/Jailbreak-R1 --local-dir $SCRATCH/jailbreak_r1
#
# Prepare chunks (from repo root):
#   python helper_scripts/perturbation/prepare_jailbreak_r1_chunks.py \
#     --input official/combined_test_dataset.csv \
#     --out-dir $SCRATCH/dp-llm-experiments/official_data/jailbreak_r1_test \
#     --chunk-size 20
#   # then sync chunks if prepared locally, and set --array to match manifest
#
# Submit:
#   mkdir -p output
#   sbatch helper_scripts/perturbation/jailbreak_r1_test.sh

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$SCRATCH/dp-llm-experiments}"
cd "$REPO_ROOT"
mkdir -p output

module load StdEnv/2023 python/3.11
# Prefer nanogcg venv (has torch/transformers); override if needed.
# shellcheck disable=SC1090
source "${VENV_ACTIVATE:-$SCRATCH/venv/nanogcg/bin/activate}"

export TRANSFORMERS_CACHE="${SLURM_TMPDIR:-/tmp}/hf_cache"
export HF_HOME="${SLURM_TMPDIR:-/tmp}/hf_home"
mkdir -p "$TRANSFORMERS_CACHE" "$HF_HOME"

IDX=$(printf "%02d" "${SLURM_ARRAY_TASK_ID}")
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/official_data/jailbreak_r1_test}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/official_data/jailbreak_r1_out}"
MODEL_PATH="${MODEL_PATH:-$SCRATCH/jailbreak_r1}"
INPUT_FILE="${DATA_ROOT}/chunk_${IDX}.csv"
OUTPUT_FILE="${OUT_ROOT}/chunk_${IDX}.csv"

mkdir -p "$OUT_ROOT"

if [[ ! -f "$INPUT_FILE" ]]; then
  echo "ERROR: missing $INPUT_FILE" >&2
  echo "Run prepare_jailbreak_r1_chunks.py first." >&2
  exit 2
fi
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "ERROR: missing Jailbreak-R1 snapshot at $MODEL_PATH" >&2
  echo "Download: huggingface-cli download yukiyounai/Jailbreak-R1 --local-dir \$SCRATCH/jailbreak_r1" >&2
  exit 2
fi

echo "input=$INPUT_FILE"
echo "output=$OUTPUT_FILE"
echo "model=$MODEL_PATH"

python "$REPO_ROOT/helper_scripts/perturbation/jailbreak_r1.py" \
  --input_file "$INPUT_FILE" \
  --model_path "$MODEL_PATH" \
  --output_file "$OUTPUT_FILE" \
  --temperature 1.0 \
  --top_p 0.95 \
  --max_new_tokens 512 \
  --dtype bfloat16 \
  --seed 0

echo "Done chunk $IDX"
