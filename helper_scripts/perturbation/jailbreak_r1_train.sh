#!/bin/bash
#SBATCH --job-name=jb_r1_train
#SBATCH --account=def-mijungp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --array=0-149
#SBATCH --output=output/jailbreak_r1_train_%A_%a.out

# Jailbreak-R1 variants for train(+val) goals.
# Default array 0-149 matches chunk-size 20 on train_plus_validation (~2996 rows).
# If your manifest differs, override: sbatch --array=0-N ...
#
# Prepare (Narval):
#   python helper_scripts/perturbation/prepare_jailbreak_r1_chunks.py \
#     --input official_data/train_plus_validation.csv \
#     --out-dir $SCRATCH/dp-llm-experiments/official_data/jailbreak_r1_train \
#     --chunk-size 20
#   cat official_data/jailbreak_r1_train/manifest.txt   # confirm array=
#
# Submit:
#   sbatch helper_scripts/perturbation/jailbreak_r1_train.sh
#   # or train.csv only:
#   DATA_ROOT=.../jailbreak_r1_train_only OUT_ROOT=.../jailbreak_r1_train_only_out \
#     sbatch --array=0-124 helper_scripts/perturbation/jailbreak_r1_train.sh

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$SCRATCH/dp-llm-experiments}"
cd "$REPO_ROOT"
mkdir -p output

module load StdEnv/2023 python/3.11
# shellcheck disable=SC1090
source "${VENV_ACTIVATE:-$SCRATCH/venv/nanogcg/bin/activate}"

export TRANSFORMERS_CACHE="${SLURM_TMPDIR:-/tmp}/hf_cache"
export HF_HOME="${SLURM_TMPDIR:-/tmp}/hf_home"
mkdir -p "$TRANSFORMERS_CACHE" "$HF_HOME"

IDX=$(printf "%02d" "${SLURM_ARRAY_TASK_ID}")
# Support >99 chunks (chunk_100.csv …) when array index ≥ 100.
if [[ "${SLURM_ARRAY_TASK_ID}" -ge 100 ]]; then
  IDX=$(printf "%03d" "${SLURM_ARRAY_TASK_ID}")
  # Prefer zero-padded width from prepare script: always try %02d then %03d.
fi

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/official_data/jailbreak_r1_train}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/official_data/jailbreak_r1_train_out}"
MODEL_PATH="${MODEL_PATH:-$SCRATCH/jailbreak_r1}"

# prepare_jailbreak_r1_chunks.py always writes chunk_00, chunk_01, … with 2-digit
# padding until 99, then chunk_100 (no forced width). Resolve flexibly:
INPUT_FILE=""
for cand in \
  "${DATA_ROOT}/chunk_$(printf '%02d' "${SLURM_ARRAY_TASK_ID}").csv" \
  "${DATA_ROOT}/chunk_$(printf '%03d' "${SLURM_ARRAY_TASK_ID}").csv" \
  "${DATA_ROOT}/chunk_${SLURM_ARRAY_TASK_ID}.csv"; do
  if [[ -f "$cand" ]]; then
    INPUT_FILE="$cand"
    break
  fi
done
OUTPUT_FILE="${OUT_ROOT}/$(basename "${INPUT_FILE:-chunk_${SLURM_ARRAY_TASK_ID}.csv}")"

mkdir -p "$OUT_ROOT"

if [[ -z "$INPUT_FILE" || ! -f "$INPUT_FILE" ]]; then
  echo "ERROR: missing chunk for task ${SLURM_ARRAY_TASK_ID} under $DATA_ROOT" >&2
  ls -1 "$DATA_ROOT"/chunk_*.csv 2>/dev/null | head -5 || true
  exit 2
fi
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "ERROR: missing Jailbreak-R1 at $MODEL_PATH" >&2
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

echo "Done train chunk task ${SLURM_ARRAY_TASK_ID}"
