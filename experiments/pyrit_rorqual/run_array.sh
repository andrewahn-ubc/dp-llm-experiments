#!/bin/bash
#SBATCH --job-name=pyrit_l3
#SBATCH --account=def-mijungp
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --array=0-259
#SBATCH --output=output/pyrit_l3_%A_%a.out

# PyRIT RedTeamingAttack on the harmful test set against 5 targets.
#
# Layout (default CHUNK_SIZE=20 → 52 chunks; 5 models → 260 tasks = array 0-259):
#   task = model_idx * N_CHUNKS + chunk_idx
#   models: 0=base  1=mixat  2=door  3=dcl_lam3_eps1  4=delman
#
# Attacker: $SCRATCH/qwen3-30b-a3b-instruct-2507  (cuda:0)
# Target:   Llama-3 / DELMAN-edited Llama-3.1     (cuda:1)
#
# DELMAN (full HF): $SCRATCH/delman_llama31_8b_instruct
#   produced from $SCRATCH/llama31_8b_instruct via experiments/delman/run_edit_llama31.sh
#
# Prep once on a login node:
#   bash experiments/pyrit_rorqual/setup_env.sh
#   python helper_scripts/perturbation/prepare_jailbreak_r1_chunks.py \
#     --input official_data/combined_test_dataset.csv \
#     --out-dir official_data/pyrit_test --chunk-size 20
#
# Submit:
#   EPOCH=2 sbatch experiments/pyrit_rorqual/run_array.sh
#
# After array finishes, score each model:
#   for tag in base mixat door dcl_lam3_eps1 delman; do
#     TARGET_TAG=$tag sbatch experiments/pyrit_rorqual/score_one.sh
#   done

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$SCRATCH/dp-llm-experiments}"
cd "$REPO_ROOT"
mkdir -p output

module load StdEnv/2023 python/3.11 cuda/12.2 || module load StdEnv/2023 python/3.11
# shellcheck disable=SC1090
source "${VENV_ACTIVATE:-$SCRATCH/venv/pyrit-rorqual/bin/activate}"

export TRANSFORMERS_CACHE="${SLURM_TMPDIR:-/tmp}/hf_cache"
export HF_HOME="${SLURM_TMPDIR:-/tmp}/hf_home"
export PYRIT_HOME="${SLURM_TMPDIR:-/tmp}/pyrit_home"
mkdir -p "$TRANSFORMERS_CACHE" "$HF_HOME" "$PYRIT_HOME"
export HOME="$PYRIT_HOME"

CHUNK_SIZE="${CHUNK_SIZE:-20}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/official_data/pyrit_test}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/official_data/pyrit_out}"
ATTACKER_PATH="${ATTACKER_PATH:-$SCRATCH/qwen3-30b-a3b-instruct-2507}"
BASE_L3="${BASE_L3:-$SCRATCH/llama_3_8b_instruct}"
MIXAT_PATH="${MIXAT_PATH:-$SCRATCH/mixat}"
DOOR_PATH="${DOOR_PATH:-$SCRATCH/door}"
DELMAN_PATH="${DELMAN_PATH:-$SCRATCH/delman_llama31_8b_instruct}"
CK_ROOT="${CHECKPOINT_ROOT:-$SCRATCH/dp-llm-sweep}"
EPOCH="${EPOCH:-2}"
MAX_TURNS="${MAX_TURNS:-5}"

# Infer chunk count from prepared data (fallback 52 for 1022 goals / 20).
N_CHUNKS="${N_CHUNKS:-}"
if [[ -z "$N_CHUNKS" ]]; then
  if [[ -f "${DATA_ROOT}/manifest.txt" ]]; then
    # manifest ends with array=0-(N-1)
    N_CHUNKS=$(python - <<PY
from pathlib import Path
t = Path("${DATA_ROOT}/manifest.txt").read_text()
import re
m = re.search(r"array=0-(\d+)", t)
print(int(m.group(1)) + 1 if m else 52)
PY
)
  else
    N_CHUNKS=52
  fi
fi

N_MODELS=5
TASK_ID="${SLURM_ARRAY_TASK_ID:?}"
MODEL_IDX=$((TASK_ID / N_CHUNKS))
CHUNK_IDX=$((TASK_ID % N_CHUNKS))

if (( MODEL_IDX >= N_MODELS )); then
  echo "TASK_ID=$TASK_ID out of range (models=$N_MODELS chunks=$N_CHUNKS)" >&2
  exit 2
fi

TAGS=(base mixat door dcl_lam3_eps1 delman)
TAG="${TAGS[$MODEL_IDX]}"
CHUNK=$(printf "chunk_%02d.csv" "$CHUNK_IDX")
# Support unpadded names if prepare script used wider indices.
if [[ ! -f "${DATA_ROOT}/${CHUNK}" ]]; then
  CHUNK="chunk_${CHUNK_IDX}.csv"
fi
INPUT_CSV="${DATA_ROOT}/${CHUNK}"
OUTPUT_CSV="${OUT_ROOT}/${TAG}/${CHUNK}"

TARGET_BASE="$BASE_L3"
TARGET_ADAPTER=""
case "$TAG" in
  base) ;;
  mixat) TARGET_BASE="$MIXAT_PATH" ;;
  door) TARGET_BASE="$DOOR_PATH" ;;
  dcl_lam3_eps1)
    TARGET_ADAPTER="${CK_ROOT}/l3_run_lr2e-05_lam3_eps1_finetuned_llm_epoch${EPOCH}"
    ;;
  delman) TARGET_BASE="$DELMAN_PATH" ;;
esac

echo "=== PyRIT task $TASK_ID ==="
echo "tag=$TAG model_idx=$MODEL_IDX chunk=$CHUNK"
echo "attacker=$ATTACKER_PATH"
echo "target_base=$TARGET_BASE"
echo "target_adapter=${TARGET_ADAPTER:-none}"
echo "input=$INPUT_CSV"
echo "output=$OUTPUT_CSV"
echo "gpus=$(nvidia-smi -L 2>/dev/null | wc -l)"

for p in "$ATTACKER_PATH" "$TARGET_BASE" "$INPUT_CSV"; do
  if [[ ! -e "$p" ]]; then
    echo "ERROR: missing path: $p" >&2
    exit 2
  fi
done
if [[ -n "$TARGET_ADAPTER" && ! -d "$TARGET_ADAPTER" ]]; then
  echo "ERROR: DCL adapter missing: $TARGET_ADAPTER" >&2
  echo "Copy from Narval or set CHECKPOINT_ROOT / EPOCH." >&2
  exit 2
fi

mkdir -p "$(dirname "$OUTPUT_CSV")"

ADAPTER_ARGS=()
if [[ -n "$TARGET_ADAPTER" ]]; then
  ADAPTER_ARGS=(--target-adapter "$TARGET_ADAPTER")
fi

python "${REPO_ROOT}/experiments/pyrit_rorqual/pyrit_attack.py" \
  --input-csv "$INPUT_CSV" \
  --output-csv "$OUTPUT_CSV" \
  --attacker-path "$ATTACKER_PATH" \
  --target-tag "$TAG" \
  --target-base "$TARGET_BASE" \
  "${ADAPTER_ARGS[@]}" \
  --max-turns "$MAX_TURNS"
