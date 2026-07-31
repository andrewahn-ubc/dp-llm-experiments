#!/bin/bash
# Prep 500-goal stratified subset + submit PyRIT attack array.
#
# Defaults:
#   225 advbench + 225 harmbench + 50 jbb
#   MAX_TURNS=3, CHUNK_SIZE=25 → 20 chunks × 4 models = 80 jobs, wall 3:30
#
#   bash experiments/pyrit_rorqual/submit_all.sh
#
# Full 1022-goal test:
#   FULL_TEST=1 bash experiments/pyrit_rorqual/submit_all.sh

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$SCRATCH/dp-llm-experiments}"
cd "$REPO_ROOT"

N_MODELS=4
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/official_data/pyrit_test}"
SUBSET_CSV="${SUBSET_CSV:-${DATA_ROOT}/pyrit_subset_500.csv}"

if [[ "${FULL_TEST:-0}" == "1" ]]; then
  CHUNK_SIZE="${CHUNK_SIZE:-20}"
  MAX_TURNS="${MAX_TURNS:-3}"
  WALL_TIME="${WALL_TIME:-6:00:00}"
  WORK_CSV="${INPUT_CSV:-official_data/combined_test_dataset.csv}"
else
  CHUNK_SIZE="${CHUNK_SIZE:-25}"
  MAX_TURNS="${MAX_TURNS:-3}"
  WALL_TIME="${WALL_TIME:-3:30:00}"
  WORK_CSV="$SUBSET_CSV"
fi

mkdir -p output official_data/pyrit_out "$DATA_ROOT"

if [[ "${FULL_TEST:-0}" != "1" ]]; then
  echo "=== 0) stratified 500-goal subset (225/225/50) ==="
  # Prefer adaptive_test on cluster if synced under official_data/
  ADAPTIVE="${ADAPTIVE_DIR:-}"
  if [[ -z "$ADAPTIVE" ]]; then
    if [[ -d official_data/adaptive_test ]]; then
      ADAPTIVE=official_data/adaptive_test
    else
      ADAPTIVE=official/splits/adaptive_test
    fi
  fi
  COMBINED="${INPUT_CSV:-official_data/combined_test_dataset.csv}"
  if [[ ! -f "$COMBINED" && -f official/combined_test_dataset.csv ]]; then
    COMBINED=official/combined_test_dataset.csv
  fi
  python experiments/pyrit_rorqual/prepare_pyrit_subset.py \
    --adaptive-dir "$ADAPTIVE" \
    --combined "$COMBINED" \
    --out "$SUBSET_CSV"
fi

echo "=== 1) chunks (size=$CHUNK_SIZE) from $WORK_CSV ==="
python helper_scripts/perturbation/prepare_jailbreak_r1_chunks.py \
  --input "$WORK_CSV" \
  --out-dir "$DATA_ROOT" \
  --chunk-size "$CHUNK_SIZE"
cat "${DATA_ROOT}/manifest.txt"

N_CHUNKS=$(python - <<PY
from pathlib import Path
import re
t = Path("${DATA_ROOT}/manifest.txt").read_text()
m = re.search(r"array=0-(\d+)", t)
print(int(m.group(1)) + 1 if m else 0)
PY
)
if [[ "$N_CHUNKS" -lt 1 ]]; then
  echo "ERROR: could not parse N_CHUNKS from manifest" >&2
  exit 2
fi

ARRAY_MAX=$((N_MODELS * N_CHUNKS - 1))
N_JOBS=$((ARRAY_MAX + 1))

echo "=== 2) submit: ${N_JOBS} jobs (array=0-${ARRAY_MAX}), time=${WALL_TIME}, MAX_TURNS=${MAX_TURNS} ==="
EPOCH="${EPOCH:-2}" \
MAX_TURNS="$MAX_TURNS" \
N_CHUNKS="$N_CHUNKS" \
DATA_ROOT="$DATA_ROOT" \
sbatch \
  --array="0-${ARRAY_MAX}" \
  --time="$WALL_TIME" \
  experiments/pyrit_rorqual/run_array.sh

echo
echo "After the array finishes, score with:"
echo "  for tag in base mixat door dcl_lam3_eps1; do TARGET_TAG=\$tag sbatch experiments/pyrit_rorqual/score_one.sh; done"
