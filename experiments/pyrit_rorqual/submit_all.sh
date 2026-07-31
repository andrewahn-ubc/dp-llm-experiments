#!/bin/bash
# One-shot prep + submit helpers for PyRIT on Rorqual (run from repo root).
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$SCRATCH/dp-llm-experiments}"
cd "$REPO_ROOT"

echo "=== 1) chunks ==="
python helper_scripts/perturbation/prepare_jailbreak_r1_chunks.py \
  --input official_data/combined_test_dataset.csv \
  --out-dir official_data/pyrit_test \
  --chunk-size "${CHUNK_SIZE:-20}"
cat official_data/pyrit_test/manifest.txt

echo "=== 2) submit attack array ==="
mkdir -p output official_data/pyrit_out
EPOCH="${EPOCH:-2}" sbatch experiments/pyrit_rorqual/run_array.sh

echo "After the array finishes, score with:"
echo "  for tag in base mixat door dcl_lam3_eps1 delman; do TARGET_TAG=\$tag sbatch experiments/pyrit_rorqual/score_one.sh; done"
