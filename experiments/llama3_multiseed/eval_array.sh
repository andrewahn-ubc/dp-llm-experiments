#!/bin/bash
#SBATCH --job-name=l3_ms_eval
#SBATCH --account=def-mijungp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --array=0-19
#SBATCH --output=output/l3_multiseed_eval_%A_%a.out
#
# Eval the 20 train configs (seeds 1–2 × 10 models). Same array indexing as train_array.sh.
#
#   TRAIN_ID=...   # from sbatch train_array.sh
#   sbatch --dependency=afterok:${TRAIN_ID} experiments/llama3_multiseed/eval_array.sh
#
# Outputs: $SCRATCH/dp-llm-eval/llama3_multiseed/seed{S}/points_{config_id}.csv
#
# Missing checkpoints fail the task (exit 2). Do not use --skip-missing unless debugging.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$SCRATCH/dp-llm-experiments}"
cd "$REPO_ROOT"
mkdir -p output
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

module load StdEnv/2023 python/3.11
# shellcheck disable=SC1090
source "${VENV_ACTIVATE:-$SCRATCH/venv/nanogcg/bin/activate}"

export TRANSFORMERS_CACHE="${SLURM_TMPDIR}/hf_cache"
export HF_HOME="${SLURM_TMPDIR}/hf_home"
mkdir -p "$TRANSFORMERS_CACHE" "$HF_HOME"

export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$SCRATCH/dp-llm-sweep/multiseed_l3}"
export EVAL_OUT_DIR="${EVAL_OUT_DIR:-$SCRATCH/dp-llm-eval/llama3_multiseed}"

# Preflight before the multi-hour eval.py call.
python - <<'PY'
import sys
from pathlib import Path

repo = Path.cwd()
ok = True
for rel in ("experiments/llama3_multiseed/eval_one.py", "eval/eval.py"):
    p = repo / rel
    if not p.is_file():
        print(f"ERROR: missing {p}", file=sys.stderr)
        ok = False
if not any((repo / r).is_file() for r in ("official_data/llama3_test.csv", "official/llama3_test.csv")):
    print("ERROR: llama3_test.csv not found under official_data/ or official/", file=sys.stderr)
    ok = False
if not any(
    (repo / r).is_file()
    for r in (
        "official_data/frr_text.csv",
        "official_data/frr_test.csv",
        "official/frr_text.csv",
        "official/frr_test.csv",
    )
):
    print("ERROR: frr_text/frr_test.csv not found under official_data/ or official/", file=sys.stderr)
    ok = False
sys.exit(0 if ok else 2)
PY

# shellcheck disable=SC2086
python experiments/llama3_multiseed/eval_one.py \
  --task-id "${SLURM_ARRAY_TASK_ID:?}" \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --out-dir "$EVAL_OUT_DIR" \
  --repo-root "$REPO_ROOT" \
  ${EXTRA_ARGS:-}
