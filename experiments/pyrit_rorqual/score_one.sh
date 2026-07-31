#!/bin/bash
#SBATCH --job-name=pyrit_score
#SBATCH --account=def-mijungp
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=output/pyrit_score_%x_%j.out

# Merge PyRIT chunks for one TARGET_TAG and score with HarmBench.
#
#   for tag in base mixat door dcl_lam3_eps1; do
#     TARGET_TAG=$tag sbatch experiments/pyrit_rorqual/score_one.sh
#   done

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$SCRATCH/dp-llm-experiments}"
cd "$REPO_ROOT"
mkdir -p output

TARGET_TAG="${TARGET_TAG:?Set TARGET_TAG=base|mixat|door|dcl_lam3_eps1}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/official_data/pyrit_out}"
CHUNKS_DIR="${OUT_ROOT}/${TARGET_TAG}"
MERGED="${OUT_ROOT}/${TARGET_TAG}_harmful.csv"

module load StdEnv/2023 python/3.11 cuda/12.2 || module load StdEnv/2023 python/3.11
# shellcheck disable=SC1090
source "${VENV_ACTIVATE:-$SCRATCH/venv/pyrit-rorqual/bin/activate}"

export TRANSFORMERS_CACHE="${SLURM_TMPDIR:-/tmp}/hf_cache"
export HF_HOME="${SLURM_TMPDIR:-/tmp}/hf_home"
mkdir -p "$TRANSFORMERS_CACHE" "$HF_HOME"

python "${REPO_ROOT}/experiments/pyrit_rorqual/merge_and_score.py" \
  --chunks-dir "$CHUNKS_DIR" \
  --output-csv "$MERGED" \
  --harmbench-path "${HARMBENCH_PATH:-$SCRATCH/harmbench_mistral_val_cls}"
