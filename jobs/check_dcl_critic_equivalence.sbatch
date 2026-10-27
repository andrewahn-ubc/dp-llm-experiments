#!/bin/bash
#SBATCH --job-name=dcl-critic-eq
#SBATCH --account=def-mijungp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
# Keep this at the submission root so a fresh pull needs no pre-created log dir.
#SBATCH --output=dcl_critic_equivalence_%j.out

# Run from the Narval repository root with:
#   sbatch jobs/check_dcl_critic_equivalence.sbatch
#
# Environment overrides:
#   PAIR=llama31|llama2|all (default: all), CASE_COUNT (default: 8),
#   OUT_DIR, REPO_ROOT, VENV_ACTIVATE, and each *_PATH below.
# Additional arguments after the script name are forwarded to the Python CLI.

set -euo pipefail

: "${SCRATCH:?SCRATCH must be set (this job is intended for Narval)}"

REPO_ROOT="${REPO_ROOT:-$SCRATCH/dp-llm-experiments}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$SCRATCH/venv/nanogcg/bin/activate}"

if [[ ! -d "$REPO_ROOT" ]]; then
  echo "ERROR: repository directory not found: $REPO_ROOT" >&2
  exit 2
fi
if [[ ! -f "$VENV_ACTIVATE" ]]; then
  echo "ERROR: Python environment activation script not found: $VENV_ACTIVATE" >&2
  exit 2
fi

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

module load StdEnv/2023 python/3.11
# shellcheck disable=SC1090
source "$VENV_ACTIVATE"

export TRANSFORMERS_CACHE="${SLURM_TMPDIR:-/tmp}/hf_cache"
export HF_HOME="${SLURM_TMPDIR:-/tmp}/hf_home"
mkdir -p "$TRANSFORMERS_CACHE" "$HF_HOME"

PAIR="${PAIR:-all}"
CASE_COUNT="${CASE_COUNT:-8}"
OUT_DIR="${OUT_DIR:-$SCRATCH/dcl_critic_equivalence/${SLURM_JOB_ID:-manual}}"

if [[ -z "${LLAMA31_STUDENT_PATH:-}" ]]; then
  # The attached plan names llama31_8b_instruct as its starting point, while
  # train/model_profiles.py is authoritative for the active DCL experiment and
  # names llama_3_8b_instruct. Prefer the configured path when it exists and
  # let the Python report record the discrepancy and checkpoint identity.
  if [[ -d "$SCRATCH/llama_3_8b_instruct" ]]; then
    LLAMA31_STUDENT_PATH="$SCRATCH/llama_3_8b_instruct"
  else
    LLAMA31_STUDENT_PATH="/home/taegyoem/scratch/llama31_8b_instruct"
  fi
fi
LLAMA31_CRITIC_PATH="${LLAMA31_CRITIC_PATH:-/home/taegyoem/scratch/llama_guard_2}"
LLAMA2_STUDENT_PATH="${LLAMA2_STUDENT_PATH:-/home/taegyoem/scratch/llama2_7b_chat_hf}"
LLAMA2_CRITIC_PATH="${LLAMA2_CRITIC_PATH:-/home/taegyoem/scratch/llama_guard_7b}"

mkdir -p "$OUT_DIR"

echo "repo_root=$REPO_ROOT"
echo "output_dir=$OUT_DIR"
echo "pair=$PAIR case_count=$CASE_COUNT"
echo "llama31_student_path=$LLAMA31_STUDENT_PATH"
echo "llama31_critic_path=$LLAMA31_CRITIC_PATH"
echo "llama2_student_path=$LLAMA2_STUDENT_PATH"
echo "llama2_critic_path=$LLAMA2_CRITIC_PATH"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  scontrol show job -o "$SLURM_JOB_ID" || true
fi
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version \
  --format=csv,noheader || true

python -u scripts/check_dcl_critic_equivalence.py \
  --pair "$PAIR" \
  --case-count "$CASE_COUNT" \
  --output-dir "$OUT_DIR" \
  --llama31-student-path "$LLAMA31_STUDENT_PATH" \
  --llama31-critic-path "$LLAMA31_CRITIC_PATH" \
  --llama2-student-path "$LLAMA2_STUDENT_PATH" \
  --llama2-critic-path "$LLAMA2_CRITIC_PATH" \
  "$@"
