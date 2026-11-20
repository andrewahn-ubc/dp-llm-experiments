#!/bin/bash
#SBATCH --job-name=l2b_control
#SBATCH --account=def-mijungp
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=output/l2b_control_%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SCRATCH}/dp-llm-experiments}"
VENV_ACTIVATE="${VENV_ACTIVATE:-${SCRATCH}/venv/nanogcg/bin/activate}"
: "${STATE:?STATE must point to the pipeline state JSON}"
: "${WAVE:?WAVE must be set}"
: "${ACTION:?ACTION must be train or eval}"
TRAINING_EPOCH="${TRAINING_EPOCH:-0}"

cd "$REPO_ROOT"
mkdir -p output
module load StdEnv/2023 python/3.11
source "$VENV_ACTIVATE"

python -m experiments.llama2_benign_sweep.pipeline continue-wave \
    --state "$STATE" \
    --wave "$WAVE" \
    --action "$ACTION" \
    --training-epoch "$TRAINING_EPOCH"
