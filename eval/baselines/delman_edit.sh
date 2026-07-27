#!/bin/bash
#SBATCH --job-name=delman_edit
#SBATCH --account=aip-mijungp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=output/delman_edit_%j.out
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=ammany01@cs.ubc.ca
#
# Runs DELMAN's model-editing step on Llama-2-7b-chat-hf using DELMAN's own
# shipped HarmBench.json edit-set (third_party/DELMAN/data/HarmBench.json).
# Produces one edited full checkpoint (not a LoRA adapter) that plugs into
# eval/eval.py via --base-llm, so it can be scored with the same ASR/FRR
# pipeline used for DCL and Adv. SFT.
#
# Optional env:
#   REPO_ROOT       default: ${SCRATCH}/dp-llm-experiments
#   BASE_MODEL      default: ${SCRATCH}/hf_models/Llama-2-7b-chat-hf  (must be HF-format dir)
#   OUT_NAME        default: DELMAN_llama2_7b_chat
#   VENV_ACTIVATE   default: ${SCRATCH}/venv/delman/bin/activate  (separate venv;
#                   DELMAN pins transformers==4.49, which may conflict with the
#                   nanogcg venv used elsewhere in this repo)
#
# Prereqs (one-time, on a login node with internet):
#   cd third_party/DELMAN
#   conda create -n delman python=3.9.20   # or: virtualenv $SCRATCH/venv/delman
#   pip install -r requirements.txt
#   # Optional but strongly recommended: download precomputed cov ("mom2") stats
#   # from the Drive link in third_party/DELMAN/README.md into data/stats/ to
#   # avoid recomputing 100k-sample Wikipedia covariance stats on first run.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${SCRATCH}/dp-llm-experiments}}"
DELMAN_DIR="${REPO_ROOT}/third_party/DELMAN"
cd "${DELMAN_DIR}"
mkdir -p output

module load StdEnv/2023 python/3.11
# shellcheck source=/dev/null
source "${VENV_ACTIVATE:-${SCRATCH}/venv/delman/bin/activate}"

export TRANSFORMERS_CACHE="${SLURM_TMPDIR:-/tmp}/hf_cache"
export HF_HOME="${SLURM_TMPDIR:-/tmp}/hf_home"
mkdir -p "${TRANSFORMERS_CACHE}" "${HF_HOME}"

BASE_MODEL="${BASE_MODEL:-${SCRATCH}/hf_models/Llama-2-7b-chat-hf}"
OUT_NAME="${OUT_NAME:-DELMAN_llama2_7b_chat}"

if [[ ! -d "${BASE_MODEL}" ]]; then
  echo "ERROR: BASE_MODEL not found: ${BASE_MODEL}" >&2
  echo "Expected an HF-format Llama-2-7b-chat-hf directory." >&2
  exit 2
fi

echo "=== DELMAN edit ==="
echo "DELMAN_DIR=${DELMAN_DIR}"
echo "BASE_MODEL=${BASE_MODEL}"
echo "OUT_NAME=${OUT_NAME}"
echo "Edit set: data/HarmBench.json (DELMAN's own, 200 prompts)"

python3 -m run_delman \
  --model_name "${BASE_MODEL}" \
  --model_path "${BASE_MODEL}" \
  --hparams_fname "Llama-2-7b-chat-hf.json" \
  --data_name "HarmBench.json" \
  --out_name "${OUT_NAME}"

EDITED_MODEL_DIR="${DELMAN_DIR}/results/${OUT_NAME}"
echo "Done. Edited model saved to:"
echo "  ${EDITED_MODEL_DIR}"
echo ""
echo "Next: point eval/baselines/delman_eval.sh at this directory, e.g."
echo "  DELMAN_MODEL_DIR=${EDITED_MODEL_DIR} sbatch eval/baselines/delman_eval.sh"
