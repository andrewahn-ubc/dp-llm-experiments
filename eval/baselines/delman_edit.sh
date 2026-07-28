#!/bin/bash
#SBATCH --job-name=delman_edit
#SBATCH --account=aip-mijungp
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=06:00:00
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
#   REPO_ROOT       default: ${HOME}/repos/dp-llm-experiments  (code lives on $HOME on
#                   Vulcan, not $SCRATCH -- override if your checkout is elsewhere)
#   BASE_MODEL      default: ${SCRATCH}/hf_models/Llama-2-7b-chat-hf  (must be HF-format dir)
#   HPARAMS_FNAME   default: Llama-2-7b-chat-hf.json  (must exist under
#                   third_party/DELMAN/hparams/; for Llama-3.1-8B-Instruct use
#                   Llama-3.1-8B-Instruct.json -- also needs its own mom2 cov-stats
#                   cache under data/stats/<model_name>/wikipedia_stats/, model_name
#                   being BASE_MODEL with "/" -> "_")
#   OUT_NAME        default: DELMAN_llama2_7b_chat
#   VENV_ACTIVATE   default: ${SCRATCH}/venv/delman/bin/activate  (separate venv;
#                   DELMAN pins transformers==4.49, which may conflict with the
#                   nanogcg venv used elsewhere in this repo)
#
# Example for Llama-3.1-8B-Instruct:
#   BASE_MODEL=${SCRATCH}/hf_models/llama3_1_8b_instruct \
#   HPARAMS_FNAME=Llama-3.1-8B-Instruct.json \
#   OUT_NAME=DELMAN_llama3_1_8b_instruct \
#     sbatch eval/baselines/delman_edit.sh
#
# Prereqs (one-time, on a login node with internet):
#   cd third_party/DELMAN
#   conda create -n delman python=3.9.20   # or: virtualenv $SCRATCH/venv/delman
#   pip install -r requirements.txt
#   # Optional but strongly recommended: download precomputed cov ("mom2") stats
#   # from the Drive link in third_party/DELMAN/README.md into data/stats/ to
#   # avoid recomputing 100k-sample Wikipedia covariance stats on first run.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${HOME}/repos/dp-llm-experiments}"
DELMAN_DIR="${REPO_ROOT}/third_party/DELMAN"
cd "${DELMAN_DIR}"
mkdir -p output

# run_delman.py writes edited checkpoints to a RESULTS_DIR that globals.yml
# sets as the *relative* path "results" (resolved from cwd, i.e. DELMAN_DIR).
# Left as-is that would write multi-GB model checkpoints under $HOME, which
# is typically small-quota on HPC clusters. Redirect via a symlink into
# $SCRATCH instead of touching the vendored globals.yml/run_delman.py.
DELMAN_RESULTS_DIR="${DELMAN_RESULTS_DIR:-${SCRATCH}/dp-llm-experiments/third_party_delman_results}"
mkdir -p "${DELMAN_RESULTS_DIR}"
if [[ -e "results" && ! -L "results" ]]; then
  echo "ERROR: ${DELMAN_DIR}/results already exists and is not a symlink." >&2
  echo "Move or remove it, then re-run (refusing to clobber existing output)." >&2
  exit 2
fi
ln -sfn "${DELMAN_RESULTS_DIR}" results
echo "results/ -> ${DELMAN_RESULTS_DIR}"

echo "[INFO] Setting up Python environment"
module purge
module load StdEnv/2023 cuda/12.2 python/3.11 gcc arrow/21.0.0 scipy-stack
# shellcheck source=/dev/null
source "${VENV_ACTIVATE:-${SCRATCH}/venv/delman/bin/activate}"

export TRANSFORMERS_CACHE="${SLURM_TMPDIR:-/tmp}/hf_cache"
export HF_HOME="${SLURM_TMPDIR:-/tmp}/hf_home"
mkdir -p "${TRANSFORMERS_CACHE}" "${HF_HOME}"

BASE_MODEL="${BASE_MODEL:-${SCRATCH}/hf_models/Llama-2-7b-chat-hf}"
HPARAMS_FNAME="${HPARAMS_FNAME:-Llama-2-7b-chat-hf.json}"
OUT_NAME="${OUT_NAME:-DELMAN_llama2_7b_chat}"

if [[ ! -d "${BASE_MODEL}" ]]; then
  echo "ERROR: BASE_MODEL not found: ${BASE_MODEL}" >&2
  echo "Expected an HF-format model directory." >&2
  exit 2
fi
if [[ ! -f "hparams/${HPARAMS_FNAME}" ]]; then
  echo "ERROR: hparams file not found: ${DELMAN_DIR}/hparams/${HPARAMS_FNAME}" >&2
  exit 2
fi

# Sanity check the cov-stats cache is present before spending time loading the
# model -- a miss silently falls back to a slow live Wikipedia computation
# inside layer_stats() rather than failing fast.
MODEL_NAME_KEY="$(python3 -c "print('${BASE_MODEL}'.replace('/', '_'))")"
STATS_CHECK_DIR="data/stats/${MODEL_NAME_KEY}/wikipedia_stats"
if [[ ! -d "${STATS_CHECK_DIR}" ]]; then
  echo "WARNING: no cached mom2 cov-stats found at ${DELMAN_DIR}/${STATS_CHECK_DIR}" >&2
  echo "         This run will fall back to a slow live Wikipedia stats computation." >&2
  echo "         See third_party/DELMAN/README.md for precomputed stats, or expect" >&2
  echo "         a much longer runtime than the ~13min seen for Llama-2-7B-chat." >&2
fi

echo "=== DELMAN edit ==="
echo "DELMAN_DIR=${DELMAN_DIR}"
echo "BASE_MODEL=${BASE_MODEL}"
echo "HPARAMS_FNAME=${HPARAMS_FNAME}"
echo "OUT_NAME=${OUT_NAME}"
echo "Edit set: data/HarmBench.json (DELMAN's own, 200 prompts)"

python3 -m run_delman \
  --model_name "${BASE_MODEL}" \
  --model_path "${BASE_MODEL}" \
  --hparams_fname "${HPARAMS_FNAME}" \
  --data_name "HarmBench.json" \
  --out_name "${OUT_NAME}"

EDITED_MODEL_DIR="${DELMAN_RESULTS_DIR}/${OUT_NAME}"
echo "Done. Edited model saved to:"
echo "  ${EDITED_MODEL_DIR}"
echo ""
echo "Next: point eval/baselines/delman_eval.sh at this directory, e.g."
echo "  DELMAN_MODEL_DIR=${EDITED_MODEL_DIR} sbatch eval/baselines/delman_eval.sh"
