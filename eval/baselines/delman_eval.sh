#!/bin/bash
#SBATCH --job-name=delman_eval
#SBATCH --account=aip-mijungp
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=10:00:00
#SBATCH --output=output/delman_eval_%j.out
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=ammany01@cs.ubc.ca
#
# Evaluates a DELMAN-edited checkpoint through the same eval/eval.py harness
# used for the base model, Adv. SFT, and DCL, against our official test set
# (official_data/combined_test_dataset.csv, official_data/frr_test.csv).
#
# Produces numbers directly comparable to Table 1 (mean ASR|FRR across
# AdvBench/HarmBench/JailbreakBench) in the DCL paper draft.
#
# Note on "held-out-family": DCL/Adv. SFT report a held-out-family protocol
# using THREE separately trained checkpoints (one per family excluded from
# training). DELMAN produces a single edited checkpoint from one fixed
# edit-set (not per-family training), so there is no equivalent held-out
# split for it -- we instead report the same single edited model's
# per-family ASR breakdown (GCG / AutoDAN / PAIR), which is directly
# comparable to the "All" row of Table 1, not to Table 2.
#
# Required env:
#   DELMAN_MODEL_DIR   path to the edited checkpoint from delman_edit.sh
#                       (e.g. $SCRATCH/dp-llm-experiments/third_party/DELMAN/results/DELMAN_llama2_7b_chat)
#
# Optional env:
#   REPO_ROOT       default: ${HOME}/repos/dp-llm-experiments  (code lives on $HOME on
#                   Vulcan, not $SCRATCH -- override if your checkout is elsewhere)
#   OFFICIAL_DATA_DIR  default: ${REPO_ROOT}/official_data
#   HARMFUL_DATA / BENIGN_DATA  override full paths to test CSVs
#   OUT_DIR         default: ${SCRATCH}/dp-llm-eval/delman
#   VENV_ACTIVATE   default: ${SCRATCH}/venv/nanogcg/bin/activate
#                   (eval.py runs fine under the existing nanogcg venv --
#                   only the DELMAN edit step itself needs the separate
#                   delman venv with transformers==4.49)

set -euo pipefail

if [[ -z "${DELMAN_MODEL_DIR:-}" ]]; then
  echo "ERROR: DELMAN_MODEL_DIR must be set to the edited checkpoint dir." >&2
  echo "  e.g. export DELMAN_MODEL_DIR=\$SCRATCH/dp-llm-experiments/third_party/DELMAN/results/DELMAN_llama2_7b_chat" >&2
  exit 2
fi
if [[ ! -d "${DELMAN_MODEL_DIR}" ]]; then
  echo "ERROR: DELMAN_MODEL_DIR not found: ${DELMAN_MODEL_DIR}" >&2
  exit 2
fi

REPO_ROOT="${REPO_ROOT:-${HOME}/repos/dp-llm-experiments}"
cd "${REPO_ROOT}"
mkdir -p output

module load StdEnv/2023 python/3.11
# shellcheck source=/dev/null
source "${VENV_ACTIVATE:-${SCRATCH}/venv/nanogcg/bin/activate}"

export TRANSFORMERS_CACHE="${SLURM_TMPDIR:-/tmp}/hf_cache"
export HF_HOME="${SLURM_TMPDIR:-/tmp}/hf_home"
mkdir -p "${TRANSFORMERS_CACHE}" "${HF_HOME}"

OFFICIAL_DATA_DIR="${OFFICIAL_DATA_DIR:-${REPO_ROOT}/official_data}"
HARMFUL_DATA="${HARMFUL_DATA:-${OFFICIAL_DATA_DIR}/combined_test_dataset.csv}"
BENIGN_DATA="${BENIGN_DATA:-${OFFICIAL_DATA_DIR}/frr_test.csv}"

if [[ ! -f "${HARMFUL_DATA}" ]]; then
  echo "ERROR: harmful test CSV not found: ${HARMFUL_DATA}" >&2
  exit 2
fi
if [[ ! -f "${BENIGN_DATA}" ]]; then
  echo "ERROR: benign test CSV not found: ${BENIGN_DATA}" >&2
  exit 2
fi

OUT_DIR="${OUT_DIR:-${SCRATCH}/dp-llm-eval/delman}"
mkdir -p "${OUT_DIR}"
RUN_TAG="delman_llama2_7b_chat"

# Vulcan keeps models under $SCRATCH/hf_models/, not directly under $SCRATCH/
# as model_profiles.py's llama_2_7b_chat profile assumes (that default was
# written for Narval). Override both judge paths explicitly rather than
# editing model_profiles.py, since other scripts share that file's defaults.
REFUSAL_JUDGE_PATH="${REFUSAL_JUDGE_PATH:-${SCRATCH}/hf_models/mistral_7b_instruct}"
JAILBREAK_CLASSIFIER_PATH="${JAILBREAK_CLASSIFIER_PATH:-${SCRATCH}/hf_models/harmbench_mistral_val_cls}"

if [[ ! -d "${REFUSAL_JUDGE_PATH}" ]]; then
  echo "ERROR: refusal judge not found: ${REFUSAL_JUDGE_PATH}" >&2
  echo "  hf download mistralai/Mistral-7B-Instruct-v0.2 --local-dir ${REFUSAL_JUDGE_PATH}" >&2
  exit 2
fi
if [[ ! -d "${JAILBREAK_CLASSIFIER_PATH}" ]]; then
  echo "ERROR: HarmBench classifier not found: ${JAILBREAK_CLASSIFIER_PATH}" >&2
  echo "  hf download cais/HarmBench-Mistral-7b-val-cls --local-dir ${JAILBREAK_CLASSIFIER_PATH}" >&2
  exit 2
fi

echo "=== DELMAN eval ==="
echo "DELMAN_MODEL_DIR=${DELMAN_MODEL_DIR}"
echo "Harmful test: ${HARMFUL_DATA}"
echo "Benign test (FRR): ${BENIGN_DATA}"
echo "Outputs under: ${OUT_DIR}"

# eval.py's FRR path expects an 'Original Prompt' column.
BENIGN_TMP="${SLURM_TMPDIR:-/tmp}/frr_eval_input_delman_$$.csv"
python "${REPO_ROOT}/eval/prep_frr_eval_input.py" "${BENIGN_DATA}" "${BENIGN_TMP}"

# --- 1. Seen-family style pass: all three families in one run (comparable
#        to Table 1's "All" row). --base-llm points straight at the edited
#        checkpoint; no --resume-from since DELMAN edits weights in place
#        rather than producing a LoRA adapter. ---
HARMFUL_OUT="${OUT_DIR}/${RUN_TAG}_harmful.csv"
BENIGN_OUT="${OUT_DIR}/${RUN_TAG}_benign.csv"

python "${REPO_ROOT}/eval/eval.py" \
  --eval-mode "seen-family" \
  --model-profile "llama_2_7b_chat" \
  --base-llm "${DELMAN_MODEL_DIR}" \
  --refusal-judge-path "${REFUSAL_JUDGE_PATH}" \
  --jailbreak-classifier-path "${JAILBREAK_CLASSIFIER_PATH}" \
  --system-prompt-mode "empty" \
  --validation-data "${HARMFUL_DATA}" \
  --benign-validation-data "${BENIGN_TMP}" \
  --harmful-output-file "${OUT_DIR}/${RUN_TAG}_harmful" \
  --benign-output-file "${OUT_DIR}/${RUN_TAG}_benign"

python "${REPO_ROOT}/eval/write_base_model_test_metrics.py" \
  "${HARMFUL_OUT}" \
  "${BENIGN_OUT}" \
  "${OUT_DIR}/${RUN_TAG}_metrics.tsv" \
  --model-profile "llama_2_7b_chat"

echo "Done."
echo "  harmful: ${HARMFUL_OUT}"
echo "  benign:  ${BENIGN_OUT}"
echo "  metrics: ${OUT_DIR}/${RUN_TAG}_metrics.tsv"
echo ""
echo "Metrics are directly comparable to Table 1 (mean ASR | FRR) in the DCL paper draft."
echo "Note: this is a single edited checkpoint scored on all three attack families at"
echo "once -- comparable to the DCL/Adv.SFT 'All' row, not to the held-out-family table,"
echo "since DELMAN does not train per-family-excluded checkpoints."
