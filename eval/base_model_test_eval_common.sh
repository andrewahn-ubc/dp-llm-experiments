#!/usr/bin/env bash
# Shared body for Narval base-model test eval (Llama-2 / Llama-3 / Mistral).
#
# Prerequisites:
#   • export MODEL_PROFILE to one of: llama_2_7b_chat, llama_3_8b_instruct, mistral_7b_instruct
#   • HF weights + judges at paths in train/model_profiles.py ($SCRATCH/..., overridable)
#
# Optional env:
#   REPO_ROOT              default: ${SCRATCH}/dp-llm-experiments
#   OFFICIAL_DATA_DIR      default: $REPO_ROOT/official_data  (use $REPO_ROOT/official if you keep CSVs there)
#   HARMFUL_DATA / BENIGN_DATA  override full paths to test CSVs
#   BASE_MODEL_TEST_OUT_DIR   default: ${SCRATCH}/dp-llm-eval/base_model_test
#   SYSTEM_PROMPT_MODE       default: empty (same as train/submit_wandb_sweep.py + test_eval_matrix.py)
#   VENV_ACTIVATE            default: ${SCRATCH}/venv/nanogcg/bin/activate
#
# Judges / protocol match eval/eval.py + eval/eval_helpers.py used by the training-eval pipeline:
#   HarmBench Mistral val classifier (HARMBENCH_STANDARD_CLS_PROMPT) for ASR;
#   Mistral-Instruct + create_formatted_frr_prompt (Yes/No refusal) for FRR.

set -euo pipefail

if [[ -z "${MODEL_PROFILE:-}" ]]; then
  echo "ERROR: MODEL_PROFILE must be set (e.g. llama_2_7b_chat)." >&2
  exit 2
fi

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${SCRATCH}/dp-llm-experiments}}"
if [[ ! -d "${REPO_ROOT}" ]]; then
  echo "ERROR: REPO_ROOT is not a directory: ${REPO_ROOT}" >&2
  echo "Fix: export REPO_ROOT=/absolute/path/to/dp-llm-experiments" >&2
  exit 2
fi
cd "${REPO_ROOT}"

OFFICIAL_DATA_DIR="${OFFICIAL_DATA_DIR:-${REPO_ROOT}/official_data}"
HARMFUL_DATA="${HARMFUL_DATA:-${OFFICIAL_DATA_DIR}/combined_test_dataset.csv}"
BENIGN_DATA="${BENIGN_DATA:-${OFFICIAL_DATA_DIR}/frr_test.csv}"

if [[ ! -f "${HARMFUL_DATA}" ]]; then
  echo "ERROR: harmful test CSV not found: ${HARMFUL_DATA}" >&2
  echo "Set HARMFUL_DATA or OFFICIAL_DATA_DIR (e.g. export OFFICIAL_DATA_DIR=\${REPO_ROOT}/official)." >&2
  exit 2
fi
if [[ ! -f "${BENIGN_DATA}" ]]; then
  echo "ERROR: benign test CSV not found: ${BENIGN_DATA}" >&2
  exit 2
fi

BASE_MODEL_TEST_OUT_DIR="${BASE_MODEL_TEST_OUT_DIR:-${SCRATCH}/dp-llm-eval/base_model_test}"
OUT_DIR="${BASE_MODEL_TEST_OUT_DIR}/${MODEL_PROFILE}"
mkdir -p "${OUT_DIR}" output

RUN_TAG="base_test_${MODEL_PROFILE}"
HARMFUL_OUT_STEM="${OUT_DIR}/${RUN_TAG}_harmful"
BENIGN_OUT_STEM="${OUT_DIR}/${RUN_TAG}_benign"
METRICS_OUT="${OUT_DIR}/${RUN_TAG}_metrics.tsv"

SYSTEM_PROMPT_MODE="${SYSTEM_PROMPT_MODE:-empty}"

module load StdEnv/2023 python/3.11
# shellcheck source=/dev/null
source "${VENV_ACTIVATE:-${SCRATCH}/venv/nanogcg/bin/activate}"

export TRANSFORMERS_CACHE="${SLURM_TMPDIR:-/tmp}/hf_cache"
export HF_HOME="${SLURM_TMPDIR:-/tmp}/hf_home"
mkdir -p "${TRANSFORMERS_CACHE}" "${HF_HOME}"

resolve_output() {
  local stem="$1"
  if [[ -f "${stem}.csv" ]]; then
    echo "${stem}.csv"
  elif [[ -f "${stem}.csv.csv" ]]; then
    echo "${stem}.csv.csv"
  else
    echo ""
  fi
}

echo "=== Base model test eval ==="
echo "MODEL_PROFILE=${MODEL_PROFILE}"
echo "REPO_ROOT=${REPO_ROOT}"
echo "Harmful test: ${HARMFUL_DATA}"
echo "Benign test (FRR): ${BENIGN_DATA}"
echo "Outputs under: ${OUT_DIR}"
echo "system_prompt_mode=${SYSTEM_PROMPT_MODE}"

HARMFUL_OUT="$(resolve_output "${HARMFUL_OUT_STEM}")"
BENIGN_OUT="$(resolve_output "${BENIGN_OUT_STEM}")"

if [[ -n "${HARMFUL_OUT}" && -n "${BENIGN_OUT}" ]]; then
  echo "Found existing eval outputs; skipping eval.py:"
  echo "  ${HARMFUL_OUT}"
  echo "  ${BENIGN_OUT}"
else
  HARMFUL_OUT="${HARMFUL_OUT_STEM}.csv"
  BENIGN_OUT="${BENIGN_OUT_STEM}.csv"

  BENIGN_TMP="${SLURM_TMPDIR:-/tmp}/frr_eval_input_${MODEL_PROFILE}_$$.csv"
  python "${REPO_ROOT}/eval/prep_frr_eval_input.py" "${BENIGN_DATA}" "${BENIGN_TMP}"

  python "${REPO_ROOT}/eval/eval.py" \
    --eval-mode "seen-family" \
    --model-profile "${MODEL_PROFILE}" \
    --system-prompt-mode "${SYSTEM_PROMPT_MODE}" \
    --validation-data "${HARMFUL_DATA}" \
    --benign-validation-data "${BENIGN_TMP}" \
    --harmful-output-file "${HARMFUL_OUT_STEM}" \
    --benign-output-file "${BENIGN_OUT_STEM}"
fi

HARMFUL_OUT="$(resolve_output "${HARMFUL_OUT_STEM}")"
BENIGN_OUT="$(resolve_output "${BENIGN_OUT_STEM}")"
if [[ -z "${HARMFUL_OUT}" || -z "${BENIGN_OUT}" ]]; then
  echo "ERROR: expected harmful/benign CSVs under stems:" >&2
  echo "  ${HARMFUL_OUT_STEM}.csv" >&2
  echo "  ${BENIGN_OUT_STEM}.csv" >&2
  exit 2
fi

python "${REPO_ROOT}/eval/write_base_model_test_metrics.py" \
  "${HARMFUL_OUT}" \
  "${BENIGN_OUT}" \
  "${METRICS_OUT}" \
  --model-profile "${MODEL_PROFILE}"

echo "Done."
echo "  harmful: ${HARMFUL_OUT}"
echo "  benign:  ${BENIGN_OUT}"
echo "  metrics: ${METRICS_OUT}"
