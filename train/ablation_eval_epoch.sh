#!/bin/bash
#SBATCH --job-name=dcl_ablation_eval
#SBATCH --account=aip-mijungp
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=4:00:00
#SBATCH --output=output/dcl_ablation_eval_%j.out
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=ammany01@cs.ubc.ca
#
# Seen-family ASR + FRR eval for one epoch checkpoint of a rebuttal core-design
# ablation, run as a SEPARATE job from training (not inline in train.py) so
# training and eval don't contend for the same GPU's VRAM and an eval failure
# can't take down the training chain. Submitted by submit_ablation.sh with
# --dependency=afterok:<that epoch's training job>, so it starts only after
# the checkpoint it evaluates has actually been saved.
#
# Uses the exact same protocol as eval/eval.py (HarmBench Mistral val
# classifier for ASR, Mistral-Instruct for FRR) so numbers are directly
# comparable to Table 1/3 in the paper draft.
#
# Required env:
#   ABLATION_TAG   one of: clean_adv_sft | symmetric | direct_unsafe | no_clean_sft
#   EPOCH          1..5 -- selects which checkpoint to evaluate
#                  ($SCRATCH/ablation_<tag>_finetuned_llm_epoch<N>)
#
# Optional env:
#   REPO_ROOT (default $HOME/repos/dp-llm-experiments)
#   VENV_ACTIVATE (default $SCRATCH/venv/dcl_train/bin/activate)
#   BASE_LLM (default $SCRATCH/hf_models/Llama-2-7b-chat-hf -- check exact
#             casing with `ls $SCRATCH/hf_models/` before relying on this default)
#   REFUSAL_JUDGE_PATH (default $SCRATCH/hf_models/mistral_7b_instruct)
#   JAILBREAK_CLASSIFIER_PATH (default $SCRATCH/hf_models/harmbench_mistral_val_cls)
#   HARMFUL_DATA (default $REPO_ROOT/official_data/combined_test_dataset.csv)
#   BENIGN_DATA (default $REPO_ROOT/official_data/frr_test.csv)
#   OUT_DIR (default $SCRATCH/dp-llm-eval/ablation/<tag>)

set -euo pipefail

if [[ -z "${ABLATION_TAG:-}" || -z "${EPOCH:-}" ]]; then
  echo "ERROR: ABLATION_TAG and EPOCH must be set." >&2
  exit 2
fi

REPO_ROOT="${REPO_ROOT:-${HOME}/repos/dp-llm-experiments}"
cd "${REPO_ROOT}"

CHECKPOINT="${SCRATCH}/ablation_${ABLATION_TAG}_finetuned_llm_epoch${EPOCH}"
if [[ ! -d "${CHECKPOINT}" ]]; then
  echo "ERROR: checkpoint not found: ${CHECKPOINT}" >&2
  echo "  (this eval job should only run after its training epoch completes --" >&2
  echo "  check the --dependency=afterok wiring in submit_ablation.sh)" >&2
  exit 2
fi

BASE_LLM="${BASE_LLM:-${SCRATCH}/hf_models/Llama-2-7b-chat-hf}"
REFUSAL_JUDGE_PATH="${REFUSAL_JUDGE_PATH:-${SCRATCH}/hf_models/mistral_7b_instruct}"
JAILBREAK_CLASSIFIER_PATH="${JAILBREAK_CLASSIFIER_PATH:-${SCRATCH}/hf_models/harmbench_mistral_val_cls}"
HARMFUL_DATA="${HARMFUL_DATA:-${REPO_ROOT}/official_data/combined_test_dataset.csv}"
BENIGN_DATA="${BENIGN_DATA:-${REPO_ROOT}/official_data/frr_test.csv}"

for p in "${BASE_LLM}" "${REFUSAL_JUDGE_PATH}" "${JAILBREAK_CLASSIFIER_PATH}"; do
  if [[ ! -d "${p}" ]]; then
    echo "ERROR: model path not found: ${p}" >&2
    exit 2
  fi
done
if [[ ! -f "${HARMFUL_DATA}" ]]; then
  echo "ERROR: harmful test CSV not found: ${HARMFUL_DATA}" >&2
  exit 2
fi
if [[ ! -f "${BENIGN_DATA}" ]]; then
  echo "ERROR: benign test CSV not found: ${BENIGN_DATA}" >&2
  exit 2
fi

OUT_DIR="${OUT_DIR:-${SCRATCH}/dp-llm-eval/ablation/${ABLATION_TAG}}"
mkdir -p "${OUT_DIR}" output
RUN_TAG="ablation_${ABLATION_TAG}_epoch${EPOCH}"

echo "[INFO] Setting up Python environment"
module purge
module load StdEnv/2023 cuda/12.2 python/3.11 gcc arrow/21.0.0 scipy-stack
# shellcheck source=/dev/null
source "${VENV_ACTIVATE:-${SCRATCH}/venv/dcl_train/bin/activate}"

export TRANSFORMERS_CACHE="${SLURM_TMPDIR:-/tmp}/hf_cache"
export HF_HOME="${SLURM_TMPDIR:-/tmp}/hf_home"
mkdir -p "${TRANSFORMERS_CACHE}" "${HF_HOME}"

echo "=== DCL ablation eval: ${ABLATION_TAG} epoch ${EPOCH} ==="
echo "checkpoint=${CHECKPOINT}"
echo "base_llm=${BASE_LLM}"
echo "refusal_judge=${REFUSAL_JUDGE_PATH}"
echo "jailbreak_classifier=${JAILBREAK_CLASSIFIER_PATH}"
echo "outputs under: ${OUT_DIR}"

# eval.py's FRR path expects an 'Original Prompt' column.
BENIGN_TMP="${SLURM_TMPDIR:-/tmp}/frr_eval_input_${RUN_TAG}_$$.csv"
python "${REPO_ROOT}/eval/prep_frr_eval_input.py" "${BENIGN_DATA}" "${BENIGN_TMP}"

HARMFUL_OUT_STEM="${OUT_DIR}/${RUN_TAG}_harmful"
BENIGN_OUT_STEM="${OUT_DIR}/${RUN_TAG}_benign"

python "${REPO_ROOT}/eval/eval.py" \
  --eval-mode "seen-family" \
  --base-llm "${BASE_LLM}" \
  --resume-from "${CHECKPOINT}" \
  --refusal-judge-path "${REFUSAL_JUDGE_PATH}" \
  --jailbreak-classifier-path "${JAILBREAK_CLASSIFIER_PATH}" \
  --system-prompt-mode "empty" \
  --validation-data "${HARMFUL_DATA}" \
  --benign-validation-data "${BENIGN_TMP}" \
  --harmful-output-file "${HARMFUL_OUT_STEM}" \
  --benign-output-file "${BENIGN_OUT_STEM}"

python "${REPO_ROOT}/eval/write_base_model_test_metrics.py" \
  "${HARMFUL_OUT_STEM}.csv" \
  "${BENIGN_OUT_STEM}.csv" \
  "${OUT_DIR}/${RUN_TAG}_metrics.tsv" \
  --model-profile "${ABLATION_TAG}_epoch${EPOCH}"

echo "Done."
echo "  harmful: ${HARMFUL_OUT_STEM}.csv"
echo "  benign:  ${BENIGN_OUT_STEM}.csv"
echo "  metrics: ${OUT_DIR}/${RUN_TAG}_metrics.tsv"
