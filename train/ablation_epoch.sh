#!/bin/bash
#SBATCH --job-name=dcl_ablation
#SBATCH --account=aip-mijungp
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=4:00:00
#SBATCH --output=output/dcl_ablation_%j.out
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=ammany01@cs.ubc.ca
#
# Single-epoch step for rebuttal Section-2 core-design ablations (seen-family,
# Llama-2-7B-Chat, lambda=0.1, epsilon=-0.5, 5 epochs total, up to 4h/epoch on an
# L40S). Runs on Vulcan (aip-mijungp). Chained via --dependency=afterok by
# submit_ablation.sh, mirroring train/epoch{1..5}.sh (which target Narval).
#
# One-time setup: train/setup_ablation_venv.sh (see that file's header).
#
# Required env:
#   ABLATION_TAG     one of: clean_adv_sft | symmetric | direct_unsafe | no_clean_sft
#   START_EPOCH      1..5
#   RESUME_FROM      checkpoint dir to resume from (unset/empty for epoch 1)
#
# Optional env:
#   LAMBDA (default 0.1), EPSILON (default -0.5), LR (default 2e-5)
#   TOTAL_EPOCHS (default 5), LORA_RANK (default 8)
#   TRAINING_DATA (default $REPO_ROOT/official_data/train_plus_validation.csv --
#              upload this file there first; it is gitignored, per CLAUDE.md)
#   REPO_ROOT (default $HOME/repos/dp-llm-experiments -- code lives on $HOME on
#              Vulcan, not $SCRATCH; see delman_eval.sh's precedent)
#   VENV_ACTIVATE (default $SCRATCH/venv/dcl_train/bin/activate)
#   BASE_LLM / HINGE_GUARD_PATH  override train/model_profiles.py's Narval-shaped
#              defaults ($SCRATCH/llama2_7b_chat_hf etc.) -- Vulcan keeps HF
#              snapshots under $SCRATCH/hf_models/ instead (see delman_eval.sh)

set -euo pipefail

if [[ -z "${ABLATION_TAG:-}" || -z "${START_EPOCH:-}" ]]; then
  echo "ERROR: ABLATION_TAG and START_EPOCH must be set." >&2
  exit 2
fi

case "${ABLATION_TAG}" in
  clean_adv_sft)
    # Item 2.1: CE(r|h) + CE(r|h'), no hinge.
    LOSS_VARIANT="dcl"
    LM_LOSS_INPUT="both"
    LAMBDA_DEFAULT=0.0
    ;;
  symmetric)
    # Item 2.2: symmetric |C(pert) - C(clean)| in place of the asymmetric hinge.
    LOSS_VARIANT="symmetric"
    LM_LOSS_INPUT="clean"
    LAMBDA_DEFAULT=0.1
    ;;
  direct_unsafe)
    # Item 2.3: regularizer is C(pert) alone, no anchor/directionality.
    LOSS_VARIANT="direct-unsafe"
    LM_LOSS_INPUT="clean"
    LAMBDA_DEFAULT=0.1
    ;;
  no_clean_sft)
    # Item 2.4: DCL hinge kept, LM term dropped entirely.
    LOSS_VARIANT="no-clean-sft"
    LM_LOSS_INPUT="clean"   # ignored by train.py when loss_variant=no-clean-sft
    LAMBDA_DEFAULT=0.1
    ;;
  *)
    echo "ERROR: unknown ABLATION_TAG=${ABLATION_TAG}" >&2
    exit 2
    ;;
esac

REPO_ROOT="${REPO_ROOT:-${HOME}/repos/dp-llm-experiments}"
cd "${REPO_ROOT}"

LAMBDA="${LAMBDA:-${LAMBDA_DEFAULT}}"
EPSILON="${EPSILON:--0.5}"
LR="${LR:-2e-5}"
LORA_RANK="${LORA_RANK:-8}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-5}"
TRAINING_DATA="${TRAINING_DATA:-${REPO_ROOT}/official_data/train_plus_validation.csv}"

# model_profiles.py's llama_2_7b_chat defaults ($SCRATCH/llama2_7b_chat_hf,
# $SCRATCH/llama_guard_7b) are Narval-shaped. Vulcan keeps HF snapshots under
# $SCRATCH/hf_models/ instead (same override pattern as delman_eval.sh).
BASE_LLM="${BASE_LLM:-${SCRATCH}/hf_models/llama2_7b_chat_hf}"
HINGE_GUARD_PATH="${HINGE_GUARD_PATH:-${SCRATCH}/hf_models/llama_guard_7b}"

if [[ ! -d "${BASE_LLM}" ]]; then
  echo "ERROR: base LLM not found: ${BASE_LLM}" >&2
  exit 2
fi
if [[ ! -d "${HINGE_GUARD_PATH}" ]]; then
  echo "ERROR: hinge guard model not found: ${HINGE_GUARD_PATH}" >&2
  echo "  Llama-Guard-3-8B (already on this cluster) is NOT a substitute --" >&2
  echo "  it's Llama-3.1-tokenized, but train.py's soft-rollout hinge concatenates" >&2
  echo "  the STUDENT model's embeddings directly with the guard's, so guard and" >&2
  echo "  student must share one tokenizer/vocab. Download the Llama-2-based guard:" >&2
  echo "    hf download meta-llama/LlamaGuard-7b --local-dir ${HINGE_GUARD_PATH}" >&2
  exit 2
fi
if [[ ! -f "${TRAINING_DATA}" ]]; then
  echo "ERROR: training data not found: ${TRAINING_DATA}" >&2
  exit 2
fi

FINETUNED_BASE="${SCRATCH}/ablation_${ABLATION_TAG}_finetuned_llm"
mkdir -p "$(dirname "${FINETUNED_BASE}")" output

echo "[INFO] Setting up Python environment"
module purge
module load StdEnv/2023 cuda/12.2 python/3.11 gcc arrow/21.0.0 scipy-stack
# shellcheck source=/dev/null
source "${VENV_ACTIVATE:-${SCRATCH}/venv/dcl_train/bin/activate}"

export TRANSFORMERS_CACHE="${SLURM_TMPDIR:-/tmp}/hf_cache"
export HF_HOME="${SLURM_TMPDIR:-/tmp}/hf_home"
mkdir -p "${TRANSFORMERS_CACHE}" "${HF_HOME}"

echo "=== DCL ablation: ${ABLATION_TAG} (epoch ${START_EPOCH}/${TOTAL_EPOCHS}) ==="
echo "loss_variant=${LOSS_VARIANT} lm_loss_input=${LM_LOSS_INPUT} lambda=${LAMBDA} epsilon=${EPSILON}"
echo "base_llm=${BASE_LLM} hinge_guard_path=${HINGE_GUARD_PATH}"
echo "resume_from=${RESUME_FROM:-<none>}"

RESUME_ARGS=()
if [[ -n "${RESUME_FROM:-}" ]]; then
  RESUME_ARGS+=(--resume-from "${RESUME_FROM}")
fi

python "${REPO_ROOT}/train/train.py" \
  --eval-mode "seen-family" \
  --finetuned-llm-path "${FINETUNED_BASE}" \
  --training-data "${TRAINING_DATA}" \
  --base-llm "${BASE_LLM}" \
  --hinge-guard-path "${HINGE_GUARD_PATH}" \
  --lr "${LR}" \
  --lambda-val "${LAMBDA}" \
  --epsilon "${EPSILON}" \
  --lora-rank "${LORA_RANK}" \
  --total-epochs "${TOTAL_EPOCHS}" \
  --start-epoch "${START_EPOCH}" \
  --loss-variant "${LOSS_VARIANT}" \
  --lm-loss-input "${LM_LOSS_INPUT}" \
  "${RESUME_ARGS[@]}"

echo "Checkpoint: ${FINETUNED_BASE}_epoch${START_EPOCH}"
