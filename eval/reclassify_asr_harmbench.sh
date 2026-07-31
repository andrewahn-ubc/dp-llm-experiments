#!/bin/bash
#SBATCH --job-name=reclass_hb_asr
#SBATCH --account=def-mijungp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=3:00:00
#SBATCH --output=output/reclassify_asr_harmbench_%j.out

# Re-judge ASR on saved harmful CSVs with HarmBench Mistral val cls, then
# regenerate overall + per-benchmark × attack stats.
#
# Default target: Llama-2 Adv-SFT (λ=0 pertlm) under test_eval_outputs.
#
# Submit from repo root on Narval:
#   mkdir -p output
#   sbatch eval/reclassify_asr_harmbench.sh
#
# Overrides:
#   sbatch --export=ALL,RESULTS_DIR=$SCRATCH/dp-llm-sweep/test_eval_outputs,\
#GLOB='pert_reg_run_lr2e-05_lam0_eps0_pertlm_ep5_*_harmful.csv' \
#     eval/reclassify_asr_harmbench.sh

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$SCRATCH/dp-llm-experiments}"
cd "$REPO_ROOT"
mkdir -p output
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

module load StdEnv/2023 python/3.11
# shellcheck disable=SC1090
source "${VENV_ACTIVATE:-$SCRATCH/venv/nanogcg/bin/activate}"

export TRANSFORMERS_CACHE="${SLURM_TMPDIR:-/tmp}/hf_cache"
export HF_HOME="${SLURM_TMPDIR:-/tmp}/hf_home"
mkdir -p "$TRANSFORMERS_CACHE" "$HF_HOME"

RESULTS_DIR="${RESULTS_DIR:-$SCRATCH/dp-llm-sweep/test_eval_outputs}"
GLOB="${GLOB:-pert_reg_run_lr2e-05_lam0_eps0_pertlm_ep5_*_harmful.csv}"
CLS_PATH="${CLS_PATH:-$SCRATCH/harmbench_mistral_val_cls}"
LABELS_CSV="${LABELS_CSV:-$REPO_ROOT/official_data/combined_test_dataset.csv}"
BATCH_SIZE="${BATCH_SIZE:-8}"

if [[ ! -f "$LABELS_CSV" && -f "$REPO_ROOT/official/combined_test_dataset.csv" ]]; then
  LABELS_CSV="$REPO_ROOT/official/combined_test_dataset.csv"
fi

echo "RESULTS_DIR=$RESULTS_DIR"
echo "GLOB=$GLOB"
echo "CLS_PATH=$CLS_PATH"
echo "LABELS_CSV=$LABELS_CSV"
echo
ls -1 "$RESULTS_DIR"/$GLOB 2>/dev/null || {
  echo "ERROR: no files match $RESULTS_DIR/$GLOB" >&2
  exit 2
}
echo

python "$REPO_ROOT/eval/reclassify_asr_harmbench.py" \
  --results-dir "$RESULTS_DIR" \
  --glob "$GLOB" \
  --cls-path "$CLS_PATH" \
  --labels-csv "$LABELS_CSV" \
  --batch-size "$BATCH_SIZE"

echo
echo "Done. Updated CSVs +:"
echo "  ${RESULTS_DIR}/pert_reg_run_lr2e-05_lam0_eps0_pertlm_ep5_metrics.tsv"
echo "  ${RESULTS_DIR}/pert_reg_run_lr2e-05_lam0_eps0_pertlm_ep5_asr_by_benchmark_attack.csv"
