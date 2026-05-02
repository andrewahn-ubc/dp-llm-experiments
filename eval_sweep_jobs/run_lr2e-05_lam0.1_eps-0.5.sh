#!/bin/bash
#SBATCH --job-name=ev_run_lr2e-05_lam0.1_eps-0.5
#SBATCH --account=rrg-mijungp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=7:00:00
#SBATCH --output=output/eval_run_lr2e-05_lam0.1_eps-0.5_%j.out

mkdir -p output

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

module load StdEnv/2023 python/3.11
source "$SCRATCH/venv/nanogcg/bin/activate"

# Node-local Hugging Face caches (same pattern as train jobs)
export TRANSFORMERS_CACHE=$SLURM_TMPDIR/hf_cache
export HF_HOME=$SLURM_TMPDIR/hf_home
mkdir -p "$TRANSFORMERS_CACHE" "$HF_HOME"

# Weights & Biases — offline on compute nodes; sync from login afterwards
export WANDB_MODE=offline
export WANDB_PROJECT="dp-llm-safety-eval"
export WANDB_DIR="$SCRATCH/wandb_offline"

export FINETUNED_BASE="$SCRATCH/dp-llm-sweep/run_lr2e-05_lam0.1_eps-0.5_finetuned_llm"
export EVAL_PY="$SCRATCH/dp-llm-experiments/eval/eval_sweep.py"
export EVAL_OUTPUT_ROOT="$SCRATCH/dp-llm-eval"
export EVAL_SWEEP_SUMMARY="$SCRATCH/dp-llm-eval/summary.tsv"
mkdir -p "$EVAL_OUTPUT_ROOT"
mkdir -p "$WANDB_DIR"
mkdir -p "$(dirname "$EVAL_SWEEP_SUMMARY")"

# -------- Epoch 1 --------
if [ ! -d "${FINETUNED_BASE}_epoch1" ]; then
  echo "WARNING: adapter not found: ${FINETUNED_BASE}_epoch1 — skipping epoch 1"
else
  export WANDB_RUN_ID="991fb89469134be8b1dfd7ee1909cbb2"
  export WANDB_RUN_NAME="eval_run_lr2e-05_lam0.1_eps-0.5_epoch1"
  python "$EVAL_PY" \
    --slug "run_lr2e-05_lam0.1_eps-0.5" \
    --epoch 1 \
    --lr 2e-05 \
    --lambda-val 0.1 \
    --epsilon -0.5 \
    --base-llm "/home/taegyoem/scratch/llama2_7b_chat_hf" \
    --resume-from "${FINETUNED_BASE}_epoch1" \
    --judge-path "/home/taegyoem/scratch/mistral_7b_instruct" \
    --validation-data "$SCRATCH/dp-llm-experiments/official_data/validation.csv" \
    --benign-validation-data "$SCRATCH/dp-llm-experiments/official_data/frr_validation.csv" \
    --harmful-output-file "$EVAL_OUTPUT_ROOT/run_lr2e-05_lam0.1_eps-0.5_epoch1_harmful.csv" \
    --benign-output-file "$EVAL_OUTPUT_ROOT/run_lr2e-05_lam0.1_eps-0.5_epoch1_benign.csv" \
    --summary-file "$EVAL_SWEEP_SUMMARY" \
    --gen-batch-size 8 \
    --judge-batch-size 8
fi

# -------- Epoch 3 --------
if [ ! -d "${FINETUNED_BASE}_epoch3" ]; then
  echo "WARNING: adapter not found: ${FINETUNED_BASE}_epoch3 — skipping epoch 3"
else
  export WANDB_RUN_ID="bec9781a88104ef3964db7a3abbe3149"
  export WANDB_RUN_NAME="eval_run_lr2e-05_lam0.1_eps-0.5_epoch3"
  python "$EVAL_PY" \
    --slug "run_lr2e-05_lam0.1_eps-0.5" \
    --epoch 3 \
    --lr 2e-05 \
    --lambda-val 0.1 \
    --epsilon -0.5 \
    --base-llm "/home/taegyoem/scratch/llama2_7b_chat_hf" \
    --resume-from "${FINETUNED_BASE}_epoch3" \
    --judge-path "/home/taegyoem/scratch/mistral_7b_instruct" \
    --validation-data "$SCRATCH/dp-llm-experiments/official_data/validation.csv" \
    --benign-validation-data "$SCRATCH/dp-llm-experiments/official_data/frr_validation.csv" \
    --harmful-output-file "$EVAL_OUTPUT_ROOT/run_lr2e-05_lam0.1_eps-0.5_epoch3_harmful.csv" \
    --benign-output-file "$EVAL_OUTPUT_ROOT/run_lr2e-05_lam0.1_eps-0.5_epoch3_benign.csv" \
    --summary-file "$EVAL_SWEEP_SUMMARY" \
    --gen-batch-size 8 \
    --judge-batch-size 8
fi

# -------- Epoch 5 --------
if [ ! -d "${FINETUNED_BASE}_epoch5" ]; then
  echo "WARNING: adapter not found: ${FINETUNED_BASE}_epoch5 — skipping epoch 5"
else
  export WANDB_RUN_ID="361d0b9b9c0c4fd189f72fe52936c2c2"
  export WANDB_RUN_NAME="eval_run_lr2e-05_lam0.1_eps-0.5_epoch5"
  python "$EVAL_PY" \
    --slug "run_lr2e-05_lam0.1_eps-0.5" \
    --epoch 5 \
    --lr 2e-05 \
    --lambda-val 0.1 \
    --epsilon -0.5 \
    --base-llm "/home/taegyoem/scratch/llama2_7b_chat_hf" \
    --resume-from "${FINETUNED_BASE}_epoch5" \
    --judge-path "/home/taegyoem/scratch/mistral_7b_instruct" \
    --validation-data "$SCRATCH/dp-llm-experiments/official_data/validation.csv" \
    --benign-validation-data "$SCRATCH/dp-llm-experiments/official_data/frr_validation.csv" \
    --harmful-output-file "$EVAL_OUTPUT_ROOT/run_lr2e-05_lam0.1_eps-0.5_epoch5_harmful.csv" \
    --benign-output-file "$EVAL_OUTPUT_ROOT/run_lr2e-05_lam0.1_eps-0.5_epoch5_benign.csv" \
    --summary-file "$EVAL_SWEEP_SUMMARY" \
    --gen-batch-size 8 \
    --judge-batch-size 8
fi

echo "Eval done for run_lr2e-05_lam0.1_eps-0.5; summary appended to $EVAL_SWEEP_SUMMARY"

