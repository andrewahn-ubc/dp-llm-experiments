#!/bin/bash
#SBATCH --job-name=ev_run_lr2e-05_lam0.1_eps-1
#SBATCH --account=rrg-mijungp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=7:00:00
#SBATCH --output=output/eval_run_lr2e-05_lam0.1_eps-1_%j.out

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

export FINETUNED_BASE="$SCRATCH/dp-llm-sweep/run_lr2e-05_lam0.1_eps-1_finetuned_llm"
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
  export WANDB_RUN_ID="3d97ecccbf9849bea5dcf90080d4eefb"
  export WANDB_RUN_NAME="eval_run_lr2e-05_lam0.1_eps-1_epoch1"
  python "$EVAL_PY" \
    --slug "run_lr2e-05_lam0.1_eps-1" \
    --epoch 1 \
    --lr 2e-05 \
    --lambda-val 0.1 \
    --epsilon -1.0 \
    --base-llm "/home/taegyoem/scratch/llama2_7b_chat_hf" \
    --resume-from "${FINETUNED_BASE}_epoch1" \
    --judge-path "/home/taegyoem/scratch/mistral_7b_instruct" \
    --validation-data "$SCRATCH/dp-llm-experiments/official_data/validation.csv" \
    --benign-validation-data "$SCRATCH/dp-llm-experiments/official_data/frr_validation.csv" \
    --harmful-output-file "$EVAL_OUTPUT_ROOT/run_lr2e-05_lam0.1_eps-1_epoch1_harmful.csv" \
    --benign-output-file "$EVAL_OUTPUT_ROOT/run_lr2e-05_lam0.1_eps-1_epoch1_benign.csv" \
    --summary-file "$EVAL_SWEEP_SUMMARY" \
    --gen-batch-size 8 \
    --judge-batch-size 8
fi

# -------- Epoch 3 --------
if [ ! -d "${FINETUNED_BASE}_epoch3" ]; then
  echo "WARNING: adapter not found: ${FINETUNED_BASE}_epoch3 — skipping epoch 3"
else
  export WANDB_RUN_ID="115b74a1c8bd44438f37c49a30b72c83"
  export WANDB_RUN_NAME="eval_run_lr2e-05_lam0.1_eps-1_epoch3"
  python "$EVAL_PY" \
    --slug "run_lr2e-05_lam0.1_eps-1" \
    --epoch 3 \
    --lr 2e-05 \
    --lambda-val 0.1 \
    --epsilon -1.0 \
    --base-llm "/home/taegyoem/scratch/llama2_7b_chat_hf" \
    --resume-from "${FINETUNED_BASE}_epoch3" \
    --judge-path "/home/taegyoem/scratch/mistral_7b_instruct" \
    --validation-data "$SCRATCH/dp-llm-experiments/official_data/validation.csv" \
    --benign-validation-data "$SCRATCH/dp-llm-experiments/official_data/frr_validation.csv" \
    --harmful-output-file "$EVAL_OUTPUT_ROOT/run_lr2e-05_lam0.1_eps-1_epoch3_harmful.csv" \
    --benign-output-file "$EVAL_OUTPUT_ROOT/run_lr2e-05_lam0.1_eps-1_epoch3_benign.csv" \
    --summary-file "$EVAL_SWEEP_SUMMARY" \
    --gen-batch-size 8 \
    --judge-batch-size 8
fi

# -------- Epoch 5 --------
if [ ! -d "${FINETUNED_BASE}_epoch5" ]; then
  echo "WARNING: adapter not found: ${FINETUNED_BASE}_epoch5 — skipping epoch 5"
else
  export WANDB_RUN_ID="ac9846638be54ac5ab1d5a4fa5d20e4b"
  export WANDB_RUN_NAME="eval_run_lr2e-05_lam0.1_eps-1_epoch5"
  python "$EVAL_PY" \
    --slug "run_lr2e-05_lam0.1_eps-1" \
    --epoch 5 \
    --lr 2e-05 \
    --lambda-val 0.1 \
    --epsilon -1.0 \
    --base-llm "/home/taegyoem/scratch/llama2_7b_chat_hf" \
    --resume-from "${FINETUNED_BASE}_epoch5" \
    --judge-path "/home/taegyoem/scratch/mistral_7b_instruct" \
    --validation-data "$SCRATCH/dp-llm-experiments/official_data/validation.csv" \
    --benign-validation-data "$SCRATCH/dp-llm-experiments/official_data/frr_validation.csv" \
    --harmful-output-file "$EVAL_OUTPUT_ROOT/run_lr2e-05_lam0.1_eps-1_epoch5_harmful.csv" \
    --benign-output-file "$EVAL_OUTPUT_ROOT/run_lr2e-05_lam0.1_eps-1_epoch5_benign.csv" \
    --summary-file "$EVAL_SWEEP_SUMMARY" \
    --gen-batch-size 8 \
    --judge-batch-size 8
fi

echo "Eval done for run_lr2e-05_lam0.1_eps-1; summary appended to $EVAL_SWEEP_SUMMARY"

