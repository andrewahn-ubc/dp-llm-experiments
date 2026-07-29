#!/bin/bash
#SBATCH --job-name=sw_heldout_gcg_l3_run_lr2e-05_lam0.1_eps-1
#SBATCH --account=def-mijungp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=3:00:00
#SBATCH --output=output/sweep_heldout_gcg_l3_run_lr2e-05_lam0.1_eps-1_%j.out

mkdir -p output

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

module load StdEnv/2023 python/3.11
source "$SCRATCH/venv/nanogcg/bin/activate"

# Node-local Hugging Face caches (same pattern as epoch*.sh)
export TRANSFORMERS_CACHE=$SLURM_TMPDIR/hf_cache
export HF_HOME=$SLURM_TMPDIR/hf_home
mkdir -p "$TRANSFORMERS_CACHE"
mkdir -p "$HF_HOME"

# Weights & Biases — offline on compute nodes; sync later from login / your laptop.
# IMPORTANT: keep WANDB_DIR on node-local fast disk ($SLURM_TMPDIR), NOT on Lustre
# ($SCRATCH). Lustre's file-metadata caching causes wandb's service-port file to
# show up after the 30s poll timeout under load, killing the job at wandb.init.
# We sync the offline run folder to a persistent dir at the end of the job.
export WANDB_MODE=offline
export WANDB_DISABLE_SERVICE=true
export WANDB_PROJECT="dp-llm-safety"
export WANDB_DIR_PERSISTENT="$SCRATCH/wandb_offline"
export WANDB_DIR="$SLURM_TMPDIR/wandb"
export WANDB_RUN_ID=5c65e5741f92469db4787f61e1f35542
export WANDB_RUN_NAME="heldout_gcg_l3_run_lr2e-05_lam0.1_eps-1"

export FINETUNED_BASE="$SCRATCH/dp-llm-sweep/heldout_gcg_l3_run_lr2e-05_lam0.1_eps-1_finetuned_llm"
export TRAIN_PY="$SCRATCH/dp-llm-experiments/train/train.py"
mkdir -p "$(dirname "$FINETUNED_BASE")"
mkdir -p "$WANDB_DIR" "$WANDB_DIR_PERSISTENT"
# Copy any wandb runtime artifacts back to persistent storage on EXIT,
# even if the job is killed or fails partway through.
trap 'cp -r "$WANDB_DIR"/* "$WANDB_DIR_PERSISTENT/" 2>/dev/null || true' EXIT

# Outer step 1 (of logical 1) - training
python "$TRAIN_PY" \
    --eval-mode "unseen-family" \
    --unseen-family "gcg" \
    --system-prompt-mode "empty" \
    --lm-loss-input "clean" \
    --model-profile "llama_3_8b_instruct" \
    --finetuned-llm-path "$FINETUNED_BASE" \
    --training-data "$SCRATCH/dp-llm-experiments/official_data/llama3_train.csv" \
    --lr 2e-05 \
    --lambda-val 0.1 \
    --epsilon -1.0 \
    --lora-rank 8 \
    --total-epochs 1 \
    --start-epoch 1

echo "Latest checkpoint from this job: ${FINETUNED_BASE}_epoch1"
echo "Train-only job; run eval/test_eval_matrix.py for test metrics."

