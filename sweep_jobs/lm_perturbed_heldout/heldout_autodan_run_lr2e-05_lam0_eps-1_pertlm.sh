#!/bin/bash
#SBATCH --job-name=sw_heldout_autodan_run_lr2e-05_lam0_eps-1_pertlm
#SBATCH --account=rrg-mijungp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=output/sweep_heldout_autodan_run_lr2e-05_lam0_eps-1_pertlm_%j.out

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
export WANDB_PROJECT="dp-llm-safety"
export WANDB_DIR_PERSISTENT="$SCRATCH/wandb_offline"
export WANDB_DIR="$SLURM_TMPDIR/wandb"
export WANDB_RUN_ID=d62ccae123fe440599328720c6c969d3
export WANDB_RUN_NAME="heldout_autodan_run_lr2e-05_lam0_eps-1_pertlm"

export FINETUNED_BASE="$SCRATCH/dp-llm-sweep/heldout_autodan_run_lr2e-05_lam0_eps-1_pertlm_finetuned_llm"
export TRAIN_PY="$SCRATCH/dp-llm-experiments/train/train.py"
export EVAL_PY="$SCRATCH/dp-llm-experiments/eval/eval.py"
export EVAL_OUT_DIR="$SCRATCH/dp-llm-sweep/eval_outputs"
mkdir -p "$(dirname "$FINETUNED_BASE")"
mkdir -p "$WANDB_DIR" "$WANDB_DIR_PERSISTENT"
mkdir -p "$EVAL_OUT_DIR"

# Copy any wandb runtime artifacts back to persistent storage on EXIT,
# even if the job is killed or fails partway through.
trap 'cp -r "$WANDB_DIR"/* "$WANDB_DIR_PERSISTENT/" 2>/dev/null || true' EXIT

# ----------------------------
# FRR input prep (eval.py expects an 'Original Prompt' column)
# ----------------------------
BENIGN_TMP="${SLURM_TMPDIR}/frr_eval_input_heldout_autodan_run_lr2e-05_lam0_eps-1_pertlm.csv"
python - "$SCRATCH/dp-llm-experiments/official_data/frr_test.csv" "$BENIGN_TMP" <<'PY'
import sys
import pandas as pd
src, dst = sys.argv[1:3]
df = pd.read_csv(src)
if 'Original Prompt' in df.columns:
    df.to_csv(dst, index=False)
    print(f'[sweep_eval] benign data already has Original Prompt -> {dst}')
    sys.exit(0)
candidates = ['adversarial','Adversarial','goal','Goal','prompt','Prompt','original_prompt','instruction','Instruction']
src_col = next((c for c in candidates if c in df.columns), None)
if src_col is None:
    raise ValueError(f'Could not build Original Prompt column from {list(df.columns)}')
df = df.copy(); df['Original Prompt'] = df[src_col].astype(str)
df.to_csv(dst, index=False)
print(f'[sweep_eval] mapped {src_col!r} -> Original Prompt -> {dst}')
PY

# Epoch 1 - training
python "$TRAIN_PY" \
    --eval-mode "unseen-family" \
    --unseen-family "autodan" \
    --system-prompt-mode "empty" \
    --lm-loss-input "perturbed" \
    --finetuned-llm-path "$FINETUNED_BASE" \
    --training-data "$SCRATCH/dp-llm-experiments/official_data/train_plus_validation.csv" \
    --lr 2e-05 \
    --lambda-val 0.0 \
    --epsilon -1.0 \
    --lora-rank 8 \
    --total-epochs 5 \
    --start-epoch 1

# Epoch 2 - training
python "$TRAIN_PY" \
    --eval-mode "unseen-family" \
    --unseen-family "autodan" \
    --system-prompt-mode "empty" \
    --lm-loss-input "perturbed" \
    --finetuned-llm-path "$FINETUNED_BASE" \
    --training-data "$SCRATCH/dp-llm-experiments/official_data/train_plus_validation.csv" \
    --lr 2e-05 \
    --lambda-val 0.0 \
    --epsilon -1.0 \
    --lora-rank 8 \
    --total-epochs 5 \
    --resume-from "${FINETUNED_BASE}_epoch1" \
    --start-epoch 2

# Epoch 3 - training
python "$TRAIN_PY" \
    --eval-mode "unseen-family" \
    --unseen-family "autodan" \
    --system-prompt-mode "empty" \
    --lm-loss-input "perturbed" \
    --finetuned-llm-path "$FINETUNED_BASE" \
    --training-data "$SCRATCH/dp-llm-experiments/official_data/train_plus_validation.csv" \
    --lr 2e-05 \
    --lambda-val 0.0 \
    --epsilon -1.0 \
    --lora-rank 8 \
    --total-epochs 5 \
    --resume-from "${FINETUNED_BASE}_epoch2" \
    --start-epoch 3

# Epoch 4 - training
python "$TRAIN_PY" \
    --eval-mode "unseen-family" \
    --unseen-family "autodan" \
    --system-prompt-mode "empty" \
    --lm-loss-input "perturbed" \
    --finetuned-llm-path "$FINETUNED_BASE" \
    --training-data "$SCRATCH/dp-llm-experiments/official_data/train_plus_validation.csv" \
    --lr 2e-05 \
    --lambda-val 0.0 \
    --epsilon -1.0 \
    --lora-rank 8 \
    --total-epochs 5 \
    --resume-from "${FINETUNED_BASE}_epoch3" \
    --start-epoch 4

# Epoch 5 - training + eval
python "$TRAIN_PY" \
    --eval-mode "unseen-family" \
    --unseen-family "autodan" \
    --system-prompt-mode "empty" \
    --lm-loss-input "perturbed" \
    --finetuned-llm-path "$FINETUNED_BASE" \
    --training-data "$SCRATCH/dp-llm-experiments/official_data/train_plus_validation.csv" \
    --lr 2e-05 \
    --lambda-val 0.0 \
    --epsilon -1.0 \
    --lora-rank 8 \
    --total-epochs 5 \
    --resume-from "${FINETUNED_BASE}_epoch4" \
    --start-epoch 5

# ----------------------------
# Epoch 5 eval (Protocol-Undefended)
# ----------------------------
CKPT_EP5="${FINETUNED_BASE}_epoch5"
HARMFUL_OUT_STEM_EP5="${EVAL_OUT_DIR}/heldout_autodan_run_lr2e-05_lam0_eps-1_pertlm_epoch5_val_output"
BENIGN_OUT_STEM_EP5="${EVAL_OUT_DIR}/heldout_autodan_run_lr2e-05_lam0_eps-1_pertlm_epoch5_frr_output"
METRICS_OUT_EP5="${EVAL_OUT_DIR}/heldout_autodan_run_lr2e-05_lam0_eps-1_pertlm_epoch5_metrics.tsv"

python "$EVAL_PY" \
    --eval-mode "seen-family" \
    --system-prompt-mode "empty" \
    --resume-from "$CKPT_EP5" \
    --validation-data "$SCRATCH/dp-llm-experiments/official_data/combined_test_dataset.csv" \
    --benign-validation-data "$BENIGN_TMP" \
    --harmful-output-file "$HARMFUL_OUT_STEM_EP5" \
    --benign-output-file "$BENIGN_OUT_STEM_EP5"

# Aggregate per-family ASR + FRR for epoch 5 into a single TSV.
python - "${HARMFUL_OUT_STEM_EP5}.csv" "${BENIGN_OUT_STEM_EP5}.csv" "$METRICS_OUT_EP5" \
    "heldout_autodan_run_lr2e-05_lam0_eps-1_pertlm" "2e-05" "0.0" "-1.0" "empty" "perturbed" "5" <<'PY'
import sys
import pandas as pd
harmful_csv, benign_csv, metrics_tsv, slug, lr, lam, eps, mode, lm_loss_input, epoch = sys.argv[1:11]
h = pd.read_csv(harmful_csv); b = pd.read_csv(benign_csv)
def mu(df, col): return float((df[col].astype(str).str.lower()=='unsafe').mean()) if col in df.columns else None
gcg = mu(h, 'GCG Safety'); ad = mu(h, 'AutoDAN Safety'); pa = mu(h, 'PAIR Safety')
vals = [v for v in (gcg, ad, pa) if v is not None]
mean_asr = sum(vals)/len(vals) if vals else None
frr = float((b['Original Safety'].astype(str).str.lower()=='refusal').mean()) if 'Original Safety' in b.columns else None
rows = [('slug',slug),('epoch',epoch),('lr',lr),('lambda',lam),('epsilon',eps),
        ('system_prompt_mode',mode),('lm_loss_input',lm_loss_input),
        ('gcg_asr',gcg),('autodan_asr',ad),('pair_asr',pa),('mean_asr',mean_asr),('frr',frr),
        ('n_harmful',len(h)),('n_benign',len(b))]
with open(metrics_tsv,'w',encoding='utf-8') as f:
    f.write('metric\tvalue\n')
    for k,v in rows: f.write(f'{k}\t{v}\n')
print(f'Wrote {metrics_tsv}')
print(f'epoch={epoch} mean_asr={mean_asr}, frr={frr}')
PY

echo "Final adapter (latest): ${FINETUNED_BASE}_epoch5"
echo "Intermediate checkpoints kept: ${FINETUNED_BASE}_epoch1 ... _epoch5"
echo "Eval epochs: 5"
echo "Per-epoch metrics in: ${EVAL_OUT_DIR}/heldout_autodan_run_lr2e-05_lam0_eps-1_pertlm_epoch<N>_metrics.tsv"

