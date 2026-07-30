# Soft-rollout length T sensitivity (Llama-2-7B-Chat)

Grid: **T ∈ {1, 3, 5, 10}** with fixed **λ=0.1, ε=-0.5, lr=2e-5, 1 epoch**,
`lm_loss=clean`, **seen-family** train + **seen ASR/FRR on the test set**.

Slugs / checkpoints:

```text
$SCRATCH/dp-llm-sweep/run_lr2e-05_lam0.1_eps-0.5_T{T}_finetuned_llm_epoch1
```

## On Rorqual (fresh env)

```bash
ssh rorqual.alliancecan.ca
cd $SCRATCH
# clone or rsync the repo if needed
cd $SCRATCH/dp-llm-experiments
git pull   # or rsync from your laptop

# 1) Fresh venv
bash experiments/rollout_T_sensitivity/setup_rorqual_env.sh
source $SCRATCH/venv/dp-llm-rorqual/bin/activate

# 2) Make sure base models + data exist under $SCRATCH (same layout as Narval):
#    $SCRATCH/llama2_7b_chat_hf
#    $SCRATCH/llama_guard_7b
#    $SCRATCH/mistral_7b_instruct          # refusal judge (eval)
#    $SCRATCH/harmbench_mistral_val_cls    # ASR judge (eval)
#    $SCRATCH/dp-llm-experiments/official_data/{train.csv,combined_test_dataset.csv,frr_test.csv}

# 3) Submit train (4 jobs) then seen-eval (4 jobs, afterok)
bash experiments/rollout_T_sensitivity/submit_all.sh
```

Or stepwise:

```bash
sbatch experiments/rollout_T_sensitivity/train_array.sh
# after train finishes:
sbatch experiments/rollout_T_sensitivity/eval_seen_array.sh
```

Override account if needed: `ACCOUNT=def-XXXX bash experiments/rollout_T_sensitivity/submit_all.sh`

## Train code flag

`train.py --rollout-length T` (alias `-T`) controls soft autoregressive rollout length
(default 5). Slugs include `_T{T}` when set via `make_run_slug(..., rollout_length=T)`.
