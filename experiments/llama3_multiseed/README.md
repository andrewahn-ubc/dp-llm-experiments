# Llama-3 multi-seed seen / heldout tables

Train and evaluate **seeds 1 and 2** for the hyperparameter cells in the paper-style seen/heldout tables, then aggregate with **seed 0 = image numbers** (no retrain) into `mean±std` tables.

## Protocol

| Item | Value |
|------|--------|
| Model | `llama_3_8b_instruct` |
| lr | `2e-5` |
| Data | `official_data/llama3_train_plus_validation.csv` |
| Eval harmful / FRR | `llama3_test.csv` / `frr_text.csv` (or `frr_test.csv`) |
| Epochs | `--total-epochs 1` equivalent via **both halves** → evaluate `*_epoch2` |
| Checkpoints | `$SCRATCH/dp-llm-sweep/multiseed_l3/` |
| Eval outputs | `$SCRATCH/dp-llm-eval/llama3_multiseed/seed{S}/` |

## Models per seed (10)

**Seen:** DCL `(1,-1)`, DCL `(1,-0.5)`, Adv-SFT `lam0` pertlm  

**Heldout:** GCG `(1,-0.5)`, GCG `(3,1)`, AutoDAN `(1,-1)`, PAIR `(30,1)`, plus Adv-SFT × `{gcg,autodan,pair}`

Array index `0–9` = seed 1, `10–19` = seed 2 (same config order). List with:

```bash
python experiments/llama3_multiseed/configs.py
```

## Submit on Narval

Before submitting, confirm data paths exist (train CSV + `llama3_test.csv` + FRR CSV under `official_data/`):

```bash
ls official_data/llama3_train_plus_validation.csv \
   official_data/llama3_test.csv \
   official_data/frr_test.csv
# optional but recommended for per-benchmark ASR:
ls official_data/combined_test_dataset.csv || ls official/combined_test_dataset.csv
```

```bash
cd $SCRATCH/dp-llm-experiments
mkdir -p output

TRAIN_ID=$(sbatch --parsable experiments/llama3_multiseed/train_array.sh)
echo "train array job: $TRAIN_ID"

EVAL_ID=$(sbatch --parsable --dependency=afterok:${TRAIN_ID} \
  experiments/llama3_multiseed/eval_array.sh)
echo "eval array job: $EVAL_ID"

AGG_ID=$(sbatch --parsable --dependency=afterok:${EVAL_ID} \
  experiments/llama3_multiseed/aggregate.sh)
echo "aggregate job: $AGG_ID"
```

W&B is **off by default** in `train_array.sh` (set `ENABLE_WANDB=1` only if you need it).

Or step-by-step after each stage finishes:

```bash
sbatch experiments/llama3_multiseed/train_array.sh
# after training:
sbatch --dependency=afterok:<train_jobid> experiments/llama3_multiseed/eval_array.sh
# after eval:
sbatch experiments/llama3_multiseed/aggregate.sh
```

## Outputs

After aggregate:

```text
$SCRATCH/dp-llm-eval/llama3_multiseed/aggregate/
  seen_family_table_multiseed.md
  seen_family_table_multiseed.csv
  heldout_family_table_multiseed.md
  heldout_family_table_multiseed.csv
  cells_long.csv
```

- **All Attacks** = unweighted mean of GCG / AutoDAN / PAIR cells (same as the image).
- **Adv-SFT** rows are added; they have no seed-0 image values (`n≤2`).
- Std is sample standard deviation (`ddof=1`) over available seeds. If a seed-1/2 cell is missing, that cell reports `n<3`.

## Seed plumbing

Training uses `--seed` (python / numpy / torch + shuffle). Slugs look like:

```text
l3_seed1_run_lr2e-05_lam1_eps-0.5
heldout_gcg_l3_seed1_run_lr2e-05_lam1_eps-0.5
l3_seed1_run_lr2e-05_lam0_eps0_pertlm
```

`train/submit_wandb_sweep.py` also accepts `--seed` for the same naming.
