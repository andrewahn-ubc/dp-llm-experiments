# Fact-Checking Experiment Runbook

Complete step-by-step guide: environment setup → debug run → overnight sweep.
All commands assume you are logged into the Narval login node unless noted.

---

## Prerequisites

- Repo cloned to `/home/ammany/repos/dp-llm-experiments`
- SLURM account: `aip-mijungp`
- GPU: L40S (`--gpus-per-node=l40s:1`)

Paths are set by `source env.sh fact-check` — check `fact_check/config.yaml` if
you need to change them. Nothing is hardcoded in the scripts.

---

## Part 1 — Environment Setup (login node, ~20 min)

Run once. Compute nodes have no internet, so everything must be downloaded here first.

### 1a. Create the virtualenv (one time only)

```bash
module load StdEnv/2023 gcc python/3.11 arrow/21.0.0

python -m venv $SCRATCH/venv/factcheck
source $SCRATCH/venv/factcheck/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
deactivate
```

### 1b. Download datasets and model weights

```bash
hostname   # must NOT start with "ng" (that's a compute node)

cd /home/ammany/repos/dp-llm-experiments
source env.sh fact-check

python fact_check/setup_fever.py
```

Downloads:
1. `copenlu/fever_gold_evidence` → `$FC_DATA_DIR/fever_train.csv` + `fever_val.csv`
2. `TalSchuster/fever_symmetric_v2` → `$FC_DATA_DIR/fever_symmetric.csv`
3. `bert-base-uncased` → `$FC_MODEL_ROOT/bert_base_uncased/`

Verify:
```bash
ls $FC_DATA_DIR
# fever_train.csv  fever_val.csv  fever_symmetric.csv

ls $FC_MODEL_ROOT/bert_base_uncased/
# config.json  tokenizer.json  pytorch_model.bin  vocab.txt  ...

wc -l $FC_DATA_DIR/fever_train.csv
# ~97,000 lines (SUPPORTS + REFUTES only)
```

**If FeverSymmetric download fails**, check your internet connection on the login
node. The script downloads directly from GitHub (no HuggingFace mirror exists).
Run manually if needed:
```bash
source env.sh fact-check
python - <<'PY'
import json, urllib.request, os
import pandas as pd
from pathlib import Path

base = "https://raw.githubusercontent.com/TalSchuster/FeverSymmetric/master/symmetric_v0.2"
rows = []
for url in [f"{base}/fever_symmetric_dev.jsonl", f"{base}/fever_symmetric_test.jsonl"]:
    with urllib.request.urlopen(url) as r:
        for line in r:
            ex = json.loads(line)
            if ex.get("gold_label") in ("SUPPORTS", "REFUTES"):
                rows.append({"claim": ex["claim"],
                             "evidence": " ".join(ex.get("evidence", [])),
                             "label": ex["gold_label"]})
out = Path(os.environ["FC_DATA_DIR"]) / "fever_symmetric.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print(f"Wrote {len(rows)} rows to {out}")
PY
```

---

## Part 2 — Generate Perturbed Training Data (login node, ~5 min)

```bash
cd /home/ammany/repos/dp-llm-experiments
source env.sh fact-check

python fact_check/perturb_fever.py

# Verify
wc -l $FC_DATA_DIR/fever_train_perturbed.csv
# expect ~3-4x the training rows
```

Perturbation families applied: `evidence_word_shuffle`, `evidence_sent_shuffle`,
`evidence_trunc`, `claim_synonym`. No-op perturbations (e.g. single-sentence
evidence for sent_shuffle) are skipped.

---

## Part 3 — Debug Run on an Interactive Node

### 3a. Request an allocation

```bash
salloc --account=aip-mijungp \
       --gpus-per-node=l40s:1 \
       --cpus-per-task=4 \
       --mem=16G \
       --time=1:00:00
```

Wait for the prompt to change to `[ammany@ngXXXX ~]$`.

### 3b. Activate the environment

```bash
cd /home/ammany/repos/dp-llm-experiments
source env.sh fact-check

python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"
```

### 3c. Smoke test via sweep_fever.py --interactive

The cleanest way to run a debug job on an interactive node is through the sweep
script with `--interactive`, which runs `train_fever.sh` directly via bash instead
of sbatch. This means all the rsync staging, venv setup, and env var checks happen
exactly as they would in a real SLURM job.

```bash
python fact_check/sweep_fever.py \
    --interactive \
    --lambdas 1.0 \
    --lrs 2e-5 \
    --epochs 1 \
    --batch-size 8
```

Output streams live to your terminal. Look for:
```
INFO | Starting training: epochs 1→1  λ=1.000  ε=0.000
INFO | ── Epoch 1 / 1 ──────────────────────────────
INFO | step=100  ce=0.XXXX  stab=0.XXXX  loss=0.XXXX  lr=2.00e-05
INFO | Epoch 1 complete — XXs  avg: loss=...  ce=...  stab=...
INFO | val_acc=0.XXXX  sym_acc=0.XXXX
```

If `stab=0.0000` for every step, the dataset pairing is broken. Check:
```bash
python -c "
import pandas as pd
df = pd.read_csv('$FC_DATA_DIR/fever_train_perturbed.csv')
print(df.columns.tolist())
print(df[['claim','original_claim','perturbation_type']].head(3))
print('NaN original_claim:', df['original_claim'].isna().sum())
"
```

If you see CUDA OOM, reduce `--batch-size` to 4.

### 3d. Exit the allocation

```bash
exit
```

---

## Part 4 — Submit the Overnight Sweep

From the **login node** after the interactive debug passes.

### 4a. Preview the grid

```bash
cd /home/ammany/repos/dp-llm-experiments
source env.sh fact-check
mkdir -p output

python fact_check/sweep_fever.py --lambdas 0.0 1.0 5.0 --lrs 2e-5 5e-5 --dry-run
```

### 4b. Submit

```bash
python fact_check/sweep_fever.py --lambdas 0.0 1.0 5.0 --lrs 2e-5 5e-5
```

You'll see a job list and a `[y/N]` confirmation before anything is submitted.

- `λ=0.0` — pure CE baseline (target: ~86.15% val / ~58.77% sym)
- `λ=1.0`, `λ=5.0` — light and strong regularization
- `lr=2e-5` — CrossAug standard; `5e-5` — faster convergence

### 4c. Monitor

```bash
squeue -u $USER

# Live SLURM output for a job:
tail -f output/fever_fever-train_JOBID.out

# Structured training log (timestamped, more detail):
tail -f $FC_OUTPUT_ROOT/<run_id>/<run_id>.log
```

### 4d. If a job gets interrupted

The SLURM header sends `SIGUSR1` 120s before wall time. Look for the resume hint
at the bottom of the `.out` file:

```
===================================================================
WALL TIME APPROACHING — printing resume command.

Latest checkpoint: .../fact_check_models/<run_id>/checkpoints/
  ls -td .../checkpoints/*/ | head -1

Re-submit with:
  RESUME_FROM="$(ls -td .../checkpoints/*/ | head -1)" \
  LAMBDA=... LR=... EPSILON=... EPOCHS=... \
  BATCH_SIZE=... LORA_RANK=... MODEL_NAME=... \
  sbatch fact_check/train_fever.sh
===================================================================
```

Resume manually:
```bash
source env.sh fact-check

RUN_ID="bert_base_uncased_lam1_eps0_lr2e-05_rank0"
LATEST=$(ls -td $FC_OUTPUT_ROOT/$RUN_ID/checkpoints/*/ | head -1)
echo "Resuming from: $LATEST"
cat $LATEST/checkpoint_meta.json   # shows epoch and global_step

RESUME_FROM="$LATEST" \
LAMBDA=1.0 LR=2e-05 EPSILON=0.0 EPOCHS=3 BATCH_SIZE=16 \
LORA_RANK=0 MODEL_NAME=bert_base_uncased \
sbatch fact_check/train_fever.sh
```

The resumed job picks up from the next epoch and appends to the existing
`training_log.json`. The scheduler is reconstructed from `global_step` — no
scheduler checkpoint needed.

### 4e. Sync W&B (offline runs only)

On clusters where W&B ran in offline mode (e.g. Narval), sync to the cloud from
the login node after jobs finish:

```bash
# Sync all unsynced runs under the output root (skips already-synced ones):
wandb sync --sync-all $FC_OUTPUT_ROOT/

# Or sync a specific run:
wandb sync $FC_OUTPUT_ROOT/<run_id>/wandb/offline-run-*
```

W&B marks runs as synced after the first upload, so re-running is safe.
On Vulcan, runs sync in real time and this step is not needed.

### 4f. Collect results

```bash
cd /home/ammany/repos/dp-llm-experiments
source env.sh fact-check

python fact_check/compare_runs.py

# run_id                                          λ     ε      lr  rank  val_acc  sym_acc
# bert_base_uncased_lam5_eps0_lr5e-05_rank0    5.00  0.00  5e-05     0   0.8489   0.6712
# bert_base_uncased_lam1_eps0_lr2e-05_rank0    1.00  0.00  2e-05     0   0.8601   0.6543
# bert_base_uncased_lam0_eps0_lr2e-05_rank0    0.00  0.00  2e-05     0   0.8618   0.5881
```

Target: `sym_acc > 65.30%` (beats PoE) with `val_acc > 85%`.

### 4g. Full sweep (after reviewing small sweep)

```bash
python fact_check/sweep_fever.py
# 12 jobs: λ ∈ {0.0, 0.1, 1.0, 5.0} × lr ∈ {1e-5, 2e-5, 5e-5}
```

---

## Troubleshooting

