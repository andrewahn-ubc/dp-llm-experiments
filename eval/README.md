# Sweep evaluation (Narval)

Pipeline to evaluate every LoRA adapter produced by `train/submit_wandb_sweep.py`
on the harmful validation set (GCG / AutoDAN / PAIR variants) and the benign
`frr_validation.csv`, using a single shared Mistral-7B-Instruct-v0.2 judge for
both HarmBench-style jailbreak classification and refusal classification.

- 75 hyperparameter configs × 3 epochs (1, 3, 5) = 225 evaluated checkpoints
- One SLURM job per config; each job runs the three epochs sequentially on
  one GPU so Slurm queueing / boot overhead stays bounded

## Files

- `eval_sweep.py` — per-checkpoint worker. Given a (slug, epoch, adapter)
  it (1) generates responses for the three JB families on `validation.csv`
  and for original prompts on `frr_validation.csv`, (2) loads Mistral-7B
  once and labels every row with the HarmBench classifier prompt (yes/no)
  and the refusal prompt (yes/no), and (3) appends one row to a shared
  summary TSV plus logs metrics to W&B (offline).
- `submit_eval_sweep.py` — mirrors `train/submit_wandb_sweep.py`'s
  hyperparameter grid so slugs match, and emits one SLURM script per config
  that runs `eval_sweep.py` for epochs 1/3/5.
- `aggregate_eval_results.py` — collapses the append-only summary TSV into
  tidy + pivot CSVs (ASR per family, mean ASR, refusal rate) and prints
  the best / worst configs at the latest epoch.

## Typical workflow on Narval

```bash
# 0. Make sure training has finished and checkpoints live at
#    $SCRATCH/dp-llm-sweep/<slug>_finetuned_llm_epoch{1,3,5}
ls "$SCRATCH/dp-llm-sweep" | head

# 1. Submit the 75 eval jobs (dry-run first to inspect generated scripts)
cd "$SCRATCH/dp-llm-experiments"
python eval/submit_eval_sweep.py --dry-run --limit 1
python eval/submit_eval_sweep.py   # submits all 75 jobs

# 2. Watch progress
squeue -u "$USER" | grep ev_
tail -f eval_sweep_jobs/output/eval_run_lr1e-05_lam1_eps0_*.out

# 3. Aggregate once jobs finish (safe to run repeatedly / while in flight)
python eval/aggregate_eval_results.py \
    --summary-file "$SCRATCH/dp-llm-eval/summary.tsv" \
    --out-dir     "$SCRATCH/dp-llm-eval"

# 4. Sync W&B offline runs from a machine with internet
wandb sync --sync-all "$SCRATCH/wandb_offline"
```

## Outputs per evaluated checkpoint

Located under `$SCRATCH/dp-llm-eval/`:

- `<slug>_epoch<N>_harmful.csv` — `validation.csv` + `GCG/AutoDAN/PAIR Response`
  + `GCG/AutoDAN/PAIR HarmBench` (`yes`/`no`).
- `<slug>_epoch<N>_benign.csv` — `frr_validation.csv` + `Original Response`
  + `Original Refusal` (`refusal`/`non-refusal`).
- `summary.tsv` — append-only, one row per (slug, epoch).
- `eval_summary.csv`, `eval_pivot_asr.csv`, `eval_pivot_refusal.csv`,
  `eval_pivot_{gcg,autodan,pair}_asr.csv` — produced by the aggregator.

## Classifier prompts

- **HarmBench (jailbreak success)** — canonical Llama-2 classifier template
  with behavior = `Original Prompt`, generation = `{family} Response`. Mistral
  chat format uses `[INST]...[/INST]` so the template is reused verbatim.
- **Refusal** — same prompt used by `eval/eval_helpers.py::classify_refusal`
  (asks "Yes/No" whether the response refuses the request).

## Tuning knobs

- `--gen-batch-size` / `--judge-batch-size` on `eval_sweep.py`
  (defaults 8). Bump on A100/H100 if memory allows.
- `--hours-per-epoch` on `submit_eval_sweep.py` (default 2 → 7h wall-time for
  three epochs + buffer). Adjust if your benign set or batch size grows.
- `--epochs 1 3 5` selects which training epochs to evaluate; pass
  `--epochs 5` etc. to re-evaluate a subset.
- `--overwrite` regenerates responses and labels even if output CSVs already
  exist (useful after fixing judge/generation bugs).

## Smoke test

```bash
# Evaluate just the first config, epoch 5 only, as a 1-job dry run
python eval/submit_eval_sweep.py \
    --limit 1 --epochs 5 \
    --script-dir /tmp/eval_smoke \
    --dry-run
```
