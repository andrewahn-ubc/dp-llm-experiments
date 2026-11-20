# Llama-2 squared-consistency + benign-LM sweep

This experiment implements

\[
L = L_{\text{harmful LM}}
  + \lambda\,[\max(C_{\text{pert}}-e^\epsilon C_{\text{clean}},0)]^2
  + \beta L_{\text{benign LM}}.
\]

It fixes the target model to Llama-2-7B-Chat, uses a soft rollout of `T=3`,
trains one validation-search seed (`0`), evaluates both seen-family and each
held-out attack family, and leaves hyperparameter selection to the researcher.
The optional final pipeline retrains one selected cell with seeds `0,1,2`.

This changes the **penalty shape**, not the critic-input construction. The
training constraint still compares the stored clean prompt/response critic
score with the adversarial prompt plus a `T=3` differentiable rollout. It is
not the separate same-text embedding-space diagnostic discussed elsewhere.

## The 100 configurations

The design avoids wasting cells on epsilon when lambda is zero:

- 20 controls: 4 learning rates × 5 benign weights at `lambda=0`, `epsilon=0`.
- 80 regularized cells: 4 learning rates × 5 positive lambdas × 4 epsilons;
  benign weight is assigned as a balanced fractional-factorial dimension.

Values:

- learning rate: `5e-6, 1e-5, 2e-5, 4e-5`
- lambda: `0, 0.1, 1, 3, 10, 30`
- epsilon for positive lambda: `-1, -0.5, 0, 0.5`
- benign weight beta: `0, 0.1, 0.3, 1, 3`

Every nonzero-lambda beta occurs exactly 16 times. `configs.py` is the source of
truth and the generated `configs.tsv` records the exact cells.

## 1. Prepare the Nectar prompt/response pairs

Nectar already contains responses tagged `llama-2-7b-chat`; no response
generation is needed. Its `good_natured` label is an approximate GPT-4 label,
so the preparation script also removes default/refusal responses, duplicates,
evaluation overlap, and long examples. Evaluation decontamination includes
normalized exact matching plus indexed 5-word-shingle Jaccard/containment
checks, so near-copies and clean goals embedded in longer wrappers are removed.
The dataset card has license/usage caveats; keep the downloaded and derived
data private and do not commit it.

On your laptop, from a networked shell:

```bash
python -m pip install -U "huggingface_hub[cli]" pyarrow pandas
mkdir -p "$HOME/Downloads/nectar"
hf download berkeley-nest/Nectar data/rlaif.parquet \
  --repo-type dataset \
  --revision 3c6b4c4 \
  --local-dir "$HOME/Downloads/nectar"
```

Then create 10,000 decontaminated candidates. Replace `REPO` with the local
path to this repository; local data in this checkout is currently under
`official/`, while Narval uses `official_data/`.

```bash
REPO="/path/to/dp-llm-experiments"
python "$REPO/experiments/llama2_benign_sweep/prepare_nectar.py" extract \
  --input-parquet "$HOME/Downloads/nectar/data/rlaif.parquet" \
  --output "$HOME/Downloads/nectar/nectar_candidates.csv" \
  --exclude "$REPO/official/validation.csv" \
  --exclude "$REPO/official/combined_test_dataset.csv" \
  --exclude "$REPO/official/frr_validation.csv" \
  --exclude "$REPO/official/frr_test.csv"
```

This also writes `nectar_candidates.csv.manifest.json` with the pinned revision,
filter counts, input hashes, and output hash.

## 2. Transfer the candidates with Globus

On the Narval login node, first create the destination and print its exact path:

```bash
mkdir -p "$SCRATCH/dp-llm-experiments/official_data"
readlink -f "$SCRATCH/dp-llm-experiments/official_data"
```

In Globus File Manager:

1. Run Globus Connect Personal on the laptop and select its personal collection
   in one panel.
2. Select **Compute Canada - Narval** in the other panel and authenticate using
   **Digital Research Alliance of Canada**.
3. In the Narval panel, paste the absolute directory printed by `readlink`.
4. Transfer both `nectar_candidates.csv` and its manifest. Leave checksum
   verification enabled.

Wait until the Globus task reports **SUCCEEDED** before running `finalize`.

Now apply the exact tokenizer limits on the Narval login node. This loads only
tokenizer files and does not use a GPU:

```bash
cd "$SCRATCH/dp-llm-experiments"
source "$SCRATCH/venv/nanogcg/bin/activate"
python -m experiments.llama2_benign_sweep.prepare_nectar finalize \
  --input "$SCRATCH/dp-llm-experiments/official_data/nectar_candidates.csv" \
  --tokenizer "$SCRATCH/llama2_7b_chat_hf" \
  --output-dir "$SCRATCH/dp-llm-experiments/official_data"
cd "$SCRATCH/dp-llm-experiments/official_data"
sha256sum -c nectar_benign.sha256
```

This produces nested datasets:

- `nectar_benign_train.csv`: 2,489 pairs for validation-search training.
- `nectar_benign_train_plus_validation.csv`: 2,996 pairs for final retraining.

## 3. Smoke test, then submit the sweep

Every GPU job has a hard `00:55:00` wall limit. The historical Llama-2 exposure
is preserved: five full training passes, each split into eight row-shard array
jobs with `aftercorr` dependencies (40 sequential steps per logical model).
Both harmful validation and FRR evaluation are split into two shards. If a
smoke-test shard approaches 55 minutes, increase `--stages` (do not increase
the wall time).

For FRR validation, `vanilla_benign` rows use `Original Prompt` and
`adversarial_benign` rows use their `adversarial` rewrite. Every evaluation CSV
is accepted as complete only after its row count, labels, and checksum pass; a
sidecar `.complete.json` prevents a timed-out partial CSV from being scored.

Pull the pushed implementation and verify the required environment first:

```bash
cd "$SCRATCH/dp-llm-experiments"
git pull --ff-only
test -f "$SCRATCH/venv/nanogcg/bin/activate"
test -d "$SCRATCH/llama2_7b_chat_hf"
test -d "$SCRATCH/llama_guard_7b"
test -d "$SCRATCH/mistral_7b_instruct"
test -d "$SCRATCH/harmbench_mistral_val_cls"
test -f "$SCRATCH/dp-llm-experiments/official_data/train.csv"
test -f "$SCRATCH/dp-llm-experiments/official_data/validation.csv"
test -f "$SCRATCH/dp-llm-experiments/official_data/frr_validation.csv"
test -f "$SCRATCH/dp-llm-experiments/official_data/nectar_benign_train.csv"
```

All `test` commands should return silently with status 0.

First inspect one cell in a separate run root:

```bash
cd "$SCRATCH/dp-llm-experiments"
python -m experiments.llama2_benign_sweep.pipeline submit-sweep \
  --config-ids 21 \
  --run-root "$SCRATCH/dcl_l2_benign_smoke" \
  --dry-run
```

Configuration 21 deliberately has both a positive consistency weight and a
positive benign-LM weight, so this smoke test exercises every new loss path.

Submit that smoke cell if the commands look right:

```bash
python -m experiments.llama2_benign_sweep.pipeline submit-sweep \
  --config-ids 21 \
  --run-root "$SCRATCH/dcl_l2_benign_smoke"
```

After it completes successfully, the full experiment is one command:

```bash
cd "$SCRATCH/dp-llm-experiments"
python -m experiments.llama2_benign_sweep.pipeline submit-sweep
```

This submits 400 logical training runs: 100 configurations × seen training plus
three held-out-family training modes. Each performs five passes × eight
resumable row shards. Evaluation runs one attack family per task and produces
per-benchmark as well as aggregate metrics. Arrays are concurrency-capped at 32
and use allocation `def-mijungp`.

The complete sweep executes 16,000 training array tasks over time, but the
launcher never queues them all at once. It partitions the sweep into five waves
of 20 configurations and queues only one eight-shard epoch for a wave at a
time. Small dependent controller jobs advance the five epochs, validation, and
then the next wave; the final controller submits the report. At the defaults,
at most 641 training tasks or 401 evaluation tasks are submitted at one time,
below Narval's 1,000-job per-user pending/running limit. The single command
above starts the entire chain.

To avoid roughly a terabyte of redundant optimizer checkpoints, each successful
step validates its new checkpoint and then removes only the immediately
superseded step for that same model. The final step is retained. Set
`RETAIN_INTERMEDIATE_CHECKPOINTS=1` before submission only if you deliberately
want all 40 checkpoints and have checked your scratch quota.

Monitor with:

```bash
squeue -u "$USER"
sacct -u "$USER" --starttime today --format=JobID,JobName,State,Elapsed,ExitCode
```

Controller logs are `output/l2b_control_*.out`. If a failed array stops the
dependency chain, inspect `sacct`, the relevant task log, and
`manifests/sweep/submission_events.jsonl`. After confirming no tasks from that
phase are still running, resubmit only the failed action. For training, replace
`N` and `E` with the affected wave and zero-based training epoch (`0` through
`4`); for evaluation, use `--action eval` and omit `--training-epoch`:

```bash
python -m experiments.llama2_benign_sweep.pipeline continue-wave \
  --state "$SCRATCH/dcl_l2_benign_sweep/manifests/sweep/pipeline_state.json" \
  --wave N \
  --action train \
  --training-epoch E \
  --resubmit
```

Workers skip complete checkpoints and CSV shards. Recorded continuations are
idempotent, so a requeued controller cannot silently submit a duplicate wave.

## 4. Sweep outputs

The report job runs only after every evaluation shard succeeds. Outputs are in:

```text
$SCRATCH/dcl_l2_benign_sweep/reports/sweep/
```

- `seed_summary.csv`: one row per configuration with seen/held-out ASR and FRR.
- `per_benchmark_metrics.csv`: tidy family × benchmark metrics and sample counts.
- `results_table.md`: compact 100-row selection table.
- `pareto_front.csv` / `.md`: non-dominated cells only; no winner is selected.
- `pareto_seen_heldout.png`: ASR–FRR tradeoffs.
- `per_benchmark_seen_vs_heldout.png`: benchmark-specific generalization.

The one-seed sweep is for model selection, not uncertainty claims.

## 5. Final three-seed evaluation after you choose

After choosing a configuration ID yourself, retrain on
`train_plus_validation.csv` and evaluate on the existing held-out test sets:

```bash
test -f "$SCRATCH/dp-llm-experiments/official_data/train_plus_validation.csv"
test -f "$SCRATCH/dp-llm-experiments/official_data/combined_test_dataset.csv"
test -f "$SCRATCH/dp-llm-experiments/official_data/frr_test.csv"
test -f "$SCRATCH/dp-llm-experiments/official_data/nectar_benign_train_plus_validation.csv"
```

```bash
cd "$SCRATCH/dp-llm-experiments"
python -m experiments.llama2_benign_sweep.pipeline submit-final --config-id 42
```

This creates 12 logical models (3 seeds × seen/three held-out training modes),
uses two ASR shards for the larger test set, and reports both seed-level values
and configuration-level mean/standard deviation. Its Pareto table and plots use
the three-seed means, with error bars, rather than treating individual seeds as
separate candidate models. It does not alter or overwrite the sweep results.
