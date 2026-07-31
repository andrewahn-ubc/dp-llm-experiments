# Jailbreak-R1 variants for the test set

Generate attack prompts with [yukiyounai/Jailbreak-R1](https://huggingface.co/yukiyounai/Jailbreak-R1)
([Guo et al., 2025](https://arxiv.org/abs/2506.00782)) for every `goal` in
`official_data/combined_test_dataset.csv` (1022 harmful behaviors).

On Narval the data dir is **`official_data/`** (not `official/`).

## 1) Download the model (login node, once)

The HF repo is gated — accept the terms on the model page, then:

```bash
module load StdEnv/2023 python/3.11
source $SCRATCH/venv/nanogcg/bin/activate   # or any venv with `huggingface_hub`
huggingface-cli login
huggingface-cli download yukiyounai/Jailbreak-R1 --local-dir $SCRATCH/jailbreak_r1
```

## 2) Chunk the test goals

```bash
cd $SCRATCH/dp-llm-experiments
python helper_scripts/perturbation/prepare_jailbreak_r1_chunks.py \
  --input official_data/combined_test_dataset.csv \
  --out-dir $SCRATCH/dp-llm-experiments/official_data/jailbreak_r1_test \
  --chunk-size 20
# → 52 chunks (array 0-51). Check official_data/jailbreak_r1_test/manifest.txt
```

## 3) Submit the array

```bash
cd $SCRATCH/dp-llm-experiments
mkdir -p output
sbatch helper_scripts/perturbation/jailbreak_r1_test.sh
```

Outputs: `$SCRATCH/dp-llm-experiments/official_data/jailbreak_r1_out/chunk_XX.csv`

## 4) Merge into the test CSV

```bash
python helper_scripts/perturbation/merge_jailbreak_r1_outputs.py \
  --chunks-dir $SCRATCH/dp-llm-experiments/official_data/jailbreak_r1_out \
  --base-csv official_data/combined_test_dataset.csv \
  --out-dir official_data/jailbreak_r1
```

You get:
- `official_data/jailbreak_r1/jailbreak_r1_variants.csv`
- `official_data/jailbreak_r1/combined_test_with_jailbreak_r1.csv` (adds `Jailbreak-R1 Variant`)

## 5) Held-out eval (Llama-3 baselines + DCL + DELMAN)

Scores ASR on `Jailbreak-R1 Variant` via `--unseen-family jailbreak_r1`, plus FRR
on `frr_test.csv`. One SLURM array task per model (6 jobs):

| # | Model | Path |
|---|--------|------|
| 0 | base L3-8B-Instruct | profile `$SCRATCH/llama_3_8b_instruct` |
| 1 | MixAT | `$SCRATCH/mixat` |
| 2 | DOOR | `$SCRATCH/door` |
| 3 | DCL λ=1, ε=-1 | `…/l3_run_lr2e-05_lam1_eps-1_finetuned_llm_epoch${EPOCH}` |
| 4 | DCL λ=3, ε=1 | `…/l3_run_lr2e-05_lam3_eps1_finetuned_llm_epoch${EPOCH}` |
| 5 | DELMAN (L3.1 edited) | `$SCRATCH/delman_llama31_8b_instruct` |

Produce DELMAN from `$SCRATCH/llama31_8b_instruct` first:
`experiments/delman/run_edit_llama31.sh` (see `experiments/delman/README.md`).

```bash
cd $SCRATCH/dp-llm-experiments
mkdir -p output
EPOCH=2 sbatch helper_scripts/perturbation/jailbreak_r1_heldout_eval_l3.sh
# overrides: MIXAT_PATH=... DOOR_PATH=... DELMAN_PATH=... EPOCH=1 OUT_DIR=...
```

Outputs: `$SCRATCH/dp-llm-eval/jailbreak_r1_heldout_l3/*_{harmful,benign,metrics}.*`

## Train (+ validation) set

Same pipeline on `official_data/train_plus_validation.csv` (~2996 rows → **150** chunks at size 20). Train CSVs use `Original Prompt` (mapped to `goal` automatically).

```bash
cd $SCRATCH/dp-llm-experiments

python helper_scripts/perturbation/prepare_jailbreak_r1_chunks.py \
  --input official_data/train_plus_validation.csv \
  --out-dir $SCRATCH/dp-llm-experiments/official_data/jailbreak_r1_train \
  --chunk-size 20
cat official_data/jailbreak_r1_train/manifest.txt   # expect array=0-149

mkdir -p output
sbatch helper_scripts/perturbation/jailbreak_r1_train.sh

# after array finishes:
python helper_scripts/perturbation/merge_jailbreak_r1_outputs.py \
  --chunks-dir $SCRATCH/dp-llm-experiments/official_data/jailbreak_r1_train_out \
  --base-csv official_data/train_plus_validation.csv \
  --out-dir official_data/jailbreak_r1
# → official_data/jailbreak_r1/train_plus_validation_with_jailbreak_r1.csv
```

For **train.csv only** (~2489 rows → 125 chunks):

```bash
python helper_scripts/perturbation/prepare_jailbreak_r1_chunks.py \
  --input official_data/train.csv \
  --out-dir $SCRATCH/dp-llm-experiments/official_data/jailbreak_r1_train_only \
  --chunk-size 20
DATA_ROOT=$SCRATCH/dp-llm-experiments/official_data/jailbreak_r1_train_only \
OUT_ROOT=$SCRATCH/dp-llm-experiments/official_data/jailbreak_r1_train_only_out \
  sbatch --array=0-124 helper_scripts/perturbation/jailbreak_r1_train.sh
```

## Notes

- Generator only — no target-model responses here. To ASR-eval, run your usual
  generation + HarmBench classifier on the `Jailbreak-R1 Variant` column.
- Parsing looks for `<attack>...</attack>` (retries twice if missing).
- `flash_attention_2` is optional (`--flash-attn`); default path works without it.
