# Jailbreak-R1 variants for the test set

Generate attack prompts with [yukiyounai/Jailbreak-R1](https://huggingface.co/yukiyounai/Jailbreak-R1)
([Guo et al., 2025](https://arxiv.org/abs/2506.00782)) for every `goal` in
`official/combined_test_dataset.csv` (1022 harmful behaviors).

## 1) Download the model (login node, once)

The HF repo is gated — accept the terms on the model page, then:

```bash
module load StdEnv/2023 python/3.11
source $SCRATCH/venv/nanogcg/bin/activate   # or any venv with `huggingface_hub`
huggingface-cli login
huggingface-cli download yukiyounai/Jailbreak-R1 --local-dir $SCRATCH/jailbreak_r1
```

## 2) Chunk the test goals

On Narval (or locally then rsync):

```bash
cd $SCRATCH/dp-llm-experiments
python helper_scripts/perturbation/prepare_jailbreak_r1_chunks.py \
  --input official/combined_test_dataset.csv \
  --out-dir $SCRATCH/dp-llm-experiments/official_data/jailbreak_r1_test \
  --chunk-size 20
# → 52 chunks (array 0-51). Check official_data/jailbreak_r1_test/manifest.txt
```

If you prepare chunks on your laptop, sync them:

```bash
rsync -av official_data/jailbreak_r1_test/ \
  narval:$SCRATCH/dp-llm-experiments/official_data/jailbreak_r1_test/
```

## 3) Submit the array

```bash
cd $SCRATCH/dp-llm-experiments
mkdir -p output
# If chunk count ≠ 52, override the array, e.g. --array=0-51
sbatch helper_scripts/perturbation/jailbreak_r1_test.sh
```

Outputs: `$SCRATCH/dp-llm-experiments/official_data/jailbreak_r1_out/chunk_XX.csv`

## 4) Merge into the test CSV

```bash
python helper_scripts/perturbation/merge_jailbreak_r1_outputs.py \
  --chunks-dir $SCRATCH/dp-llm-experiments/official_data/jailbreak_r1_out \
  --base-csv official/combined_test_dataset.csv \
  --out-dir official/jailbreak_r1
```

You get:
- `official/jailbreak_r1/jailbreak_r1_variants.csv`
- `official/jailbreak_r1/combined_test_with_jailbreak_r1.csv` (adds `Jailbreak-R1 Variant`)

## Notes

- Generator only — no target-model responses here. To ASR-eval, run your usual
  generation + HarmBench classifier on the `Jailbreak-R1 Variant` column.
- Parsing looks for `<attack>...</attack>` (retries twice if missing).
- `flash_attention_2` is optional (`--flash-attn`); default path works without it.
