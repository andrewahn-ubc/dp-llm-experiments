# PyRIT held-out attacks on Rorqual (Llama-3 family + DELMAN)

Multi-turn [PyRIT](https://github.com/microsoft/PyRIT) `RedTeamingAttack` against
the harmful test set (`combined_test_dataset.csv`), with:

| Role | Model | Device |
|------|--------|--------|
| Attacker + judge | `$SCRATCH/qwen3-30b-a3b-instruct-2507` | `cuda:0` |
| Target | base / MixAT / DOOR / DCL / DELMAN | `cuda:1` |

Targets (via array):

| Tag | Path |
|-----|------|
| `base` | `$SCRATCH/llama_3_8b_instruct` |
| `mixat` | `$SCRATCH/mixat` |
| `door` | `$SCRATCH/door` |
| `dcl_lam3_eps1` | base + LoRA `$SCRATCH/dp-llm-sweep/l3_run_lr2e-05_lam3_eps1_finetuned_llm_epoch${EPOCH}` |
| `delman` | `$SCRATCH/delman_llama31_8b_instruct` (edited from `$SCRATCH/llama31_8b_instruct`) |

Each job needs **2× H100** (attacker ~30B MoE + 8B target).

## 0) DELMAN edited weights (once)

See `experiments/delman/README.md`. Short version:

```bash
# cov stats in $SCRATCH/DELMAN/data/stats, offset=2 in rome/repr_tools.py
sbatch experiments/delman/run_edit_llama31.sh
# → $SCRATCH/delman_llama31_8b_instruct
```

## 1) Setup (login node, once)

```bash
cd $SCRATCH/dp-llm-experiments
bash experiments/pyrit_rorqual/setup_env.sh

python helper_scripts/perturbation/prepare_jailbreak_r1_chunks.py \
  --input official_data/combined_test_dataset.csv \
  --out-dir official_data/pyrit_test \
  --chunk-size 20
```

## 2) Launch attacks (260 jobs = 5 models × 52 chunks)

```bash
cd $SCRATCH/dp-llm-experiments
mkdir -p output official_data/pyrit_out
EPOCH=2 sbatch experiments/pyrit_rorqual/run_array.sh
```

Overrides: `ATTACKER_PATH`, `MIXAT_PATH`, `DOOR_PATH`, `DELMAN_PATH`, `BASE_L3`,
`CHECKPOINT_ROOT`, `MAX_TURNS`, `N_CHUNKS`, `VENV_ACTIVATE`.

If you change `--chunk-size`, set `#SBATCH --array=0-$((5*N_CHUNKS-1))`.

## 3) Merge + HarmBench ASR

```bash
for tag in base mixat door dcl_lam3_eps1 delman; do
  TARGET_TAG=$tag sbatch experiments/pyrit_rorqual/score_one.sh
done
```

Outputs under `official_data/pyrit_out/<tag>/` and `<tag>_harmful.csv`.

## Notes

- Target decoding is **greedy** with **no system prompt** (same protocol as `eval.py`).
- Reported ASR uses HarmBench with `{behavior}=goal` for the `pyrit` family.
- DELMAN is Llama-**3.1**; other L3 baselines are Llama-**3**. Same chat-template family
  for generation; treat cross-version comparison carefully in writeups.
