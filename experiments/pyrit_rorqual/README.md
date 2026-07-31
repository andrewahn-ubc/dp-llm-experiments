# PyRIT held-out attacks on Rorqual (Llama-3 family)

Multi-turn [PyRIT](https://github.com/microsoft/PyRIT) `RedTeamingAttack` on a
**500-goal stratified subset**:

- **225** advbench + **225** harmbench + **50** jailbreakbench  
- `MAX_TURNS=3`  
- ~80 jobs (4 models × 20 chunks of 25), wall **3:30**

| Role | Model | Device |
|------|--------|--------|
| Attacker + judge | `$SCRATCH/qwen3-30b-a3b-instruct-2507` | `cuda:0` |
| Target | base / MixAT / DOOR / DCL | `cuda:1` |

| Tag | Path |
|-----|------|
| `base` | `$SCRATCH/llama_3_8b_instruct` |
| `mixat` | `$SCRATCH/mixat` |
| `door` | `$SCRATCH/door` |
| `dcl_lam3_eps1` | LoRA `…/l3_run_lr2e-05_lam3_eps1_finetuned_llm_epoch${EPOCH}` |

## Setup + submit

```bash
cd $SCRATCH/dp-llm-experiments
# setup loads gcc+arrow before activate (Alliance pyarrow) and Rust for base2048
bash experiments/pyrit_rorqual/setup_env.sh   # once
bash experiments/pyrit_rorqual/submit_all.sh
```

Full 1022-goal run: `FULL_TEST=1 bash experiments/pyrit_rorqual/submit_all.sh`

## Score

```bash
for tag in base mixat door dcl_lam3_eps1; do
  TARGET_TAG=$tag sbatch experiments/pyrit_rorqual/score_one.sh
done
```

Outputs under `official_data/pyrit_out/`.
