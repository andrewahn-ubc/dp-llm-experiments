# DELMAN (Llama-3.1-8B-Instruct) baseline

Edit `$SCRATCH/llama31_8b_instruct` with [DELMAN](https://github.com/wanglne/DELMAN),
then evaluate the edited weights as a full-HF baseline (same slot as MixAT/DOOR)
on **Jailbreak-R1** and **PyRIT**.

## Produce the edited model

```bash
# once
git clone https://github.com/wanglne/DELMAN.git $SCRATCH/DELMAN
# unpack your Llama-3.1 cov zip into:
mkdir -p $SCRATCH/DELMAN/data/stats && unzip … -d $SCRATCH/DELMAN/data/stats

# REQUIRED for Llama 3.1: in $SCRATCH/DELMAN/rome/repr_tools.py set
#   offset = 2

# Dedicated venv (do not reuse nanogcg — transformers pin conflicts with nanogcg).
# Load Arrow *before* activating so pyarrow is real, not the dummy wheel:
deactivate 2>/dev/null || true
module load StdEnv/2023 gcc arrow python/3.11
python -m venv $SCRATCH/venv/delman
source $SCRATCH/venv/delman/bin/activate
pip install --upgrade pip
pip install --no-index torch
pip install -r experiments/delman/requirements_cc.txt
# If you see: pyarrow has no attribute 'PyExtensionType'
#   → old datasets vs arrow>=21; upgrade:
#   pip install -U 'datasets>=2.20,<4'

cd $SCRATCH/dp-llm-experiments && mkdir -p output
sbatch experiments/delman/run_edit_llama31.sh   # defaults to $SCRATCH/venv/delman
# → $SCRATCH/delman_llama31_8b_instruct
```

Overrides: `BASE_L31`, `DELMAN_OUT`, `DELMAN_REPO`, `VENV_ACTIVATE`.

## Use as baseline

Both pipelines default `DELMAN_PATH=$SCRATCH/delman_llama31_8b_instruct`
(full HF dir, no LoRA).

- Jailbreak-R1: array task `delman` in `helper_scripts/perturbation/jailbreak_r1_heldout_eval_l3.sh`
- PyRIT: tag `delman` in `experiments/pyrit_rorqual/run_array.sh`
