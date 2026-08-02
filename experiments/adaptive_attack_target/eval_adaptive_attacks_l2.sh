#!/bin/bash
#SBATCH --job-name=adapt_eval_l2
#SBATCH --account=def-mijungp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=5:00:00
#SBATCH --output=output/adaptive_eval_l2_%j.out

# Eval defended Llama-2 (adaptive-attack target) on adaptive GCG/AutoDAN/PAIR.
# ASR only (skips FRR). HarmBench Mistral val classifier.
#
# From repo root on Narval:
#   mkdir -p output
#   # once: sync the combined CSV
#   rsync -av official/adaptive_llama2_combined_test.csv \
#     $SCRATCH/dp-llm-experiments/official_data/
#   git pull   # need eval.py with --skip-frr
#   sbatch experiments/adaptive_attack_target/eval_adaptive_attacks_l2.sh
#
# Overrides:
#   MODEL_PATH, HARMFUL_DATA, OUT_DIR, CLS_PATH, REPO_ROOT

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$SCRATCH/dp-llm-experiments}"
cd "$REPO_ROOT"
mkdir -p output

module load StdEnv/2023 python/3.11
# shellcheck disable=SC1090
source "${VENV_ACTIVATE:-$SCRATCH/venv/nanogcg/bin/activate}"

export TRANSFORMERS_CACHE="${SLURM_TMPDIR:-/tmp}/hf_cache"
export HF_HOME="${SLURM_TMPDIR:-/tmp}/hf_home"
mkdir -p "$TRANSFORMERS_CACHE" "$HF_HOME"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

MODEL_PATH="${MODEL_PATH:-$SCRATCH/merged_adaptive_l2_lam0.1_eps-0.5_ep5}"
HARMFUL_DATA="${HARMFUL_DATA:-$SCRATCH/dp-llm-experiments/official_data/adaptive_llama2_combined_test.csv}"
CLS_PATH="${CLS_PATH:-$SCRATCH/harmbench_mistral_val_cls}"
OUT_DIR="${OUT_DIR:-$SCRATCH/dp-llm-eval/adaptive_l2}"
RUN_TAG="${RUN_TAG:-adaptive_l2_lam0.1_eps-0.5_ep5}"
HARMFUL_OUT_STEM="${OUT_DIR}/${RUN_TAG}_harmful"

mkdir -p "$OUT_DIR"

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "ERROR: model not found: $MODEL_PATH" >&2
  exit 2
fi
if [[ ! -f "$HARMFUL_DATA" ]]; then
  echo "ERROR: missing harmful data: $HARMFUL_DATA" >&2
  echo "  rsync official/adaptive_llama2_combined_test.csv \\" >&2
  echo "    \$SCRATCH/dp-llm-experiments/official_data/" >&2
  exit 2
fi
if [[ ! -d "$CLS_PATH" ]]; then
  echo "ERROR: HarmBench classifier not found: $CLS_PATH" >&2
  exit 2
fi

echo "MODEL_PATH=$MODEL_PATH"
echo "HARMFUL_DATA=$HARMFUL_DATA"
echo "CLS_PATH=$CLS_PATH"
echo "HARMFUL_OUT=${HARMFUL_OUT_STEM}.csv"
echo

python "$REPO_ROOT/eval/eval.py" \
  --model-profile llama_2_7b_chat \
  --base-llm "$MODEL_PATH" \
  --eval-mode seen-family \
  --system-prompt-mode empty \
  --skip-frr \
  --validation-data "$HARMFUL_DATA" \
  --jailbreak-classifier-path "$CLS_PATH" \
  --harmful-output-file "$HARMFUL_OUT_STEM"

HARMFUL_CSV="${HARMFUL_OUT_STEM}.csv"
echo
echo "Wrote: $HARMFUL_CSV"

python - "$HARMFUL_CSV" <<'PY'
import sys
from pathlib import Path
import pandas as pd

path = Path(sys.argv[1])
df = pd.read_csv(path)
cols = {
    "gcg": "GCG Safety",
    "autodan": "AutoDAN Safety",
    "pair": "PAIR Safety",
}
print(f"n={len(df)}")
asrs = []
for fam, col in cols.items():
    if col not in df.columns:
        print(f"{fam}: missing column {col}")
        continue
    asr = float((df[col].astype(str).str.lower() == "unsafe").mean())
    asrs.append(asr)
    print(f"{fam}_asr={asr:.6f}")
if asrs:
    print(f"mean_asr={sum(asrs)/len(asrs):.6f}")
if "dataset" in df.columns:
    print("\nper-dataset mean ASR (avg over families present):")
    for ds, sub in df.groupby(df["dataset"].astype(str).str.lower()):
        rates = []
        for col in cols.values():
            if col in sub.columns:
                rates.append(float((sub[col].astype(str).str.lower() == "unsafe").mean()))
        if rates:
            print(f"  {ds}: {sum(rates)/len(rates):.6f}  (n={len(sub)})")
PY

echo "Done."
