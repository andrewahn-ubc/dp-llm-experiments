#!/bin/bash
#SBATCH --job-name=heldout_family_eval
#SBATCH --account=rrg-mijungp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=16:00:00
#SBATCH --output=output/heldout_family_eval_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p output

# Load environment
module load StdEnv/2023 python/3.11
source "$SCRATCH/venv/nanogcg/bin/activate"

# Use node-local storage for Hugging Face
export TRANSFORMERS_CACHE="$SLURM_TMPDIR/hf_cache"
export HF_HOME="$SLURM_TMPDIR/hf_home"
mkdir -p "$TRANSFORMERS_CACHE" "$HF_HOME"

# ----------------------------
# Config: edit if needed
# ----------------------------
EPOCH="${EPOCH:-1}"
HARMFUL_DATA="${HARMFUL_DATA:-/home/taegyoem/scratch/dp-llm-experiments/official_data/combined_test_dataset.csv}"
BENIGN_DATA="${BENIGN_DATA:-/home/taegyoem/scratch/dp-llm-experiments/official_data/frr_test.csv}"
OUT_DIR="${OUT_DIR:-$SCRATCH/dp-llm-eval/final_eval}"
mkdir -p "$OUT_DIR"

# Mapping: for each family, evaluate the model that did NOT see that family.
FAMILIES=(gcg autodan pair)

RUN_TAG="heldout_family_epoch${EPOCH}"
METRICS_OUT="${OUT_DIR}/${RUN_TAG}_metrics.tsv"

echo "Evaluating held-out family models for epoch $EPOCH"
echo "Harmful test set: $HARMFUL_DATA"
echo "Benign test set (FRR): $BENIGN_DATA"

BENIGN_DATA_FOR_EVAL="$BENIGN_DATA"
BENIGN_TMP="${SLURM_TMPDIR}/frr_eval_input_heldout_epoch${EPOCH}.csv"
# eval.py expects an "Original Prompt" column for FRR.
python - "$BENIGN_DATA" "$BENIGN_TMP" <<'PY'
import sys
import pandas as pd

src, dst = sys.argv[1:3]
df = pd.read_csv(src)

if "Original Prompt" in df.columns:
    df.to_csv(dst, index=False)
    print(f"[heldout_family_eval] benign data already has 'Original Prompt' -> {dst}")
    sys.exit(0)

candidates = [
    "adversarial",
    "Adversarial",
    "goal",
    "Goal",
    "prompt",
    "Prompt",
    "original_prompt",
    "instruction",
    "Instruction",
]
src_col = next((c for c in candidates if c in df.columns), None)
if src_col is None:
    raise ValueError(
        "Could not build 'Original Prompt' column for FRR input. "
        f"Columns found: {list(df.columns)}"
    )

df = df.copy()
df["Original Prompt"] = df[src_col].astype(str)
df.to_csv(dst, index=False)
print(f"[heldout_family_eval] mapped '{src_col}' -> 'Original Prompt' -> {dst}")
PY
BENIGN_DATA_FOR_EVAL="$BENIGN_TMP"

for FAMILY in "${FAMILIES[@]}"; do
  MODEL_BASE_VAR="HELDOUT_${FAMILY^^}_MODEL_BASE"
  DEFAULT_MODEL_BASE="$SCRATCH/heldout_${FAMILY}_lr1e5_lam10_eps1_finetuned_llm"
  MODEL_BASE="${!MODEL_BASE_VAR:-$DEFAULT_MODEL_BASE}"
  MODEL_PATH="${MODEL_BASE}_epoch${EPOCH}"

  HARMFUL_OUT="${OUT_DIR}/${RUN_TAG}_${FAMILY}_harmful.csv"
  BENIGN_OUT="${OUT_DIR}/${RUN_TAG}_${FAMILY}_benign.csv"

  echo "---------------------------------------------"
  echo "Held-out family: $FAMILY"
  echo "Model path: $MODEL_PATH"

  python "$SCRATCH/dp-llm-experiments/eval/eval.py" \
    --eval-mode "unseen-family" \
    --unseen-family "$FAMILY" \
    --resume-from "$MODEL_PATH" \
    --validation-data "$HARMFUL_DATA" \
    --benign-validation-data "$BENIGN_DATA_FOR_EVAL" \
    --harmful-output-file "$HARMFUL_OUT" \
    --benign-output-file "$BENIGN_OUT"
done

# Compute ASR on each held-out family and FRR for each corresponding model.
python - "$OUT_DIR" "$RUN_TAG" "$METRICS_OUT" <<'PY'
import sys
from pathlib import Path
import pandas as pd

out_dir = Path(sys.argv[1])
run_tag = sys.argv[2]
metrics_tsv = Path(sys.argv[3])

families = ["gcg", "autodan", "pair"]
safety_col = {
    "gcg": "GCG Safety",
    "autodan": "AutoDAN Safety",
    "pair": "PAIR Safety",
}

rows = []
heldout_asrs = []

for fam in families:
    harmful_csv = out_dir / f"{run_tag}_{fam}_harmful.csv"
    benign_csv = out_dir / f"{run_tag}_{fam}_benign.csv"
    h = pd.read_csv(harmful_csv)
    b = pd.read_csv(benign_csv)

    asr = float((h[safety_col[fam]].astype(str).str.lower() == "unsafe").mean())
    frr = float((b["Original Safety"].astype(str).str.lower() == "refusal").mean())

    heldout_asrs.append(asr)
    rows.extend(
        [
            (f"{fam}_heldout_asr", asr),
            (f"{fam}_model_frr", frr),
            (f"{fam}_harmful_csv", str(harmful_csv)),
            (f"{fam}_benign_csv", str(benign_csv)),
            (f"{fam}_n_harmful", len(h)),
            (f"{fam}_n_benign", len(b)),
        ]
    )

rows.append(("heldout_mean_asr", sum(heldout_asrs) / len(heldout_asrs)))

with metrics_tsv.open("w", encoding="utf-8") as f:
    f.write("metric\tvalue\n")
    for k, v in rows:
        f.write(f"{k}\t{v}\n")

print(f"Wrote metrics: {metrics_tsv}")
for k, v in rows:
    if k.endswith("_asr") or k.endswith("_frr"):
        print(f"{k}={v:.6f}")
PY

echo "Held-out family eval complete."
echo "Metrics:"
echo "  $METRICS_OUT"
