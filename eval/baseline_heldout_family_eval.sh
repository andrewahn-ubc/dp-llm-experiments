#!/bin/bash
# Held-out *family* metrics for the **bare** base Llama-2-7B-Chat (no LoRA).
# Reuses eval/eval.py in unseen-family mode three times (gcg, autodan, pair) and
# the same aggregation logic as eval/heldout_family_eval_nosys.sh, but omits
# --resume-from so load_model uses the base weights only.
#
# Default protocol: --system-prompt-mode empty (no system role).
# To use a profile-defined system string instead (if you set one in model_profiles):
#   SYSTEM_PROMPT_MODE=default sbatch eval/baseline_heldout_family_eval.sh
#
#SBATCH --job-name=baseline_heldout
#SBATCH --account=rrg-mijungp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=16:00:00
#SBATCH --output=output/baseline_heldout_family_eval_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p output

# Load environment
module load StdEnv/2023 python/3.11
source "$SCRATCH/venv/nanogcg/bin/activate"

# Use node-local storage for Hugging Face
export TRANSFORMERS_CACHE="${SLURM_TMPDIR:-/tmp}/hf_cache"
export HF_HOME="${SLURM_TMPDIR:-/tmp}/hf_home"
mkdir -p "$TRANSFORMERS_CACHE" "$HF_HOME"

# ----------------------------
# Config: edit or override via env
# ----------------------------
REPO_ROOT="${REPO_ROOT:-$SCRATCH/dp-llm-experiments}"
EVAL_PY="${EVAL_PY:-$REPO_ROOT/eval/eval.py}"
EPOCH="${EPOCH:-1}"
SYSTEM_PROMPT_MODE="${SYSTEM_PROMPT_MODE:-empty}"

HARMFUL_DATA="${HARMFUL_DATA:-/home/taegyoem/scratch/dp-llm-experiments/official_data/combined_test_dataset.csv}"
BENIGN_DATA="${BENIGN_DATA:-/home/taegyoem/scratch/dp-llm-experiments/official_data/frr_test.csv}"
OUT_DIR="${OUT_DIR:-$SCRATCH/dp-llm-eval/final_eval}"
mkdir -p "$OUT_DIR"

FAMILIES=(gcg autodan pair)
RUN_TAG="baseline_heldout_family_${SYSTEM_PROMPT_MODE}_epoch${EPOCH}"
METRICS_OUT="${OUT_DIR}/${RUN_TAG}_metrics.tsv"

resolve_output() {
  local stem="$1"
  if [[ -f "${stem}.csv" ]]; then
    echo "${stem}.csv"
  elif [[ -f "${stem}.csv.csv" ]]; then
    echo "${stem}.csv.csv"
  else
    echo ""
  fi
}

echo "Baseline (no LoRA) held-out family eval — system_prompt_mode=${SYSTEM_PROMPT_MODE}"
echo "eval.py: $EVAL_PY"
echo "Harmful test set: $HARMFUL_DATA"
echo "Benign test set (FRR): $BENIGN_DATA"
echo "RUN_TAG=$RUN_TAG"

BENIGN_TMP="${SLURM_TMPDIR:-/tmp}/frr_eval_input_${RUN_TAG}.csv"
python - "$BENIGN_DATA" "$BENIGN_TMP" <<'PY'
import sys
import pandas as pd

src, dst = sys.argv[1:3]
df = pd.read_csv(src)

if "Original Prompt" in df.columns:
    df.to_csv(dst, index=False)
    print(f"[baseline_heldout_family_eval] benign data already has 'Original Prompt' -> {dst}")
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
print(f"[baseline_heldout_family_eval] mapped '{src_col}' -> 'Original Prompt' -> {dst}")
PY
BENIGN_DATA_FOR_EVAL="$BENIGN_TMP"

for FAMILY in "${FAMILIES[@]}"; do
  HARMFUL_OUT_STEM="${OUT_DIR}/${RUN_TAG}_${FAMILY}_harmful"
  BENIGN_OUT_STEM="${OUT_DIR}/${RUN_TAG}_${FAMILY}_benign"

  HARMFUL_OUT="$(resolve_output "$HARMFUL_OUT_STEM")"
  BENIGN_OUT="$(resolve_output "$BENIGN_OUT_STEM")"

  echo "---------------------------------------------"
  echo "Held-out family (unseen-family scoring): $FAMILY  (base model, no --resume-from)"

  if [[ -n "$HARMFUL_OUT" && -n "$BENIGN_OUT" ]]; then
    echo "Found existing eval outputs for $FAMILY; skipping eval.py:"
    echo "  $HARMFUL_OUT"
    echo "  $BENIGN_OUT"
  else
    HARMFUL_OUT="${HARMFUL_OUT_STEM}.csv"
    BENIGN_OUT="${BENIGN_OUT_STEM}.csv"
    python "$EVAL_PY" \
      --eval-mode "unseen-family" \
      --unseen-family "$FAMILY" \
      --system-prompt-mode "$SYSTEM_PROMPT_MODE" \
      --validation-data "$HARMFUL_DATA" \
      --benign-validation-data "$BENIGN_DATA_FOR_EVAL" \
      --harmful-output-file "$HARMFUL_OUT_STEM" \
      --benign-output-file "$BENIGN_OUT_STEM"
  fi
done

python - "$OUT_DIR" "$RUN_TAG" "$METRICS_OUT" "$SYSTEM_PROMPT_MODE" <<'PY'
import sys
from pathlib import Path
import pandas as pd

out_dir = Path(sys.argv[1])
run_tag = sys.argv[2]
metrics_tsv = Path(sys.argv[3])
system_prompt_mode = sys.argv[4]

families = ["gcg", "autodan", "pair"]
safety_col = {
    "gcg": "GCG Safety",
    "autodan": "AutoDAN Safety",
    "pair": "PAIR Safety",
}


def resolve(stem: Path) -> Path:
    """Match bash resolve_output: prefer stem.csv, fall back to stem.csv.csv."""
    p1 = Path(str(stem) + ".csv")
    p2 = Path(str(stem) + ".csv.csv")
    if p1.exists():
        return p1
    if p2.exists():
        return p2
    raise FileNotFoundError(f"Neither {p1} nor {p2} exists")


rows = [
    ("model", "baseline"),
    ("system_prompt_mode", system_prompt_mode),
]
heldout_asrs = []

for fam in families:
    harmful_stem = out_dir / f"{run_tag}_{fam}_harmful"
    benign_stem = out_dir / f"{run_tag}_{fam}_benign"
    harmful_csv = resolve(harmful_stem)
    benign_csv = resolve(benign_stem)
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

echo "Baseline held-out family eval complete."
echo "Metrics:"
echo "  $METRICS_OUT"
