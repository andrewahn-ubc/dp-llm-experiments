#!/bin/bash
#SBATCH --job-name=baseline_eval
#SBATCH --account=rrg-mijungp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=7:00:00
#SBATCH --output=output/baseline_eval_%j.out

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
HARMFUL_DATA="${HARMFUL_DATA:-/home/taegyoem/scratch/dp-llm-experiments/official_data/combined_test_dataset.csv}"
BENIGN_DATA="${BENIGN_DATA:-/home/taegyoem/scratch/dp-llm-experiments/official_data/frr_test.csv}"

OUT_DIR="${OUT_DIR:-$SCRATCH/dp-llm-eval/final_eval}"
RUN_TAG="baseline"
mkdir -p "$OUT_DIR"

# NOTE: eval.py appends ".csv" to --harmful-output-file / --benign-output-file,
# so we keep these as stems (no extension) and resolve the actual file path
# below, supporting both the correct "stem.csv" form and the legacy
# "stem.csv.csv" form left over from earlier runs.
HARMFUL_OUT_STEM="${OUT_DIR}/${RUN_TAG}_harmful"
BENIGN_OUT_STEM="${OUT_DIR}/${RUN_TAG}_benign"
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

echo "Evaluating baseline model (no LoRA adapter)"
echo "Harmful test set: $HARMFUL_DATA"
echo "Benign test set (FRR): $BENIGN_DATA"

# If the eval.py outputs already exist (from a previous run), reuse them and
# skip straight to the metrics step. Otherwise run eval.py.
HARMFUL_OUT="$(resolve_output "$HARMFUL_OUT_STEM")"
BENIGN_OUT="$(resolve_output "$BENIGN_OUT_STEM")"

if [[ -n "$HARMFUL_OUT" && -n "$BENIGN_OUT" ]]; then
  echo "Found existing eval outputs; skipping eval.py:"
  echo "  $HARMFUL_OUT"
  echo "  $BENIGN_OUT"
  SKIP_EVAL=1
else
  SKIP_EVAL=0
  HARMFUL_OUT="${HARMFUL_OUT_STEM}.csv"
  BENIGN_OUT="${BENIGN_OUT_STEM}.csv"
fi

if [[ "$SKIP_EVAL" -eq 0 ]]; then
  BENIGN_DATA_FOR_EVAL="$BENIGN_DATA"
  BENIGN_TMP="${SLURM_TMPDIR}/frr_eval_input_baseline.csv"
  # eval.py expects an "Original Prompt" column for FRR.
  python - "$BENIGN_DATA" "$BENIGN_TMP" <<'PY'
import sys
import pandas as pd

src, dst = sys.argv[1:3]
df = pd.read_csv(src)

if "Original Prompt" in df.columns:
    df.to_csv(dst, index=False)
    print(f"[baseline_eval] benign data already has 'Original Prompt' -> {dst}")
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
print(f"[baseline_eval] mapped '{src_col}' -> 'Original Prompt' -> {dst}")
PY
  BENIGN_DATA_FOR_EVAL="$BENIGN_TMP"

  # Note: no --resume-from flag, so eval.py loads the bare base model (baseline).
  # Pass the stems (without .csv); eval.py appends ".csv" itself.
  python "$SCRATCH/dp-llm-experiments/eval/eval.py" \
    --eval-mode "seen-family" \
    --validation-data "$HARMFUL_DATA" \
    --benign-validation-data "$BENIGN_DATA_FOR_EVAL" \
    --harmful-output-file "$HARMFUL_OUT_STEM" \
    --benign-output-file "$BENIGN_OUT_STEM"
fi

# Compute per-family ASR, mean ASR (3 variants), and FRR.
python - "$HARMFUL_OUT" "$BENIGN_OUT" "$METRICS_OUT" <<'PY'
import sys
import pandas as pd

harmful_csv, benign_csv, metrics_tsv = sys.argv[1:4]
h = pd.read_csv(harmful_csv)
b = pd.read_csv(benign_csv)

def mean_is_unsafe(df, col):
    return float((df[col].astype(str).str.lower() == "unsafe").mean())

gcg_asr = mean_is_unsafe(h, "GCG Safety")
autodan_asr = mean_is_unsafe(h, "AutoDAN Safety")
pair_asr = mean_is_unsafe(h, "PAIR Safety")
mean_asr = (gcg_asr + autodan_asr + pair_asr) / 3.0
frr = float((b["Original Safety"].astype(str).str.lower() == "refusal").mean())

rows = [
    ("model", "baseline"),
    ("harmful_csv", harmful_csv),
    ("benign_csv", benign_csv),
    ("gcg_asr", gcg_asr),
    ("autodan_asr", autodan_asr),
    ("pair_asr", pair_asr),
    ("mean_asr", mean_asr),
    ("frr", frr),
    ("n_harmful", len(h)),
    ("n_benign", len(b)),
]

with open(metrics_tsv, "w", encoding="utf-8") as f:
    f.write("metric\tvalue\n")
    for k, v in rows:
        f.write(f"{k}\t{v}\n")

print(f"Wrote metrics: {metrics_tsv}")
print(f"gcg_asr={gcg_asr:.6f}, autodan_asr={autodan_asr:.6f}, pair_asr={pair_asr:.6f}")
print(f"mean_asr={mean_asr:.6f}, frr={frr:.6f}")
PY

echo "Baseline eval complete."
echo "Outputs:"
echo "  $HARMFUL_OUT"
echo "  $BENIGN_OUT"
echo "  $METRICS_OUT"
