#!/bin/bash
#SBATCH --job-name=rollout_T_eval
#SBATCH --account=def-mijungp
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --array=0-3
#SBATCH --output=output/rollout_T_eval_%A_%a.out

# Seen-family ASR + FRR on the test set for each T∈{1,3,5,10}.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$SCRATCH/dp-llm-experiments}"
cd "$REPO_ROOT"
mkdir -p output
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

module load StdEnv/2023 python/3.11
# shellcheck disable=SC1090
source "${VENV_ACTIVATE:-$SCRATCH/venv/dp-llm-rorqual/bin/activate}"

export TRANSFORMERS_CACHE="${SLURM_TMPDIR}/hf_cache"
export HF_HOME="${SLURM_TMPDIR}/hf_home"
mkdir -p "$TRANSFORMERS_CACHE" "$HF_HOME"

TS=(1 3 5 10)
T="${TS[$SLURM_ARRAY_TASK_ID]}"

LR="${LR:-2e-5}"
LAM="${LAM:-0.1}"
EPS="${EPS:--0.5}"
EPOCH="${EPOCH:-1}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$SCRATCH/dp-llm-sweep}"
MODEL_PROFILE="${MODEL_PROFILE:-llama_2_7b_chat}"
HARMFUL_DATA="${HARMFUL_DATA:-$REPO_ROOT/official_data/combined_test_dataset.csv}"
BENIGN_DATA="${BENIGN_DATA:-$REPO_ROOT/official_data/frr_test.csv}"
OUT_DIR="${OUT_DIR:-$CHECKPOINT_ROOT/rollout_T_sensitivity}"

SLUG=$(
  python - <<PY
from train.model_profiles import make_run_slug
print(make_run_slug(float("$LR"), float("$LAM"), float("$EPS"), "clean",
                    model_profile="$MODEL_PROFILE", rollout_length=int("$T")))
PY
)
SEEN_MODEL_PATH="${CHECKPOINT_ROOT}/${SLUG}_finetuned_llm_epoch${EPOCH}"
RUN_TAG="seen_${SLUG}_ep${EPOCH}"

mkdir -p "$OUT_DIR"
HARMFUL_OUT="${OUT_DIR}/${RUN_TAG}_harmful.csv"
BENIGN_OUT="${OUT_DIR}/${RUN_TAG}_benign.csv"
METRICS_OUT="${OUT_DIR}/${RUN_TAG}_metrics.tsv"

echo "=== rollout-T seen eval: T=$T ==="
echo "resume_from=$SEEN_MODEL_PATH"
echo "harmful=$HARMFUL_DATA"
echo "benign=$BENIGN_DATA"

if [[ ! -d "$SEEN_MODEL_PATH" ]]; then
  echo "ERROR: missing checkpoint $SEEN_MODEL_PATH" >&2
  exit 2
fi

BENIGN_TMP="${SLURM_TMPDIR}/frr_eval_input_T${T}.csv"
python - "$BENIGN_DATA" "$BENIGN_TMP" <<'PY'
import sys
import pandas as pd

src, dst = sys.argv[1:3]
df = pd.read_csv(src)
if "Original Prompt" in df.columns:
    df.to_csv(dst, index=False)
    raise SystemExit(0)
candidates = [
    "adversarial", "Adversarial", "goal", "Goal", "prompt", "Prompt",
    "original_prompt", "instruction", "Instruction",
]
src_col = next((c for c in candidates if c in df.columns), None)
if src_col is None:
    raise ValueError(f"No prompt column in FRR CSV; cols={list(df.columns)}")
df = df.copy()
df["Original Prompt"] = df[src_col].astype(str)
df.to_csv(dst, index=False)
PY

python "$REPO_ROOT/eval/eval.py" \
  --eval-mode seen-family \
  --system-prompt-mode empty \
  --model-profile "$MODEL_PROFILE" \
  --resume-from "$SEEN_MODEL_PATH" \
  --validation-data "$HARMFUL_DATA" \
  --benign-validation-data "$BENIGN_TMP" \
  --harmful-output-file "$HARMFUL_OUT" \
  --benign-output-file "$BENIGN_OUT"

python - "$HARMFUL_OUT" "$BENIGN_OUT" "$METRICS_OUT" "$T" "$SLUG" "$EPOCH" <<'PY'
import sys
from pathlib import Path
import pandas as pd

harmful_csv, benign_csv, metrics_tsv, T, slug, epoch = sys.argv[1:7]

def resolve_csv(path: str) -> Path:
    """Prefer path as given; fall back to legacy *.csv.csv double-suffix."""
    p = Path(path)
    if p.is_file():
        return p
    alt = Path(str(path) + ".csv")
    if alt.is_file():
        return alt
    raise FileNotFoundError(f"missing {path} (also tried {alt})")

h = pd.read_csv(resolve_csv(harmful_csv))
b = pd.read_csv(resolve_csv(benign_csv))

def mean_is_unsafe(df, col):
    return float((df[col].astype(str).str.lower() == "unsafe").mean())

gcg_asr = mean_is_unsafe(h, "GCG Safety")
autodan_asr = mean_is_unsafe(h, "AutoDAN Safety")
pair_asr = mean_is_unsafe(h, "PAIR Safety")
mean_asr = (gcg_asr + autodan_asr + pair_asr) / 3.0
frr = float((b["Original Safety"].astype(str).str.lower() == "refusal").mean())

rows = [
    ("slug", slug),
    ("rollout_length_T", T),
    ("epoch", epoch),
    ("mode", "seen-family"),
    ("gcg_asr", gcg_asr),
    ("autodan_asr", autodan_asr),
    ("pair_asr", pair_asr),
    ("mean_asr", mean_asr),
    ("frr", frr),
    ("n_harmful", len(h)),
    ("n_benign", len(b)),
    ("harmful_csv", harmful_csv),
    ("benign_csv", benign_csv),
]
with open(metrics_tsv, "w", encoding="utf-8") as f:
    f.write("metric\tvalue\n")
    for k, v in rows:
        f.write(f"{k}\t{v}\n")
print(f"Wrote {metrics_tsv}")
print(f"T={T} mean_asr={mean_asr:.6f} frr={frr:.6f}")
PY

echo "Done: $METRICS_OUT"
