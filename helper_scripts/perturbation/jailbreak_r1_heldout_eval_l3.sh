#!/bin/bash
#SBATCH --job-name=jb_r1_heldout_l3
#SBATCH --account=def-mijungp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=5:00:00
#SBATCH --array=0-5
#SBATCH --output=output/jb_r1_heldout_l3_%A_%a.out

# Jailbreak-R1 held-out ASR (+ FRR). One array task = one model:
#
#   0  base Llama-3-8B-Instruct          ($SCRATCH/llama_3_8b_instruct)
#   1  MixAT                             $SCRATCH/mixat
#   2  DOOR                              $SCRATCH/door
#   3  DCL seen λ=1, ε=-1                LoRA
#   4  DCL seen λ=3, ε=1                 LoRA
#   5  DELMAN (Llama-3.1 edited)         $SCRATCH/delman_llama31_8b_instruct
#
# DELMAN is produced from $SCRATCH/llama31_8b_instruct via
#   experiments/delman/run_edit_llama31.sh
#
#   EPOCH=2 sbatch helper_scripts/perturbation/jailbreak_r1_heldout_eval_l3.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p output

module load StdEnv/2023 python/3.11
# shellcheck source=/dev/null
source "${VENV_ACTIVATE:-$SCRATCH/venv/nanogcg/bin/activate}"

export TRANSFORMERS_CACHE="${SLURM_TMPDIR:-/tmp}/hf_cache"
export HF_HOME="${SLURM_TMPDIR:-/tmp}/hf_home"
mkdir -p "$TRANSFORMERS_CACHE" "$HF_HOME"

REPO_ROOT="${REPO_ROOT:-$SCRATCH/dp-llm-experiments}"
CK_ROOT="${CHECKPOINT_ROOT:-$SCRATCH/dp-llm-sweep}"
EPOCH="${EPOCH:-2}"
SYSTEM_PROMPT_MODE="${SYSTEM_PROMPT_MODE:-empty}"

HARMFUL_DATA="${HARMFUL_DATA:-${REPO_ROOT}/official_data/jailbreak_r1/combined_test_with_jailbreak_r1.csv}"
BENIGN_DATA="${BENIGN_DATA:-${REPO_ROOT}/official_data/frr_test.csv}"
OUT_DIR="${OUT_DIR:-$SCRATCH/dp-llm-eval/jailbreak_r1_heldout_l3}"
mkdir -p "$OUT_DIR"

MIXAT_PATH="${MIXAT_PATH:-$SCRATCH/mixat}"
DOOR_PATH="${DOOR_PATH:-$SCRATCH/door}"
DELMAN_PATH="${DELMAN_PATH:-$SCRATCH/delman_llama31_8b_instruct}"

TAGS=(base mixat door dcl_lam1_eps-1 dcl_lam3_eps1 delman)
TAG="${TAGS[$SLURM_ARRAY_TASK_ID]}"

BASE_LLM_ARGS=()
RESUME_ARGS=()
NEED_DIR=""

case "$TAG" in
  base)
    RUN_TAG="base_jb_r1"
    ;;
  mixat)
    BASE_LLM_ARGS=(--base-llm "$MIXAT_PATH")
    NEED_DIR="$MIXAT_PATH"
    RUN_TAG="mixat_jb_r1"
    ;;
  door)
    BASE_LLM_ARGS=(--base-llm "$DOOR_PATH")
    NEED_DIR="$DOOR_PATH"
    RUN_TAG="door_jb_r1"
    ;;
  dcl_lam1_eps-1)
    CKPT="${CK_ROOT}/l3_run_lr2e-05_lam1_eps-1_finetuned_llm_epoch${EPOCH}"
    RESUME_ARGS=(--resume-from "$CKPT")
    NEED_DIR="$CKPT"
    RUN_TAG="dcl_lam1_eps-1_ep${EPOCH}_jb_r1"
    ;;
  dcl_lam3_eps1)
    CKPT="${CK_ROOT}/l3_run_lr2e-05_lam3_eps1_finetuned_llm_epoch${EPOCH}"
    RESUME_ARGS=(--resume-from "$CKPT")
    NEED_DIR="$CKPT"
    RUN_TAG="dcl_lam3_eps1_ep${EPOCH}_jb_r1"
    ;;
  delman)
    # Full HF edited Llama-3.1 (not LoRA). Chat template is Llama-3 family.
    BASE_LLM_ARGS=(--base-llm "$DELMAN_PATH")
    NEED_DIR="$DELMAN_PATH"
    RUN_TAG="delman_l31_jb_r1"
    ;;
  *)
    echo "Unknown TAG=$TAG" >&2
    exit 2
    ;;
esac

if [[ -n "$NEED_DIR" && ! -d "$NEED_DIR" ]]; then
  echo "ERROR: model path not found: $NEED_DIR" >&2
  echo "For DELMAN: run experiments/delman/run_edit_llama31.sh first" >&2
  echo "(base=$SCRATCH/llama31_8b_instruct → out=\$DELMAN_PATH)." >&2
  exit 2
fi

if [[ ! -f "$HARMFUL_DATA" ]]; then
  echo "ERROR: Jailbreak-R1 test CSV missing: $HARMFUL_DATA" >&2
  exit 2
fi

BENIGN_TMP="${SLURM_TMPDIR:-/tmp}/frr_jb_r1_${TAG}.csv"
if [[ -f "${REPO_ROOT}/eval/prep_frr_eval_input.py" ]]; then
  python "${REPO_ROOT}/eval/prep_frr_eval_input.py" "$BENIGN_DATA" "$BENIGN_TMP"
else
  python - "$BENIGN_DATA" "$BENIGN_TMP" <<'PY'
import sys
import pandas as pd
src, dst = sys.argv[1:3]
df = pd.read_csv(src)
if "Original Prompt" not in df.columns:
    for c in ("adversarial", "goal", "prompt"):
        if c in df.columns:
            df = df.copy()
            df["Original Prompt"] = df[c].astype(str)
            break
    else:
        raise SystemExit(f"no prompt column in {src}: {list(df.columns)}")
df.to_csv(dst, index=False)
PY
fi

HARMFUL_OUT="${OUT_DIR}/${RUN_TAG}_harmful"
BENIGN_OUT="${OUT_DIR}/${RUN_TAG}_benign"

echo "=== Jailbreak-R1 held-out eval ==="
echo "tag=$TAG  task=$SLURM_ARRAY_TASK_ID"
echo "harmful=$HARMFUL_DATA"
[[ -n "$NEED_DIR" ]] && echo "model_dir=$NEED_DIR"
echo "outs: ${HARMFUL_OUT}.csv / ${BENIGN_OUT}.csv"

python "${REPO_ROOT}/eval/eval.py" \
  --model-profile llama_3_8b_instruct \
  --eval-mode unseen-family \
  --unseen-family jailbreak_r1 \
  --system-prompt-mode "$SYSTEM_PROMPT_MODE" \
  --benign-system-prompt-mode "$SYSTEM_PROMPT_MODE" \
  "${BASE_LLM_ARGS[@]}" \
  "${RESUME_ARGS[@]}" \
  --validation-data "$HARMFUL_DATA" \
  --benign-validation-data "$BENIGN_TMP" \
  --harmful-output-file "$HARMFUL_OUT" \
  --benign-output-file "$BENIGN_OUT"

python - "$OUT_DIR" "$RUN_TAG" <<'PY'
import sys
from pathlib import Path
import pandas as pd

out_dir, tag = Path(sys.argv[1]), sys.argv[2]
h = out_dir / f"{tag}_harmful.csv"
b = out_dir / f"{tag}_benign.csv"
if not h.is_file():
    h = out_dir / f"{tag}_harmful.csv.csv"
if not b.is_file():
    b = out_dir / f"{tag}_benign.csv.csv"

hdf = pd.read_csv(h)
bdf = pd.read_csv(b)
col = "Jailbreak-R1 Safety"
asr = (hdf[col].astype(str).str.lower() == "unsafe").mean() if col in hdf.columns else float("nan")
scol = "Original Safety"
frr = (bdf[scol].astype(str).str.lower() == "refusal").mean() if scol in bdf.columns else float("nan")
print(f"[{tag}] jailbreak_r1_asr={asr:.4f}  frr={frr:.4f}  n_harm={len(hdf)} n_benign={len(bdf)}")
(out_dir / f"{tag}_metrics.tsv").write_text(
    f"metric\tvalue\njailbreak_r1_asr\t{asr}\nfrr\t{frr}\nn_harmful\t{len(hdf)}\nn_benign\t{len(bdf)}\n"
)
print(f"wrote {out_dir / (tag + '_metrics.tsv')}")
PY
