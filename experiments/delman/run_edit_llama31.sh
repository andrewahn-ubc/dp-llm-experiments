#!/bin/bash
#SBATCH --job-name=delman_l31
#SBATCH --account=def-mijungp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=output/delman_l31_%j.out

# Run DELMAN editing on Llama-3.1-8B-Instruct, then copy the edited
# weights to a stable path for Jailbreak-R1 / PyRIT baselines.
#
# Prerequisites (once):
#   git clone https://github.com/wanglne/DELMAN.git $SCRATCH/DELMAN
#   # unpack cov zip into $SCRATCH/DELMAN/data/stats
#   # set offset=2 in $SCRATCH/DELMAN/rome/repr_tools.py for Llama 3.1
#   # install DELMAN deps into $SCRATCH/venv/delman (or reuse nanogcg + pip install -r)
#
#   cd $SCRATCH/dp-llm-experiments && mkdir -p output
#   sbatch experiments/delman/run_edit_llama31.sh
#
# Base weights:   $SCRATCH/llama31_8b_instruct
# Edited output:  $SCRATCH/delman_llama31_8b_instruct   (override with DELMAN_OUT)

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$SCRATCH/dp-llm-experiments}"
cd "$REPO_ROOT"
mkdir -p output

DELMAN_REPO="${DELMAN_REPO:-$SCRATCH/DELMAN}"
BASE_L31="${BASE_L31:-$SCRATCH/llama31_8b_instruct}"
DELMAN_OUT="${DELMAN_OUT:-$SCRATCH/delman_llama31_8b_instruct}"
HPARAMS="${HPARAMS_FNAME:-Llama-3.1-8B-Instruct.json}"
DATA_NAME="${DATA_NAME:-HarmBench.json}"
OUT_NAME="${OUT_NAME:-DELMAN_llama3.1}"

module load StdEnv/2023 python/3.11 cuda/12.2 || module load StdEnv/2023 python/3.11
# shellcheck disable=SC1090
source "${VENV_ACTIVATE:-$SCRATCH/venv/delman/bin/activate}"

export TRANSFORMERS_CACHE="${SLURM_TMPDIR:-/tmp}/hf_cache"
export HF_HOME="${SLURM_TMPDIR:-/tmp}/hf_home"
mkdir -p "$TRANSFORMERS_CACHE" "$HF_HOME"

if [[ ! -d "$DELMAN_REPO" ]]; then
  echo "ERROR: DELMAN repo missing: $DELMAN_REPO" >&2
  exit 2
fi
if [[ ! -d "$BASE_L31" ]]; then
  echo "ERROR: Llama-3.1 base missing: $BASE_L31" >&2
  exit 2
fi
if [[ ! -d "$DELMAN_REPO/data/stats" ]]; then
  echo "ERROR: unpack cov matrices into $DELMAN_REPO/data/stats" >&2
  exit 2
fi

# Llama 3.1 needs offset=2 (DELMAN README).
OFFSET_LINE=$(grep -nE '^\s*offset\s*=' "$DELMAN_REPO/rome/repr_tools.py" | head -1 || true)
echo "repr_tools offset line: ${OFFSET_LINE:-not found}"
if ! grep -qE 'offset\s*=\s*2' "$DELMAN_REPO/rome/repr_tools.py"; then
  echo "WARN: set offset=2 in $DELMAN_REPO/rome/repr_tools.py for Llama 3.1" >&2
fi

cd "$DELMAN_REPO"
echo "Running DELMAN: model=$BASE_L31 hparams=$HPARAMS out_name=$OUT_NAME"

python -m run_delman \
  --model_name "$BASE_L31" \
  --hparams_fname "$HPARAMS" \
  --data_name "$DATA_NAME" \
  --out_name "$OUT_NAME"

# Prefer an explicit HF save if the driver wrote one; otherwise copy common out dirs.
CANDIDATES=(
  "$DELMAN_REPO/results/$OUT_NAME"
  "$DELMAN_REPO/outputs/$OUT_NAME"
  "$DELMAN_REPO/$OUT_NAME"
)
SRC=""
for c in "${CANDIDATES[@]}"; do
  if [[ -d "$c" ]] && [[ -f "$c/config.json" || -f "$c/model.safetensors" || -f "$c/pytorch_model.bin" || -n "$(ls "$c"/*.safetensors 2>/dev/null | head -1)" ]]; then
    SRC="$c"
    break
  fi
done

if [[ -z "$SRC" ]]; then
  echo "DELMAN finished but no HF dir auto-detected under results/outputs."
  echo "Locate the edited checkpoint and copy manually to: $DELMAN_OUT"
  echo "Searched:"; printf '  %s\n' "${CANDIDATES[@]}"
  find "$DELMAN_REPO" -maxdepth 3 -name config.json 2>/dev/null | head -20 || true
  exit 3
fi

mkdir -p "$(dirname "$DELMAN_OUT")"
rm -rf "$DELMAN_OUT"
cp -a "$SRC" "$DELMAN_OUT"
echo "DELMAN edited model -> $DELMAN_OUT"
ls -la "$DELMAN_OUT" | head
