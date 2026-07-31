#!/bin/bash
# Create a fresh Python venv for the rollout-T sensitivity experiment on Rorqual.
# Run on a login node (has network for any pip fallsbacks that need it).
#
# Usage:
#   bash experiments/rollout_T_sensitivity/setup_rorqual_env.sh
#
# Then point jobs at:
#   export VENV_ACTIVATE="$SCRATCH/venv/dp-llm-rorqual/bin/activate"

set -euo pipefail

VENV_DIR="${VENV_DIR:-$SCRATCH/venv/dp-llm-rorqual}"

module load StdEnv/2023 python/3.11 cuda/12.2 || module load StdEnv/2023 python/3.11

if [[ -d "$VENV_DIR" ]]; then
  echo "Venv already exists: $VENV_DIR"
  echo "To recreate: rm -rf \"$VENV_DIR\" && re-run this script."
else
  python -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

pip install --upgrade pip setuptools wheel
# Prefer Alliance wheelhouse (no internet). Fall back to PyPI if a package is missing.
# sentencepiece + tiktoken are required to load Llama / HarmBench / Mistral tokenizers.
PKGS=(
  torch transformers peft accelerate
  pandas numpy scipy tqdm psutil
  sentencepiece tiktoken protobuf
)
pip install --no-index "${PKGS[@]}" 2>/dev/null || pip install "${PKGS[@]}"

# Optional but used by training scripts:
pip install --no-index wandb 2>/dev/null || pip install wandb || true

python - <<'PY'
import torch, transformers, peft, sentencepiece, tiktoken
print("ok torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
print("ok transformers", transformers.__version__, "peft", peft.__version__)
print("ok sentencepiece", sentencepiece.__version__, "tiktoken", tiktoken.__version__)
PY

echo
echo "Activate with:"
echo "  source $VENV_DIR/bin/activate"
