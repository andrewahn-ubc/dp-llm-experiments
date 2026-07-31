#!/bin/bash
# One-time venv setup on Vulcan (aip-mijungp) for train/train.py and the
# rebuttal core-design ablations (train/ablation_epoch.sh). Run ONCE on a
# login node (needs internet for pip); training jobs then just `source` the
# resulting venv.
#
# Usage:
#   bash train/setup_ablation_venv.sh
#
# Override the target venv path with VENV_DIR (default: $SCRATCH/venv/dcl_train,
# matching ablation_epoch.sh's VENV_ACTIVATE default).

set -euo pipefail

VENV_DIR="${VENV_DIR:-${SCRATCH}/venv/dcl_train}"

echo "[INFO] Loading modules (same set ablation_epoch.sh uses at run time)"
module purge
module load StdEnv/2023 cuda/12.2 python/3.11 gcc arrow/21.0.0 scipy-stack

if [[ -d "${VENV_DIR}" ]]; then
  echo "[INFO] venv already exists at ${VENV_DIR} -- reusing it, just upgrading deps."
else
  echo "[INFO] Creating venv at ${VENV_DIR}"
  python -m venv "${VENV_DIR}"
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

pip install --upgrade pip

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Compute Canada wheelhouse (--no-index) is the default; it's built against
# the currently-loaded modules (cuda/12.2 etc.) and avoids pulling arbitrary
# wheels from PyPI on a cluster that discourages it. Set USE_PYPI=1 to fall
# back to PyPI instead (e.g. if a package isn't in the CC wheelhouse).
if [[ "${USE_PYPI:-0}" == "1" ]]; then
  echo "[INFO] Installing from PyPI (USE_PYPI=1)"
  pip install -r "${SCRIPT_DIR}/requirements-train.txt"
else
  echo "[INFO] Installing from Compute Canada wheelhouse (--no-index)"
  pip install --no-index -r "${SCRIPT_DIR}/requirements-train.txt"
fi

echo ""
echo "[INFO] Done. Sanity check (import only -- cuda availability is expected"
echo "to print False here since login nodes have no GPU; that's normal):"
python - <<'PY'
import torch, transformers, peft, pandas, psutil
print(f"torch {torch.__version__}  cuda available: {torch.cuda.is_available()}")
print(f"transformers {transformers.__version__}")
print(f"peft {peft.__version__}")
PY

echo ""
echo "venv ready at: ${VENV_DIR}"
echo "ablation_epoch.sh will use it automatically (VENV_ACTIVATE default =" \
     "\${SCRATCH}/venv/dcl_train/bin/activate)."
