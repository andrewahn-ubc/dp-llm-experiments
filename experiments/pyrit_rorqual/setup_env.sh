#!/bin/bash
# Create a Rorqual venv with PyRIT + transformers stack.
#
#   bash experiments/pyrit_rorqual/setup_env.sh
#   export VENV_ACTIVATE=$SCRATCH/venv/pyrit-rorqual/bin/activate

set -euo pipefail

VENV_DIR="${VENV_DIR:-$SCRATCH/venv/pyrit-rorqual}"

module load StdEnv/2023 python/3.11 cuda/12.2 || module load StdEnv/2023 python/3.11
# Linking / some crates need a compiler; base2048 (pyrit dep) needs Cargo.
module load gcc 2>/dev/null || true
module load rust 2>/dev/null || module spider rust 2>/dev/null | head -5 || true

export CARGO_HOME="${CARGO_HOME:-$SCRATCH/.cargo}"
export RUSTUP_HOME="${RUSTUP_HOME:-$SCRATCH/.rustup}"
export PATH="$CARGO_HOME/bin:$PATH"
if ! command -v cargo >/dev/null 2>&1; then
  echo "Installing Rust toolchain into $CARGO_HOME (needed to build base2048)..."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain stable --profile minimal --no-modify-path
  export PATH="$CARGO_HOME/bin:$PATH"
fi
cargo --version
rustc --version

if [[ -d "$VENV_DIR" ]]; then
  echo "Venv already exists: $VENV_DIR"
  echo "To recreate: rm -rf \"$VENV_DIR\" && re-run."
else
  python -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel

# Core ML stack (Alliance wheelhouse when available).
pip install --no-index torch transformers peft accelerate pandas numpy scipy tqdm \
  sentencepiece tiktoken protobuf 2>/dev/null \
  || pip install torch transformers peft accelerate pandas numpy scipy tqdm \
       sentencepiece tiktoken protobuf

# PyRIT pulls many deps from PyPI; base2048 compiles with Cargo.
pip install 'pyrit==1.0.1'

python - <<'PY'
import torch, transformers, peft, pyrit
print("ok torch", torch.__version__, "cuda", torch.cuda.is_available())
print("ok transformers", transformers.__version__, "peft", peft.__version__)
print("ok pyrit", pyrit.__version__)
PY

echo
echo "Activate with:  source $VENV_DIR/bin/activate"
