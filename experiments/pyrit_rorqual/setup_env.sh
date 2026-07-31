#!/bin/bash
# Create a Rorqual venv with PyRIT + transformers stack.
#
# Alliance notes:
#   - load gcc + arrow BEFORE activating (real pyarrow, not dummy wheel)
#   - base2048 needs Cargo (installs under $SCRATCH/.cargo if needed)
#   - pillow: use wheelhouse binary (login nodes often fail source builds)
#   - pyodbc: unused for local LLM attacks; install a stub if no wheel/unixodbc
#
#   bash experiments/pyrit_rorqual/setup_env.sh

set -euo pipefail

VENV_DIR="${VENV_DIR:-$SCRATCH/venv/pyrit-rorqual}"

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  deactivate 2>/dev/null || true
fi

module load StdEnv/2023 python/3.11 cuda/12.2 || module load StdEnv/2023 python/3.11
module load gcc arrow
module load rust 2>/dev/null || true
module load unixodbc 2>/dev/null || true

export CARGO_HOME="${CARGO_HOME:-$SCRATCH/.cargo}"
export RUSTUP_HOME="${RUSTUP_HOME:-$SCRATCH/.rustup}"
export PATH="$CARGO_HOME/bin:$PATH"
# Avoid parallel compile storms on login nodes if anything still builds from source.
export MAX_JOBS="${MAX_JOBS:-1}"
export MAKEFLAGS="${MAKEFLAGS:--j1}"

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

# Core ML stack from Alliance wheelhouse when possible.
pip install --no-index torch transformers peft accelerate pandas numpy scipy tqdm \
  sentencepiece tiktoken protobuf 2>/dev/null \
  || pip install torch transformers peft accelerate pandas numpy scipy tqdm \
       sentencepiece tiktoken protobuf

# Prefer binary pillow (source build hits "can't start new thread" on login nodes).
pip install --no-index pillow \
  || pip install --only-binary=:all: pillow \
  || pip install pillow

# pyodbc: try wheelhouse / binary; else stub (PyRIT lists it; we never call ODBC).
if ! python -c "import pyodbc" 2>/dev/null; then
  if pip install --no-index pyodbc 2>/dev/null \
    || pip install --only-binary=:all: pyodbc 2>/dev/null; then
    echo "Installed binary pyodbc"
  else
    STUB_DIR="${TMPDIR:-/tmp}/pyodbc_stub_$$"
    mkdir -p "$STUB_DIR"
    cat >"$STUB_DIR/pyodbc.py" <<'PY'
"""Stub pyodbc for Alliance installs — not used by local-LLM PyRIT attacks."""

class Error(Exception):
    pass


def connect(*_a, **_k):
    raise Error("pyodbc stub: ODBC not available; not needed for local PyRIT")
PY
    cat >"$STUB_DIR/setup.py" <<'PY'
from setuptools import setup

setup(name="pyodbc", version="5.3.0", py_modules=["pyodbc"])
PY
    pip install --no-deps "$STUB_DIR"
    rm -rf "$STUB_DIR"
    echo "Installed pyodbc stub (no unixODBC headers on this node)"
  fi
fi

# Do not upgrade pillow/pyodbc to source builds if already satisfied.
pip install 'pyrit==1.0.1' --prefer-binary --upgrade-strategy only-if-needed

python - <<'PY'
import torch, transformers, peft, pyrit, pyarrow
import PIL
print("ok torch", torch.__version__, "cuda", torch.cuda.is_available())
print("ok transformers", transformers.__version__, "peft", peft.__version__)
print("ok pyrit", pyrit.__version__, "pyarrow", pyarrow.__version__)
print("ok pillow", PIL.__version__)
from pyrit.executor.attack import RedTeamingAttack  # noqa: F401
print("ok RedTeamingAttack import")
PY

echo
echo "Activate with (modules first):"
echo "  module load StdEnv/2023 gcc arrow python/3.11"
echo "  source $VENV_DIR/bin/activate"
