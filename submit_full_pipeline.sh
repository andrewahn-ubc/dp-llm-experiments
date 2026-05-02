#!/usr/bin/env bash
# One-shot launcher: submit all training sweeps + test-eval array from repo root.
# Same as: python run_final_pipeline.py "$@"
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec python3 "${ROOT}/run_final_pipeline.py" "$@"
