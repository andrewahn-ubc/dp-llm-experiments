#!/bin/bash
# Deprecated wrapper: use submit_narval_5ep.sh (5× 3h chained jobs).
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$SCRATCH/dp-llm-experiments}"
exec bash "$REPO_ROOT/experiments/adaptive_attack_target/submit_narval_5ep.sh"
