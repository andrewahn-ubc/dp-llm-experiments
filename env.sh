#!/bin/bash
# Activate the correct venv and set experiment-specific env variables.
#
# Usage: source ./env.sh <experiment>
#
# Experiments:
#   fact-check — FEVER fact-checking with symmetric stability regularizer

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Error: this script must be sourced, not executed."
    echo "Usage: source ./env.sh <experiment>"
    exit 1
fi

if [ -z "${1:-}" ]; then
    echo "Usage: source ./env.sh <experiment>"
    echo "Experiments: fact-check"
    return 1
fi

EXPERIMENT="$1"

# ── Compute Canada modules ───────────────────────────────────────────────────
module purge

case "$EXPERIMENT" in
    fact-check)
        module load StdEnv/2023 python/3.11
        source "$SCRATCH/venv/factcheck/bin/activate"
        ;;
    *)
        echo "Unknown experiment: $EXPERIMENT"
        echo "Valid options: fact-check"
        return 1
        ;;
esac

# ── Shared HuggingFace cache (node-local when inside a SLURM job) ────────────
if [ -n "${SLURM_TMPDIR:-}" ]; then
    export TRANSFORMERS_CACHE="$SLURM_TMPDIR/hf_cache"
    export HF_HOME="$SLURM_TMPDIR/hf_home"
    mkdir -p "$TRANSFORMERS_CACHE" "$HF_HOME"
else
    export TRANSFORMERS_CACHE="$SCRATCH/hf_cache"
    export HF_HOME="$SCRATCH/hf_home"
fi

# ── Experiment-specific variables ────────────────────────────────────────────
case "$EXPERIMENT" in
    fact-check)
        export FC_REPO_ROOT="$HOME/repos/dp-llm-experiments"
        export FC_DATA_DIR="$SCRATCH/dp-llm-experiments/fact_check_data"
        export FC_MODEL_ROOT="$SCRATCH/hf_models"
        export FC_OUTPUT_ROOT="$SCRATCH/dp-llm-experiments/fact_check_models"
        export FC_VENV="$SCRATCH/venv/factcheck"
        ;;
esac

echo "[$EXPERIMENT] environment ready."
echo "  venv:       $VIRTUAL_ENV"
echo "  repo:       $FC_REPO_ROOT"
echo "  data:       $FC_DATA_DIR"
echo "  model_root: $FC_MODEL_ROOT"
echo "  output:     $FC_OUTPUT_ROOT"
