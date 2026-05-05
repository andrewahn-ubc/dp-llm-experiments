#!/usr/bin/env python3
"""
Deprecated: use ``plot_hyperparameter_heatmaps.py`` instead.

That script writes **clean-reg LM only**, ``aggregate/`` (full test + per-family held-out
ASR/FRR) and ``by_dataset/<benchmark>/`` (stratified ASR + duplicated global FRR panels).

Example::

  python eval/plot_hyperparameter_heatmaps.py \\
    --metrics-dir \"$SCRATCH/dp-llm-sweep/test_eval_outputs\" \\
    --labels \"$SCRATCH/dp-llm-experiments/official/combined_test_dataset.csv\"
"""

from __future__ import annotations

import sys


def main() -> int:
    print(
        "plot_hyperparameter_heatmaps_by_dataset.py is deprecated.\n"
        "Use: python eval/plot_hyperparameter_heatmaps.py "
        "--metrics-dir <test_eval_outputs> [--labels combined_test_dataset.csv]",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
