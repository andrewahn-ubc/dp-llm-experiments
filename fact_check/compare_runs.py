"""
compare_runs.py — Collect training_log.json from all sweep runs and print a
sorted comparison table.

Usage (on login node or locally after syncing):
    python fact_check/compare_runs.py \
        --models-dir $SCRATCH/dp-llm-experiments/fact_check_models

Prints a table sorted by FeverSymmetric accuracy (the robustness metric),
showing the trade-off between standard validation accuracy and symmetric
accuracy.  A good run should improve sym_acc without tanking val_acc.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from fact_check.config_loader import load_config


def main():
    cfg = load_config()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models-dir",
        default=str(cfg.paths.output_root),
        help="Directory containing one subfolder per run, each with training_log.json "
             "(default: paths.output_root from config.yaml)"
    )
    args = parser.parse_args()

    base = Path(args.models_dir)
    logs = list(base.glob("*/training_log.json"))

    if not logs:
        print(f"No training_log.json files found under {base}")
        return

    rows = []
    for log_path in logs:
        with open(log_path) as f:
            entries = json.load(f)
        # Take the last epoch (best checkpoint policy: last epoch)
        last = entries[-1]
        rows.append(last)

    # Sort by FeverSymmetric accuracy descending
    rows.sort(key=lambda r: r.get("sym_acc", 0), reverse=True)

    # Print table
    header = f"{'run_id':<55} {'λ':>5} {'ε':>5} {'lr':>7} {'rank':>4} {'val_acc':>8} {'sym_acc':>8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r.get('run_id','?'):<55} "
            f"{r.get('lambda',0):>5.2f} "
            f"{r.get('epsilon',0):>5.2f} "
            f"{r.get('lr',0):>7.0e} "
            f"{r.get('lora_rank',0):>4} "
            f"{r.get('val_acc',0):>8.4f} "
            f"{r.get('sym_acc',0):>8.4f}"
        )

    print(f"\n{len(rows)} runs found.")


if __name__ == "__main__":
    main()
