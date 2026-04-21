"""
compare_runs.py — Collect training_log.json from all sweep runs, print a
sorted comparison table, and optionally plot training curves.

Usage (on login node or locally after syncing):
    python fact_check/compare_runs.py \
        --models-dir $SCRATCH/dp-llm-experiments/fact_check_models

    # Also save training curve plots:
    python fact_check/compare_runs.py --plot --out figures/

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


def plot_curves(all_entries: list[list[dict]], out_dir: Path | None) -> None:
    import matplotlib.pyplot as plt

    metrics = [
        ("loss",    "Loss"),
        ("ce",      "CE Loss"),
        ("stab",    "Stability Loss"),
        ("val_acc", "Val Accuracy"),
        ("sym_acc", "Sym Accuracy"),
    ]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 3 * len(metrics)))

    for ax, (key, label) in zip(axes, metrics):
        for entries in all_entries:
            if not entries:
                continue
            run_id = entries[-1].get("run_id", "?")
            lam    = entries[-1].get("lambda", 0)
            eps    = entries[-1].get("epsilon", 0)
            lr     = entries[-1].get("lr", 0)
            legend = f"λ={lam} ε={eps} lr={lr:.0e}"
            epochs = [e["epoch"] for e in entries if key in e]
            values = [e[key]    for e in entries if key in e]
            if epochs:
                ax.plot(epochs, values, marker="o", markersize=4, label=legend)
        ax.set_xlabel("epoch")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "training_curves.png"
        plt.savefig(path, dpi=150)
        print(f"\nPlot saved to {path}")
    else:
        plt.show()


def main():
    cfg = load_config()

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models-dir",
        default=str(cfg.paths.output_root),
        help="Directory containing one subfolder per run, each with training_log.json "
             "(default: paths.output_root from config.yaml)"
    )
    parser.add_argument("--plot",   action="store_true", help="Plot training curves")
    parser.add_argument("--out",    default=None, help="Directory to save plot (default: show interactively)")
    parser.add_argument("--filter", default=None, help="Only include runs whose directory name contains this string (e.g. a timestamp)")
    args = parser.parse_args()

    base = Path(args.models_dir)
    logs = [p for p in base.glob("*/training_log.json")
            if args.filter is None or args.filter in p.parent.name]

    if not logs:
        print(f"No training_log.json files found under {base}")
        return

    consistency_csv = base / "consistency_eval.csv"
    consistency_map: dict[str, dict] = {}
    if consistency_csv.exists():
        import pandas as pd
        cdf = pd.read_csv(consistency_csv)
        for _, row in cdf.iterrows():
            consistency_map[row["run_id"]] = row.to_dict()

    all_entries = []
    summary_rows = []
    for log_path in logs:
        with open(log_path) as f:
            entries = json.load(f)
        if not entries:
            continue
        all_entries.append(entries)
        summary_rows.append(entries[-1])  # last epoch as summary

    # Sort by FeverSymmetric accuracy descending
    summary_rows.sort(key=lambda r: r.get("sym_acc", 0), reverse=True)

    has_consistency = bool(consistency_map)

    # Print table
    header = (
        f"{'run_id':<60} {'dataset':<10} {'model':<25} {'mode':<6} "
        f"{'λ':>5} {'ε':>5} {'lr':>7} {'rank':>4} {'val_acc':>8} {'sym_acc':>8}"
    )
    if has_consistency:
        header += f" {'gap_all':>8} {'gap_wrd':>8}"
    print(header)
    print("-" * len(header))
    for r in summary_rows:
        mode = "aug" if r.get("augmentation_only") else "reg"
        line = (
            f"{r.get('run_id','?'):<60} "
            f"{r.get('dataset','fever'):<10} "
            f"{r.get('model_name','?'):<25} "
            f"{mode:<6} "
            f"{r.get('lambda',0):>5.2f} "
            f"{r.get('epsilon',0):>5.2f} "
            f"{r.get('lr',0):>7.0e} "
            f"{r.get('lora_rank',0):>4} "
            f"{r.get('val_acc',0):>8.4f} "
            f"{r.get('sym_acc',0):>8.4f}"
        )
        if has_consistency:
            cons = consistency_map.get(r.get("run_id", ""), {})
            gap_all = cons.get("gap_all", float("nan"))
            gap_wrd = cons.get("gap_word_shuffle", float("nan"))
            line += f" {gap_all:>8.4f} {gap_wrd:>8.4f}"
        print(line)

    print(f"\n{len(summary_rows)} runs found.")

    if args.plot:
        plot_curves(all_entries, Path(args.out) if args.out else None)


if __name__ == "__main__":
    main()
