#!/usr/bin/env python3
"""
Aggregate eval metrics from train/submit_wandb_sweep.py jobs.

Each job writes TSVs::

    {eval_output_dir}/{slug}_epoch{N}_metrics.tsv

with columns ``metric`` and ``value`` (slug, epoch, lr, lambda, epsilon,
system_prompt_mode, lm_loss_input, gcg_asr, autodan_asr, pair_asr, mean_asr,
frr, n_harmful, n_benign).

This script globs all ``*_metrics.tsv`` under ``--eval-output-dir``, builds one
row per file, writes ``sweep_metrics_summary.csv``, and prints ranked tables.

Default sort: ``mean_asr`` ascending (lower attack success = better defense).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


def _read_metrics_tsv(path: Path) -> dict:
    df = pd.read_csv(path, sep="\t")
    if "metric" not in df.columns or "value" not in df.columns:
        raise ValueError(f"expected columns metric, value; got {list(df.columns)} in {path}")
    row = df.set_index("metric")["value"].to_dict()
    row["_metrics_path"] = str(path)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--eval-output-dir",
        default=os.environ.get("EVAL_OUTPUT_DIR", "$SCRATCH/dp-llm-sweep/eval_outputs"),
        help="Directory containing *_metrics.tsv (matches submit_wandb_sweep --eval-output-dir).",
    )
    ap.add_argument(
        "--out-csv",
        default=None,
        help="Output CSV (default: <eval-output-dir>/sweep_metrics_summary.csv).",
    )
    ap.add_argument(
        "--sort-by",
        default="mean_asr",
        choices=["mean_asr", "frr", "gcg_asr", "autodan_asr", "pair_asr"],
        help="Rank key (default: mean_asr).",
    )
    ap.add_argument(
        "--ascending",
        action="store_true",
        help="Sort ascending (overrides default).",
    )
    ap.add_argument(
        "--descending",
        action="store_true",
        help="Sort descending (overrides default).",
    )
    args = ap.parse_args()

    root = Path(os.path.expandvars(args.eval_output_dir)).resolve()
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")

    paths = sorted(root.glob("*_metrics.tsv"))
    if not paths:
        raise SystemExit(f"no *_metrics.tsv under {root}")

    rows: list[dict] = []
    for p in paths:
        try:
            rows.append(_read_metrics_tsv(p))
        except Exception as e:
            print(f"[skip] {p}: {e}")

    if not rows:
        raise SystemExit("no valid metrics files")

    out = pd.DataFrame(rows)
    numeric = [
        "lr",
        "lambda",
        "epsilon",
        "epoch",
        "gcg_asr",
        "autodan_asr",
        "pair_asr",
        "mean_asr",
        "frr",
        "n_harmful",
        "n_benign",
    ]
    for c in numeric:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    preferred = [
        "slug",
        "epoch",
        "lr",
        "lambda",
        "epsilon",
        "system_prompt_mode",
        "lm_loss_input",
        "mean_asr",
        "gcg_asr",
        "autodan_asr",
        "pair_asr",
        "frr",
        "n_harmful",
        "n_benign",
        "_metrics_path",
    ]
    cols = [c for c in preferred if c in out.columns] + [c for c in out.columns if c not in preferred]
    out = out[cols]

    sort_col = args.sort_by
    if sort_col not in out.columns:
        raise SystemExit(f"missing sort column {sort_col!r}; columns: {list(out.columns)}")

    if args.descending:
        ascending = False
    elif args.ascending:
        ascending = True
    else:
        ascending = sort_col != "frr"

    out_sorted = out.sort_values(sort_col, ascending=ascending, na_position="last")

    out_path = (
        Path(os.path.expandvars(args.out_csv))
        if args.out_csv
        else root / "sweep_metrics_summary.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_sorted.to_csv(out_path, index=False)
    print(f"Wrote {len(out_sorted)} rows -> {out_path}")

    display = [c for c in ("slug", "epoch", "lr", "lambda", "epsilon", "mean_asr", "gcg_asr", "autodan_asr", "pair_asr", "frr") if c in out_sorted.columns]
    print(f"\nTop 20 by {sort_col} ({'asc' if ascending else 'desc'}ending):")
    print(out_sorted[display].head(20).to_string(index=False))
    print(f"\nBottom 10 by {sort_col}:")
    print(out_sorted[display].tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
