#!/usr/bin/env python3
"""Aggregate per-epoch metrics TSVs from the rebuttal core-design ablations
(train/ablation_eval_epoch.sh) into one table: rows = ablation x epoch,
columns = gcg_asr/autodan_asr/pair_asr/mean_asr/frr.

Reads $SCRATCH/dp-llm-eval/ablation/<tag>/ablation_<tag>_epoch<N>_metrics.tsv,
written by eval/write_base_model_test_metrics.py.

Usage:
    python train/aggregate_ablation_metrics.py
    python train/aggregate_ablation_metrics.py --root /scratch/ammany/dp-llm-eval/ablation
    python train/aggregate_ablation_metrics.py --csv-out ablation_summary.csv
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import pandas as pd

METRICS_FILENAME_RE = re.compile(r"^ablation_(?P<tag>.+)_epoch(?P<epoch>\d+)_metrics\.tsv$")

METRIC_COLUMNS = ["gcg_asr", "autodan_asr", "pair_asr", "mean_asr", "frr", "n_harmful", "n_benign"]


def parse_metrics_tsv(path: Path) -> dict:
    metrics = {}
    with path.open() as f:
        next(f)  # header row: "metric\tvalue"
        for line in f:
            if "\t" not in line:
                continue
            key, value = line.rstrip("\n").split("\t", 1)
            metrics[key] = value
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=os.path.expandvars("$SCRATCH/dp-llm-eval/ablation"),
        help="Directory containing one subdir per ablation tag (default: $SCRATCH/dp-llm-eval/ablation).",
    )
    parser.add_argument(
        "--csv-out",
        default=None,
        help="Optional path to also write the aggregated table as CSV.",
    )
    args = parser.parse_args()

    root = Path(os.path.expandvars(os.path.expanduser(args.root)))
    if not root.is_dir():
        raise SystemExit(f"ERROR: root not found: {root}")

    rows = []
    for tsv_path in sorted(root.glob("*/*_metrics.tsv")):
        m = METRICS_FILENAME_RE.match(tsv_path.name)
        if not m:
            print(f"[skip] unrecognized filename: {tsv_path.name}")
            continue
        tag = m.group("tag")
        epoch = int(m.group("epoch"))
        metrics = parse_metrics_tsv(tsv_path)
        row = {"ablation": tag, "epoch": epoch}
        for col in METRIC_COLUMNS:
            row[col] = metrics.get(col)
        rows.append(row)

    if not rows:
        raise SystemExit(
            f"No *_metrics.tsv files found under {root}. "
            "Eval jobs may not have finished yet -- check `squeue -u $USER`."
        )

    df = pd.DataFrame(rows)
    for col in ["gcg_asr", "autodan_asr", "pair_asr", "mean_asr", "frr"]:
        df[col] = pd.to_numeric(df[col], errors="coerce") * 100  # fractions -> percent
    for col in ["n_harmful", "n_benign"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    df = df.sort_values(["ablation", "epoch"]).reset_index(drop=True)

    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", lambda x: f"{x:.1f}")
    print(df.to_string(index=False))

    n_expected_tags = 4
    n_found_tags = df["ablation"].nunique()
    if n_found_tags < n_expected_tags:
        found = sorted(df["ablation"].unique())
        print(
            f"\n[note] found {n_found_tags}/{n_expected_tags} ablation tags so far: {found} "
            "-- some eval jobs may still be running/queued."
        )

    if args.csv_out:
        out_path = Path(args.csv_out)
        df.to_csv(out_path, index=False)
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
