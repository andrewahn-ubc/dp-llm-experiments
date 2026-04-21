#!/usr/bin/env python3
"""
Aggregate per-checkpoint eval summaries into a single CSV.

Reads the append-only summary TSV produced by eval/eval_sweep.py (one row per
(slug, epoch) combination), deduplicates on the (slug, epoch) key (keeping the
last entry), and writes:

  * `<out_dir>/eval_summary.csv`  — tidy, one row per evaluated checkpoint
  * `<out_dir>/eval_pivot_asr.csv` — wide table (slug × epoch) of mean ASR
  * `<out_dir>/eval_pivot_refusal.csv` — wide table (slug × epoch) of refusal rate

Falls back to scanning a directory of enriched CSVs (`*_harmful.csv` /
`*_benign.csv`) if a summary TSV is unavailable — useful if jobs died after
writing outputs but before appending a row.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import pandas as pd

HARMFUL_RE = re.compile(r"(?P<slug>.+)_epoch(?P<epoch>\d+)_harmful\.csv$")
BENIGN_RE = re.compile(r"(?P<slug>.+)_epoch(?P<epoch>\d+)_benign\.csv$")

HARMBENCH_COLS = {
    "gcg": "GCG HarmBench",
    "autodan": "AutoDAN HarmBench",
    "pair": "PAIR HarmBench",
}


def _asr(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return float("nan")
    return float((df[col].astype(str).str.lower() == "yes").mean())


def _refusal_rate(df: pd.DataFrame) -> float:
    if "Original Refusal" not in df.columns:
        return float("nan")
    return float((df["Original Refusal"].astype(str).str.lower() == "refusal").mean())


def rebuild_from_csvs(eval_output_root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    harmful_files = {}
    benign_files = {}
    for p in eval_output_root.glob("*_harmful.csv"):
        m = HARMFUL_RE.match(p.name)
        if m:
            harmful_files[(m.group("slug"), int(m.group("epoch")))] = p
    for p in eval_output_root.glob("*_benign.csv"):
        m = BENIGN_RE.match(p.name)
        if m:
            benign_files[(m.group("slug"), int(m.group("epoch")))] = p

    keys = sorted(set(harmful_files.keys()) | set(benign_files.keys()))
    for slug, epoch in keys:
        row: dict = {"slug": slug, "epoch": epoch}
        if (slug, epoch) in harmful_files:
            hp = harmful_files[(slug, epoch)]
            hdf = pd.read_csv(hp)
            row["n_harmful"] = len(hdf)
            row["gcg_asr"] = _asr(hdf, HARMBENCH_COLS["gcg"])
            row["autodan_asr"] = _asr(hdf, HARMBENCH_COLS["autodan"])
            row["pair_asr"] = _asr(hdf, HARMBENCH_COLS["pair"])
            asrs = [row[k] for k in ("gcg_asr", "autodan_asr", "pair_asr")]
            row["mean_asr"] = sum(asrs) / len(asrs) if all(a == a for a in asrs) else float("nan")
            row["harmful_csv"] = str(hp)
        if (slug, epoch) in benign_files:
            bp = benign_files[(slug, epoch)]
            bdf = pd.read_csv(bp)
            row["n_benign"] = len(bdf)
            row["refusal_rate"] = _refusal_rate(bdf)
            row["benign_csv"] = str(bp)
        rows.append(row)

    return pd.DataFrame(rows)


def parse_slug(slug: str) -> dict:
    m = re.match(r"run_lr([^_]+)_lam([^_]+)_eps(.+)$", slug)
    if not m:
        return {"lr": float("nan"), "lambda_val": float("nan"), "epsilon": float("nan")}
    try:
        return {
            "lr": float(m.group(1)),
            "lambda_val": float(m.group(2)),
            "epsilon": float(m.group(3)),
        }
    except ValueError:
        return {"lr": float("nan"), "lambda_val": float("nan"), "epsilon": float("nan")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--summary-file",
        default=os.path.expandvars("$SCRATCH/dp-llm-eval/summary.tsv"),
        help="TSV produced by eval_sweep.py (append-only).",
    )
    ap.add_argument(
        "--eval-output-root",
        default=os.path.expandvars("$SCRATCH/dp-llm-eval"),
        help="Directory containing *_harmful.csv / *_benign.csv (used if summary is missing).",
    )
    ap.add_argument(
        "--out-dir",
        default=os.path.expandvars("$SCRATCH/dp-llm-eval"),
        help="Where aggregated CSVs are written.",
    )
    args = ap.parse_args()

    summary_path = Path(args.summary_file)
    if summary_path.exists():
        df = pd.read_csv(summary_path, sep="\t")
        print(f"Loaded {len(df)} rows from {summary_path}")
    else:
        print(f"Summary not found at {summary_path}; rebuilding from {args.eval_output_root}")
        df = rebuild_from_csvs(Path(args.eval_output_root))
        print(f"Rebuilt {len(df)} rows from enriched CSVs")

    if df.empty:
        print("No data to aggregate.")
        return

    # Ensure hyperparameter columns are present (fill from slug if missing)
    for key in ("lr", "lambda_val", "epsilon"):
        if key not in df.columns:
            df[key] = float("nan")
    missing_hp = df["lr"].isna() | df["lambda_val"].isna() | df["epsilon"].isna()
    if missing_hp.any():
        parsed = df.loc[missing_hp, "slug"].apply(parse_slug).apply(pd.Series)
        for col in ("lr", "lambda_val", "epsilon"):
            df.loc[missing_hp, col] = parsed[col]

    df = df.sort_values(["slug", "epoch"]).drop_duplicates(
        ["slug", "epoch"], keep="last"
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tidy_path = out_dir / "eval_summary.csv"
    df.to_csv(tidy_path, index=False)
    print(f"Wrote tidy summary: {tidy_path} ({len(df)} rows)")

    if "mean_asr" in df.columns:
        pivot_asr = df.pivot_table(index="slug", columns="epoch", values="mean_asr")
        pivot_asr.to_csv(out_dir / "eval_pivot_asr.csv")
        print(f"Wrote ASR pivot: {out_dir / 'eval_pivot_asr.csv'}")
    if "refusal_rate" in df.columns:
        pivot_ref = df.pivot_table(index="slug", columns="epoch", values="refusal_rate")
        pivot_ref.to_csv(out_dir / "eval_pivot_refusal.csv")
        print(f"Wrote refusal pivot: {out_dir / 'eval_pivot_refusal.csv'}")

    # Per-family pivot for ASR
    for fam in ("gcg", "autodan", "pair"):
        col = f"{fam}_asr"
        if col in df.columns:
            pvt = df.pivot_table(index="slug", columns="epoch", values=col)
            pvt.to_csv(out_dir / f"eval_pivot_{fam}_asr.csv")
            print(f"Wrote {fam} ASR pivot: {out_dir / f'eval_pivot_{fam}_asr.csv'}")

    # Top-10 best & worst by mean_asr at the latest available epoch
    if "mean_asr" in df.columns:
        last_epoch = int(df["epoch"].max())
        last = df[df["epoch"] == last_epoch].sort_values("mean_asr")
        print(f"\nBest 10 configs at epoch {last_epoch} (lowest mean_asr):")
        print(
            last.head(10)[
                ["slug", "mean_asr", "gcg_asr", "autodan_asr", "pair_asr", "refusal_rate"]
            ].to_string(index=False)
        )
        print(f"\nWorst 10 configs at epoch {last_epoch} (highest mean_asr):")
        print(
            last.tail(10)[
                ["slug", "mean_asr", "gcg_asr", "autodan_asr", "pair_asr", "refusal_rate"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
