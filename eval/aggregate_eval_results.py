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

# Default location is alongside this script (eval/), so the aggregator works
# out-of-the-box when run on a local laptop after copying CSVs down from the
# cluster. Pass --summary-file / --eval-output-root / --out-dir to override.
_HERE = Path(__file__).resolve().parent
DEFAULT_SUMMARY = _HERE / "summary.tsv"
DEFAULT_EVAL_ROOT = _HERE
DEFAULT_OUT_DIR = _HERE

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
        default=str(DEFAULT_SUMMARY),
        help="TSV produced by eval_sweep.py (append-only). "
             f"Default: {DEFAULT_SUMMARY}",
    )
    ap.add_argument(
        "--eval-output-root",
        default=str(DEFAULT_EVAL_ROOT),
        help="Directory containing *_harmful.csv / *_benign.csv (used if summary is missing). "
             f"Default: {DEFAULT_EVAL_ROOT}",
    )
    ap.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help=f"Where aggregated CSVs are written. Default: {DEFAULT_OUT_DIR}",
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

    # Top-10 best & worst by mean_asr at every evaluated epoch
    if "mean_asr" in df.columns:
        display_cols = [
            "slug",
            "mean_asr",
            "gcg_asr",
            "autodan_asr",
            "pair_asr",
            "refusal_rate",
        ]
        epochs = sorted(int(e) for e in df["epoch"].dropna().unique())
        leaderboard_rows: list[pd.DataFrame] = []
        for ep in epochs:
            sub = df[df["epoch"] == ep].copy()
            if sub.empty or sub["mean_asr"].isna().all():
                print(f"\nEpoch {ep}: no mean_asr values available, skipping leaderboard")
                continue
            sub = sub.sort_values("mean_asr", na_position="last")
            best = sub.head(10)
            worst = sub.dropna(subset=["mean_asr"]).tail(10).iloc[::-1]

            print(f"\nBest 10 configs at epoch {ep} (lowest mean_asr):")
            print(best[display_cols].to_string(index=False))
            print(f"\nWorst 10 configs at epoch {ep} (highest mean_asr):")
            print(worst[display_cols].to_string(index=False))

            best_lb = best[display_cols].assign(epoch=ep, rank_type="best")
            worst_lb = worst[display_cols].assign(epoch=ep, rank_type="worst")
            leaderboard_rows.append(best_lb)
            leaderboard_rows.append(worst_lb)

        if leaderboard_rows:
            leaderboard = pd.concat(leaderboard_rows, ignore_index=True)
            leaderboard = leaderboard[
                ["epoch", "rank_type"] + display_cols
            ]
            lb_path = out_dir / "eval_leaderboard.csv"
            leaderboard.to_csv(lb_path, index=False)
            print(f"\nWrote per-epoch best/worst leaderboard: {lb_path}")


if __name__ == "__main__":
    main()
