#!/usr/bin/env python3
"""Merge Jailbreak-R1 chunk CSVs and join onto combined_test_dataset.csv.

Writes:
  - jailbreak_r1_variants.csv  (goal, Jailbreak-R1 Variant, …)
  - combined_test_with_jailbreak_r1.csv  (original cols + Jailbreak-R1 Variant)

Example::

  python helper_scripts/perturbation/merge_jailbreak_r1_outputs.py \\
    --chunks-dir $SCRATCH/dp-llm-experiments/official_data/jailbreak_r1_out \\
    --base-csv official_data/combined_test_dataset.csv \\
    --out-dir official_data/jailbreak_r1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chunks-dir", required=True)
    p.add_argument(
        "--base-csv",
        default="official_data/combined_test_dataset.csv",
        help="Test set to join Jailbreak-R1 Variant onto (Narval: official_data/).",
    )
    p.add_argument("--out-dir", default="official_data/jailbreak_r1")
    p.add_argument("--pattern", default="chunk_*.csv")
    args = p.parse_args()

    chunks_dir = Path(args.chunks_dir)
    files = sorted(chunks_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files matching {args.pattern} in {chunks_dir}")

    parts = [pd.read_csv(f) for f in files]
    variants = pd.concat(parts, ignore_index=True)
    if "goal" not in variants.columns or "Jailbreak-R1 Variant" not in variants.columns:
        raise SystemExit(f"Unexpected columns: {list(variants.columns)}")

    # Prefer successfully parsed rows if duplicates.
    variants["_ok"] = variants.get("Jailbreak-R1 Parsed", True)
    if "Jailbreak-R1 Parsed" in variants.columns:
        variants = variants.sort_values("_ok", ascending=False)
    variants = variants.drop_duplicates(subset=["goal"], keep="first").drop(columns=["_ok"], errors="ignore")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    var_path = out_dir / "jailbreak_r1_variants.csv"
    variants.to_csv(var_path, index=False)

    base = pd.read_csv(args.base_csv)
    base["goal"] = base["goal"].astype(str)
    variants["goal"] = variants["goal"].astype(str)
    join_cols = ["goal", "Jailbreak-R1 Variant"]
    merged = base.merge(variants[join_cols], on="goal", how="left")
    n_ok = int(merged["Jailbreak-R1 Variant"].notna().sum())
    n_nonempty = int(merged["Jailbreak-R1 Variant"].fillna("").astype(str).str.strip().ne("").sum())

    merged_path = out_dir / "combined_test_with_jailbreak_r1.csv"
    merged.to_csv(merged_path, index=False)

    print(f"chunks merged: {len(files)}")
    print(f"unique goals with variants: {len(variants)}")
    print(f"joined onto base ({len(base)} rows): non-null={n_ok} non-empty={n_nonempty}")
    print(f"wrote {var_path}")
    print(f"wrote {merged_path}")
    missing = merged["Jailbreak-R1 Variant"].isna() | merged["Jailbreak-R1 Variant"].astype(str).str.strip().eq("")
    if missing.any():
        miss_path = out_dir / "missing_goals.csv"
        merged.loc[missing, ["goal"] + [c for c in ("target", "dataset") if c in merged.columns]].to_csv(
            miss_path, index=False
        )
        print(f"WARNING: {int(missing.sum())} goals missing variants → {miss_path}")


if __name__ == "__main__":
    main()
