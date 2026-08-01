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
    p.add_argument(
        "--keep-missing",
        action="store_true",
        help="Keep base rows with empty Jailbreak-R1 Variant (default: drop them).",
    )
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
    variants["goal"] = variants["goal"].astype(str)
    # Join key: test uses "goal"; train/val use "Original Prompt".
    if "goal" in base.columns:
        base = base.copy()
        base["goal"] = base["goal"].astype(str)
        join_key = "goal"
    elif "Original Prompt" in base.columns:
        base = base.copy()
        base["_jb_r1_join"] = base["Original Prompt"].astype(str)
        variants = variants.rename(columns={"goal": "_jb_r1_join"})
        join_key = "_jb_r1_join"
    else:
        raise SystemExit(f"base CSV needs goal or Original Prompt; cols={list(base.columns)}")

    join_cols = [join_key, "Jailbreak-R1 Variant"]
    merged = base.merge(variants[join_cols], on=join_key, how="left")
    if join_key == "_jb_r1_join":
        merged = merged.drop(columns=["_jb_r1_join"])
    missing = merged["Jailbreak-R1 Variant"].isna() | merged[
        "Jailbreak-R1 Variant"
    ].astype(str).str.strip().eq("")
    n_missing = int(missing.sum())
    if missing.any():
        miss_path = out_dir / "missing_goals.csv"
        cols = [c for c in ("goal", "Original Prompt", "target", "dataset") if c in merged.columns]
        merged.loc[missing, cols].to_csv(miss_path, index=False)
        print(f"WARNING: {n_missing} goals missing variants → {miss_path}")
        if not args.keep_missing:
            merged = merged.loc[~missing].reset_index(drop=True)
            print(f"Dropped {n_missing} rows without variants (eval-safe). Use --keep-missing to retain.")

    n_ok = int(merged["Jailbreak-R1 Variant"].notna().sum())
    n_nonempty = int(
        merged["Jailbreak-R1 Variant"].fillna("").astype(str).str.strip().ne("").sum()
    )

    merged_name = Path(args.base_csv).stem + "_with_jailbreak_r1.csv"
    merged_path = out_dir / merged_name
    merged.to_csv(merged_path, index=False)

    print(f"chunks merged: {len(files)}")
    print(f"unique goals with variants: {len(variants)}")
    print(f"eval CSV rows: {len(merged)} (non-null={n_ok} non-empty={n_nonempty})")
    print(f"wrote {var_path}")
    print(f"wrote {merged_path}")


if __name__ == "__main__":
    main()
