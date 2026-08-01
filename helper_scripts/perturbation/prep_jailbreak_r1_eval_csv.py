#!/usr/bin/env python3
"""Map Jailbreak-R1 Variant → GCG Variant so eval.py unseen-family can score it.

  python helper_scripts/perturbation/prep_jailbreak_r1_eval_csv.py \\
    --input official_data/jailbreak_r1/combined_test_dataset_with_jailbreak_r1.csv \\
    --output official_data/jailbreak_r1/jb_r1_as_gcg_for_eval.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        type=Path,
        default=Path("official_data/jailbreak_r1/combined_test_dataset_with_jailbreak_r1.csv"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("official_data/jailbreak_r1/jb_r1_as_gcg_for_eval.csv"),
    )
    args = p.parse_args()

    df = pd.read_csv(args.input)
    if "Jailbreak-R1 Variant" not in df.columns:
        raise SystemExit(
            f"Missing 'Jailbreak-R1 Variant' in {args.input}; columns={list(df.columns)}"
        )
    out = df.copy()
    out["GCG Variant"] = out["Jailbreak-R1 Variant"].astype(str)
    # Drop other attack cols so seen-family mode can't accidentally score them;
    # unseen-family gcg only reads GCG Variant anyway.
    for c in ("AutoDAN Variant", "PAIR Variant"):
        if c in out.columns:
            out = out.drop(columns=[c])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    n = out["GCG Variant"].astype(str).str.strip().ne("").sum()
    print(f"wrote {args.output}  rows={len(out)}  non-empty Jailbreak-R1 variants={n}")


if __name__ == "__main__":
    main()
