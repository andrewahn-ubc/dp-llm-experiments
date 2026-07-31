#!/usr/bin/env python3
"""Split combined_test_dataset.csv into fixed-size chunks for Jailbreak-R1 jobs.

Default: 20 goals/chunk → 52 chunks for the 1022-goal test set.

Example::

  python helper_scripts/perturbation/prepare_jailbreak_r1_chunks.py \\
    --input official_data/combined_test_dataset.csv \\
    --out-dir official_data/jailbreak_r1_test \\
    --chunk-size 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        default="official_data/combined_test_dataset.csv",
        help="CSV with at least a 'goal' column (and ideally target/dataset). "
        "On Narval this is official_data/ (local checkout may use official/).",
    )
    p.add_argument(
        "--out-dir",
        default="official_data/jailbreak_r1_test",
        help="Directory for chunk_XX.csv files.",
    )
    p.add_argument("--chunk-size", type=int, default=20)
    args = p.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    if "goal" not in df.columns:
        raise SystemExit(f"Need goal column; got {list(df.columns)}")
    # Deduplicate goals; keep first target/dataset.
    keep = [c for c in ("goal", "target", "dataset") if c in df.columns]
    df = df[keep].drop_duplicates(subset=["goal"], keep="first").reset_index(drop=True)

    n = len(df)
    n_chunks = (n + args.chunk_size - 1) // args.chunk_size
    for i in range(n_chunks):
        sl = df.iloc[i * args.chunk_size : (i + 1) * args.chunk_size]
        path = out_dir / f"chunk_{i:02d}.csv"
        sl.to_csv(path, index=False)
        print(f"wrote {path} ({len(sl)} rows)")

    manifest = out_dir / "manifest.txt"
    manifest.write_text(
        f"input={in_path.resolve()}\nn_goals={n}\nchunk_size={args.chunk_size}\nn_chunks={n_chunks}\n"
        f"array=0-{n_chunks - 1}\n",
        encoding="utf-8",
    )
    print(f"wrote {manifest}")
    print(f"Submit with: #SBATCH --array=0-{n_chunks - 1}")


if __name__ == "__main__":
    main()
