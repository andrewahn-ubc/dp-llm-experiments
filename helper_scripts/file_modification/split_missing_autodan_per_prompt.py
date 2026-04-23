"""Split official/missing_from_autodan.csv into one CSV per prompt.

Each output file has columns `goal,target` (target pulled from
official/test_behaviours.csv) so AutoDAN's autodan_hga_eval.py can consume it
directly. Output files are named missing_autodan_<IDX>.csv with 2-digit padding.
"""

import math
from pathlib import Path

import pandas as pd

OFFICIAL = Path(__file__).resolve().parents[2] / "official"
MISSING = OFFICIAL / "missing_from_autodan.csv"
REFERENCE = OFFICIAL / "test_behaviours.csv"
OUT_DIR = OFFICIAL / "missing_autodan_splits"

OUT_DIR.mkdir(parents=True, exist_ok=True)

missing = pd.read_csv(MISSING)
ref = pd.read_csv(REFERENCE)[["goal", "target"]]

merged = missing.merge(ref, on="goal", how="left")
unmatched = merged[merged["target"].isna()]
if len(unmatched) > 0:
    print(f"WARNING: {len(unmatched)} goals had no target in {REFERENCE.name}")
    for g in unmatched["goal"].head(3):
        print(f"  - {g[:80]}...")

n = len(merged)
width = max(2, int(math.log10(max(n - 1, 1))) + 1)

for i, row in enumerate(merged.itertuples(index=False)):
    idx = str(i).zfill(width)
    out_path = OUT_DIR / f"missing_autodan_{idx}.csv"
    pd.DataFrame([row._asdict()]).to_csv(out_path, index=False)

print(f"Wrote {n} per-prompt CSVs to {OUT_DIR}")
print(f"SLURM array range: 0-{n - 1}")
