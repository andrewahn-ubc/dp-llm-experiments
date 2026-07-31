#!/bin/bash
# Build PAIR chunks for (full test − adaptive_test) goals.
#
#   bash helper_scripts/perturbation/prep_pair_test_remainder.sh
# → official_data/pair_test_rest/chunk_00.csv …  (~772 goals @ 5/chunk → 155 files)

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$SCRATCH/dp-llm-experiments}"
cd "$REPO_ROOT"

COMBINED="${COMBINED:-official_data/combined_test_dataset.csv}"
if [[ ! -f "$COMBINED" && -f official/combined_test_dataset.csv ]]; then
  COMBINED=official/combined_test_dataset.csv
fi
ADAPTIVE_DIR="${ADAPTIVE_DIR:-official_data/adaptive_test}"
if [[ ! -d "$ADAPTIVE_DIR" && -d official/splits/adaptive_test ]]; then
  ADAPTIVE_DIR=official/splits/adaptive_test
fi
OUT_DIR="${OUT_DIR:-official_data/pair_test_rest}"
CHUNK_SIZE="${CHUNK_SIZE:-5}"

python - <<PY
import pandas as pd
from pathlib import Path

combined = Path("$COMBINED")
adaptive_dir = Path("$ADAPTIVE_DIR")
out_dir = Path("$OUT_DIR")
chunk_size = int("$CHUNK_SIZE")

df = pd.read_csv(combined)
if "goal" not in df.columns:
    raise SystemExit(f"need goal in {combined}")
df = df.drop_duplicates("goal", keep="first")

adapt_files = sorted(adaptive_dir.glob("test_*.csv"))
if not adapt_files:
    raise SystemExit(f"no test_*.csv in {adaptive_dir}")
adapt = pd.concat([pd.read_csv(f) for f in adapt_files], ignore_index=True)
adapt_goals = set(adapt["goal"].astype(str))

rest = df[~df["goal"].astype(str).isin(adapt_goals)].copy()
keep = [c for c in ("goal", "target", "dataset") if c in rest.columns]
rest = rest[keep].reset_index(drop=True)
print(f"full={len(df)} adaptive={len(adapt_goals)} rest={len(rest)}")

out_dir.mkdir(parents=True, exist_ok=True)
for p in out_dir.glob("chunk_*.csv"):
    p.unlink()
n_chunks = (len(rest) + chunk_size - 1) // chunk_size
for i in range(n_chunks):
    sl = rest.iloc[i * chunk_size : (i + 1) * chunk_size]
    path = out_dir / f"chunk_{i:02d}.csv"
    sl.to_csv(path, index=False)
    print(f"wrote {path} ({len(sl)})")
(out_dir / "manifest.txt").write_text(
    f"rows={len(rest)}\nchunk_size={chunk_size}\nn_chunks={n_chunks}\narray=0-{n_chunks-1}\n"
)
print(f"manifest: array=0-{n_chunks-1}")
PY

echo
echo "Next:"
echo "  1) In \$SCRATCH/pair/pair.py set CSV to:"
echo "       os.environ.get('PAIR_DATASET_PATH', '${OUT_DIR}/chunk_00.csv')"
echo "     and --local-llama-path to your merged model (same as adaptive)."
echo "  2) mkdir -p logs && sbatch --array=\$(grep array= ${OUT_DIR}/manifest.txt | cut -d= -f2) helper_scripts/perturbation/pair_test_rest.sh"
