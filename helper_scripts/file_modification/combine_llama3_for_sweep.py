"""
Finalize the Llama 3 sweep CSVs so ``python run_final_pipeline.py --model
llama_3_8b_instruct`` works end to end.

This merges the base-model "Original Response" completions into the
train / validation / train+validation splits and guarantees the exact column
names train.py and eval.py require. It is idempotent: safe to re-run.

train.py drops any row missing one of (see train/train.py ``required_cols``):
    Original Prompt, Original Response, GCG Variant, AutoDAN Variant, PAIR Variant

Inputs (in --data-dir, default official_data/):
  - llama3_train.csv, llama3_validation.csv, llama3_train_plus_validation.csv
        each has: goal, GCG Variant, AutoDAN Variant, PAIR Variant
        (from combine_llama3_jb_fams.py + split_llama3_train_val.py)
  - llama3_original_responses.csv  (--responses; optional if the CSVs already
        carry an Original Response column)
        has: goal, Original Response
        (from helper_scripts/inference/inference.py)

Outputs: the same CSVs rewritten in place with columns ordered
    goal, Original Prompt, Original Response, GCG Variant, AutoDAN Variant, PAIR Variant
Rows still missing any required column are reported, then dropped (matching train.py).

Usage (from repo root on the cluster)::

    python helper_scripts/file_modification/combine_llama3_for_sweep.py
    python helper_scripts/file_modification/combine_llama3_for_sweep.py \
        --data-dir $SCRATCH/dp-llm-experiments/official_data
"""

import argparse
import os
import sys

import pandas as pd

# helper_scripts/file_modification/<this file> -> repo root is three levels up.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

PROMPT_COL = "goal"
RESPONSE_COL = "Original Response"
VARIANT_COLS = ["GCG Variant", "AutoDAN Variant", "PAIR Variant"]
# Columns train.py requires (must be non-null per row or the row is dropped).
REQUIRED_COLS = ["Original Prompt", RESPONSE_COL] + VARIANT_COLS
# Preferred on-disk column order (extras are appended after these).
COLUMN_ORDER = [PROMPT_COL, "Original Prompt", RESPONSE_COL] + VARIANT_COLS

TARGET_FILES = ["llama3_train.csv", "llama3_validation.csv", "llama3_train_plus_validation.csv"]


def expand(p: str) -> str:
    return os.path.expandvars(os.path.expanduser(p))


def load_responses(path: str) -> dict[str, str]:
    """Return {stripped goal -> Original Response} from the standalone lookup CSV."""
    rdf = pd.read_csv(path)
    prompt_col = PROMPT_COL if PROMPT_COL in rdf.columns else None
    if prompt_col is None:
        for c in ("Original Prompt", "prompt", "Prompt", "adversarial", "instruction"):
            if c in rdf.columns:
                prompt_col = c
                break
    if prompt_col is None:
        raise ValueError(
            f"{path}: cannot find a prompt column (have {list(rdf.columns)})"
        )
    response_col = RESPONSE_COL if RESPONSE_COL in rdf.columns else None
    if response_col is None:
        for c in ("response", "Response", "output", "completion"):
            if c in rdf.columns:
                response_col = c
                break
    if response_col is None:
        raise ValueError(
            f"{path}: cannot find a response column (have {list(rdf.columns)})"
        )

    mapping: dict[str, str] = {}
    for goal, resp in zip(rdf[prompt_col].astype(str), rdf[response_col]):
        key = goal.strip()
        if key and pd.notna(resp):
            mapping[key] = resp
    print(f"[combine] loaded {len(mapping)} responses from {path}", flush=True)
    return mapping


def reorder(df: pd.DataFrame) -> pd.DataFrame:
    front = [c for c in COLUMN_ORDER if c in df.columns]
    rest = [c for c in df.columns if c not in front]
    return df[front + rest]


def finalize_one(path: str, responses: dict[str, str] | None) -> bool:
    """Add Original Prompt + Original Response, validate, drop incomplete rows.

    Returns True if the written file has every REQUIRED_COLS column present.
    """
    print(f"\n{'=' * 70}\n{os.path.basename(path)}\n{'=' * 70}")
    df = pd.read_csv(path)
    n_start = len(df)

    if PROMPT_COL not in df.columns:
        raise ValueError(
            f"{path}: missing {PROMPT_COL!r} column (have {list(df.columns)}). "
            "Run combine_llama3_jb_fams.py + split_llama3_train_val.py first."
        )

    # Original Prompt is the clean goal (never the perturbed variant).
    df["Original Prompt"] = df[PROMPT_COL].astype(str)

    # Original Response: prefer the lookup; fall back to any existing column value.
    if responses is not None:
        mapped = df[PROMPT_COL].astype(str).str.strip().map(responses)
        if RESPONSE_COL in df.columns:
            df[RESPONSE_COL] = mapped.where(mapped.notna(), df[RESPONSE_COL])
        else:
            df[RESPONSE_COL] = mapped
    elif RESPONSE_COL not in df.columns:
        raise ValueError(
            f"{path}: no --responses lookup given and no existing {RESPONSE_COL!r} "
            "column. Generate responses first (helper_scripts/inference/inference.py)."
        )

    missing_cols = [c for c in VARIANT_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"{path}: missing variant column(s) {missing_cols} (have {list(df.columns)})."
        )

    # Treat empty / whitespace-only strings as missing before the NaN drop.
    for c in REQUIRED_COLS:
        s = df[c].astype("object")
        df[c] = s.where(s.isna(), s.astype(str).str.strip()).replace({"": None})

    for c in REQUIRED_COLS:
        print(f"  missing {c}: {int(df[c].isna().sum())}")

    complete = df.dropna(subset=REQUIRED_COLS).reset_index(drop=True)
    complete = reorder(complete)
    complete.to_csv(path, index=False)

    ok = all(c in complete.columns for c in REQUIRED_COLS)
    print(
        f"  rows: {n_start} -> {len(complete)} written "
        f"({n_start - len(complete)} dropped for missing required values)"
    )
    print(f"  columns: {list(complete.columns)}")
    print(f"  train.py-ready: {'YES' if ok and len(complete) else 'NO'}")
    return ok and len(complete) > 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--data-dir",
        default=os.path.join(_REPO_ROOT, "official_data"),
        help="Directory holding the llama3_*.csv splits (default: <repo>/official_data).",
    )
    p.add_argument(
        "--responses",
        default=None,
        help="Lookup CSV (goal, Original Response). Default: <data-dir>/llama3_original_responses.csv "
        "if it exists; otherwise reuse an existing Original Response column.",
    )
    p.add_argument(
        "--files",
        default=",".join(TARGET_FILES),
        help="Comma-separated CSV names (in --data-dir) to finalize.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = expand(args.data_dir)

    responses_path = expand(args.responses) if args.responses else os.path.join(
        data_dir, "llama3_original_responses.csv"
    )
    responses = load_responses(responses_path) if os.path.exists(responses_path) else None
    if responses is None:
        print(
            f"[combine] no responses lookup at {responses_path}; "
            "will reuse existing Original Response columns if present.",
            flush=True,
        )

    all_ok = True
    for name in [x.strip() for x in args.files.split(",") if x.strip()]:
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            print(f"\n[combine] SKIP (not found): {path}", flush=True)
            # train + validation are required for the sweep; the pool file is optional.
            if name in ("llama3_train.csv", "llama3_validation.csv"):
                all_ok = False
            continue
        all_ok = finalize_one(path, responses) and all_ok

    print(f"\n{'=' * 70}")
    if all_ok:
        print("DONE. Llama 3 train/validation CSVs are ready for the sweep. Run:")
        print("  python run_final_pipeline.py --model llama_3_8b_instruct")
    else:
        print(
            "INCOMPLETE. At least one required file is missing columns/rows. "
            "See the warnings above (generate responses or re-run the combine/split steps)."
        )
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
