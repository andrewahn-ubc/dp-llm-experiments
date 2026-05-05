#!/usr/bin/env python3
"""Ensure a benign CSV has an ``Original Prompt`` column for ``eval/eval.py`` FRR eval."""

from __future__ import annotations

import argparse
import sys

import pandas as pd

_PROMPT_CANDIDATES = [
    "adversarial",
    "Adversarial",
    "goal",
    "Goal",
    "prompt",
    "Prompt",
    "original_prompt",
    "instruction",
    "Instruction",
]


def prep(src: str, dst: str) -> None:
    df = pd.read_csv(src)
    if "Original Prompt" in df.columns:
        df.to_csv(dst, index=False)
        print(f"[prep_frr_eval_input] already has 'Original Prompt' -> {dst}", flush=True)
        return

    src_col = next((c for c in _PROMPT_CANDIDATES if c in df.columns), None)
    if src_col is None:
        raise ValueError(
            "Could not build 'Original Prompt' column for FRR input. "
            f"Columns found: {list(df.columns)}"
        )

    out = df.copy()
    out["Original Prompt"] = out[src_col].astype(str)
    out.to_csv(dst, index=False)
    print(f"[prep_frr_eval_input] mapped {src_col!r} -> 'Original Prompt' -> {dst}", flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("src_csv", help="Benign CSV (e.g. frr_test.csv).")
    p.add_argument("dst_csv", help="Output CSV path (e.g. under $SLURM_TMPDIR).")
    args = p.parse_args(argv)
    try:
        prep(args.src_csv, args.dst_csv)
    except Exception as e:
        print(f"[prep_frr_eval_input] error: {e}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
