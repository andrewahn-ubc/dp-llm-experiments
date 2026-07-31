#!/usr/bin/env python3
"""Build the PyRIT eval subset: 225 advbench + 225 harmbench + 50 jailbreakbench (=500).

If ``adaptive_test`` (100+100+50) is present, keep those goals and add 125 more
advbench + 125 more harmbench from ``combined_test_dataset.csv`` (excluding
already-selected goals). Otherwise sample 225/225/50 with a fixed seed.

Example::

  python experiments/pyrit_rorqual/prepare_pyrit_subset.py \\
    --out official_data/pyrit_test/pyrit_subset_500.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

TARGET_COUNTS = {"advbench": 225, "harmbench": 225, "jailbreakbench": 50}
ADAPTIVE_COUNTS = {"advbench": 100, "harmbench": 100, "jailbreakbench": 50}


def _load_adaptive(adaptive_dir: Path) -> pd.DataFrame | None:
    files = sorted(adaptive_dir.glob("test_*.csv"))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if "goal" not in df.columns or "dataset" not in df.columns:
        return None
    df = df.drop_duplicates(subset=["goal"], keep="first").reset_index(drop=True)
    return df


def _sample_extra(
    pool: pd.DataFrame,
    dataset: str,
    n: int,
    exclude_goals: set[str],
    seed: int,
) -> pd.DataFrame:
    sub = pool[
        (pool["dataset"].astype(str) == dataset)
        & (~pool["goal"].astype(str).isin(exclude_goals))
    ]
    if len(sub) < n:
        raise SystemExit(
            f"need {n} extra {dataset} (excluding {len(exclude_goals)} taken), found {len(sub)}"
        )
    return sub.sample(n=n, random_state=seed)


def build_subset(
    *,
    adaptive_dir: Path | None,
    combined: Path,
    seed: int,
) -> pd.DataFrame:
    keep_cols_pref = ("goal", "target", "dataset")

    combined_df = pd.read_csv(combined)
    if "goal" not in combined_df.columns or "dataset" not in combined_df.columns:
        raise SystemExit(f"need goal+dataset in {combined}")
    combined_df = combined_df.drop_duplicates(subset=["goal"], keep="first")

    adaptive = _load_adaptive(adaptive_dir) if adaptive_dir and adaptive_dir.is_dir() else None
    parts: list[pd.DataFrame] = []

    if adaptive is not None and len(adaptive) > 0:
        print(f"base adaptive_test from {adaptive_dir}: n={len(adaptive)}")
        taken = set(adaptive["goal"].astype(str))
        parts.append(adaptive)
        # Add extras for advbench / harmbench; jbb stays at 50 from adaptive.
        for ds, target_n in TARGET_COUNTS.items():
            have = int((adaptive["dataset"].astype(str) == ds).sum())
            need = target_n - have
            if need <= 0:
                continue
            extra = _sample_extra(combined_df, ds, need, taken, seed=seed + hash(ds) % 1000)
            taken |= set(extra["goal"].astype(str))
            parts.append(extra)
            print(f"  +{need} {ds} from combined")
    else:
        print(f"sampling {TARGET_COUNTS} from {combined} seed={seed}")
        for ds, n in TARGET_COUNTS.items():
            sub = combined_df[combined_df["dataset"].astype(str) == ds]
            if len(sub) < n:
                raise SystemExit(f"need {n} {ds}, found {len(sub)}")
            parts.append(sub.sample(n=n, random_state=seed + hash(ds) % 1000))

    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates(subset=["goal"], keep="first")
    keep = [c for c in keep_cols_pref if c in out.columns]
    out = out[keep]
    order = {k: i for i, k in enumerate(TARGET_COUNTS)}
    out["_ord"] = out["dataset"].map(order)
    out = out.sort_values(["_ord"]).drop(columns=["_ord"]).reset_index(drop=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adaptive-dir", default="official/splits/adaptive_test")
    p.add_argument("--combined", default="official_data/combined_test_dataset.csv")
    p.add_argument("--out", default="official_data/pyrit_test/pyrit_subset_500.csv")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    adaptive_candidates = [
        Path(args.adaptive_dir),
        Path("official_data/adaptive_test"),
        Path("official/splits/adaptive_test"),
    ]
    adaptive_dir = next((d for d in adaptive_candidates if d.is_dir()), None)

    combined = Path(args.combined)
    if not combined.is_file():
        alt = Path("official/combined_test_dataset.csv")
        if alt.is_file():
            combined = alt

    df = build_subset(adaptive_dir=adaptive_dir, combined=combined, seed=args.seed)
    vc = df["dataset"].astype(str).value_counts().to_dict()
    print("counts:", vc, "total:", len(df))
    if len(df) != 500:
        print(f"[warn] expected 500 rows, got {len(df)}")
    for k, n in TARGET_COUNTS.items():
        got = vc.get(k, 0)
        if got != n:
            print(f"[warn] {k}: expected {n}, got {got}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
