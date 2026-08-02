#!/usr/bin/env python3
"""Aggregate seed0 (image) + seed1/2 eval points into mean±std seen/heldout tables.

Reads:
  * ``seed0_from_image.csv`` (this directory)
  * ``$EVAL_OUT_DIR/seed{1,2}/points_*.csv`` from ``eval_one.py``

Writes (under ``--out-dir``):
  * ``seen_family_table_multiseed.md`` / ``.csv``
  * ``heldout_family_table_multiseed.md`` / ``.csv``
  * ``cells_long.csv`` (per-seed leaf cells + n)

ASR/FRR are in percent (matching the image). Sample std uses ``ddof=1``.
All Attacks = unweighted mean of GCG/AutoDAN/PAIR cells.
All Benchmarks = unweighted mean over AdvBench/HarmBench/JailbreakBench.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.llama3_multiseed.configs import (  # noqa: E402
    BENCHMARKS,
    FAMILIES,
    TABLE_CELLS,
)

BENCH_LABEL = {
    "advbench": "AdvBench",
    "harmbench": "HarmBench",
    "jailbreakbench": "JailbreakBench",
    "mean": "All Benchmarks",
}

SEEN_ROW_ORDER = ("all", "gcg", "autodan", "pair", "advsft")
HELDOUT_ROW_ORDER = (
    "all",
    "gcg",
    "autodan",
    "pair",
    "advsft_gcg",
    "advsft_autodan",
    "advsft_pair",
)
ROW_LABEL = {
    "all": "All Attacks",
    "gcg": "GCG",
    "autodan": "AutoDAN",
    "pair": "PAIR",
    "advsft": "Adv-SFT",
    "advsft_gcg": "Adv-SFT (GCG heldout)",
    "advsft_autodan": "Adv-SFT (AutoDAN heldout)",
    "advsft_pair": "Adv-SFT (PAIR heldout)",
}


def expand_path(p: str) -> str:
    return os.path.expandvars(os.path.expanduser(p))


def _fmt(mean: float | None, std: float | None, n: int) -> str:
    if mean is None or mean != mean:
        return "—"
    if n < 2 or std is None or std != std:
        return f"{mean:.1f} (n={n})"
    return f"{mean:.1f}±{std:.1f}"


def _mean_std(vals: list[float]) -> tuple[float | None, float | None, int]:
    good = [float(v) for v in vals if v is not None and v == v]
    n = len(good)
    if n == 0:
        return None, None, 0
    mean = float(np.mean(good))
    std = float(np.std(good, ddof=1)) if n >= 2 else None
    return mean, std, n


def _lookup_point(
    points: pd.DataFrame,
    *,
    seed: int,
    config_id: str,
    family: str,
    benchmark: str,
) -> tuple[float | None, float | None]:
    """Return (asr, frr) percent for one seed cell, or (None, None)."""
    if points.empty:
        return None, None
    # For Adv-SFT seen, points use family=all; for heldout Adv-SFT, family is attack name.
    fam = family
    if family == "all" and config_id == "seen_advsft":
        fam = "all"
    elif family.startswith("advsft_"):
        fam = family.split("_", 1)[1]  # advsft_gcg → gcg
    elif family == "advsft":
        fam = "all"

    sub = points[
        (points["seed"] == seed)
        & (points["config_id"] == config_id)
        & (points["family"] == fam)
        & (points["benchmark"] == benchmark)
    ]
    if sub.empty:
        # Fallback: any matching config_id + benchmark (heldout single-family files)
        sub = points[
            (points["seed"] == seed)
            & (points["config_id"] == config_id)
            & (points["benchmark"] == benchmark)
        ]
        if family in FAMILIES:
            sub = sub[sub["family"] == family]
    if sub.empty:
        return None, None
    row = sub.iloc[0]
    asr = row["asr"] if pd.notna(row["asr"]) else None
    frr = row["frr"] if pd.notna(row["frr"]) else None
    return (None if asr is None else float(asr), None if frr is None else float(frr))


def load_seed0(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["seed"] = 0
    return df


def load_eval_points(eval_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for seed in (1, 2):
        d = eval_root / f"seed{seed}"
        if not d.is_dir():
            print(f"[warn] missing {d}", flush=True)
            continue
        for p in sorted(d.glob("points_*.csv")):
            part = pd.read_csv(p)
            if "seed" not in part.columns:
                part["seed"] = seed
            frames.append(part)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def collect_leaf_cells(seed0: pd.DataFrame, points: pd.DataFrame) -> pd.DataFrame:
    """One row per (seed, mode, family, benchmark) leaf cell used in the tables."""
    rows: list[dict] = []
    for cell in TABLE_CELLS:
        assert cell.source_config_id
        for seed in (0, 1, 2):
            if seed == 0:
                sub = seed0[
                    (seed0["mode"] == cell.mode)
                    & (seed0["family"] == cell.family)
                    & (seed0["benchmark"] == cell.benchmark)
                ]
                # Adv-SFT has no seed0
                if cell.family.startswith("advsft") or cell.family == "advsft":
                    asr, frr = None, None
                elif sub.empty:
                    asr, frr = None, None
                else:
                    asr = float(sub.iloc[0]["asr"])
                    frr = float(sub.iloc[0]["frr"])
            else:
                asr, frr = _lookup_point(
                    points,
                    seed=seed,
                    config_id=cell.source_config_id,
                    family=cell.family,
                    benchmark=cell.benchmark,
                )
            rows.append(
                {
                    "seed": seed,
                    "mode": cell.mode,
                    "family": cell.family,
                    "benchmark": cell.benchmark,
                    "config_id": cell.source_config_id,
                    "asr": asr,
                    "frr": frr,
                }
            )
    return pd.DataFrame(rows)


def _add_derived(leaf: pd.DataFrame) -> pd.DataFrame:
    """Add All Benchmarks (per family) and All Attacks (per benchmark) rows."""
    extra: list[dict] = []
    for mode in ("seen", "heldout"):
        fams = (
            list(FAMILIES) + ["advsft"]
            if mode == "seen"
            else list(FAMILIES) + ["advsft_gcg", "advsft_autodan", "advsft_pair"]
        )
        for seed in (0, 1, 2):
            # All Benchmarks per family
            for fam in fams:
                sub = leaf[
                    (leaf["mode"] == mode)
                    & (leaf["family"] == fam)
                    & (leaf["seed"] == seed)
                    & (leaf["benchmark"].isin(BENCHMARKS))
                ]
                if sub.empty:
                    continue
                asrs = [v for v in sub["asr"].tolist() if v is not None and v == v]
                frrs = [v for v in sub["frr"].tolist() if v is not None and v == v]
                extra.append(
                    {
                        "seed": seed,
                        "mode": mode,
                        "family": fam,
                        "benchmark": "mean",
                        "config_id": "",
                        "asr": float(np.mean(asrs)) if asrs else None,
                        "frr": float(np.mean(frrs)) if frrs else None,
                    }
                )
            # All Attacks = mean of GCG/AutoDAN/PAIR (not Adv-SFT)
            for bench in list(BENCHMARKS) + ["mean"]:
                if bench == "mean":
                    # mean of family×bench means, or mean of per-bench all-attack — use family means
                    fam_means = []
                    fam_frrs = []
                    for fam in FAMILIES:
                        sub = leaf[
                            (leaf["mode"] == mode)
                            & (leaf["family"] == fam)
                            & (leaf["seed"] == seed)
                            & (leaf["benchmark"].isin(BENCHMARKS))
                        ]
                        asrs = [v for v in sub["asr"].tolist() if v is not None and v == v]
                        frrs = [v for v in sub["frr"].tolist() if v is not None and v == v]
                        if asrs:
                            fam_means.append(float(np.mean(asrs)))
                        if frrs:
                            fam_frrs.append(float(np.mean(frrs)))
                    extra.append(
                        {
                            "seed": seed,
                            "mode": mode,
                            "family": "all",
                            "benchmark": "mean",
                            "config_id": "",
                            "asr": float(np.mean(fam_means)) if fam_means else None,
                            "frr": float(np.mean(fam_frrs)) if fam_frrs else None,
                        }
                    )
                    continue
                sub = leaf[
                    (leaf["mode"] == mode)
                    & (leaf["family"].isin(FAMILIES))
                    & (leaf["seed"] == seed)
                    & (leaf["benchmark"] == bench)
                ]
                asrs = [v for v in sub["asr"].tolist() if v is not None and v == v]
                frrs = [v for v in sub["frr"].tolist() if v is not None and v == v]
                extra.append(
                    {
                        "seed": seed,
                        "mode": mode,
                        "family": "all",
                        "benchmark": bench,
                        "config_id": "",
                        "asr": float(np.mean(asrs)) if asrs else None,
                        "frr": float(np.mean(frrs)) if frrs else None,
                    }
                )
    return pd.concat([leaf, pd.DataFrame(extra)], ignore_index=True)


def build_table(cells: pd.DataFrame, mode: str, row_order: tuple[str, ...]) -> pd.DataFrame:
    benches = list(BENCHMARKS) + ["mean"]
    out_rows: list[dict] = []
    for fam in row_order:
        row: dict = {"Attack": ROW_LABEL.get(fam, fam)}
        for bench in benches:
            asrs, frrs = [], []
            for seed in (0, 1, 2):
                sub = cells[
                    (cells["mode"] == mode)
                    & (cells["family"] == fam)
                    & (cells["benchmark"] == bench)
                    & (cells["seed"] == seed)
                ]
                if sub.empty:
                    continue
                a, f = sub.iloc[0]["asr"], sub.iloc[0]["frr"]
                if a is not None and a == a:
                    asrs.append(float(a))
                if f is not None and f == f:
                    frrs.append(float(f))
            am, as_, an = _mean_std(asrs)
            fm, fs, fn = _mean_std(frrs)
            n = min(an, fn) if an and fn else max(an, fn)
            label = BENCH_LABEL[bench]
            row[f"{label} ASR"] = _fmt(am, as_, an)
            row[f"{label} FRR"] = _fmt(fm, fs, fn)
            row[f"{label}"] = (
                f"{_fmt(am, as_, an)} / {_fmt(fm, fs, fn)}" if n else "—"
            )
            row[f"_{bench}_asr_mean"] = am
            row[f"_{bench}_asr_std"] = as_
            row[f"_{bench}_frr_mean"] = fm
            row[f"_{bench}_frr_std"] = fs
            row[f"_{bench}_n"] = n
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def to_markdown(df: pd.DataFrame, title: str) -> str:
    cols = ["Attack"] + [BENCH_LABEL[b] for b in list(BENCHMARKS) + ["mean"]]
    lines = [f"# {title}", "", "| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        cells = [str(r["Attack"])] + [str(r[BENCH_LABEL[b]]) for b in list(BENCHMARKS) + ["mean"]]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "Values are ASR/FRR in percent as `mean±std` over seeds "
        "(seed0=image for DCL cells; Adv-SFT has n≤2). Sample std with ddof=1."
    )
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--seed0-csv",
        default=str(_HERE / "seed0_from_image.csv"),
    )
    p.add_argument(
        "--eval-root",
        default=os.environ.get("EVAL_OUT_DIR", "$SCRATCH/dp-llm-eval/llama3_multiseed"),
    )
    p.add_argument(
        "--out-dir",
        default="",
        help="Default: <eval-root>/aggregate",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    seed0_path = Path(expand_path(args.seed0_csv))
    eval_root = Path(expand_path(args.eval_root))
    out_dir = Path(expand_path(args.out_dir)) if args.out_dir else eval_root / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)

    seed0 = load_seed0(seed0_path)
    points = load_eval_points(eval_root)
    print(f"[aggregate] seed0 rows={len(seed0)} eval_points={len(points)}", flush=True)

    leaf = collect_leaf_cells(seed0, points)
    cells = _add_derived(leaf)
    cells.to_csv(out_dir / "cells_long.csv", index=False)

    seen = build_table(cells, "seen", SEEN_ROW_ORDER)
    held = build_table(cells, "heldout", HELDOUT_ROW_ORDER)

    # Compact CSV: Attack + combined ASR/FRR strings
    def compact(df: pd.DataFrame) -> pd.DataFrame:
        cols = ["Attack"] + [BENCH_LABEL[b] for b in list(BENCHMARKS) + ["mean"]]
        return df[cols]

    compact(seen).to_csv(out_dir / "seen_family_table_multiseed.csv", index=False)
    compact(held).to_csv(out_dir / "heldout_family_table_multiseed.csv", index=False)
    (out_dir / "seen_family_table_multiseed.md").write_text(
        to_markdown(seen, "Seen-family (Llama-3 multi-seed)"), encoding="utf-8"
    )
    (out_dir / "heldout_family_table_multiseed.md").write_text(
        to_markdown(held, "Heldout-family (Llama-3 multi-seed)"), encoding="utf-8"
    )
    # Full numeric dump
    seen.to_csv(out_dir / "seen_family_table_multiseed_full.csv", index=False)
    held.to_csv(out_dir / "heldout_family_table_multiseed_full.csv", index=False)

    print(f"[aggregate] Wrote tables under {out_dir}", flush=True)
    print(to_markdown(seen, "Seen-family (preview)"))
    print(to_markdown(held, "Heldout-family (preview)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
