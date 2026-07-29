#!/usr/bin/env python3
"""
Build 32 ASR-vs-FRR Pareto charts from final-test ``points_*.csv`` files.

Layout (2 × 4 × 4 = 32)::

  mode ∈ {seen, heldout}
  family ∈ {all, gcg, autodan, pair}
  benchmark ∈ {advbench, harmbench, jailbreakbench, mean}

Each chart has one point per (λ, ε) cell (default 6), with the min-min Pareto
frontier drawn and labeled.

Usage::

  python eval/plot_final_test_pareto.py \\
    --points-dir $SCRATCH/dp-llm-sweep/final_test_outputs/lr2e-5
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

MODES = ("seen", "heldout")
FAMILIES = ("all", "gcg", "autodan", "pair")
BENCHMARKS = ("advbench", "harmbench", "jailbreakbench", "mean")


def expand_path(p: str) -> str:
    return os.path.expandvars(os.path.expanduser(p))


def pareto_indices(asr: list[float], frr: list[float]) -> list[int]:
    pts = [(i, a, f) for i, (a, f) in enumerate(zip(asr, frr)) if not (math.isnan(a) or math.isnan(f))]
    keep = []
    for i, a, f in pts:
        dominated = any(
            (a2 <= a and f2 <= f and (a2 < a or f2 < f)) for j, a2, f2 in pts if j != i
        )
        if not dominated:
            keep.append(i)
    keep.sort(key=lambda i: (frr[i], asr[i]))
    return keep


def load_points(points_dir: Path) -> pd.DataFrame:
    files = sorted(points_dir.glob("points_*.csv"))
    # Also accept nested cell dirs
    files += sorted(points_dir.glob("lam*_eps*/points.csv"))
    if not files:
        raise SystemExit(f"No points_*.csv (or lam*/points.csv) under {points_dir}")
    frames = [pd.read_csv(p) for p in files]
    df = pd.concat(frames, ignore_index=True)
    # De-dupe identical cells if both flat and nested copies exist
    df = df.drop_duplicates(subset=["mode", "family", "benchmark", "lambda", "epsilon"], keep="last")
    print(f"[plot] loaded {len(df)} rows from {len(files)} files", flush=True)
    return df


def plot_one(sub: pd.DataFrame, title: str, out_path: Path) -> None:
    # One row per (λ, ε)
    rows = []
    for (lam, eps), g in sub.groupby(["lambda", "epsilon"], dropna=False):
        # Should be a single row; if not, take first
        r = g.iloc[0]
        asr = float(r["asr"]) if pd.notna(r["asr"]) else float("nan")
        frr = float(r["frr"]) if pd.notna(r["frr"]) else float("nan")
        rows.append((float(lam), float(eps), asr, frr))
    if not rows:
        print(f"[plot] skip empty: {title}", flush=True)
        return

    lams = [r[0] for r in rows]
    epss = [r[1] for r in rows]
    asrs = [r[2] for r in rows]
    frrs = [r[3] for r in rows]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(frrs, asrs, c="#9aa0a6", s=55, zorder=2, label="(λ, ε) cells")

    valid = [i for i, (a, f) in enumerate(zip(asrs, frrs)) if not (math.isnan(a) or math.isnan(f))]
    front: list[int] = []
    if len(valid) >= 1:
        v_asr = [asrs[i] for i in valid]
        v_frr = [frrs[i] for i in valid]
        local_front = pareto_indices(v_asr, v_frr)
        front = [valid[i] for i in local_front]
        order = sorted(front, key=lambda i: frrs[i])
        ax.plot(
            [frrs[i] for i in order],
            [asrs[i] for i in order],
            "-o",
            color="#1a73e8",
            zorder=3,
            label="Pareto frontier",
        )
        for i in front:
            ax.annotate(
                f"λ={lams[i]:g},ε={epss[i]:g}",
                (frrs[i], asrs[i]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=7,
                color="#1a73e8",
            )

    for i in range(len(rows)):
        if i in front:
            continue
        if math.isnan(asrs[i]) or math.isnan(frrs[i]):
            continue
        ax.annotate(
            f"λ={lams[i]:g},ε={epss[i]:g}",
            (frrs[i], asrs[i]),
            textcoords="offset points",
            xytext=(4, -10),
            fontsize=6,
            color="#5f6368",
        )

    ax.set_xlabel("FRR  — lower is more helpful")
    ax.set_ylabel("ASR  — lower is safer")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {out_path}", flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--points-dir",
        default="",
        help="Directory containing points_*.csv (default: $CHECKPOINT_ROOT/final_test_outputs).",
    )
    p.add_argument(
        "--out-dir",
        default="",
        help="Where to write PNGs (default: <points-dir>/pareto_charts).",
    )
    args = p.parse_args(argv)

    scr = os.environ.get("SCRATCH", "")
    if not args.points_dir:
        ck = os.environ.get("CHECKPOINT_ROOT") or (f"{scr}/dp-llm-sweep" if scr else "")
        if not ck:
            raise SystemExit("pass --points-dir or set CHECKPOINT_ROOT / SCRATCH")
        # Prefer lr2e-5 subdir if present
        base = Path(expand_path(ck)) / "final_test_outputs"
        cand = base / "lr2e-5"
        args.points_dir = str(cand if cand.is_dir() else base)
    points_dir = Path(expand_path(args.points_dir))
    out_dir = Path(expand_path(args.out_dir)) if args.out_dir else points_dir / "pareto_charts"

    df = load_points(points_dir)
    n = 0
    for mode in MODES:
        for fam in FAMILIES:
            for bench in BENCHMARKS:
                sub = df[
                    (df["mode"] == mode)
                    & (df["family"] == fam)
                    & (df["benchmark"] == bench)
                ]
                title = f"{mode} | {fam} | {bench}"
                fname = f"pareto_{mode}_{fam}_{bench}.png"
                plot_one(sub, title, out_dir / fname)
                n += 1
    print(f"[plot] done: {n} charts → {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
