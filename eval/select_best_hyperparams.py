#!/usr/bin/env python3
"""
Pick the best (λ, ε) from a validation sweep's ``test_eval_matrix.py`` ``*_metrics.tsv``.

Reads every ``*_metrics.tsv`` in ``--metrics-dir`` (one per (λ, ε) cell), builds a summary
table of ASR (attack success rate; lower is safer) and FRR (false-refusal rate; lower is
more helpful), computes the ASR-vs-FRR **Pareto frontier**, and recommends a cell under a
few explicit policies. λ=0 rows are labelled ``clean`` (regularizer off); λ>0 are
``perturbed`` (stability regularizer on).

Objective (which ASR/FRR to optimize; both minimized):
  seen      — seen-family ASR + seen FRR (default)
  heldout   — held-out mean ASR + held-out mean FRR (avg over gcg/autodan/pair adapters)
  combined  — mean of seen and held-out for each of ASR / FRR

Selection policies reported (all on the chosen objective):
  1. FRR-capped min-ASR : among cells with FRR ≤ --frr-cap, the lowest ASR (the usual pick).
  2. Balanced / knee    : min Euclidean distance to the utopia point after min-max
                          normalizing ASR and FRR across the grid.
  3. Weighted sum       : min (w·ASR_norm + (1-w)·FRR_norm), w = --asr-weight.

Outputs (in --out-dir, default <metrics-dir>/selection/):
  hyperparameter_summary.csv  all cells, all metrics, sorted by (ASR, FRR)
  pareto_frontier.csv         Pareto-optimal cells only
  recommendation.md           human-readable table + the three recommended picks
  pareto_curve.png            ASR-vs-FRR scatter (frontier highlighted, picks annotated)

Usage::

  python eval/select_best_hyperparams.py --metrics-dir $SCRATCH/dp-llm-sweep/val_eval_outputs/lr2e-5
  python eval/select_best_hyperparams.py --metrics-dir ... --objective seen --frr-cap 0.1
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any

FAMILIES = ("gcg", "autodan", "pair")


def expand_path(p: str) -> str:
    return os.path.expandvars(os.path.expanduser(p))


def _to_float(s: str) -> float:
    s = (s or "").strip()
    if not s or s.lower() in ("none", "nan", "skipped_seen_only", ""):
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _parse_metrics_tsv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "\t" not in line:
            continue
        k, v = line.split("\t", 1)
        out[k.strip()] = v.strip()
    return out


def _nanmean(vals: list[float]) -> float:
    xs = [v for v in vals if not math.isnan(v)]
    return sum(xs) / len(xs) if xs else float("nan")


def collect_rows(metrics_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(metrics_dir.glob("*_metrics.tsv")):
        d = _parse_metrics_tsv(path)
        if "lambda" not in d or "epsilon" not in d:
            continue
        lam = _to_float(d["lambda"])
        eps = _to_float(d["epsilon"])
        if math.isnan(lam) or math.isnan(eps):
            continue

        ho_frr_parts = [_to_float(d.get(f"{fam}_model_frr", "")) for fam in FAMILIES]
        ho_asr_parts = [_to_float(d.get(f"{fam}_heldout_asr", "")) for fam in FAMILIES]
        row: dict[str, Any] = {
            "kind": "clean" if abs(lam) < 1e-12 else "perturbed",
            "lambda": lam,
            "epsilon": eps,
            "lr": d.get("lr", ""),
            "slug": d.get("slug", ""),
            "model_profile": d.get("model_profile", ""),
            "seen_mean_asr": _to_float(d.get("seen_mean_asr", "")),
            "seen_frr": _to_float(d.get("seen_frr", "")),
            "seen_gcg_asr": _to_float(d.get("seen_gcg_asr", "")),
            "seen_autodan_asr": _to_float(d.get("seen_autodan_asr", "")),
            "seen_pair_asr": _to_float(d.get("seen_pair_asr", "")),
            "heldout_mean_asr": _to_float(d.get("heldout_mean_asr", "")),
            "heldout_mean_frr": _nanmean(ho_frr_parts),
            "heldout_gcg_asr": ho_asr_parts[0],
            "heldout_autodan_asr": ho_asr_parts[1],
            "heldout_pair_asr": ho_asr_parts[2],
            "_tsv": path.name,
        }
        rows.append(row)
    return rows


def objective_values(row: dict[str, Any], objective: str) -> tuple[float, float]:
    """Return (asr, frr) for the requested objective (both minimized)."""
    if objective == "seen":
        return row["seen_mean_asr"], row["seen_frr"]
    if objective == "heldout":
        return row["heldout_mean_asr"], row["heldout_mean_frr"]
    if objective == "combined":
        asr = _nanmean([row["seen_mean_asr"], row["heldout_mean_asr"]])
        frr = _nanmean([row["seen_frr"], row["heldout_mean_frr"]])
        return asr, frr
    raise ValueError(f"unknown objective: {objective!r}")


def pareto_frontier(points: list[tuple[int, float, float]]) -> list[int]:
    """points = (idx, asr, frr); return indices on the min-min Pareto frontier."""
    valid = [(i, a, f) for (i, a, f) in points if not (math.isnan(a) or math.isnan(f))]
    keep: list[int] = []
    for i, a, f in valid:
        dominated = False
        for j, a2, f2 in valid:
            if j == i:
                continue
            if a2 <= a and f2 <= f and (a2 < a or f2 < f):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    # Sort frontier by ASR ascending for readability.
    keep.sort(key=lambda i: (points[i][1], points[i][2]))
    return keep


def _norm(vals: list[float]) -> tuple[float, float]:
    xs = [v for v in vals if not math.isnan(v)]
    if not xs:
        return 0.0, 1.0
    lo, hi = min(xs), max(xs)
    return lo, (hi if hi > lo else lo + 1e-9)


def fmt(v: Any) -> str:
    if isinstance(v, float):
        return "—" if math.isnan(v) else f"{v:.4f}"
    return str(v)


def cell_label(row: dict[str, Any]) -> str:
    return f"λ={row['lambda']:g}, ε={row['epsilon']:g} ({row['kind']})"


def write_summary_csv(path: Path, rows: list[dict[str, Any]], cols: list[str]) -> None:
    import csv

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([fmt(r.get(c)) if isinstance(r.get(c), float) else r.get(c) for c in cols])


def _md_table(rows: list[dict[str, Any]], objective: str) -> list[str]:
    hdr = ["λ", "ε", "kind", "obj_ASR", "obj_FRR", "seen_ASR", "seen_FRR", "heldout_ASR", "heldout_FRR"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]
    for r in rows:
        a, fr = objective_values(r, objective)
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{r['lambda']:g}",
                    f"{r['epsilon']:g}",
                    r["kind"],
                    fmt(a),
                    fmt(fr),
                    fmt(r["seen_mean_asr"]),
                    fmt(r["seen_frr"]),
                    fmt(r["heldout_mean_asr"]),
                    fmt(r["heldout_mean_frr"]),
                ]
            )
            + " |"
        )
    return lines


def plot_pareto(
    out_path: Path,
    rows: list[dict[str, Any]],
    objective: str,
    frontier_idx: list[int],
    picks: dict[str, int | None],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - matplotlib optional
        print(f"[select] matplotlib unavailable ({exc}); skipping pareto_curve.png", flush=True)
        return

    xs, ys, labels = [], [], []
    for r in rows:
        a, fr = objective_values(r, objective)
        if math.isnan(a) or math.isnan(fr):
            continue
        xs.append(fr)
        ys.append(a)
        labels.append((r["lambda"], r["epsilon"]))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(xs, ys, c="#9aa0a6", s=40, label="all cells", zorder=2)

    fx = [objective_values(rows[i], objective)[1] for i in frontier_idx]
    fy = [objective_values(rows[i], objective)[0] for i in frontier_idx]
    order = sorted(range(len(fx)), key=lambda k: fx[k])
    ax.plot([fx[k] for k in order], [fy[k] for k in order], "-o", color="#1a73e8",
            label="Pareto frontier", zorder=3)

    colors = {"frr_capped": "#188038", "balanced": "#e37400", "weighted": "#d93025"}
    for name, idx in picks.items():
        if idx is None:
            continue
        a, fr = objective_values(rows[idx], objective)
        ax.scatter([fr], [a], s=180, marker="*", color=colors.get(name, "black"),
                   edgecolor="black", zorder=4, label=f"{name}: {cell_label(rows[idx])}")

    ax.set_xlabel(f"FRR ({objective})  — lower is more helpful")
    ax.set_ylabel(f"ASR ({objective})  — lower is safer")
    ax.set_title("Validation ASR vs FRR (utopia = bottom-left)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metrics-dir", required=True, help="Dir with *_metrics.tsv from test_eval_matrix.")
    p.add_argument("--out-dir", default=None, help="Output dir (default: <metrics-dir>/selection).")
    p.add_argument(
        "--objective",
        default="seen",
        choices=("seen", "heldout", "combined"),
        help="Which ASR/FRR to optimize (default: seen).",
    )
    p.add_argument(
        "--frr-cap",
        type=float,
        default=0.10,
        help="Max acceptable FRR for the FRR-capped min-ASR policy (default: 0.10).",
    )
    p.add_argument(
        "--asr-weight",
        type=float,
        default=0.5,
        help="Weight on normalized ASR for the weighted-sum policy (default: 0.5).",
    )
    args = p.parse_args(argv)

    mdir = Path(expand_path(args.metrics_dir))
    if not mdir.is_dir():
        print(f"ERROR: metrics dir not found: {mdir}", file=sys.stderr)
        return 2
    out_dir = Path(expand_path(args.out_dir)) if args.out_dir else mdir / "selection"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(mdir)
    if not rows:
        print(f"ERROR: no usable *_metrics.tsv in {mdir}", file=sys.stderr)
        return 2

    obj = args.objective
    # Sort by (ASR, FRR) of the chosen objective (NaNs last).
    def sort_key(r: dict[str, Any]) -> tuple[float, float]:
        a, fr = objective_values(r, obj)
        return (a if not math.isnan(a) else float("inf"), fr if not math.isnan(fr) else float("inf"))

    rows.sort(key=sort_key)

    # Pareto frontier.
    pts = [(i, *objective_values(r, obj)) for i, r in enumerate(rows)]
    frontier_idx = pareto_frontier(pts)

    # Policy 1: FRR-capped min-ASR.
    capped = [
        i
        for i, r in enumerate(rows)
        if (lambda af: not math.isnan(af[0]) and not math.isnan(af[1]) and af[1] <= args.frr_cap)(
            objective_values(r, obj)
        )
    ]
    pick_capped = (
        min(capped, key=lambda i: (objective_values(rows[i], obj)[0], objective_values(rows[i], obj)[1]))
        if capped
        else None
    )

    # Normalize for balanced / weighted policies (over cells with both metrics present).
    asr_all = [objective_values(r, obj)[0] for r in rows]
    frr_all = [objective_values(r, obj)[1] for r in rows]
    a_lo, a_hi = _norm(asr_all)
    f_lo, f_hi = _norm(frr_all)

    def norm_pair(i: int) -> tuple[float, float] | None:
        a, fr = objective_values(rows[i], obj)
        if math.isnan(a) or math.isnan(fr):
            return None
        return (a - a_lo) / (a_hi - a_lo), (fr - f_lo) / (f_hi - f_lo)

    valid_idx = [i for i in range(len(rows)) if norm_pair(i) is not None]
    pick_balanced = (
        min(valid_idx, key=lambda i: math.hypot(*norm_pair(i))) if valid_idx else None  # type: ignore[misc]
    )
    w = args.asr_weight
    pick_weighted = (
        min(valid_idx, key=lambda i: w * norm_pair(i)[0] + (1 - w) * norm_pair(i)[1])  # type: ignore[index]
        if valid_idx
        else None
    )

    picks = {"frr_capped": pick_capped, "balanced": pick_balanced, "weighted": pick_weighted}

    summary_cols = [
        "kind", "lambda", "epsilon", "lr", "seen_mean_asr", "seen_frr",
        "heldout_mean_asr", "heldout_mean_frr", "seen_gcg_asr", "seen_autodan_asr",
        "seen_pair_asr", "heldout_gcg_asr", "heldout_autodan_asr", "heldout_pair_asr",
        "slug", "_tsv",
    ]
    write_summary_csv(out_dir / "hyperparameter_summary.csv", rows, summary_cols)
    write_summary_csv(
        out_dir / "pareto_frontier.csv", [rows[i] for i in frontier_idx], summary_cols
    )
    plot_pareto(out_dir / "pareto_curve.png", rows, obj, frontier_idx, picks)

    # recommendation.md
    md: list[str] = []
    md.append(f"# Hyperparameter selection (objective: {obj})\n")
    md.append(
        f"Metrics dir: `{mdir}`  \nCells: {len(rows)}  |  Pareto-optimal: {len(frontier_idx)}  "
        f"|  FRR cap: {args.frr_cap:g}  |  ASR weight (weighted sum): {w:g}\n"
    )
    md.append("ASR = attack success rate (lower safer). FRR = false-refusal rate (lower more helpful). "
              "λ=0 = clean (no stability regularizer); λ>0 = perturbed (regularizer on).\n")

    md.append("## Recommended picks\n")
    policy_desc = {
        "frr_capped": f"Lowest ASR with FRR ≤ {args.frr_cap:g}",
        "balanced": "Closest to the utopia point (min-max normalized ASR/FRR)",
        "weighted": f"Min weighted sum ({w:g}·ASR + {1 - w:g}·FRR, normalized)",
    }
    for name, idx in picks.items():
        if idx is None:
            md.append(f"- **{name}** ({policy_desc[name]}): no cell qualifies.")
            continue
        a, fr = objective_values(rows[idx], obj)
        md.append(
            f"- **{name}** ({policy_desc[name]}): **{cell_label(rows[idx])}** "
            f"→ ASR={fmt(a)}, FRR={fmt(fr)} "
            f"(seen ASR={fmt(rows[idx]['seen_mean_asr'])}/FRR={fmt(rows[idx]['seen_frr'])}, "
            f"held-out ASR={fmt(rows[idx]['heldout_mean_asr'])}/FRR={fmt(rows[idx]['heldout_mean_frr'])})."
        )
    md.append("")
    md.append("## Pareto frontier (sorted by ASR)\n")
    md.extend(_md_table([rows[i] for i in frontier_idx], obj))
    md.append("")
    md.append("## All cells (sorted by objective ASR, then FRR)\n")
    md.extend(_md_table(rows, obj))
    md.append("")
    (out_dir / "recommendation.md").write_text("\n".join(md), encoding="utf-8")

    # Console summary.
    print(f"[select] objective={obj}  cells={len(rows)}  pareto={len(frontier_idx)}", flush=True)
    for name, idx in picks.items():
        if idx is None:
            print(f"[select] {name}: (no qualifying cell)", flush=True)
            continue
        a, fr = objective_values(rows[idx], obj)
        print(f"[select] {name}: {cell_label(rows[idx])}  ASR={fmt(a)} FRR={fmt(fr)}", flush=True)
    print(f"[select] wrote: {out_dir}/hyperparameter_summary.csv, pareto_frontier.csv, "
          f"recommendation.md, pareto_curve.png", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
