#!/usr/bin/env python3
"""
Build 8 λ×ε heatmaps from ``test_eval_matrix.py`` per-task ``*_metrics.tsv`` files.

Two metrics × two LM modes (clean vs perturbed reg) × two protocol views (seen-family vs
held-out average across gcg/autodan/pair):

  * Seen: ``seen_mean_asr``, ``seen_frr``
  * Unseen (held-out): ``heldout_mean_asr``, mean of ``*_model_frr`` per family

Grid axes match ``train/submit_wandb_sweep.py`` (``LAMBDAS`` × ``EPSILONS``). For **λ=0**
only one ε is trained; heatmaps **repeat that value across every ε column** in that row
(since ε does not affect the loss at λ=0).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from train.submit_wandb_sweep import EPSILONS, LAMBDAS  # noqa: E402


def _parse_metrics_tsv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "\t" not in line:
            continue
        k, v = line.split("\t", 1)
        out[k.strip()] = v.strip()
    return out


def _to_float(s: str) -> float:
    s = (s or "").strip()
    if not s or s == "skipped_seen_only":
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _grid_index(val: float, grid: tuple[float, ...]) -> int | None:
    for i, g in enumerate(grid):
        if abs(float(val) - float(g)) <= 1e-6 * max(1.0, abs(g)):
            return i
    return None


def _collect_rows(metrics_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(metrics_dir.glob("*_metrics.tsv")):
        d = _parse_metrics_tsv(path)
        if "lambda" not in d or "epsilon" not in d or "mode" not in d:
            continue
        lam = _to_float(d["lambda"])
        eps = _to_float(d["epsilon"])
        mode = d["mode"].strip()
        seen_asr = _to_float(d.get("seen_mean_asr", ""))
        seen_frr = _to_float(d.get("seen_frr", ""))
        ho_asr = _to_float(d.get("heldout_mean_asr", ""))
        frr_parts = [
            _to_float(d.get(f"{fam}_model_frr", "")) for fam in ("gcg", "autodan", "pair")
        ]
        if all(not np.isnan(x) for x in frr_parts):
            ho_frr = float(np.nanmean(frr_parts))
        else:
            ho_frr = float("nan")
        rows.append(
            {
                "path": path,
                "lambda": lam,
                "epsilon": eps,
                "mode": mode,
                "seen_mean_asr": seen_asr,
                "seen_frr": seen_frr,
                "heldout_mean_asr": ho_asr,
                "heldout_mean_frr": ho_frr,
            }
        )
    return rows


def _fill_grid(rows: list[dict[str, Any]], mode: str, col: str) -> np.ndarray:
    mat = np.full((len(LAMBDAS), len(EPSILONS)), np.nan, dtype=float)
    for r in rows:
        if r["mode"] != mode:
            continue
        i = _grid_index(r["lambda"], LAMBDAS)
        j = _grid_index(r["epsilon"], EPSILONS)
        if i is None or j is None:
            continue
        v = r.get(col)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        mat[i, j] = float(v)
    return mat


def _broadcast_lambda_zero_across_epsilon(mat: np.ndarray) -> np.ndarray:
    """For rows with λ≈0, copy the measured metric across all ε columns (degenerate at λ=0)."""
    out = np.array(mat, copy=True, dtype=float)
    for i, lam in enumerate(LAMBDAS):
        if abs(float(lam)) > 1e-12:
            continue
        row = out[i, :]
        finite = row[~np.isnan(row)]
        if finite.size == 0:
            continue
        val = float(np.nanmean(finite))
        out[i, :] = val
    return out


def _plot_matrix(
    mat: np.ndarray,
    *,
    title: str,
    out_path: Path,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(EPSILONS)))
    ax.set_yticks(np.arange(len(LAMBDAS)))
    ax.set_xticklabels([f"{e:g}" for e in EPSILONS])
    ax.set_yticklabels([f"{lam:g}" for lam in LAMBDAS])
    ax.set_xlabel("ε")
    ax.set_ylabel("λ")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            txt = "—" if np.isnan(v) else f"{v:.3f}"
            ax.text(j, i, txt, ha="center", va="center", color="white", fontsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--metrics-dir",
        type=Path,
        required=True,
        help="Directory containing *_metrics.tsv from test_eval_matrix (e.g. .../test_eval_outputs).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write PNGs (default: <metrics-dir>/heatmaps).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Load data and print planned outputs only.",
    )
    args = p.parse_args(argv)

    mdir = args.metrics_dir.expanduser()
    if not mdir.is_dir():
        print(f"ERROR: metrics directory not found: {mdir}", file=sys.stderr)
        return 2

    out_dir = args.output_dir.expanduser() if args.output_dir else mdir / "heatmaps"
    rows = _collect_rows(mdir)
    if not rows:
        print(f"ERROR: no *_metrics.tsv files under {mdir}", file=sys.stderr)
        return 2

    specs: list[tuple[str, str, str, str]] = [
        ("clean_reg", "seen_mean_asr", "Seen-family mean ASR (clean LM)", "seen_mean_asr_clean_lm.png"),
        ("clean_reg", "seen_frr", "Seen-family FRR (clean LM)", "seen_frr_clean_lm.png"),
        ("pert_reg", "seen_mean_asr", "Seen-family mean ASR (perturbed LM)", "seen_mean_asr_pert_lm.png"),
        ("pert_reg", "seen_frr", "Seen-family FRR (perturbed LM)", "seen_frr_pert_lm.png"),
        ("clean_reg", "heldout_mean_asr", "Held-out mean ASR (clean LM)", "heldout_mean_asr_clean_lm.png"),
        ("clean_reg", "heldout_mean_frr", "Held-out mean FRR (clean LM)", "heldout_mean_frr_clean_lm.png"),
        ("pert_reg", "heldout_mean_asr", "Held-out mean ASR (perturbed LM)", "heldout_mean_asr_pert_lm.png"),
        ("pert_reg", "heldout_mean_frr", "Held-out mean FRR (perturbed LM)", "heldout_mean_frr_pert_lm.png"),
    ]

    if args.dry_run:
        print(f"Would write {len(specs)} heatmaps to {out_dir} from {len(rows)} metric rows")
        for mode, col, title, fname in specs:
            print(f"  {fname}: {title}")
        return 0

    for mode, col, title, fname in specs:
        mat = _fill_grid(rows, mode, col)
        mat = _broadcast_lambda_zero_across_epsilon(mat)
        out_path = out_dir / fname
        _plot_matrix(mat, title=title, out_path=out_path)
        print(f"Wrote {out_path}")

    print(f"Done. {len(specs)} heatmaps in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
