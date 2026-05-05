#!/usr/bin/env python3
"""
Build λ×ε heatmaps from ``test_eval_matrix.py`` per-task ``*_metrics.tsv`` files.

**Clean-reg LM only** (``mode=clean_reg``); perturbed-LM runs are ignored.

For each (λ, ε) cell, metrics come from the saved CSV paths embedded in each TSV when
possible (attack ASR), else fall back to scalar fields in the TSV.

**Aggregate** (full harmful test set, no benchmark split) — written under
``<output-dir>/aggregate/``:

  * Seen ASR: mean across jailbreak variants on the harmful CSV, plus **per family**
    (GCG / AutoDAN / PAIR ``* Safety`` columns); ``seen_frr``
  * Held-out ASR: mean across families, plus **per jailbreak family** (GCG / AutoDAN / PAIR)
  * Held-out FRR: mean across families, plus **per family** (from ``*_model_frr`` in the TSV)

**Per harmful benchmark** — written under ``<output-dir>/by_dataset/<dataset>/``:

  * Same ASR panels as aggregate but restricted to prompts with that ``dataset`` label
    (join on ``goal``, ``target`` via ``--labels``).
  * **FRR** panels are **identical** to aggregate (the benign FRR set is not split by
    harmful benchmark); files are duplicated into each folder for a self-contained figure set.

Artifact paths in each ``*_metrics.tsv`` are stored **relative to the eval output directory**
when possible; the plotter also falls back to ``<metrics-dir>/<basename>`` so copies that
keep TSVs and CSVs together still work.

Grid axes match ``train/submit_wandb_sweep.py`` (``LAMBDAS`` × ``EPSILONS``). For **λ=0**,
heatmaps repeat the measured value across every ε column in that row.

Requires **pandas** (same as other eval utilities). You do **not** need to export
``MODEL_PROFILE`` or ``LR``: the default output folder is inferred from ``*_metrics.tsv``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from train.submit_wandb_sweep import EPSILONS, LAMBDAS  # noqa: E402

MODE = "clean_reg"
SAFETY_COL = {
    "gcg": "GCG Safety",
    "autodan": "AutoDAN Safety",
    "pair": "PAIR Safety",
}
FAMILIES = ("gcg", "autodan", "pair")


def expand_path(p: str) -> str:
    return os.path.expandvars(os.path.expanduser(p))


def _resolve_csv_path(metrics_dir: Path, raw: str) -> Path | None:
    """Resolve a path from metrics TSV: absolute, relative to metrics_dir, or basename fallback."""
    raw = (raw or "").strip()
    if not raw:
        return None
    p = Path(expand_path(raw))
    if p.is_file():
        return p
    rel = metrics_dir / raw
    if rel.is_file():
        return rel
    by_name = metrics_dir / Path(raw).name
    if by_name.is_file():
        return by_name
    return None


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


def _infer_model_profile_and_lr(metrics_dir: Path) -> tuple[str, str] | None:
    """Use the first ``clean_reg`` ``*_metrics.tsv`` to name output (no env exports needed)."""
    for path in sorted(metrics_dir.glob("*_metrics.tsv")):
        d = _parse_metrics_tsv(path)
        if d.get("mode", "").strip() != MODE:
            continue
        mp = d.get("model_profile", "").strip()
        lr = d.get("lr", "").strip()
        if mp and lr:
            return mp, lr
    return None


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


def _mean_asr_seen(h: pd.DataFrame) -> float:
    if len(h) == 0:
        return float("nan")
    cols = [c for c in ("GCG Safety", "AutoDAN Safety", "PAIR Safety") if c in h.columns]
    if not cols:
        return float("nan")
    return float(
        np.mean([(h[c].astype(str).str.lower() == "unsafe").mean() for c in cols])
    )


def _asr_family(h: pd.DataFrame, family: str) -> float:
    if len(h) == 0:
        return float("nan")
    col = SAFETY_COL[family]
    if col not in h.columns:
        return float("nan")
    return float((h[col].astype(str).str.lower() == "unsafe").mean())


def _heldout_asrs_by_subset(
    metrics: dict[str, str],
    ds_mask_fn: Callable[[pd.DataFrame], pd.Series],
    metrics_dir: Path,
) -> tuple[float, dict[str, float]]:
    """Returns (mean_asr_across_loaded_families, per_family_asr)."""
    per: dict[str, float] = {fam: float("nan") for fam in FAMILIES}
    parts: list[float] = []
    for fam in FAMILIES:
        key = f"{fam}_harmful_csv"
        if key not in metrics:
            continue
        p = _resolve_csv_path(metrics_dir, metrics[key])
        if p is None:
            continue
        h = pd.read_csv(p)
        h = h.loc[ds_mask_fn(h)].copy()
        if len(h) == 0:
            v = float("nan")
        else:
            v = _asr_family(h, fam)
        per[fam] = v
        parts.append(v)
    if not parts:
        mean_v = float("nan")
    elif any(np.isnan(parts)):
        mean_v = float("nan")
    else:
        mean_v = float(np.mean(parts))
    return mean_v, per


def _all_rows_mask(df: pd.DataFrame) -> pd.Series:
    return pd.Series(True, index=df.index)


def _frr_fields_from_tsv(d: dict[str, str]) -> dict[str, float]:
    seen_frr = _to_float(d.get("seen_frr", ""))
    parts = [_to_float(d.get(f"{fam}_model_frr", "")) for fam in FAMILIES]
    per_frr = {fam: _to_float(d.get(f"{fam}_model_frr", "")) for fam in FAMILIES}
    if all(not np.isnan(x) for x in parts):
        ho_mean_frr = float(np.nanmean(parts))
    else:
        ho_mean_frr = float("nan")
    return {
        "seen_frr": seen_frr,
        "heldout_mean_frr": ho_mean_frr,
        "heldout_gcg_frr": per_frr["gcg"],
        "heldout_autodan_frr": per_frr["autodan"],
        "heldout_pair_frr": per_frr["pair"],
    }


def _collect_aggregate_rows(metrics_dir: Path) -> list[dict[str, Any]]:
    """One row per clean_reg metrics TSV with ASR from CSVs when possible."""
    rows: list[dict[str, Any]] = []
    for path in sorted(metrics_dir.glob("*_metrics.tsv")):
        d = _parse_metrics_tsv(path)
        if "lambda" not in d or "epsilon" not in d or "mode" not in d:
            continue
        if d["mode"].strip() != MODE:
            continue
        lam = _to_float(d["lambda"])
        eps = _to_float(d["epsilon"])
        frr = _frr_fields_from_tsv(d)

        row: dict[str, Any] = {
            "path": path,
            "lambda": lam,
            "epsilon": eps,
            "mode": MODE,
            **frr,
        }

        seen_path = d.get("seen_harmful_csv", "").strip()
        sp = _resolve_csv_path(metrics_dir, seen_path)
        if sp is not None:
            h_seen = pd.read_csv(sp)
            row["seen_mean_asr"] = _mean_asr_seen(h_seen)
            for fam in FAMILIES:
                row[f"seen_{fam}_asr"] = _asr_family(h_seen, fam)
        else:
            row["seen_mean_asr"] = _to_float(d.get("seen_mean_asr", ""))
            for fam in FAMILIES:
                row[f"seen_{fam}_asr"] = _to_float(d.get(f"seen_{fam}_asr", ""))

        ho_mean, ho_per = _heldout_asrs_by_subset(d, _all_rows_mask, metrics_dir)
        has_any_heldout_csv = any(
            _resolve_csv_path(metrics_dir, d.get(f"{fam}_harmful_csv", "")) is not None
            for fam in FAMILIES
            if f"{fam}_harmful_csv" in d
        )
        if np.isnan(ho_mean) and not has_any_heldout_csv:
            ho_mean = _to_float(d.get("heldout_mean_asr", ""))
            ho_per = {fam: float("nan") for fam in FAMILIES}

        row["heldout_mean_asr"] = ho_mean
        row["heldout_gcg_asr"] = ho_per["gcg"]
        row["heldout_autodan_asr"] = ho_per["autodan"]
        row["heldout_pair_asr"] = ho_per["pair"]

        rows.append(row)
    return rows


def _collect_per_dataset_rows(
    metrics_dir: Path,
    labels_df: pd.DataFrame,
) -> dict[str, list[dict[str, Any]]]:
    keycols = ["goal", "target"]
    lab = labels_df[keycols + ["dataset"]].drop_duplicates(subset=keycols)
    by_ds: dict[str, list[dict[str, Any]]] = {}

    for path in sorted(metrics_dir.glob("*_metrics.tsv")):
        d = _parse_metrics_tsv(path)
        if "lambda" not in d or "epsilon" not in d or "mode" not in d:
            continue
        if d["mode"].strip() != MODE:
            continue
        lam = _to_float(d["lambda"])
        eps = _to_float(d["epsilon"])
        frr = _frr_fields_from_tsv(d)

        seen_path = d.get("seen_harmful_csv", "").strip()
        sp = _resolve_csv_path(metrics_dir, seen_path)
        if sp is None:
            continue
        h_seen = pd.read_csv(sp)
        if any(c not in h_seen.columns for c in keycols):
            continue

        merged_seen = h_seen.merge(lab, on=keycols, how="left")
        if merged_seen["dataset"].isna().any():
            n_bad = int(merged_seen["dataset"].isna().sum())
            print(
                f"[warn] {path.name}: {n_bad} harmful rows without dataset label",
                file=sys.stderr,
            )

        for ds in sorted(x for x in merged_seen["dataset"].dropna().unique()):
            mask_seen = merged_seen["dataset"] == ds

            def ds_mask(df: pd.DataFrame, dataset: str = ds) -> pd.Series:
                m = df.merge(lab, on=keycols, how="left")
                return m["dataset"] == dataset

            h_sub = merged_seen.loc[mask_seen]
            seen_asr = _mean_asr_seen(h_sub)
            ho_mean, ho_per = _heldout_asrs_by_subset(d, ds_mask, metrics_dir)

            row = {
                "path": path,
                "lambda": lam,
                "epsilon": eps,
                "mode": MODE,
                "seen_mean_asr": seen_asr,
                **{f"seen_{fam}_asr": _asr_family(h_sub, fam) for fam in FAMILIES},
                "heldout_mean_asr": ho_mean,
                "heldout_gcg_asr": ho_per["gcg"],
                "heldout_autodan_asr": ho_per["autodan"],
                "heldout_pair_asr": ho_per["pair"],
                **frr,
            }
            by_ds.setdefault(str(ds), []).append(row)

    return by_ds


def _fill_grid(rows: list[dict[str, Any]], col: str) -> np.ndarray:
    mat = np.full((len(LAMBDAS), len(EPSILONS)), np.nan, dtype=float)
    for r in rows:
        if r["mode"] != MODE:
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


# (column_key, title, filename)
AGGREGATE_SPECS: list[tuple[str, str, str]] = [
    ("seen_mean_asr", "Seen-family mean ASR — avg. jailbreak variants (clean LM)", "seen_mean_asr_clean_lm.png"),
    ("seen_gcg_asr", "Seen-family ASR — GCG jailbreak (clean LM)", "seen_gcg_asr_clean_lm.png"),
    ("seen_autodan_asr", "Seen-family ASR — AutoDAN jailbreak (clean LM)", "seen_autodan_asr_clean_lm.png"),
    ("seen_pair_asr", "Seen-family ASR — PAIR jailbreak (clean LM)", "seen_pair_asr_clean_lm.png"),
    ("seen_frr", "Seen-family FRR (clean LM)", "seen_frr_clean_lm.png"),
    ("heldout_mean_asr", "Held-out mean ASR — avg. jailbreak families (clean LM)", "heldout_mean_asr_clean_lm.png"),
    ("heldout_gcg_asr", "Held-out ASR — GCG family (clean LM)", "heldout_gcg_asr_clean_lm.png"),
    ("heldout_autodan_asr", "Held-out ASR — AutoDAN family (clean LM)", "heldout_autodan_asr_clean_lm.png"),
    ("heldout_pair_asr", "Held-out ASR — PAIR family (clean LM)", "heldout_pair_asr_clean_lm.png"),
    ("heldout_mean_frr", "Held-out mean FRR — avg. families (clean LM)", "heldout_mean_frr_clean_lm.png"),
    ("heldout_gcg_frr", "Held-out FRR — GCG-trained adapter (clean LM)", "heldout_gcg_frr_clean_lm.png"),
    ("heldout_autodan_frr", "Held-out FRR — AutoDAN-trained adapter (clean LM)", "heldout_autodan_frr_clean_lm.png"),
    ("heldout_pair_frr", "Held-out FRR — PAIR-trained adapter (clean LM)", "heldout_pair_frr_clean_lm.png"),
]

_ASR_SPEC_KEYS = frozenset(
    {
        "seen_mean_asr",
        "seen_gcg_asr",
        "seen_autodan_asr",
        "seen_pair_asr",
        "heldout_mean_asr",
        "heldout_gcg_asr",
        "heldout_autodan_asr",
        "heldout_pair_asr",
    }
)


def _default_heatmap_dirname(metrics_dir: Path) -> str:
    """``heatmaps_<MODEL_PROFILE>_lr<LR>/`` from env if set, else inferred from metrics TSVs."""
    mp = os.environ.get("MODEL_PROFILE", "").strip()
    lr = os.environ.get("LR", "").strip()
    if mp and lr:
        return f"heatmaps_{mp}_lr{lr}".replace("/", "_")
    inferred = _infer_model_profile_and_lr(metrics_dir)
    if inferred:
        mp, lr = inferred
        return f"heatmaps_{mp}_lr{lr}".replace("/", "_")
    return "heatmaps"


def main(argv: list[str] | None = None) -> int:
    scr = os.environ.get("SCRATCH", "")
    default_labels = f"{scr}/dp-llm-experiments/official_data/combined_test_dataset.csv" if scr else ""

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--metrics-dir",
        type=Path,
        required=True,
        help="Directory containing *_metrics.tsv from test_eval_matrix.",
    )
    p.add_argument(
        "--labels",
        type=Path,
        default=Path(default_labels) if default_labels else None,
        help="CSV with goal, target, dataset (default: $SCRATCH/.../official_data/combined_test_dataset.csv).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Root output directory (default: <metrics-dir>/heatmaps_<model>_lr<rate> from "
            "the first clean_reg *_metrics.tsv, or MODEL_PROFILE+LR env if both set)."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned PNG paths only.",
    )
    args = p.parse_args(argv)

    mdir = Path(expand_path(str(args.metrics_dir)))
    if not mdir.is_dir():
        print(f"ERROR: metrics directory not found: {mdir}", file=sys.stderr)
        return 2

    out_root = (
        Path(expand_path(str(args.output_dir)))
        if args.output_dir
        else mdir / _default_heatmap_dirname(mdir)
    )
    if not args.dry_run:
        print(f"[plot_hyperparameter_heatmaps] output root: {out_root}", flush=True)
    agg_dir = out_root / "aggregate"
    by_ds_root = out_root / "by_dataset"

    agg_rows = _collect_aggregate_rows(mdir)
    if not agg_rows:
        print(
            f"ERROR: no clean_reg *_metrics.tsv under {mdir}",
            file=sys.stderr,
        )
        return 2

    labels_path = Path(expand_path(str(args.labels))) if args.labels else None
    by_ds: dict[str, list[dict[str, Any]]] = {}
    if labels_path and labels_path.is_file():
        labels_df = pd.read_csv(labels_path)
        for c in ("goal", "target", "dataset"):
            if c not in labels_df.columns:
                print(f"ERROR: --labels must contain column {c!r}", file=sys.stderr)
                return 2
        by_ds = _collect_per_dataset_rows(mdir, labels_df)
        if not by_ds:
            print("[warn] no per-dataset rows (check seen_harmful_csv paths vs labels)", file=sys.stderr)
    elif args.labels:
        print(f"[warn] labels file not found ({labels_path}); skipping by_dataset/", file=sys.stderr)

    asr_specs = [s for s in AGGREGATE_SPECS if s[0] in _ASR_SPEC_KEYS]
    frr_specs = [s for s in AGGREGATE_SPECS if s[0] not in _ASR_SPEC_KEYS]

    if args.dry_run:
        print(f"Aggregate ({len(AGGREGATE_SPECS)} PNGs) -> {agg_dir}/")
        for _c, _t, fname in AGGREGATE_SPECS:
            print(f"  {fname}")
        if by_ds:
            print(f"By dataset: {', '.join(sorted(by_ds.keys()))} -> {by_ds_root}/")
            for ds in sorted(by_ds.keys()):
                print(f"  [{ds}] {len(asr_specs)} ASR + {len(frr_specs)} FRR (global) panels")
                for _c, _t, fname in asr_specs + frr_specs:
                    print(f"    {fname}")
        print(f"Total aggregate: {len(AGGREGATE_SPECS)}")
        print(f"Output root: {out_root}")
        return 0

    def write_specs(rows: list[dict[str, Any]], out_dir: Path, title_suffix: str) -> None:
        for col, title_stem, fname in AGGREGATE_SPECS:
            mat = _fill_grid(rows, col)
            mat = _broadcast_lambda_zero_across_epsilon(mat)
            title = f"{title_stem}{title_suffix}"
            out_path = out_dir / fname
            _plot_matrix(mat, title=title, out_path=out_path)
            print(f"Wrote {out_path}")

    write_specs(agg_rows, agg_dir, "")

    if by_ds:
        for ds, rows_ds in sorted(by_ds.items()):
            ds_slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in ds)
            ds_dir = by_ds_root / ds_slug
            suffix = f" — {ds}"
            for col, title_stem, fname in asr_specs:
                mat = _fill_grid(rows_ds, col)
                mat = _broadcast_lambda_zero_across_epsilon(mat)
                title = f"{title_stem}{suffix}"
                _plot_matrix(mat, title=title, out_path=ds_dir / fname)
                print(f"Wrote {ds_dir / fname}")
            # Global FRR (same matrix as aggregate): duplicate into each dataset folder
            for col, title_stem, fname in frr_specs:
                mat = _fill_grid(agg_rows, col)
                mat = _broadcast_lambda_zero_across_epsilon(mat)
                title = f"{title_stem} (global benign set){suffix}"
                _plot_matrix(mat, title=title, out_path=ds_dir / fname)
                print(f"Wrote {ds_dir / fname}")

    print(f"Done. Output root: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
