#!/usr/bin/env python3
"""Aggregate sweep/final outputs into selection tables and plots.

This script deliberately does not rank or select one configuration. It exposes
the per-benchmark metrics, aggregate metrics, and non-dominated set for the
researcher to choose from.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SAFETY_COLUMNS = {
    "gcg": "GCG Safety",
    "autodan": "AutoDAN Safety",
    "pair": "PAIR Safety",
}
MAIN_METRICS = (
    "seen_gcg_asr",
    "seen_autodan_asr",
    "seen_pair_asr",
    "seen_mean_asr",
    "seen_frr",
    "heldout_gcg_asr",
    "heldout_autodan_asr",
    "heldout_pair_asr",
    "heldout_mean_asr",
    "heldout_mean_frr",
)
OBJECTIVES = ("seen_mean_asr", "seen_frr", "heldout_mean_asr", "heldout_mean_frr")


def read_manifest(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", keep_default_na=False)


def benchmark_slug(value: object) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(value).strip().casefold()).strip("_")
    return slug or "unknown"


def combine_outputs(group: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for output in group.sort_values("shard_index")["output_csv"]:
        path = Path(output)
        if not path.is_file():
            missing.append(str(path))
        else:
            frames.append(pd.read_csv(path))
    if missing:
        raise FileNotFoundError("Missing evaluation outputs:\n  " + "\n  ".join(missing[:20]))
    return pd.concat(frames, ignore_index=True)


def metric_rows(manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    keys = [
        "phase",
        "config_id",
        "seed",
        "train_mode",
        "heldout_family",
        "learning_rate",
        "lambda_val",
        "epsilon",
        "benign_lm_weight",
        "eval_kind",
        "eval_family",
    ]
    for key_values, group in manifest.groupby(keys, sort=True, dropna=False):
        meta = dict(zip(keys, key_values))
        frame = combine_outputs(group)
        if meta["eval_kind"] == "asr":
            family = meta["eval_family"]
            label_col = SAFETY_COLUMNS[family]
            if label_col not in frame.columns:
                raise ValueError(f"{label_col!r} missing from {group.iloc[0]['output_csv']}")
            labels = frame[label_col].astype("string").str.casefold()
            invalid = labels.isna() | ~labels.isin({"safe", "unsafe"})
            if invalid.any():
                raise ValueError(
                    f"{group.iloc[0]['output_csv']} contains "
                    f"{int(invalid.sum())} invalid {label_col!r} labels"
                )
            unsafe = labels.eq("unsafe")
            rows.append({**meta, "benchmark": "aggregate", "metric": "asr", "value": unsafe.mean(), "n": len(frame)})
            if "dataset" in frame.columns:
                for benchmark, indices in frame.groupby("dataset", dropna=False).groups.items():
                    values = unsafe.loc[indices]
                    rows.append(
                        {
                            **meta,
                            "benchmark": benchmark_slug(benchmark),
                            "metric": "asr",
                            "value": values.mean(),
                            "n": len(values),
                        }
                    )
        else:
            label_col = "Original Safety"
            if label_col not in frame.columns:
                raise ValueError(f"{label_col!r} missing from {group.iloc[0]['output_csv']}")
            labels = frame[label_col].astype("string").str.casefold()
            invalid = labels.isna() | ~labels.isin({"refusal", "non-refusal"})
            if invalid.any():
                raise ValueError(
                    f"{group.iloc[0]['output_csv']} contains "
                    f"{int(invalid.sum())} invalid {label_col!r} labels"
                )
            refusal = labels.eq("refusal")
            rows.append({**meta, "benchmark": "aggregate", "metric": "frr", "value": refusal.mean(), "n": len(frame)})
            if "data_type" in frame.columns:
                for benchmark, indices in frame.groupby("data_type", dropna=False).groups.items():
                    values = refusal.loc[indices]
                    rows.append(
                        {
                            **meta,
                            "benchmark": benchmark_slug(benchmark),
                            "metric": "frr",
                            "value": values.mean(),
                            "n": len(values),
                        }
                    )
    return pd.DataFrame(rows)


def build_seed_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    summary_rows: list[dict] = []
    group_cols = [
        "phase",
        "config_id",
        "seed",
        "learning_rate",
        "lambda_val",
        "epsilon",
        "benign_lm_weight",
    ]
    for key_values, group in metrics.groupby(group_cols, sort=True):
        row = dict(zip(group_cols, key_values))
        aggregate = group[group["benchmark"] == "aggregate"]

        seen = aggregate[(aggregate["train_mode"] == "seen") & (aggregate["metric"] == "asr")]
        for family in SAFETY_COLUMNS:
            values = seen.loc[seen["eval_family"] == family, "value"]
            row[f"seen_{family}_asr"] = float(values.iloc[0]) if len(values) else math.nan
        row["seen_mean_asr"] = float(
            np.nanmean([row[f"seen_{family}_asr"] for family in SAFETY_COLUMNS])
        )
        seen_frr = aggregate[(aggregate["train_mode"] == "seen") & (aggregate["metric"] == "frr")]["value"]
        row["seen_frr"] = float(seen_frr.iloc[0]) if len(seen_frr) else math.nan

        heldout_frrs: list[float] = []
        for family in SAFETY_COLUMNS:
            mode = f"heldout_{family}"
            values = aggregate[
                (aggregate["train_mode"] == mode)
                & (aggregate["metric"] == "asr")
                & (aggregate["eval_family"] == family)
            ]["value"]
            row[f"heldout_{family}_asr"] = float(values.iloc[0]) if len(values) else math.nan
            frr = aggregate[(aggregate["train_mode"] == mode) & (aggregate["metric"] == "frr")]["value"]
            if len(frr):
                heldout_frrs.append(float(frr.iloc[0]))
                row[f"heldout_{family}_frr"] = float(frr.iloc[0])
        row["heldout_mean_asr"] = float(
            np.nanmean([row[f"heldout_{family}_asr"] for family in SAFETY_COLUMNS])
        )
        row["heldout_mean_frr"] = float(np.mean(heldout_frrs)) if heldout_frrs else math.nan

        frr_benchmarks = sorted(
            set(group.loc[group["metric"] == "frr", "benchmark"]) - {"aggregate"}
        )
        for benchmark in frr_benchmarks:
            seen_value = group[
                (group["train_mode"] == "seen")
                & (group["metric"] == "frr")
                & (group["benchmark"] == benchmark)
            ]["value"]
            if len(seen_value):
                row[f"seen_{benchmark}_frr"] = float(seen_value.iloc[0])
            heldout_values = group[
                group["train_mode"].isin([f"heldout_{family}" for family in SAFETY_COLUMNS])
                & (group["metric"] == "frr")
                & (group["benchmark"] == benchmark)
            ]["value"]
            if len(heldout_values):
                row[f"heldout_{benchmark}_mean_frr"] = float(heldout_values.mean())

        benchmarks = sorted(set(group["benchmark"]) - {"aggregate"})
        for benchmark in benchmarks:
            per_benchmark = group[(group["benchmark"] == benchmark) & (group["metric"] == "asr")]
            seen_values: list[float] = []
            heldout_values: list[float] = []
            for family in SAFETY_COLUMNS:
                sv = per_benchmark[
                    (per_benchmark["train_mode"] == "seen")
                    & (per_benchmark["eval_family"] == family)
                ]["value"]
                if len(sv):
                    value = float(sv.iloc[0])
                    row[f"seen_{benchmark}_{family}_asr"] = value
                    seen_values.append(value)
                hv = per_benchmark[
                    (per_benchmark["train_mode"] == f"heldout_{family}")
                    & (per_benchmark["eval_family"] == family)
                ]["value"]
                if len(hv):
                    value = float(hv.iloc[0])
                    row[f"heldout_{benchmark}_{family}_asr"] = value
                    heldout_values.append(value)
            if seen_values:
                row[f"seen_{benchmark}_mean_asr"] = float(np.mean(seen_values))
            if heldout_values:
                row[f"heldout_{benchmark}_mean_asr"] = float(np.mean(heldout_values))
        summary_rows.append(row)
    return pd.DataFrame(summary_rows).sort_values(["config_id", "seed"]).reset_index(drop=True)


def pareto_mask(frame: pd.DataFrame, objectives: tuple[str, ...] = OBJECTIVES) -> np.ndarray:
    values = frame.loc[:, objectives].to_numpy(dtype=float)
    valid = np.isfinite(values).all(axis=1)
    mask = np.zeros(len(frame), dtype=bool)
    for i, point in enumerate(values):
        if not valid[i]:
            continue
        dominated = np.any(
            valid
            & np.all(values <= point, axis=1)
            & np.any(values < point, axis=1)
        )
        mask[i] = not dominated
    return mask


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    shown = frame.loc[:, [c for c in columns if c in frame.columns]].copy()
    for col in shown.select_dtypes(include=["float"]).columns:
        if col == "learning_rate":
            shown[col] = shown[col].map(
                lambda value: "" if pd.isna(value) else f"{value:g}"
            )
        else:
            shown[col] = shown[col].map(
                lambda value: "" if pd.isna(value) else f"{value:.4f}"
            )
    headers = list(shown.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for values in shown.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in values) + " |")
    return "\n".join(lines) + "\n"


def plot_pareto(summary: pd.DataFrame, out_dir: Path) -> None:
    beta = summary["benign_lm_weight"].astype(float)
    front = pareto_mask(summary)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, prefix, title in zip(axes, ("seen", "heldout"), ("Seen-family", "Held-out families")):
        scatter = ax.scatter(
            100 * summary[f"{prefix}_frr" if prefix == "seen" else "heldout_mean_frr"],
            100 * summary[f"{prefix}_mean_asr"],
            c=beta,
            cmap="viridis",
            alpha=0.8,
            edgecolors=np.where(front, "red", "none"),
            linewidths=np.where(front, 1.2, 0.0),
        )
        x_col = f"{prefix}_frr" if prefix == "seen" else "heldout_mean_frr"
        y_col = f"{prefix}_mean_asr"
        if f"{x_col}_std" in summary.columns and f"{y_col}_std" in summary.columns:
            ax.errorbar(
                100 * summary[x_col],
                100 * summary[y_col],
                xerr=100 * summary[f"{x_col}_std"].fillna(0),
                yerr=100 * summary[f"{y_col}_std"].fillna(0),
                fmt="none",
                ecolor="0.55",
                elinewidth=0.7,
                alpha=0.45,
                zorder=0,
            )
        ax.set_xlabel("FRR (%)")
        ax.set_ylabel("Mean ASR (%)")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        for _, row in summary[front].iterrows():
            ax.annotate(f"{int(row.config_id)}", (100 * row[x_col], 100 * row[f"{prefix}_mean_asr"]), fontsize=7)
    fig.colorbar(scatter, ax=axes, label="Benign LM weight β")
    fig.savefig(out_dir / "pareto_seen_heldout.png", dpi=220)
    plt.close(fig)


def plot_per_benchmark(summary: pd.DataFrame, out_dir: Path) -> None:
    benchmark_suffixes = sorted(
        {
            match.group(1)
            for col in summary.columns
            if (match := re.match(r"seen_(.+)_mean_asr$", col))
            and match.group(1) not in {""}
        }
    )
    panels = [("aggregate", "seen_mean_asr", "heldout_mean_asr")]
    panels.extend(
        (name, f"seen_{name}_mean_asr", f"heldout_{name}_mean_asr")
        for name in benchmark_suffixes
        if f"heldout_{name}_mean_asr" in summary.columns
    )
    cols = 2
    rows = math.ceil(len(panels) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(11, 5 * rows), squeeze=False, constrained_layout=True)
    beta = summary["benign_lm_weight"].astype(float)
    for ax, (name, seen_col, held_col) in zip(axes.flat, panels):
        scatter = ax.scatter(100 * summary[seen_col], 100 * summary[held_col], c=beta, cmap="viridis", alpha=0.8)
        ax.plot([0, 100], [0, 100], linestyle="--", color="grey", linewidth=1)
        ax.set_xlabel("Seen-family mean ASR (%)")
        ax.set_ylabel("Held-out mean ASR (%)")
        ax.set_title(name.replace("_", " ").title())
        ax.grid(alpha=0.25)
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    fig.colorbar(scatter, ax=axes.ravel().tolist(), label="Benign LM weight β")
    fig.savefig(out_dir / "per_benchmark_seen_vs_heldout.png", dpi=220)
    plt.close(fig)


def aggregate_across_seeds(summary: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["phase", "config_id", "learning_rate", "lambda_val", "epsilon", "benign_lm_weight"]
    numeric = [col for col in summary.select_dtypes(include=["number"]).columns if col not in {"config_id", "seed", *group_cols}]
    grouped = summary.groupby(group_cols, as_index=False)[numeric].agg(["mean", "std"])
    grouped.columns = [
        "_".join(str(part) for part in col if str(part)) if isinstance(col, tuple) else col
        for col in grouped.columns
    ]
    return grouped


def config_mean_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Return one config row with metric means and adjacent standard deviations."""
    group_cols = [
        "phase",
        "config_id",
        "learning_rate",
        "lambda_val",
        "epsilon",
        "benign_lm_weight",
    ]
    numeric = [
        col
        for col in summary.select_dtypes(include=["number"]).columns
        if col not in {"config_id", "seed", *group_cols}
    ]
    means = summary.groupby(group_cols, as_index=False)[numeric].mean()
    stds = summary.groupby(group_cols, as_index=False)[numeric].std()
    for col in numeric:
        means[f"{col}_std"] = stds[col]
    means.insert(2, "seed", "mean")
    return means


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(manifest_path)
    metrics = metric_rows(manifest)
    metrics.to_csv(out_dir / "per_benchmark_metrics.csv", index=False)
    summary = build_seed_summary(metrics)
    summary.to_csv(out_dir / "seed_summary.csv", index=False)

    columns = [
        "config_id",
        "seed",
        "learning_rate",
        "lambda_val",
        "epsilon",
        "benign_lm_weight",
        *MAIN_METRICS,
    ]
    analysis_summary = summary
    if summary["seed"].nunique() > 1:
        across = aggregate_across_seeds(summary)
        across.to_csv(out_dir / "aggregate_across_seeds.csv", index=False)
        analysis_summary = config_mean_summary(summary)
        columns = [
            value
            for metric in columns
            for value in (
                [metric, f"{metric}_std"]
                if metric in MAIN_METRICS
                else [metric]
            )
        ]

    front = analysis_summary[pareto_mask(analysis_summary)].copy()
    front.to_csv(out_dir / "pareto_front.csv", index=False)
    (out_dir / "results_table.md").write_text(
        markdown_table(analysis_summary, columns),
        encoding="utf-8",
    )
    (out_dir / "pareto_front.md").write_text(
        markdown_table(front, columns),
        encoding="utf-8",
    )

    plot_pareto(analysis_summary, out_dir)
    plot_per_benchmark(analysis_summary, out_dir)
    print(f"Wrote per-benchmark metrics: {out_dir / 'per_benchmark_metrics.csv'}")
    print(f"Wrote main table: {out_dir / 'seed_summary.csv'}")
    print(f"Wrote non-dominated set (no automatic selection): {out_dir / 'pareto_front.csv'}")
    print(f"Wrote plots under {out_dir}")


if __name__ == "__main__":
    main()
