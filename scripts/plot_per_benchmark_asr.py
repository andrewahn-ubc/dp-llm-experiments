#!/usr/bin/env python3
"""Grouped bar chart: per-benchmark seen-family ASR for Base / Adv. SFT / DCL (Table 1 aggregate)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("figures/per_benchmark_asr.pdf"),
        help="Output PDF path.",
    )
    args = p.parse_args()

    # Table 1 — seen-family, “All attacks (aggregate)” ASR (%)
    benchmarks = ["AdvBench", "HarmBench", "JailbreakBench"]
    base = [10.5, 23.2, 18.2]
    adv_sft = [2.5, 14.0, 3.0]
    dcl = [2.1, 9.5, 8.5]

    rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "mathtext.fontset": "dejavuserif",
    }

    x = np.arange(len(benchmarks), dtype=float)
    width = 0.24

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(rc=rc):
        fig, ax = plt.subplots(figsize=(6.2, 3.6), layout="constrained")
        ax.bar(
            x - width,
            base,
            width,
            label="Base",
            color="#4c678a",
            edgecolor="black",
            linewidth=0.4,
        )
        ax.bar(
            x,
            adv_sft,
            width,
            label="Adv. SFT",
            color="#c98c5d",
            edgecolor="black",
            linewidth=0.4,
        )
        ax.bar(
            x + width,
            dcl,
            width,
            label="DCL (ours)",
            color="#5a9e6f",
            edgecolor="black",
            linewidth=0.4,
        )

        ax.set_ylabel("ASR (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(benchmarks)
        ax.set_ylim(0, 28)
        ax.legend(frameon=True, loc="upper right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.savefig(args.output, format="pdf")
        print(f"Wrote {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
