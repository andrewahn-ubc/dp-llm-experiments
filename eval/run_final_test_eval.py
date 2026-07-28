#!/usr/bin/env python3
"""
Final **test-mode** evaluation: run the four chosen (λ, ε) checkpoints with the
per-family mapping from hyperparameter search, then write SEEN-FAMILY and
HELDOUT-FAMILY summary tables.

Model mapping (default; override via CLI)::

  SEEN (gcg / autodan / pair):   λ=1, ε=-0.5   → seen adapter
  HELDOUT autodan:               λ=3, ε=0      → heldout_autodan adapter
  HELDOUT gcg:                   λ=3, ε=1      → heldout_gcg adapter
  HELDOUT pair:                  λ=1, ε=-1     → heldout_pair adapter

Outputs under ``--out-dir``:

  * per-eval harmful/benign CSVs (same naming as test_eval_matrix)
  * ``final_test_metrics.tsv`` — scalar ASR/FRR used for the tables
  * ``seen_family_table.{md,csv}`` / ``heldout_family_table.{md,csv}``

Table layout (ASR / FRR in each cell)::

  rows:    all attack families | gcg | autodan | pair
  columns: advbench | harmbench | jailbreakbench | mean across benchmarks

FRR is computed on the benign CSV (``frr_test.csv`` / ``frr_text.csv``) and is
**not** split by benchmark — the same FRR is shown in every benchmark column for
a given row. For HELDOUT "all attack families", FRR is the mean of the three
held-out models' FRRs.

Benchmark labels: if the harmful CSV lacks a ``dataset`` column, rows are joined
to ``--labels-csv`` on ``goal`` (default: ``combined_test_dataset.csv``).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EVAL_DIR = Path(__file__).resolve().parent
# Prefer the eval/ directory for sibling imports (test_eval_matrix), then repo root
# for train.*. Do NOT leave this file's parent alone at sys.path[0] under the name
# "eval" — that shadows the package and pulls in eval/eval.py (heavy deps).
if str(_EVAL_DIR) in sys.path:
    sys.path.remove(str(_EVAL_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from test_eval_matrix import (  # noqa: E402
    FAMILIES,
    SAFETY_COL,
    Task,
    asr_one_family,
    expand_path,
    frr_benign,
    heldout_ckpt,
    mean_asr_harmful,
    prep_benign_csv,
    resolve_output,
    run_eval_py,
    seen_asr_one_family,
    seen_ckpt,
    write_metrics_tsv,
)
from train.model_profiles import (  # noqa: E402
    MODEL_PROFILE_CHOICES,
)

BENCHMARKS = ("advbench", "harmbench", "jailbreakbench")
ROW_KEYS = ("all", "gcg", "autodan", "pair")
ROW_LABELS = {
    "all": "all attack families",
    "gcg": "gcg",
    "autodan": "autodan",
    "pair": "pair",
}

# Default hyperparameter → eval mapping from validation selection.
DEFAULT_SEEN_LAM, DEFAULT_SEEN_EPS = 1.0, -0.5
DEFAULT_HELDOUT: dict[str, tuple[float, float]] = {
    "autodan": (3.0, 0.0),
    "gcg": (3.0, 1.0),
    "pair": (1.0, -1.0),
}


def _parse_pair(raw: str) -> tuple[float, float]:
    if ":" not in raw:
        raise argparse.ArgumentTypeError(f"expected 'lam:eps', got {raw!r}")
    a, b = raw.split(":", 1)
    return float(a.strip()), float(b.strip())


def _attach_dataset(df: pd.DataFrame, labels: pd.DataFrame | None) -> pd.DataFrame:
    out = df.copy()
    if "dataset" in out.columns:
        out["dataset"] = out["dataset"].astype(str).str.strip().str.lower()
        return out
    if labels is None or "goal" not in out.columns:
        out["dataset"] = ""
        return out
    lab = labels.copy()
    lab["goal"] = lab["goal"].astype(str)
    lab["dataset"] = lab["dataset"].astype(str).str.strip().str.lower()
    lab = lab.drop_duplicates(subset=["goal"], keep="first")
    out["goal"] = out["goal"].astype(str)
    out = out.merge(lab[["goal", "dataset"]], on="goal", how="left")
    out["dataset"] = out["dataset"].fillna("").astype(str)
    return out


def _load_labels(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.is_file():
        return None
    df = pd.read_csv(path)
    if "goal" not in df.columns or "dataset" not in df.columns:
        print(
            f"[final_test_eval] WARN: labels CSV {path} missing goal/dataset; "
            "benchmark columns may be empty.",
            flush=True,
        )
        return None
    return df


def _asr_on_subset(h: pd.DataFrame, family: str | None) -> float | None:
    """family=None → mean over available family safety columns."""
    if h is None or len(h) == 0:
        return None
    if family is None:
        return mean_asr_harmful(h)
    col = SAFETY_COL[family]
    if col not in h.columns:
        return None
    return float((h[col].astype(str).str.lower() == "unsafe").mean())


def _fmt_cell(asr: float | None, frr: float | None) -> str:
    asr_s = f"{asr:.4f}" if asr is not None and asr == asr else "n/a"
    frr_s = f"{frr:.4f}" if frr is not None and frr == frr else "n/a"
    return f"ASR {asr_s} / FRR {frr_s}"


def _mean_skip_nan(vals: list[float | None]) -> float | None:
    good = [v for v in vals if v is not None and v == v]
    if not good:
        return None
    return sum(good) / len(good)


def build_family_table(
    *,
    per_family_harmful: dict[str, pd.DataFrame],
    per_family_frr: dict[str, float | None],
    labels: pd.DataFrame | None,
    shared_frr: float | None = None,
) -> pd.DataFrame:
    """Build one SEEN or HELDOUT table.

    ``per_family_harmful`` maps family → harmful eval CSV (with safety cols).
    For SEEN, all three families share one CSV (same DataFrame thrice) and
    ``shared_frr`` is used for every row. For HELDOUT, each family has its own
    CSV + FRR; the "all" row averages ASRs and FRRs across families.
    """
    labeled: dict[str, pd.DataFrame] = {
        fam: _attach_dataset(df, labels) for fam, df in per_family_harmful.items()
    }

    rows_out: list[dict[str, str]] = []
    for row_key in ROW_KEYS:
        cells: dict[str, str] = {"family": ROW_LABELS[row_key]}
        bench_asrs: list[float | None] = []
        bench_frrs: list[float | None] = []

        for bench in list(BENCHMARKS) + ["mean"]:
            if bench == "mean":
                asr = _mean_skip_nan(bench_asrs)
                frr = _mean_skip_nan(bench_frrs)
                cells["mean across benchmarks"] = _fmt_cell(asr, frr)
                continue

            if row_key == "all":
                fam_asrs: list[float | None] = []
                fam_frrs: list[float | None] = []
                for fam in FAMILIES:
                    hdf = labeled[fam]
                    sub = hdf[hdf["dataset"] == bench]
                    fam_asrs.append(_asr_on_subset(sub, fam))
                    if shared_frr is not None:
                        fam_frrs.append(shared_frr)
                    else:
                        fam_frrs.append(per_family_frr.get(fam))
                asr = _mean_skip_nan(fam_asrs)
                frr = _mean_skip_nan(fam_frrs)
            else:
                hdf = labeled[row_key]
                sub = hdf[hdf["dataset"] == bench]
                asr = _asr_on_subset(sub, row_key)
                frr = shared_frr if shared_frr is not None else per_family_frr.get(row_key)

            bench_asrs.append(asr)
            bench_frrs.append(frr)
            cells[bench] = _fmt_cell(asr, frr)

        rows_out.append(cells)

    cols = ["family"] + list(BENCHMARKS) + ["mean across benchmarks"]
    return pd.DataFrame(rows_out, columns=cols)


def table_to_markdown(df: pd.DataFrame, title: str) -> str:
    lines = [f"## {title}", ""]
    headers = list(df.columns)
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in headers) + " |")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    scr = os.environ.get("SCRATCH", "")
    default_repo = f"{scr}/dp-llm-experiments" if scr else str(_REPO_ROOT)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-missing", action="store_true", default=True)
    p.add_argument("--no-skip-missing", action="store_false", dest="skip_missing")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epoch", type=int, default=2)
    p.add_argument(
        "--model-profile",
        default=os.environ.get("MODEL_PROFILE", "llama_3_8b_instruct"),
        choices=list(MODEL_PROFILE_CHOICES),
    )
    p.add_argument("--checkpoint-root", default="$SCRATCH/dp-llm-sweep")
    p.add_argument("--repo-root", default=default_repo)
    p.add_argument("--eval-py", default="")
    p.add_argument(
        "--harmful-test",
        default="",
        help="Harmful ASR test CSV (default: official_data/llama3_test.csv).",
    )
    p.add_argument(
        "--benign-test",
        default="",
        help=(
            "Benign FRR test CSV. Default: official_data/frr_test.csv "
            "(alias often called frr_text.csv)."
        ),
    )
    p.add_argument(
        "--labels-csv",
        default="",
        help=(
            "CSV with goal+dataset for benchmark splits (default: "
            "official_data/combined_test_dataset.csv). Ignored if harmful CSV "
            "already has a dataset column."
        ),
    )
    p.add_argument("--out-dir", default="")
    p.add_argument("--system-prompt-mode", choices=("default", "empty"), default="empty")
    p.add_argument("--benign-system-prompt-mode", choices=("default", "empty"), default="empty")
    p.add_argument("--seen-pair", type=_parse_pair, default=(DEFAULT_SEEN_LAM, DEFAULT_SEEN_EPS))
    p.add_argument("--heldout-autodan-pair", type=_parse_pair, default=DEFAULT_HELDOUT["autodan"])
    p.add_argument("--heldout-gcg-pair", type=_parse_pair, default=DEFAULT_HELDOUT["gcg"])
    p.add_argument("--heldout-pair-pair", type=_parse_pair, default=DEFAULT_HELDOUT["pair"])
    args = p.parse_args(argv)

    repo = Path(expand_path(args.repo_root))
    if not args.eval_py:
        args.eval_py = str(repo / "eval" / "eval.py")
    if not args.harmful_test:
        args.harmful_test = str(repo / "official_data" / "llama3_test.csv")
    if not args.benign_test:
        # Prefer frr_text.csv if present (user naming); else frr_test.csv.
        text = repo / "official_data" / "frr_text.csv"
        test = repo / "official_data" / "frr_test.csv"
        args.benign_test = str(text if text.is_file() else test)
    if not args.labels_csv:
        args.labels_csv = str(repo / "official_data" / "combined_test_dataset.csv")
    if not args.out_dir:
        args.out_dir = str(Path(expand_path(args.checkpoint_root)) / "final_test_outputs")

    held = {
        "autodan": args.heldout_autodan_pair,
        "gcg": args.heldout_gcg_pair,
        "pair": args.heldout_pair_pair,
    }
    args.heldout_map = held
    return args


def _make_task(lr: float, lam: float, eps: float, model_profile: str) -> Task:
    return Task(0, "clean_reg", lr, lam, eps, "clean", model_profile)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ck_root = Path(expand_path(args.checkpoint_root))
    eval_py = Path(expand_path(args.eval_py))
    harmful_test = Path(expand_path(args.harmful_test))
    benign_test = Path(expand_path(args.benign_test))
    out_dir = Path(expand_path(args.out_dir))
    labels_path = Path(expand_path(args.labels_csv)) if args.labels_csv else None
    labels = _load_labels(labels_path)

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    tmp_benign = Path(os.environ.get("SLURM_TMPDIR", "/tmp")) / "frr_final_test_eval.csv"
    prep_benign_csv(benign_test, tmp_benign) if not args.dry_run else None

    seen_lam, seen_eps = args.seen_pair
    seen_task = _make_task(args.lr, seen_lam, seen_eps, args.model_profile)
    seen_model = seen_ckpt(ck_root, seen_task, args.epoch)

    metrics_rows: list[tuple[str, Any]] = [
        ("lr", args.lr),
        ("epoch", args.epoch),
        ("model_profile", args.model_profile),
        ("seen_lambda", seen_lam),
        ("seen_epsilon", seen_eps),
        ("seen_slug", seen_task.slug()),
        ("harmful_test", str(harmful_test)),
        ("benign_test", str(benign_test)),
        ("labels_csv", str(labels_path) if labels_path else ""),
    ]

    # --- Seen-family (one adapter for all three attack families) ---
    if not seen_model.is_dir():
        msg = f"[final_test_eval] Missing seen checkpoint: {seen_model}"
        if args.dry_run:
            print(f"[dry-run] WARN: {msg}", flush=True)
        elif args.skip_missing:
            print(msg, flush=True)
            return 0
        else:
            raise FileNotFoundError(msg)

    seen_tag = f"final_seen_{seen_task.slug()}_ep{args.epoch}"
    seen_h_stem = out_dir / f"{seen_tag}_harmful"
    seen_b_stem = out_dir / f"{seen_tag}_benign"

    if args.dry_run:
        print(f"[dry-run] seen-family resume_from={seen_model}", flush=True)
    else:
        run_eval_py(
            eval_py=eval_py,
            resume_from=seen_model,
            eval_mode="seen-family",
            unseen_family=None,
            harmful_csv=harmful_test,
            benign_csv=tmp_benign,
            harmful_stem=seen_h_stem,
            benign_stem=seen_b_stem,
            system_prompt_mode=args.system_prompt_mode,
            benign_system_prompt_mode=args.benign_system_prompt_mode,
            model_profile=args.model_profile,
        )

    # --- Held-out per family (each with its own (λ, ε)) ---
    held_h: dict[str, Path] = {}
    held_b: dict[str, Path] = {}
    for fam in FAMILIES:
        lam, eps = args.heldout_map[fam]
        task = _make_task(args.lr, lam, eps, args.model_profile)
        h_model = heldout_ckpt(ck_root, fam, task, args.epoch)
        metrics_rows.extend(
            [
                (f"heldout_{fam}_lambda", lam),
                (f"heldout_{fam}_epsilon", eps),
                (f"heldout_{fam}_slug", task.slug()),
            ]
        )
        if not h_model.is_dir():
            msg = f"[final_test_eval] Missing held-out checkpoint for {fam}: {h_model}"
            if args.dry_run:
                print(f"[dry-run] WARN: {msg}", flush=True)
            elif args.skip_missing:
                print(msg, flush=True)
                return 0
            else:
                raise FileNotFoundError(msg)

        tag = f"final_heldout_{fam}_{task.slug()}_ep{args.epoch}"
        h_stem = out_dir / f"{tag}_harmful"
        b_stem = out_dir / f"{tag}_benign"
        held_h[fam] = h_stem
        held_b[fam] = b_stem

        if args.dry_run:
            print(
                f"[dry-run] unseen-family {fam} λ={lam:g} ε={eps:g} resume_from={h_model}",
                flush=True,
            )
            continue

        run_eval_py(
            eval_py=eval_py,
            resume_from=h_model,
            eval_mode="unseen-family",
            unseen_family=fam,
            harmful_csv=harmful_test,
            benign_csv=tmp_benign,
            harmful_stem=h_stem,
            benign_stem=b_stem,
            system_prompt_mode=args.system_prompt_mode,
            benign_system_prompt_mode=args.benign_system_prompt_mode,
            model_profile=args.model_profile,
        )

    if args.dry_run:
        print("[dry-run] skipping table build", flush=True)
        return 0

    # Load outputs
    sh = resolve_output(seen_h_stem)
    sb = resolve_output(seen_b_stem)
    if sh is None or sb is None:
        raise FileNotFoundError(f"Missing seen outputs under {out_dir}")
    h_seen = pd.read_csv(sh)
    b_seen = pd.read_csv(sb)
    seen_frr = frr_benign(b_seen)
    metrics_rows.extend(
        [
            ("seen_mean_asr", mean_asr_harmful(h_seen)),
            ("seen_frr", seen_frr),
            ("seen_harmful_csv", str(sh)),
            ("seen_benign_csv", str(sb)),
        ]
    )
    for fam in FAMILIES:
        metrics_rows.append((f"seen_{fam}_asr", seen_asr_one_family(h_seen, fam)))

    held_dfs: dict[str, pd.DataFrame] = {}
    held_frrs: dict[str, float | None] = {}
    for fam in FAMILIES:
        hf = resolve_output(held_h[fam])
        bf = resolve_output(held_b[fam])
        if hf is None or bf is None:
            raise FileNotFoundError(f"Missing held-out outputs for {fam}")
        hdf = pd.read_csv(hf)
        bdf = pd.read_csv(bf)
        held_dfs[fam] = hdf
        held_frrs[fam] = frr_benign(bdf)
        metrics_rows.extend(
            [
                (f"{fam}_heldout_asr", asr_one_family(hdf, fam)),
                (f"{fam}_model_frr", held_frrs[fam]),
                (f"{fam}_harmful_csv", str(hf)),
                (f"{fam}_benign_csv", str(bf)),
            ]
        )

    write_metrics_tsv(out_dir / "final_test_metrics.tsv", metrics_rows)

    # Tables: SEEN uses the same harmful CSV for every family column.
    seen_table = build_family_table(
        per_family_harmful={fam: h_seen for fam in FAMILIES},
        per_family_frr={fam: seen_frr for fam in FAMILIES},
        labels=labels,
        shared_frr=seen_frr,
    )
    held_table = build_family_table(
        per_family_harmful=held_dfs,
        per_family_frr=held_frrs,
        labels=labels,
        shared_frr=None,
    )

    seen_csv = out_dir / "seen_family_table.csv"
    held_csv = out_dir / "heldout_family_table.csv"
    seen_md = out_dir / "seen_family_table.md"
    held_md = out_dir / "heldout_family_table.md"
    report_md = out_dir / "final_test_report.md"

    seen_table.to_csv(seen_csv, index=False)
    held_table.to_csv(held_csv, index=False)
    seen_md.write_text(table_to_markdown(seen_table, "SEEN-FAMILY"), encoding="utf-8")
    held_md.write_text(table_to_markdown(held_table, "HELDOUT-FAMILY"), encoding="utf-8")
    report_md.write_text(
        "# Final test results\n\n"
        f"Seen model: λ={seen_lam:g}, ε={seen_eps:g} (`{seen_task.slug()}`)\n\n"
        + "\n".join(
            f"Held-out {fam}: λ={args.heldout_map[fam][0]:g}, "
            f"ε={args.heldout_map[fam][1]:g}"
            for fam in FAMILIES
        )
        + "\n\n"
        + table_to_markdown(seen_table, "SEEN-FAMILY")
        + "\n"
        + table_to_markdown(held_table, "HELDOUT-FAMILY"),
        encoding="utf-8",
    )

    print(f"[final_test_eval] Wrote {seen_csv}", flush=True)
    print(f"[final_test_eval] Wrote {held_csv}", flush=True)
    print(f"[final_test_eval] Wrote {report_md}", flush=True)
    print(table_to_markdown(seen_table, "SEEN-FAMILY"), flush=True)
    print(table_to_markdown(held_table, "HELDOUT-FAMILY"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
