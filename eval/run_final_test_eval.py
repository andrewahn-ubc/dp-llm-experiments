#!/usr/bin/env python3
"""
Final **test-mode** evaluation for **one** (λ, ε) cell.

Each SLURM array task evaluates four checkpoints that share the same (λ, ε):

  • seen-family adapter
  • held-out gcg / autodan / pair adapters

Default grid (task-id → cell), matching ``run_final_pipeline._TEST_SWEEP_PAIRS``::

  0: λ=1, ε=-0.5
  1: λ=3, ε=0
  2: λ=3, ε=1
  3: λ=1, ε=-1
  4: λ=30, ε=0
  5: λ=30, ε=1

Writes under ``--out-dir / lam{lam}_eps{eps}/``:

  * harmful/benign CSVs
  * ``metrics.tsv``
  * ``points.csv`` — long-form rows for Pareto plotting
    (mode ∈ {seen, heldout}, family ∈ {all,gcg,autodan,pair},
     benchmark ∈ {advbench,harmbench,jailbreakbench,mean}, asr, frr, …)

After all array tasks finish, run ``eval/plot_final_test_pareto.py`` to build
the 32 Pareto charts (2 modes × 4 families × 4 benchmarks).
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
from train.model_profiles import MODEL_PROFILE_CHOICES  # noqa: E402

# Must match run_final_pipeline._TEST_SWEEP_PAIRS (order = SLURM array index).
TEST_PAIRS: list[tuple[float, float]] = [
    (1.0, -0.5),
    (3.0, 0.0),
    (3.0, 1.0),
    (1.0, -1.0),
    (30.0, 0.0),
    (30.0, 1.0),
]

BENCHMARKS = ("advbench", "harmbench", "jailbreakbench")
FAMILY_KEYS = ("all",) + FAMILIES


def _parse_pair(raw: str) -> tuple[float, float]:
    if ":" not in raw:
        raise argparse.ArgumentTypeError(f"expected 'lam:eps', got {raw!r}")
    a, b = raw.split(":", 1)
    return float(a.strip()), float(b.strip())


def _load_labels(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        print("[final_test_eval] WARN: no --labels-csv; benchmark ASR may be n/a.", flush=True)
        return None
    if not path.is_file():
        print(
            f"[final_test_eval] WARN: labels CSV not found: {path}; "
            "benchmark ASR may be n/a.",
            flush=True,
        )
        return None
    df = pd.read_csv(path)
    if "goal" not in df.columns or "dataset" not in df.columns:
        print(
            f"[final_test_eval] WARN: labels CSV {path} missing goal/dataset.",
            flush=True,
        )
        return None
    return df


def _attach_dataset(df: pd.DataFrame, labels: pd.DataFrame | None) -> pd.DataFrame:
    out = df.copy()
    if "dataset" in out.columns and out["dataset"].astype(str).str.strip().ne("").any():
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
    if "dataset" in out.columns:
        out = out.drop(columns=["dataset"])
    out = out.merge(lab[["goal", "dataset"]], on="goal", how="left")
    out["dataset"] = out["dataset"].fillna("").astype(str)
    n_ok = int((out["dataset"] != "").sum())
    print(
        f"[final_test_eval] attached dataset labels to {n_ok}/{len(out)} harmful rows",
        flush=True,
    )
    return out


def _asr_family(h: pd.DataFrame, family: str | None) -> float | None:
    if h is None or len(h) == 0:
        return None
    if family is None or family == "all":
        return mean_asr_harmful(h)
    col = SAFETY_COL[family]
    if col not in h.columns:
        return None
    return float((h[col].astype(str).str.lower() == "unsafe").mean())


def _mean_skip_nan(vals: list[float | None]) -> float | None:
    good = [v for v in vals if v is not None and v == v]
    if not good:
        return None
    return sum(good) / len(good)


def _cell_tag(lam: float, eps: float) -> str:
    return f"lam{lam:g}_eps{eps:g}"


def _make_task(lr: float, lam: float, eps: float, model_profile: str) -> Task:
    return Task(0, "clean_reg", lr, lam, eps, "clean", model_profile)


def build_points_for_mode(
    *,
    mode: str,
    per_family_harmful: dict[str, pd.DataFrame],
    per_family_frr: dict[str, float | None],
    labels: pd.DataFrame | None,
    shared_frr: float | None,
    lam: float,
    eps: float,
    lr: float,
) -> list[dict[str, Any]]:
    """Build long-form points for one mode (seen or heldout)."""
    labeled = {fam: _attach_dataset(df, labels) for fam, df in per_family_harmful.items()}
    rows: list[dict[str, Any]] = []

    for fam_key in FAMILY_KEYS:
        for bench in list(BENCHMARKS) + ["mean"]:
            if fam_key == "all":
                fam_asrs: list[float | None] = []
                fam_frrs: list[float | None] = []
                for fam in FAMILIES:
                    hdf = labeled[fam]
                    if bench == "mean":
                        # mean across benchmarks for this family first, then average families
                        bench_asrs = [
                            _asr_family(hdf[hdf["dataset"] == b], fam) for b in BENCHMARKS
                        ]
                        fam_asrs.append(_mean_skip_nan(bench_asrs))
                    else:
                        sub = hdf[hdf["dataset"] == bench]
                        fam_asrs.append(_asr_family(sub, fam))
                    if shared_frr is not None:
                        fam_frrs.append(shared_frr)
                    else:
                        fam_frrs.append(per_family_frr.get(fam))
                asr = _mean_skip_nan(fam_asrs)
                frr = _mean_skip_nan(fam_frrs)
            else:
                hdf = labeled[fam_key]
                if bench == "mean":
                    bench_asrs = [
                        _asr_family(hdf[hdf["dataset"] == b], fam_key) for b in BENCHMARKS
                    ]
                    asr = _mean_skip_nan(bench_asrs)
                else:
                    sub = hdf[hdf["dataset"] == bench]
                    asr = _asr_family(sub, fam_key)
                frr = shared_frr if shared_frr is not None else per_family_frr.get(fam_key)

            rows.append(
                {
                    "mode": mode,
                    "family": fam_key,
                    "benchmark": bench,
                    "asr": asr,
                    "frr": frr,
                    "lambda": lam,
                    "epsilon": eps,
                    "lr": lr,
                    "cell": _cell_tag(lam, eps),
                }
            )
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    scr = os.environ.get("SCRATCH", "")
    default_repo = f"{scr}/dp-llm-experiments" if scr else str(_REPO_ROOT)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-missing", action="store_true", default=True)
    p.add_argument("--no-skip-missing", action="store_false", dest="skip_missing")
    p.add_argument("--list-tasks", action="store_true", help="Print task grid and exit.")
    p.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="Index into TEST_PAIRS (or SLURM_ARRAY_TASK_ID). Ignored if --pair is set.",
    )
    p.add_argument(
        "--pair",
        type=_parse_pair,
        default=None,
        help="Explicit lam:eps (overrides --task-id).",
    )
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
    p.add_argument("--harmful-test", default="")
    p.add_argument("--benign-test", default="")
    p.add_argument("--labels-csv", default="")
    p.add_argument("--out-dir", default="")
    p.add_argument("--system-prompt-mode", choices=("default", "empty"), default="empty")
    p.add_argument("--benign-system-prompt-mode", choices=("default", "empty"), default="empty")
    args = p.parse_args(argv)

    repo = Path(expand_path(args.repo_root))
    if not args.eval_py:
        args.eval_py = str(repo / "eval" / "eval.py")
    if not args.harmful_test:
        args.harmful_test = str(repo / "official_data" / "llama3_test.csv")
    if not args.benign_test:
        text = repo / "official_data" / "frr_text.csv"
        test = repo / "official_data" / "frr_test.csv"
        args.benign_test = str(text if text.is_file() else test)
    if not args.labels_csv:
        args.labels_csv = str(repo / "official_data" / "combined_test_dataset.csv")
    if not args.out_dir:
        args.out_dir = str(Path(expand_path(args.checkpoint_root)) / "final_test_outputs")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.list_tasks:
        print(f"task_count={len(TEST_PAIRS)}")
        for i, (lam, eps) in enumerate(TEST_PAIRS):
            print(f"  {i}\tλ={lam:g}\tε={eps:g}\t{_cell_tag(lam, eps)}")
        return 0

    if args.pair is not None:
        lam, eps = args.pair
        task_id = None
    else:
        task_id = args.task_id
        if task_id is None and os.environ.get("SLURM_ARRAY_TASK_ID") is not None:
            task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        if task_id is None:
            print(
                "ERROR: pass --task-id N, --pair lam:eps, --list-tasks, "
                "or run under SLURM with SLURM_ARRAY_TASK_ID",
                file=sys.stderr,
            )
            return 2
        if task_id < 0 or task_id >= len(TEST_PAIRS):
            print(
                f"ERROR: task_id {task_id} out of range [0, {len(TEST_PAIRS) - 1}]",
                file=sys.stderr,
            )
            return 2
        lam, eps = TEST_PAIRS[task_id]

    cell = _cell_tag(lam, eps)
    ck_root = Path(expand_path(args.checkpoint_root))
    eval_py = Path(expand_path(args.eval_py))
    harmful_test = Path(expand_path(args.harmful_test))
    benign_test = Path(expand_path(args.benign_test))
    base_out = Path(expand_path(args.out_dir))
    out_dir = base_out / cell
    labels_path = Path(expand_path(args.labels_csv)) if args.labels_csv else None
    labels = _load_labels(labels_path)

    print(
        f"[final_test_eval] cell={cell} λ={lam:g} ε={eps:g} "
        f"(task_id={task_id if task_id is not None else 'pair'})",
        flush=True,
    )

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    tmp_benign = (
        Path(os.environ.get("SLURM_TMPDIR", "/tmp"))
        / f"frr_final_test_eval_{cell}.csv"
    )
    if not args.dry_run:
        prep_benign_csv(benign_test, tmp_benign)

    task = _make_task(args.lr, lam, eps, args.model_profile)
    seen_model = seen_ckpt(ck_root, task, args.epoch)

    metrics_rows: list[tuple[str, Any]] = [
        ("lr", args.lr),
        ("epoch", args.epoch),
        ("model_profile", args.model_profile),
        ("lambda", lam),
        ("epsilon", eps),
        ("cell", cell),
        ("slug", task.slug()),
        ("harmful_test", str(harmful_test)),
        ("benign_test", str(benign_test)),
        ("labels_csv", str(labels_path) if labels_path else ""),
    ]

    if not seen_model.is_dir():
        msg = f"[final_test_eval] Missing seen checkpoint: {seen_model}"
        if args.dry_run:
            print(f"[dry-run] WARN: {msg}", flush=True)
        elif args.skip_missing:
            print(msg, flush=True)
            return 0
        else:
            raise FileNotFoundError(msg)

    seen_tag = f"seen_{task.slug()}_ep{args.epoch}"
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

    held_h: dict[str, Path] = {}
    held_b: dict[str, Path] = {}
    for fam in FAMILIES:
        h_model = heldout_ckpt(ck_root, fam, task, args.epoch)
        if not h_model.is_dir():
            msg = f"[final_test_eval] Missing held-out checkpoint for {fam}: {h_model}"
            if args.dry_run:
                print(f"[dry-run] WARN: {msg}", flush=True)
            elif args.skip_missing:
                print(msg, flush=True)
                return 0
            else:
                raise FileNotFoundError(msg)

        tag = f"heldout_{fam}_{task.slug()}_ep{args.epoch}"
        held_h[fam] = out_dir / f"{tag}_harmful"
        held_b[fam] = out_dir / f"{tag}_benign"

        if args.dry_run:
            print(f"[dry-run] unseen-family {fam} resume_from={h_model}", flush=True)
            continue

        run_eval_py(
            eval_py=eval_py,
            resume_from=h_model,
            eval_mode="unseen-family",
            unseen_family=fam,
            harmful_csv=harmful_test,
            benign_csv=tmp_benign,
            harmful_stem=held_h[fam],
            benign_stem=held_b[fam],
            system_prompt_mode=args.system_prompt_mode,
            benign_system_prompt_mode=args.benign_system_prompt_mode,
            model_profile=args.model_profile,
        )

    if args.dry_run:
        print("[dry-run] skipping metrics / points", flush=True)
        return 0

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

    write_metrics_tsv(out_dir / "metrics.tsv", metrics_rows)

    points = []
    points.extend(
        build_points_for_mode(
            mode="seen",
            per_family_harmful={fam: h_seen for fam in FAMILIES},
            per_family_frr={fam: seen_frr for fam in FAMILIES},
            labels=labels,
            shared_frr=seen_frr,
            lam=lam,
            eps=eps,
            lr=args.lr,
        )
    )
    points.extend(
        build_points_for_mode(
            mode="heldout",
            per_family_harmful=held_dfs,
            per_family_frr=held_frrs,
            labels=labels,
            shared_frr=None,
            lam=lam,
            eps=eps,
            lr=args.lr,
        )
    )
    points_df = pd.DataFrame(points)
    points_path = out_dir / "points.csv"
    points_df.to_csv(points_path, index=False)
    # Also copy to base out-dir for easy globbing by the plotter.
    base_out.mkdir(parents=True, exist_ok=True)
    points_df.to_csv(base_out / f"points_{cell}.csv", index=False)

    print(f"[final_test_eval] Wrote {out_dir / 'metrics.tsv'}", flush=True)
    print(f"[final_test_eval] Wrote {points_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
