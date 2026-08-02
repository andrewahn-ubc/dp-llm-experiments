#!/usr/bin/env python3
"""Eval one Llama-3 multi-seed train config (array task) and write points CSV.

Seen adapters → ``--eval-mode seen-family`` (all three attack columns + FRR).
Held-out adapters → ``--eval-mode unseen-family`` for that family only + FRR.

Outputs under ``--out-dir/seed{S}/``:
  * ``{tag}_harmful.csv`` / ``{tag}_benign.csv``
  * ``{tag}_metrics.tsv``
  * ``points_{config_id}.csv`` (long-form; percent ASR/FRR for aggregation)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from eval.run_final_test_eval import (  # noqa: E402
    BENCHMARKS,
    _attach_dataset,
    _asr_family,
    _load_labels,
    _mean_skip_nan,
    _resolve_labels_path,
)
from eval.test_eval_matrix import (  # noqa: E402
    expand_path,
    frr_benign,
    mean_asr_harmful,
    prep_benign_csv,
    resolve_output,
    run_eval_py,
    seen_asr_one_family,
    write_metrics_tsv,
)
from experiments.llama3_multiseed.configs import (  # noqa: E402
    EPOCH,
    FAMILIES,
    LR,
    MODEL_PROFILE,
    TRAIN_CONFIGS,
    array_index_to_seed_config,
)

SAFETY_COL = {"gcg": "GCG Safety", "autodan": "AutoDAN Safety", "pair": "PAIR Safety"}


def _pct(x: float | None) -> float | None:
    if x is None or x != x:
        return None
    return float(x) * 100.0


def _soft_attach_dataset(df: pd.DataFrame, labels: pd.DataFrame | None) -> pd.DataFrame:
    """Like ``_attach_dataset`` but never abort after a successful GPU eval.

    Missing / unmatched labels → empty ``dataset`` (per-benchmark ASR becomes NaN);
    aggregate still gets pooled cells from family means where possible.
    """
    out = df.copy()
    if "dataset" in out.columns and out["dataset"].astype(str).str.strip().ne("").any():
        out["dataset"] = out["dataset"].astype(str).str.strip().str.lower()
        return out
    if labels is None or "goal" not in out.columns:
        out["dataset"] = ""
        print("[eval_one] WARN: no dataset labels; per-benchmark ASR will be empty", flush=True)
        return out
    try:
        return _attach_dataset(out, labels)
    except SystemExit as e:
        print(f"[eval_one] WARN: dataset join failed ({e}); continuing without benches", flush=True)
        out["dataset"] = ""
        return out


def _build_points(
    *,
    mode: str,
    seed: int,
    config_id: str,
    lam: float,
    eps: float,
    harmful: pd.DataFrame,
    frr: float | None,
    labels: pd.DataFrame | None,
    families: tuple[str, ...],
) -> list[dict[str, Any]]:
    labeled = _soft_attach_dataset(harmful, labels)
    rows: list[dict[str, Any]] = []
    for fam in families:
        for bench in list(BENCHMARKS) + ["mean"]:
            if bench == "mean":
                asr = _mean_skip_nan(
                    [_asr_family(labeled[labeled["dataset"] == b], fam) for b in BENCHMARKS]
                )
            else:
                asr = _asr_family(labeled[labeled["dataset"] == bench], fam)
            # Adv-SFT seen: mean over families for "all"
            if fam == "all":
                if bench == "mean":
                    asr = _mean_skip_nan(
                        [
                            _mean_skip_nan(
                                [
                                    _asr_family(labeled[labeled["dataset"] == b], f)
                                    for b in BENCHMARKS
                                ]
                            )
                            for f in FAMILIES
                        ]
                    )
                else:
                    asr = _mean_skip_nan(
                        [_asr_family(labeled[labeled["dataset"] == bench], f) for f in FAMILIES]
                    )
            rows.append(
                {
                    "seed": seed,
                    "mode": mode,
                    "config_id": config_id,
                    "family": fam,
                    "benchmark": bench,
                    "asr": _pct(asr),
                    "frr": _pct(frr),
                    "lambda": lam,
                    "epsilon": eps,
                    "lr": LR,
                    "epoch": EPOCH,
                }
            )
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    scr = os.environ.get("SCRATCH", "")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task-id", type=int, default=None)
    p.add_argument("--list-tasks", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--checkpoint-root",
        default=os.environ.get("CHECKPOINT_ROOT", "$SCRATCH/dp-llm-sweep/multiseed_l3"),
    )
    p.add_argument(
        "--out-dir",
        default=os.environ.get("EVAL_OUT_DIR", "$SCRATCH/dp-llm-eval/llama3_multiseed"),
    )
    p.add_argument("--repo-root", default=scr and f"{scr}/dp-llm-experiments" or str(_REPO))
    p.add_argument("--harmful-test", default="")
    p.add_argument("--benign-test", default="")
    p.add_argument("--labels-csv", default="")
    p.add_argument("--model-profile", default=MODEL_PROFILE)
    p.add_argument("--epoch", type=int, default=EPOCH)
    # Default False: missing checkpoints must fail the SLURM task (not silent success).
    p.add_argument("--skip-missing", action="store_true", default=False)
    p.add_argument("--no-skip-missing", action="store_false", dest="skip_missing")
    args = p.parse_args(argv)

    repo = Path(expand_path(args.repo_root))
    if not args.harmful_test:
        for rel in (
            Path("official_data") / "llama3_test.csv",
            Path("official") / "llama3_test.csv",
        ):
            cand = repo / rel
            if cand.is_file():
                args.harmful_test = str(cand)
                break
        else:
            args.harmful_test = str(repo / "official_data" / "llama3_test.csv")
    if not args.benign_test:
        for rel in (
            Path("official_data") / "frr_text.csv",
            Path("official_data") / "frr_test.csv",
            Path("official") / "frr_text.csv",
            Path("official") / "frr_test.csv",
        ):
            cand = repo / rel
            if cand.is_file():
                args.benign_test = str(cand)
                break
        else:
            args.benign_test = str(repo / "official_data" / "frr_test.csv")
    if not args.labels_csv:
        resolved = _resolve_labels_path(repo, "")
        # Prefer an existing file; do not keep a non-existent default path.
        if resolved is not None and Path(expand_path(str(resolved))).is_file():
            args.labels_csv = str(resolved)
        else:
            args.labels_csv = ""
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.list_tasks:
        from experiments.llama3_multiseed.configs import list_train_jobs

        list_train_jobs()
        return 0

    task_id = args.task_id
    if task_id is None and os.environ.get("SLURM_ARRAY_TASK_ID") is not None:
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if task_id is None:
        print("ERROR: pass --task-id or set SLURM_ARRAY_TASK_ID", file=sys.stderr)
        return 2

    seed, cfg = array_index_to_seed_config(task_id)
    ck_root = expand_path(args.checkpoint_root)
    out_base = Path(expand_path(args.out_dir))
    out_dir = out_base / f"seed{seed}"
    repo = Path(expand_path(args.repo_root))
    eval_py = repo / "eval" / "eval.py"
    harmful_test = Path(expand_path(args.harmful_test))
    benign_test = Path(expand_path(args.benign_test))
    # Never abort the job solely for missing labels (GPU eval is the expensive part).
    labels = _load_labels(
        Path(expand_path(args.labels_csv)) if args.labels_csv else None,
        required=False,
    )

    ckpt = Path(cfg.ckpt_path(ck_root, seed, args.epoch))
    tag = f"{cfg.config_id}_{cfg.run_slug(seed)}_ep{args.epoch}"
    print(
        f"[eval_one] task={task_id} seed={seed} config={cfg.config_id} ckpt={ckpt}",
        flush=True,
    )

    if not harmful_test.is_file():
        print(f"ERROR: harmful test CSV missing: {harmful_test}", file=sys.stderr)
        return 2
    if not benign_test.is_file():
        print(f"ERROR: benign/FRR CSV missing: {benign_test}", file=sys.stderr)
        return 2
    if not eval_py.is_file():
        print(f"ERROR: eval.py missing: {eval_py}", file=sys.stderr)
        return 2

    if not ckpt.is_dir() or not (ckpt / "adapter_config.json").is_file():
        msg = f"Missing/incomplete checkpoint: {ckpt}"
        if args.dry_run:
            print(f"[dry-run] WARN: {msg}", flush=True)
            return 0
        if args.skip_missing:
            print(msg, "; skip", flush=True)
            return 0
        print(f"ERROR: {msg}", file=sys.stderr)
        return 2

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    tmp_benign = (
        Path(os.environ.get("SLURM_TMPDIR", "/tmp")) / f"frr_l3ms_{seed}_{cfg.config_id}.csv"
    )
    if not args.dry_run:
        prep_benign_csv(benign_test, tmp_benign)

    h_stem = out_dir / f"{tag}_harmful"
    b_stem = out_dir / f"{tag}_benign"

    if cfg.role == "seen":
        eval_mode = "seen-family"
        unseen = None
        point_families: tuple[str, ...] = FAMILIES + ("all",)
        mode = "seen"
    else:
        eval_mode = "unseen-family"
        unseen = cfg.heldout_family
        point_families = (cfg.heldout_family,)  # type: ignore[assignment]
        mode = "heldout"

    if args.dry_run:
        print(f"[dry-run] {eval_mode} resume_from={ckpt} unseen={unseen}", flush=True)
        return 0

    run_eval_py(
        eval_py=eval_py,
        resume_from=ckpt,
        eval_mode=eval_mode,
        unseen_family=unseen,
        harmful_csv=harmful_test,
        benign_csv=tmp_benign,
        harmful_stem=h_stem,
        benign_stem=b_stem,
        system_prompt_mode="empty",
        benign_system_prompt_mode="empty",
        model_profile=args.model_profile,
    )

    sh = resolve_output(h_stem)
    sb = resolve_output(b_stem)
    if sh is None or sb is None:
        raise FileNotFoundError(f"Missing eval outputs under {out_dir}")
    h = pd.read_csv(sh)
    b = pd.read_csv(sb)
    frr = frr_benign(b)

    metrics: list[tuple[str, Any]] = [
        ("seed", seed),
        ("config_id", cfg.config_id),
        ("slug", cfg.run_slug(seed)),
        ("role", cfg.role),
        ("lambda", cfg.lam),
        ("epsilon", cfg.eps),
        ("lm_loss_input", cfg.lm_loss_input),
        ("epoch", args.epoch),
        ("frr", frr),
        ("mean_asr", mean_asr_harmful(h)),
        ("harmful_csv", str(sh)),
        ("benign_csv", str(sb)),
    ]
    if cfg.role == "seen":
        for fam in FAMILIES:
            metrics.append((f"seen_{fam}_asr", seen_asr_one_family(h, fam)))
    else:
        fam = cfg.heldout_family or ""
        col = SAFETY_COL.get(fam)
        if col and col in h.columns:
            metrics.append(
                (f"{fam}_heldout_asr", float((h[col].astype(str).str.lower() == "unsafe").mean()))
            )

    write_metrics_tsv(out_dir / f"{tag}_metrics.tsv", metrics)

    points = _build_points(
        mode=mode,
        seed=seed,
        config_id=cfg.config_id,
        lam=cfg.lam,
        eps=cfg.eps,
        harmful=h,
        frr=frr,
        labels=labels,
        families=point_families,
    )
    points_path = out_dir / f"points_{cfg.config_id}.csv"
    pd.DataFrame(points).to_csv(points_path, index=False)
    print(f"[eval_one] Wrote {points_path}", flush=True)
    return 0


if __name__ == "__main__":
    # silence unused import warning for TRAIN_CONFIGS (used via array helper)
    _ = TRAIN_CONFIGS
    sys.exit(main())
