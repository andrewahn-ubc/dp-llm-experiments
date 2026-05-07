#!/usr/bin/env python3
"""
Single entry point for the **final** experiment on Narval:

  1. Submit **seen-family** training with **clean** LM (λ×ε grid via
     ``train/submit_wandb_sweep.LAMBDAS`` / ``EPSILONS`` and ``lambda_epsilon_pairs``). Each cell is **one** SLURM job that runs **only**
     ``train.py`` once by default on the **first half** of the shuffled CSV (``--training-halves-phase first``,
     ``--total-epochs 1`` in the sweep sense → checkpoint ``…_finetuned_llm_epoch1``). Use
     ``python run_final_pipeline.py --training-half second`` (same ``--model`` / ``--lr``) for the
     second half → ``…_epoch2``. SLURM wall time defaults to **2 hours** per half (``--time 2:00:00``),
     i.e. **4 hours per full epoch** of data; ``--training-half full`` defaults to **4 hours** for one
     full-data pass.
     Test metrics use ``submit_test_eval_matrix.sh`` with ``EPOCH`` matching the half (1 or 2).

  2. Submit **seen-family** training with **perturbed** LM only at **λ=0** (one run;
     representative ε matches ``lambda_epsilon_pairs`` for λ=0), via
     ``--lm-loss-input perturbed --perturbed-sweep-subset lambda0_only``; slugs use ``_pertlm``.

  3. Submit **held-out** training for **clean** LM (three families × same grid as (1)).

  4. Submit **held-out** training for **perturbed** LM at λ=0 only (same as (2)).

  5. Submit the SLURM array for ``test_eval_matrix.py`` (task count matches the grid in
     ``submit_wandb_sweep`` / ``eval/test_eval_matrix.py``; default pipeline: **2** tasks:
     1 clean (λ=0.1, ε=-1) + 1 perturbed-at-λ=0; 1× seen + 3× unseen eval per task).

  6. Submit **CPU** heatmaps: **13** per-metric panel PNGs plus **combined_clean_lm_dashboard.png**
     under ``…/heatmaps_<MODEL>_lr<LR>/aggregate/`` and each
     ``…/heatmaps_<MODEL>_lr<LR>/by_dataset/<benchmark>/``. Folder names come from the eval
     ``*_metrics.tsv`` files (no manual exports). Labels default to
     ``official_data/combined_test_dataset.csv`` (must include ``dataset``). Eval artifacts live under
     ``$CHECKPOINT_ROOT/test_eval_outputs`` (same as ``test_eval_matrix``).

By default the eval array waits on training (``--dependency=after:<ids>``). Use
``--parallel-eval`` to overlap eval with training.

Training manifests land under ``sweep_jobs/``; job ids append to ``sweep_jobs/training_job_ids.txt``.

Run from the repo root on the cluster::

  python run_final_pipeline.py --model llama_3_8b_instruct
  python run_final_pipeline.py --model llama_3_8b_instruct --lr 1e-5

``--lr`` must be ``2e-5`` or ``1e-5``: passed to ``submit_wandb_sweep --learning-rates`` and
``LR`` for ``submit_test_eval_matrix.sh``.

Forward extra arguments to ``train/submit_wandb_sweep.py`` (all **four** training passes) after ``--``::

  python run_final_pipeline.py --model mistral_7b_instruct -- --dry-run --limit 2

Arguments after ``--`` override launcher defaults such as ``--total-epochs``, ``--time``, or
``--embed-sweep-eval`` (see ``train/submit_wandb_sweep.py --help``).

Launcher-only flags (before ``--``)::

  --model NAME             Base LLM preset (default: llama_2_7b_chat); see train/model_profiles.py.

  --lr RATE                ``2e-5`` or ``1e-5`` for all training sweeps + eval array (see above).

  --skip-training          Skip all ``submit_wandb_sweep`` calls (only sbatch eval array).
  --skip-held-out-training Submit seen-family sweeps only (no held-out training jobs).
  --skip-eval              Submit training only (no ``submit_test_eval_matrix.sh``).
  --skip-heatmaps          Do not submit ``eval/submit_plot_heatmaps.sh`` after eval.
  --parallel-eval          Submit eval immediately; do not wait for training jobs to finish.

  --training-half {first,second,full}
                           Default ``first``: first shuffled half → ``*_epoch1``. ``second``: resume
                           ``*_epoch1``, second half → ``*_epoch2``. ``full``: legacy one pass on all rows.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from train.model_profiles import DEFAULT_MODEL_PROFILE, MODEL_PROFILE_CHOICES  # noqa: E402

HELD_OUT_FAMS = "gcg,autodan,pair"

# Single SLURM script per (lr,λ,ε) cell; train-only; default **first half** of one data
# pass (``--total-epochs 1`` + ``--training-halves-phase first`` → ``*_epoch1``), then run
# ``python run_final_pipeline.py --training-half second`` for the second half → ``*_epoch2``.
# ``submit_test_eval_matrix.sh`` reads ``EPOCH``; we set ``EPOCH`` from ``--training-half``
# and any ``--total-epochs`` override after ``--``.
_PIPELINE_TRAIN_EPOCHS = "1"
# Budget: 4 h per full pass over all training rows; each 0.5-epoch half-job gets 2 h.
_TRAIN_TIME_PER_FULL_EPOCH = "4:00:00"
_TRAIN_TIME_PER_HALF_EPOCH = "2:00:00"


def _default_submit_train_args(training_half: str) -> list[str]:
    wall = (
        _TRAIN_TIME_PER_HALF_EPOCH
        if training_half in ("first", "second")
        else _TRAIN_TIME_PER_FULL_EPOCH
    )
    base = [
        "--total-epochs",
        _PIPELINE_TRAIN_EPOCHS,
        "--skip-embedded-eval",
        "--time",
        wall,
    ]
    if training_half == "first":
        return base[:2] + ["--training-halves-phase", "first"] + base[2:]
    if training_half == "second":
        return base[:2] + ["--training-halves-phase", "second"] + base[2:]
    return base


def _epoch_env_for_matrix(training_half: str, forward: list[str]) -> str:
    """``test_eval_matrix`` checkpoint suffix ``_epoch{N}``.

    After the default first-half training, evaluate ``_epoch1``. After ``--training-half
    second``, evaluate ``_epoch2``. Otherwise ``N`` comes from the effective ``--total-epochs``
    in merged defaults + ``forward`` (after ``--``).
    """
    if training_half == "first":
        return "1"
    if training_half == "second":
        return "2"
    merged = list(_default_submit_train_args("full")) + list(forward)
    last: str | None = None
    i = 0
    while i < len(merged):
        t = merged[i]
        if t == "--total-epochs" and i + 1 < len(merged):
            last = merged[i + 1]
            i += 2
            continue
        if t.startswith("--total-epochs="):
            last = t.split("=", 1)[1].strip()
            i += 1
            continue
        i += 1
    raw = last if last is not None else _PIPELINE_TRAIN_EPOCHS
    try:
        n = int(raw)
    except ValueError:
        print(
            f"[WARN] Invalid --total-epochs {raw!r}; using EPOCH={_PIPELINE_TRAIN_EPOCHS}",
            file=sys.stderr,
            flush=True,
        )
        return _PIPELINE_TRAIN_EPOCHS
    if n < 1:
        return _PIPELINE_TRAIN_EPOCHS
    return str(n)


def _read_job_ids(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [
        ln.strip()
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]


def _submit_sbatch(
    repo: Path,
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
) -> str | None:
    """Run sbatch from repo cwd; print stdout/stderr; return job id if parsed."""
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    proc = subprocess.run(cmd, cwd=str(repo), text=True, capture_output=True, env=run_env)
    if proc.stdout:
        print(proc.stdout, end="", flush=True)
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr, flush=True)
    if proc.returncode != 0:
        print(
            f"[ERROR] sbatch failed (exit {proc.returncode}): {' '.join(cmd)}",
            file=sys.stderr,
            flush=True,
        )
        return None
    combined = (proc.stdout or "") + (proc.stderr or "")
    m = re.search(r"Submitted batch job (\d+)", combined)
    return m.group(1) if m else None


def _submit_eval_array(
    *,
    repo: Path,
    eval_sh: Path,
    dependency_job_ids: list[str] | None,
    model_profile: str,
    lr: str | None = None,
    matrix_epoch: str | None = None,
) -> str | None:
    """Submit eval SLURM script; optionally wait until all listed training jobs finish."""
    cmd: list[str] = ["sbatch"]
    if dependency_job_ids:
        cmd.append(f"--dependency=after:{':'.join(dependency_job_ids)}")
    cmd.append(str(eval_sh))
    print(" ", " ".join(cmd), flush=True)
    run_env = os.environ.copy()
    run_env["REPO_ROOT"] = str(repo.resolve())
    run_env["MODEL_PROFILE"] = model_profile
    run_env["LR"] = lr if lr is not None else "2e-5"
    run_env["EPOCH"] = matrix_epoch if matrix_epoch is not None else _PIPELINE_TRAIN_EPOCHS
    print(f"  (eval matrix EPOCH={run_env['EPOCH']} → *_finetuned_llm_epoch{run_env['EPOCH']})", flush=True)
    return _submit_sbatch(repo, cmd, env=run_env)


def _submit_heatmap_job(*, repo: Path, heatmap_sh: Path, after_job_id: str) -> None:
    """Chain heatmap CPU job after the eval array.

    Try ``afterok`` first (typical on Compute Canada / job arrays: run after all tasks
    exit 0). Some Slurm builds reject ``after:`` or behave oddly with arrays; fall back
    to ``afterany`` then ``after``.

    Ensures ``METRICS_DIR`` / ``CHECKPOINT_ROOT`` match ``test_eval_matrix`` defaults
    (``$CHECKPOINT_ROOT/test_eval_outputs``) when not already set in the parent environment,
    so ``by_dataset/`` heatmaps resolve CSV paths next to ``*_metrics.tsv``.

    Output folder name is chosen inside ``plot_hyperparameter_heatmaps.py`` from the metrics
    TSVs (no ``MODEL_PROFILE`` / ``LR`` exports required). Set ``HEATMAP_OUT_DIR`` only to
    force a fixed output root.
    """
    scr = os.environ.get("SCRATCH", "")
    heatmap_env: dict[str, str] = {"REPO_ROOT": str(repo.resolve())}
    if "CHECKPOINT_ROOT" not in os.environ and scr:
        heatmap_env["CHECKPOINT_ROOT"] = f"{scr}/dp-llm-sweep"
    ck = os.environ.get("CHECKPOINT_ROOT", heatmap_env.get("CHECKPOINT_ROOT", ""))
    if "METRICS_DIR" not in os.environ and ck:
        heatmap_env["METRICS_DIR"] = str(Path(ck) / "test_eval_outputs")
    dep_styles = (
        f"afterok:{after_job_id}",
        f"afterany:{after_job_id}",
        f"after:{after_job_id}",
    )
    for dep in dep_styles:
        cmd = ["sbatch", f"--dependency={dep}", str(heatmap_sh)]
        print(" ", " ".join(cmd), flush=True)
        jid = _submit_sbatch(repo, cmd, env=heatmap_env)
        if jid is not None:
            return
        print(f"[WARN] Heatmap sbatch with --dependency={dep} failed; trying next...", flush=True)
    print(
        "[WARN] Could not submit heatmap job (dependency or sbatch error). "
        "After eval finishes, run manually:\n"
        f"  sbatch {heatmap_sh}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" in argv:
        idx = argv.index("--")
        launcher_args = argv[:idx]
        forward = argv[idx + 1 :]
    else:
        launcher_args = argv
        forward = []

    lp = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    lp.add_argument(
        "--skip-training",
        action="store_true",
        help="Only submit the test-eval SLURM array (skip submit_wandb_sweep).",
    )
    lp.add_argument(
        "--skip-held-out-training",
        action="store_true",
        help="Only submit seen-family sweeps (clean + perturbed@λ=0); skip held-out training.",
    )
    lp.add_argument(
        "--skip-eval",
        action="store_true",
        help="Only run training sweep submissions (skip sbatch test matrix).",
    )
    lp.add_argument(
        "--skip-heatmaps",
        action="store_true",
        help="Do not sbatch eval/submit_plot_heatmaps.sh after the eval array.",
    )
    lp.add_argument(
        "--parallel-eval",
        action="store_true",
        help=(
            "Submit the eval array right away (no SLURM dependency on training). "
            "Default is to chain eval after all training jobs complete."
        ),
    )
    lp.add_argument(
        "--training-half",
        choices=("first", "second", "full"),
        default="first",
        help=(
            "Training data slice for each sweep job: default ``first`` runs train.py on the "
            "first half of the shuffled rows (checkpoint *_epoch1). ``second`` resumes *_epoch1 "
            "and trains the second half (*_epoch2). ``full`` uses the whole CSV in one pass "
            "(``--training-halves-phase full``; legacy behavior)."
        ),
    )
    lp.add_argument(
        "--repo-root",
        default="",
        help="Repo path on the cluster (default: directory containing this script).",
    )
    lp.add_argument(
        "--model",
        dest="model_profile",
        default=DEFAULT_MODEL_PROFILE,
        choices=list(MODEL_PROFILE_CHOICES),
        help="Which base LLM + hinge/eval preset (train/model_profiles.py). Propagates to training, eval array, and MODEL_PROFILE.",
    )
    lp.add_argument(
        "--lr",
        dest="learning_rate",
        default=None,
        metavar="RATE",
        choices=("2e-5", "1e-5"),
        help=(
            "Single learning rate for all submit_wandb_sweep passes (--learning-rates) "
            "and for the eval SLURM array (LR env → test_eval_matrix --lr). "
            "Omit to keep submit_wandb_sweep default (2e-5) and eval LR default (2e-5)."
        ),
    )
    args = lp.parse_args(launcher_args)

    repo = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parent
    submit_py = repo / "train" / "submit_wandb_sweep.py"
    eval_sh = repo / "eval" / "submit_test_eval_matrix.sh"
    sweep_root = repo / "sweep_jobs"
    sweep_root.mkdir(parents=True, exist_ok=True)
    job_ids_path = sweep_root / "training_job_ids.txt"

    heatmap_sh = repo / "eval" / "submit_plot_heatmaps.sh"

    if not submit_py.is_file():
        print(f"ERROR: missing {submit_py}", file=sys.stderr)
        return 2
    if not eval_sh.is_file():
        print(f"ERROR: missing {eval_sh}", file=sys.stderr)
        return 2
    if not args.skip_eval and not args.skip_heatmaps and not heatmap_sh.is_file():
        print(f"ERROR: missing {heatmap_sh} (use --skip-heatmaps if you do not want plots)", file=sys.stderr)
        return 2

    seen_sweeps = [
        ("clean", "seen-family", sweep_root / "lm_clean_seen", ()),
        (
            "perturbed",
            "seen-family",
            sweep_root / "lm_perturbed_seen",
            ("--perturbed-sweep-subset", "lambda0_only"),
        ),
    ]
    held_sweeps = [
        ("clean", "held-out", sweep_root / "lm_clean_heldout", ()),
        (
            "perturbed",
            "held-out",
            sweep_root / "lm_perturbed_heldout",
            ("--perturbed-sweep-subset", "lambda0_only"),
        ),
    ]

    chain_eval = not args.parallel_eval and not args.skip_eval

    lr_train_args: list[str] = (
        ["--learning-rates", args.learning_rate] if args.learning_rate is not None else []
    )
    train_submit_args = _default_submit_train_args(args.training_half)

    if not args.skip_training:
        job_ids_path.write_text("", encoding="utf-8")
        for lm, desc, script_dir, sweep_extra in seen_sweeps:
            cmd = (
                [
                    sys.executable,
                    str(submit_py),
                    "--model-profile",
                    args.model_profile,
                    "--lm-loss-input",
                    lm,
                    "--script-dir",
                    str(script_dir),
                ]
                + list(sweep_extra)
                + lr_train_args
                + train_submit_args
                + forward
            )
            if chain_eval:
                cmd.extend(["--record-job-ids", str(job_ids_path)])
            print(f"\n=== submit_wandb_sweep ({desc}, {lm} LM) ===", flush=True)
            print(" ", " ".join(cmd), flush=True)
            subprocess.run(cmd, cwd=str(repo), check=True)

        if not args.skip_held_out_training:
            for lm, desc, script_dir, sweep_extra in held_sweeps:
                cmd = (
                    [
                        sys.executable,
                        str(submit_py),
                        "--model-profile",
                        args.model_profile,
                        "--lm-loss-input",
                        lm,
                        "--held-out-families",
                        HELD_OUT_FAMS,
                        "--script-dir",
                        str(script_dir),
                    ]
                    + list(sweep_extra)
                    + lr_train_args
                    + train_submit_args
                    + forward
                )
                if chain_eval:
                    cmd.extend(["--record-job-ids", str(job_ids_path)])
                print(f"\n=== submit_wandb_sweep ({desc}, {lm} LM; {HELD_OUT_FAMS}) ===", flush=True)
                print(" ", " ".join(cmd), flush=True)
                subprocess.run(cmd, cwd=str(repo), check=True)

    if not args.skip_eval:
        train_ids = _read_job_ids(job_ids_path) if chain_eval and not args.skip_training else []
        use_dep = bool(train_ids) and chain_eval and not args.skip_training

        if chain_eval and not args.skip_training and not train_ids:
            print(
                "\n[WARN] No training job ids recorded (e.g. --dry-run on sweeps). "
                "Submitting eval without SLURM dependency.",
                flush=True,
            )

        print("\n=== sbatch test_eval_matrix array (seen + unseen × 3 families) ===", flush=True)
        matrix_epoch = _epoch_env_for_matrix(args.training_half, forward)
        eval_job_id = _submit_eval_array(
            repo=repo,
            eval_sh=eval_sh,
            dependency_job_ids=train_ids if use_dep else None,
            model_profile=args.model_profile,
            lr=args.learning_rate,
            matrix_epoch=matrix_epoch,
        )

        if not args.skip_heatmaps:
            if eval_job_id is None:
                print(
                    "[WARN] Could not parse eval job id; skipping heatmap submission. "
                    "Run manually: sbatch eval/submit_plot_heatmaps.sh",
                    flush=True,
                )
            else:
                print("\n=== sbatch plot hyperparameter heatmaps (after eval array) ===", flush=True)
                _submit_heatmap_job(repo=repo, heatmap_sh=heatmap_sh, after_job_id=eval_job_id)

    if args.training_half == "first" and not args.skip_training:
        lr_hint = args.learning_rate or "2e-5"
        exe = Path(sys.argv[0]).name
        extra = " --skip-held-out-training" if args.skip_held_out_training else ""
        print(
            "\n=== After *_epoch1 checkpoints exist: second half (same model / LR) ===\n"
            f"  python {exe} --model {args.model_profile} --lr {lr_hint} "
            f"--training-half second{extra}\n"
            "  Job scripts pin ``--training-shuffle-seed`` from each run slug so the second "
            "half sees the same shuffled order as the first.\n"
            "  (Append the same ``--`` forward args you used for the first half, if any.)\n",
            flush=True,
        )

    print(
        "\nDone. Eval + metrics TSVs + CSVs: ``$CHECKPOINT_ROOT/test_eval_outputs/`` (default "
        "``$SCRATCH/dp-llm-sweep/test_eval_outputs``). Heatmaps: "
        "``…/test_eval_outputs/heatmaps_<MODEL_PROFILE>_lr<LR>/{aggregate,by_dataset}/`` "
        "(names from metrics TSVs unless ``HEATMAP_OUT_DIR`` is set). Override ``CHECKPOINT_ROOT`` / "
        "``METRICS_DIR`` / ``HEATMAP_OUT_DIR`` when sbatching ``submit_plot_heatmaps.sh`` manually.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
