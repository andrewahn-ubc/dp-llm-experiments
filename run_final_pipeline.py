#!/usr/bin/env python3
"""
Single entry point for the **final** experiment on Narval:

  1. Submit **seen-family** training with **clean** LM (λ×ε grid via
     ``train/submit_wandb_sweep.LAMBDAS`` / ``EPSILONS`` and ``lambda_epsilon_pairs``). Each cell is **one** SLURM job that runs **only**
     ``train.py`` once by default on the **first half** of the shuffled CSV (``--training-halves-phase first``,
     ``--total-epochs 1`` in the sweep sense → checkpoint ``…_finetuned_llm_epoch1``). Use
     ``python run_final_pipeline.py --training-half second`` (same ``--model`` / ``--lr``) for the
     second half → ``…_epoch2``. SLURM wall time defaults to **3 hours** (``--time 3:00:00``) per
     training job (each half-epoch pass or a single full-data pass when using ``--training-half full``).
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
  python run_final_pipeline.py --model llama_3_8b_instruct --lr 2e-5,1e-5

``--lr`` is a comma-separated subset of ``{2e-5, 1e-5}``: passed to
``submit_wandb_sweep --learning-rates`` and used as ``LR`` for
``submit_test_eval_matrix.sh`` (one eval array + heatmap job per LR). Omit to use the
default (``2e-5``).

Forward extra arguments to ``train/submit_wandb_sweep.py`` (all **four** training passes) after ``--``::

  python run_final_pipeline.py --model mistral_7b_instruct -- --dry-run --limit 2

Arguments after ``--`` override launcher defaults such as ``--total-epochs``, ``--time``, or
``--embed-sweep-eval`` (see ``train/submit_wandb_sweep.py --help``).

Launcher-only flags (before ``--``)::

  --model NAME             Base LLM preset (default: llama_2_7b_chat); see train/model_profiles.py.

  --lr RATE[,RATE...]      Comma-separated subset of {2e-5, 1e-5} for all training sweeps +
                           eval array (one eval array + heatmap per LR). Omit to use the
                           default (2e-5).

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
from train.submit_wandb_sweep import (  # noqa: E402
    EPSILONS as SWEEP_EPSILONS,
    LAMBDAS as SWEEP_LAMBDAS,
    lambda_epsilon_pairs,
)

HELD_OUT_FAMS = "gcg,autodan,pair"

# Learning rates accepted by --lr. Default (when --lr is omitted) matches the
# submit_wandb_sweep --learning-rates default so the eval arrays cover every trained LR.
_ALLOWED_LRS = ("2e-5", "1e-5")
_DEFAULT_SWEEP_LRS = ("2e-5",)


def _parse_lr_list(raw: str | None) -> list[str] | None:
    """Parse the --lr value into an ordered, de-duplicated list of allowed rates.

    Returns None when --lr is omitted (keep submit_wandb_sweep / eval defaults).
    """
    if raw is None:
        return None
    parts = [x.strip() for x in raw.split(",") if x.strip()]
    if not parts:
        raise SystemExit("--lr must list at least one learning rate")
    out: list[str] = []
    for p in parts:
        if p not in _ALLOWED_LRS:
            raise SystemExit(
                f"--lr: unknown rate {p!r}; allowed values: {', '.join(_ALLOWED_LRS)}"
            )
        if p not in out:
            out.append(p)
    return out

# Single SLURM script per (lr,λ,ε) cell; train-only; default **first half** of one data
# pass (``--total-epochs 1`` + ``--training-halves-phase first`` → ``*_epoch1``), then run
# ``python run_final_pipeline.py --training-half second`` for the second half → ``*_epoch2``.
# ``submit_test_eval_matrix.sh`` reads ``EPOCH``; we set ``EPOCH`` from ``--training-half``
# and any ``--total-epochs`` override after ``--``.
_PIPELINE_TRAIN_EPOCHS = "1"
_TRAIN_WALL_TIME = "3:00:00"

# Llama 3 data on the compute node ($SCRATCH sync of the repo's official/ dir).
# training-data: llama3 train split; validation-data: llama3 validation split (same
# goal partition as the Llama 2 train/validation split). Override via --training-data /
# --validation-data if needed.
_L3_TRAINING_DATA = "$SCRATCH/dp-llm-experiments/official_data/llama3_train.csv"
_L3_VALIDATION_DATA = "$SCRATCH/dp-llm-experiments/official_data/llama3_validation.csv"
# Benign (FRR) validation set for over-refusal measurement during the validation sweep.
_L3_FRR_VALIDATION_DATA = "$SCRATCH/dp-llm-experiments/official_data/frr_validation.csv"


def _eval_grid_str() -> tuple[str, str]:
    """λ and ε grids (comma-separated) that the eval matrix must match — from the sweep."""
    lam = ",".join(f"{x:g}" for x in SWEEP_LAMBDAS)
    eps = ",".join(f"{x:g}" for x in SWEEP_EPSILONS)
    return lam, eps


def _eval_task_count() -> int:
    """Number of test_eval_matrix tasks for the pipeline's grid.

    Matches eval/test_eval_matrix.build_tasks with --perturbed-reg-subset lambda0_only:
    one clean_reg task per (λ, ε) cell, plus one perturbed λ=0 task.
    """
    return len(lambda_epsilon_pairs(SWEEP_LAMBDAS, SWEEP_EPSILONS)) + 1


def _default_submit_train_args(training_half: str) -> list[str]:
    base = [
        "--total-epochs",
        _PIPELINE_TRAIN_EPOCHS,
        "--skip-embedded-eval",
        "--time",
        _TRAIN_WALL_TIME,
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
    array_range: str | None = None,
    harmful_data: str | None = None,
    benign_data: str | None = None,
    out_dir: str | None = None,
    lambdas: str | None = None,
    epsilons: str | None = None,
    system_prompt_mode: str | None = None,
    benign_system_prompt_mode: str | None = None,
) -> str | None:
    """Submit eval SLURM script; optionally wait until all listed training jobs finish.

    Optional args parameterize eval/submit_test_eval_matrix.sh (via env) so the same
    array can evaluate the validation set (harmful ASR + FRR) over the full λ×ε grid,
    with an independent system-prompt policy for FRR.
    """
    cmd: list[str] = ["sbatch"]
    if dependency_job_ids:
        cmd.append(f"--dependency=after:{':'.join(dependency_job_ids)}")
    if array_range:
        cmd.append(f"--array={array_range}")
    cmd.append(str(eval_sh))
    print(" ", " ".join(cmd), flush=True)
    run_env = os.environ.copy()
    run_env["REPO_ROOT"] = str(repo.resolve())
    run_env["MODEL_PROFILE"] = model_profile
    run_env["LR"] = lr if lr is not None else "2e-5"
    run_env["EPOCH"] = matrix_epoch if matrix_epoch is not None else _PIPELINE_TRAIN_EPOCHS
    if harmful_data is not None:
        run_env["HARMFUL_TEST"] = harmful_data
    if benign_data is not None:
        run_env["BENIGN_TEST"] = benign_data
    if out_dir is not None:
        run_env["OUT_DIR"] = out_dir
    if lambdas is not None:
        run_env["LAMBDAS"] = lambdas
    if epsilons is not None:
        run_env["EPSILONS"] = epsilons
    if system_prompt_mode is not None:
        run_env["SYSTEM_PROMPT_MODE"] = system_prompt_mode
    if benign_system_prompt_mode is not None:
        run_env["BENIGN_SYSTEM_PROMPT_MODE"] = benign_system_prompt_mode
    print(f"  (eval matrix EPOCH={run_env['EPOCH']} → *_finetuned_llm_epoch{run_env['EPOCH']})", flush=True)
    return _submit_sbatch(repo, cmd, env=run_env)


def _submit_heatmap_job(
    *,
    repo: Path,
    heatmap_sh: Path,
    after_job_id: str,
    metrics_dir: str | None = None,
    labels_csv: str | None = None,
) -> None:
    """Chain heatmap CPU job after the eval array.

    Try ``afterok`` first (typical on Compute Canada / job arrays: run after all tasks
    exit 0). Some Slurm builds reject ``after:`` or behave oddly with arrays; fall back
    to ``afterany`` then ``after``.

    ``metrics_dir`` selects which ``*_metrics.tsv`` folder to plot (e.g. the per-lr
    validation output dir); ``labels_csv`` sets the harmful CSV used for the by_dataset
    split (omit / use a goal-only CSV to get aggregate heatmaps only).

    Output folder name is chosen inside ``plot_hyperparameter_heatmaps.py`` from the metrics
    TSVs (no ``MODEL_PROFILE`` / ``LR`` exports required). Set ``HEATMAP_OUT_DIR`` only to
    force a fixed output root.
    """
    scr = os.environ.get("SCRATCH", "")
    heatmap_env: dict[str, str] = {"REPO_ROOT": str(repo.resolve())}
    if "CHECKPOINT_ROOT" not in os.environ and scr:
        heatmap_env["CHECKPOINT_ROOT"] = f"{scr}/dp-llm-sweep"
    ck = os.environ.get("CHECKPOINT_ROOT", heatmap_env.get("CHECKPOINT_ROOT", ""))
    if metrics_dir is not None:
        heatmap_env["METRICS_DIR"] = metrics_dir
    elif "METRICS_DIR" not in os.environ and ck:
        heatmap_env["METRICS_DIR"] = str(Path(ck) / "test_eval_outputs")
    if labels_csv is not None:
        heatmap_env["LABELS_CSV"] = labels_csv
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
        default="llama_3_8b_instruct",
        choices=list(MODEL_PROFILE_CHOICES),
        help="Which base LLM + hinge/eval preset (train/model_profiles.py). Propagates to training, eval array, and MODEL_PROFILE. Default: llama_3_8b_instruct.",
    )
    lp.add_argument(
        "--training-data",
        default=_L3_TRAINING_DATA,
        help=(
            "Training CSV forwarded to submit_wandb_sweep --training-data (all passes). "
            "Default: the Llama 3 train split."
        ),
    )
    lp.add_argument(
        "--validation-data",
        default=_L3_VALIDATION_DATA,
        help=(
            "Harmful validation CSV. Forwarded to submit_wandb_sweep --validation-data and "
            "used as the harmful (ASR) set for the post-training validation eval array + "
            "heatmaps. Default: the Llama 3 validation split."
        ),
    )
    lp.add_argument(
        "--benign-validation-data",
        default=_L3_FRR_VALIDATION_DATA,
        help=(
            "Benign (FRR) validation CSV for the validation eval array. FRR generation "
            "always runs WITHOUT a system prompt. Default: frr_validation.csv."
        ),
    )
    lp.add_argument(
        "--lr",
        dest="learning_rate",
        default=None,
        metavar="RATE[,RATE...]",
        help=(
            "Comma-separated learning rate(s) from {2e-5, 1e-5} for all submit_wandb_sweep "
            "passes (--learning-rates) and for the eval SLURM array (one array + heatmap job "
            "per LR; LR env → test_eval_matrix --lr). Omit to use the submit_wandb_sweep "
            "default (2e-5) for both training and eval."
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

    lr_list = _parse_lr_list(args.learning_rate)
    lr_train_args: list[str] = (
        ["--learning-rates", ",".join(lr_list)] if lr_list is not None else []
    )
    # LRs the eval arrays must cover: the explicit list, else the sweep default (both LRs).
    eval_lrs = lr_list if lr_list is not None else list(_DEFAULT_SWEEP_LRS)
    # Llama 3 train/validation data forwarded to every submit_wandb_sweep pass. Placed
    # before `forward`, so user args after `--` can still override them.
    data_args = [
        "--training-data",
        args.training_data,
        "--validation-data",
        args.validation_data,
    ]
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
                + data_args
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
                    + data_args
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

        matrix_epoch = _epoch_env_for_matrix(args.training_half, forward)
        lam_grid, eps_grid = _eval_grid_str()
        array_range = f"0-{_eval_task_count() - 1}"
        base_ck = os.environ.get("CHECKPOINT_ROOT") or "$SCRATCH/dp-llm-sweep"
        for lr in eval_lrs:
            # Per-lr output dir keeps each lr's λ×ε metrics (and heatmaps) separate.
            val_out_dir = f"{base_ck}/val_eval_outputs/lr{lr}"
            print(
                f"\n=== sbatch VALIDATION eval array (lr={lr}; λ×ε grid, seen + unseen × 3 "
                f"families; FRR without system prompt) ===",
                flush=True,
            )
            eval_job_id = _submit_eval_array(
                repo=repo,
                eval_sh=eval_sh,
                dependency_job_ids=train_ids if use_dep else None,
                model_profile=args.model_profile,
                lr=lr,
                matrix_epoch=matrix_epoch,
                array_range=array_range,
                harmful_data=args.validation_data,
                benign_data=args.benign_validation_data,
                out_dir=val_out_dir,
                lambdas=lam_grid,
                epsilons=eps_grid,
                system_prompt_mode="empty",
                benign_system_prompt_mode="empty",
            )

            if not args.skip_heatmaps:
                if eval_job_id is None:
                    print(
                        f"[WARN] Could not parse eval job id for lr={lr}; skipping heatmap "
                        "submission. Run manually: sbatch eval/submit_plot_heatmaps.sh",
                        flush=True,
                    )
                else:
                    print(
                        f"\n=== sbatch VALIDATION heatmaps (lr={lr}, after eval array) ===",
                        flush=True,
                    )
                    _submit_heatmap_job(
                        repo=repo,
                        heatmap_sh=heatmap_sh,
                        after_job_id=eval_job_id,
                        metrics_dir=val_out_dir,
                        labels_csv=args.validation_data,
                    )

    if args.training_half == "first" and not args.skip_training:
        lr_hint = args.learning_rate or ",".join(_DEFAULT_SWEEP_LRS)
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
        "\nDone. VALIDATION eval metrics TSVs + per-example CSVs: "
        "``$CHECKPOINT_ROOT/val_eval_outputs/lr<LR>/`` (default "
        "``$SCRATCH/dp-llm-sweep/val_eval_outputs/lr<LR>``). Validation heatmaps (aggregate "
        "ASR + FRR): ``…/val_eval_outputs/lr<LR>/heatmaps_<MODEL_PROFILE>_lr<LR>/aggregate/``. "
        "FRR was measured WITHOUT a system prompt. Pick the best (λ, ε, lr) from these, then "
        "run the TEST eval separately (e.g. sbatch eval/submit_test_eval_matrix.sh with the "
        "test CSVs) for the final numbers.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
