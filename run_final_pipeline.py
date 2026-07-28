#!/usr/bin/env python3
"""
Single entry point for the **final** experiment on Narval.

Modes
-----
``--mode sweep`` (default)
  Hyperparameter search on the validation split:

  1. Submit **seen-family** training over the λ×ε grid (via
     ``train/submit_wandb_sweep.LAMBDAS`` / ``EPSILONS``).
  2. Submit **held-out** training over the same grid (three families × grid).
  3. Submit the SLURM array for ``test_eval_matrix.py`` on the **validation** set.
  4. Submit CPU heatmaps + hyperparameter selection under
     ``…/val_eval_outputs/lr<LR>/selection/``.

``--mode test``
  Final models + benchmark tables (no hyperparam heatmaps):

  1. Train **four** checkpoints on ``llama3_train_plus_validation.csv``:
       • seen-family          λ=1, ε=-0.5
       • held-out autodan     λ=3, ε=0
       • held-out gcg         λ=3, ε=1
       • held-out pair        λ=1, ε=-1
  2. Evaluate on ``llama3_test.csv`` (+ FRR on ``frr_test.csv`` / ``frr_text.csv``)
     with the mapping above (seen attacks all use the λ=1/ε=-0.5 model).
  3. Write SEEN-FAMILY and HELDOUT-FAMILY tables
     (columns: advbench, harmbench, jailbreakbench, mean; rows: all / gcg /
     autodan / pair; cells: ASR / FRR) under
     ``$CHECKPOINT_ROOT/final_test_outputs/``.

**clean/perturbed is purely λ-based** (all runs use ``--lm-loss-input clean``):
**λ=0** = regularizer OFF; **λ>0** = regularizer ON.

Run from the repo root on the cluster::

  python run_final_pipeline.py --mode sweep --model llama_3_8b_instruct
  python run_final_pipeline.py --mode test  --model llama_3_8b_instruct

``--lr`` is a comma-separated subset of ``{2e-5, 1e-5}``. Omit to use ``2e-5``.

Forward extra arguments to ``train/submit_wandb_sweep.py`` after ``--``::

  python run_final_pipeline.py --mode test -- --dry-run
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

from train.model_profiles import MODEL_PROFILE_CHOICES  # noqa: E402
from train.submit_wandb_sweep import (  # noqa: E402
    EPSILONS as SWEEP_EPSILONS,
    LAMBDAS as SWEEP_LAMBDAS,
    lambda_epsilon_pairs,
)

HELD_OUT_FAMS = "gcg,autodan,pair"

_ALLOWED_LRS = ("2e-5", "1e-5")
_DEFAULT_SWEEP_LRS = ("2e-5",)

_PIPELINE_TRAIN_EPOCHS = "1"
_TRAIN_WALL_TIME = "3:00:00"
_TRAIN_WALL_TIME_BOTH = "9:00:00"
_EVAL_WALL_TIME = "4:30:00"
_FINAL_TEST_EVAL_WALL_TIME = "6:00:00"

# Sweep-mode (validation) defaults
_L3_TRAINING_DATA = "$SCRATCH/dp-llm-experiments/official_data/llama3_train.csv"
_L3_VALIDATION_DATA = "$SCRATCH/dp-llm-experiments/official_data/llama3_validation.csv"
_L3_FRR_VALIDATION_DATA = "$SCRATCH/dp-llm-experiments/official_data/frr_validation.csv"

# Test-mode defaults
_L3_TRAIN_PLUS_VAL = (
    "$SCRATCH/dp-llm-experiments/official_data/llama3_train_plus_validation.csv"
)
_L3_TEST_DATA = "$SCRATCH/dp-llm-experiments/official_data/llama3_test.csv"
_L3_FRR_TEST_DATA = "$SCRATCH/dp-llm-experiments/official_data/frr_test.csv"
_L3_DATASET_LABELS = (
    "$SCRATCH/dp-llm-experiments/official_data/combined_test_dataset.csv"
)

# Final test-mode train jobs: (description, held_out_family|None, lam, eps)
_TEST_TRAIN_JOBS: list[tuple[str, str | None, float, float]] = [
    ("seen-family λ=1 ε=-0.5", None, 1.0, -0.5),
    ("held-out autodan λ=3 ε=0", "autodan", 3.0, 0.0),
    ("held-out gcg λ=3 ε=1", "gcg", 3.0, 1.0),
    ("held-out pair λ=1 ε=-1", "pair", 1.0, -1.0),
]


def _parse_lr_list(raw: str | None) -> list[str] | None:
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


def _eval_grid_str() -> tuple[str, str]:
    lam = ",".join(f"{x:g}" for x in SWEEP_LAMBDAS)
    eps = ",".join(f"{x:g}" for x in SWEEP_EPSILONS)
    return lam, eps


def _eval_task_count() -> int:
    return len(lambda_epsilon_pairs(SWEEP_LAMBDAS, SWEEP_EPSILONS))


def _default_submit_train_args(training_half: str) -> list[str]:
    wall = _TRAIN_WALL_TIME_BOTH if training_half == "both" else _TRAIN_WALL_TIME
    args = ["--total-epochs", _PIPELINE_TRAIN_EPOCHS]
    if training_half in ("first", "second", "both"):
        args += ["--training-halves-phase", training_half]
    args += ["--skip-embedded-eval", "--time", wall]
    return args


def _epoch_env_for_matrix(training_half: str, forward: list[str]) -> str:
    if training_half == "first":
        return "1"
    if training_half in ("second", "both"):
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
    wall_time: str | None = None,
) -> str | None:
    cmd: list[str] = ["sbatch", f"--time={wall_time or _EVAL_WALL_TIME}"]
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
    print(
        f"  (eval matrix EPOCH={run_env['EPOCH']} → *_finetuned_llm_epoch{run_env['EPOCH']})",
        flush=True,
    )
    return _submit_sbatch(repo, cmd, env=run_env)


def _submit_final_test_eval(
    *,
    repo: Path,
    eval_sh: Path,
    dependency_job_ids: list[str] | None,
    model_profile: str,
    lr: str,
    matrix_epoch: str,
    harmful_data: str,
    benign_data: str,
    labels_csv: str,
    out_dir: str,
) -> str | None:
    cmd: list[str] = ["sbatch", f"--time={_FINAL_TEST_EVAL_WALL_TIME}"]
    if dependency_job_ids:
        cmd.append(f"--dependency=after:{':'.join(dependency_job_ids)}")
    cmd.append(str(eval_sh))
    print(" ", " ".join(cmd), flush=True)
    run_env = os.environ.copy()
    run_env["REPO_ROOT"] = str(repo.resolve())
    run_env["MODEL_PROFILE"] = model_profile
    run_env["LR"] = lr
    run_env["EPOCH"] = matrix_epoch
    run_env["HARMFUL_TEST"] = harmful_data
    run_env["BENIGN_TEST"] = benign_data
    run_env["LABELS_CSV"] = labels_csv
    run_env["OUT_DIR"] = out_dir
    run_env["SYSTEM_PROMPT_MODE"] = "empty"
    run_env["BENIGN_SYSTEM_PROMPT_MODE"] = "empty"
    print(
        f"  (final test eval EPOCH={matrix_epoch}; out={out_dir})",
        flush=True,
    )
    return _submit_sbatch(repo, cmd, env=run_env)


def _submit_heatmap_job(
    *,
    repo: Path,
    heatmap_sh: Path,
    after_job_id: str,
    metrics_dir: str | None = None,
    labels_csv: str | None = None,
) -> None:
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


def _run_test_mode(
    *,
    args: argparse.Namespace,
    repo: Path,
    submit_py: Path,
    eval_sh: Path,
    sweep_root: Path,
    job_ids_path: Path,
    forward: list[str],
) -> int:
    """Train the four mapped checkpoints and run final test eval + tables."""
    chain_eval = not args.parallel_eval and not args.skip_eval
    lr_list = _parse_lr_list(args.learning_rate)
    lr_train_args: list[str] = (
        ["--learning-rates", ",".join(lr_list)] if lr_list is not None else []
    )
    eval_lrs = lr_list if lr_list is not None else list(_DEFAULT_SWEEP_LRS)
    train_submit_args = _default_submit_train_args(args.training_half)
    data_args = [
        "--training-data",
        args.training_data,
        # Embedded eval is skipped; still pass a harmless placeholder so the
        # sweep CLI stays happy if someone re-enables embedded eval.
        "--validation-data",
        args.test_data,
    ]

    if not args.skip_training:
        job_ids_path.write_text("", encoding="utf-8")
        for desc, fam, lam, eps in _TEST_TRAIN_JOBS:
            pair = f"{lam:g}:{eps:g}"
            script_dir = (
                sweep_root / "test_seen"
                if fam is None
                else sweep_root / f"test_heldout_{fam}"
            )
            cmd = [
                sys.executable,
                str(submit_py),
                "--model-profile",
                args.model_profile,
                "--lm-loss-input",
                "clean",
                "--script-dir",
                str(script_dir),
                "--lambda-epsilon-pairs",
                pair,
            ]
            if fam is not None:
                cmd += ["--held-out-families", fam]
            cmd += lr_train_args + data_args + train_submit_args + forward
            if chain_eval:
                cmd.extend(["--record-job-ids", str(job_ids_path)])
            print(f"\n=== submit_wandb_sweep (TEST: {desc}) ===", flush=True)
            print(" ", " ".join(cmd), flush=True)
            subprocess.run(cmd, cwd=str(repo), check=True)

    if not args.skip_eval:
        train_ids = _read_job_ids(job_ids_path) if chain_eval and not args.skip_training else []
        use_dep = bool(train_ids) and chain_eval and not args.skip_training
        if chain_eval and not args.skip_training and not train_ids:
            print(
                "\n[WARN] No training job ids recorded (e.g. --dry-run on sweeps). "
                "Submitting final test eval without SLURM dependency.",
                flush=True,
            )

        matrix_epoch = _epoch_env_for_matrix(args.training_half, forward)
        base_ck = os.environ.get("CHECKPOINT_ROOT") or "$SCRATCH/dp-llm-sweep"
        for lr in eval_lrs:
            out_dir = f"{base_ck}/final_test_outputs/lr{lr}"
            print(
                f"\n=== sbatch FINAL TEST eval (lr={lr}; mapped (λ,ε) per family; "
                f"SEEN/HELDOUT tables) ===",
                flush=True,
            )
            _submit_final_test_eval(
                repo=repo,
                eval_sh=eval_sh,
                dependency_job_ids=train_ids if use_dep else None,
                model_profile=args.model_profile,
                lr=lr,
                matrix_epoch=matrix_epoch,
                harmful_data=args.test_data,
                benign_data=args.benign_test_data,
                labels_csv=args.dataset_labels,
                out_dir=out_dir,
            )

    print(
        "\nDone (TEST mode). Final tables land in "
        "``$CHECKPOINT_ROOT/final_test_outputs/lr<LR>/`` "
        "(seen_family_table.md / heldout_family_table.md / final_test_report.md). "
        "No hyperparameter heatmaps are submitted in test mode.\n"
        "Model mapping: seen → λ=1,ε=-0.5; heldout autodan → λ=3,ε=0; "
        "heldout gcg → λ=3,ε=1; heldout pair → λ=1,ε=-1.\n",
        flush=True,
    )
    return 0


def _run_sweep_mode(
    *,
    args: argparse.Namespace,
    repo: Path,
    submit_py: Path,
    eval_sh: Path,
    heatmap_sh: Path,
    sweep_root: Path,
    job_ids_path: Path,
    forward: list[str],
) -> int:
    seen_sweeps = [
        ("clean", "seen-family", sweep_root / "lm_seen", ()),
    ]
    held_sweeps = [
        ("clean", "held-out", sweep_root / "lm_heldout", ()),
    ]

    chain_eval = not args.parallel_eval and not args.skip_eval
    lr_list = _parse_lr_list(args.learning_rate)
    lr_train_args: list[str] = (
        ["--learning-rates", ",".join(lr_list)] if lr_list is not None else []
    )
    eval_lrs = lr_list if lr_list is not None else list(_DEFAULT_SWEEP_LRS)
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
                print(
                    f"\n=== submit_wandb_sweep ({desc}, {lm} LM; {HELD_OUT_FAMS}) ===",
                    flush=True,
                )
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
            f"  python {exe} --mode sweep --model {args.model_profile} --lr {lr_hint} "
            f"--training-half second{extra}\n",
            flush=True,
        )

    print(
        "\nDone (SWEEP mode). VALIDATION metrics: "
        "``$CHECKPOINT_ROOT/val_eval_outputs/lr<LR>/``; selection under "
        "``…/selection/``. Then run ``--mode test`` for final numbers.\n",
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" in argv:
        idx = argv.index("--")
        launcher_args = argv[:idx]
        forward = argv[idx + 1 :]
    else:
        launcher_args = argv
        forward = []

    lp = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    lp.add_argument(
        "--mode",
        choices=("sweep", "test"),
        default="sweep",
        help=(
            "sweep = hyperparameter search on validation (heatmaps + selection). "
            "test = train the 4 selected (λ,ε) models on train+val and evaluate on "
            "the test set with SEEN/HELDOUT tables (no heatmaps)."
        ),
    )
    lp.add_argument(
        "--skip-training",
        action="store_true",
        help="Only submit eval (skip submit_wandb_sweep).",
    )
    lp.add_argument(
        "--skip-held-out-training",
        action="store_true",
        help="SWEEP mode only: seen-family sweep only (no held-out training).",
    )
    lp.add_argument(
        "--skip-eval",
        action="store_true",
        help="Only run training sweep submissions (skip eval sbatch).",
    )
    lp.add_argument(
        "--skip-heatmaps",
        action="store_true",
        help="SWEEP mode only: do not sbatch heatmaps after the eval array.",
    )
    lp.add_argument(
        "--parallel-eval",
        action="store_true",
        help=(
            "Submit eval immediately (no SLURM dependency on training). "
            "Default is to chain eval after all training jobs complete."
        ),
    )
    lp.add_argument(
        "--training-half",
        choices=("both", "first", "second", "full"),
        default="both",
        help=(
            "Training data slice for each sweep job. Default ``both``: first half → "
            "*_epoch1 then second half → *_epoch2 (evaluated)."
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
        help="Base LLM + hinge/eval preset (train/model_profiles.py).",
    )
    lp.add_argument(
        "--training-data",
        default=None,
        help=(
            "Training CSV for submit_wandb_sweep. Default depends on --mode: "
            "sweep → llama3_train.csv; test → llama3_train_plus_validation.csv."
        ),
    )
    lp.add_argument(
        "--validation-data",
        default=_L3_VALIDATION_DATA,
        help="SWEEP mode: harmful validation CSV (ASR + heatmap labels).",
    )
    lp.add_argument(
        "--benign-validation-data",
        default=_L3_FRR_VALIDATION_DATA,
        help="SWEEP mode: benign FRR validation CSV.",
    )
    lp.add_argument(
        "--test-data",
        default=_L3_TEST_DATA,
        help="TEST mode: harmful test CSV (ASR). Default: llama3_test.csv.",
    )
    lp.add_argument(
        "--benign-test-data",
        default=_L3_FRR_TEST_DATA,
        help=(
            "TEST mode: benign FRR CSV. Default: frr_test.csv "
            "(use frr_text.csv if that is your local name)."
        ),
    )
    lp.add_argument(
        "--dataset-labels",
        default=_L3_DATASET_LABELS,
        help=(
            "TEST mode: CSV with goal+dataset for advbench/harmbench/jailbreakbench "
            "splits (default: combined_test_dataset.csv). Used when the test CSV "
            "lacks a dataset column."
        ),
    )
    lp.add_argument(
        "--lr",
        dest="learning_rate",
        default=None,
        metavar="RATE[,RATE...]",
        help="Comma-separated learning rate(s) from {2e-5, 1e-5}. Default: 2e-5.",
    )
    args = lp.parse_args(launcher_args)

    # Mode-specific training-data default.
    if args.training_data is None:
        args.training_data = (
            _L3_TRAIN_PLUS_VAL if args.mode == "test" else _L3_TRAINING_DATA
        )

    repo = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parent
    submit_py = repo / "train" / "submit_wandb_sweep.py"
    sweep_root = repo / "sweep_jobs"
    sweep_root.mkdir(parents=True, exist_ok=True)
    job_ids_path = sweep_root / "training_job_ids.txt"
    heatmap_sh = repo / "eval" / "submit_plot_heatmaps.sh"

    if not submit_py.is_file():
        print(f"ERROR: missing {submit_py}", file=sys.stderr)
        return 2

    if args.mode == "test":
        eval_sh = repo / "eval" / "submit_final_test_eval.sh"
        if not eval_sh.is_file():
            print(f"ERROR: missing {eval_sh}", file=sys.stderr)
            return 2
        return _run_test_mode(
            args=args,
            repo=repo,
            submit_py=submit_py,
            eval_sh=eval_sh,
            sweep_root=sweep_root,
            job_ids_path=job_ids_path,
            forward=forward,
        )

    eval_sh = repo / "eval" / "submit_test_eval_matrix.sh"
    if not eval_sh.is_file():
        print(f"ERROR: missing {eval_sh}", file=sys.stderr)
        return 2
    if not args.skip_eval and not args.skip_heatmaps and not heatmap_sh.is_file():
        print(
            f"ERROR: missing {heatmap_sh} (use --skip-heatmaps if you do not want plots)",
            file=sys.stderr,
        )
        return 2

    return _run_sweep_mode(
        args=args,
        repo=repo,
        submit_py=submit_py,
        eval_sh=eval_sh,
        heatmap_sh=heatmap_sh,
        sweep_root=sweep_root,
        job_ids_path=job_ids_path,
        forward=forward,
    )


if __name__ == "__main__":
    sys.exit(main())
