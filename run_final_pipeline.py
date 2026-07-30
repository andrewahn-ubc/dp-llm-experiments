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
  Final test-set hyperparameter sweep on ``llama3_train_plus_validation.csv``
  (no validation heatmaps). For each (λ, ε) in the test grid, trains **seen** +
  **held-out gcg / autodan / pair** (4 modes). Default grid is 6 cells → **24**
  models; four already-trained checkpoints are skipped by default → **20** jobs.

  Optional eval: SLURM array over the 6 (λ, ε) cells (3h each). Every task
  runs seen + held-out gcg/autodan/pair for **that** cell on ``llama3_test.csv``,
  then a CPU job builds **32** ASR–FRR Pareto charts
  (seen/heldout × {all,gcg,autodan,pair} × {advbench,harmbench,jailbreakbench,mean}).
  Use ``--skip-eval`` while the test set is still being updated.

``--mode adv-sft``
  Standard **adversarial SFT** baseline (R2D2-style): ``--lm-loss-input perturbed``
  at **λ=0** (stability regularizer off; ε inert). Trains on
  ``llama3_train_plus_validation.csv``:

    • seen-family adapter
    • held-out gcg / autodan / pair adapters (unless ``--skip-held-out-training``)

  Then evaluates those ``*_pertlm`` checkpoints on ``llama3_test.csv`` /
  ``frr_test.csv`` via ``test_eval_matrix.py --perturbed-reg-subset perturbed_only``.

**clean/perturbed is purely λ-based** for ``sweep`` / ``test`` (``--lm-loss-input clean``):
**λ=0** = regularizer OFF; **λ>0** = regularizer ON. ``adv-sft`` is the separate
perturbed-LM baseline.

Run from the repo root on the cluster::

  python run_final_pipeline.py --mode sweep --model llama_3_8b_instruct
  python run_final_pipeline.py --mode test --model llama_3_8b_instruct --skip-eval
  python run_final_pipeline.py --mode adv-sft --model llama_3_8b_instruct
  python run_final_pipeline.py --mode test --model llama_3_8b_instruct --skip-eval -- --dry-run

``--lr`` is a comma-separated subset of ``{2e-5, 1e-5}``. Omit to use ``2e-5``.

Forward extra arguments to ``train/submit_wandb_sweep.py`` after ``--``.
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
_TEST_TRAIN_WALL_TIME = "4:00:00"
_EVAL_WALL_TIME = "4:30:00"
_ADV_SFT_EVAL_WALL_TIME = "5:30:00"
_FINAL_TEST_EVAL_WALL_TIME = "3:00:00"

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

# Test-mode (λ, ε) grid: each cell × {seen, heldout gcg/autodan/pair} = 24 models.
_TEST_SWEEP_PAIRS: list[tuple[float, float]] = [
    (1.0, -0.5),
    (3.0, 0.0),
    (3.0, 1.0),
    (1.0, -1.0),
    (30.0, 0.0),
    (30.0, 1.0),
]
_FINAL_TEST_EVAL_ARRAY = f"0-{len(_TEST_SWEEP_PAIRS) - 1}"  # one eval task per (λ, ε)
_TEST_TRAIN_FAMILIES: tuple[str | None, ...] = (None, "gcg", "autodan", "pair")
# Already trained in the earlier 4-model final run; skipped unless
# --no-skip-existing-test-jobs is set. Keys: (held_out_family|None, λ, ε).
_TEST_SKIP_TRAIN_JOBS: set[tuple[str | None, float, float]] = {
    (None, 1.0, -0.5),       # seen-family
    ("autodan", 3.0, 0.0),   # held-out autodan
    ("gcg", 3.0, 1.0),       # held-out gcg
    ("pair", 1.0, -1.0),     # held-out pair
}


def _iter_test_train_jobs(
    *,
    skip_existing: bool,
) -> list[tuple[str, str | None, float, float]]:
    """Return (description, held_out_family|None, lam, eps) jobs for test mode."""
    jobs: list[tuple[str, str | None, float, float]] = []
    for lam, eps in _TEST_SWEEP_PAIRS:
        for fam in _TEST_TRAIN_FAMILIES:
            if skip_existing and (fam, lam, eps) in _TEST_SKIP_TRAIN_JOBS:
                continue
            label = "seen-family" if fam is None else f"held-out {fam}"
            jobs.append((f"{label} λ={lam:g} ε={eps:g}", fam, lam, eps))
    return jobs


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


def _default_submit_train_args(
    training_half: str,
    *,
    wall: str | None = None,
) -> list[str]:
    if wall is None:
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
    perturbed_reg_subset: str | None = None,
    extra_args: str | None = None,
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
    if perturbed_reg_subset is not None:
        run_env["PERTURBED_REG_SUBSET"] = perturbed_reg_subset
    if extra_args is not None:
        run_env["EXTRA_ARGS"] = extra_args
    print(
        f"  (eval matrix EPOCH={run_env['EPOCH']} → *_finetuned_llm_epoch{run_env['EPOCH']}"
        + (
            f"; PERTURBED_REG_SUBSET={perturbed_reg_subset}"
            if perturbed_reg_subset is not None
            else ""
        )
        + (f"; EXTRA_ARGS={extra_args}" if extra_args else "")
        + ")",
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
    """Submit the 6-task final-test eval array (one task per (λ, ε))."""
    cmd: list[str] = [
        "sbatch",
        f"--time={_FINAL_TEST_EVAL_WALL_TIME}",
        f"--array={_FINAL_TEST_EVAL_ARRAY}",
    ]
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
        f"  (final test eval array={_FINAL_TEST_EVAL_ARRAY}, "
        f"EPOCH={matrix_epoch}, wall={_FINAL_TEST_EVAL_WALL_TIME}; out={out_dir})",
        flush=True,
    )
    return _submit_sbatch(repo, cmd, env=run_env)


def _submit_final_test_pareto(
    *,
    repo: Path,
    pareto_sh: Path,
    after_job_id: str,
    points_dir: str,
) -> None:
    """Chain the 32-chart Pareto CPU job after the final-test eval array."""
    env = {
        "REPO_ROOT": str(repo.resolve()),
        "POINTS_DIR": points_dir,
        "PARETO_OUT_DIR": f"{points_dir.rstrip('/')}/pareto_charts",
    }
    dep_styles = (
        f"afterok:{after_job_id}",
        f"afterany:{after_job_id}",
        f"after:{after_job_id}",
    )
    for dep in dep_styles:
        cmd = ["sbatch", f"--dependency={dep}", str(pareto_sh)]
        print(" ", " ".join(cmd), flush=True)
        jid = _submit_sbatch(repo, cmd, env=env)
        if jid is not None:
            return
        print(f"[WARN] Pareto sbatch with --dependency={dep} failed; trying next...", flush=True)
    print(
        "[WARN] Could not submit final-test Pareto job. After eval finishes, run:\n"
        f"  POINTS_DIR={points_dir} sbatch {pareto_sh}",
        flush=True,
    )


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
    """Train the test-mode (λ, ε) × {seen, held-out ×3} grid; optional final eval."""
    chain_eval = not args.parallel_eval and not args.skip_eval
    lr_list = _parse_lr_list(args.learning_rate)
    lr_train_args: list[str] = (
        ["--learning-rates", ",".join(lr_list)] if lr_list is not None else []
    )
    eval_lrs = lr_list if lr_list is not None else list(_DEFAULT_SWEEP_LRS)
    # Test-mode wall is fixed at 4h (overrides the longer sweep-mode "both" budget).
    train_submit_args = _default_submit_train_args(
        args.training_half, wall=_TEST_TRAIN_WALL_TIME
    )
    data_args = [
        "--training-data",
        args.training_data,
        # Embedded eval is skipped; placeholder keeps the sweep CLI happy.
        "--validation-data",
        args.test_data,
    ]

    jobs = _iter_test_train_jobs(skip_existing=not args.no_skip_existing_test_jobs)
    n_full = len(_TEST_SWEEP_PAIRS) * len(_TEST_TRAIN_FAMILIES)
    n_skip = n_full - len(jobs)
    print(
        f"\n=== TEST mode training plan: {len(jobs)} jobs "
        f"({n_full} full grid − {n_skip} skipped; wall={_TEST_TRAIN_WALL_TIME}) ===",
        flush=True,
    )
    if not args.no_skip_existing_test_jobs and n_skip:
        for fam, lam, eps in sorted(
            _TEST_SKIP_TRAIN_JOBS, key=lambda t: (t[0] or "", t[1], t[2])
        ):
            label = "seen-family" if fam is None else f"held-out {fam}"
            print(f"  skip (already trained): {label} λ={lam:g} ε={eps:g}", flush=True)
    for desc, fam, lam, eps in jobs:
        print(f"  train: {desc}", flush=True)

    if not args.skip_training:
        job_ids_path.write_text("", encoding="utf-8")
        for desc, fam, lam, eps in jobs:
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
        pareto_sh = repo / "eval" / "submit_plot_final_test_pareto.sh"
        for lr in eval_lrs:
            out_dir = f"{base_ck}/final_test_outputs/lr{lr}"
            print(
                f"\n=== sbatch FINAL TEST eval array (lr={lr}; "
                f"{len(_TEST_SWEEP_PAIRS)} (λ,ε) cells × seen+3 heldout; "
                f"wall={_FINAL_TEST_EVAL_WALL_TIME}) ===",
                flush=True,
            )
            eval_job_id = _submit_final_test_eval(
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
            if eval_job_id is None:
                print(
                    f"[WARN] Could not parse final-test eval job id for lr={lr}; "
                    "skip Pareto submission. After eval finishes, run:\n"
                    f"  POINTS_DIR={out_dir} sbatch eval/submit_plot_final_test_pareto.sh",
                    flush=True,
                )
            elif not pareto_sh.is_file():
                print(f"[WARN] missing {pareto_sh}; skip Pareto submission.", flush=True)
            else:
                print(
                    f"\n=== sbatch FINAL TEST Pareto charts (lr={lr}, after eval array) ===",
                    flush=True,
                )
                _submit_final_test_pareto(
                    repo=repo,
                    pareto_sh=pareto_sh,
                    after_job_id=eval_job_id,
                    points_dir=out_dir,
                )
    else:
        print("\n=== TEST mode: --skip-eval set; not submitting final test eval ===", flush=True)

    print(
        f"\nDone (TEST mode). Training jobs use wall {_TEST_TRAIN_WALL_TIME}; "
        f"final-test eval array uses wall {_FINAL_TEST_EVAL_WALL_TIME} "
        f"(array {_FINAL_TEST_EVAL_ARRAY}). "
        "Per-cell outputs under ``$CHECKPOINT_ROOT/final_test_outputs/lr<LR>/lam*_eps*/``; "
        "32 Pareto PNGs under ``…/pareto_charts/``.\n",
        flush=True,
    )
    return 0


def _run_adv_sft_mode(
    *,
    args: argparse.Namespace,
    repo: Path,
    submit_py: Path,
    eval_sh: Path,
    sweep_root: Path,
    job_ids_path: Path,
    forward: list[str],
) -> int:
    """Train adversarial SFT (perturbed LM, λ=0) then eval on the test set."""
    chain_eval = not args.parallel_eval and not args.skip_eval
    lr_list = _parse_lr_list(args.learning_rate)
    lr_train_args: list[str] = (
        ["--learning-rates", ",".join(lr_list)] if lr_list is not None else []
    )
    eval_lrs = lr_list if lr_list is not None else list(_DEFAULT_SWEEP_LRS)
    train_submit_args = _default_submit_train_args(
        args.training_half, wall=_TEST_TRAIN_WALL_TIME
    )
    data_args = [
        "--training-data",
        args.training_data,
        "--validation-data",
        args.test_data,
    ]

    print(
        "\n=== ADV-SFT mode: adversarial SFT (lm_loss=perturbed, λ=0) "
        f"on train+val; wall={_TEST_TRAIN_WALL_TIME} ===",
        flush=True,
    )
    print("  train: seen-family (perturbed λ=0)", flush=True)
    if not args.skip_held_out_training:
        print(f"  train: held-out {HELD_OUT_FAMS} (perturbed λ=0)", flush=True)
    else:
        print("  skip held-out training (--skip-held-out-training)", flush=True)

    if not args.skip_training:
        job_ids_path.write_text("", encoding="utf-8")
        seen_cmd = [
            sys.executable,
            str(submit_py),
            "--model-profile",
            args.model_profile,
            "--lm-loss-input",
            "perturbed",
            "--perturbed-sweep-subset",
            "lambda0_only",
            "--script-dir",
            str(sweep_root / "test_adv_sft_seen"),
        ] + lr_train_args + data_args + train_submit_args + forward
        if chain_eval:
            seen_cmd.extend(["--record-job-ids", str(job_ids_path)])
        print("\n=== submit_wandb_sweep (ADV-SFT: seen-family) ===", flush=True)
        print(" ", " ".join(seen_cmd), flush=True)
        subprocess.run(seen_cmd, cwd=str(repo), check=True)

        if not args.skip_held_out_training:
            held_cmd = [
                sys.executable,
                str(submit_py),
                "--model-profile",
                args.model_profile,
                "--lm-loss-input",
                "perturbed",
                "--perturbed-sweep-subset",
                "lambda0_only",
                "--held-out-families",
                HELD_OUT_FAMS,
                "--script-dir",
                str(sweep_root / "test_adv_sft_heldout"),
            ] + lr_train_args + data_args + train_submit_args + forward
            if chain_eval:
                held_cmd.extend(["--record-job-ids", str(job_ids_path)])
            print(
                f"\n=== submit_wandb_sweep (ADV-SFT: held-out {HELD_OUT_FAMS}) ===",
                flush=True,
            )
            print(" ", " ".join(held_cmd), flush=True)
            subprocess.run(held_cmd, cwd=str(repo), check=True)

    if not args.skip_eval:
        train_ids = _read_job_ids(job_ids_path) if chain_eval and not args.skip_training else []
        use_dep = bool(train_ids) and chain_eval and not args.skip_training
        if chain_eval and not args.skip_training and not train_ids:
            print(
                "\n[WARN] No training job ids recorded (e.g. --dry-run on sweeps). "
                "Submitting adv-SFT test eval without SLURM dependency.",
                flush=True,
            )

        matrix_epoch = _epoch_env_for_matrix(args.training_half, forward)
        base_ck = os.environ.get("CHECKPOINT_ROOT") or "$SCRATCH/dp-llm-sweep"
        seen_only_args = "--seen-only" if args.skip_held_out_training else None
        for lr in eval_lrs:
            out_dir = f"{base_ck}/adv_sft_test_outputs/lr{lr}"
            print(
                f"\n=== sbatch ADV-SFT test eval (lr={lr}; "
                f"perturbed_only λ=0; wall={_ADV_SFT_EVAL_WALL_TIME}) ===",
                flush=True,
            )
            eval_job_id = _submit_eval_array(
                repo=repo,
                eval_sh=eval_sh,
                dependency_job_ids=train_ids if use_dep else None,
                model_profile=args.model_profile,
                lr=lr,
                matrix_epoch=matrix_epoch,
                array_range="0-0",
                harmful_data=args.test_data,
                benign_data=args.benign_test_data,
                out_dir=out_dir,
                lambdas="0",
                epsilons="0",
                system_prompt_mode="empty",
                benign_system_prompt_mode="empty",
                wall_time=_ADV_SFT_EVAL_WALL_TIME,
                perturbed_reg_subset="perturbed_only",
                extra_args=seen_only_args,
            )
            if eval_job_id is None:
                print(
                    f"[WARN] Could not parse adv-SFT eval job id for lr={lr}.",
                    flush=True,
                )
    else:
        print("\n=== ADV-SFT mode: --skip-eval set; not submitting test eval ===", flush=True)

    print(
        "\nDone (ADV-SFT mode). Checkpoints: "
        "``$CHECKPOINT_ROOT/l3_run_lr…_lam0_eps0_pertlm_finetuned_llm_epoch*`` "
        "(and ``heldout_{gcg,autodan,pair}_…``). "
        "Test metrics: ``$CHECKPOINT_ROOT/adv_sft_test_outputs/lr<LR>/``.\n",
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
        choices=("sweep", "test", "adv-sft"),
        default="sweep",
        help=(
            "sweep = hyperparameter search on validation (heatmaps + selection). "
            "test = train the test (λ,ε) grid × {seen, held-out×3} on train+val "
            "(default skips 4 already-trained cells); optional test-set eval. "
            "adv-sft = adversarial SFT baseline (perturbed LM, λ=0) on train+val, "
            "then test-set eval."
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
        help=(
            "SWEEP / ADV-SFT: seen-family only (no held-out training). "
            "ADV-SFT also passes --seen-only to test eval."
        ),
    )
    lp.add_argument(
        "--skip-eval",
        action="store_true",
        help="Only run training sweep submissions (skip eval sbatch).",
    )
    lp.add_argument(
        "--no-skip-existing-test-jobs",
        action="store_true",
        help=(
            "TEST mode only: do not skip the 4 checkpoints already trained "
            "(seen λ=1/ε=-0.5; held-out autodan λ=3/ε=0; gcg λ=3/ε=1; pair λ=1/ε=-1). "
            "Default skips those 4 → 20 jobs instead of 24."
        ),
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
            "sweep → llama3_train.csv; test/adv-sft → llama3_train_plus_validation.csv."
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
        help="TEST / ADV-SFT mode: harmful test CSV (ASR). Default: llama3_test.csv.",
    )
    lp.add_argument(
        "--benign-test-data",
        default=_L3_FRR_TEST_DATA,
        help=(
            "TEST / ADV-SFT mode: benign FRR CSV. Default: frr_test.csv "
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
            _L3_TRAIN_PLUS_VAL
            if args.mode in ("test", "adv-sft")
            else _L3_TRAINING_DATA
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

    if args.mode == "adv-sft":
        eval_sh = repo / "eval" / "submit_test_eval_matrix.sh"
        if not eval_sh.is_file():
            print(f"ERROR: missing {eval_sh}", file=sys.stderr)
            return 2
        return _run_adv_sft_mode(
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
