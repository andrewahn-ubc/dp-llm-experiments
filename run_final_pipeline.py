#!/usr/bin/env python3
"""
Single entry point for the **final** experiment on Narval:

  1. Submit **seen-family** training sweep with **clean** LM loss
     (``--lm-loss-input clean``): default λ × ε grid × LR, ``train_plus_validation.csv``.

  2. Submit **seen-family** sweep with **perturbed** LM loss
     (``--lm-loss-input perturbed``); slugs get ``_pertlm`` suffix.

  3. Submit **held-out** training for **clean** LM: one job per (λ, ε, held-out family)
     for families ``gcg``, ``autodan``, ``pair`` — checkpoints under
     ``heldout_{family}_{slug}_finetuned_llm_epoch*`` (matches ``eval/test_eval_matrix.py``).

  4. Same **held-out** grid for **perturbed** LM.

  5. Submit the SLURM array for ``test_eval_matrix.py`` (1× seen-family + 3× unseen-family
     eval per cell): **62** tasks (31 λ×ε cells × 2 LM modes; λ=0 uses one ε only).

  6. Submit a short **CPU** job that builds **8 λ×ε heatmaps** (PNG) from
     ``test_eval_outputs/*_metrics.tsv``, chained after the eval array finishes.

By default the eval array is submitted with a SLURM dependency so it starts **after all
training jobs have terminated** (``--dependency=after:<ids>`` — success or failure).
Use ``--parallel-eval`` to submit eval immediately (legacy behavior: overlaps with training;
useful with ``test_eval_matrix.py --skip-missing`` while checkpoints stream in).

Training manifests and ``.sh`` files land in separate subdirs under ``sweep_jobs/`` so
waves do not overwrite each other. Submitted training job ids are appended to
``sweep_jobs/training_job_ids.txt`` when using chained eval (you can inspect or reuse).

Run from the repo root on the cluster::

  python run_final_pipeline.py

Or::

  ./submit_full_pipeline.sh

Forward extra arguments to ``train/submit_wandb_sweep.py`` (all four training sweep passes) after ``--``::

  python run_final_pipeline.py -- --dry-run --limit 2

Launcher-only flags (before ``--``)::

  --skip-training          Skip all ``submit_wandb_sweep`` calls (only sbatch eval array).
  --skip-held-out-training Submit seen-family sweeps only (no held-out training jobs).
  --skip-eval              Submit training only (no ``submit_test_eval_matrix.sh``).
  --skip-heatmaps          Do not submit ``eval/submit_plot_heatmaps.sh`` after eval.
  --parallel-eval          Submit eval immediately; do not wait for training jobs to finish.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

HELD_OUT_FAMS = "gcg,autodan,pair"


def _read_job_ids(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [
        ln.strip()
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]


def _submit_sbatch(repo: Path, cmd: list[str]) -> str | None:
    """Run sbatch from repo cwd; print stdout/stderr; return job id if parsed."""
    proc = subprocess.run(cmd, cwd=str(repo), text=True, capture_output=True, check=True)
    if proc.stdout:
        print(proc.stdout, end="", flush=True)
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr, flush=True)
    combined = (proc.stdout or "") + (proc.stderr or "")
    m = re.search(r"Submitted batch job (\d+)", combined)
    return m.group(1) if m else None


def _submit_eval_array(
    *,
    repo: Path,
    eval_sh: Path,
    dependency_job_ids: list[str] | None,
) -> str | None:
    """Submit eval SLURM script; optionally wait until all listed training jobs finish."""
    cmd: list[str] = ["sbatch"]
    if dependency_job_ids:
        cmd.append(f"--dependency=after:{':'.join(dependency_job_ids)}")
    cmd.append(str(eval_sh))
    print(" ", " ".join(cmd), flush=True)
    return _submit_sbatch(repo, cmd)


def _submit_heatmap_job(*, repo: Path, heatmap_sh: Path, after_job_id: str) -> None:
    cmd = ["sbatch", f"--dependency=after:{after_job_id}", str(heatmap_sh)]
    print(" ", " ".join(cmd), flush=True)
    jid = _submit_sbatch(repo, cmd)
    if jid is None:
        print("[WARN] Could not parse heatmap batch job id from sbatch output.", flush=True)


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
        help="Only submit seen-family sweeps (clean + perturbed); skip held-out training.",
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
        "--repo-root",
        default="",
        help="Repo path on the cluster (default: directory containing this script).",
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
        ("clean", "seen-family", sweep_root / "lm_clean_seen"),
        ("perturbed", "seen-family", sweep_root / "lm_perturbed_seen"),
    ]
    held_sweeps = [
        ("clean", "held-out", sweep_root / "lm_clean_heldout"),
        ("perturbed", "held-out", sweep_root / "lm_perturbed_heldout"),
    ]

    chain_eval = not args.parallel_eval and not args.skip_eval

    if not args.skip_training:
        job_ids_path.write_text("", encoding="utf-8")
        for lm, desc, script_dir in seen_sweeps:
            cmd = [
                sys.executable,
                str(submit_py),
                "--lm-loss-input",
                lm,
                "--script-dir",
                str(script_dir),
            ] + forward
            if chain_eval:
                cmd.extend(["--record-job-ids", str(job_ids_path)])
            print(f"\n=== submit_wandb_sweep ({desc}, {lm} LM) ===", flush=True)
            print(" ", " ".join(cmd), flush=True)
            subprocess.run(cmd, cwd=str(repo), check=True)

        if not args.skip_held_out_training:
            for lm, desc, script_dir in held_sweeps:
                cmd = [
                    sys.executable,
                    str(submit_py),
                    "--lm-loss-input",
                    lm,
                    "--held-out-families",
                    HELD_OUT_FAMS,
                    "--script-dir",
                    str(script_dir),
                ] + forward
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
        eval_job_id = _submit_eval_array(
            repo=repo,
            eval_sh=eval_sh,
            dependency_job_ids=train_ids if use_dep else None,
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

    print(
        "\nDone. Check sweep_jobs/*/ manifests and SLURM queue. "
        "PNG heatmaps default to CHECKPOINT_ROOT/test_eval_outputs/heatmaps "
        "(override METRICS_DIR / HEATMAP_OUT_DIR in eval/submit_plot_heatmaps.sh).",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
