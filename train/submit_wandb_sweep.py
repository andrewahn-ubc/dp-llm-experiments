#!/usr/bin/env python3
"""
Submit a grid of SLURM training jobs (Narval / Compute Canada friendly).

Each job runs five sequential train.py invocations (epochs 1–5), matching
submit_all_train.sh, with a unique checkpoint prefix so every hyperparameter
configuration produces its own saved adapter for inference (all retained):

  {FINETUNED_BASE}_epoch1 … {FINETUNED_BASE}_epoch5

Use the latest (or any intermediate) folder for inference (PeftModel).

Weights & Biases is configured for offline mode on air-gapped compute nodes.
After jobs finish, sync from a machine with internet, from the same WANDB_DIR:

  wandb sync path/to/wandb/offline-run-...

Or: wandb sync --sync-all (see https://docs.wandb.ai/guides/track/public-api-guide/#syncing-offline-runs)

Requires: pip install wandb (in the same venv you use for training.)
"""

from __future__ import annotations

import argparse
import itertools
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path

# Follow-up sweep (Cartesian product = 1 * 3 * 3 = 9 jobs)
# Previous sweep:
# LEARNING_RATES = (1e-6, 1e-5, 2e-5)
# LAMBDAS = (0.1, 0.3, 1.0, 3.0, 10.0)
# EPSILONS = (-1.0, -0.5, 0.0, 0.5, 1.0)
LEARNING_RATES = (1e-5,)
LAMBDAS = (10.0, 20.0, 30.0)
EPSILONS = (1.0, 1.5, 2.0)


def make_run_slug(lr: float, lam: float, eps: float) -> str:
    """Stable, filesystem-friendly id that encodes the three hyperparameters."""
    return f"run_lr{lr:g}_lam{lam:g}_eps{eps:g}".replace(" ", "_")


def render_job_script(
    *,
    slug: str,
    run_id: str,
    lr: float,
    lam: float,
    eps: float,
    wandb_project: str,
    wandb_dir: str,
    finetuned_base: str,
    train_py: str,
    training_csv: str,
    validation_csv: str,
    benign_csv: str,
    sbatch_job_name: str,
    account: str,
    gres: str,
    cpus: int,
    mem: str,
    time_limit: str,
    module_line: str,
    venv_activate: str,
    total_epochs: int,
) -> str:
    # W&B run display name (visible after sync)
    run_name = slug

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={sbatch_job_name}",
        f"#SBATCH --account={account}",
        f"#SBATCH --gres={gres}",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --output=output/sweep_{slug}_%j.out",
        "",
        "mkdir -p output",
        "",
        "set -euo pipefail",
        'cd "${SLURM_SUBMIT_DIR:-.}"',
        "",
    ]
    if module_line:
        lines.append(module_line)
    lines += [
        f'source "{venv_activate}"',
        "",
        "# Node-local Hugging Face caches (same pattern as epoch*.sh)",
        "export TRANSFORMERS_CACHE=$SLURM_TMPDIR/hf_cache",
        "export HF_HOME=$SLURM_TMPDIR/hf_home",
        "mkdir -p \"$TRANSFORMERS_CACHE\"",
        "mkdir -p \"$HF_HOME\"",
        "",
        "# Weights & Biases — offline on compute nodes; sync later from login / your laptop",
        "export WANDB_MODE=offline",
        f'export WANDB_PROJECT="{wandb_project}"',
        f'export WANDB_DIR="{wandb_dir}"',
        f"export WANDB_RUN_ID={run_id}",
        f'export WANDB_RUN_NAME="{run_name}"',
        "",
        f'export FINETUNED_BASE="{finetuned_base}"',
        f'export TRAIN_PY="{train_py}"',
        'mkdir -p "$(dirname "$FINETUNED_BASE")"',
        'mkdir -p "$WANDB_DIR"',
        "",
    ]

    for ep in range(1, total_epochs + 1):
        lines.append(f"# Epoch {ep}")
        lines.append('python "$TRAIN_PY" \\')
        lines.append('    --eval-mode "seen-family" \\')
        lines.append('    --finetuned-llm-path "$FINETUNED_BASE" \\')
        if ep == total_epochs:
            lines.append(f'    --training-data "{training_csv}" \\')
            lines.append(f'    --validation-data "{validation_csv}" \\')
            lines.append(f'    --benign-validation-data "{benign_csv}" \\')
            lines.append(
                f'    --harmful-output-file "$SCRATCH/dp-llm-experiments/official_data/{slug}_val_output" \\'
            )
            lines.append(
                f'    --benign-output-file "$SCRATCH/dp-llm-experiments/official_data/{slug}_frr_output" \\'
            )
        else:
            lines.append(f'    --training-data "{training_csv}" \\')
        lines.append(f"    --lr {lr} \\")
        lines.append(f"    --lambda-val {lam} \\")
        lines.append(f"    --epsilon {eps} \\")
        lines.append("    --lora-rank 8 \\")
        lines.append(f"    --total-epochs {total_epochs} \\")
        if ep > 1:
            lines.append(f'    --resume-from "${{FINETUNED_BASE}}_epoch{ep - 1}" \\')
        lines.append(f"    --start-epoch {ep}")
        lines.append("")

    lines += [
        f'echo "Final adapter (latest): ${{FINETUNED_BASE}}_epoch{total_epochs}"',
        f'echo "Intermediate checkpoints kept: ${{FINETUNED_BASE}}_epoch1 ... _epoch{total_epochs}"',
        "",
    ]
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Submit SLURM jobs for a W&B-logged hyperparameter sweep (offline-friendly)."
    )
    p.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT", "dp-llm-safety"),
        help="W&B project name (default: env WANDB_PROJECT or dp-llm-safety).",
    )
    p.add_argument(
        "--wandb-dir",
        default="$SCRATCH/wandb_offline",
        help="Where offline W&B runs are stored (expand $SCRATCH on the cluster).",
    )
    p.add_argument(
        "--checkpoint-root",
        default="$SCRATCH/dp-llm-sweep",
        help="Directory prefix for finetuned checkpoints; each run uses "
        "{checkpoint-root}/{slug}_finetuned_llm.",
    )
    p.add_argument(
        "--train-py",
        default="$SCRATCH/dp-llm-experiments/train/train.py",
        help="Path to train.py on the cluster.",
    )
    p.add_argument(
        "--training-data",
        default="$SCRATCH/dp-llm-experiments/official_data/train.csv",
        help="Training CSV path.",
    )
    p.add_argument(
        "--validation-data",
        default="$SCRATCH/dp-llm-experiments/official_data/validation.csv",
        help="Validation CSV (epoch 3 only; matches epoch3.sh).",
    )
    p.add_argument(
        "--benign-validation-data",
        default="$SCRATCH/dp-llm-experiments/official_data/frr_validation.csv",
        help="Benign validation CSV (epoch 3 only).",
    )
    p.add_argument(
        "--account",
        default="rrg-mijungp",
        help="SLURM --account (see submit_all_train / your allocation).",
    )
    p.add_argument("--gres", default="gpu:1", help="SLURM --gres string.")
    p.add_argument("--cpus-per-task", type=int, default=6, help="SLURM CPUs per task.")
    p.add_argument("--mem", default="40G", help="SLURM memory.")
    p.add_argument(
        "--hours-per-epoch",
        type=int,
        default=2,
        help="Budget per epoch (hours); used to set default --time as hours-per-epoch × total-epochs.",
    )
    p.add_argument(
        "--time",
        dest="time_limit",
        default=None,
        help="SLURM wall time for the entire sweep job (e.g. 15:00:00). "
        "Default: hours-per-epoch × total-epochs (one job runs all epochs back-to-back).",
    )
    p.add_argument(
        "--module-line",
        default="module load StdEnv/2023 python/3.11",
        help="Module load line (empty string to skip).",
    )
    p.add_argument(
        "--venv-activate",
        default="$SCRATCH/venv/nanogcg/bin/activate",
        help="Path to venv activate script.",
    )
    p.add_argument(
        "--script-dir",
        type=Path,
        default=None,
        help="Directory to write generated .sh files (default: ./sweep_jobs under cwd).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Write scripts and print sbatch commands without submitting.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only submit the first N combinations (useful for smoke tests).",
    )
    p.add_argument(
        "--total-epochs",
        type=int,
        default=5,
        help="Number of sequential train.py invocations per job (each saves its own checkpoint).",
    )
    args = p.parse_args(argv)
    if args.time_limit is None:
        total_h = args.hours_per_epoch * args.total_epochs
        args.time_limit = f"{total_h}:00:00"
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    combos = list(itertools.product(LEARNING_RATES, LAMBDAS, EPSILONS))
    if args.limit is not None:
        combos = combos[: args.limit]

    script_dir = args.script_dir or (Path.cwd() / "sweep_jobs")
    script_dir.mkdir(parents=True, exist_ok=True)

    manifest = script_dir / "sweep_manifest.txt"
    submitted: list[str] = []

    with manifest.open("w", encoding="utf-8") as mf:
        mf.write("# slug\tlr\tlambda\tepsilon\twandb_run_id\tfinetuned_base\tjob_script\n")

        for lr, lam, eps in combos:
            slug = make_run_slug(lr, lam, eps)
            run_id = uuid.uuid4().hex
            finetuned_base = f"{args.checkpoint_root.rstrip('/')}/{slug}_finetuned_llm"

            # SLURM job name: max 64 chars on many systems; keep short
            sbatch_job_name = f"sw_{slug}"[:64]

            body = render_job_script(
                slug=slug,
                run_id=run_id,
                lr=lr,
                lam=lam,
                eps=eps,
                wandb_project=args.wandb_project,
                wandb_dir=args.wandb_dir,
                finetuned_base=finetuned_base,
                train_py=args.train_py,
                training_csv=args.training_data,
                validation_csv=args.validation_data,
                benign_csv=args.benign_validation_data,
                sbatch_job_name=sbatch_job_name,
                account=args.account,
                gres=args.gres,
                cpus=args.cpus_per_task,
                mem=args.mem,
                time_limit=args.time_limit,
                module_line=args.module_line,
                venv_activate=args.venv_activate,
                total_epochs=args.total_epochs,
            )

            sh_path = script_dir / f"{slug}.sh"
            sh_path.write_text(body, encoding="utf-8")
            os.chmod(sh_path, 0o755)

            mf.write(
                f"{slug}\t{lr}\t{lam}\t{eps}\t{run_id}\t{finetuned_base}\t{sh_path}\n"
            )

            cmd = ["sbatch", str(sh_path)]
            if args.dry_run:
                print("Would run:", " ".join(cmd))
            else:
                proc = subprocess.run(
                    cmd,
                    check=False,
                    text=True,
                    capture_output=True,
                )
                if proc.returncode != 0:
                    print(proc.stderr or proc.stdout, file=sys.stderr)
                    return proc.returncode
                out = proc.stdout.strip()
                m = re.search(r"Submitted batch job (\d+)", out)
                job_id = m.group(1) if m else out
                print(out, sh_path)
                submitted.append(job_id)

    print(f"Wrote {len(combos)} job scripts to {script_dir}")
    print(f"Manifest: {manifest}")
    if args.dry_run:
        print("Dry run: no jobs submitted.")
    else:
        print(f"Submitted {len(submitted)} jobs.")
    print()
    te = args.total_epochs
    print(
        f"Inference: each run saves {te} checkpoints; latest is "
        f"${{FINETUNED_BASE}}_epoch{te} (see manifest)."
    )
    print("W&B: set WANDB_MODE=offline in jobs (already in scripts). After jobs finish, from a machine")
    print("with network access, run: wandb sync <offline-run-folder> under your WANDB_DIR.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
