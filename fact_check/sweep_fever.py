"""
sweep_fever.py — Submit a hyperparameter grid as independent SLURM jobs.

Reads grid and paths from fact_check/config.yaml.

    python fact_check/sweep_fever.py --dry-run              # preview only
    python fact_check/sweep_fever.py                        # submit via sbatch
    python fact_check/sweep_fever.py --interactive          # run directly (interactive node)

CLI flags override config.yaml values. Anything not overridden uses the
config defaults, so you only specify what differs from the baseline.

Options:
    --config            Path to config.yaml (default: fact_check/config.yaml)
    --dry-run           Print job list without submitting or running
    --interactive       Run the shell script directly instead of via sbatch.
                        Jobs run sequentially in the current shell — use this
                        on an interactive allocation (salloc) for debugging.
    --lambdas           Space-separated list (overrides sweep.lambdas)
    --lrs               Space-separated list (overrides sweep.lrs)
    --epsilons          Space-separated list (overrides sweep.epsilons)
    --lora-ranks        Space-separated list (overrides sweep.lora_ranks)
    --epochs            Override training.epochs
    --batch-size        Override training.batch_size
    --ckpt-every-steps  Override training.ckpt_every_steps
    --resume-from       Checkpoint path passed to ALL jobs (for resuming a grid)
    --slurm-script      Path to SLURM/shell script (default: fact_check/train_fever.sh)
"""

import argparse
import itertools
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from fact_check.config_loader import load_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt(v: float) -> str:
    return f"{v:g}"


def build_env(model_name: str, lam: float, lr: float, eps: float, rank: int,
              epochs: int, batch_size: int, ckpt_every_steps: int,
              resume_from: str | None, timestamp: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "MODEL_NAME":        model_name,
        "LAMBDA":            fmt(lam),
        "LR":                fmt(lr),
        "EPSILON":           fmt(eps),
        "LORA_RANK":         str(rank),
        "EPOCHS":            str(epochs),
        "BATCH_SIZE":        str(batch_size),
        "CKPT_EVERY_STEPS":  str(ckpt_every_steps),
        "SWEEP_TIMESTAMP":   timestamp,
        # FC_* vars are already in os.environ (from env.sh) and forwarded
        # automatically to sbatch via os.environ.copy().
    })
    if resume_from:
        env["RESUME_FROM"] = resume_from
    return env


def submit(slurm_script: str, env: dict) -> str | None:
    """Submit via sbatch. Returns job ID string or None on failure."""
    result = subprocess.run(
        ["sbatch", "--parsable", slurm_script],
        env=env, capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr.strip()}", file=sys.stderr)
        return None
    return result.stdout.strip()


def run_interactive(slurm_script: str, env: dict) -> bool:
    """Run the script directly in the current shell (for interactive allocations).
    Streams output live. Returns True on success."""
    result = subprocess.run(["bash", slurm_script], env=env)
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config",           default=None)
    parser.add_argument("--dry-run",          action="store_true")
    parser.add_argument("--interactive",      action="store_true",
                        help="Run script directly via bash instead of sbatch (sequential).")
    parser.add_argument("--lambdas",          nargs="+", type=float, default=None)
    parser.add_argument("--lrs",              nargs="+", type=float, default=None)
    parser.add_argument("--epsilons",         nargs="+", type=float, default=None)
    parser.add_argument("--lora-ranks",       nargs="+", type=int,   default=None)
    parser.add_argument("--epochs",           type=int,   default=None)
    parser.add_argument("--batch-size",       type=int,   default=None)
    parser.add_argument("--ckpt-every-steps", type=int,   default=None)
    parser.add_argument("--resume-from",      default=None)
    parser.add_argument("--slurm-script",     default="fact_check/train_fever.sh")
    args = parser.parse_args()

    if args.resume_from:
        meta_path = Path(args.resume_from) / "checkpoint_meta.json"
        if meta_path.exists():
            import json
            with open(meta_path) as f:
                timestamp = json.load(f).get("sweep_timestamp") or datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    cfg = load_config(Path(args.config) if args.config else None)
    sweep = cfg.sweep
    train = cfg.training

    # CLI overrides config; config is the default
    lambdas    = args.lambdas    or list(sweep.lambdas)
    lrs        = args.lrs        or list(sweep.lrs)
    epsilons   = args.epsilons   or list(sweep.epsilons)
    lora_ranks = args.lora_ranks or list(sweep.lora_ranks)
    epochs          = args.epochs          or train.epochs
    batch_size      = args.batch_size      or train.batch_size
    ckpt_every      = args.ckpt_every_steps or train.ckpt_every_steps

    slurm_script = args.slurm_script
    if not Path(slurm_script).exists():
        sys.exit(f"ERROR: SLURM script not found: {slurm_script}")

    combos = list(itertools.product(lambdas, lrs, epsilons, lora_ranks))

    print(f"Config:    {Path(__file__).parent / 'config.yaml'}")
    print(f"Timestamp: {timestamp}")
    print(f"Grid:      {len(combos)} jobs")
    print(f"  lambdas={lambdas}  lrs={lrs}  epsilons={epsilons}  lora_ranks={lora_ranks}")
    print(f"  epochs={epochs}  batch={batch_size}  ckpt_every={ckpt_every}")
    if args.resume_from:
        print(f"  resume_from={args.resume_from}  (all jobs)")
    print()
    print("Jobs to be submitted:")
    for lam, lr, eps, rank in combos:
        run_id = f"{cfg.model.name}_lam{fmt(lam)}_eps{fmt(eps)}_lr{fmt(lr)}_rank{rank}_{timestamp}"
        print(f"  {run_id}")
    print()

    if args.resume_from and len(combos) > 1:
        sys.exit(f"ERROR: --resume-from is for a single job but grid has {len(combos)} combos. "
                 f"Narrow the grid to exactly one lambda/lr/epsilon/rank combination.")

    if args.dry_run:
        print("[DRY RUN] — no jobs submitted.")
        return

    mode = "interactive (bash)" if args.interactive else "SLURM (sbatch)"
    if args.interactive and len(combos) > 1:
        print(f"NOTE: interactive mode runs {len(combos)} jobs sequentially in this shell.")
        print()

    try:
        confirm = input(f"Run {len(combos)} job(s) via {mode}? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.")
        return

    if confirm != "y":
        print("Aborted.")
        return

    print()
    succeeded, failed = [], []

    for lam, lr, eps, rank in combos:
        env    = build_env(cfg.model.name, lam, lr, eps, rank, epochs, batch_size, ckpt_every, args.resume_from, timestamp)
        run_id = f"{cfg.model.name}_lam{fmt(lam)}_eps{fmt(eps)}_lr{fmt(lr)}_rank{rank}_{timestamp}"

        if args.interactive:
            print(f"── Running {run_id} ──")
            ok = run_interactive(slurm_script, env)
            if ok:
                print(f"  Done  →  {run_id}")
                succeeded.append(run_id)
            else:
                print(f"  FAILED  →  {run_id}", file=sys.stderr)
                failed.append(run_id)
        else:
            job_id = submit(slurm_script, env)
            if job_id:
                print(f"  Submitted {job_id}  →  {run_id}")
                succeeded.append(job_id)
            else:
                failed.append(run_id)

    print()
    print(f"{'Ran' if args.interactive else 'Submitted'} {len(succeeded)} / {len(combos)} jobs.")
    if failed:
        print(f"Failed: {failed}")
    if not args.interactive:
        print("Monitor:  squeue -u $USER")
    print("Results:  python fact_check/compare_runs.py")


if __name__ == "__main__":
    main()
