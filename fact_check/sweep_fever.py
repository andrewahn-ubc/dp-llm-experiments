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
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from fact_check.config_loader import load_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt(v: float) -> str:
    return f"{v:g}"


def build_env(model_name: str, max_seq_len: int,
              lam: float, lr: float, eps: float, rank: int,
              epochs: int, batch_size: int, ckpt_every_steps: int,
              resume_from: str | None, timestamp: str,
              augmentation_only: bool = False,
              dataset: str = "fever", mem: str | None = None,
              held_out_perturbations: list[str] | None = None,
              loo_prefix: str = "") -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "MODEL_NAME":               model_name,
        "MAX_SEQ_LEN":              str(max_seq_len),
        "LAMBDA":                   fmt(lam),
        "LR":                       fmt(lr),
        "EPSILON":                  fmt(eps),
        "LORA_RANK":                str(rank),
        "EPOCHS":                   str(epochs),
        "BATCH_SIZE":               str(batch_size),
        "CKPT_EVERY_STEPS":         str(ckpt_every_steps),
        "SWEEP_TIMESTAMP":          timestamp,
        "AUGMENTATION_ONLY":        "1" if augmentation_only else "0",
        "DATASET":                  dataset,
        "HELD_OUT_PERTURBATIONS":   " ".join(held_out_perturbations) if held_out_perturbations else "",
        "LOO_PREFIX":               loo_prefix,
        # FC_* vars are already in os.environ (from env.sh) and forwarded
        # automatically to sbatch via os.environ.copy().
    })
    if resume_from:
        env["RESUME_FROM"] = resume_from
    return env


def submit(slurm_script: str, env: dict, mem: str | None = None) -> str | None:
    """Submit via sbatch. Returns job ID string or None on failure."""
    cmd = ["sbatch", "--parsable"]
    if mem:
        cmd += [f"--mem={mem}"]
    cmd.append(slurm_script)
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
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
    parser.add_argument("--model",            default="bert",
                        choices=["bert", "deberta", "llama"],
                        help="Which model to train (bert|deberta|llama). Selects name/hf_repo/max_seq_len from config.")
    parser.add_argument("--dataset",          choices=["fever", "vitaminc"], default="fever",
                        help="Training dataset.")
    parser.add_argument("--mem",              default=None,
                        help="Override SLURM --mem (e.g. '48G' for Llama). Passed as --mem to sbatch.")
    parser.add_argument("--lambdas",          nargs="+", type=float, default=None)
    parser.add_argument("--lrs",              nargs="+", type=float, default=None)
    parser.add_argument("--epsilons",         nargs="+", type=float, default=None)
    parser.add_argument("--lora-ranks",       nargs="+", type=int,   default=None)
    parser.add_argument("--epochs",           type=int,   default=None)
    parser.add_argument("--batch-size",       type=int,   default=None)
    parser.add_argument("--ckpt-every-steps", type=int,   default=None)
    parser.add_argument("--resume-from",        default=None)
    parser.add_argument("--augmentation-only", action="store_true",
                        help="Run augmentation-only baseline (CE on all rows, λ forced to 0, no stability loss).")
    parser.add_argument("--held-out-perturbations", nargs="*", default=[],
                        help="Passed through to train_fever.py. Exclude these families from "
                             "stab loss. Use --loo-sweep to run all four leave-one-out configs.")
    parser.add_argument("--loo-sweep", action="store_true",
                        help="Run leave-one-out sweep: submit 4 × grid jobs, one per held-out "
                             "perturbation family. Each job trains on the other 3 families for "
                             "the regularizer. Overrides --held-out-perturbations.")
    parser.add_argument("--slurm-script",      default="fact_check/train_fever.sh")
    parser.add_argument("--submit-delay",     type=float, default=2.0,
                        help="Seconds to wait between sbatch submissions (default: 2)")
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

    model_cfg   = getattr(cfg.models, args.model)
    model_name  = model_cfg.name
    max_seq_len = model_cfg.max_seq_len

    # Per-model sweep overrides sit between global defaults and CLI flags.
    # Priority: CLI > model.sweep > global sweep
    model_sweep = getattr(model_cfg, "sweep", None)
    def _model_sweep(key, global_val):
        if model_sweep and hasattr(model_sweep, key):
            return list(getattr(model_sweep, key))
        return list(global_val)

    lambdas    = args.lambdas    or list(sweep.lambdas)
    lrs        = args.lrs        or _model_sweep("lrs",        sweep.lrs)
    epsilons   = args.epsilons   or list(sweep.epsilons)
    lora_ranks = args.lora_ranks or _model_sweep("lora_ranks", sweep.lora_ranks)
    epochs     = args.epochs     or train.epochs
    # batch_size: CLI > model.sweep.batch_size > training.batch_size
    batch_size = (args.batch_size
                  or (int(model_sweep.batch_size) if model_sweep and hasattr(model_sweep, "batch_size") else None)
                  or train.batch_size)
    ckpt_every = args.ckpt_every_steps or train.ckpt_every_steps

    slurm_script = args.slurm_script
    if not Path(slurm_script).exists():
        sys.exit(f"ERROR: SLURM script not found: {slurm_script}")

    combos = list(itertools.product(lambdas, lrs, epsilons, lora_ranks))

    if model_name == "bert_base_uncased" and any(r > 0 for r in lora_ranks):
        sys.exit("ERROR: LoRA is not supported for BERT. Use --lora-ranks 0 with --model bert.")

    ALL_FAMILIES = [
        "evidence_word_shuffle",
        "evidence_sent_shuffle",
        "evidence_trunc",
        "claim_synonym",
    ]
    LOO_SHORT_NAME = {
        "evidence_word_shuffle": "loo_wordshuffle",
        "evidence_sent_shuffle": "loo_sentshuffle",
        "evidence_trunc":        "loo_trunc",
        "claim_synonym":         "loo_synonym",
    }

    # Build list of (held_out_family_or_None, loo_prefix) configs to iterate over.
    if args.loo_sweep:
        loo_configs = [(f, LOO_SHORT_NAME[f]) for f in ALL_FAMILIES]
    else:
        held_out = args.held_out_perturbations or []
        if held_out:
            short = "_".join(
                LOO_SHORT_NAME.get(f, f"loo_{f.split('_')[-1]}") for f in held_out
            )
            loo_configs = [(held_out, short)]
        else:
            loo_configs = [([], "")]

    total_jobs = len(combos) * len(loo_configs)

    print(f"Config:    {Path(__file__).parent / 'config.yaml'}")
    print(f"Model:     {model_name}  (max_seq_len={max_seq_len})")
    print(f"Dataset:   {args.dataset}")
    print(f"Timestamp: {timestamp}")
    if args.loo_sweep:
        print(f"Mode:      LOO sweep ({len(ALL_FAMILIES)} held-out families × {len(combos)} grid = {total_jobs} jobs)")
    print(f"Grid:      {len(combos)} combos × {len(loo_configs)} LOO config(s) = {total_jobs} jobs")
    print(f"  lambdas={lambdas}  lrs={lrs}  epsilons={epsilons}  lora_ranks={lora_ranks}")
    print(f"  epochs={epochs}  batch={batch_size}  ckpt_every={ckpt_every}")
    if args.mem:
        print(f"  mem override: {args.mem}")
    if args.resume_from:
        print(f"  resume_from={args.resume_from}  (all jobs)")
    print()
    aug_prefix = "aug_" if args.augmentation_only else ""
    print("Jobs to be submitted:")
    for held_out_val, loo_prefix in loo_configs:
        held_out_list = [held_out_val] if isinstance(held_out_val, str) else held_out_val
        loo_label = f"  [held-out: {held_out_val}]" if held_out_val else ""
        for lam, lr, eps, rank in combos:
            run_prefix = f"{loo_prefix}_" if loo_prefix else aug_prefix
            run_id = f"{run_prefix}{model_name}_{args.dataset}_lam{fmt(lam)}_eps{fmt(eps)}_lr{fmt(lr)}_rank{rank}_{timestamp}"
            print(f"  {run_id}{loo_label}")
    print()

    if args.resume_from and total_jobs > 1:
        sys.exit(f"ERROR: --resume-from is for a single job but grid has {total_jobs} combos. "
                 f"Narrow the grid to exactly one lambda/lr/epsilon/rank combination.")

    if args.dry_run:
        print("[DRY RUN] — no jobs submitted.")
        return

    mode = "interactive (bash)" if args.interactive else "SLURM (sbatch)"
    if args.interactive and total_jobs > 1:
        print(f"NOTE: interactive mode runs {total_jobs} jobs sequentially in this shell.")
        print()

    try:
        confirm = input(f"Run {total_jobs} job(s) via {mode}? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.")
        return

    if confirm != "y":
        print("Aborted.")
        return

    print()
    succeeded, failed = [], []

    for held_out_val, loo_prefix in loo_configs:
        held_out_list = [held_out_val] if isinstance(held_out_val, str) else held_out_val
        for lam, lr, eps, rank in combos:
            run_prefix = f"{loo_prefix}_" if loo_prefix else aug_prefix
            run_id = f"{run_prefix}{model_name}_{args.dataset}_lam{fmt(lam)}_eps{fmt(eps)}_lr{fmt(lr)}_rank{rank}_{timestamp}"
            env = build_env(model_name, max_seq_len, lam, lr, eps, rank,
                            epochs, batch_size, ckpt_every, args.resume_from, timestamp,
                            args.augmentation_only, dataset=args.dataset, mem=args.mem,
                            held_out_perturbations=held_out_list,
                            loo_prefix=loo_prefix)

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
                job_id = submit(slurm_script, env, mem=args.mem)
                if job_id is None:
                    failed.append(run_id)
                else:
                    print(f"  Submitted {job_id}  →  {run_id}")
                    succeeded.append(job_id)
                time.sleep(args.submit_delay)

    print()
    print(f"{'Ran' if args.interactive else 'Submitted'} {len(succeeded)} / {len(combos)} jobs.")
    if failed:
        print(f"Failed: {failed}")
    if not args.interactive:
        print("Monitor:  squeue -u $USER")
    print("Results:  python fact_check/compare_runs.py")


if __name__ == "__main__":
    main()
