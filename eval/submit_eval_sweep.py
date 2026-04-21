#!/usr/bin/env python3
"""
Submit SLURM evaluation jobs for the 75-config training sweep.

Each SLURM job runs `eval_sweep.py` three times — once for epoch 1, 3, and 5 —
against the checkpoints produced by the corresponding training job. With the
default --epochs, this totals 75 jobs × 3 eval passes = 225 evaluated
checkpoints.

The hyperparameter grid is intentionally kept in sync with
`train/submit_wandb_sweep.py` so the slugs match exactly and we can look up the
adapter paths that training produced:

    {checkpoint_root}/{slug}_finetuned_llm_epoch{N}

Each eval job generates model responses on validation.csv (GCG, AutoDAN, PAIR)
and frr_validation.csv (benign), then judges them with a single shared
Mistral-7B-Instruct-v0.2 instance (HarmBench prompt for jailbreaks, refusal
prompt for benign), writing enriched CSVs + one summary row to an append-only
TSV.

Weights & Biases is run in offline mode on compute nodes (same pattern as
training); sync from a node with internet afterwards:

    wandb sync --sync-all
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

# Keep in sync with train/submit_wandb_sweep.py (3 × 5 × 5 = 75 configs)
LEARNING_RATES = (1e-6, 1e-5, 2e-5)
LAMBDAS = (0.1, 0.3, 1.0, 3.0, 10.0)
EPSILONS = (-1.0, -0.5, 0.0, 0.5, 1.0)

DEFAULT_EVAL_EPOCHS = (1, 3, 5)


def make_run_slug(lr: float, lam: float, eps: float) -> str:
    return f"run_lr{lr:g}_lam{lam:g}_eps{eps:g}".replace(" ", "_")


def render_job_script(
    *,
    slug: str,
    lr: float,
    lam: float,
    eps: float,
    epochs: tuple[int, ...],
    wandb_project: str,
    wandb_dir: str,
    wandb_run_ids: dict[int, str],
    finetuned_base: str,
    eval_py: str,
    base_llm: str,
    judge_path: str,
    validation_csv: str,
    benign_csv: str,
    eval_output_root: str,
    summary_file: str,
    sbatch_job_name: str,
    account: str,
    gres: str,
    cpus: int,
    mem: str,
    time_limit: str,
    module_line: str,
    venv_activate: str,
    gen_batch_size: int,
    judge_batch_size: int,
    overwrite: bool,
) -> str:
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={sbatch_job_name}",
        f"#SBATCH --account={account}",
        f"#SBATCH --gres={gres}",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --output=output/eval_{slug}_%j.out",
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
        "# Node-local Hugging Face caches (same pattern as train jobs)",
        "export TRANSFORMERS_CACHE=$SLURM_TMPDIR/hf_cache",
        "export HF_HOME=$SLURM_TMPDIR/hf_home",
        'mkdir -p "$TRANSFORMERS_CACHE" "$HF_HOME"',
        "",
        "# Weights & Biases — offline on compute nodes; sync from login afterwards",
        "export WANDB_MODE=offline",
        f'export WANDB_PROJECT="{wandb_project}"',
        f'export WANDB_DIR="{wandb_dir}"',
        "",
        f'export FINETUNED_BASE="{finetuned_base}"',
        f'export EVAL_PY="{eval_py}"',
        f'export EVAL_OUTPUT_ROOT="{eval_output_root}"',
        f'export EVAL_SWEEP_SUMMARY="{summary_file}"',
        'mkdir -p "$EVAL_OUTPUT_ROOT"',
        'mkdir -p "$WANDB_DIR"',
        'mkdir -p "$(dirname "$EVAL_SWEEP_SUMMARY")"',
        "",
    ]

    for ep in epochs:
        run_id = wandb_run_ids[ep]
        run_name = f"eval_{slug}_epoch{ep}"
        adapter = f"${{FINETUNED_BASE}}_epoch{ep}"
        harmful_out = f"$EVAL_OUTPUT_ROOT/{slug}_epoch{ep}_harmful.csv"
        benign_out = f"$EVAL_OUTPUT_ROOT/{slug}_epoch{ep}_benign.csv"

        lines.append(f"# -------- Epoch {ep} --------")
        lines.append(f'if [ ! -d "{adapter}" ]; then')
        lines.append(f'  echo "WARNING: adapter not found: {adapter} — skipping epoch {ep}"')
        lines.append("else")
        lines.append(f'  export WANDB_RUN_ID="{run_id}"')
        lines.append(f'  export WANDB_RUN_NAME="{run_name}"')
        lines.append('  python "$EVAL_PY" \\')
        lines.append(f'    --slug "{slug}" \\')
        lines.append(f"    --epoch {ep} \\")
        lines.append(f"    --lr {lr} \\")
        lines.append(f"    --lambda-val {lam} \\")
        lines.append(f"    --epsilon {eps} \\")
        lines.append(f'    --base-llm "{base_llm}" \\')
        lines.append(f'    --resume-from "{adapter}" \\')
        lines.append(f'    --judge-path "{judge_path}" \\')
        lines.append(f'    --validation-data "{validation_csv}" \\')
        lines.append(f'    --benign-validation-data "{benign_csv}" \\')
        lines.append(f'    --harmful-output-file "{harmful_out}" \\')
        lines.append(f'    --benign-output-file "{benign_out}" \\')
        lines.append(f'    --summary-file "$EVAL_SWEEP_SUMMARY" \\')
        lines.append(f"    --gen-batch-size {gen_batch_size} \\")
        lines.append(f"    --judge-batch-size {judge_batch_size}"
                     + (" \\" if overwrite else ""))
        if overwrite:
            lines.append("    --overwrite")
        lines.append("fi")
        lines.append("")

    lines += [
        f'echo "Eval done for {slug}; summary appended to $EVAL_SWEEP_SUMMARY"',
        "",
    ]
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "dp-llm-safety-eval"))
    p.add_argument("--wandb-dir", default="$SCRATCH/wandb_offline")
    p.add_argument(
        "--checkpoint-root",
        default="$SCRATCH/dp-llm-sweep",
        help="Matches --checkpoint-root used by train/submit_wandb_sweep.py.",
    )
    p.add_argument(
        "--eval-output-root",
        default="$SCRATCH/dp-llm-eval",
        help="Where enriched CSVs land.",
    )
    p.add_argument(
        "--summary-file",
        default="$SCRATCH/dp-llm-eval/summary.tsv",
        help="Append-only TSV shared across all eval jobs.",
    )
    p.add_argument(
        "--eval-py",
        default="$SCRATCH/dp-llm-experiments/eval/eval_sweep.py",
    )
    p.add_argument(
        "--base-llm",
        default="/home/taegyoem/scratch/llama2_7b_chat_hf",
    )
    p.add_argument(
        "--judge-path",
        default="/home/taegyoem/scratch/mistral_7b_instruct",
    )
    p.add_argument(
        "--validation-data",
        default="$SCRATCH/dp-llm-experiments/official_data/validation.csv",
    )
    p.add_argument(
        "--benign-validation-data",
        default="$SCRATCH/dp-llm-experiments/official_data/frr_validation.csv",
    )
    p.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        default=list(DEFAULT_EVAL_EPOCHS),
        help="Which training epochs to evaluate per config.",
    )

    p.add_argument("--account", default="rrg-mijungp")
    p.add_argument("--gres", default="gpu:1")
    p.add_argument("--cpus-per-task", type=int, default=6)
    p.add_argument("--mem", default="40G")
    p.add_argument(
        "--hours-per-epoch",
        type=int,
        default=2,
        help="Budget per evaluated epoch (hours). Default wall time = hours-per-epoch × N-epochs + 1.",
    )
    p.add_argument("--time", dest="time_limit", default=None)
    p.add_argument("--module-line", default="module load StdEnv/2023 python/3.11")
    p.add_argument("--venv-activate", default="$SCRATCH/venv/nanogcg/bin/activate")
    p.add_argument("--gen-batch-size", type=int, default=8)
    p.add_argument("--judge-batch-size", type=int, default=8)

    p.add_argument("--script-dir", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="First N configs only (smoke test).")
    p.add_argument("--overwrite", action="store_true", help="Regenerate even if outputs exist.")

    args = p.parse_args(argv)
    if args.time_limit is None:
        total_h = args.hours_per_epoch * len(args.epochs) + 1
        args.time_limit = f"{total_h}:00:00"
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    epochs = tuple(args.epochs)
    combos = list(itertools.product(LEARNING_RATES, LAMBDAS, EPSILONS))
    if args.limit is not None:
        combos = combos[: args.limit]

    script_dir = args.script_dir or (Path.cwd() / "eval_sweep_jobs")
    script_dir.mkdir(parents=True, exist_ok=True)

    manifest = script_dir / "eval_sweep_manifest.tsv"
    submitted: list[str] = []

    with manifest.open("w", encoding="utf-8") as mf:
        mf.write(
            "slug\tlr\tlambda\tepsilon\tepochs\t"
            "wandb_run_ids\tfinetuned_base\tjob_script\n"
        )

        for lr, lam, eps in combos:
            slug = make_run_slug(lr, lam, eps)
            finetuned_base = f"{args.checkpoint_root.rstrip('/')}/{slug}_finetuned_llm"
            wandb_run_ids = {ep: uuid.uuid4().hex for ep in epochs}

            sbatch_job_name = f"ev_{slug}"[:64]

            body = render_job_script(
                slug=slug,
                lr=lr,
                lam=lam,
                eps=eps,
                epochs=epochs,
                wandb_project=args.wandb_project,
                wandb_dir=args.wandb_dir,
                wandb_run_ids=wandb_run_ids,
                finetuned_base=finetuned_base,
                eval_py=args.eval_py,
                base_llm=args.base_llm,
                judge_path=args.judge_path,
                validation_csv=args.validation_data,
                benign_csv=args.benign_validation_data,
                eval_output_root=args.eval_output_root,
                summary_file=args.summary_file,
                sbatch_job_name=sbatch_job_name,
                account=args.account,
                gres=args.gres,
                cpus=args.cpus_per_task,
                mem=args.mem,
                time_limit=args.time_limit,
                module_line=args.module_line,
                venv_activate=args.venv_activate,
                gen_batch_size=args.gen_batch_size,
                judge_batch_size=args.judge_batch_size,
                overwrite=args.overwrite,
            )

            sh_path = script_dir / f"{slug}.sh"
            sh_path.write_text(body, encoding="utf-8")
            os.chmod(sh_path, 0o755)

            mf.write(
                "\t".join(
                    [
                        slug,
                        f"{lr}",
                        f"{lam}",
                        f"{eps}",
                        ",".join(str(e) for e in epochs),
                        ",".join(f"{e}:{wandb_run_ids[e]}" for e in epochs),
                        finetuned_base,
                        str(sh_path),
                    ]
                )
                + "\n"
            )

            cmd = ["sbatch", str(sh_path)]
            if args.dry_run:
                print("Would run:", " ".join(cmd))
            else:
                proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
                if proc.returncode != 0:
                    print(proc.stderr or proc.stdout, file=sys.stderr)
                    return proc.returncode
                out = proc.stdout.strip()
                m = re.search(r"Submitted batch job (\d+)", out)
                job_id = m.group(1) if m else out
                print(out, sh_path)
                submitted.append(job_id)

    print(f"Wrote {len(combos)} eval job scripts to {script_dir}")
    print(f"Manifest: {manifest}")
    print(f"Eval epochs per config: {epochs}")
    if args.dry_run:
        print("Dry run: no jobs submitted.")
    else:
        print(f"Submitted {len(submitted)} jobs ({len(combos)} configs × {len(epochs)} epochs each).")
    print()
    print("Summary TSV (one row per evaluated checkpoint):")
    print(f"  {args.summary_file}")
    print("After jobs finish, aggregate with:")
    print("  python eval/aggregate_eval_results.py --summary-file <summary.tsv>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
