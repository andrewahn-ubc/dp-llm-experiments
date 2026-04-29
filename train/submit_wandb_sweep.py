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

LEARNING_RATES = (1e-6, 1e-5, 2e-5)
LAMBDAS = (0.1, 0.3, 1.0, 3.0, 10.0, 20.0, 30.0)
EPSILONS = (-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)


def make_run_slug(lr: float, lam: float, eps: float) -> str:
    """Stable, filesystem-friendly id that encodes the three hyperparameters."""
    return f"run_lr{lr:g}_lam{lam:g}_eps{eps:g}".replace(" ", "_")


def _render_eval_block(
    *,
    ep: int,
    slug: str,
    lr: float,
    lam: float,
    eps: float,
    validation_csv: str,
    system_prompt_mode: str,
) -> list[str]:
    """Lines that evaluate ${FINETUNED_BASE}_epoch{ep} via $EVAL_PY.

    Assumes the caller has already exported $EVAL_PY, $EVAL_OUT_DIR, and
    $BENIGN_TMP (the FRR input with an 'Original Prompt' column).
    """
    return [
        "# ----------------------------",
        f"# Epoch {ep} eval (Protocol-{'Defended' if system_prompt_mode == 'default' else 'Undefended'})",
        "# ----------------------------",
        f'CKPT_EP{ep}="${{FINETUNED_BASE}}_epoch{ep}"',
        f'HARMFUL_OUT_STEM_EP{ep}="${{EVAL_OUT_DIR}}/{slug}_epoch{ep}_val_output"',
        f'BENIGN_OUT_STEM_EP{ep}="${{EVAL_OUT_DIR}}/{slug}_epoch{ep}_frr_output"',
        f'METRICS_OUT_EP{ep}="${{EVAL_OUT_DIR}}/{slug}_epoch{ep}_metrics.tsv"',
        "",
        'python "$EVAL_PY" \\',
        '    --eval-mode "seen-family" \\',
        f'    --system-prompt-mode "{system_prompt_mode}" \\',
        f'    --resume-from "$CKPT_EP{ep}" \\',
        f'    --validation-data "{validation_csv}" \\',
        '    --benign-validation-data "$BENIGN_TMP" \\',
        f'    --harmful-output-file "$HARMFUL_OUT_STEM_EP{ep}" \\',
        f'    --benign-output-file "$BENIGN_OUT_STEM_EP{ep}"',
        "",
        f"# Aggregate per-family ASR + FRR for epoch {ep} into a single TSV.",
        f'python - "${{HARMFUL_OUT_STEM_EP{ep}}}.csv" "${{BENIGN_OUT_STEM_EP{ep}}}.csv" "$METRICS_OUT_EP{ep}" \\',
        f'    "{slug}" "{lr}" "{lam}" "{eps}" "{system_prompt_mode}" "{ep}" <<\'PY\'',
        "import sys",
        "import pandas as pd",
        "harmful_csv, benign_csv, metrics_tsv, slug, lr, lam, eps, mode, epoch = sys.argv[1:10]",
        "h = pd.read_csv(harmful_csv); b = pd.read_csv(benign_csv)",
        "def mu(df, col): return float((df[col].astype(str).str.lower()=='unsafe').mean()) if col in df.columns else None",
        "gcg = mu(h, 'GCG Safety'); ad = mu(h, 'AutoDAN Safety'); pa = mu(h, 'PAIR Safety')",
        "vals = [v for v in (gcg, ad, pa) if v is not None]",
        "mean_asr = sum(vals)/len(vals) if vals else None",
        "frr = float((b['Original Safety'].astype(str).str.lower()=='refusal').mean()) if 'Original Safety' in b.columns else None",
        "rows = [('slug',slug),('epoch',epoch),('lr',lr),('lambda',lam),('epsilon',eps),('system_prompt_mode',mode),",
        "        ('gcg_asr',gcg),('autodan_asr',ad),('pair_asr',pa),('mean_asr',mean_asr),('frr',frr),",
        "        ('n_harmful',len(h)),('n_benign',len(b))]",
        "with open(metrics_tsv,'w',encoding='utf-8') as f:",
        "    f.write('metric\\tvalue\\n')",
        "    for k,v in rows: f.write(f'{k}\\t{v}\\n')",
        "print(f'Wrote {metrics_tsv}')",
        "print(f'epoch={epoch} mean_asr={mean_asr}, frr={frr}')",
        "PY",
        "",
    ]


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
    eval_py: str,
    training_csv: str,
    validation_csv: str,
    benign_csv: str,
    eval_output_dir: str,
    sbatch_job_name: str,
    account: str,
    gres: str,
    cpus: int,
    mem: str,
    time_limit: str,
    module_line: str,
    venv_activate: str,
    total_epochs: int,
    system_prompt_mode: str,
    eval_epochs: list[int],
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
        "# Weights & Biases — offline on compute nodes; sync later from login / your laptop.",
        "# IMPORTANT: keep WANDB_DIR on node-local fast disk ($SLURM_TMPDIR), NOT on Lustre",
        "# ($SCRATCH). Lustre's file-metadata caching causes wandb's service-port file to",
        "# show up after the 30s poll timeout under load, killing the job at wandb.init.",
        "# We sync the offline run folder to a persistent dir at the end of the job.",
        "export WANDB_MODE=offline",
        f'export WANDB_PROJECT="{wandb_project}"',
        f'export WANDB_DIR_PERSISTENT="{wandb_dir}"',
        'export WANDB_DIR="$SLURM_TMPDIR/wandb"',
        f"export WANDB_RUN_ID={run_id}",
        f'export WANDB_RUN_NAME="{run_name}"',
        "",
        f'export FINETUNED_BASE="{finetuned_base}"',
        f'export TRAIN_PY="{train_py}"',
        f'export EVAL_PY="{eval_py}"',
        f'export EVAL_OUT_DIR="{eval_output_dir}"',
        'mkdir -p "$(dirname "$FINETUNED_BASE")"',
        'mkdir -p "$WANDB_DIR" "$WANDB_DIR_PERSISTENT"',
        'mkdir -p "$EVAL_OUT_DIR"',
        "",
        "# Copy any wandb runtime artifacts back to persistent storage on EXIT,",
        "# even if the job is killed or fails partway through.",
        "trap 'cp -r \"$WANDB_DIR\"/* \"$WANDB_DIR_PERSISTENT/\" 2>/dev/null || true' EXIT",
        "",
    ]

    # Pre-build the FRR input once (renames {adversarial,goal,...} -> 'Original Prompt'
    # so eval.py finds it). Reused across all per-epoch eval calls.
    lines += [
        "# ----------------------------",
        "# FRR input prep (eval.py expects an 'Original Prompt' column)",
        "# ----------------------------",
        f'BENIGN_TMP="${{SLURM_TMPDIR}}/frr_eval_input_{slug}.csv"',
        f"python - \"{benign_csv}\" \"$BENIGN_TMP\" <<'PY'",
        "import sys",
        "import pandas as pd",
        "src, dst = sys.argv[1:3]",
        "df = pd.read_csv(src)",
        "if 'Original Prompt' in df.columns:",
        "    df.to_csv(dst, index=False)",
        "    print(f'[sweep_eval] benign data already has Original Prompt -> {dst}')",
        "    sys.exit(0)",
        "candidates = ['adversarial','Adversarial','goal','Goal','prompt','Prompt','original_prompt','instruction','Instruction']",
        "src_col = next((c for c in candidates if c in df.columns), None)",
        "if src_col is None:",
        "    raise ValueError(f'Could not build Original Prompt column from {list(df.columns)}')",
        "df = df.copy(); df['Original Prompt'] = df[src_col].astype(str)",
        "df.to_csv(dst, index=False)",
        "print(f'[sweep_eval] mapped {src_col!r} -> Original Prompt -> {dst}')",
        "PY",
        "",
    ]

    eval_eps_set = set(eval_epochs)
    for ep in range(1, total_epochs + 1):
        # Train epoch ep
        do_eval_after = ep in eval_eps_set
        suffix = " + eval" if do_eval_after else ""
        lines.append(f"# Epoch {ep} - training{suffix}")
        lines.append('python "$TRAIN_PY" \\')
        lines.append('    --eval-mode "seen-family" \\')
        lines.append(f'    --system-prompt-mode "{system_prompt_mode}" \\')
        lines.append('    --finetuned-llm-path "$FINETUNED_BASE" \\')
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

        if do_eval_after:
            lines += _render_eval_block(
                ep=ep,
                slug=slug,
                lr=lr,
                lam=lam,
                eps=eps,
                validation_csv=validation_csv,
                system_prompt_mode=system_prompt_mode,
            )

    eval_eps_str = ",".join(str(e) for e in sorted(eval_eps_set))
    lines += [
        f'echo "Final adapter (latest): ${{FINETUNED_BASE}}_epoch{total_epochs}"',
        f'echo "Intermediate checkpoints kept: ${{FINETUNED_BASE}}_epoch1 ... _epoch{total_epochs}"',
        f'echo "Eval epochs: {eval_eps_str}"',
        f'echo "Per-epoch metrics in: ${{EVAL_OUT_DIR}}/{slug}_epoch<N>_metrics.tsv"',
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
        "--eval-py",
        default="$SCRATCH/dp-llm-experiments/eval/eval.py",
        help="Path to eval.py on the cluster (run after the final training epoch).",
    )
    p.add_argument(
        "--eval-output-dir",
        default="$SCRATCH/dp-llm-sweep/eval_outputs",
        help="Directory where each run's harmful/benign CSVs and metrics.tsv are written.",
    )
    p.add_argument(
        "--system-prompt-mode",
        default="empty",
        choices=["default", "empty"],
        help=(
            "System prompt mode applied at BOTH train-time (in train.py) and "
            "eval-time (in eval.py). 'default' uses Llama-2-Chat's helpful/safe "
            "system prompt; 'empty' omits the system role entirely "
            "(Protocol-Undefended). Defaults to 'empty' for this sweep."
        ),
    )
    p.add_argument(
        "--eval-epochs",
        default="1,3,5",
        help=(
            "Comma-separated list of epochs at which to evaluate the model "
            "(checkpointed adapter at end of that epoch). Each produces "
            "{slug}_epoch{N}_{val_output,frr_output,metrics.tsv} files. "
            "Default: '1,3,5'."
        ),
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
        help="Budget per epoch (hours); used to set default --time as hours-per-epoch × total-epochs + eval-hours.",
    )
    p.add_argument(
        "--eval-hours",
        type=int,
        default=2,
        help="Hours added to wall time PER eval epoch (default: 2). Total eval "
        "budget = eval-hours × len(eval-epochs).",
    )
    p.add_argument(
        "--time",
        dest="time_limit",
        default=None,
        help="SLURM wall time for the entire sweep job (e.g. 15:00:00). "
        "Default: hours-per-epoch × total-epochs + eval-hours.",
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

    # Parse + validate --eval-epochs ("1,3,5") into a sorted unique list of ints.
    eval_eps: list[int] = []
    for tok in args.eval_epochs.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            v = int(tok)
        except ValueError:
            raise SystemExit(f"--eval-epochs got non-integer entry: {tok!r}")
        if v < 1 or v > args.total_epochs:
            raise SystemExit(
                f"--eval-epochs entry {v} out of range [1, {args.total_epochs}]"
            )
        eval_eps.append(v)
    args.eval_epochs_list = sorted(set(eval_eps))
    if not args.eval_epochs_list:
        raise SystemExit("--eval-epochs must contain at least one epoch")

    if args.time_limit is None:
        total_h = (
            args.hours_per_epoch * args.total_epochs
            + args.eval_hours * len(args.eval_epochs_list)
        )
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
                eval_py=args.eval_py,
                training_csv=args.training_data,
                validation_csv=args.validation_data,
                benign_csv=args.benign_validation_data,
                eval_output_dir=args.eval_output_dir,
                sbatch_job_name=sbatch_job_name,
                account=args.account,
                gres=args.gres,
                cpus=args.cpus_per_task,
                mem=args.mem,
                time_limit=args.time_limit,
                module_line=args.module_line,
                venv_activate=args.venv_activate,
                total_epochs=args.total_epochs,
                system_prompt_mode=args.system_prompt_mode,
                eval_epochs=args.eval_epochs_list,
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
