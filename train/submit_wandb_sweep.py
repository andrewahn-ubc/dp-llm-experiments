#!/usr/bin/env python3
"""
Submit a grid of SLURM training jobs (Narval / Compute Canada friendly).

Defaults target **final** runs (not train/val hyperparameter search):

  • Learning rates: ``2e-5`` only (override via ``--learning-rates``).
  • λ × ε grid: **7 × 5** full factorial except **λ=0** uses a single representative ε
    (see ``lambda_epsilon_pairs``): fewer jobs than 7×5×LR because ε does not affect
    training when λ=0.
  • Training CSV: ``official_data/train_plus_validation.csv``.
  • Embedded ``eval.py`` after training uses **test** files:
    ``combined_test_dataset.csv`` + ``frr_test.csv``.
  • ``--eval-epochs`` default: ``5`` (checkpoint + metrics once at end).

Each job still runs ``total_epochs`` sequential ``train.py`` invocations (default 5),
saving ``{FINETUNED_BASE}_epoch1 … _epoch5``. Use epoch 5 for reporting.

Use ``python run_final_pipeline.py`` to submit **both** LM modes (clean + perturbed),
**held-out** training (three families × same λ/ε grid), and the test-eval SLURM array.

Held-out jobs are submitted via ``--held-out-families gcg,autodan,pair`` (see below).

Weights & Biases is configured for offline mode on air-gapped compute nodes.
After jobs finish, sync from a machine with internet, from the same WANDB_DIR:

  wandb sync path/to/wandb/offline-run-...

Or: wandb sync --sync-all (see https://docs.wandb.ai/guides/track/public-api-guide/#syncing-offline-runs)

After jobs finish, aggregate eval TSVs (see train/aggregate_sweep_metrics.py):

  python train/aggregate_sweep_metrics.py --eval-output-dir $SCRATCH/dp-llm-sweep/eval_outputs

Requires: pip install wandb (in the same venv you use for training.)
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path

LAMBDAS = (0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
EPSILONS = (-1.0, -0.5, 0.0, 0.5, 1.0)


def lambda_epsilon_pairs(
    lambdas: tuple[float, ...],
    epsilons: tuple[float, ...],
) -> list[tuple[float, float]]:
    """Enumerate (λ, ε) sweep cells.

    When λ=0 the stability term does not contribute, so ε is unused in training; only
    one representative ε is scheduled (``0.0`` if it appears in ``epsilons``, else the
    middle ε value) to reduce redundant jobs while keeping slugs consistent with the grid.
    """
    if not epsilons:
        raise ValueError("epsilons must be non-empty")
    has_zero_eps = any(abs(float(e)) < 1e-15 for e in epsilons)
    eps_at_lambda_zero = 0.0 if has_zero_eps else float(epsilons[len(epsilons) // 2])
    out: list[tuple[float, float]] = []
    for lam in lambdas:
        lam_f = float(lam)
        if abs(lam_f) < 1e-12:
            out.append((lam_f, eps_at_lambda_zero))
        else:
            for eps in epsilons:
                out.append((lam_f, float(eps)))
    return out


def sweep_lr_lambda_epsilon_combos(
    learning_rates: tuple[float, ...],
    lambdas: tuple[float, ...],
    epsilons: tuple[float, ...],
) -> list[tuple[float, float, float]]:
    pairs = lambda_epsilon_pairs(lambdas, epsilons)
    return [(lr, lam, eps) for lr in learning_rates for lam, eps in pairs]


def make_run_slug(lr: float, lam: float, eps: float, lm_loss_input: str = "clean") -> str:
    """Stable, filesystem-friendly id that encodes the hyperparameters.

    The default lm_loss_input ("clean") produces the original slug format so
    pre-existing checkpoints/eval outputs are reused as-is. Any non-default
    LM-loss-input gets a short suffix (e.g. "_pertlm") so the two variants do
    not clobber each other when the sweep is run twice.
    """
    base = f"run_lr{lr:g}_lam{lam:g}_eps{eps:g}".replace(" ", "_")
    if lm_loss_input == "perturbed":
        return f"{base}_pertlm"
    return base


def _render_eval_block(
    *,
    ep: int,
    run_slug: str,
    lr: float,
    lam: float,
    eps: float,
    validation_csv: str,
    system_prompt_mode: str,
    lm_loss_input: str,
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
        f'HARMFUL_OUT_STEM_EP{ep}="${{EVAL_OUT_DIR}}/{run_slug}_epoch{ep}_val_output"',
        f'BENIGN_OUT_STEM_EP{ep}="${{EVAL_OUT_DIR}}/{run_slug}_epoch{ep}_frr_output"',
        f'METRICS_OUT_EP{ep}="${{EVAL_OUT_DIR}}/{run_slug}_epoch{ep}_metrics.tsv"',
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
        f'    "{run_slug}" "{lr}" "{lam}" "{eps}" "{system_prompt_mode}" "{lm_loss_input}" "{ep}" <<\'PY\'',
        "import sys",
        "import pandas as pd",
        "harmful_csv, benign_csv, metrics_tsv, slug, lr, lam, eps, mode, lm_loss_input, epoch = sys.argv[1:11]",
        "h = pd.read_csv(harmful_csv); b = pd.read_csv(benign_csv)",
        "def mu(df, col): return float((df[col].astype(str).str.lower()=='unsafe').mean()) if col in df.columns else None",
        "gcg = mu(h, 'GCG Safety'); ad = mu(h, 'AutoDAN Safety'); pa = mu(h, 'PAIR Safety')",
        "vals = [v for v in (gcg, ad, pa) if v is not None]",
        "mean_asr = sum(vals)/len(vals) if vals else None",
        "frr = float((b['Original Safety'].astype(str).str.lower()=='refusal').mean()) if 'Original Safety' in b.columns else None",
        "rows = [('slug',slug),('epoch',epoch),('lr',lr),('lambda',lam),('epsilon',eps),",
        "        ('system_prompt_mode',mode),('lm_loss_input',lm_loss_input),",
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
    run_slug: str,
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
    lm_loss_input: str,
    eval_epochs: list[int],
    train_eval_mode: str = "seen-family",
    train_unseen_family: str | None = None,
) -> str:
    # W&B run display name (visible after sync)
    run_name = run_slug

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={sbatch_job_name}",
        f"#SBATCH --account={account}",
        f"#SBATCH --gres={gres}",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --output=output/sweep_{run_slug}_%j.out",
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
        f'BENIGN_TMP="${{SLURM_TMPDIR}}/frr_eval_input_{run_slug}.csv"',
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
        lines.append(f'    --eval-mode "{train_eval_mode}" \\')
        if train_eval_mode == "unseen-family":
            if not train_unseen_family:
                raise ValueError("train_unseen_family required when train_eval_mode is unseen-family")
            lines.append(f'    --unseen-family "{train_unseen_family}" \\')
        lines.append(f'    --system-prompt-mode "{system_prompt_mode}" \\')
        lines.append(f'    --lm-loss-input "{lm_loss_input}" \\')
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
                run_slug=run_slug,
                lr=lr,
                lam=lam,
                eps=eps,
                validation_csv=validation_csv,
                system_prompt_mode=system_prompt_mode,
                lm_loss_input=lm_loss_input,
            )

    eval_eps_str = ",".join(str(e) for e in sorted(eval_eps_set))
    lines += [
        f'echo "Final adapter (latest): ${{FINETUNED_BASE}}_epoch{total_epochs}"',
        f'echo "Intermediate checkpoints kept: ${{FINETUNED_BASE}}_epoch1 ... _epoch{total_epochs}"',
        f'echo "Eval epochs: {eval_eps_str}"',
        f'echo "Per-epoch metrics in: ${{EVAL_OUT_DIR}}/{run_slug}_epoch<N>_metrics.tsv"',
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
        help="Directory prefix for finetuned checkpoints. Seen-family runs use "
        "{checkpoint-root}/{slug}_finetuned_llm; with --held-out-families, each job "
        "uses {checkpoint-root}/heldout_{family}_{slug}_finetuned_llm.",
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
        "--lm-loss-input",
        default="clean",
        choices=["clean", "perturbed"],
        help=(
            "Which prompt the language-modelling cross-entropy in train.py is "
            "conditioned on. 'clean' (default) uses the unperturbed Original "
            "Prompt — the original method as written; refusal behavior on "
            "attacks is transferred only via the stability regularizer. "
            "'perturbed' uses the GCG/AutoDAN/PAIR-perturbed prompt "
            "(R2D2-style adversarial SFT). Combine with --lambda-val 0 in your "
            "lambda grid to obtain a pure adversarial-SFT baseline; combine "
            "with lambda>0 to obtain adversarial SFT + stability regularizer "
            "(the strict-superset variant of the original method). When set to "
            "'perturbed' the per-run slug gets a '_pertlm' suffix so artifacts "
            "do not collide with the 'clean' variant."
        ),
    )
    p.add_argument(
        "--eval-epochs",
        default="5",
        help=(
            "Comma-separated epochs after which to run eval.py (checkpoint at end of epoch). "
            "Default: '5' only (final checkpoint). Use '1,3,5' for multi-epoch monitoring."
        ),
    )
    p.add_argument(
        "--training-data",
        default="$SCRATCH/dp-llm-experiments/official_data/train_plus_validation.csv",
        help="Training CSV (default: train ∪ validation for final runs).",
    )
    p.add_argument(
        "--validation-data",
        default="$SCRATCH/dp-llm-experiments/official_data/combined_test_dataset.csv",
        help=(
            "Harmful prompts for eval.py inside each training job (default: **test** set). "
            "Override if you want intermediate validation-only metrics."
        ),
    )
    p.add_argument(
        "--benign-validation-data",
        default="$SCRATCH/dp-llm-experiments/official_data/frr_test.csv",
        help="Benign prompts for FRR in eval.py (default: **test** set).",
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
        help=(
            "Budget per epoch (hours); used to set default --time as "
            "hours-per-epoch × total-epochs + eval-hours × len(eval-epochs). "
            "Default 2 gives 12h with total-epochs=5, eval-epochs=5 only "
            "(2×5 + 2×1)."
        ),
    )
    p.add_argument(
        "--eval-hours",
        type=int,
        default=2,
        help=(
            "Hours added to wall time PER eval epoch (default: 2). Total eval "
            "budget = eval-hours × len(eval-epochs)."
        ),
    )
    p.add_argument(
        "--time",
        dest="time_limit",
        default=None,
        help="SLURM wall time for the entire sweep job (e.g. 15:00:00). "
        "Default: hours-per-epoch × total-epochs + eval-hours × len(eval-epochs).",
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
        "--record-job-ids",
        type=Path,
        default=None,
        help=(
            "Append each successfully submitted SLURM batch job id to this file (one id per line). "
            "Used by run_final_pipeline.py to chain the test-eval array after training."
        ),
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only submit the first N combinations (useful for smoke tests).",
    )
    p.add_argument(
        "--learning-rates",
        default="2e-5",
        help=(
            "Comma-separated learning rates for the sweep (default: 2e-5 only). "
            "Example for search: '1e-6,1e-5,2e-5'."
        ),
    )
    p.add_argument(
        "--held-out-families",
        default="",
        help=(
            "Comma-separated subset of {gcg,autodan,pair}. When non-empty, each "
            "(lr,λ,ε) combo submits **one SLURM job per family** that trains with "
            "train.py --eval-mode unseen-family --unseen-family <family>, saving "
            "checkpoints under "
            "{checkpoint-root}/heldout_{family}_{slug}_finetuned_llm_epoch* "
            "(matching eval/test_eval_matrix.py). Omit or leave empty for the "
            "usual seen-family sweep only (one job per combo)."
        ),
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

    lr_parts = [float(x.strip()) for x in args.learning_rates.split(",") if x.strip()]
    if not lr_parts:
        raise SystemExit("--learning-rates must list at least one value")
    args.learning_rates_tuple = tuple(lr_parts)

    held_parts = [x.strip().lower() for x in args.held_out_families.split(",") if x.strip()]
    valid_fams = {"gcg", "autodan", "pair"}
    for x in held_parts:
        if x not in valid_fams:
            raise SystemExit(
                f"--held-out-families: unknown family {x!r}; expected one of {sorted(valid_fams)}"
            )
    args.held_out_families_tuple = tuple(held_parts)

    if args.time_limit is None:
        total_h = (
            args.hours_per_epoch * args.total_epochs
            + args.eval_hours * len(args.eval_epochs_list)
        )
        args.time_limit = f"{total_h}:00:00"
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    combos = sweep_lr_lambda_epsilon_combos(args.learning_rates_tuple, LAMBDAS, EPSILONS)
    if args.limit is not None:
        combos = combos[: args.limit]

    if args.held_out_families_tuple:
        families: tuple[str | None, ...] = args.held_out_families_tuple
    else:
        families = (None,)

    script_dir = args.script_dir or (Path.cwd() / "sweep_jobs")
    script_dir.mkdir(parents=True, exist_ok=True)

    manifest = script_dir / "sweep_manifest.txt"
    submitted: list[str] = []

    ck_root = args.checkpoint_root.rstrip("/")

    with manifest.open("w", encoding="utf-8") as mf:
        mf.write(
            "# slug\tlr\tlambda\tepsilon\tlm_loss_input\tsystem_prompt_mode\t"
            "held_out_family\twandb_run_id\tfinetuned_base\tjob_script\n"
        )

        for lr, lam, eps in combos:
            base_slug = make_run_slug(lr, lam, eps, lm_loss_input=args.lm_loss_input)
            for fam in families:
                if fam is None:
                    run_slug = base_slug
                    finetuned_base = f"{ck_root}/{base_slug}_finetuned_llm"
                    train_eval_mode = "seen-family"
                    train_unseen: str | None = None
                    held_col = ""
                else:
                    run_slug = f"heldout_{fam}_{base_slug}"
                    finetuned_base = f"{ck_root}/heldout_{fam}_{base_slug}_finetuned_llm"
                    train_eval_mode = "unseen-family"
                    train_unseen = fam
                    held_col = fam

                run_id = uuid.uuid4().hex
                sbatch_job_name = f"sw_{run_slug}"[:64]

                body = render_job_script(
                    run_slug=run_slug,
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
                    lm_loss_input=args.lm_loss_input,
                    eval_epochs=args.eval_epochs_list,
                    train_eval_mode=train_eval_mode,
                    train_unseen_family=train_unseen,
                )

                sh_path = script_dir / f"{run_slug}.sh"
                sh_path.write_text(body, encoding="utf-8")
                os.chmod(sh_path, 0o755)

                mf.write(
                    f"{run_slug}\t{lr}\t{lam}\t{eps}\t{args.lm_loss_input}"
                    f"\t{args.system_prompt_mode}\t{held_col}\t{run_id}\t{finetuned_base}\t{sh_path}\n"
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
                    if args.record_job_ids is not None:
                        args.record_job_ids.parent.mkdir(parents=True, exist_ok=True)
                        with args.record_job_ids.open("a", encoding="utf-8") as jf:
                            jf.write(f"{job_id}\n")

    n_scripts = len(combos) * len(families)
    print(f"Wrote {n_scripts} job scripts to {script_dir}")
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
