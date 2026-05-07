#!/usr/bin/env python3
"""
Run seen-family + unseen-family (held-out) test evaluation for a grid of
hyperparameter checkpoints and training modes.

Checkpoint naming matches train/submit_wandb_sweep.py:

  {checkpoint_root}/{slug}_finetuned_llm_epoch{E}

where slug = make_run_slug(lr, lam, eps, lm_loss_input).

Default grid matches ``submit_wandb_sweep.py`` (``LAMBDAS`` / ``EPSILONS`` there), with
**λ=0** collapsed to a single representative ε when λ=0 is in the λ list (``lambda_epsilon_pairs``),
fixed ``lr=2e-5``, ``epoch=5``.

Default test CSVs under ``--repo-root``: ``official_data/combined_test_dataset.csv`` and
``official_data/frr_test.csv`` (cluster layout: ``$SCRATCH/dp-llm-experiments/official_data/``).

Metrics TSVs record artifact paths **relative to ``--out-dir``** when outputs live there, so
post-hoc plotting (``plot_hyperparameter_heatmaps.py``) can always open CSVs next to the TSVs.

Modes:

  clean_reg — lm_loss_input=clean. λ=0 is vanilla refusal SFT (no stability term);
              λ>0 is your stability-regularized method.

  pert_reg  — lm_loss_input=perturbed. With default ``--perturbed-reg-subset lambda0_only``,
              a **single** λ=0 checkpoint (representative ε, same rule as clean λ=0) matches
              ``--perturbed-sweep-subset lambda0_only`` training. Use ``--perturbed-reg-subset full``
              for the legacy full perturbed λ×ε grid (62 tasks total).

For **clean** LM, when λ=0 only **one** ε is used (``lambda_epsilon_pairs``). The same
representative ε is used for the lone **perturbed** λ=0 run when using ``lambda0_only``.

Seen-family:    eval.py --eval-mode seen-family
Unseen-family:  eval.py --eval-mode unseen-family --unseen-family {gcg|autodan|pair}

Held-out adapters (when not using ``--seen-only``) must live at::

  {checkpoint_root}/heldout_{family}_{slug}_finetuned_llm_epoch{E}

Use the same ``slug`` as the seen sweep (including ``_pertlm`` when LM loss used
perturbed prompts). Train with ``train.py --eval-mode unseen-family``. If your
saved folder names differ, symlink or rename to match.

If you only have seen checkpoints, pass ``--seen-only``.

Usage (Narval):
  # List tasks (0 .. N-1); default N=32 (clean full grid + one perturbed @λ=0)
  python eval/test_eval_matrix.py --list-tasks

  # Dry-run task 0
  python eval/test_eval_matrix.py --task-id 0 --dry-run

  # Run task 0 for real
  python eval/test_eval_matrix.py --task-id 0

  # submit_test_eval_matrix.sh uses SLURM_ARRAY_TASK_ID (array range must match task count)

Outputs under --out-dir:
  per-task metrics TSV + optional master summary rows.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from train.model_profiles import (  # noqa: E402
    DEFAULT_MODEL_PROFILE,
    MODEL_PROFILE_CHOICES,
    make_run_slug,
)
from train.submit_wandb_sweep import lambda_epsilon_pairs  # noqa: E402

# ---------------------------------------------------------------------------
# Slug helper (delegates to train.model_profiles.make_run_slug; must match sweep)
# ---------------------------------------------------------------------------
def expand_path(p: str) -> str:
    """Expand $SCRATCH / $HOME style tokens for defaults."""
    return os.path.expandvars(os.path.expanduser(p))


# ---------------------------------------------------------------------------
# Defaults from your sweep / paper slice (override via CLI)
# ---------------------------------------------------------------------------

DEFAULT_LR = 2e-5
DEFAULT_EPOCH = 5
# Default grid: must match train/submit_wandb_sweep.py LAMBDAS × EPSILONS (+ perturbed task
# when --perturbed-reg-subset lambda0_only). With pipeline defaults: 2 clean_reg + 1 pert_reg = 3 tasks.
# Use --perturbed-reg-subset full for full perturbed grid, or none for clean only.
DEFAULT_LAMBDAS = (0.1,)
DEFAULT_EPSILONS = (-1.0, -0.5)

FAMILIES = ("gcg", "autodan", "pair")

SAFETY_COL = {
    "gcg": "GCG Safety",
    "autodan": "AutoDAN Safety",
    "pair": "PAIR Safety",
}


def resolve_output(stem: Path) -> Path | None:
    p1 = Path(str(stem) + ".csv")
    p2 = Path(str(stem) + ".csv.csv")
    if p1.is_file():
        return p1
    if p2.is_file():
        return p2
    return None


def _artifact_relpath(out_dir: Path, artifact: Path) -> str:
    """Store paths relative to eval out_dir so heatmaps work when METRICS_DIR == out_dir."""
    try:
        return str(artifact.resolve().relative_to(out_dir.resolve()))
    except ValueError:
        return str(artifact)


def prep_benign_csv(src: Path, dst: Path) -> None:
    df = pd.read_csv(src)
    if "Original Prompt" in df.columns:
        df.to_csv(dst, index=False)
        return
    candidates = [
        "adversarial",
        "Adversarial",
        "goal",
        "Goal",
        "prompt",
        "Prompt",
        "original_prompt",
        "instruction",
        "Instruction",
    ]
    src_col = next((c for c in candidates if c in df.columns), None)
    if src_col is None:
        raise ValueError(f"Could not build Original Prompt from columns: {list(df.columns)}")
    df = df.copy()
    df["Original Prompt"] = df[src_col].astype(str)
    df.to_csv(dst, index=False)


def mean_asr_harmful(h: pd.DataFrame) -> float | None:
    cols = [c for c in ("GCG Safety", "AutoDAN Safety", "PAIR Safety") if c in h.columns]
    if not cols:
        return None
    vals = []
    for c in cols:
        vals.append(float((h[c].astype(str).str.lower() == "unsafe").mean()))
    return sum(vals) / len(vals)


def frr_benign(b: pd.DataFrame) -> float | None:
    if "Original Safety" not in b.columns:
        return None
    return float((b["Original Safety"].astype(str).str.lower() == "refusal").mean())


def asr_one_family(h: pd.DataFrame, family: str) -> float:
    col = SAFETY_COL[family]
    if col not in h.columns:
        raise ValueError(f"Expected column {col!r} in harmful CSV; got {list(h.columns)}")
    return float((h[col].astype(str).str.lower() == "unsafe").mean())


def seen_asr_one_family(h: pd.DataFrame, family: str) -> float | None:
    """Single-column ASR for seen-family CSV (same file holds GCG/AutoDAN/PAIR safety)."""
    col = SAFETY_COL[family]
    if col not in h.columns:
        return None
    return float((h[col].astype(str).str.lower() == "unsafe").mean())


@dataclass
class Task:
    task_id: int
    mode: str  # clean_reg | pert_reg
    lr: float
    lam: float
    eps: float
    lm_loss_input: str
    model_profile: str

    def slug(self) -> str:
        return make_run_slug(
            self.lr, self.lam, self.eps, self.lm_loss_input, model_profile=self.model_profile
        )


def build_tasks(
    lr: float,
    lambdas: tuple[float, ...],
    epsilons: tuple[float, ...],
    model_profile: str,
    *,
    perturbed_reg_subset: str = "lambda0_only",
) -> list[Task]:
    """perturbed_reg_subset: 'none' | 'lambda0_only' | 'full' (must match training sweep)."""
    tasks: list[Task] = []
    tid = 0
    pairs = lambda_epsilon_pairs(lambdas, epsilons)

    for lam, eps in pairs:
        tasks.append(Task(tid, "clean_reg", lr, lam, eps, "clean", model_profile))
        tid += 1

    if perturbed_reg_subset == "none":
        return tasks
    if perturbed_reg_subset == "full":
        for lam, eps in pairs:
            tasks.append(Task(tid, "pert_reg", lr, lam, eps, "perturbed", model_profile))
            tid += 1
        return tasks
    if perturbed_reg_subset == "lambda0_only":
        lam, eps = lambda_epsilon_pairs((0.0,), epsilons)[0]
        tasks.append(Task(tid, "pert_reg", lr, lam, eps, "perturbed", model_profile))
        tid += 1
        return tasks
    raise ValueError(f"Unknown perturbed_reg_subset: {perturbed_reg_subset!r}")


def seen_ckpt(root: Path, task: Task, epoch: int) -> Path:
    slug = task.slug()
    return root / f"{slug}_finetuned_llm_epoch{epoch}"


def heldout_ckpt(root: Path, family: str, task: Task, epoch: int) -> Path:
    slug = task.slug()
    # heldout_gcg_run_lr…_finetuned_llm_epoch5
    return root / f"heldout_{family}_{slug}_finetuned_llm_epoch{epoch}"


def run_eval_py(
    *,
    eval_py: Path,
    resume_from: Path,
    eval_mode: str,
    unseen_family: str | None,
    harmful_csv: Path,
    benign_csv: Path,
    harmful_stem: Path,
    benign_stem: Path,
    system_prompt_mode: str,
    model_profile: str,
) -> None:
    cmd = [
        sys.executable,
        str(eval_py),
        "--eval-mode",
        eval_mode,
        "--system-prompt-mode",
        system_prompt_mode,
        "--model-profile",
        model_profile,
        "--resume-from",
        str(resume_from),
        "--validation-data",
        str(harmful_csv),
        "--benign-validation-data",
        str(benign_csv),
        "--harmful-output-file",
        str(harmful_stem),
        "--benign-output-file",
        str(benign_stem),
    ]
    if eval_mode == "unseen-family":
        if not unseen_family:
            raise ValueError("unseen_family required")
        cmd.extend(["--unseen-family", unseen_family])
    print("[test_eval_matrix] Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def write_metrics_tsv(path: Path, rows: list[tuple[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("metric\tvalue\n")
        for k, v in rows:
            f.write(f"{k}\t{v}\n")


def run_one_task(
    task: Task,
    *,
    epoch: int,
    checkpoint_root: Path,
    eval_py: Path,
    harmful_test: Path,
    benign_test: Path,
    out_dir: Path,
    system_prompt_mode: str,
    model_profile: str,
    tmp_benign: Path,
    dry_run: bool,
    skip_missing: bool,
    seen_only: bool,
) -> int:
    slug = task.slug()
    tag = f"{task.mode}_{slug}_ep{epoch}"

    seen_model = seen_ckpt(checkpoint_root, task, epoch)
    if not seen_model.is_dir():
        msg = f"[test_eval_matrix] Missing seen checkpoint: {seen_model}"
        if skip_missing:
            print(msg, "; skip task", task.task_id, flush=True)
            return 0
        raise FileNotFoundError(msg)

    if not seen_only:
        for fam in FAMILIES:
            hp = heldout_ckpt(checkpoint_root, fam, task, epoch)
            if not hp.is_dir():
                msg = f"[test_eval_matrix] Missing held-out checkpoint for {fam}: {hp}"
                if skip_missing:
                    print(msg, "; skip task", task.task_id, flush=True)
                    return 0
                raise FileNotFoundError(msg)

    prep_benign_csv(benign_test, tmp_benign)

    # --- Seen-family ---
    seen_h_stem = out_dir / f"{tag}_seen_harmful"
    seen_b_stem = out_dir / f"{tag}_seen_benign"

    if dry_run:
        print(f"[dry-run] seen-family resume_from={seen_model}", flush=True)
    else:
        run_eval_py(
            eval_py=eval_py,
            resume_from=seen_model,
            eval_mode="seen-family",
            unseen_family=None,
            harmful_csv=harmful_test,
            benign_csv=tmp_benign,
            harmful_stem=seen_h_stem,
            benign_stem=seen_b_stem,
            system_prompt_mode=system_prompt_mode,
            model_profile=model_profile,
        )

    # --- Unseen each family ---
    held_rows: list[tuple[str, Any]] = []
    held_asrs: list[float] = []

    if seen_only and not dry_run:
        print("[test_eval_matrix] --seen-only: skipping held-out eval.", flush=True)

    for fam in FAMILIES:
        if seen_only:
            continue
        h_model = heldout_ckpt(checkpoint_root, fam, task, epoch)
        h_stem = out_dir / f"{tag}_heldout_{fam}_harmful"
        b_stem = out_dir / f"{tag}_heldout_{fam}_benign"

        if dry_run:
            print(f"[dry-run] unseen-family {fam} resume_from={h_model}", flush=True)
            continue

        run_eval_py(
            eval_py=eval_py,
            resume_from=h_model,
            eval_mode="unseen-family",
            unseen_family=fam,
            harmful_csv=harmful_test,
            benign_csv=tmp_benign,
            harmful_stem=h_stem,
            benign_stem=b_stem,
            system_prompt_mode=system_prompt_mode,
            model_profile=model_profile,
        )

        hf = resolve_output(h_stem)
        bf = resolve_output(b_stem)
        if hf is None or bf is None:
            raise FileNotFoundError(f"Expected outputs for {fam}: {h_stem}, {b_stem}")

        hdf = pd.read_csv(hf)
        bdf = pd.read_csv(bf)
        asr_f = asr_one_family(hdf, fam)
        frr_f = frr_benign(bdf)
        held_asrs.append(asr_f)
        held_rows.extend(
            [
                (f"{fam}_heldout_asr", asr_f),
                (f"{fam}_model_frr", frr_f),
                (f"{fam}_harmful_csv", _artifact_relpath(out_dir, hf)),
                (f"{fam}_benign_csv", _artifact_relpath(out_dir, bf)),
            ]
        )

    if not dry_run:
        sh = resolve_output(seen_h_stem)
        sb = resolve_output(seen_b_stem)
        if sh is None or sb is None:
            raise FileNotFoundError(f"Missing seen outputs for {tag}")

        h_seen = pd.read_csv(sh)
        b_seen = pd.read_csv(sb)
        seen_mean = mean_asr_harmful(h_seen)
        seen_frr = frr_benign(b_seen)

        metrics_path = out_dir / f"{tag}_metrics.tsv"
        rows: list[tuple[str, Any]] = [
            ("task_id", task.task_id),
            ("mode", task.mode),
            ("lr", task.lr),
            ("lambda", task.lam),
            ("epsilon", task.eps),
            ("epoch", epoch),
            ("slug", slug),
            ("model_profile", model_profile),
            ("lm_loss_input", task.lm_loss_input),
            ("system_prompt_mode", system_prompt_mode),
            ("seen_mean_asr", seen_mean),
            ("seen_frr", seen_frr),
            ("seen_harmful_csv", _artifact_relpath(out_dir, sh)),
            ("seen_benign_csv", _artifact_relpath(out_dir, sb)),
        ]
        for fam in FAMILIES:
            v = seen_asr_one_family(h_seen, fam)
            rows.append((f"seen_{fam}_asr", v if v is not None else ""))
        if seen_only:
            rows.append(("heldout_mean_asr", "skipped_seen_only"))
        elif held_asrs:
            rows.append(("heldout_mean_asr", sum(held_asrs) / len(held_asrs)))
        rows.extend(held_rows)

        write_metrics_tsv(metrics_path, rows)
        print(f"[test_eval_matrix] Wrote {metrics_path}", flush=True)

    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    scr = os.environ.get("SCRATCH", "")
    default_repo = f"{scr}/dp-llm-experiments" if scr else ""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task-id", type=int, default=None, help="Single task index (or SLURM_ARRAY_TASK_ID).")
    p.add_argument("--list-tasks", action="store_true", help="Print task count and exit.")
    p.add_argument("--dry-run", action="store_true", help="Print what would run; do not call eval.py.")
    p.add_argument(
        "--seen-only",
        action="store_true",
        help="Only run seen-family eval (skip unseen/held-out; only require seen checkpoint).",
    )
    p.add_argument(
        "--perturbed-reg-subset",
        dest="perturbed_reg_subset",
        default="lambda0_only",
        choices=("none", "lambda0_only", "full"),
        help=(
            "Which perturbed-LM checkpoints to evaluate. 'lambda0_only' (default): one "
            "task at λ=0 with the same representative ε as clean λ=0 (when λ=0 is in the λ list). "
            "'full': one perturbed task per (λ, ε) clean cell. 'none': clean grid only."
        ),
    )
    p.add_argument("--skip-missing", action="store_true", default=True, help="Skip task if any checkpoint is missing (default: on).")
    p.add_argument("--no-skip-missing", action="store_false", dest="skip_missing", help="Fail if a checkpoint is missing.")

    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--epoch", type=int, default=DEFAULT_EPOCH)
    p.add_argument(
        "--lambdas",
        type=str,
        default=",".join(str(x) for x in DEFAULT_LAMBDAS),
        help="Comma-separated λ grid (must match training sweep).",
    )
    p.add_argument(
        "--epsilons",
        type=str,
        default=",".join(str(x) for x in DEFAULT_EPSILONS),
        help="Comma-separated ε grid (must match training sweep).",
    )

    p.add_argument(
        "--checkpoint-root",
        type=str,
        default="$SCRATCH/dp-llm-sweep",
        help="Directory containing {slug}_finetuned_llm_epoch* and heldout_* checkpoints.",
    )
    p.add_argument(
        "--eval-py",
        type=str,
        default="",
        help="Path to eval.py (default: <repo>/eval/eval.py).",
    )
    p.add_argument(
        "--repo-root",
        type=str,
        default=default_repo or ".",
        help="Repo root on cluster (for default eval.py path).",
    )
    p.add_argument(
        "--harmful-test",
        type=str,
        default="",
        help="combined_test_dataset.csv (harmful / ASR).",
    )
    p.add_argument(
        "--benign-test",
        type=str,
        default="",
        help="frr_test.csv (benign / FRR).",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Eval outputs + metrics (default: <checkpoint-root>/test_eval_outputs).",
    )
    p.add_argument(
        "--system-prompt-mode",
        choices=("default", "empty"),
        default="empty",
        help="Must match training/eval protocol (default empty = nosys, same as your sweep).",
    )
    p.add_argument(
        "--model-profile",
        default=os.environ.get("MODEL_PROFILE", DEFAULT_MODEL_PROFILE),
        choices=list(MODEL_PROFILE_CHOICES),
        help="Must match train/submit_wandb_sweep.py --model-profile for checkpoint slugs.",
    )
    args = p.parse_args(argv)

    repo = expand_path(args.repo_root)
    if not args.eval_py:
        args.eval_py = str(Path(repo) / "eval" / "eval.py")
    if not args.harmful_test:
        args.harmful_test = str(Path(repo) / "official_data" / "combined_test_dataset.csv")
    if not args.benign_test:
        args.benign_test = str(Path(repo) / "official_data" / "frr_test.csv")
    if not args.out_dir:
        args.out_dir = str(Path(expand_path(args.checkpoint_root)) / "test_eval_outputs")

    lam_parts = [float(x.strip()) for x in args.lambdas.split(",") if x.strip()]
    eps_parts = [float(x.strip()) for x in args.epsilons.split(",") if x.strip()]
    args.lambdas_tuple = tuple(lam_parts)
    args.epsilons_tuple = tuple(eps_parts)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    tasks = build_tasks(
        args.lr,
        args.lambdas_tuple,
        args.epsilons_tuple,
        args.model_profile,
        perturbed_reg_subset=args.perturbed_reg_subset,
    )

    if args.list_tasks:
        print(f"task_count={len(tasks)}")
        for t in tasks:
            slug = t.slug()
            print(f"  {t.task_id}\t{t.mode}\tprofile={t.model_profile}\tlam={t.lam}\teps={t.eps}\t{slug}")
        return 0

    # SLURM array index or explicit --task-id
    if args.task_id is None:
        if os.environ.get("SLURM_ARRAY_TASK_ID") is not None:
            args.task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        else:
            print(
                "ERROR: pass --task-id N, --list-tasks, or run under SLURM with SLURM_ARRAY_TASK_ID",
                file=sys.stderr,
            )
            return 2

    if args.task_id < 0 or args.task_id >= len(tasks):
        print(f"ERROR: task_id {args.task_id} out of range [0, {len(tasks) - 1}]", file=sys.stderr)
        return 2

    task = tasks[args.task_id]
    ck_root = Path(expand_path(args.checkpoint_root))
    eval_py = Path(expand_path(args.eval_py))
    harmful_test = Path(expand_path(args.harmful_test))
    benign_test = Path(expand_path(args.benign_test))
    out_dir = Path(expand_path(args.out_dir))

    tmp_benign = Path(os.environ.get("SLURM_TMPDIR", "/tmp")) / f"frr_test_eval_task{args.task_id}.csv"

    return run_one_task(
        task,
        epoch=args.epoch,
        checkpoint_root=ck_root,
        eval_py=eval_py,
        harmful_test=harmful_test,
        benign_test=benign_test,
        out_dir=out_dir,
        system_prompt_mode=args.system_prompt_mode,
        model_profile=args.model_profile,
        tmp_benign=tmp_benign,
        dry_run=args.dry_run,
        skip_missing=args.skip_missing,
        seen_only=args.seen_only,
    )


if __name__ == "__main__":
    sys.exit(main())
