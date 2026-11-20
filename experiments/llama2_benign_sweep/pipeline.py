#!/usr/bin/env python3
"""Create manifests and submit the Llama-2 validation/final pipelines."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .configs import SweepConfig, get_config, sweep_configs
from train.model_profiles import resolve_profile


REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
TRAIN_MODES: tuple[tuple[str, str], ...] = (
    ("seen", ""),
    ("heldout_gcg", "gcg"),
    ("heldout_autodan", "autodan"),
    ("heldout_pair", "pair"),
)
FAMILIES = ("gcg", "autodan", "pair")


def expand_path(raw: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(raw))).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_provenance() -> dict[str, object]:
    result: dict[str, object] = {"commit": "unknown", "tracked_dirty": None}
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    if revision.returncode == 0:
        result["commit"] = revision.stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    if status.returncode == 0:
        result["tracked_dirty"] = bool(status.stdout.strip())
    return result


def parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            value = int(token)
            if value not in values:
                values.append(value)
    if not values:
        raise ValueError("expected at least one integer")
    return values


def write_tsv(path: Path, rows: list[dict], *, replace: bool) -> None:
    if not rows:
        raise ValueError("cannot write an empty manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, dialect="excel-tab", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    content = buffer.getvalue()
    if path.exists():
        previous = path.read_text(encoding="utf-8")
        if previous != content and not replace:
            raise FileExistsError(
                f"Refusing to replace a different manifest: {path}. "
                "Use --replace-manifests only when intentionally starting a new run."
            )
        if previous == content:
            return
    path.write_text(content, encoding="utf-8")


def build_train_rows(
    *,
    phase: str,
    configs: list[SweepConfig],
    seeds: list[int],
    run_root: Path,
    training_data: Path,
    benign_training_data: Path,
) -> list[dict]:
    rows: list[dict] = []
    for config in configs:
        for seed in seeds:
            for train_mode, heldout_family in TRAIN_MODES:
                checkpoint_prefix = (
                    run_root
                    / "checkpoints"
                    / phase
                    / config.slug
                    / f"seed{seed}"
                    / train_mode
                    / "adapter"
                )
                rows.append(
                    {
                        "task_id": len(rows),
                        "phase": phase,
                        "config_id": config.config_id,
                        "seed": seed,
                        "train_mode": train_mode,
                        "heldout_family": heldout_family,
                        "learning_rate": f"{config.learning_rate:.12g}",
                        "lambda_val": f"{config.lambda_val:.12g}",
                        "epsilon": f"{config.epsilon:.12g}",
                        "benign_lm_weight": f"{config.benign_lm_weight:.12g}",
                        "rollout_length": config.rollout_length,
                        "stability_loss": config.stability_loss,
                        "training_data": str(training_data),
                        "benign_training_data": str(benign_training_data),
                        "checkpoint_prefix": str(checkpoint_prefix),
                    }
                )
    return rows


def build_eval_rows(
    *,
    train_rows: list[dict],
    run_root: Path,
    harmful_eval_data: Path,
    benign_eval_data: Path,
    final_checkpoint_step: int,
    asr_shards: int,
    frr_shards: int,
) -> list[dict]:
    rows: list[dict] = []
    for train in train_rows:
        phase = train["phase"]
        config_id = int(train["config_id"])
        seed = int(train["seed"])
        train_mode = train["train_mode"]
        families = FAMILIES if train_mode == "seen" else (train["heldout_family"],)
        checkpoint = f"{train['checkpoint_prefix']}_step{final_checkpoint_step}"
        output_dir = (
            run_root
            / "eval_outputs"
            / phase
            / f"cfg{config_id:03d}"
            / f"seed{seed}"
            / train_mode
        )
        common = {
            "phase": phase,
            "config_id": config_id,
            "seed": seed,
            "train_mode": train_mode,
            "heldout_family": train["heldout_family"],
            "learning_rate": train["learning_rate"],
            "lambda_val": train["lambda_val"],
            "epsilon": train["epsilon"],
            "benign_lm_weight": train["benign_lm_weight"],
            "checkpoint_path": checkpoint,
        }
        for family in families:
            for shard_index in range(asr_shards):
                rows.append(
                    {
                        "eval_task_id": len(rows),
                        **common,
                        "eval_kind": "asr",
                        "eval_family": family,
                        "shard_index": shard_index,
                        "shard_count": asr_shards,
                        "source_csv": str(harmful_eval_data),
                        "output_csv": str(output_dir / f"asr_{family}_shard{shard_index}.csv"),
                    }
                )
        for shard_index in range(frr_shards):
            rows.append(
                {
                    "eval_task_id": len(rows),
                    **common,
                    "eval_kind": "frr",
                    "eval_family": "",
                    "shard_index": shard_index,
                    "shard_count": frr_shards,
                    "source_csv": str(benign_eval_data),
                    "output_csv": str(output_dir / f"frr_shard{shard_index}.csv"),
                }
            )
    return rows


def validate_inputs(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit("Missing required input files:\n  " + "\n  ".join(missing))


def sbatch(
    command: list[str],
    *,
    dry_run: bool,
    placeholder: str,
) -> tuple[str, str]:
    printable = shlex.join(command)
    print(printable, flush=True)
    if dry_run:
        return placeholder, printable
    proc = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    if proc.stdout:
        print(proc.stdout, end="", flush=True)
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr, flush=True)
    if proc.returncode != 0:
        raise SystemExit(f"sbatch failed with exit code {proc.returncode}")
    job_id = proc.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise SystemExit(f"Could not parse sbatch job id from: {proc.stdout!r}")
    return job_id, printable


def export_arg(**values: object) -> str:
    items = ["ALL"]
    for key, value in values.items():
        rendered = str(value)
        if "," in rendered:
            raise ValueError(f"SLURM --export value cannot contain a comma: {key}={rendered}")
        items.append(f"{key}={rendered}")
    return ",".join(items)


def load_state(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def event_already_submitted(
    state_path: Path,
    *,
    wave: int,
    action: str,
    training_epoch: int | None = None,
) -> bool:
    """Return whether this continuation was recorded as fully submitted.

    Controller jobs are deliberately tiny, but Slurm can requeue a batch job.
    Treating an already-recorded continuation as a no-op prevents a requeued
    controller from duplicating hundreds of array tasks.
    """
    events_path = state_path.with_name("submission_events.jsonl")
    if not events_path.is_file():
        return False
    expected = f"{action}_wave_submitted"
    for line in events_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        same_epoch = (
            action != "train"
            or int(event.get("training_epoch", -1)) == int(training_epoch)
        )
        if (
            event.get("event") == expected
            and int(event.get("wave", -1)) == wave
            and same_epoch
        ):
            return True
    return False


def append_event(state_path: Path, event: dict) -> None:
    event = {"at_utc": datetime.now(timezone.utc).isoformat(), **event}
    events_path = state_path.with_name("submission_events.jsonl")
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def submit_controller(
    *,
    state_path: Path,
    wave: int,
    action: str,
    training_epoch: int,
    dependency_ids: list[str],
    dry_run: bool,
) -> tuple[str, str]:
    command = ["sbatch", "--parsable"]
    if dependency_ids:
        command.append(f"--dependency=afterok:{':'.join(dependency_ids)}")
    command.extend(
        [
            "--export="
            + export_arg(
                REPO_ROOT=REPO_ROOT,
                STATE=state_path,
                WAVE=wave,
                ACTION=action,
                TRAINING_EPOCH=training_epoch,
            ),
            str(HERE / "controller_job.sh"),
        ]
    )
    return sbatch(
        command,
        dry_run=dry_run,
        placeholder=f"DRY_CONTROL_{action}_{wave}",
    )


def submit_train_wave(
    state_path: Path,
    wave_index: int,
    training_epoch: int,
    *,
    dry_run: bool,
) -> dict:
    state = load_state(state_path)
    wave = state["waves"][wave_index]
    if not 0 <= training_epoch < int(state["epochs"]):
        raise ValueError(
            f"training_epoch must be in 0..{int(state['epochs']) - 1}, "
            f"got {training_epoch}"
        )
    commands: list[str] = []
    previous: str | None = None
    stage_jobs: list[str] = []
    for stage in range(int(state["stages"])):
        cmd = [
            "sbatch",
            "--parsable",
            f"--array=0-{int(wave['train_count']) - 1}%{int(state['train_concurrency'])}",
        ]
        if previous is not None:
            cmd.append(f"--dependency=aftercorr:{previous}")
        cmd.extend(
            [
                "--export="
                + export_arg(
                    REPO_ROOT=REPO_ROOT,
                    MANIFEST=state["train_manifest"],
                    TASK_OFFSET=wave["train_offset"],
                    EPOCH=training_epoch,
                    EPOCHS=state["epochs"],
                    STAGE=stage,
                    STAGES=state["stages"],
                ),
                str(HERE / "train_array.sh"),
            ]
        )
        job_id, rendered = sbatch(
            cmd,
            dry_run=dry_run,
            placeholder=f"DRY_TRAIN_{stage}",
        )
        commands.append(rendered)
        stage_jobs.append(job_id)
        previous = job_id
    if training_epoch + 1 < int(state["epochs"]):
        next_action = "train"
        next_wave = wave_index
        next_training_epoch = training_epoch + 1
    elif state["skip_eval"]:
        next_action = "train"
        next_wave = wave_index + 1
        next_training_epoch = 0
    else:
        next_action = "eval"
        next_wave = wave_index
        next_training_epoch = 0
    controller_id = None
    if next_wave < len(state["waves"]):
        controller_id, rendered = submit_controller(
            state_path=state_path,
            wave=next_wave,
            action=next_action,
            training_epoch=next_training_epoch,
            dependency_ids=[previous] if previous else [],
            dry_run=dry_run,
        )
        commands.append(rendered)
    result = {
        "event": "train_wave_submitted",
        "wave": wave_index,
        "training_epoch": training_epoch,
        "dry_run": dry_run,
        "train_stage_job_ids": stage_jobs,
        "controller_job_id": controller_id,
        "commands": commands,
    }
    if not dry_run:
        append_event(state_path, result)
    return result


def submit_eval_wave(state_path: Path, wave_index: int, *, dry_run: bool) -> dict:
    state = load_state(state_path)
    wave = state["waves"][wave_index]
    commands: list[str] = []
    eval_jobs: list[str] = []
    start = int(wave["eval_offset"])
    stop = start + int(wave["eval_count"])
    max_array_size = int(state["max_array_size"])
    for offset in range(start, stop, max_array_size):
        chunk_size = min(max_array_size, stop - offset)
        cmd = [
            "sbatch",
            "--parsable",
            f"--array=0-{chunk_size - 1}%{int(state['eval_concurrency'])}",
            "--export="
            + export_arg(
                REPO_ROOT=REPO_ROOT,
                MANIFEST=state["eval_manifest"],
                TASK_OFFSET=offset,
            ),
            str(HERE / "eval_array.sh"),
        ]
        eval_job, rendered = sbatch(
            cmd,
            dry_run=dry_run,
            placeholder=f"DRY_EVAL_{wave_index}_{offset}",
        )
        eval_jobs.append(eval_job)
        commands.append(rendered)

    next_controller = None
    report_job = None
    if wave_index + 1 < len(state["waves"]):
        next_controller, rendered = submit_controller(
            state_path=state_path,
            wave=wave_index + 1,
            action="train",
            training_epoch=0,
            dependency_ids=eval_jobs,
            dry_run=dry_run,
        )
        commands.append(rendered)
    elif not state["skip_report"]:
        cmd = [
            "sbatch",
            "--parsable",
            f"--dependency=afterok:{':'.join(eval_jobs)}",
            "--export="
            + export_arg(
                REPO_ROOT=REPO_ROOT,
                MANIFEST=state["eval_manifest"],
                REPORT_DIR=state["report_dir"],
            ),
            str(HERE / "report_job.sh"),
        ]
        report_job, rendered = sbatch(cmd, dry_run=dry_run, placeholder="DRY_REPORT")
        commands.append(rendered)
    result = {
        "event": "eval_wave_submitted",
        "wave": wave_index,
        "dry_run": dry_run,
        "eval_job_ids": eval_jobs,
        "next_controller_job_id": next_controller,
        "report_job_id": report_job,
        "commands": commands,
    }
    if not dry_run:
        append_event(state_path, result)
    return result


def run(args: argparse.Namespace) -> None:
    run_root = expand_path(args.run_root)
    repo_data = expand_path(args.data_root)
    if args.command == "submit-sweep":
        config_ids = parse_int_list(args.config_ids) if args.config_ids else list(range(100))
        configs = [get_config(config_id) for config_id in config_ids]
        seeds = [0]
        phase = "sweep"
        training_data = expand_path(args.training_data or str(repo_data / "train.csv"))
        benign_training_data = expand_path(
            args.benign_training_data or str(repo_data / "nectar_benign_train.csv")
        )
        harmful_eval_data = expand_path(
            args.harmful_eval_data or str(repo_data / "validation.csv")
        )
        benign_eval_data = expand_path(
            args.benign_eval_data or str(repo_data / "frr_validation.csv")
        )
        asr_shards = args.asr_shards if args.asr_shards is not None else 2
        frr_shards = args.frr_shards if args.frr_shards is not None else 2
    else:
        config = get_config(args.config_id)
        configs = [config]
        seeds = parse_int_list(args.seeds)
        if len(seeds) != 3:
            raise SystemExit("Final evaluation requires exactly three distinct seeds")
        phase = f"final_cfg{config.config_id:03d}"
        training_data = expand_path(
            args.training_data or str(repo_data / "train_plus_validation.csv")
        )
        benign_training_data = expand_path(
            args.benign_training_data
            or str(repo_data / "nectar_benign_train_plus_validation.csv")
        )
        harmful_eval_data = expand_path(
            args.harmful_eval_data or str(repo_data / "combined_test_dataset.csv")
        )
        benign_eval_data = expand_path(
            args.benign_eval_data or str(repo_data / "frr_test.csv")
        )
        asr_shards = args.asr_shards if args.asr_shards is not None else 2
        frr_shards = args.frr_shards if args.frr_shards is not None else 1

    if args.epochs < 1:
        raise SystemExit("--epochs must be >=1")
    if args.stages < 2:
        raise SystemExit("--stages must be >=2; row shards enforce the sub-hour limit")
    if asr_shards < 1:
        raise SystemExit("--asr-shards must be >=1")
    if frr_shards < 1:
        raise SystemExit("--frr-shards must be >=1")
    if args.max_array_size < 1:
        raise SystemExit("--max-array-size must be >=1")
    if args.wave_size < 1:
        raise SystemExit("--wave-size must be >=1")
    if args.max_pending_jobs < 1:
        raise SystemExit("--max-pending-jobs must be >=1")
    if args.train_concurrency < 1:
        raise SystemExit("--train-concurrency must be >=1")
    if args.eval_concurrency < 1:
        raise SystemExit("--eval-concurrency must be >=1")
    validate_inputs([training_data, benign_training_data, harmful_eval_data, benign_eval_data])

    train_rows = build_train_rows(
        phase=phase,
        configs=configs,
        seeds=seeds,
        run_root=run_root,
        training_data=training_data,
        benign_training_data=benign_training_data,
    )
    eval_rows = build_eval_rows(
        train_rows=train_rows,
        run_root=run_root,
        harmful_eval_data=harmful_eval_data,
        benign_eval_data=benign_eval_data,
        final_checkpoint_step=args.epochs * args.stages,
        asr_shards=asr_shards,
        frr_shards=frr_shards,
    )
    manifest_dir = run_root / "manifests" / phase
    train_manifest = manifest_dir / "train.tsv"
    eval_manifest = manifest_dir / "eval.tsv"
    config_manifest = manifest_dir / "configs.tsv"
    write_tsv(train_manifest, train_rows, replace=args.replace_manifests)
    write_tsv(eval_manifest, eval_rows, replace=args.replace_manifests)
    write_tsv(
        config_manifest,
        [config.as_dict() for config in configs],
        replace=args.replace_manifests,
    )

    metadata = {
        "phase": phase,
        "configs": [config.as_dict() for config in configs],
        "seeds": seeds,
        "epochs": args.epochs,
        "stages": args.stages,
        "training_steps": args.epochs * args.stages,
        "asr_shards": asr_shards,
        "frr_shards": frr_shards,
        "train_tasks": len(train_rows),
        "eval_tasks": len(eval_rows),
        "code": git_provenance(),
        "model_profile": {
            "name": "llama_2_7b_chat",
            "base_llm": os.path.expandvars(resolve_profile("llama_2_7b_chat").base_llm),
            "hinge_guard": os.path.expandvars(
                resolve_profile("llama_2_7b_chat").hinge_guard_path
            ),
            "refusal_judge": os.path.expandvars(
                resolve_profile("llama_2_7b_chat").eval_refusal_judge_path
            ),
            "jailbreak_classifier": os.path.expandvars(
                resolve_profile("llama_2_7b_chat").eval_jailbreak_classifier_path
            ),
        },
        "inputs": {
            str(path): sha256_file(path)
            for path in (training_data, benign_training_data, harmful_eval_data, benign_eval_data)
        },
    }
    metadata_path = manifest_dir / "metadata.json"
    metadata_json = json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    if metadata_path.exists() and metadata_path.read_text(encoding="utf-8") != metadata_json:
        if not args.replace_manifests:
            raise SystemExit(f"Metadata differs from existing run: {metadata_path}")
    metadata_path.write_text(metadata_json, encoding="utf-8")

    train_per_config = len(train_rows) // len(configs)
    eval_per_config = len(eval_rows) // len(configs)
    wave_size = min(args.wave_size, len(configs))
    if wave_size < 1:
        raise SystemExit("--wave-size must be >=1")
    max_train_pending = wave_size * train_per_config * args.stages + 1
    max_eval_pending = wave_size * eval_per_config + 1
    if max(max_train_pending, max_eval_pending) > args.max_pending_jobs:
        raise SystemExit(
            f"--wave-size {wave_size} can create up to "
            f"{max(max_train_pending, max_eval_pending)} pending/running tasks, above "
            f"--max-pending-jobs {args.max_pending_jobs}. Reduce --wave-size."
        )
    waves: list[dict] = []
    for start_config in range(0, len(configs), wave_size):
        count = min(wave_size, len(configs) - start_config)
        waves.append(
            {
                "wave": len(waves),
                "config_ids": [
                    config.config_id
                    for config in configs[start_config : start_config + count]
                ],
                "train_offset": start_config * train_per_config,
                "train_count": count * train_per_config,
                "eval_offset": start_config * eval_per_config,
                "eval_count": count * eval_per_config,
            }
        )

    report_dir = run_root / "reports" / phase
    state = {
        "schema_version": 2,
        "phase": phase,
        "repo_root": str(REPO_ROOT),
        "train_manifest": str(train_manifest),
        "eval_manifest": str(eval_manifest),
        "report_dir": str(report_dir),
        "epochs": args.epochs,
        "stages": args.stages,
        "train_concurrency": args.train_concurrency,
        "eval_concurrency": args.eval_concurrency,
        "max_array_size": args.max_array_size,
        "max_pending_jobs": args.max_pending_jobs,
        "skip_eval": args.skip_eval,
        "skip_report": args.skip_report,
        "waves": waves,
    }
    state_path = manifest_dir / "pipeline_state.json"
    state_json = json.dumps(state, indent=2, sort_keys=True) + "\n"
    if state_path.exists() and state_path.read_text(encoding="utf-8") != state_json:
        if not args.replace_manifests:
            raise SystemExit(f"Pipeline state differs from existing run: {state_path}")
    state_path.write_text(state_json, encoding="utf-8")

    manifest_dir.mkdir(parents=True, exist_ok=True)
    submission_path = manifest_dir / "submission.json"
    if submission_path.exists() and not args.dry_run and not args.resubmit:
        prior = json.loads(submission_path.read_text(encoding="utf-8"))
        if not prior.get("dry_run", False):
            raise SystemExit(
                f"This phase already has a real submission record: {submission_path}. "
                "Use --resubmit only after checking squeue/sacct and existing outputs."
            )
    # Slurm resolves #SBATCH --output before the batch script starts, so this
    # directory must already exist when the first job is submitted.
    (REPO_ROOT / "output").mkdir(parents=True, exist_ok=True)
    submission = submit_train_wave(
        state_path,
        0,
        training_epoch=0,
        dry_run=args.dry_run,
    )
    submission.update(
        {
            "phase": phase,
            "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
            "state": str(state_path),
            "wave_count": len(waves),
        }
    )
    submission_path.write_text(
        json.dumps(submission, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Training manifest: {train_manifest} ({len(train_rows)} tasks per stage)")
    print(f"Evaluation manifest: {eval_manifest} ({len(eval_rows)} tasks)")
    print(
        f"Waves: {len(waves)} (up to {wave_size} configs each; "
        f"worst-case pending/running tasks={max(max_train_pending, max_eval_pending)})"
    )
    print(f"Pipeline state: {state_path}")
    print(f"Submission record: {submission_path}")
    print(f"Report directory: {report_dir}")


def continue_wave(args: argparse.Namespace) -> None:
    """Internal entry point used by the small dependent controller jobs."""
    state_path = expand_path(args.state)
    if not state_path.is_file():
        raise SystemExit(f"Pipeline state does not exist: {state_path}")
    state = load_state(state_path)
    waves = state.get("waves", [])
    if not 0 <= args.wave < len(waves):
        raise SystemExit(f"--wave must be in 0..{len(waves) - 1}, got {args.wave}")
    if args.action == "eval" and state.get("skip_eval", False):
        raise SystemExit("This pipeline was created with --skip-eval")
    if event_already_submitted(
        state_path,
        wave=args.wave,
        action=args.action,
        training_epoch=args.training_epoch,
    ) and not args.resubmit:
        print(
            f"Wave {args.wave} action {args.action!r} is already recorded as submitted; "
            "leaving the pipeline unchanged. Use --resubmit only after checking sacct."
        )
        return

    (REPO_ROOT / "output").mkdir(parents=True, exist_ok=True)
    if args.action == "train":
        if not 0 <= args.training_epoch < int(state["epochs"]):
            raise SystemExit(
                f"--training-epoch must be in 0..{int(state['epochs']) - 1}, "
                f"got {args.training_epoch}"
            )
        result = submit_train_wave(
            state_path,
            args.wave,
            training_epoch=args.training_epoch,
            dry_run=args.dry_run,
        )
    else:
        if args.training_epoch != 0:
            raise SystemExit("--training-epoch only applies to --action train")
        result = submit_eval_wave(state_path, args.wave, dry_run=args.dry_run)
    print(json.dumps(result, indent=2, sort_keys=True))


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-root", default="$SCRATCH/dcl_l2_benign_sweep")
    parser.add_argument("--data-root", default="$SCRATCH/dp-llm-experiments/official_data")
    parser.add_argument("--training-data")
    parser.add_argument("--benign-training-data")
    parser.add_argument("--harmful-eval-data")
    parser.add_argument("--benign-eval-data")
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Full passes over the harmful and paired benign training sets.",
    )
    parser.add_argument(
        "--stages",
        type=int,
        default=8,
        help="Sequential row shards per epoch, each submitted as a sub-hour job.",
    )
    parser.add_argument("--asr-shards", type=int)
    parser.add_argument("--frr-shards", type=int)
    parser.add_argument("--train-concurrency", type=int, default=32)
    parser.add_argument("--eval-concurrency", type=int, default=32)
    parser.add_argument(
        "--max-array-size",
        type=int,
        default=1000,
        help="Split evaluation manifests across arrays no larger than this.",
    )
    parser.add_argument(
        "--wave-size",
        type=int,
        default=20,
        help=(
            "Configurations submitted at once. The default keeps all staged "
            "array elements below Narval's per-user pending/running job cap."
        ),
    )
    parser.add_argument(
        "--max-pending-jobs",
        type=int,
        default=1000,
        help="Refuse a wave design that can exceed this many submitted tasks.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-report", action="store_true")
    parser.add_argument("--replace-manifests", action="store_true")
    parser.add_argument(
        "--resubmit",
        action="store_true",
        help="Allow a new initial submission for an existing recorded phase.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sweep = sub.add_parser("submit-sweep")
    add_common(sweep)
    sweep.add_argument(
        "--config-ids",
        help="Comma-separated subset for a smoke test; default submits all 0..99.",
    )
    final = sub.add_parser("submit-final")
    add_common(final)
    final.add_argument("--config-id", type=int, required=True)
    final.add_argument("--seeds", default="0,1,2")
    continuation = sub.add_parser(
        "continue-wave",
        help=argparse.SUPPRESS,
    )
    continuation.add_argument("--state", required=True)
    continuation.add_argument("--wave", type=int, required=True)
    continuation.add_argument("--action", choices=["train", "eval"], required=True)
    continuation.add_argument(
        "--training-epoch",
        type=int,
        default=0,
        help="Zero-based training epoch for an internal train continuation.",
    )
    continuation.add_argument("--dry-run", action="store_true")
    continuation.add_argument(
        "--resubmit",
        action="store_true",
        help="Resubmit an already-recorded continuation after checking sacct.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "continue-wave":
        continue_wave(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
