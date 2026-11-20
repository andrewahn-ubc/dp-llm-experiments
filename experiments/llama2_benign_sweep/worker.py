#!/usr/bin/env python3
"""SLURM array worker for staged training and sharded evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
VARIANT_COLUMNS = {
    "gcg": "GCG Variant",
    "autodan": "AutoDAN Variant",
    "pair": "PAIR Variant",
}
SAFETY_COLUMNS = {
    "gcg": "GCG Safety",
    "autodan": "AutoDAN Safety",
    "pair": "PAIR Safety",
}
CHECKPOINT_MARKER = ".dcl_checkpoint_complete.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_eval_output(
    path: Path,
    *,
    expected_rows: int,
    eval_kind: str,
    eval_family: str,
) -> dict[str, int]:
    frame = pd.read_csv(path)
    if len(frame) != expected_rows:
        raise ValueError(f"{path} has {len(frame)} rows; expected {expected_rows}")
    if eval_kind == "asr":
        label_col = SAFETY_COLUMNS[eval_family]
        allowed = {"safe", "unsafe"}
    else:
        label_col = "Original Safety"
        allowed = {"refusal", "non-refusal"}
    if label_col not in frame.columns:
        raise ValueError(f"{path} is missing required label column {label_col!r}")
    labels = frame[label_col].astype("string").str.casefold()
    invalid = labels.isna() | ~labels.isin(allowed)
    if invalid.any():
        values = sorted(set(labels[invalid].dropna().astype(str)))
        raise ValueError(f"{path} has {int(invalid.sum())} invalid labels: {values}")
    return {str(label): int(count) for label, count in labels.value_counts().items()}


def completion_marker_valid(
    marker_path: Path,
    output_path: Path,
    *,
    expected_rows: int,
    eval_kind: str,
    eval_family: str,
) -> bool:
    if not marker_path.is_file() or not output_path.is_file():
        return False
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        counts = validate_eval_output(
            output_path,
            expected_rows=expected_rows,
            eval_kind=eval_kind,
            eval_family=eval_family,
        )
        return (
            int(marker["rows"]) == expected_rows
            and marker["sha256"] == sha256_file(output_path)
            and marker["label_counts"] == counts
        )
    except (KeyError, TypeError, ValueError, OSError, pd.errors.ParserError):
        return False


def read_manifest_row(path: Path, task_id: int, id_column: str) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, dialect="excel-tab"))
    if not 0 <= task_id < len(rows):
        raise SystemExit(f"task id {task_id} outside manifest range 0..{len(rows)-1}")
    row = rows[task_id]
    if int(row[id_column]) != task_id:
        raise SystemExit(
            f"manifest row mismatch: position={task_id}, {id_column}={row[id_column]}"
        )
    return row


def checkpoint_artifacts(path: Path) -> dict[str, Path] | None:
    model_file = path / "adapter_model.safetensors"
    if not model_file.is_file():
        model_file = path / "adapter_model.bin"
    artifacts = {
        "adapter_config.json": path / "adapter_config.json",
        model_file.name: model_file,
        "optimizer.pt": path / "optimizer.pt",
    }
    if not path.is_dir() or any(
        not artifact.is_file() or artifact.stat().st_size == 0
        for artifact in artifacts.values()
    ):
        return None
    return artifacts


def checkpoint_metadata(path: Path) -> dict | None:
    artifacts = checkpoint_artifacts(path)
    if artifacts is None:
        return None
    return {
        "schema_version": 1,
        "files": {
            name: {"bytes": file.stat().st_size, "sha256": sha256_file(file)}
            for name, file in artifacts.items()
        },
    }


def validate_loadable_checkpoint(path: Path) -> dict:
    """Load checkpoint artifacts once before publishing the success marker."""

    artifacts = checkpoint_artifacts(path)
    if artifacts is None:
        raise ValueError(f"missing or empty checkpoint artifacts in {path}")
    config = json.loads(artifacts["adapter_config.json"].read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"adapter_config.json is not an object in {path}")

    model_name = next(
        name for name in artifacts if name in {"adapter_model.safetensors", "adapter_model.bin"}
    )
    model_path = artifacts[model_name]
    if model_name.endswith(".safetensors"):
        from safetensors.torch import load_file

        model_state = load_file(str(model_path), device="cpu")
    else:
        import torch

        try:
            model_state = torch.load(model_path, map_location="cpu", weights_only=True)
        except TypeError:  # PyTorch versions before weights_only
            model_state = torch.load(model_path, map_location="cpu")
    if not isinstance(model_state, dict) or not model_state:
        raise ValueError(f"adapter model state is empty or invalid in {path}")

    import torch

    try:
        optimizer_state = torch.load(
            artifacts["optimizer.pt"], map_location="cpu", weights_only=True
        )
    except TypeError:  # PyTorch versions before weights_only
        optimizer_state = torch.load(artifacts["optimizer.pt"], map_location="cpu")
    if not isinstance(optimizer_state, dict) or not {
        "state",
        "param_groups",
    }.issubset(optimizer_state):
        raise ValueError(f"optimizer state is invalid in {path}")
    metadata = checkpoint_metadata(path)
    if metadata is None:  # guarded above; keeps the return type explicit
        raise ValueError(f"checkpoint artifacts disappeared while validating {path}")
    return metadata


def write_checkpoint_marker(path: Path) -> None:
    metadata = validate_loadable_checkpoint(path)
    marker_path = path / CHECKPOINT_MARKER
    marker_tmp = path / f"{CHECKPOINT_MARKER}.tmp.{os.getpid()}"
    marker_tmp.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(marker_tmp, marker_path)


def checkpoint_complete(path: Path) -> bool:
    marker_path = path / CHECKPOINT_MARKER
    if not marker_path.is_file():
        return False
    try:
        recorded = json.loads(marker_path.read_text(encoding="utf-8"))
        observed = checkpoint_metadata(path)
        return observed is not None and recorded == observed
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError):
        return False


def run_train(args: argparse.Namespace) -> None:
    row = read_manifest_row(Path(args.manifest), args.task_id, "task_id")
    if not 0 <= args.stage < args.stages:
        raise SystemExit(f"shard stage must be in 0..{args.stages-1}")
    if not 0 <= args.epoch < args.epochs:
        raise SystemExit(f"epoch must be in 0..{args.epochs-1}")

    checkpoint_prefix = Path(row["checkpoint_prefix"])
    global_step = args.epoch * args.stages + args.stage
    output_checkpoint = Path(f"{checkpoint_prefix}_step{global_step + 1}")
    if not args.overwrite:
        total_steps = args.epochs * args.stages
        for later_step in range(total_steps, global_step + 1, -1):
            later = Path(f"{checkpoint_prefix}_step{later_step}")
            if checkpoint_complete(later):
                print(
                    f"[worker] later complete checkpoint supersedes this step; "
                    f"skipping: {later}",
                    flush=True,
                )
                return
    if checkpoint_complete(output_checkpoint) and not args.overwrite:
        print(f"[worker] complete checkpoint exists; skipping: {output_checkpoint}")
        return

    # A timed-out save can leave a directory containing truncated files but no
    # atomic success marker.  Rebuild only this precisely derived output path.
    if output_checkpoint.exists():
        expected = Path(f"{checkpoint_prefix}_step{global_step + 1}")
        if output_checkpoint != expected or output_checkpoint.parent != checkpoint_prefix.parent:
            raise RuntimeError(f"refusing to replace unexpected checkpoint path: {output_checkpoint}")
        shutil.rmtree(output_checkpoint)
        print(f"[worker] removed incomplete checkpoint: {output_checkpoint}", flush=True)

    resume_from: Path | None = None
    if global_step > 0:
        resume_from = Path(f"{checkpoint_prefix}_step{global_step}")
        if not checkpoint_complete(resume_from):
            raise SystemExit(f"previous training-step checkpoint is incomplete: {resume_from}")

    train_mode = row["train_mode"]
    seed = int(row["seed"])
    stage_seed = seed + global_step * 1009
    command = [
        sys.executable,
        str(REPO_ROOT / "train" / "train.py"),
        "--model-profile",
        "llama_2_7b_chat",
        "--eval-mode",
        "seen-family" if train_mode == "seen" else "unseen-family",
        "--system-prompt-mode",
        "empty",
        "--lm-loss-input",
        "clean",
        "--stability-loss",
        row["stability_loss"],
        "--rollout-length",
        row["rollout_length"],
        "--finetuned-llm-path",
        str(checkpoint_prefix),
        "--checkpoint-output",
        str(output_checkpoint),
        "--training-data",
        row["training_data"],
        "--benign-training-data",
        row["benign_training_data"],
        "--benign-lm-weight",
        row["benign_lm_weight"],
        "--lr",
        row["learning_rate"],
        "--lambda-val",
        row["lambda_val"],
        "--epsilon",
        row["epsilon"],
        "--lora-rank",
        "8",
        "--total-epochs",
        str(args.epochs),
        "--start-epoch",
        str(args.epoch + 1),
        "--train-data-frac-start",
        f"{args.stage / args.stages:.17g}",
        "--train-data-frac-end",
        f"{(args.stage + 1) / args.stages:.17g}",
        "--seed",
        str(stage_seed),
        "--training-shuffle-seed",
        str(seed),
        "--benign-shuffle-seed",
        str(seed + 1),
    ]
    if train_mode != "seen":
        command.extend(["--unseen-family", row["heldout_family"]])
    if resume_from is not None:
        command.extend(["--resume-from", str(resume_from)])

    checkpoint_prefix.parent.mkdir(parents=True, exist_ok=True)
    print("[worker]", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    try:
        write_checkpoint_marker(output_checkpoint)
    except (ImportError, json.JSONDecodeError, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(
            f"training returned successfully but checkpoint validation failed: "
            f"{output_checkpoint}: {exc}"
        ) from exc
    if not checkpoint_complete(output_checkpoint):  # verifies marker hashes after publication
        raise SystemExit(f"checkpoint marker validation failed: {output_checkpoint}")
    if resume_from is not None and not args.retain_intermediate_checkpoints:
        expected_name = f"{checkpoint_prefix.name}_step{global_step}"
        if resume_from.name != expected_name:
            raise RuntimeError(
                f"refusing to remove unexpected checkpoint path: {resume_from}"
            )
        shutil.rmtree(resume_from)
        print(f"[worker] removed superseded checkpoint: {resume_from}", flush=True)


def prepare_eval_shard(
    source: Path,
    destination: Path,
    *,
    shard_index: int,
    shard_count: int,
    eval_kind: str,
    eval_family: str,
) -> int:
    df = pd.read_csv(source)
    if eval_kind == "asr":
        variant_col = VARIANT_COLUMNS[eval_family]
        if variant_col not in df.columns:
            raise ValueError(f"{source} is missing {variant_col!r}")
        df = df[df[variant_col].notna() & df[variant_col].astype(str).str.strip().ne("")]
    else:
        df = df.copy()
        if "Original Prompt" not in df.columns:
            candidates = (
                "adversarial",
                "goal",
                "prompt",
                "original_prompt",
                "instruction",
            )
            prompt_col = next((col for col in candidates if col in df.columns), None)
            if prompt_col is None:
                raise ValueError(f"cannot create Original Prompt from columns {list(df.columns)}")
            df["Original Prompt"] = df[prompt_col].astype(str)
        # The FRR validation file contains both the clean prompt and a benign
        # adversarial rewrite.  Evaluate the rewrite for adversarial_benign
        # rows so validation selection matches the intended stratum (and the
        # adversarial-only final FRR set), while retaining clean prompts for
        # vanilla_benign rows.
        if "adversarial" in df.columns and "data_type" in df.columns:
            adversarial = df["adversarial"].fillna("").astype(str).str.strip()
            use_adversarial = (
                df["data_type"].astype(str).str.casefold().eq("adversarial_benign")
                & adversarial.ne("")
            )
            df.loc[use_adversarial, "Original Prompt"] = adversarial[use_adversarial]
    shard = df.iloc[shard_index::shard_count].reset_index(drop=True)
    if shard.empty:
        raise ValueError(
            f"empty evaluation shard {shard_index}/{shard_count} from {source}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shard.to_csv(destination, index=False)
    return len(shard)


def run_eval(args: argparse.Namespace) -> None:
    row = read_manifest_row(Path(args.manifest), args.task_id, "eval_task_id")
    output = Path(row["output_csv"])
    checkpoint = Path(row["checkpoint_path"])
    if not checkpoint_complete(checkpoint):
        raise SystemExit(f"checkpoint is incomplete: {checkpoint}")

    tmp_root = Path(os.environ.get("SLURM_TMPDIR", "/tmp"))
    shard_path = tmp_root / f"l2bs_eval_{args.task_id}.csv"
    shard_count = int(row["shard_count"])
    shard_index = int(row["shard_index"])
    n_rows = prepare_eval_shard(
        Path(row["source_csv"]),
        shard_path,
        shard_index=shard_index,
        shard_count=shard_count,
        eval_kind=row["eval_kind"],
        eval_family=row["eval_family"],
    )
    marker_path = Path(f"{output}.complete.json")
    if not args.overwrite and completion_marker_valid(
        marker_path,
        output,
        expected_rows=n_rows,
        eval_kind=row["eval_kind"],
        eval_family=row["eval_family"],
    ):
        print(f"[worker] validated complete evaluation exists; skipping: {output}")
        return
    if output.exists() or marker_path.exists():
        print(
            f"[worker] stale or incomplete evaluation artifacts will be replaced: {output}",
            flush=True,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    unused = tmp_root / f"l2bs_unused_{args.task_id}.csv"
    command = [
        sys.executable,
        str(REPO_ROOT / "eval" / "eval.py"),
        "--model-profile",
        "llama_2_7b_chat",
        "--system-prompt-mode",
        "empty",
        "--benign-system-prompt-mode",
        "empty",
        "--resume-from",
        str(checkpoint),
        "--validation-data",
        str(shard_path),
        "--benign-validation-data",
        str(shard_path),
        "--harmful-output-file",
        str(output if row["eval_kind"] == "asr" else unused),
        "--benign-output-file",
        str(output if row["eval_kind"] == "frr" else unused),
    ]
    if row["eval_kind"] == "asr":
        command.extend(
            [
                "--eval-mode",
                "unseen-family",
                "--unseen-family",
                row["eval_family"],
                "--skip-frr",
            ]
        )
    else:
        command.append("--skip-harmful")

    print(f"[worker] evaluating {n_rows} rows", flush=True)
    print("[worker]", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    if not output.is_file() or output.stat().st_size == 0:
        raise SystemExit(f"evaluation returned successfully but output is missing: {output}")
    label_counts = validate_eval_output(
        output,
        expected_rows=n_rows,
        eval_kind=row["eval_kind"],
        eval_family=row["eval_family"],
    )
    marker = {
        "rows": n_rows,
        "sha256": sha256_file(output),
        "label_counts": label_counts,
    }
    marker_tmp = marker_path.with_suffix(marker_path.suffix + ".tmp")
    marker_tmp.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(marker_tmp, marker_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--manifest", required=True)
    p_train.add_argument("--task-id", type=int, required=True)
    p_train.add_argument("--epoch", type=int, required=True)
    p_train.add_argument("--epochs", type=int, required=True)
    p_train.add_argument("--stage", type=int, required=True)
    p_train.add_argument("--stages", type=int, required=True)
    p_train.add_argument("--overwrite", action="store_true")
    p_train.add_argument(
        "--retain-intermediate-checkpoints",
        action="store_true",
        help="Keep superseded step checkpoints for debugging (large scratch cost).",
    )
    p_train.set_defaults(func=run_train)

    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--manifest", required=True)
    p_eval.add_argument("--task-id", type=int, required=True)
    p_eval.add_argument("--overwrite", action="store_true")
    p_eval.set_defaults(func=run_eval)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
