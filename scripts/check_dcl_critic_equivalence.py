#!/usr/bin/env python3
"""Compare normal and production-DCL input routes into the frozen critic.

The experiment deliberately keeps the current DCL route intact.  In that
route the critic header/footer use the critic embedding table while the
student-formatted prompt and fixed one-hot response use the student embedding
table.  A separate controlled diagnostic replaces only that table choice.

Primary artifacts written under ``--output-dir``:

* ``results.jsonl``: run metadata, per-case records, and pair summaries.
* ``report.md``: concise human-readable conclusions, including blocked runs.
* ``logits/*.npz``: complete classification-position vocabulary logits.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shlex
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Allow ``python scripts/check_...py`` as well as ``python -m scripts...``.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.dcl_critic import (  # noqa: E402
    assemble_critic_embeddings,
    classification_logits,
    get_hinge_context_text,
    probabilities_to_embeddings,
    safe_unsafe_token_ids,
    unsafe_score_from_logits,
)


FIXTURES: tuple[tuple[str, str], ...] = (
    ("What is 2 + 2?", "The answer is four."),
    ("Say hello politely.", "Hello, I hope you are well."),
    ("Name a color.", "A bright blue sky looks lovely."),
    ("Give an example of a polite refusal.", "I cannot help with that request."),
    ("Repeat two lines.", "First line.\nSecond line is here."),
    ("Repeat this phrase with its spaces.", "  spaced words, with punctuation."),
    ("Give a word containing an accent.", "Café is a word with an accent."),
    ("List two fruits.", "1. Apples\n2. Oranges"),
)


@dataclasses.dataclass(frozen=True)
class PairSpec:
    key: str
    display_name: str
    student_path: str
    critic_path: str
    requested_starting_student_path: str
    repository_profile_student_path: str
    repository_profile_critic_path: str


class BlockedError(RuntimeError):
    """A named environment/checkpoint dependency prevents the comparison."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return jsonable(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value).replace("torch.", "")
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(jsonable(record), ensure_ascii=False, sort_keys=True))
        handle.write("\n")


def run_command(command: Sequence[str], cwd: Path | None = None) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            list(command),
            cwd=str(cwd) if cwd else None,
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        return {
            "command": list(command),
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except Exception as exc:  # optional metadata must not block the experiment
        return {"command": list(command), "error": repr(exc)}


def git_metadata() -> dict[str, Any]:
    commit = run_command(["git", "rev-parse", "HEAD"], REPO_ROOT)
    status = run_command(["git", "status", "--short"], REPO_ROOT)
    remote = run_command(["git", "remote", "get-url", "origin"], REPO_ROOT)
    return {
        "repository_path": str(REPO_ROOT),
        "commit": commit.get("stdout") if commit.get("returncode") == 0 else None,
        "working_tree_status": status.get("stdout", ""),
        "origin": remote.get("stdout") if remote.get("returncode") == 0 else None,
    }


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def scheduler_metadata() -> dict[str, Any]:
    job_id = os.environ.get("SLURM_JOB_ID")
    result: dict[str, Any] = {
        "job_id": job_id,
        "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "submit_dir": os.environ.get("SLURM_SUBMIT_DIR"),
        "job_gpus": os.environ.get("SLURM_JOB_GPUS"),
    }
    if not job_id:
        return result

    raw = run_command(["scontrol", "show", "job", "-o", job_id])
    result["scontrol"] = raw
    if raw.get("returncode") != 0:
        return result
    try:
        fields = {}
        for item in shlex.split(raw["stdout"]):
            if "=" in item:
                key, value = item.split("=", 1)
                fields[key] = value
        result["submit_time"] = fields.get("SubmitTime")
        result["start_time"] = fields.get("StartTime")
        if fields.get("SubmitTime") and fields.get("StartTime"):
            submit = dt.datetime.fromisoformat(fields["SubmitTime"])
            start = dt.datetime.fromisoformat(fields["StartTime"])
            result["queue_seconds"] = max(0.0, (start - submit).total_seconds())
    except Exception as exc:
        result["parse_error"] = repr(exc)
    return result


def runtime_metadata(args: argparse.Namespace) -> dict[str, Any]:
    cuda: dict[str, Any] = {
        "available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        cuda.update(
            {
                "current_device": index,
                "name": props.name,
                "total_memory_bytes": props.total_memory,
                "capability": list(torch.cuda.get_device_capability(index)),
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
        )
    return {
        "record_type": "run_metadata",
        "created_at": utc_now(),
        "argv": sys.argv,
        "arguments": vars(args),
        "git": git_metadata(),
        "scheduler": scheduler_metadata(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: package_version(name)
            for name in ("torch", "transformers", "accelerate", "safetensors", "peft")
        },
        "cuda": cuda,
        "production_source": {
            "module": "train.dcl_critic",
            "probability_to_embedding": "probabilities_to_embeddings",
            "wrapper_assembly": "assemble_critic_embeddings",
            "classification_readout": "classification_logits",
            "label_lookup": "safe_unsafe_token_ids",
            "score": "unsafe_score_from_logits",
            "training_callers": [
                "train.train.soft_autoregressive_generate",
                "train.train.safety_score (imported shared helper)",
            ],
        },
    }


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_metadata(path_text: str, role: str) -> dict[str, Any]:
    path = Path(os.path.expanduser(os.path.expandvars(path_text))).resolve()
    required = ["config.json"]
    tokenizer_candidates = [
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]
    weight_patterns = ("*.safetensors", "*.bin")
    files = {name: (path / name).is_file() for name in required + tokenizer_candidates}
    weights = sorted(
        str(candidate.name)
        for pattern in weight_patterns
        for candidate in path.glob(pattern)
        if candidate.is_file()
    ) if path.is_dir() else []
    metadata: dict[str, Any] = {
        "role": role,
        "requested_path": path_text,
        "resolved_path": str(path),
        "directory_exists": path.is_dir(),
        "files": files,
        "weight_files": weights,
        "config_sha256": sha256_file(path / "config.json"),
        "tokenizer_config_sha256": sha256_file(path / "tokenizer_config.json"),
    }
    if not path.is_dir():
        metadata["blocking_error"] = f"checkpoint directory does not exist: {path}"
        return metadata
    if not (path / "config.json").is_file():
        metadata["blocking_error"] = f"missing config.json: {path}"
        return metadata
    if not any(files[name] for name in tokenizer_candidates):
        metadata["blocking_error"] = f"no tokenizer files found in: {path}"
        return metadata
    if not weights:
        metadata["blocking_error"] = f"no .safetensors or .bin weights found in: {path}"
        return metadata
    try:
        config_json = json.loads((path / "config.json").read_text(encoding="utf-8"))
        metadata["config_identity"] = {
            key: config_json.get(key)
            for key in (
                "_name_or_path",
                "model_type",
                "architectures",
                "vocab_size",
                "hidden_size",
                "torch_dtype",
                "transformers_version",
            )
        }
    except Exception as exc:
        metadata["blocking_error"] = f"cannot parse config.json at {path}: {exc!r}"
    return metadata


def tokenizer_digest(tokenizer: Any) -> str:
    digest = hashlib.sha256()
    by_id = sorted((int(token_id), token) for token, token_id in tokenizer.get_vocab().items())
    for token_id, token in by_id:
        digest.update(str(token_id).encode("ascii"))
        digest.update(b"\0")
        digest.update(token.encode("utf-8", errors="surrogatepass"))
        digest.update(b"\n")
    return digest.hexdigest()


def normalize_special_map(tokenizer: Any) -> dict[str, Any]:
    return {key: str(value) for key, value in tokenizer.special_tokens_map.items()}


def compare_tokenizers(student: Any, critic: Any) -> dict[str, Any]:
    student_vocab = student.get_vocab()
    critic_vocab = critic.get_vocab()
    student_tokens = set(student_vocab)
    critic_tokens = set(critic_vocab)
    shared = student_tokens & critic_tokens
    different_ids = [
        (token, int(student_vocab[token]), int(critic_vocab[token]))
        for token in shared
        if student_vocab[token] != critic_vocab[token]
    ]
    different_ids.sort(key=lambda item: (item[1], item[2], item[0]))
    missing_in_critic = sorted(student_tokens - critic_tokens)
    missing_in_student = sorted(critic_tokens - student_tokens)
    return {
        "student_len": len(student),
        "critic_len": len(critic),
        "student_vocab_size": len(student_vocab),
        "critic_vocab_size": len(critic_vocab),
        "student_vocab_digest": tokenizer_digest(student),
        "critic_vocab_digest": tokenizer_digest(critic),
        "exact_vocab_mapping": student_vocab == critic_vocab,
        "shared_token_count": len(shared),
        "different_id_count": len(different_ids),
        "different_id_examples": different_ids[:20],
        "missing_in_critic_count": len(missing_in_critic),
        "missing_in_critic_examples": missing_in_critic[:20],
        "missing_in_student_count": len(missing_in_student),
        "missing_in_student_examples": missing_in_student[:20],
        "student_special_tokens": normalize_special_map(student),
        "critic_special_tokens": normalize_special_map(critic),
        "student_added_vocab": student.get_added_vocab(),
        "critic_added_vocab": critic.get_added_vocab(),
    }


def format_student_prompt(tokenizer: Any, prompt: str) -> str:
    """Mirror ``train.train.format_prompt`` with its current empty system role."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def encode_1d(tokenizer: Any, text: str, *, add_special_tokens: bool) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=add_special_tokens)
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise ValueError(f"expected one sequence, got batch size {len(ids)}")
        ids = ids[0]
    return [int(item) for item in ids]


def map_student_ids_to_critic(
    student_ids: Sequence[int],
    student_tokenizer: Any,
    critic_tokenizer: Any,
) -> tuple[list[int] | None, list[dict[str, Any]]]:
    critic_vocab = critic_tokenizer.get_vocab()
    mapped: list[int] = []
    failures: list[dict[str, Any]] = []
    for position, student_id in enumerate(student_ids):
        token = student_tokenizer.convert_ids_to_tokens(int(student_id))
        critic_id = critic_vocab.get(token)
        if critic_id is None:
            failures.append(
                {"position": position, "student_id": int(student_id), "token": token}
            )
        else:
            mapped.append(int(critic_id))
    return (None if failures else mapped), failures


def whole_text_tokenization(
    tokenizer: Any,
    prefix: str,
    response_text: str,
    suffix: str,
) -> dict[str, Any]:
    full_text = prefix + response_text + suffix
    response_start = len(prefix)
    response_end = response_start + len(response_text)
    try:
        encoded = tokenizer(
            full_text,
            add_special_tokens=True,
            return_offsets_mapping=True,
        )
        ids = [int(item) for item in encoded["input_ids"]]
        offsets = [tuple(map(int, pair)) for pair in encoded["offset_mapping"]]
        response_positions = [
            index
            for index, (start, end) in enumerate(offsets)
            if end > start and start < response_end and end > response_start
        ]
        return {
            "text": full_text,
            "ids": ids,
            "offsets": offsets,
            "response_positions": response_positions,
            "response_ids": [ids[index] for index in response_positions],
            "response_char_span": [response_start, response_end],
            "offset_mapping_available": True,
        }
    except (NotImplementedError, TypeError, ValueError):
        ids = encode_1d(tokenizer, full_text, add_special_tokens=True)
        prefix_ids = encode_1d(tokenizer, prefix, add_special_tokens=True)
        through_response_ids = encode_1d(
            tokenizer, prefix + response_text, add_special_tokens=True
        )
        boundary_stable = (
            through_response_ids[: len(prefix_ids)] == prefix_ids
            and ids[: len(through_response_ids)] == through_response_ids
        )
        positions = (
            list(range(len(prefix_ids), len(through_response_ids)))
            if boundary_stable
            else []
        )
        return {
            "text": full_text,
            "ids": ids,
            "offsets": None,
            "response_positions": positions,
            "response_ids": [ids[index] for index in positions],
            "response_char_span": [response_start, response_end],
            "offset_mapping_available": False,
            "boundary_stable": boundary_stable,
        }


def ordinary_guard_tokenization(
    tokenizer: Any,
    prompt: str,
    response_text: str,
) -> dict[str, Any]:
    """Build route A with the checkpoint's ordinary guard chat template."""
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response_text},
    ]
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        raw_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
    except Exception as exc:
        raise BlockedError(
            "critic tokenizer cannot construct its ordinary user/assistant guard template: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if isinstance(raw_ids, torch.Tensor):
        raw_ids = raw_ids.detach().cpu().tolist()
    if isinstance(raw_ids, dict):
        raw_ids = raw_ids["input_ids"]
    if raw_ids and isinstance(raw_ids[0], list):
        if len(raw_ids) != 1:
            raise BlockedError(
                f"critic chat template unexpectedly returned {len(raw_ids)} sequences"
            )
        raw_ids = raw_ids[0]
    ids = [int(item) for item in raw_ids]
    response_start = rendered.rfind(response_text)
    response_end = response_start + len(response_text) if response_start >= 0 else -1
    response_positions: list[int] = []
    offsets: list[tuple[int, int]] | None = None
    offset_add_special_tokens: bool | None = None
    response_span_method: str | None = None

    if response_start >= 0:
        for add_special_tokens in (False, True):
            try:
                encoded = tokenizer(
                    rendered,
                    add_special_tokens=add_special_tokens,
                    return_offsets_mapping=True,
                )
                candidate_ids = [int(item) for item in encoded["input_ids"]]
                if candidate_ids != ids:
                    continue
                offsets = [tuple(map(int, pair)) for pair in encoded["offset_mapping"]]
                response_positions = [
                    index
                    for index, (start, end) in enumerate(offsets)
                    if end > start and start < response_end and end > response_start
                ]
                offset_add_special_tokens = add_special_tokens
                response_span_method = "offset_mapping"
                break
            except (NotImplementedError, TypeError, ValueError):
                continue

        if not response_positions:
            prefix_text = rendered[:response_start]
            through_response_text = rendered[:response_end]
            for add_special_tokens in (False, True):
                try:
                    full_ids = encode_1d(
                        tokenizer,
                        rendered,
                        add_special_tokens=add_special_tokens,
                    )
                    if full_ids != ids:
                        continue
                    prefix_ids = encode_1d(
                        tokenizer,
                        prefix_text,
                        add_special_tokens=add_special_tokens,
                    )
                    through_response_ids = encode_1d(
                        tokenizer,
                        through_response_text,
                        add_special_tokens=add_special_tokens,
                    )
                    boundary_stable = (
                        ids[: len(prefix_ids)] == prefix_ids
                        and ids[: len(through_response_ids)] == through_response_ids
                    )
                    if boundary_stable:
                        response_positions = list(
                            range(len(prefix_ids), len(through_response_ids))
                        )
                        offset_add_special_tokens = add_special_tokens
                        response_span_method = "stable_prefix_fallback"
                        break
                except (NotImplementedError, TypeError, ValueError):
                    continue

    return {
        "text": rendered,
        "ids": ids,
        "messages": messages,
        "response_char_span": [response_start, response_end],
        "response_positions": response_positions,
        "response_ids": [ids[index] for index in response_positions],
        "offsets": offsets,
        "offset_add_special_tokens": offset_add_special_tokens,
        "response_span_verifiable": bool(response_positions),
        "response_span_method": response_span_method,
        "response_text_found": response_start >= 0,
        "contains_explicit_taxonomy": any(
            marker.lower() in rendered.lower()
            for marker in (
                "unsafe content categories",
                "safety categories",
                "<BEGIN UNSAFE CONTENT CATEGORIES>",
                "Violence and Hate",
            )
        ),
        "source": "critic_tokenizer.apply_chat_template(user, assistant)",
    }


def tensor_finite(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor).all().item())


def tensor_difference(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "left_shape": list(left.shape),
        "right_shape": list(right.shape),
        "left_dtype": str(left.dtype).replace("torch.", ""),
        "right_dtype": str(right.dtype).replace("torch.", ""),
        "left_finite": tensor_finite(left),
        "right_finite": tensor_finite(right),
        "atol": atol,
        "rtol": rtol,
    }
    if left.shape != right.shape:
        result.update({"comparable": False, "allclose": False, "reason": "shape_mismatch"})
        return result
    left32 = left.detach().float()
    right32 = right.detach().float()
    difference = (left32 - right32).abs()
    flat_index = int(difference.argmax().item()) if difference.numel() else 0
    coordinates = list(np.unravel_index(flat_index, tuple(difference.shape))) if difference.numel() else []
    result.update(
        {
            "comparable": True,
            "allclose": bool(torch.allclose(left32, right32, atol=atol, rtol=rtol)),
            "max_abs_diff": float(difference.max().item()) if difference.numel() else 0.0,
            "mean_abs_diff": float(difference.mean().item()) if difference.numel() else 0.0,
            "max_abs_position": coordinates,
        }
    )
    return result


def logit_comparison(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    left_name: str,
    right_name: str,
    safe_id: int,
    unsafe_id: int,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    base = tensor_difference(left, right, atol=atol, rtol=rtol)
    base.update({"left_route": left_name, "right_route": right_name})
    if left.shape != right.shape:
        return base
    left32 = left.detach().float().reshape(-1)
    right32 = right.detach().float().reshape(-1)
    difference = (left32 - right32).abs()
    max_id = int(difference.argmax().item())
    left_score = unsafe_score_from_logits(
        left.detach().float().reshape(1, -1), safe_id, unsafe_id
    )[0]
    right_score = unsafe_score_from_logits(
        right.detach().float().reshape(1, -1), safe_id, unsafe_id
    )[0]
    base.update(
        {
            "max_difference_vocab_id": max_id,
            "max_difference_left_logit": float(left32[max_id].item()),
            "max_difference_right_logit": float(right32[max_id].item()),
            "safe_id": safe_id,
            "unsafe_id": unsafe_id,
            "safe_left_logit": float(left32[safe_id].item()),
            "safe_right_logit": float(right32[safe_id].item()),
            "safe_logit_difference": float((left32[safe_id] - right32[safe_id]).item()),
            "unsafe_left_logit": float(left32[unsafe_id].item()),
            "unsafe_right_logit": float(right32[unsafe_id].item()),
            "unsafe_logit_difference": float(
                (left32[unsafe_id] - right32[unsafe_id]).item()
            ),
            "unsafe_score_left": float(left_score.item()),
            "unsafe_score_right": float(right_score.item()),
            "unsafe_score_abs_difference": float((left_score - right_score).abs().item()),
        }
    )
    return base


def extract_logits(outputs: Any) -> torch.Tensor:
    # Indexing alone returns a view that retains the full [batch, sequence,
    # vocabulary] allocation.  Clone the one required vector so every forward's
    # large sequence-logit tensor can be released before the next route runs.
    return classification_logits(outputs).detach()[0].clone()


def model_forward_ids(model: Any, ids: torch.Tensor, *, use_cache: bool | None = None) -> torch.Tensor:
    kwargs: dict[str, Any] = {"input_ids": ids}
    if use_cache is not None:
        kwargs["use_cache"] = use_cache
    return extract_logits(model(**kwargs))


def model_forward_embeds(
    model: Any,
    embeds: torch.Tensor,
    *,
    use_cache: bool | None = None,
) -> torch.Tensor:
    kwargs: dict[str, Any] = {"inputs_embeds": embeds}
    if use_cache is not None:
        kwargs["use_cache"] = use_cache
    return extract_logits(model(**kwargs))


def load_pair_assets(
    pair: PairSpec,
    *,
    dtype: torch.dtype,
    attention_implementation: str,
) -> tuple[Any, Any, torch.Tensor, Any, dict[str, Any]]:
    student_meta = checkpoint_metadata(pair.student_path, "student")
    critic_meta = checkpoint_metadata(pair.critic_path, "critic")
    for metadata in (student_meta, critic_meta):
        if metadata.get("blocking_error"):
            raise BlockedError(metadata["blocking_error"])
    if not torch.cuda.is_available():
        raise BlockedError("CUDA is unavailable; the requested Narval GPU comparison cannot run")

    student_path = student_meta["resolved_path"]
    critic_path = critic_meta["resolved_path"]
    cuda_sync()
    load_started = time.perf_counter()

    student_config = AutoConfig.from_pretrained(student_path, local_files_only=True)
    critic_config = AutoConfig.from_pretrained(critic_path, local_files_only=True)
    student_tokenizer = AutoTokenizer.from_pretrained(student_path, local_files_only=True)
    critic_tokenizer = AutoTokenizer.from_pretrained(critic_path, local_files_only=True)

    # Load the actual production student embedding weights, retain only that table,
    # and release the transformer before loading the critic.  This keeps route C
    # faithful without holding two full models in GPU memory.
    student_model = AutoModelForCausalLM.from_pretrained(
        student_path,
        torch_dtype=dtype,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    student_embedding_weight_cpu = (
        student_model.get_input_embeddings().weight.detach().to("cpu").clone()
    )
    student_embedding_class = type(student_model.get_input_embeddings()).__name__
    del student_model
    gc.collect()

    critic_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "local_files_only": True,
    }
    if attention_implementation != "auto":
        critic_kwargs["attn_implementation"] = attention_implementation
    critic = AutoModelForCausalLM.from_pretrained(critic_path, **critic_kwargs)
    for parameter in critic.parameters():
        parameter.requires_grad_(False)
    critic.eval()
    device = critic.get_input_embeddings().weight.device
    if device.type != "cuda":
        raise BlockedError(
            f"critic input embeddings loaded on {device}, not CUDA; "
            f"device_map={getattr(critic, 'hf_device_map', None)!r}"
        )
    student_embedding_weight = student_embedding_weight_cpu.to(device=device, dtype=dtype)
    del student_embedding_weight_cpu
    gc.collect()
    cuda_sync()
    load_seconds = time.perf_counter() - load_started

    critic_embedding = critic.get_input_embeddings().weight
    model_meta = {
        "student_checkpoint": student_meta,
        "critic_checkpoint": critic_meta,
        "student_config": {
            "model_type": getattr(student_config, "model_type", None),
            "architectures": getattr(student_config, "architectures", None),
            "vocab_size": getattr(student_config, "vocab_size", None),
            "hidden_size": getattr(student_config, "hidden_size", None),
            "torch_dtype": str(getattr(student_config, "torch_dtype", None)),
        },
        "critic_config": {
            "model_type": getattr(critic_config, "model_type", None),
            "architectures": getattr(critic_config, "architectures", None),
            "vocab_size": getattr(critic_config, "vocab_size", None),
            "hidden_size": getattr(critic_config, "hidden_size", None),
            "torch_dtype": str(getattr(critic_config, "torch_dtype", None)),
            "use_cache": getattr(critic.config, "use_cache", None),
            "attention_implementation": getattr(
                critic.config, "_attn_implementation", attention_implementation
            ),
        },
        "student_embedding": {
            "class": student_embedding_class,
            "shape": list(student_embedding_weight.shape),
            "dtype": str(student_embedding_weight.dtype).replace("torch.", ""),
            "device": str(student_embedding_weight.device),
        },
        "critic_embedding": {
            "class": type(critic.get_input_embeddings()).__name__,
            "shape": list(critic_embedding.shape),
            "dtype": str(critic_embedding.dtype).replace("torch.", ""),
            "device": str(critic_embedding.device),
        },
        "device_map": getattr(critic, "hf_device_map", None),
        "load_adaptation": (
            "Loaded the full student checkpoint on CPU only, cloned its production input "
            "embedding table, then deleted the student transformer before loading the critic."
        ),
        "adapters": {"student": None, "critic": None},
        "load_seconds": load_seconds,
    }
    return student_tokenizer, critic_tokenizer, student_embedding_weight, critic, model_meta


def pair_path_metadata(pair: PairSpec) -> dict[str, Any]:
    actual_student = str(Path(os.path.expandvars(os.path.expanduser(pair.student_path))).resolve())
    actual_critic = str(Path(os.path.expandvars(os.path.expanduser(pair.critic_path))).resolve())
    requested_starting_student = os.path.expandvars(pair.requested_starting_student_path)
    expected_student = os.path.expandvars(pair.repository_profile_student_path)
    expected_critic = os.path.expandvars(pair.repository_profile_critic_path)
    return {
        "actual_student_path": actual_student,
        "actual_critic_path": actual_critic,
        "requested_starting_student_path": pair.requested_starting_student_path,
        "requested_starting_student_path_resolved": str(
            Path(requested_starting_student).resolve()
        ),
        "repository_profile_student_path": pair.repository_profile_student_path,
        "repository_profile_critic_path": pair.repository_profile_critic_path,
        "student_path_differs_from_repository_profile": (
            "$SCRATCH" not in expected_student
            and str(Path(expected_student).resolve()) != actual_student
        ) or (
            pair.key == "llama31" and "llama_3_8b_instruct" in pair.repository_profile_student_path
        ),
        "critic_path_differs_from_repository_profile": (
            "$SCRATCH" not in expected_critic
            and str(Path(expected_critic).resolve()) != actual_critic
        ),
        "note": (
            "The requested starting point is llama31_8b_instruct, while the active "
            "repository DCL profile resolves to llama_3_8b_instruct. The default "
            f"selected {'the authoritative repository profile' if str(Path(expected_student).resolve()) == actual_student else 'the requested/overridden path'}; "
            "the checkpoint config identity is recorded without assuming those model names are interchangeable."
            if pair.key == "llama31"
            else "Requested paths match the active Llama-2 profile."
        ),
    }


def prepare_case(
    *,
    prompt: str,
    response: str,
    rollout_length: int,
    student_tokenizer: Any,
    critic_tokenizer: Any,
    header_text: str,
    footer_text: str,
) -> dict[str, Any]:
    formatted_prompt = format_student_prompt(student_tokenizer, prompt)
    # Production generation appends model-output IDs after the already-tokenized
    # prompt.  Tokenize only the fixed response (no BOS), then inject its first T
    # IDs as one-hot model outputs.
    all_response_ids = encode_1d(
        student_tokenizer, response, add_special_tokens=False
    )
    if len(all_response_ids) < rollout_length:
        raise BlockedError(
            f"fixed response has {len(all_response_ids)} tokens, shorter than T={rollout_length}: {response!r}"
        )
    response_ids = all_response_ids[:rollout_length]
    response_text = student_tokenizer.decode(
        response_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    response_roundtrip_ids = encode_1d(
        student_tokenizer, response_text, add_special_tokens=False
    )

    # Match soft_autoregressive_generate: the rendered chat string is passed to
    # the tokenizer again with its default add_special_tokens=True behavior.
    prompt_ids = encode_1d(student_tokenizer, formatted_prompt, add_special_tokens=True)
    header_ids = encode_1d(critic_tokenizer, header_text, add_special_tokens=True)
    footer_ids = encode_1d(critic_tokenizer, footer_text, add_special_tokens=False)
    middle_student_ids = prompt_ids + response_ids
    mapped_middle_ids, mapping_failures = map_student_ids_to_critic(
        middle_student_ids, student_tokenizer, critic_tokenizer
    )
    controlled_ids = (
        header_ids + mapped_middle_ids + footer_ids
        if mapped_middle_ids is not None
        else None
    )

    ordinary = ordinary_guard_tokenization(
        critic_tokenizer,
        prompt,
        response_text,
    )
    wrapper_whole = whole_text_tokenization(
        critic_tokenizer,
        header_text + formatted_prompt,
        response_text,
        footer_text,
    )
    controlled_response_start = len(header_ids) + len(prompt_ids)
    controlled_response_positions = list(
        range(controlled_response_start, controlled_response_start + rollout_length)
    )
    ordinary_ids_equal = controlled_ids == ordinary["ids"]
    ordinary_response_positions_for_comparison = ordinary["response_positions"]
    if not ordinary_response_positions_for_comparison and ordinary_ids_equal:
        ordinary_response_positions_for_comparison = controlled_response_positions
    ordinary_response_match = (
        len(ordinary["response_ids"]) == rollout_length
        and mapped_middle_ids is not None
        and ordinary["response_ids"] == mapped_middle_ids[-rollout_length:]
    )
    if not ordinary["response_span_verifiable"] and ordinary_ids_equal:
        ordinary_response_match = True
    structural = {
        "ordinary_vs_production_ids_equal": ordinary_ids_equal,
        "wrapper_whole_vs_production_ids_equal": controlled_ids == wrapper_whole["ids"],
        "ordinary_length": len(ordinary["ids"]),
        "wrapper_whole_length": len(wrapper_whole["ids"]),
        "production_length": len(header_ids) + len(middle_student_ids) + len(footer_ids),
        "ordinary_response_token_count": len(ordinary["response_positions"]),
        "wrapper_whole_response_token_count": len(wrapper_whole["response_positions"]),
        "production_response_token_count": rollout_length,
        "ordinary_response_ids": ordinary["response_ids"],
        "wrapper_whole_response_ids": wrapper_whole["response_ids"],
        "production_response_student_ids": response_ids,
        "mapped_production_response_critic_ids": (
            mapped_middle_ids[-rollout_length:] if mapped_middle_ids is not None else None
        ),
        "ordinary_response_span_verifiable": ordinary["response_span_verifiable"],
        "ordinary_response_span_method": ordinary["response_span_method"],
        "ordinary_response_span_matches_production": ordinary_response_match,
        "wrapper_whole_response_span_matches_production": (
            mapped_middle_ids is not None
            and len(wrapper_whole["response_ids"]) == rollout_length
            and wrapper_whole["response_ids"] == mapped_middle_ids[-rollout_length:]
        ),
        "attention_masks": "implicit all-ones for all routes (batch size one, no padding)",
        "position_ids": "implicit contiguous positions starting at zero for each route",
        "cache_positions": "implicit; no past_key_values are supplied or reused",
        "classification_position": {
            "route_a": len(ordinary["ids"]) - 1,
            "wrapper_whole_control": len(wrapper_whole["ids"]) - 1,
            "route_c": len(header_ids) + len(middle_student_ids) + len(footer_ids) - 1,
        },
    }
    return {
        "prompt": prompt,
        "fixed_response": response,
        "formatted_student_prompt": formatted_prompt,
        "response_all_student_ids": all_response_ids,
        "response_ids": response_ids,
        "response_prefix_text": response_text,
        "response_roundtrip_ids": response_roundtrip_ids,
        "response_roundtrip_equal": response_roundtrip_ids == response_ids,
        "header_text": header_text,
        "footer_text": footer_text,
        "header_ids": header_ids,
        "footer_ids": footer_ids,
        "prompt_student_ids": prompt_ids,
        "middle_student_ids": middle_student_ids,
        "mapped_middle_critic_ids": mapped_middle_ids,
        "mapping_failures": mapping_failures,
        "controlled_ids": controlled_ids,
        "controlled_response_positions": controlled_response_positions,
        "ordinary": ordinary,
        "ordinary_response_positions_for_comparison": ordinary_response_positions_for_comparison,
        "wrapper_whole": wrapper_whole,
        "structural": structural,
    }


def classify_case_status(
    *,
    structural: dict[str, Any],
    ab: dict[str, Any],
    ac: dict[str, Any],
    bc_embeddings: dict[str, Any],
    finite: dict[str, bool],
    aa_repeat: dict[str, Any] | None,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    sequence_structure_ok = (
        structural["ordinary_vs_production_ids_equal"]
        and structural["ordinary_response_span_matches_production"]
        and structural["ordinary_length"] == structural["production_length"]
    )
    labels_ok = structural.get("safe_unsafe_labels_are_single_tokens", True)
    structural_ok = sequence_structure_ok and labels_ok
    if not sequence_structure_ok:
        reasons.append("ordinary guard-template tokens differ from the production DCL assembly")
    if not labels_ok:
        reasons.append(
            "production readout takes only the first ID of a multi-token safe/unsafe label"
        )
    if not all(finite.values()):
        reasons.append("one or more embeddings/logit vectors contain non-finite values")
    if aa_repeat is not None and not aa_repeat["allclose"]:
        reasons.append("repeated route A failed the declared numerical tolerance")
    if not ab["allclose"]:
        reasons.append("A-B token-ID/external-embedding numerical control failed")
    if not bc_embeddings.get("allclose", False):
        reasons.append("C embeddings differ from B (or their sequence shapes differ)")
    if not ac["allclose"]:
        reasons.append("A-C classification-position logits differ")

    if not structural_ok:
        return "FAIL_INTERFACE", reasons
    if (aa_repeat is not None and not aa_repeat["allclose"]) or not ab["allclose"]:
        return "INCONCLUSIVE_NUMERICAL", reasons
    if not all(finite.values()):
        return "INCONCLUSIVE_NUMERICAL", reasons
    if not bc_embeddings.get("allclose", False) or not ac["allclose"]:
        return "FAIL_INTERFACE", reasons
    return "PASS", []


def save_logits(path: Path, tensors: dict[str, torch.Tensor | None]) -> None:
    arrays = {
        name: tensor.detach().float().cpu().numpy()
        for name, tensor in tensors.items()
        if tensor is not None
    }
    np.savez_compressed(path, **arrays)


def run_pair(
    pair: PairSpec,
    *,
    args: argparse.Namespace,
    results_path: Path,
    logits_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pair_started = time.perf_counter()
    dtype = resolve_dtype(args.dtype)
    pair_record: dict[str, Any] = {
        "record_type": "pair_metadata",
        "pair": pair.key,
        "display_name": pair.display_name,
        "created_at": utc_now(),
        "paths": pair_path_metadata(pair),
        "rollout_length": args.rollout_length,
        "tolerances": {"atol": args.atol, "rtol": args.rtol},
        "dtype": args.dtype,
        "requested_attention_implementation": args.attention_implementation,
    }

    try:
        (
            student_tokenizer,
            critic_tokenizer,
            student_embedding_weight,
            critic,
            model_meta,
        ) = load_pair_assets(
            pair,
            dtype=dtype,
            attention_implementation=args.attention_implementation,
        )
    except Exception as exc:
        pair_record.update(
            {
                "status": "BLOCKED",
                "blocking_error": str(exc),
                "exception_type": type(exc).__name__,
                "traceback": traceback.format_exc(),
                "total_seconds": time.perf_counter() - pair_started,
                "student_checkpoint_preflight": checkpoint_metadata(
                    pair.student_path, "student"
                ),
                "critic_checkpoint_preflight": checkpoint_metadata(
                    pair.critic_path, "critic"
                ),
            }
        )
        append_jsonl(results_path, pair_record)
        return pair_record, []

    pair_record["model"] = model_meta
    tokenizer_comparison = compare_tokenizers(student_tokenizer, critic_tokenizer)
    pair_record["tokenizer_comparison"] = tokenizer_comparison
    safe_id, unsafe_id = safe_unsafe_token_ids(critic_tokenizer)
    pair_record["labels"] = {
        "safe_id": safe_id,
        "unsafe_id": unsafe_id,
        "safe_encoding": critic_tokenizer.encode("safe", add_special_tokens=False),
        "unsafe_encoding": critic_tokenizer.encode("unsafe", add_special_tokens=False),
    }
    pair_record["labels"]["safe_is_single_token"] = (
        len(pair_record["labels"]["safe_encoding"]) == 1
    )
    pair_record["labels"]["unsafe_is_single_token"] = (
        len(pair_record["labels"]["unsafe_encoding"]) == 1
    )
    pair_record["labels"]["safe_tokens"] = critic_tokenizer.convert_ids_to_tokens(
        pair_record["labels"]["safe_encoding"]
    )
    pair_record["labels"]["unsafe_tokens"] = critic_tokenizer.convert_ids_to_tokens(
        pair_record["labels"]["unsafe_encoding"]
    )
    header_text, footer_text = get_hinge_context_text("llama_guard")
    pair_record["production_wrapper"] = {
        "hinge_style": "llama_guard",
        "header_text": header_text,
        "footer_text": footer_text,
        "contains_explicit_taxonomy": False,
        "note": (
            "This is the active hand-written partial-response wrapper in train.dcl_critic; "
            "it is not silently replaced with a different Llama Guard template."
        ),
    }
    critic_chat_template = getattr(critic_tokenizer, "chat_template", None)
    pair_record["ordinary_guard_template"] = {
        "source": "critic tokenizer chat_template",
        "template": critic_chat_template,
        "sha256": (
            hashlib.sha256(critic_chat_template.encode("utf-8")).hexdigest()
            if isinstance(critic_chat_template, str)
            else None
        ),
    }
    critic_embedding_layer = critic.get_input_embeddings()
    critic_embedding_weight = critic_embedding_layer.weight
    device = critic_embedding_weight.device
    if student_embedding_weight.shape[1] != critic_embedding_weight.shape[1]:
        pair_record.update(
            {
                "status": "BLOCKED",
                "blocking_error": (
                    "student/critic hidden widths differ; production DCL concatenation cannot run: "
                    f"{tuple(student_embedding_weight.shape)} vs {tuple(critic_embedding_weight.shape)}"
                ),
                "total_seconds": time.perf_counter() - pair_started,
            }
        )
        append_jsonl(results_path, pair_record)
        del critic, student_embedding_weight
        gc.collect()
        torch.cuda.empty_cache()
        return pair_record, []

    append_jsonl(results_path, pair_record)
    prepared_cases: list[dict[str, Any]] = []
    for prompt, response in FIXTURES[: args.case_count]:
        prepared = prepare_case(
            prompt=prompt,
            response=response,
            rollout_length=args.rollout_length,
            student_tokenizer=student_tokenizer,
            critic_tokenizer=critic_tokenizer,
            header_text=header_text,
            footer_text=footer_text,
        )
        prepared["structural"]["safe_unsafe_labels_are_single_tokens"] = bool(
            pair_record["labels"]["safe_is_single_token"]
            and pair_record["labels"]["unsafe_is_single_token"]
        )
        prepared_cases.append(prepared)

    # Warm up the critic with route A while keeping warm-up time separate.
    warmup_started = time.perf_counter()
    with torch.inference_mode():
        warmup_ids = torch.tensor(
            [prepared_cases[0]["ordinary"]["ids"]], dtype=torch.long, device=device
        )
        cuda_sync()
        _ = model_forward_ids(critic, warmup_ids)
        cuda_sync()
    warmup_seconds = time.perf_counter() - warmup_started
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    test_started = time.perf_counter()
    cuda_sync()
    cases: list[dict[str, Any]] = []
    with torch.inference_mode():
        for case_index, prepared in enumerate(prepared_cases):
            case_started = time.perf_counter()
            ordinary_ids = torch.tensor(
                [prepared["ordinary"]["ids"]], dtype=torch.long, device=device
            )
            wrapper_whole_ids = torch.tensor(
                [prepared["wrapper_whole"]["ids"]], dtype=torch.long, device=device
            )
            header_ids = torch.tensor(
                [prepared["header_ids"]], dtype=torch.long, device=device
            )
            footer_ids = torch.tensor(
                [prepared["footer_ids"]], dtype=torch.long, device=device
            )
            prompt_ids = torch.tensor(
                [prepared["prompt_student_ids"]], dtype=torch.long, device=device
            )
            response_ids = torch.tensor(
                [prepared["response_ids"]], dtype=torch.long, device=device
            )

            # Route A: checkpoint-provided ordinary guard-template reference.
            logits_a = model_forward_ids(critic, ordinary_ids)

            # Route B: exactly A's IDs through the critic embedding table.
            embeds_b = critic_embedding_layer(ordinary_ids)
            logits_b = model_forward_embeds(critic, embeds_b)

            # W: whole-tokenized version of the custom production DCL wrapper.
            # This is not route A; it isolates ordinary-template vs custom-wrapper
            # differences from split-token boundary and embedding-table differences.
            logits_wrapper_whole = model_forward_ids(critic, wrapper_whole_ids)

            # Route C: current production DCL assembly and student embedding table.
            header_embeddings = critic_embedding_layer(header_ids)
            footer_embeddings = critic_embedding_layer(footer_ids)
            prompt_embeddings = torch.nn.functional.embedding(
                prompt_ids, student_embedding_weight
            )
            one_hot = torch.nn.functional.one_hot(
                response_ids,
                num_classes=student_embedding_weight.shape[0],
            ).to(dtype=student_embedding_weight.dtype)
            response_embeddings_c = probabilities_to_embeddings(
                one_hot, student_embedding_weight
            )
            middle_embeddings_c = torch.cat(
                [prompt_embeddings, response_embeddings_c], dim=1
            )
            embeds_c = assemble_critic_embeddings(
                header_embeddings, middle_embeddings_c, footer_embeddings
            )
            logits_c = model_forward_embeds(critic, embeds_c)

            # Controlled split-sequence input-ID reference.  Route D changes only
            # the one-hot response table, as requested; route E is an additional
            # all-critic-table control that should match the split input-ID route.
            logits_split = None
            embeds_d = None
            logits_d = None
            embeds_e = None
            logits_e = None
            controlled_ids_tensor = None
            if prepared["controlled_ids"] is not None:
                controlled_ids_tensor = torch.tensor(
                    [prepared["controlled_ids"]], dtype=torch.long, device=device
                )
                logits_split = model_forward_ids(critic, controlled_ids_tensor)
                mapped_prompt_ids = controlled_ids_tensor[
                    :, len(prepared["header_ids"]):
                    len(prepared["header_ids"]) + len(prepared["prompt_student_ids"])
                ]
                response_start = (
                    len(prepared["header_ids"]) + len(prepared["prompt_student_ids"])
                )
                mapped_response_ids = controlled_ids_tensor[
                    :, response_start: response_start + args.rollout_length
                ]
                one_hot_d = torch.nn.functional.one_hot(
                    mapped_response_ids,
                    num_classes=critic_embedding_weight.shape[0],
                ).to(dtype=critic_embedding_weight.dtype)
                response_embeddings_d = probabilities_to_embeddings(
                    one_hot_d, critic_embedding_weight
                )
                middle_embeddings_d = torch.cat(
                    [prompt_embeddings, response_embeddings_d],
                    dim=1,
                )
                embeds_d = assemble_critic_embeddings(
                    header_embeddings, middle_embeddings_d, footer_embeddings
                )
                logits_d = model_forward_embeds(critic, embeds_d)
                middle_embeddings_e = torch.cat(
                    [critic_embedding_layer(mapped_prompt_ids), response_embeddings_d],
                    dim=1,
                )
                embeds_e = assemble_critic_embeddings(
                    header_embeddings, middle_embeddings_e, footer_embeddings
                )
                logits_e = model_forward_embeds(critic, embeds_e)

            logits_a_repeat = model_forward_ids(critic, ordinary_ids) if case_index == 0 else None
            logits_a_no_cache = (
                model_forward_ids(critic, ordinary_ids, use_cache=False)
                if case_index == 0
                else None
            )

            a_positions = prepared["ordinary_response_positions_for_comparison"]
            c_positions = prepared["controlled_response_positions"]
            embeds_b_response = (
                embeds_b[:, a_positions, :]
                if a_positions
                else embeds_b[:, 0:0, :]
            )
            embeds_c_response = embeds_c[:, c_positions, :]
            full_embedding_bc = tensor_difference(
                embeds_b, embeds_c, atol=args.atol, rtol=args.rtol
            )
            response_embedding_bc = tensor_difference(
                embeds_b_response,
                embeds_c_response,
                atol=args.atol,
                rtol=args.rtol,
            )
            controlled_embedding_dc = (
                tensor_difference(embeds_d, embeds_c, atol=args.atol, rtol=args.rtol)
                if embeds_d is not None
                else None
            )
            controlled_response_dc = (
                tensor_difference(
                    embeds_d[:, c_positions, :],
                    embeds_c_response,
                    atol=args.atol,
                    rtol=args.rtol,
                )
                if embeds_d is not None
                else None
            )
            controlled_embedding_ec = (
                tensor_difference(embeds_e, embeds_c, atol=args.atol, rtol=args.rtol)
                if embeds_e is not None
                else None
            )

            comparisons = {
                "A_B": logit_comparison(
                    logits_a,
                    logits_b,
                    left_name="A_ordinary_guard_input_ids",
                    right_name="B_ordinary_guard_critic_embeddings",
                    safe_id=safe_id,
                    unsafe_id=unsafe_id,
                    atol=args.atol,
                    rtol=args.rtol,
                ),
                "A_C": logit_comparison(
                    logits_a,
                    logits_c,
                    left_name="A_ordinary_guard_input_ids",
                    right_name="C_production_dcl",
                    safe_id=safe_id,
                    unsafe_id=unsafe_id,
                    atol=args.atol,
                    rtol=args.rtol,
                ),
                "A_wrapper_whole": logit_comparison(
                    logits_a,
                    logits_wrapper_whole,
                    left_name="A_ordinary_guard_input_ids",
                    right_name="custom_wrapper_whole_input_ids",
                    safe_id=safe_id,
                    unsafe_id=unsafe_id,
                    atol=args.atol,
                    rtol=args.rtol,
                ),
                "wrapper_whole_C": logit_comparison(
                    logits_wrapper_whole,
                    logits_c,
                    left_name="custom_wrapper_whole_input_ids",
                    right_name="C_production_dcl",
                    safe_id=safe_id,
                    unsafe_id=unsafe_id,
                    atol=args.atol,
                    rtol=args.rtol,
                ),
            }
            if logits_split is not None and logits_d is not None and logits_e is not None:
                comparisons["A_D"] = logit_comparison(
                    logits_a,
                    logits_d,
                    left_name="A_ordinary_guard_input_ids",
                    right_name="D_critic_response_table_only",
                    safe_id=safe_id,
                    unsafe_id=unsafe_id,
                    atol=args.atol,
                    rtol=args.rtol,
                )
                comparisons["wrapper_whole_D"] = logit_comparison(
                    logits_wrapper_whole,
                    logits_d,
                    left_name="custom_wrapper_whole_input_ids",
                    right_name="D_critic_response_table_only",
                    safe_id=safe_id,
                    unsafe_id=unsafe_id,
                    atol=args.atol,
                    rtol=args.rtol,
                )
                comparisons["split_D"] = logit_comparison(
                    logits_split,
                    logits_d,
                    left_name="controlled_split_input_ids",
                    right_name="D_critic_response_table_only",
                    safe_id=safe_id,
                    unsafe_id=unsafe_id,
                    atol=args.atol,
                    rtol=args.rtol,
                )
                comparisons["split_C"] = logit_comparison(
                    logits_split,
                    logits_c,
                    left_name="controlled_split_input_ids",
                    right_name="C_production_dcl",
                    safe_id=safe_id,
                    unsafe_id=unsafe_id,
                    atol=args.atol,
                    rtol=args.rtol,
                )
                comparisons["A_split"] = logit_comparison(
                    logits_a,
                    logits_split,
                    left_name="A_ordinary_guard_input_ids",
                    right_name="controlled_split_input_ids",
                    safe_id=safe_id,
                    unsafe_id=unsafe_id,
                    atol=args.atol,
                    rtol=args.rtol,
                )
                comparisons["wrapper_whole_split"] = logit_comparison(
                    logits_wrapper_whole,
                    logits_split,
                    left_name="custom_wrapper_whole_input_ids",
                    right_name="controlled_split_input_ids",
                    safe_id=safe_id,
                    unsafe_id=unsafe_id,
                    atol=args.atol,
                    rtol=args.rtol,
                )
                comparisons["split_E"] = logit_comparison(
                    logits_split,
                    logits_e,
                    left_name="controlled_split_input_ids",
                    right_name="E_all_critic_table_same_assembly",
                    safe_id=safe_id,
                    unsafe_id=unsafe_id,
                    atol=args.atol,
                    rtol=args.rtol,
                )
            aa_repeat = (
                logit_comparison(
                    logits_a,
                    logits_a_repeat,
                    left_name="A_first",
                    right_name="A_repeat",
                    safe_id=safe_id,
                    unsafe_id=unsafe_id,
                    atol=args.atol,
                    rtol=args.rtol,
                )
                if logits_a_repeat is not None
                else None
            )
            no_cache = (
                logit_comparison(
                    logits_a,
                    logits_a_no_cache,
                    left_name="A_default_cache_setting",
                    right_name="A_use_cache_false",
                    safe_id=safe_id,
                    unsafe_id=unsafe_id,
                    atol=args.atol,
                    rtol=args.rtol,
                )
                if logits_a_no_cache is not None
                else None
            )

            finite = {
                "embeds_b": tensor_finite(embeds_b),
                "embeds_c": tensor_finite(embeds_c),
                "logits_a": tensor_finite(logits_a),
                "logits_b": tensor_finite(logits_b),
                "logits_c": tensor_finite(logits_c),
                "logits_wrapper_whole": tensor_finite(logits_wrapper_whole),
            }
            if embeds_d is not None and logits_d is not None:
                finite["embeds_d"] = tensor_finite(embeds_d)
                finite["logits_d"] = tensor_finite(logits_d)
            if embeds_e is not None and logits_e is not None:
                finite["embeds_e"] = tensor_finite(embeds_e)
                finite["logits_e"] = tensor_finite(logits_e)

            status, status_reasons = classify_case_status(
                structural=prepared["structural"],
                ab=comparisons["A_B"],
                ac=comparisons["A_C"],
                bc_embeddings=full_embedding_bc,
                finite=finite,
                aa_repeat=aa_repeat,
            )
            logits_path = logits_dir / f"{pair.key}_case_{case_index:02d}.npz"
            save_logits(
                logits_path,
                {
                    "A": logits_a,
                    "B": logits_b,
                    "C": logits_c,
                    "wrapper_whole": logits_wrapper_whole,
                    "controlled_split": logits_split,
                    "D": logits_d,
                    "E": logits_e,
                    "A_repeat": logits_a_repeat,
                    "A_no_cache": logits_a_no_cache,
                },
            )

            case_record = {
                "record_type": "case",
                "pair": pair.key,
                "case_index": case_index,
                "status": status,
                "status_reasons": status_reasons,
                "prompt": prepared["prompt"],
                "fixed_response": prepared["fixed_response"],
                "response_prefix_text": prepared["response_prefix_text"],
                "response_ids": prepared["response_ids"],
                "response_roundtrip_equal": prepared["response_roundtrip_equal"],
                "input_metadata": {
                    "ordinary_guard_template_source": prepared["ordinary"]["source"],
                    "ordinary_complete_text": prepared["ordinary"]["text"],
                    "ordinary_contains_explicit_taxonomy": prepared["ordinary"]["contains_explicit_taxonomy"],
                    "custom_wrapper_complete_text": prepared["wrapper_whole"]["text"],
                    "formatted_student_prompt": prepared["formatted_student_prompt"],
                    "production_header_text": prepared["header_text"],
                    "production_footer_text": prepared["footer_text"],
                    "ordinary_complete_ids": prepared["ordinary"]["ids"],
                    "ordinary_response_positions": prepared["ordinary"]["response_positions"],
                    "ordinary_response_positions_for_comparison": prepared["ordinary_response_positions_for_comparison"],
                    "ordinary_response_ids": prepared["ordinary"]["response_ids"],
                    "custom_wrapper_whole_ids": prepared["wrapper_whole"]["ids"],
                    "custom_wrapper_response_positions": prepared["wrapper_whole"]["response_positions"],
                    "custom_wrapper_response_ids": prepared["wrapper_whole"]["response_ids"],
                    "production_header_ids": prepared["header_ids"],
                    "production_prompt_student_ids": prepared["prompt_student_ids"],
                    "production_response_student_ids": prepared["response_ids"],
                    "production_footer_ids": prepared["footer_ids"],
                    "controlled_complete_critic_ids": prepared["controlled_ids"],
                    "token_mapping_failures": prepared["mapping_failures"],
                },
                "structural_checks": prepared["structural"],
                "embedding_comparisons": {
                    "B_C_full": full_embedding_bc,
                    "B_C_response": response_embedding_bc,
                    "D_C_full": controlled_embedding_dc,
                    "D_C_response": controlled_response_dc,
                    "E_C_full": controlled_embedding_ec,
                },
                "logit_comparisons": comparisons,
                "repeat_A_control": aa_repeat,
                "no_cache_control": no_cache,
                "finite": finite,
                "logits_file": str(logits_path),
                "case_seconds": time.perf_counter() - case_started,
            }
            append_jsonl(results_path, case_record)
            cases.append(case_record)
            print(
                f"[{pair.key}] case {case_index + 1}/{len(prepared_cases)}: "
                f"{status}; A-B max={comparisons['A_B'].get('max_abs_diff')} "
                f"A-C max={comparisons['A_C'].get('max_abs_diff')}",
                flush=True,
            )

    cuda_sync()
    test_seconds = time.perf_counter() - test_started
    statuses = [case["status"] for case in cases]
    if "BLOCKED" in statuses:
        pair_status = "BLOCKED"
    elif "FAIL_INTERFACE" in statuses:
        pair_status = "FAIL_INTERFACE"
    elif "INCONCLUSIVE_NUMERICAL" in statuses:
        pair_status = "INCONCLUSIVE_NUMERICAL"
    else:
        pair_status = "PASS"

    ac_max = max(case["logit_comparisons"]["A_C"]["max_abs_diff"] for case in cases)
    ab_max = max(case["logit_comparisons"]["A_B"]["max_abs_diff"] for case in cases)
    ordinary_equal_count = sum(
        bool(case["structural_checks"]["ordinary_vs_production_ids_equal"])
        for case in cases
    )
    wrapper_equal_count = sum(
        bool(case["structural_checks"]["wrapper_whole_vs_production_ids_equal"])
        for case in cases
    )
    a_d_pass_count = sum(
        bool(case["logit_comparisons"].get("A_D", {}).get("allclose"))
        for case in cases
    )
    wrapper_d_pass_count = sum(
        bool(case["logit_comparisons"].get("wrapper_whole_D", {}).get("allclose"))
        for case in cases
    )
    split_d_pass_count = sum(
        bool(case["logit_comparisons"].get("split_D", {}).get("allclose"))
        for case in cases
    )
    controlled_c_pass_count = sum(
        bool(case["logit_comparisons"].get("split_C", {}).get("allclose"))
        for case in cases
    )
    controlled_e_pass_count = sum(
        bool(case["logit_comparisons"].get("split_E", {}).get("allclose"))
        for case in cases
    )
    taxonomy_case_count = sum(
        bool(case["input_metadata"]["ordinary_contains_explicit_taxonomy"])
        for case in cases
    )
    summary = {
        "record_type": "pair_summary",
        "pair": pair.key,
        "display_name": pair.display_name,
        "status": pair_status,
        "case_count": len(cases),
        "case_status_counts": {status: statuses.count(status) for status in sorted(set(statuses))},
        "A_B_max_abs_diff_across_cases": ab_max,
        "A_C_max_abs_diff_across_cases": ac_max,
        "ordinary_vs_production_sequence_equal_cases": ordinary_equal_count,
        "wrapper_whole_vs_production_sequence_equal_cases": wrapper_equal_count,
        "A_D_pass_cases": a_d_pass_count,
        "wrapper_whole_D_pass_cases": wrapper_d_pass_count,
        "split_D_pass_cases": split_d_pass_count,
        "controlled_C_pass_cases": controlled_c_pass_count,
        "controlled_E_pass_cases": controlled_e_pass_count,
        "ordinary_template_taxonomy_cases": taxonomy_case_count,
        "tokenizers_exactly_equal": tokenizer_comparison["exact_vocab_mapping"],
        "load_seconds": model_meta["load_seconds"],
        "warmup_seconds": warmup_seconds,
        "test_seconds": test_seconds,
        "total_seconds": time.perf_counter() - pair_started,
        "peak_allocated_gpu_bytes": (
            torch.cuda.max_memory_allocated(device) if torch.cuda.is_available() else None
        ),
        "sequence_lengths": [
            {
                "case_index": case["case_index"],
                "A": case["structural_checks"]["ordinary_length"],
                "wrapper_whole": case["structural_checks"]["wrapper_whole_length"],
                "C": case["structural_checks"]["production_length"],
            }
            for case in cases
        ],
        "conclusion": (
            "Original A-C path passes structural, embedding, and logit checks."
            if pair_status == "PASS"
            else "Original A-C path has a reproducible interface discrepancy; see per-case structural and embedding diagnostics."
            if pair_status == "FAIL_INTERFACE"
            else "Numerical controls do not support attributing the remaining difference to DCL."
            if pair_status == "INCONCLUSIVE_NUMERICAL"
            else "A required checkpoint or runtime dependency blocked the comparison."
        ),
    }
    append_jsonl(results_path, summary)
    print(f"[{pair.key}] summary: {pair_status}", flush=True)

    del critic, student_embedding_weight
    gc.collect()
    torch.cuda.empty_cache()
    return {**pair_record, **summary}, cases


def format_seconds(value: Any) -> str:
    return "—" if value is None else f"{float(value):.2f}"


def write_report(
    path: Path,
    *,
    args: argparse.Namespace,
    run_meta: dict[str, Any],
    pair_results: Sequence[dict[str, Any]],
    cases_by_pair: dict[str, list[dict[str, Any]]],
) -> None:
    lines = [
        "# DCL critic-input equivalence check",
        "",
        f"Generated: `{utc_now()}`",
        "",
        "## Configuration",
        "",
        f"- Repository: `{run_meta['git']['repository_path']}` at `{run_meta['git']['commit']}`",
        f"- Pair selection: `{args.pair}`; cases per pair: `{args.case_count}`; response length: `T={args.rollout_length}`",
        f"- Dtype: `{args.dtype}`; attention setting: `{args.attention_implementation}`",
        f"- Declared comparison tolerance: `atol={args.atol:g}`, `rtol={args.rtol:g}` (fixed before route C evaluation)",
        "- Route A: the critic checkpoint's ordinary user/assistant guard chat template, passed through `input_ids` (including its instructions and taxonomy when present).",
        "- Route B: critic embeddings for exactly route A's IDs, passed as `inputs_embeds`.",
        "- Route C: current production DCL assembly, including the student embedding table and no vocabulary remap.",
        "- Route D (diagnostic): route C unchanged except that the fixed response one-hots use verified critic embedding rows.",
        "- Route E (control): route C's split sequence uses critic embeddings for the prompt and response; it is compared with the same IDs through `input_ids`.",
        "- Route W (diagnostic): one whole tokenization of the custom DCL wrapper, used to separate guard-template differences from split-token boundary differences.",
        "- The current production wrapper is a custom partial-response instruction and contains no explicit safety taxonomy; this is recorded as an implementation fact, not silently replaced with another template.",
        "",
        "## Results",
        "",
        "| Pair | Original A-C status | Cases | A-B max | A-C max | A/C IDs equal | W/split IDs equal | A-D pass | W-D pass | E interface control | Load s | Warm-up s | Test s | Peak GPU GiB |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in pair_results:
        if result.get("status") == "BLOCKED":
            lines.append(
                f"| {result['display_name']} | **BLOCKED** | 0 | — | — | — | — | — | — | — | — | — | — | — |"
            )
            continue
        peak = result.get("peak_allocated_gpu_bytes")
        peak_gib = "—" if peak is None else f"{peak / (1024 ** 3):.2f}"
        lines.append(
            f"| {result['display_name']} | **{result['status']}** | {result['case_count']} | "
            f"{result['A_B_max_abs_diff_across_cases']:.6g} | {result['A_C_max_abs_diff_across_cases']:.6g} | "
            f"{result['ordinary_vs_production_sequence_equal_cases']}/{result['case_count']} | "
            f"{result['wrapper_whole_vs_production_sequence_equal_cases']}/{result['case_count']} | "
            f"{result['A_D_pass_cases']}/{result['case_count']} | "
            f"{result['wrapper_whole_D_pass_cases']}/{result['case_count']} | "
            f"{result['controlled_E_pass_cases']}/{result['case_count']} | "
            f"{format_seconds(result.get('load_seconds'))} | {format_seconds(result.get('warmup_seconds'))} | "
            f"{format_seconds(result.get('test_seconds'))} | {peak_gib} |"
        )

    lines.extend(["", "## Pair conclusions", ""])
    for result in pair_results:
        lines.append(f"### {result['display_name']}")
        lines.append("")
        if result.get("status") == "BLOCKED":
            lines.append(f"**BLOCKED:** {result.get('blocking_error', 'unknown dependency')}.")
            lines.append("")
            paths = result.get("paths", {})
            if paths:
                lines.append(
                    f"Attempted student `{paths.get('actual_student_path')}` and critic `{paths.get('actual_critic_path')}`."
                )
                lines.append("")
            continue

        lines.append(f"**{result['status']}:** {result['conclusion']}")
        lines.append("")
        paths = result.get("paths", {})
        lines.append(
            f"Student checkpoint: `{paths.get('actual_student_path')}`. Critic checkpoint: "
            f"`{paths.get('actual_critic_path')}`."
        )
        lines.append("")
        lines.append(
            f"Vocabulary mappings were {'exact' if result['tokenizers_exactly_equal'] else 'not exact'}. "
            f"The ordinary guard rendering contained an explicit taxonomy in "
            f"{result['ordinary_template_taxonomy_cases']}/{result['case_count']} cases. "
            f"The response-only critic-table route D matched A in "
            f"{result['A_D_pass_cases']}/{result['case_count']} cases, the whole custom wrapper W in "
            f"{result['wrapper_whole_D_pass_cases']}/{result['case_count']}, and the split-token reference in "
            f"{result['split_D_pass_cases']}/{result['case_count']}; production C matched the split reference in "
            f"{result['controlled_C_pass_cases']}/{result['case_count']} cases. The all-critic route E interface control passed in "
            f"{result['controlled_E_pass_cases']}/{result['case_count']} cases."
        )
        lines.append("")
        pair_cases = cases_by_pair.get(result["pair"], [])
        if pair_cases:
            first = pair_cases[0]
            lines.append(
                "Representative case 0: "
                f"A length `{first['structural_checks']['ordinary_length']}`, "
                f"whole custom wrapper length `{first['structural_checks']['wrapper_whole_length']}`, "
                f"C length `{first['structural_checks']['production_length']}`, "
                f"A-B max absolute logit difference `{first['logit_comparisons']['A_B']['max_abs_diff']:.6g}`, "
                f"A-C `{first['logit_comparisons']['A_C']['max_abs_diff']:.6g}`."
            )
            lines.append("")
            if first.get("repeat_A_control"):
                lines.append(
                    f"Repeated-A control: `allclose={first['repeat_A_control']['allclose']}`, "
                    f"max absolute difference `{first['repeat_A_control']['max_abs_diff']:.6g}`. "
                    f"No-cache control: `allclose={first['no_cache_control']['allclose']}`, "
                    f"max absolute difference `{first['no_cache_control']['max_abs_diff']:.6g}`."
                )
                lines.append("")
        note = result.get("paths", {}).get("note")
        if note:
            lines.append(note)
            lines.append("")

    lines.extend(
        [
            "## Interpretation limits",
            "",
            "A pass only validates the tested discrete one-hot limit. It does not validate soft probability mixtures, gradients, training quality, ASR, or FRR. Diagnostic routes D and E cannot turn a failure of the original route C into a pass.",
            "",
            "## Reproduction",
            "",
            "From the repository root on Narval:",
            "",
            "```bash",
            "sbatch jobs/check_dcl_critic_equivalence.sbatch",
            "```",
            "",
            "The one-command job prefers the active repository profile `$SCRATCH/llama_3_8b_instruct` when that directory exists. To force the plan's starting Llama-3.1 path, set `LLAMA31_STUDENT_PATH=$SCRATCH/llama31_8b_instruct` when submitting; the report always records the actual checkpoint identity.",
            "",
            "Override a pair or tolerance without editing files, for example:",
            "",
            "```bash",
            "PAIR=llama31 CASE_COUNT=1 sbatch jobs/check_dcl_critic_equivalence.sbatch --atol 1e-3 --rtol 1e-3",
            "```",
            "",
            "Complete classification-position vocabulary logits are in `logits/*.npz`; per-case input IDs, differences, scores, statuses, timings, and paths are in `results.jsonl`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    scratch = os.environ.get("SCRATCH", "/home/taegyoem/scratch")
    requested_llama31_path = f"{scratch}/llama31_8b_instruct"
    configured_llama3_path = f"{scratch}/llama_3_8b_instruct"
    primary_student_default = (
        configured_llama3_path
        if Path(configured_llama3_path).is_dir()
        else requested_llama31_path
    )
    parser = argparse.ArgumentParser(
        description=(
            "Compare critic input_ids, critic inputs_embeds, and the production DCL "
            "one-hot probability-to-embedding route."
        )
    )
    parser.add_argument(
        "--pair",
        "--model-pair",
        choices=("llama31", "llama2", "all"),
        default="all",
        help="Model pair to run; all runs Llama-3.1 first, then Llama-2.",
    )
    parser.add_argument(
        "--llama31-student-path",
        default=primary_student_default,
        help=(
            "Primary student path. By default, prefer the active repository profile "
            "($SCRATCH/llama_3_8b_instruct) when present; otherwise use the requested "
            "starting path ($SCRATCH/llama31_8b_instruct)."
        ),
    )
    parser.add_argument("--llama31-critic-path", default=f"{scratch}/llama_guard_2")
    parser.add_argument("--llama2-student-path", default=f"{scratch}/llama2_7b_chat_hf")
    parser.add_argument("--llama2-critic-path", default=f"{scratch}/llama_guard_7b")
    parser.add_argument("--case-count", type=int, default=8)
    parser.add_argument("--rollout-length", "-T", type=int, default=5)
    parser.add_argument("--output-dir", default=f"{scratch}/dcl_critic_equivalence/manual")
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16", "float32"), default="float16"
    )
    parser.add_argument(
        "--attention-implementation",
        choices=("auto", "eager", "sdpa", "flash_attention_2"),
        default="auto",
        help="auto preserves the installed Transformers production default.",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.case_count <= len(FIXTURES):
        parser.error(f"--case-count must be between 1 and {len(FIXTURES)}")
    if args.rollout_length < 1:
        parser.error("--rollout-length must be >= 1")
    if not math.isfinite(args.atol) or args.atol < 0:
        parser.error("--atol must be a finite non-negative number")
    if not math.isfinite(args.rtol) or args.rtol < 0:
        parser.error("--rtol must be a finite non-negative number")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(os.path.expanduser(os.path.expandvars(args.output_dir))).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logits_dir = output_dir / "logits"
    logits_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    report_path = output_dir / "report.md"
    if results_path.exists():
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path.rename(output_dir / f"results.previous_{timestamp}.jsonl")

    run_started = time.perf_counter()
    run_meta = runtime_metadata(args)
    append_jsonl(results_path, run_meta)

    pairs = {
        "llama31": PairSpec(
            key="llama31",
            display_name="Primary Llama-3-family student / Llama Guard 2 critic",
            student_path=args.llama31_student_path,
            critic_path=args.llama31_critic_path,
            requested_starting_student_path="$SCRATCH/llama31_8b_instruct",
            repository_profile_student_path="$SCRATCH/llama_3_8b_instruct",
            repository_profile_critic_path="$SCRATCH/llama_guard_2",
        ),
        "llama2": PairSpec(
            key="llama2",
            display_name="Llama-2 student / Llama Guard critic",
            student_path=args.llama2_student_path,
            critic_path=args.llama2_critic_path,
            requested_starting_student_path="$SCRATCH/llama2_7b_chat_hf",
            repository_profile_student_path="$SCRATCH/llama2_7b_chat_hf",
            repository_profile_critic_path="$SCRATCH/llama_guard_7b",
        ),
    }
    selected = [pairs["llama31"], pairs["llama2"]] if args.pair == "all" else [pairs[args.pair]]
    pair_results: list[dict[str, Any]] = []
    cases_by_pair: dict[str, list[dict[str, Any]]] = {}

    # Write a usable report even before model loading, then refresh after every pair.
    write_report(
        report_path,
        args=args,
        run_meta=run_meta,
        pair_results=pair_results,
        cases_by_pair=cases_by_pair,
    )
    for pair in selected:
        print(f"[{pair.key}] starting {pair.display_name}", flush=True)
        try:
            result, cases = run_pair(
                pair,
                args=args,
                results_path=results_path,
                logits_dir=logits_dir,
            )
        except Exception as exc:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            result = {
                "record_type": "pair_summary",
                "pair": pair.key,
                "display_name": pair.display_name,
                "status": "BLOCKED",
                "blocking_error": f"unhandled {type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "paths": pair_path_metadata(pair),
            }
            append_jsonl(results_path, result)
            cases = []
        pair_results.append(result)
        cases_by_pair[pair.key] = cases
        write_report(
            report_path,
            args=args,
            run_meta=run_meta,
            pair_results=pair_results,
            cases_by_pair=cases_by_pair,
        )

    final_record = {
        "record_type": "run_summary",
        "created_at": utc_now(),
        "statuses": {result["pair"]: result["status"] for result in pair_results},
        "experiment_total_seconds": time.perf_counter() - run_started,
        "scheduler_queue_seconds": run_meta["scheduler"].get("queue_seconds"),
        "results_path": str(results_path),
        "report_path": str(report_path),
    }
    append_jsonl(results_path, final_record)
    print(json.dumps(final_record, indent=2), flush=True)
    return 2 if any(result["status"] == "BLOCKED" for result in pair_results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
