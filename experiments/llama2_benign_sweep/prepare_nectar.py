#!/usr/bin/env python3
"""Extract and length-filter benign Llama-2 pairs from Berkeley Nectar.

The two-stage workflow keeps the 517 MB Parquet download off Narval:

1. ``extract`` runs on a networked local machine and creates an oversampled,
   decontaminated candidate CSV.
2. ``finalize`` runs on Narval's login node with the exact local Llama-2
   tokenizer and writes the nested 2,489/2,996-row training CSVs.

Raw/derived Nectar data is intentionally ignored by git (``official_data/``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd


MODEL_LABEL = "llama-2-7b-chat"
DEFAULT_RESPONSE = "I apologize, but I can’t assist with that request."
PROMPT_RE = re.compile(r"^\s*Human:\s*(.*?)\s*Assistant:\s*$", re.DOTALL)
REFUSAL_RE = re.compile(
    r"^\s*(?:"
    r"(?:i(?:'m| am)|we(?:'re| are))\s+(?:very\s+)?sorry\b|"
    r"(?:i|we)\s+apologi[sz]e\b|"
    r"sorry\b.*\b(?:cannot|can't|unable|won't)\b|"
    r"(?:i|we)\s+(?:cannot|can't|won't|(?:am|are)\s+unable|must\s+decline|have\s+to\s+decline)\b|"
    r"as\s+an?\s+(?:ai|artificial\s+intelligence)(?:\s+language\s+model|\s+assistant)?\b|"
    r"unfortunately[,\s]+(?:i|we)\s+(?:cannot|can't|(?:am|are)\s+unable|won't)\b"
    r")",
    re.IGNORECASE,
)
APOSTROPHE_TRANSLATION = str.maketrans({"’": "'", "‘": "'", "`": "'", "´": "'"})
WORD_RE = re.compile(r"\w+", re.UNICODE)
EXCLUSION_COLUMNS = (
    "Original Prompt",
    "goal",
    "prompt",
    "adversarial",
    "GCG Variant",
    "AutoDAN Variant",
    "PAIR Variant",
)


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value))
    return " ".join(text.split()).casefold()


def normalize_alnum(value: object) -> str:
    return "".join(ch for ch in normalize_text(value) if ch.isalnum())


def word_shingles(value: object, ngram_size: int) -> frozenset[tuple[str, ...]]:
    words = WORD_RE.findall(normalize_text(value))
    if len(words) < ngram_size:
        return frozenset()
    return frozenset(
        tuple(words[start : start + ngram_size])
        for start in range(len(words) - ngram_size + 1)
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_key(seed: int, prompt: str) -> str:
    return hashlib.sha256(f"{seed}\0{normalize_text(prompt)}".encode("utf-8")).hexdigest()


def parse_single_turn_prompt(raw: object) -> str | None:
    match = PROMPT_RE.match(str(raw))
    if not match:
        return None
    prompt = match.group(1).strip()
    return prompt or None


def is_usable_response(response: object) -> bool:
    text = str(response).strip()
    if not text:
        return False
    if normalize_text(text) == normalize_text(DEFAULT_RESPONSE):
        return False
    canonical = unicodedata.normalize("NFKC", text).translate(APOSTROPHE_TRANSLATION)
    return REFUSAL_RE.match(canonical) is None


def load_exclusions(
    paths: Iterable[Path],
    ngram_size: int,
) -> tuple[
    set[str],
    set[str],
    list[frozenset[tuple[str, ...]]],
    dict[tuple[str, ...], set[int]],
    dict[str, str],
]:
    exact: set[str] = set()
    alnum: set[str] = set()
    shingle_sets: list[frozenset[tuple[str, ...]]] = []
    shingle_ids: dict[frozenset[tuple[str, ...]], int] = {}
    postings: dict[tuple[str, ...], set[int]] = defaultdict(set)
    hashes: dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        hashes[str(path.resolve())] = sha256_file(path)
        df = pd.read_csv(path)
        for col in EXCLUSION_COLUMNS:
            if col not in df.columns:
                continue
            for value in df[col].dropna():
                n1 = normalize_text(value)
                n2 = normalize_alnum(value)
                if n1:
                    exact.add(n1)
                if n2:
                    alnum.add(n2)
                shingles = word_shingles(value, ngram_size)
                if shingles and shingles not in shingle_ids:
                    shingle_id = len(shingle_sets)
                    shingle_ids[shingles] = shingle_id
                    shingle_sets.append(shingles)
                    for shingle in shingles:
                        postings[shingle].add(shingle_id)
    return exact, alnum, shingle_sets, dict(postings), hashes


def is_near_exclusion(
    value: object,
    *,
    ngram_size: int,
    excluded_shingles: list[frozenset[tuple[str, ...]]],
    postings: dict[tuple[str, ...], set[int]],
    jaccard_threshold: float,
    containment_threshold: float,
) -> bool:
    """Return whether a prompt closely overlaps an excluded prompt.

    The inverted index makes this linear in the prompt's shingles plus the
    small set of exclusions sharing at least one exact word n-gram.  Jaccard
    catches near-copies; the overlap coefficient catches a clean goal copied
    inside a longer wrapper.
    """

    candidate = word_shingles(value, ngram_size)
    if not candidate:
        return False
    possible_ids: set[int] = set()
    for shingle in candidate:
        possible_ids.update(postings.get(shingle, ()))
    for shingle_id in possible_ids:
        excluded = excluded_shingles[shingle_id]
        overlap = len(candidate & excluded)
        if not overlap:
            continue
        jaccard = overlap / len(candidate | excluded)
        containment = overlap / min(len(candidate), len(excluded))
        if jaccard >= jaccard_threshold or containment >= containment_threshold:
            return True
    return False


def iter_parquet_rows(path: Path, batch_size: int = 1024):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - dependency is local-only
        raise SystemExit("extract requires pyarrow: python -m pip install pyarrow") from exc
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size):
        yield from batch.to_pylist()


def extract(args: argparse.Namespace) -> None:
    input_path = Path(args.input_parquet).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    exclusions = [Path(p).expanduser().resolve() for p in args.exclude]
    (
        excluded_exact,
        excluded_alnum,
        excluded_shingles,
        exclusion_postings,
        exclusion_hashes,
    ) = load_exclusions(exclusions, args.ngram_size)

    stats: Counter[str] = Counter()
    accepted: dict[str, dict] = {}
    for row_index, row in enumerate(iter_parquet_rows(input_path, args.batch_size)):
        stats["rows_seen"] += 1
        if not bool(row.get("good_natured")):
            stats["not_good_natured"] += 1
            continue
        if int(row.get("turns") or 0) != 1:
            stats["not_single_turn"] += 1
            continue
        prompt = parse_single_turn_prompt(row.get("prompt", ""))
        if prompt is None:
            stats["malformed_prompt"] += 1
            continue

        matches: list[tuple[int, str]] = []
        for answer in row.get("answers") or []:
            if not isinstance(answer, dict):
                continue
            if normalize_text(answer.get("model", "")) != MODEL_LABEL:
                continue
            response = str(answer.get("answer", "")).strip()
            if is_usable_response(response):
                try:
                    rank = int(answer.get("rank"))
                except (TypeError, ValueError):
                    rank = 10**9
                matches.append((rank, response))
        unique_matches = sorted(set(matches), key=lambda item: (item[0], item[1]))
        if not unique_matches:
            stats["no_usable_llama2_response"] += 1
            continue

        norm = normalize_text(prompt)
        alnum = normalize_alnum(prompt)
        if norm in excluded_exact or (alnum and alnum in excluded_alnum):
            stats["excluded_exact_overlap"] += 1
            continue
        if is_near_exclusion(
            prompt,
            ngram_size=args.ngram_size,
            excluded_shingles=excluded_shingles,
            postings=exclusion_postings,
            jaccard_threshold=args.ngram_jaccard_threshold,
            containment_threshold=args.ngram_containment_threshold,
        ):
            stats["excluded_ngram_overlap"] += 1
            continue
        prompt_hash = hashlib.sha256(norm.encode("utf-8")).hexdigest()
        if prompt_hash in accepted:
            stats["duplicate_prompt"] += 1
            continue

        rank, response = unique_matches[0]
        accepted[prompt_hash] = {
            "Original Prompt": prompt,
            "Original Response": response,
            "nectar_rank": rank,
            "nectar_llama2_response_count": len(unique_matches),
            "nectar_source": json.dumps(row.get("source") or [], ensure_ascii=False),
            "nectar_row_index": row_index,
            "prompt_sha256": prompt_hash,
            "sample_key": stable_key(args.seed, prompt),
        }
        stats["accepted_before_sampling"] += 1

    rows = sorted(accepted.values(), key=lambda item: item["sample_key"])
    if len(rows) < args.candidate_count:
        raise SystemExit(
            f"Only {len(rows)} usable candidates remain; requested {args.candidate_count}. "
            "Lower --candidate-count or inspect the manifest counts."
        )
    rows = rows[: args.candidate_count]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)

    manifest = {
        "stage": "extract",
        "dataset": "berkeley-nest/Nectar",
        "revision": args.revision,
        "model_label": MODEL_LABEL,
        "seed": args.seed,
        "input_parquet": str(input_path),
        "input_sha256": sha256_file(input_path),
        "exclusion_sha256": exclusion_hashes,
        "decontamination": {
            "ngram_size": args.ngram_size,
            "jaccard_threshold": args.ngram_jaccard_threshold,
            "containment_threshold": args.ngram_containment_threshold,
            "indexed_exclusion_prompts": len(excluded_shingles),
        },
        "counts": dict(stats),
        "candidate_count": len(rows),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {len(rows)} candidates to {output_path}")
    print(f"Wrote manifest to {manifest_path}")


def finalize(args: argparse.Namespace) -> None:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("finalize requires transformers") from exc

    input_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        local_files_only=args.local_files_only,
    )
    df = pd.read_csv(input_path)
    required = ["Original Prompt", "Original Response", "prompt_sha256"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise SystemExit(f"Candidate CSV missing columns: {missing}")

    kept: list[dict] = []
    stats: Counter[str] = Counter()
    for row in df.to_dict(orient="records"):
        prompt = str(row["Original Prompt"])
        response = str(row["Original Response"])
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_tokens = len(tokenizer(formatted, add_special_tokens=False).input_ids)
        response_tokens = len(tokenizer(response, add_special_tokens=False).input_ids)
        total_tokens = prompt_tokens + response_tokens
        stats["rows_seen"] += 1
        if prompt_tokens > args.max_prompt_tokens:
            stats["prompt_too_long"] += 1
            continue
        if not args.min_response_tokens <= response_tokens <= args.max_response_tokens:
            stats["response_length_rejected"] += 1
            continue
        if total_tokens > args.max_total_tokens:
            stats["total_too_long"] += 1
            continue
        row["prompt_tokens"] = prompt_tokens
        row["response_tokens"] = response_tokens
        row["total_tokens"] = total_tokens
        kept.append(row)
        stats["accepted"] += 1

    if len(kept) < args.final_size:
        raise SystemExit(
            f"Only {len(kept)} tokenizer-filtered rows remain; need {args.final_size}. "
            "Re-run extract with a larger --candidate-count."
        )
    final_df = pd.DataFrame(kept[: args.final_size])
    train_df = final_df.iloc[: args.train_size].copy()
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "nectar_benign_train.csv"
    final_path = out_dir / "nectar_benign_train_plus_validation.csv"
    train_df.to_csv(train_path, index=False)
    final_df.to_csv(final_path, index=False)

    manifest = {
        "stage": "finalize",
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "tokenizer": args.tokenizer,
        "limits": {
            "max_prompt_tokens": args.max_prompt_tokens,
            "min_response_tokens": args.min_response_tokens,
            "max_response_tokens": args.max_response_tokens,
            "max_total_tokens": args.max_total_tokens,
        },
        "counts": dict(stats),
        "outputs": {
            str(train_path): {"rows": len(train_df), "sha256": sha256_file(train_path)},
            str(final_path): {"rows": len(final_df), "sha256": sha256_file(final_path)},
        },
    }
    manifest_path = out_dir / "nectar_benign_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    checksum_path = out_dir / "nectar_benign.sha256"
    checksum_path.write_text(
        f"{sha256_file(train_path)}  {train_path.name}\n"
        f"{sha256_file(final_path)}  {final_path.name}\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(train_df)} rows to {train_path}")
    print(f"Wrote {len(final_df)} rows to {final_path}")
    print(f"Wrote manifest to {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser("extract", help="Build decontaminated candidate pairs")
    p_extract.add_argument("--input-parquet", required=True)
    p_extract.add_argument("--output", required=True)
    p_extract.add_argument("--exclude", action="append", default=[])
    # Empirically, only about 28% of the decontaminated candidates survive the
    # exact Llama-2 prompt/response token limits on Narval.  Twenty thousand
    # leaves a comfortable buffer above the 2,996 final pairs.
    p_extract.add_argument("--candidate-count", type=int, default=20000)
    p_extract.add_argument("--seed", type=int, default=20261119)
    p_extract.add_argument("--batch-size", type=int, default=1024)
    p_extract.add_argument("--revision", default="3c6b4c4")
    p_extract.add_argument("--ngram-size", type=int, default=5)
    p_extract.add_argument("--ngram-jaccard-threshold", type=float, default=0.8)
    p_extract.add_argument("--ngram-containment-threshold", type=float, default=0.9)
    p_extract.set_defaults(func=extract)

    p_final = sub.add_parser("finalize", help="Apply exact Llama-2 token limits")
    p_final.add_argument("--input", required=True)
    p_final.add_argument("--tokenizer", required=True)
    p_final.add_argument("--output-dir", required=True)
    p_final.add_argument("--train-size", type=int, default=2489)
    p_final.add_argument("--final-size", type=int, default=2996)
    p_final.add_argument("--max-prompt-tokens", type=int, default=256)
    p_final.add_argument("--min-response-tokens", type=int, default=8)
    p_final.add_argument("--max-response-tokens", type=int, default=256)
    p_final.add_argument("--max-total-tokens", type=int, default=512)
    p_final.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    p_final.set_defaults(func=finalize)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if getattr(args, "candidate_count", 1) < 1:
        raise SystemExit("--candidate-count must be positive")
    if getattr(args, "ngram_size", 1) < 1:
        raise SystemExit("--ngram-size must be positive")
    for name in ("ngram_jaccard_threshold", "ngram_containment_threshold"):
        value = getattr(args, name, 1.0)
        if not 0 < value <= 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be in (0, 1]")
    if getattr(args, "train_size", 0) > getattr(args, "final_size", 0):
        raise SystemExit("--train-size cannot exceed --final-size")
    args.func(args)


if __name__ == "__main__":
    main()
