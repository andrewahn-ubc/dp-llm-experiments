#!/usr/bin/env python3
"""Merge PyRIT chunk CSVs and score ASR with HarmBench (same judge as eval.py)."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from eval.eval_helpers import classify_all_jb_safety  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chunks-dir", required=True, help="Dir with chunk_*.csv for one target_tag")
    p.add_argument("--output-csv", required=True)
    p.add_argument(
        "--harmbench-path",
        default=os.path.expandvars("$SCRATCH/harmbench_mistral_val_cls"),
    )
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--skip-score", action="store_true", help="Only merge, do not run HarmBench.")
    args = p.parse_args()

    chunks_dir = Path(args.chunks_dir)
    files = sorted(chunks_dir.glob("chunk_*.csv"))
    if not files:
        raise SystemExit(f"No chunk_*.csv under {chunks_dir}")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if "goal" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["goal"], keep="first").reset_index(drop=True)
        print(f"merged {before} rows → {len(df)} unique goals from {len(files)} chunks")

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.skip_score:
        df.to_csv(out, index=False)
        print("wrote", out)
        return

    # Ensure columns for classify_all_jb_safety(unseen_family=pyrit)
    if "PyRIT Variant" not in df.columns or "PyRIT Response" not in df.columns:
        raise SystemExit("Need PyRIT Variant + PyRIT Response columns")

    hb = os.path.expandvars(os.path.expanduser(args.harmbench_path))
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    tok = AutoTokenizer.from_pretrained(hb)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        hb, torch_dtype=dtype, device_map="auto", low_cpu_mem_usage=True
    )
    for param in model.parameters():
        param.requires_grad = False

    classify_all_jb_safety(
        df,
        batch_size=args.batch_size,
        guard_model=model,
        guard_tokenizer=tok,
        testing_mode="unseen-family",
        unseen_family="pyrit",
        output_file=str(out),
        GUARD_LLM_PATH=hb,
    )
    scored = pd.read_csv(out)
    asr = (scored["PyRIT Safety"].astype(str).str.lower() == "unsafe").mean()
    print(f"wrote {out}  pyrit_asr={asr:.4f}  n={len(scored)}")
    metrics = out.with_name(out.stem + "_metrics.tsv")
    metrics.write_text(f"metric\tvalue\npyrit_asr\t{asr}\nn\t{len(scored)}\n")
    print("wrote", metrics)


if __name__ == "__main__":
    main()
