#!/usr/bin/env python3
"""Preflight check for Jailbreak-R1 held-out eval (Llama-3 array).

Verifies paths, CSVs, tokenizer load, and PEFT adapter_config (no full GPU
weight load by default — use --load-weights on a GPU node if you want that).

  module load StdEnv/2023 python/3.11
  source $SCRATCH/venv/nanogcg/bin/activate
  python helper_scripts/perturbation/verify_jailbreak_r1_heldout_models.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _ok(msg: str) -> None:
    print(f"  OK  {msg}")


def _fail(msg: str, errors: list[str]) -> None:
    print(f"  FAIL {msg}")
    errors.append(msg)


def _warn(msg: str) -> None:
    print(f"  WARN {msg}")


def check_csv(path: Path, need_cols: list[str], errors: list[str]) -> None:
    import pandas as pd

    if not path.is_file():
        _fail(f"missing CSV: {path}", errors)
        return
    df = pd.read_csv(path, nrows=5)
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        _fail(f"{path.name}: missing columns {missing}; have {list(df.columns)}", errors)
        return
    full = pd.read_csv(path, usecols=need_cols)
    n = len(full)
    if "Jailbreak-R1 Variant" in need_cols:
        empty = (
            full["Jailbreak-R1 Variant"].isna()
            | full["Jailbreak-R1 Variant"].astype(str).str.strip().eq("")
        ).sum()
        if empty:
            _warn(f"{path.name}: {empty}/{n} empty Jailbreak-R1 Variant (eval will drop)")
        else:
            _ok(f"{path.name}: {n} rows, variants non-empty")
    else:
        _ok(f"{path.name}: {n} rows")


def check_full_hf(path: Path, label: str, errors: list[str], load_tok: bool) -> None:
    if not path.is_dir():
        _fail(f"{label}: not a directory: {path}", errors)
        return
    if not (path / "config.json").is_file():
        _fail(f"{label}: no config.json under {path}", errors)
        return
    has_tok = (path / "tokenizer.json").is_file() or (path / "tokenizer_config.json").is_file()
    if not has_tok:
        _fail(f"{label}: no tokenizer files under {path}", errors)
        return
    weights = list(path.glob("*.safetensors")) + list(path.glob("pytorch_model*.bin"))
    if not weights and not (path / "model.safetensors.index.json").is_file():
        _fail(f"{label}: no model weights under {path}", errors)
        return
    _ok(f"{label}: full HF dir ({path})")
    if load_tok:
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)
            _ok(f"{label}: tokenizer loads (vocab={len(tok)})")
        except Exception as e:
            _fail(f"{label}: tokenizer load failed: {e}", errors)


def check_peft(adapter: Path, base: Path, label: str, errors: list[str], load_tok: bool) -> None:
    if not adapter.is_dir():
        _fail(f"{label}: adapter dir missing: {adapter}", errors)
        return
    cfg = adapter / "adapter_config.json"
    if not cfg.is_file():
        _fail(f"{label}: no adapter_config.json (is this a full model?): {adapter}", errors)
        return
    w = list(adapter.glob("adapter_model*.safetensors")) + list(adapter.glob("adapter_model.bin"))
    if not w:
        _fail(f"{label}: no adapter_model weights in {adapter}", errors)
        return
    try:
        meta = json.loads(cfg.read_text())
        base_in_cfg = meta.get("base_model_name_or_path")
        _ok(f"{label}: PEFT adapter ({adapter.name}); config base={base_in_cfg!r}")
    except Exception as e:
        _fail(f"{label}: bad adapter_config.json: {e}", errors)
        return
    check_full_hf(base, f"{label} base", errors, load_tok=load_tok)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scratch", default=os.environ.get("SCRATCH", ""))
    p.add_argument("--repo", default=None, help="dp-llm-experiments root")
    p.add_argument("--epoch", type=int, default=int(os.environ.get("EPOCH", "2")))
    p.add_argument(
        "--load-tokenizer",
        action="store_true",
        default=True,
        help="Try AutoTokenizer.from_pretrained (default on)",
    )
    p.add_argument("--no-load-tokenizer", action="store_false", dest="load_tokenizer")
    p.add_argument(
        "--load-weights",
        action="store_true",
        help="Also load each model on CPU (slow; use on a GPU/compute node)",
    )
    args = p.parse_args()

    scratch = Path(args.scratch or os.environ.get("SCRATCH", "")).expanduser()
    if not scratch.is_dir():
        print("ERROR: set SCRATCH or --scratch", file=sys.stderr)
        return 2
    repo = Path(args.repo or f"{scratch}/dp-llm-experiments").expanduser()

    base_l3 = Path(os.environ.get("BASE_L3", scratch / "llama_3_8b_instruct"))
    mixat = Path(os.environ.get("MIXAT_PATH", scratch / "mixat"))
    door = Path(os.environ.get("DOOR_PATH", scratch / "door"))
    delman = Path(os.environ.get("DELMAN_PATH", scratch / "delman_llama31_8b_instruct"))
    ck = Path(os.environ.get("CHECKPOINT_ROOT", scratch / "dp-llm-sweep"))
    dcl1 = ck / f"l3_run_lr2e-05_lam1_eps-1_finetuned_llm_epoch{args.epoch}"
    dcl3 = ck / f"l3_run_lr2e-05_lam3_eps1_finetuned_llm_epoch{args.epoch}"
    advsft = ck / f"l3_run_lr2e-05_lam0_eps0_pertlm_finetuned_llm_epoch{args.epoch}"

    harmful = repo / "official_data/jailbreak_r1/combined_test_dataset_with_jailbreak_r1.csv"
    if not harmful.is_file():
        alt = repo / "official_data/jailbreak_r1/combined_test_with_jailbreak_r1.csv"
        if alt.is_file():
            harmful = alt
    benign = Path(os.environ.get("BENIGN_DATA", repo / "official_data/frr_test.csv"))

    errors: list[str] = []
    print("=== data ===")
    check_csv(harmful, ["Jailbreak-R1 Variant"], errors)
    # FRR CSV may use Original Prompt or other names; just existence + rows
    if benign.is_file():
        import pandas as pd

        n = len(pd.read_csv(benign))
        _ok(f"{benign.name}: {n} rows")
    else:
        _fail(f"missing benign CSV: {benign}", errors)

    print("\n=== models (as held-out array will load them) ===")
    check_full_hf(base_l3, "0 base", errors, args.load_tokenizer)

    if (mixat / "adapter_config.json").is_file():
        check_peft(mixat, base_l3, "1 mixat", errors, args.load_tokenizer)
    else:
        check_full_hf(mixat, "1 mixat (full HF)", errors, args.load_tokenizer)

    if (door / "adapter_config.json").is_file():
        check_peft(door, base_l3, "2 door", errors, args.load_tokenizer)
    else:
        check_full_hf(door, "2 door (full HF)", errors, args.load_tokenizer)

    check_peft(dcl1, base_l3, "3 dcl_lam1_eps-1", errors, args.load_tokenizer)
    check_peft(dcl3, base_l3, "4 dcl_lam3_eps1", errors, args.load_tokenizer)
    check_full_hf(delman, "5 delman", errors, args.load_tokenizer)
    check_peft(advsft, base_l3, "6 advsft (seen pertlm)", errors, args.load_tokenizer)

    if args.load_weights:
        print("\n=== weight load (CPU / slow) ===")
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        def load_base(path: Path, label: str) -> None:
            try:
                AutoModelForCausalLM.from_pretrained(
                    str(path), torch_dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True
                )
                _ok(f"{label}: weights load")
            except Exception as e:
                _fail(f"{label}: weight load failed: {e}", errors)

        def load_peft(adapter: Path, base: Path, label: str) -> None:
            try:
                m = AutoModelForCausalLM.from_pretrained(
                    str(base), torch_dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True
                )
                PeftModel.from_pretrained(m, str(adapter))
                _ok(f"{label}: base+adapter load")
            except Exception as e:
                _fail(f"{label}: base+adapter load failed: {e}", errors)

        load_base(base_l3, "0 base")
        if (mixat / "adapter_config.json").is_file():
            load_peft(mixat, base_l3, "1 mixat")
        else:
            load_base(mixat, "1 mixat")
        if (door / "adapter_config.json").is_file():
            load_peft(door, base_l3, "2 door")
        else:
            load_base(door, "2 door")
        load_peft(dcl1, base_l3, "3 dcl_lam1_eps-1")
        load_peft(dcl3, base_l3, "4 dcl_lam3_eps1")
        load_base(delman, "5 delman")
        load_peft(advsft, base_l3, "6 advsft")

    print()
    if errors:
        print(f"{len(errors)} failure(s). Fix before sbatch.")
        return 1
    print("All checks passed. Safe to submit:")
    print("  EPOCH=2 sbatch helper_scripts/perturbation/jailbreak_r1_heldout_eval_l3.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
