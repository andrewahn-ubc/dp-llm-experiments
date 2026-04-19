"""
setup_fever.py — One-time setup for the fact-checking experiment.

Reads paths and model config from fact_check/config.yaml (with env var expansion).
Run on a login node (internet access required) after sourcing env.sh:

    source env.sh fact-check
    python fact_check/setup_fever.py

This script will:
  1. Download FEVER train/val datasets
  2. Download FeverSymmetric test set
  3. Download bert-base-uncased model weights

Options:
    --config        Path to config.yaml (default: fact_check/config.yaml)
    --skip-model    Skip model download
    --skip-data     Skip dataset downloads
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from fact_check.config_loader import load_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pip_install(*packages: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])


def ensure_deps() -> None:
    try:
        import datasets   # noqa: F401
        import pandas     # noqa: F401
        import pyarrow    # noqa: F401
        from huggingface_hub import snapshot_download  # noqa: F401
    except ImportError:
        print("Installing required packages into current Python...")
        pip_install("huggingface_hub[cli]", "datasets", "pandas", "pyarrow")


# ---------------------------------------------------------------------------
# Dataset downloads
# ---------------------------------------------------------------------------

def download_fever(data_dir: Path) -> None:
    from datasets import load_dataset
    import pandas as pd

    train_out = data_dir / "fever_train.csv"
    val_out   = data_dir / "fever_val.csv"

    if train_out.exists() and val_out.exists():
        print(f"[FEVER] already exists at {data_dir}, skipping.")
        return

    print("Downloading copenlu/fever_gold_evidence ...")
    ds = load_dataset("copenlu/fever_gold_evidence")

    def to_df(split):
        rows = []
        for ex in split:
            label    = ex["label"]
            claim    = ex["claim"]
            evidence = " ".join(e[2] for e in ex["evidence"])
            rows.append({"claim": claim, "evidence": evidence, "label": label})
        df = pd.DataFrame(rows)
        return df[df["label"] != "NOT ENOUGH INFO"].reset_index(drop=True)

    train_df = to_df(ds["train"])
    val_df   = to_df(ds["validation"])

    train_df.to_csv(train_out, index=False)
    val_df.to_csv(val_out,   index=False)

    print(f"  Train: {len(train_df):,} rows → {train_out}")
    print(f"  Val:   {len(val_df):,} rows → {val_out}")
    print(f"  Label dist (train): {train_df['label'].value_counts().to_dict()}")


def download_fever_symmetric(data_dir: Path) -> None:
    import pandas as pd

    out = data_dir / "fever_symmetric.csv"
    if out.exists():
        print(f"[FeverSymmetric] already exists at {out}, skipping.")
        return

    print("Downloading TalSchuster/fever_symmetric_v2 ...")
    try:
        from datasets import load_dataset
        ds = load_dataset("TalSchuster/fever_symmetric_v2", split="test")
        rows = []
        for ex in ds:
            label = ex.get("gold_label", "")
            if label not in ("SUPPORTS", "REFUTES"):
                continue
            rows.append({
                "claim":    ex["claim"],
                "evidence": " ".join(ex.get("evidence", [])),
                "label":    label,
            })
        df = pd.DataFrame(rows)
    except Exception as e:
        print(f"  HuggingFace load failed ({e}), trying GitHub raw fallback...")
        import json, urllib.request
        url = (
            "https://raw.githubusercontent.com/TalSchuster/"
            "FeverSymmetric/master/symmetric_v2/fever_symmetric_generated.jsonl"
        )
        rows = []
        with urllib.request.urlopen(url) as resp:
            for line in resp:
                ex = json.loads(line)
                label = ex.get("gold_label", "")
                if label not in ("SUPPORTS", "REFUTES"):
                    continue
                rows.append({
                    "claim":    ex["claim"],
                    "evidence": " ".join(ex.get("evidence_sentence", [])),
                    "label":    label,
                })
        df = pd.DataFrame(rows)

    df.to_csv(out, index=False)
    print(f"  FeverSymmetric: {len(df):,} rows → {out}")
    print(f"  Label dist: {df['label'].value_counts().to_dict()}")


# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------

def download_model(model_path: Path, hf_repo: str) -> None:
    if model_path.exists():
        print(f"[model] already exists at {model_path}, skipping.")
        return

    print(f"Downloading {hf_repo} → {model_path} ...")
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=hf_repo, local_dir=str(model_path))
    print(f"  Done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config",      default=None, help="Path to config.yaml")
    parser.add_argument("--skip-model",  action="store_true")
    parser.add_argument("--skip-data",   action="store_true")
    args = parser.parse_args()

    cfg = load_config(Path(args.config) if args.config else None)

    data_dir   = Path(cfg.paths.data_dir)
    model_path = Path(cfg.paths.model_root) / cfg.model.name
    hf_repo    = cfg.model.hf_repo

    data_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Config:    {Path(__file__).parent / 'config.yaml'}")
    print(f"Data dir:  {data_dir}")
    print(f"Model:     {model_path}  (from {hf_repo})")
    print()

    ensure_deps()

    if not args.skip_data:
        download_fever(data_dir)
        print()
        download_fever_symmetric(data_dir)
        print()

    if not args.skip_model:
        download_model(model_path, hf_repo)
        print()

    print("=" * 55)
    print("Setup complete.")
    print(f"  Data:  {data_dir}/{{fever_train,fever_val,fever_symmetric}}.csv")
    print(f"  Model: {model_path}")
    print()
    print("CrossAug baselines (BERT-base, for reference):")
    print("  No augmentation: 86.15% FEVER dev / 58.77% FeverSymmetric")
    print("  CrossAug:        85.34% FEVER dev / 68.90% FeverSymmetric")
    print()
    print("Next steps:")
    print("  1. python fact_check/perturb_fever.py  (generates perturbed training data)")
    print("  2. python fact_check/sweep_fever.py --dry-run")
    print("  3. python fact_check/sweep_fever.py")


if __name__ == "__main__":
    main()
