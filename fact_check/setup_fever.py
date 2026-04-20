"""
setup_fever.py — One-time setup for the fact-checking experiment.

Reads paths and model config from fact_check/config.yaml (with env var expansion).
Run on a login node (internet access required) after sourcing env.sh:

    source env.sh fact-check
    python fact_check/setup_fever.py

This script will:
  1. Download FEVER train/val datasets
  2. Download FeverSymmetric test set
  3. Download VitaminC train/val/test datasets
  4. Download bert-base-uncased model weights
  5. Optionally download DeBERTa-v3-large-nli and Llama-3.1-8B-Instruct

Options:
    --config          Path to config.yaml (default: fact_check/config.yaml)
    --skip-model      Skip BERT model download (alias: --skip-bert)
    --skip-bert       Skip BERT model download
    --skip-vitaminc   Skip VitaminC dataset download
    --skip-deberta    Skip DeBERTa model download
    --skip-llama      Skip Llama model download
    --skip-data       Skip FEVER dataset downloads
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from fact_check.config_loader import load_config


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
    import json, urllib.request
    import pandas as pd

    out = data_dir / "fever_symmetric.csv"
    if out.exists():
        print(f"[FeverSymmetric] already exists at {out}, skipping.")
        return

    # FeverSymmetric v0.2 (dev + test combined, ~1420 pairs).
    # No HuggingFace mirror exists — download directly from GitHub.
    base = "https://raw.githubusercontent.com/TalSchuster/FeverSymmetric/master/symmetric_v0.2"
    urls = [
        f"{base}/fever_symmetric_dev.jsonl",
        f"{base}/fever_symmetric_test.jsonl",
    ]

    print("Downloading FeverSymmetric v0.2 (dev + test) from GitHub ...")
    rows = []
    for url in urls:
        with urllib.request.urlopen(url) as resp:
            for line in resp:
                ex = json.loads(line)
                label = ex.get("gold_label", "")
                if label not in ("SUPPORTS", "REFUTES"):
                    continue
                rows.append({
                    "claim":    ex["claim"],
                    "evidence": " ".join(ex.get("evidence", [])),
                    "label":    label,
                })
    df = pd.DataFrame(rows)

    df.to_csv(out, index=False)
    print(f"  FeverSymmetric: {len(df):,} rows → {out}")
    print(f"  Label dist: {df['label'].value_counts().to_dict()}")


def download_vitaminc(data_dir: Path) -> None:
    """Download VitaminC (Schuster et al., NAACL 2021) from HuggingFace.

    Saves train/val/test splits with columns: claim, evidence, label, case_id.
    NOT ENOUGH INFO rows are dropped to keep binary classification consistent
    with FEVER. The case_id column is required for contrastive pairing in
    VitaminCDataset.
    """
    from datasets import load_dataset
    import pandas as pd

    train_out = data_dir / "vitaminc_train.csv"
    val_out   = data_dir / "vitaminc_val.csv"
    test_out  = data_dir / "vitaminc_test.csv"

    if train_out.exists() and val_out.exists() and test_out.exists():
        print(f"[VitaminC] already exists at {data_dir}, skipping.")
        return

    print("Downloading tals/vitaminc ...")
    ds = load_dataset("tals/vitaminc")

    def to_df(split):
        df = split.to_pandas()[["claim", "evidence", "label", "case_id"]]
        return df[df["label"].isin(("SUPPORTS", "REFUTES"))].reset_index(drop=True)

    train_df = to_df(ds["train"])
    val_df   = to_df(ds["validation"])
    test_df  = to_df(ds["test"])

    train_df.to_csv(train_out, index=False)
    val_df.to_csv(val_out,   index=False)
    test_df.to_csv(test_out,  index=False)

    print(f"  Train: {len(train_df):,} rows → {train_out}")
    print(f"  Val:   {len(val_df):,} rows → {val_out}")
    print(f"  Test:  {len(test_df):,} rows → {test_out}")
    print(f"  Label dist (train): {train_df['label'].value_counts().to_dict()}")
    grp_sizes = train_df.groupby("case_id").size()
    print(f"  case_id group sizes (train): mean={grp_sizes.mean():.1f}  "
          f"min={grp_sizes.min()}  max={grp_sizes.max()}")


# ---------------------------------------------------------------------------
# Model downloads
# ---------------------------------------------------------------------------

def download_model(model_path: Path, hf_repo: str) -> None:
    if model_path.exists():
        print(f"[model] already exists at {model_path}, skipping.")
        return

    print(f"Downloading {hf_repo} → {model_path} ...")
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=hf_repo, local_dir=str(model_path))
    print(f"  Done.")


def download_llama(model_path: Path, hf_repo: str) -> None:
    """Download Llama-3.1-8B-Instruct weights.

    Requires a HuggingFace account with Llama access approved at:
    https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
    Set HF_TOKEN env var or run 'huggingface-cli login' before calling.
    """
    if not os.environ.get("HF_TOKEN"):
        print("WARNING: HF_TOKEN is not set. Llama download may fail if you are not")
        print("         already logged in. Run: huggingface-cli login")
        print()
    download_model(model_path, hf_repo)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config",          default=None, help="Path to config.yaml")
    parser.add_argument("--skip-model",      action="store_true", help="Skip BERT download (backwards compat)")
    parser.add_argument("--skip-bert",       action="store_true", help="Skip BERT download")
    parser.add_argument("--skip-vitaminc",   action="store_true", help="Skip VitaminC dataset download")
    parser.add_argument("--skip-deberta",    action="store_true", help="Skip DeBERTa download")
    parser.add_argument("--skip-llama",      action="store_true", help="Skip Llama download")
    parser.add_argument("--skip-data",       action="store_true", help="Skip FEVER dataset downloads")
    args = parser.parse_args()

    cfg = load_config(Path(args.config) if args.config else None)

    data_dir    = Path(cfg.paths.data_dir)
    model_root  = Path(cfg.paths.model_root)

    bert_path    = model_root / cfg.models.bert.name
    deberta_path = model_root / cfg.models.deberta.name
    llama_path   = model_root / cfg.models.llama.name

    data_dir.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)

    print(f"Config:    {Path(__file__).parent / 'config.yaml'}")
    print(f"Data dir:  {data_dir}")
    print(f"Models:    {model_root}")
    print()

    if not args.skip_data:
        download_fever(data_dir)
        print()
        download_fever_symmetric(data_dir)
        print()

    if not args.skip_vitaminc:
        download_vitaminc(data_dir)
        print()

    skip_bert = args.skip_bert or args.skip_model
    if not skip_bert:
        download_model(bert_path, cfg.models.bert.hf_repo)
        print()

    if not args.skip_deberta:
        download_model(deberta_path, cfg.models.deberta.hf_repo)
        print()

    if not args.skip_llama:
        download_llama(llama_path, cfg.models.llama.hf_repo)
        print()

    print("=" * 60)
    print("Setup complete.")
    print(f"  FEVER:      {data_dir}/{{fever_train,fever_val,fever_symmetric}}.csv")
    print(f"  VitaminC:   {data_dir}/{{vitaminc_train,vitaminc_val,vitaminc_test}}.csv")
    print(f"  BERT:       {bert_path}")
    print(f"  DeBERTa:    {deberta_path}")
    print(f"  Llama:      {llama_path}  (~16GB)")
    print()
    print("CrossAug baselines (BERT-base, for reference):")
    print("  No augmentation: 86.15% FEVER dev / 58.77% FeverSymmetric")
    print("  CrossAug:        85.34% FEVER dev / 68.90% FeverSymmetric")
    print()
    print("VitaminC baselines (ALBERT-base, for reference):")
    print("  No contrastive training: ~72% VitaminC test")
    print("  VitaminC-trained:        ~85% VitaminC test")
    print()
    print("Next steps:")
    print("  1. python fact_check/perturb_fever.py   (generates perturbed FEVER training data)")
    print("  2. python fact_check/sweep_fever.py --dry-run")
    print("  3. python fact_check/sweep_fever.py")
    print("  4. python fact_check/sweep_fever.py --model deberta --dataset vitaminc --dry-run")
    print("  5. python fact_check/sweep_fever.py --model llama --dataset vitaminc --lora-ranks 16 --mem 48G")


if __name__ == "__main__":
    main()
