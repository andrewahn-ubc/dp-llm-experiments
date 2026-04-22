"""
eval_consistency.py — Evaluate output consistency of trained FEVER models
under surface perturbations.

PURPOSE
-------
Experiment 1 trains BERT with a symmetric stability regularizer on
surface-perturbed FEVER evidence. The regularizer's claim is:

    λ > 0 reduces |P(SUPPORTS|clean) - P(SUPPORTS|perturbed)|
    without significantly degrading val_acc.

FeverSymmetric (Schuster et al., EMNLP 2019) tests claim-level negation
shortcuts — an orthogonal robustness property that surface perturbations
cannot address. This script instead measures the regularizer's actual
target: output consistency under evidence and claim surface perturbations.

METRICS
-------
  gap_<perturbation_type>  mean |P(SUPPORTS|clean) - P(SUPPORTS|pert)|
                           lower = more consistent = better
  gap_indist               mean gap over in-distribution families (regularizer saw them)
  gap_ood                  mean gap over OOD (held-out) families only
  gap_all                  mean across all perturbation families
  ood_family               which family was held out (empty string if none / baseline run)
  val_acc                  standard FEVER dev accuracy (sanity check)
  sym_acc                  FeverSymmetric accuracy (secondary, ~50% expected
                           for Experiment 1 — reported for completeness)

USAGE
-----
  # Evaluate all runs from a specific sweep
  python fact_check/eval_consistency.py --filter 20260420_161433

  # Evaluate all LOO runs
  python fact_check/eval_consistency.py --filter loo_ --batch-size 64 --device cuda

  # Evaluate specific runs
  python fact_check/eval_consistency.py \\
      --run-ids bert_base_uncased_lam0_eps0_lr2e-05_rank0_20260420_161433 \\
                bert_base_uncased_lam5_eps0_lr2e-05_rank0_20260420_161433

  # Evaluate on GPU with larger batch
  python fact_check/eval_consistency.py \\
      --filter 20260420_161433 --batch-size 128 --device cuda
"""

import argparse
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from fact_check.config_loader import load_config
from fact_check.train_fever import LABEL2ID, NUM_LABELS, ID2LABEL, FeverDataset, supports_prob
from fact_check.perturb_fever import PERTURBATION_FNS, CLAIM_PERTURBATIONS

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def find_model_path(run_dir: Path) -> Path | None:
    """Return the path to load the model from: final dir or latest epoch checkpoint."""
    if (run_dir / "config.json").exists():
        return run_dir
    ckpt_base = run_dir / "checkpoints"
    if ckpt_base.exists():
        epoch_dirs = sorted(ckpt_base.glob("epoch_*"),
                            key=lambda p: int(p.name.split("_")[1]))
        if epoch_dirs:
            return epoch_dirs[-1]
    return None


def read_log_meta(run_dir: Path) -> dict:
    log_path = run_dir / "training_log.json"
    if not log_path.exists():
        return {}
    with open(log_path) as f:
        entries = json.load(f)
    return entries[-1] if entries else {}


def load_model(run_dir: Path, device: torch.device):
    """Load tokenizer + model from a run directory. Returns (tokenizer, model)."""
    meta = read_log_meta(run_dir)
    lora_rank  = meta.get("lora_rank", 0)
    model_name = meta.get("model_name", "")

    if lora_rank > 0 and model_name == "bert_base_uncased":
        raise ValueError("BERT+LoRA is unsupported — skipping")

    model_path = find_model_path(run_dir)
    if model_path is None:
        raise FileNotFoundError(f"No model found in {run_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=False,
    )

    if lora_rank > 0:
        model = PeftModel.from_pretrained(base, model_path, is_trainable=False)
    else:
        model = base

    model.eval()
    model.to(device)
    return tokenizer, model


# ---------------------------------------------------------------------------
# Accuracy evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_accuracy(model, csv_path: str, tokenizer, max_seq_len: int,
                  batch_size: int, device: torch.device) -> float:
    ds = FeverDataset(csv_path, tokenizer, max_length=max_seq_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=1)
    correct = total = 0
    for batch in loader:
        ids  = batch["input_ids_clean"].to(device)
        mask = batch["attention_mask_clean"].to(device)
        lbls = batch["label"].to(device)
        logits = model(input_ids=ids, attention_mask=mask).logits
        correct += (logits.argmax(dim=-1) == lbls).sum().item()
        total   += lbls.size(0)
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Consistency evaluation
# ---------------------------------------------------------------------------

def encode_batch(tokenizer, claims: list[str], evidences: list[str],
                 max_length: int) -> dict:
    return tokenizer(claims, evidences, max_length=max_length, truncation=True,
                     padding="max_length", return_tensors="pt")


@torch.no_grad()
def eval_consistency(
    model,
    tokenizer,
    val_df: pd.DataFrame,
    perturbation_types: list[str],
    max_seq_len: int,
    batch_size: int,
    device: torch.device,
    rng: random.Random,
) -> dict[str, dict]:
    """
    For each perturbation type, compute mean |P(SUPPORTS|clean) - P(SUPPORTS|pert)|
    over all val examples where the perturbation had a non-trivial effect.

    Returns dict: perturbation_type -> {"mean_gap": float, "std": float, "n": int}
    """
    claims    = val_df["claim"].astype(str).tolist()
    evidences = val_df["evidence"].astype(str).tolist()
    n = len(claims)

    # Get clean P(SUPPORTS) in batches — one pass for all perturbation types
    clean_probs = []
    for start in range(0, n, batch_size):
        batch_claims = claims[start:start + batch_size]
        batch_evids  = evidences[start:start + batch_size]
        enc = encode_batch(tokenizer, batch_claims, batch_evids, max_seq_len)
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(input_ids=enc["input_ids"],
                       attention_mask=enc["attention_mask"]).logits
        clean_probs.append(supports_prob(logits).cpu())
    clean_probs = torch.cat(clean_probs)  # (N,)

    results = {}
    for ptype in perturbation_types:
        fn = PERTURBATION_FNS[ptype]

        pert_claims, pert_evids, valid_indices = [], [], []
        for i, (claim, evid) in enumerate(zip(claims, evidences)):
            if ptype in CLAIM_PERTURBATIONS:
                pert_c = fn(claim, rng)
                pert_e = evid
            elif ptype == "evidence_trunc":
                pert_c = claim
                pert_e = fn(evid)
            else:
                pert_c = claim
                pert_e = fn(evid, rng)

            if pert_c == claim and pert_e == evid:
                continue  # no-op

            pert_claims.append(pert_c)
            pert_evids.append(pert_e)
            valid_indices.append(i)

        if not valid_indices:
            results[ptype] = {"mean_gap": 0.0, "std": 0.0, "n": 0}
            continue

        pert_probs = []
        for start in range(0, len(valid_indices), batch_size):
            batch_claims = pert_claims[start:start + batch_size]
            batch_evids  = pert_evids[start:start + batch_size]
            enc = encode_batch(tokenizer, batch_claims, batch_evids, max_seq_len)
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(input_ids=enc["input_ids"],
                           attention_mask=enc["attention_mask"]).logits
            pert_probs.append(supports_prob(logits).cpu())
        pert_probs = torch.cat(pert_probs)

        clean_subset = clean_probs[valid_indices]
        gaps = torch.abs(clean_subset - pert_probs)

        results[ptype] = {
            "mean_gap": gaps.mean().item(),
            "std":      gaps.std().item(),
            "n":        len(valid_indices),
        }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = load_config()

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--models-dir",    default=str(cfg.paths.output_root))
    parser.add_argument("--val-csv",       default=str(Path(cfg.paths.data_dir) / "fever_val.csv"))
    parser.add_argument("--sym-val-csv",   default=str(Path(cfg.paths.data_dir) / "fever_symmetric.csv"))
    parser.add_argument("--filter",        default=None,
                        help="Only evaluate runs whose directory name contains this string")
    parser.add_argument("--run-ids",       nargs="+", default=None,
                        help="Explicit run_ids to evaluate (alternative to --filter)")
    parser.add_argument("--batch-size",    type=int, default=32)
    parser.add_argument("--max-seq-len",   type=int, default=128)
    parser.add_argument("--output",        default=None,
                        help="Path for consistency_eval.csv (default: models-dir/consistency_eval.csv)")
    parser.add_argument("--device",        default=None, choices=["cuda", "cpu"])
    parser.add_argument("--perturbations", nargs="+", default=list(PERTURBATION_FNS.keys()))
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    models_dir = Path(args.models_dir)
    output_csv = Path(args.output) if args.output else models_dir / "consistency_eval.csv"

    # Discover run directories
    if args.run_ids:
        run_dirs = [models_dir / rid for rid in args.run_ids]
    else:
        run_dirs = [p for p in models_dir.iterdir()
                    if p.is_dir() and (args.filter is None or args.filter in p.name)
                    and (p / "training_log.json").exists()]

    run_dirs = sorted(run_dirs)
    if not run_dirs:
        print(f"No matching run directories found under {models_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(run_dirs)} run(s) to evaluate.", file=sys.stderr)

    # Load val data once
    val_df = pd.read_csv(args.val_csv)
    val_df = val_df[val_df["label"].isin(LABEL2ID)].reset_index(drop=True)

    # Load existing results if CSV exists
    if output_csv.exists():
        existing = pd.read_csv(output_csv)
    else:
        existing = pd.DataFrame()

    new_rows = []

    for i, run_dir in enumerate(run_dirs, 1):
        run_id = run_dir.name
        print(f"\n[run {i}/{len(run_dirs)}] {run_id}", file=sys.stderr)

        last = read_log_meta(run_dir)
        if not last:
            print(f"  Skipping — no training_log.json or empty", file=sys.stderr)
            continue

        # Read held-out perturbations from log (empty for baseline/non-LOO runs)
        held_out = set(last.get("held_out_perturbations", []))
        ood_family = ", ".join(sorted(held_out)) if held_out else "none"
        if held_out:
            print(f"  Held-out (OOD) family: {ood_family}", file=sys.stderr)

        # Load model
        t0 = time.time()
        print(f"  Loading model...", end=" ", file=sys.stderr, flush=True)
        try:
            tokenizer, model = load_model(run_dir, device)
        except Exception as e:
            print(f"FAILED: {e}", file=sys.stderr)
            continue
        print(f"done ({time.time() - t0:.1f}s)", file=sys.stderr)

        # val_acc
        print(f"  val_acc eval...", end=" ", file=sys.stderr, flush=True)
        val_acc = eval_accuracy(model, args.val_csv, tokenizer,
                                args.max_seq_len, args.batch_size, device)
        print(f"done ({val_acc:.4f})", file=sys.stderr)

        # sym_acc
        print(f"  sym_acc eval...", end=" ", file=sys.stderr, flush=True)
        sym_acc = eval_accuracy(model, args.sym_val_csv, tokenizer,
                                args.max_seq_len, args.batch_size, device)
        print(f"done ({sym_acc:.4f})", file=sys.stderr)

        # Consistency gaps
        print(f"  Perturbation gaps:", file=sys.stderr)
        rng = random.Random(args.seed)
        consistency = eval_consistency(
            model, tokenizer, val_df,
            args.perturbations, args.max_seq_len, args.batch_size, device, rng,
        )

        indist_gaps, ood_gaps = [], []
        for ptype, res in consistency.items():
            tag = "[OOD]  " if ptype in held_out else "[TRAIN]"
            print(f"    {ptype:<25} {tag} : n={res['n']:<6} mean_gap={res['mean_gap']:.4f}  std={res['std']:.4f}",
                  file=sys.stderr)
            if ptype in held_out:
                ood_gaps.append(res["mean_gap"])
            else:
                indist_gaps.append(res["mean_gap"])

        gap_indist = sum(indist_gaps) / len(indist_gaps) if indist_gaps else float("nan")
        gap_ood    = sum(ood_gaps)    / len(ood_gaps)    if ood_gaps    else float("nan")
        all_gap_values = indist_gaps + ood_gaps
        gap_all = sum(all_gap_values) / len(all_gap_values) if all_gap_values else 0.0
        gap_std = statistics.stdev(all_gap_values) if len(all_gap_values) > 1 else 0.0
        print(f"    gap_indist={gap_indist:.4f}  gap_ood={'nan' if gap_ood != gap_ood else f'{gap_ood:.4f}'}  gap_all={gap_all:.4f}",
              file=sys.stderr)

        row = {
            "run_id":           run_id,
            "lambda":           last.get("lambda", 0),
            "epsilon":          last.get("epsilon", 0),
            "lr":               last.get("lr", 0),
            "lora_rank":        last.get("lora_rank", 0),
            "ood_family":       ood_family,
            "val_acc":          val_acc,
            "sym_acc":          sym_acc,
            "gap_word_shuffle": consistency.get("evidence_word_shuffle", {}).get("mean_gap", None),
            "gap_sent_shuffle": consistency.get("evidence_sent_shuffle", {}).get("mean_gap", None),
            "gap_trunc":        consistency.get("evidence_trunc",        {}).get("mean_gap", None),
            "gap_synonym":      consistency.get("claim_synonym",         {}).get("mean_gap", None),
            "gap_indist":       gap_indist,
            "gap_ood":          gap_ood,
            "gap_all":          gap_all,
            "gap_std":          gap_std,
            "model_path":       str(find_model_path(run_dir)),
            "eval_timestamp":   datetime.now().strftime("%Y%m%d_%H%M%S"),
        }
        new_rows.append(row)

    if not new_rows:
        print("No runs evaluated.", file=sys.stderr)
        return

    new_df = pd.DataFrame(new_rows)

    # Merge with existing: new rows overwrite existing by run_id
    if not existing.empty:
        combined = pd.concat([existing[~existing["run_id"].isin(new_df["run_id"])], new_df],
                             ignore_index=True)
    else:
        combined = new_df

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_csv, index=False)
    print(f"\nResults written to {output_csv}", file=sys.stderr)

    # ── Per-family grouped table ─────────────────────────────────────────────
    loo_df = new_df[new_df["ood_family"] != "none"].copy()
    baseline_df = new_df[new_df["ood_family"] == "none"].copy()

    if not baseline_df.empty:
        print("\n── Baseline runs (no held-out family) ──────────────────────────────")
        _print_run_table(baseline_df.sort_values("gap_all"))

    if not loo_df.empty:
        for fam in sorted(loo_df["ood_family"].unique()):
            subset = loo_df[loo_df["ood_family"] == fam].sort_values("lambda")
            print(f"\nOOD family: {fam}")
            _print_run_table(subset, indent="  ")

        # Generalization summary: mean gap_ood by lambda, averaged across OOD families
        loo_df_valid = loo_df[loo_df["gap_ood"].notna()]
        if not loo_df_valid.empty:
            print("\nGeneralization summary (mean gap_ood by λ, averaged across all OOD families):")
            for lam_val in sorted(loo_df_valid["lambda"].unique()):
                sub = loo_df_valid[loo_df_valid["lambda"] == lam_val]["gap_ood"]
                mean_v = sub.mean()
                std_v  = sub.std() if len(sub) > 1 else 0.0
                print(f"  λ={lam_val:<5.2f}  gap_ood={mean_v:.4f} ± {std_v:.4f}  (n={len(sub)} runs)")

    print(f"\n{len(new_rows)} run(s) evaluated.")


def _print_run_table(df: pd.DataFrame, indent: str = "") -> None:
    header = (
        f"{indent}{'run_id':<65} {'λ':>5} {'ε':>5} {'lr':>7} {'rank':>4} "
        f"{'val_acc':>8} {'sym_acc':>8} "
        f"{'gap_wrd':>9} {'gap_snt':>9} {'gap_trn':>9} {'gap_syn':>9} "
        f"{'gap_in':>8} {'gap_ood':>8} {'gap_all':>8}"
    )
    print(header)
    print(indent + "-" * (len(header) - len(indent)))

    def _fmt(v) -> str:
        if v is None or (isinstance(v, float) and v != v):
            return f"{'nan':>8}"
        return f"{v:>8.4f}"

    for _, r in df.iterrows():
        print(
            f"{indent}{r['run_id']:<65} "
            f"{r['lambda']:>5.2f} {r['epsilon']:>5.2f} {r['lr']:>7.0e} {int(r['lora_rank']):>4} "
            f"{r['val_acc']:>8.4f} {r['sym_acc']:>8.4f} "
            f"{_fmt(r.get('gap_word_shuffle'))} {_fmt(r.get('gap_sent_shuffle'))} "
            f"{_fmt(r.get('gap_trunc'))} {_fmt(r.get('gap_synonym'))} "
            f"{_fmt(r.get('gap_indist'))} {_fmt(r.get('gap_ood'))} {r['gap_all']:>8.4f}"
        )


if __name__ == "__main__":
    main()
