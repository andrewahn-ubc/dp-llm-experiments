"""
train_fever.py — Fine-tune a fact-checking model with the symmetric stability loss.

=======================================================================
WHAT THIS EXPERIMENT IS SHOWING
=======================================================================
Standard NLI/fact-checking models trained on FEVER learn annotation
shortcuts: they fire on surface features like negation words ("not",
"never") rather than truly reasoning over evidence.  This means they're
brittle, swap the evidence order, drop a sentence, or paraphrase a
claim and the predicted label can flip, even though the underlying
fact hasn't changed.

We apply the symmetric regularizer from the paper:

    L_stab = | C(claim, evidence_clean) - C(claim, evidence_perturbed) |

where C(·) is the model's own confidence in "SUPPORTS" (a scalar in [0,1]).
Because both inputs represent the same underlying fact, the model should
give them similar confidence scores.  The regularizer penalizes the gap.

Since the model's output is a single token (SUPPORTS / REFUTES), 
we don't need a separate safety classifier we can read the probability 
directly from the model's own output head. This avoids the embedding-compatibility 
constraint that required Llama Guard to share the same embedding 
space as the LLM being trained.

Full loss:
    L = L_CE(clean) + λ * max(0, |P_SUPPORTS(clean) - P_SUPPORTS(perturbed)| - ε)

=======================================================================
HYPERPARAMETERS (all injectable via env vars — see train_fever.sh)
=======================================================================
  LAMBDA            — weight of the stability term (default 1.0)
  EPSILON           — allowed gap before penalty kicks in (default 0.0)
  LR                — learning rate (default 2e-5)
  BATCH_SIZE        — examples per gradient step (default 16)
  EPOCHS            — number of training epochs (default 3)
  LORA_RANK         — LoRA rank; 0 means full fine-tuning (default 0)
  CKPT_EVERY_STEPS  — save a mid-epoch checkpoint every N steps (default 0 = epoch-only)

=======================================================================
CHECKPOINTING & RESUMPTION
=======================================================================
The script saves a checkpoint after every epoch and optionally mid-epoch
(every CKPT_EVERY_STEPS steps).  Each checkpoint directory contains:
  - model weights (pytorch_model.bin / adapter_model.bin)
  - tokenizer files
  - optimizer.pt
  - checkpoint_meta.json  ← epoch, global_step, log_rows so far
  (scheduler is NOT saved — it is reconstructed from global_step on resume)

To resume after an interruption:
    python train_fever.py ... --resume-from-checkpoint /path/to/checkpoint/dir

The script will pick up from the saved epoch/step and append to the
existing training_log.json in the final output directory.

=======================================================================
"""

import os
import argparse
import json
import random
import time
from pathlib import Path

import wandb
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

from fact_check.logging_utils import get_logger

# ---------------------------------------------------------------------------
# Label map
# ---------------------------------------------------------------------------
LABEL2ID = {"SUPPORTS": 0, "REFUTES": 1}
ID2LABEL  = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = 2   # binary; NOT ENOUGH INFO excluded at setup stage

# MoritzLaurer DeBERTa-v3-large-mnli-fever-anli-ling-wanli has 3 labels:
#   0=entailment  1=neutral  2=contradiction
# We preserve the pretrained head and remap our binary labels onto these indices.
DEBERTA_NLI_LABEL_MAP = {"SUPPORTS": 0, "REFUTES": 2}  # entailment / contradiction

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
def encode(tokenizer, claim: str, evidence: str, max_length: int):
    return tokenizer(claim, evidence, max_length=max_length, truncation=True,
                     padding="max_length", return_tensors="pt")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class FeverDataset(Dataset):
    """
    Loads a CSV with at minimum: claim, evidence, label.
    If original_claim / original_evidence columns are present (the perturbed
    training CSV), pairs each clean row with one of its perturbed variants
    on the fly.
    """

    def __init__(self, csv_path: str, tokenizer, max_length: int = 128,
                 label2id: dict | None = None,
                 held_out_perturbations: set[str] = frozenset()):
        self.label2id = label2id or LABEL2ID
        df = pd.read_csv(csv_path)
        df = df[df["label"].isin(self.label2id)].reset_index(drop=True)
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.held_out_perturbations = held_out_perturbations

        if "original_claim" in df.columns:
            self.perturbed = True
            self.rows = df
            self.orig_groups: dict[tuple, list[int]] = {}
            for idx, row in df.iterrows():
                key = (str(row["original_claim"]), str(row["original_evidence"]))
                self.orig_groups.setdefault(key, []).append(idx)
            # originals: rows where claim == original_claim
            mask = df["claim"].astype(str) == df["original_claim"].astype(str)
            self.originals = df[mask].reset_index(drop=True)
        else:
            self.perturbed = False
            self.originals = df

    def as_augmented(self) -> "FeverDataset":
        """Return a copy that trains CE on ALL rows (originals + perturbations),
        used for the augmentation-only baseline (λ=0, all data as CE signal)."""
        clone = object.__new__(FeverDataset)
        clone.tokenizer  = self.tokenizer
        clone.max_length = self.max_length
        clone.label2id   = self.label2id
        clone.held_out_perturbations = self.held_out_perturbations
        clone.perturbed  = False   # disables stability pairing in __getitem__
        clone.originals  = self.rows if self.perturbed else self.originals
        return clone

    def _encode(self, claim: str, evid: str) -> dict:
        return encode(self.tokenizer, claim, evid, self.max_length)

    def __len__(self):
        return len(self.originals)

    def __getitem__(self, idx):
        row   = self.originals.iloc[idx]
        claim = str(row["claim"])
        evid  = str(row["evidence"])
        label = self.label2id[row["label"]]

        enc_clean = self._encode(claim, evid)
        item = {
            "input_ids_clean":      enc_clean["input_ids"].squeeze(0),
            "attention_mask_clean": enc_clean["attention_mask"].squeeze(0),
            "label":                torch.tensor(label, dtype=torch.long),
        }

        if self.perturbed:
            key       = (claim, evid)   # already str() from above
            pert_idxs = self.orig_groups.get(key, [])
            if pert_idxs:
                eligible = [
                    i for i in pert_idxs
                    if self.rows.loc[i, "perturbation_type"] not in self.held_out_perturbations
                ] if self.held_out_perturbations else pert_idxs
                if eligible:
                    pert_row   = self.rows.loc[random.choice(eligible)]
                    pert_claim = str(pert_row["claim"])
                    pert_evid  = str(pert_row["evidence"])
                else:
                    # all neighbors are held out — use clean as its own pair (stab = 0)
                    enc_pert = self._encode(claim, evid)
                    item["input_ids_pert"]      = enc_pert["input_ids"].squeeze(0)
                    item["attention_mask_pert"] = enc_pert["attention_mask"].squeeze(0)
                    return item
            else:
                pert_claim = claim
                pert_evid  = evid

            enc_pert = self._encode(pert_claim, pert_evid)
            item["input_ids_pert"]      = enc_pert["input_ids"].squeeze(0)
            item["attention_mask_pert"] = enc_pert["attention_mask"].squeeze(0)

        return item


# ---------------------------------------------------------------------------
# VitaminC dataset
# ---------------------------------------------------------------------------
class VitaminCDataset(Dataset):
    """
    Loads VitaminC (Schuster et al., NAACL 2021) contrastive fact-verification data.

    VitaminC contains naturally contrastive claim-evidence pairs from real Wikipedia
    revision history. Each `case_id` groups 2-4 rows sharing the same claim with
    different evidence passages — one typically supports the claim, another refutes
    or is insufficient. These are real neighbor pairs; no synthetic perturbation needed.

    __getitem__ returns an anchor row and a randomly-selected different row from the
    same case_id group as the "perturbed" input. This matches FeverDataset's output
    dict keys exactly so the training loop in main() requires zero modification.

    Important: anchor and neighbor may have *different* labels (SUPPORTS vs REFUTES).
    This is intentional and correct. The symmetric regularizer penalizes the *magnitude*
    of the gap in P(SUPPORTS) between anchor and neighbor. In the FEVER case both inputs
    share the same label (surface perturbation, same truth value). Here the gap being
    large is expected when labels differ — the model should be sensitive to evidence
    changes — but the *rate* of change should still be bounded by ε. This is the
    contrastive version of the regularizer and the point of using VitaminC.

    Reference: Schuster et al. (2021) "VitaminC: Robust Fact Verification with
    Contrastive Evidence" NAACL 2021. https://aclanthology.org/2021.naacl-main.52
    """

    def __init__(self, csv_path: str, tokenizer, max_length: int = 128,
                 label2id: dict | None = None):
        self.label2id = label2id or LABEL2ID
        df = pd.read_csv(csv_path)
        df = df[df["label"].isin(self.label2id)].reset_index(drop=True)
        self.tokenizer  = tokenizer
        self.max_length = max_length

        # Group rows by case_id; each group is a set of contrastive evidence passages.
        self.groups: list[list[int]] = []
        for _, grp in df.groupby("case_id"):
            self.groups.append(grp.index.tolist())
        self.df = df

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        idxs       = self.groups[idx]
        anchor_pos = random.choice(range(len(idxs)))
        anchor     = self.df.iloc[idxs[anchor_pos]]

        claim = str(anchor["claim"])
        evid  = str(anchor["evidence"])
        label = self.label2id[anchor["label"]]

        enc_clean = encode(self.tokenizer, claim, evid, self.max_length)
        item = {
            "input_ids_clean":      enc_clean["input_ids"].squeeze(0),
            "attention_mask_clean": enc_clean["attention_mask"].squeeze(0),
            "label":                torch.tensor(label, dtype=torch.long),
        }

        # Pick a different row from the same case_id group as the contrastive input.
        if len(idxs) > 1:
            other_positions = [i for i in range(len(idxs)) if i != anchor_pos]
            pert_row = self.df.iloc[idxs[random.choice(other_positions)]]
        else:
            pert_row = anchor  # single-row group: stab loss will be zero

        enc_pert = encode(self.tokenizer, str(pert_row["claim"]), str(pert_row["evidence"]), self.max_length)
        item["input_ids_pert"]      = enc_pert["input_ids"].squeeze(0)
        item["attention_mask_pert"] = enc_pert["attention_mask"].squeeze(0)

        return item


# ---------------------------------------------------------------------------
# Stability loss
# ---------------------------------------------------------------------------
def supports_prob(logits: torch.Tensor, supports_idx: int = 0) -> torch.Tensor:
    """P(SUPPORTS) from classification logits. Shape: (B,)"""
    return torch.softmax(logits, dim=-1)[:, supports_idx]


def stability_loss(
    logits_clean: torch.Tensor,
    model,
    input_ids_pert,  attention_mask_pert,
    epsilon: float,
    supports_idx: int = 0,
) -> torch.Tensor:
    """
    Symmetric stability regularizer:
        max(0, |P_SUPPORTS(clean) - P_SUPPORTS(pert)| - epsilon)

    Takes pre-computed clean logits to avoid a redundant forward pass —
    the training loop already ran the model on the clean input for CE loss.
    """
    logits_pert = model(input_ids=input_ids_pert, attention_mask=attention_mask_pert).logits
    gap = torch.abs(supports_prob(logits_clean, supports_idx) - supports_prob(logits_pert, supports_idx))
    return torch.clamp(gap - epsilon, min=0.0).mean()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, dataloader, device, log, valid_label_indices: list[int] | None = None) -> dict:
    """
    valid_label_indices: for 3-class DeBERTa NLI, pass [0, 2] so argmax is taken
    only over entailment and contradiction columns (ignoring neutral). Labels in
    the batch are already remapped (SUPPORTS→0, REFUTES→2), so direct comparison works.
    """
    model.eval()
    try:
        correct = total = 0
        for batch in dataloader:
            input_ids = batch["input_ids_clean"].to(device)
            attn_mask = batch["attention_mask_clean"].to(device)
            labels    = batch["label"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attn_mask).logits
            if valid_label_indices is not None:
                # Mask out neutral (index 1); pick argmax among valid columns only.
                mask = torch.full(logits.shape, float("-inf"), device=device, dtype=logits.dtype)
                mask[:, valid_label_indices] = logits[:, valid_label_indices]
                preds = mask.argmax(dim=-1)
            else:
                preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    finally:
        model.train()
    acc = correct / total if total > 0 else 0.0
    log.debug("evaluate: correct=%d / total=%d  acc=%.4f", correct, total, acc)
    return {"accuracy": acc}


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def save_checkpoint(
    ckpt_dir: Path,
    model,
    tokenizer,
    optimizer,
    epoch: int,
    global_step: int,
    log_rows: list,
    log,
    mid_epoch: bool = False,
):
    """Save a resumable checkpoint to ckpt_dir."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
    meta = {
        "epoch":           epoch,
        "global_step":     global_step,
        "log_rows":        log_rows,
        "wandb_run_id":    wandb.run.id if wandb.run else None,
        "sweep_timestamp": os.environ.get("SWEEP_TIMESTAMP", ""),
        "mid_epoch":       mid_epoch,
    }
    with open(ckpt_dir / "checkpoint_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Checkpoint saved → %s  (epoch=%d  step=%d)", ckpt_dir, epoch, global_step)


def load_checkpoint(ckpt_dir: Path, model, optimizer, log):
    """
    Load optimizer state and metadata from a checkpoint directory.
    Model weights must already be loaded via from_pretrained before calling this.
    The scheduler is NOT saved/loaded — it is reconstructed from global_step
    in main() to avoid total_iters mismatch on resume.
    Returns (start_epoch, global_step, log_rows).
    """
    opt_path  = ckpt_dir / "optimizer.pt"
    meta_path = ckpt_dir / "checkpoint_meta.json"

    if not meta_path.exists():
        raise FileNotFoundError(f"No checkpoint_meta.json found in {ckpt_dir}")

    with open(meta_path) as f:
        meta = json.load(f)

    if opt_path.exists():
        optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))
        log.info("Loaded optimizer state from %s", opt_path)
    else:
        log.warning("optimizer.pt not found in checkpoint — starting optimizer fresh")

    mid_epoch   = meta.get("mid_epoch", False)
    # Mid-epoch checkpoint: resume within the same epoch.
    # End-of-epoch checkpoint: resume from the next epoch.
    start_epoch = meta["epoch"] if mid_epoch else meta["epoch"] + 1
    global_step = meta["global_step"]
    log_rows    = meta.get("log_rows", [])
    log.info(
        "Resuming from checkpoint: epoch=%d  mid_epoch=%s  global_step=%d  "
        "log_rows_so_far=%d",
        meta["epoch"], mid_epoch, global_step, len(log_rows),
    )
    return start_epoch, global_step, log_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args):
    out_dir  = Path(args.output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_base = out_dir / "checkpoints"

    log = get_logger(__name__, output_dir=str(out_dir), run_id=args.run_id)
    log.info("=" * 60)
    log.info("Run ID: %s", args.run_id)
    log.info("Args: %s", vars(args))
    log.info("Output dir: %s", out_dir)

    # ── Resume flag (needed by W&B init below and model loading below) ──────────
    is_resume = args.resume_from_checkpoint is not None
    ckpt_dir  = Path(args.resume_from_checkpoint) if is_resume else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    if device.type == "cuda":
        log.info(
            "GPU: %s  (%.1f GB total)",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )

    # ── Model + tokenizer ────────────────────────────────────────────────────
    if is_resume:
        log.info("Resuming — loading model weights from checkpoint: %s", ckpt_dir)
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    else:
        log.info("Fresh run — loading base model from: %s", args.model_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Llama (and other causal LMs) have no pad token by default.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_path = ckpt_dir if is_resume else args.model_path

    # DeBERTa-v3 pretrained NLI models have a 3-class head (entailment/neutral/contradiction).
    # Overriding num_labels=2 discards the pretrained head and randomly reinitializes it,
    # causing the model to collapse to majority-class prediction. Instead, preserve the
    # pretrained head and remap SUPPORTS→entailment(0), REFUTES→contradiction(2) at loss time.
    # We detect this from the saved config (works for both fresh runs and resumes).
    import json as _json
    _cfg_path = load_path / "config.json" if isinstance(load_path, Path) else Path(load_path) / "config.json"
    _saved_num_labels = len(_json.load(open(_cfg_path)).get("id2label", {})) if _cfg_path.exists() else NUM_LABELS
    is_deberta_nli = (
        "deberta" in args.model_name.lower()
        and _saved_num_labels == 3
    )
    if is_deberta_nli:
        base = AutoModelForSequenceClassification.from_pretrained(load_path)
        log.info("DeBERTa NLI: preserving pretrained 3-class head (entailment/neutral/contradiction)")
        log.info("Label remap: SUPPORTS→0 (entailment), REFUTES→2 (contradiction)")
    else:
        base = AutoModelForSequenceClassification.from_pretrained(
            load_path,
            num_labels=NUM_LABELS,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=not is_resume,
        )
    args._deberta_nli = is_deberta_nli  # propagate to loss/eval helpers

    if args.lora_rank > 0:
        if args.model_name == "bert_base_uncased":
            raise ValueError("LoRA is not supported for BERT (target_modules mismatch). Use --lora-rank 0 with BERT.")
        if is_resume:
            model = PeftModel.from_pretrained(base, ckpt_dir, is_trainable=True)
            log.info("Loaded LoRA adapter from checkpoint")
        else:
            lora_cfg = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=args.lora_rank,
                lora_alpha=16,
                lora_dropout=0.05,
                target_modules=["q_proj", "v_proj"],
            )
            model = get_peft_model(base, lora_cfg)
            trainable, total = model.get_nb_trainable_parameters()
            log.info("LoRA rank=%d  trainable params: %d / %d", args.lora_rank, trainable, total)
    else:
        model = base

    model.to(device)
    log.info("Model loaded and moved to %s", device)

    # ── Label mapping (model-specific) ───────────────────────────────────────
    # For DeBERTa NLI (3-class pretrained head), remap our binary labels onto
    # entailment(0) and contradiction(2). All other models use the standard 2-class map.
    if is_deberta_nli:
        active_label2id      = DEBERTA_NLI_LABEL_MAP          # SUPPORTS→0, REFUTES→2
        supports_idx         = DEBERTA_NLI_LABEL_MAP["SUPPORTS"]   # 0
        valid_label_indices  = [0, 2]                          # ignore neutral(1) at eval
    else:
        active_label2id      = LABEL2ID
        supports_idx         = LABEL2ID["SUPPORTS"]            # 0
        valid_label_indices  = None

    # ── Datasets ─────────────────────────────────────────────────────────────
    log.info("Loading datasets (dataset=%s) ...", args.dataset)
    held_out = set(args.held_out_perturbations) if args.held_out_perturbations else set()
    if held_out:
        log.info("Held-out perturbations (excluded from stab loss): %s", sorted(held_out))
    ds_kwargs = dict(max_length=args.max_seq_len, label2id=active_label2id)

    if args.dataset == "vitaminc":
        if not args.vitaminc_train_csv or not args.vitaminc_val_csv:
            raise ValueError("--vitaminc-train-csv and --vitaminc-val-csv are required when --dataset vitaminc")
        train_dataset = VitaminCDataset(args.vitaminc_train_csv, tokenizer, **ds_kwargs)
        val_dataset   = FeverDataset(args.vitaminc_val_csv, tokenizer, **ds_kwargs)
        log.info("Training on VitaminC; val on VitaminC val; FeverSymmetric as cross-dataset probe")
    elif args.dataset == "fever":
        if not args.train_csv or not args.val_csv:
            raise ValueError("--train-csv and --val-csv are required when --dataset fever")
        train_dataset = FeverDataset(args.train_csv, tokenizer, **ds_kwargs,
                                     held_out_perturbations=held_out)
        val_dataset   = FeverDataset(args.val_csv,   tokenizer, **ds_kwargs)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset!r}. Expected 'fever' or 'vitaminc'.")

    sym_dataset = FeverDataset(args.sym_val_csv, tokenizer, **ds_kwargs)

    if args.augmentation_only:
        if args.dataset != "fever":
            raise ValueError("--augmentation-only is only supported with --dataset fever")
        train_dataset = train_dataset.as_augmented()
        log.info("Mode: augmentation-only (CE on all rows, no stability loss)")

    log.info(
        "Dataset sizes — train: %d  val: %d  fever_sym: %d",
        len(train_dataset), len(val_dataset), len(sym_dataset),
    )
    if hasattr(train_dataset, "perturbed") and train_dataset.perturbed:
        log.debug("Perturbation pairing: train has %d orig_groups", len(train_dataset.orig_groups))
    elif hasattr(train_dataset, "groups"):
        log.debug("VitaminC pairing: train has %d case_id groups", len(train_dataset.groups))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    sym_loader   = DataLoader(sym_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # ── Optimizer + scheduler ────────────────────────────────────────────────
    # eps=1e-6 required for DeBERTa-v3 stability (default 1e-8 causes NaN)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, eps=1e-6)

    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(0.06 * total_steps)
    log.info("Optimizer: AdamW  lr=%.2e  warmup_steps=%d / total_steps=%d", args.lr, warmup_steps, total_steps)

    # ── Resume state ─────────────────────────────────────────────────────────
    if is_resume:
        start_epoch, global_step, log_rows = load_checkpoint(
            ckpt_dir, model, optimizer, log
        )
    else:
        start_epoch, global_step, log_rows = 1, 0, []

    # Scheduler is rebuilt from scratch after loading the resume global_step so
    # that total_iters and the internal last_epoch counter are always consistent.
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-3,
        end_factor=1.0,
        total_iters=warmup_steps,
        last_epoch=global_step - 1,  # -1 because scheduler.step() will increment
    )

    epsilon_val      = args.epsilon
    ckpt_every_steps = args.ckpt_every_steps

    # ── W&B init — done here (just before first log call) to avoid timeout ───
    wandb_run_id = None
    if is_resume:
        meta_path = ckpt_dir / "checkpoint_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                wandb_run_id = json.load(f).get("wandb_run_id")

    wandb.init(
        project="fact-check-fever",
        name=args.run_id,
        id=wandb_run_id,
        dir=os.environ.get("WANDB_DIR", str(out_dir)),
        config={
            "lambda":           args.lambda_val,
            "epsilon":          args.epsilon,
            "lr":               args.lr,
            "batch_size":       args.batch_size,
            "epochs":           args.epochs,
            "lora_rank":        args.lora_rank,
            "model_name":       args.model_name,
            "dataset":          args.dataset,
        },
        resume="allow",
    )

    # ── Training loop ────────────────────────────────────────────────────────
    log.info("Starting training: epochs %d→%d  λ=%.3f  ε=%.3f", start_epoch, args.epochs, args.lambda_val, epsilon_val)

    steps_per_epoch = len(train_loader)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_ce = epoch_stab = epoch_total = 0.0
        n_batches   = 0
        epoch_start = time.time()

        # On a mid-epoch resume, skip batches already processed in this epoch.
        batches_done = global_step % steps_per_epoch if epoch == start_epoch and is_resume else 0
        if batches_done:
            log.info("── Epoch %d / %d  (skipping first %d batches) ──────────────────────────────", epoch, args.epochs, batches_done)
        else:
            log.info("── Epoch %d / %d ──────────────────────────────", epoch, args.epochs)

        for batch_idx, batch in enumerate(train_loader):
            if batch_idx < batches_done:
                continue
            ids_c  = batch["input_ids_clean"].to(device)
            attn_c = batch["attention_mask_clean"].to(device)
            labels = batch["label"].to(device)

            out          = model(input_ids=ids_c, attention_mask=attn_c, labels=labels)
            ce           = out.loss
            logits_clean = out.logits  # reused below — avoids a second forward pass

            if "input_ids_pert" in batch:
                ids_p  = batch["input_ids_pert"].to(device)
                attn_p = batch["attention_mask_pert"].to(device)
                stab   = stability_loss(logits_clean, model, ids_p, attn_p, epsilon_val, supports_idx)
            else:
                stab = torch.tensor(0.0, device=device)

            loss = ce + args.lambda_val * stab

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_ce    += ce.item()
            epoch_stab  += stab.item()
            epoch_total += loss.item()
            n_batches   += 1
            global_step += 1

            log.debug(
                "step=%d  epoch=%d  batch=%d  ce=%.4f  stab=%.4f  loss=%.4f  lr=%.2e",
                global_step, epoch, batch_idx,
                ce.item(), stab.item(), loss.item(),
                scheduler.get_last_lr()[0],
            )

            if global_step % 100 == 0:
                current_lr = scheduler.get_last_lr()[0]
                log.info(
                    "step=%d  ce=%.4f  stab=%.4f  loss=%.4f  lr=%.2e",
                    global_step, ce.item(), stab.item(), loss.item(),
                    current_lr,
                )
                wandb.log({
                    "train/ce":    ce.item(),
                    "train/stab":  stab.item(),
                    "train/loss":  loss.item(),
                    "train/lr":    current_lr,
                    "global_step": global_step,
                })

            # ── Mid-epoch checkpoint ──────────────────────────────────────────
            if ckpt_every_steps > 0 and global_step % ckpt_every_steps == 0:
                mid_ckpt = ckpt_base / f"step_{global_step}"
                save_checkpoint(
                    mid_ckpt, model, tokenizer, optimizer,
                    epoch, global_step, log_rows, log, mid_epoch=True,
                )

        # ── End-of-epoch ─────────────────────────────────────────────────────
        epoch_elapsed = time.time() - epoch_start
        avg_ce   = epoch_ce   / n_batches
        avg_stab = epoch_stab / n_batches
        avg_loss = epoch_total / n_batches

        log.info("Epoch %d complete — %.1fs  avg: loss=%.4f  ce=%.4f  stab=%.4f",
                 epoch, epoch_elapsed, avg_loss, avg_ce, avg_stab)
        log.info("Running evaluation...")

        val_metrics = evaluate(model, val_loader, device, log, valid_label_indices)
        sym_metrics = evaluate(model, sym_loader, device, log, valid_label_indices)

        log.info(
            "Epoch %d results — val_acc=%.4f  sym_acc=%.4f",
            epoch, val_metrics["accuracy"], sym_metrics["accuracy"],
        )
        wandb.log({
            "epoch/loss":    avg_loss,
            "epoch/ce":      avg_ce,
            "epoch/stab":    avg_stab,
            "epoch/val_acc": val_metrics["accuracy"],
            "epoch/sym_acc": sym_metrics["accuracy"],
            "epoch":         epoch,
            "global_step":   global_step,
        })

        epoch_record = {
            "epoch":                   epoch,
            "loss":                    avg_loss,
            "ce":                      avg_ce,
            "stab":                    avg_stab,
            "val_acc":                 val_metrics["accuracy"],
            "sym_acc":                 sym_metrics["accuracy"],
            "lambda":                  args.lambda_val,
            "epsilon":                 args.epsilon,
            "lr":                      args.lr,
            "lora_rank":               args.lora_rank,
            "augmentation_only":       args.augmentation_only,
            "model_name":              args.model_name,
            "dataset":                 args.dataset,
            "run_id":                  args.run_id,
            "global_step":             global_step,
            "held_out_perturbations":  sorted(held_out),
        }
        log_rows.append(epoch_record)

        log_path = out_dir / "training_log.json"
        with open(log_path, "w") as f:
            json.dump(log_rows, f, indent=2)
        log.debug("Training log updated: %s", log_path)

        # ── End-of-epoch checkpoint ───────────────────────────────────────────
        epoch_ckpt = ckpt_base / f"epoch_{epoch}"
        save_checkpoint(
            epoch_ckpt, model, tokenizer, optimizer,
            epoch, global_step, log_rows, log
        )

    # ── Final model save ─────────────────────────────────────────────────────
    log.info("Training complete. Saving final model to %s", out_dir)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    final = log_rows[-1]
    log.info("=" * 60)
    log.info("FINAL RESULTS  run_id=%s", args.run_id)
    log.info("  val_acc  = %.4f", final["val_acc"])
    log.info("  sym_acc  = %.4f", final["sym_acc"])
    log.info("  λ=%.3f  ε=%.3f  lr=%.2e  lora_rank=%d", args.lambda_val, args.epsilon, args.lr, args.lora_rank)
    log.info("  Log file: %s", out_dir / f"{args.run_id}.log")
    log.info(
        "  To resume from last checkpoint:\n"
        "    --resume-from-checkpoint %s",
        ckpt_base / f"epoch_{final['epoch']}",
    )
    log.info("=" * 60)
    wandb.finish()


if __name__ == "__main__":
    t0 = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path",    required=True)
    parser.add_argument("--train-csv",     default="",
                        help="Path to fever_train_perturbed.csv (used when --dataset fever)")
    parser.add_argument("--val-csv",       default="",
                        help="Path to fever_val.csv (used when --dataset fever)")
    parser.add_argument("--sym-val-csv",   required=True,
                        help="Path to fever_symmetric.csv (always used as cross-dataset eval)")
    parser.add_argument("--output-path",   required=True)
    parser.add_argument("--run-id",        required=True)
    parser.add_argument("--lambda-val",    type=float, default=float(os.getenv("LAMBDA",            "1.0")))
    parser.add_argument("--epsilon",       type=float, default=float(os.getenv("EPSILON",           "0.0")))
    parser.add_argument("--lr",            type=float, default=float(os.getenv("LR",                "2e-5")))
    parser.add_argument("--batch-size",    type=int,   default=int(os.getenv("BATCH_SIZE",          "16")))
    parser.add_argument("--epochs",        type=int,   default=int(os.getenv("EPOCHS",              "3")))
    parser.add_argument("--lora-rank",     type=int,   default=int(os.getenv("LORA_RANK",           "0")))
    parser.add_argument("--ckpt-every-steps", type=int, default=int(os.getenv("CKPT_EVERY_STEPS",  "0")),
                        help="Save a mid-epoch checkpoint every N steps. 0 = epoch-only checkpoints.")
    parser.add_argument("--resume-from-checkpoint", default=None,
                        help="Path to a checkpoint directory to resume from.")
    parser.add_argument("--augmentation-only", action="store_true",
                        default=os.getenv("AUGMENTATION_ONLY", "0") == "1",
                        help="Augmentation-only baseline: CE on all rows (originals + perturbations), no stability loss.")
    parser.add_argument("--model-name",   default=os.getenv("MODEL_NAME", "bert_base_uncased"),
                        help="Short model name (used in training_log.json for compare_runs).")
    parser.add_argument("--max-seq-len",  type=int, default=128,
                        help="Max tokenizer sequence length.")
    parser.add_argument("--dataset",      choices=["fever", "vitaminc"],
                        default=os.getenv("DATASET", "fever"),
                        help="Training dataset. 'fever' uses fever_train_perturbed.csv with "
                             "synthetic perturbations. 'vitaminc' uses vitaminc_train.csv with "
                             "natural contrastive pairs grouped by case_id.")
    parser.add_argument("--vitaminc-train-csv", default="",
                        help="Path to vitaminc_train.csv (required when --dataset vitaminc)")
    parser.add_argument("--vitaminc-val-csv",   default="",
                        help="Path to vitaminc_val.csv (required when --dataset vitaminc)")
    parser.add_argument("--held-out-perturbations", nargs="*", default=[],
                        help="Perturbation families to exclude from the stability loss pairing. "
                             "These families still appear in the training CSV and contribute CE "
                             "signal, but are never used as the (h, h') neighbor pair for the "
                             "regularizer. Used for leave-one-out generalization eval. "
                             "Example: --held-out-perturbations claim_synonym")

    args = parser.parse_args()
    main(args)

    elapsed = time.time() - t0
    print(f"\nTotal wall time: {elapsed/3600:.2f}h  ({elapsed/60:.1f}min)")
