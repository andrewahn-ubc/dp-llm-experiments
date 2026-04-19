"""
perturb_fever.py — Generate perturbed (claim, evidence) pairs for FEVER training.

PURPOSE
-------
Our symmetric regularizer penalizes prediction inconsistency across
label-preserving perturbations: pairs where the surface form changes but
the underlying fact (and therefore the correct label) does not.

WHICH PERTURBATIONS AND WHY

We include four families, ordered roughly by expected difficulty:

  1. evidence_word_shuffle  — randomly reorder words within each evidence
                              sentence (inspired by FairFlow's PGra / Sinha
                              et al. 2021 "UnNatural Language Inference",
                              which showed BERT is largely word-order
                              invariant, so this tests a real surface bias)

  2. evidence_sent_shuffle  — randomly reorder evidence sentences
                              (weak version of 1; included as a sanity check)

  3. evidence_trunc         — drop the last evidence sentence; tests
                              robustness to partial evidence presentation

  4. claim_synonym          — replace up to 2 content words in the claim
                              with WordNet synonyms; tests lexical shortcut
                              sensitivity (the primary bias FeverSymmetric
                              was designed to expose)



Output CSV columns:
  claim, evidence, label, original_claim, original_evidence, perturbation_type

Usage:
    python fact_check/perturb_fever.py \
        --input  $SCRATCH/dp-llm-experiments/fact_check_data/fever_train.csv \
        --output $SCRATCH/dp-llm-experiments/fact_check_data/fever_train_perturbed.csv \
        --seed 42
"""

import argparse
import random
import re
import pandas as pd
from pathlib import Path


def split_sentences(text: str) -> list[str]:
    """Naive sentence splitter adequate for single-sentence Wikipedia evidence."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p for p in parts if p.strip()]


def perturb_evidence_word_shuffle(evidence: str, rng: random.Random) -> str:
    """
    Shuffle words within each evidence sentence independently.
    Inspired by FairFlow's PGra and Sinha et al. (2021), who showed
    BERT is largely invariant to word order, meaning it tends to
    assign confident predictions to ungrammatical inputs, which is a
    form of shortcut reliance. We penalize that with our regularizer.
    """
    sentences = split_sentences(evidence)
    shuffled = []
    for s in sentences:
        words = s.split()
        rng.shuffle(words)
        shuffled.append(" ".join(words))
    return " ".join(shuffled)


def perturb_evidence_sent_shuffle(evidence: str, rng: random.Random) -> str:
    """Randomly reorder evidence sentences (weaker than word shuffle)."""
    sentences = split_sentences(evidence)
    if len(sentences) <= 1:
        return evidence
    rng.shuffle(sentences)
    return " ".join(sentences)


def perturb_evidence_trunc(evidence: str) -> str:
    """
    Drop the last evidence sentence.
    Tests whether the model's confidence degrades gracefully with
    partial evidence , it should remain relatively stable when the
    dropped sentence is not the key fact.
    """
    sentences = split_sentences(evidence)
    if len(sentences) <= 1:
        return evidence
    return " ".join(sentences[:-1])


def perturb_claim_synonym(claim: str, rng: random.Random) -> str:
    """
    Replace up to 2 content words in the claim with WordNet synonyms.

    This directly targets the primary annotation artifact in FEVER:
    models learn that negation words ('not', 'never', 'failed to')
    predict REFUTES regardless of evidence. FeverSymmetric was
    designed to catch exactly this. Synonym-perturbed claims preserve
    the fact but change surface lexical patterns the model might rely on.

    Falls back to original claim if nltk/wordnet is unavailable.
    """
    try:
        from nltk.corpus import wordnet
        import nltk
        nltk.download("wordnet", quiet=True)
        nltk.download("punkt", quiet=True)
    except ImportError:
        return claim

    words = claim.split()
    replaceable = []
    for i, w in enumerate(words):
        synsets = wordnet.synsets(w)
        synonyms = set()
        for syn in synsets:
            for lemma in syn.lemmas():
                name = lemma.name().replace("_", " ")
                if name.lower() != w.lower() and " " not in name:
                    synonyms.add(name)
        if synonyms:
            replaceable.append((i, list(synonyms)))

    rng.shuffle(replaceable)
    for idx, syns in replaceable[:2]:
        words[idx] = rng.choice(syns)

    return " ".join(words)


PERTURBATION_FNS = {
    "evidence_word_shuffle": perturb_evidence_word_shuffle,
    "evidence_sent_shuffle": perturb_evidence_sent_shuffle,
    "evidence_trunc":        perturb_evidence_trunc,
    "claim_synonym":         perturb_claim_synonym,
}

# Which perturbations modify the claim (vs the evidence)
CLAIM_PERTURBATIONS = {"claim_synonym"}


def main():
    from fact_check.config_loader import load_config

    cfg      = load_config()
    data_dir = Path(cfg.paths.data_dir)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--input",  default=str(data_dir / "fever_train.csv"),
                        help="Path to fever_train.csv (default: from config)")
    parser.add_argument("--output", default=str(data_dir / "fever_train_perturbed.csv"),
                        help="Output path (default: from config)")
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument(
        "--perturbations",
        nargs="+",
        default=["evidence_word_shuffle", "evidence_sent_shuffle",
                 "evidence_trunc", "claim_synonym"],
        choices=list(PERTURBATION_FNS.keys()),
        help="Which perturbation families to apply."
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")

    rows = []
    for _, row in df.iterrows():
        claim    = str(row["claim"])
        evidence = str(row["evidence"])
        label    = row["label"]

        for ptype in args.perturbations:
            fn = PERTURBATION_FNS[ptype]

            if ptype in CLAIM_PERTURBATIONS:
                perturbed_claim    = fn(claim, rng)
                perturbed_evidence = evidence
            else:
                perturbed_claim    = claim
                # evidence_trunc doesn't take rng
                if ptype == "evidence_trunc":
                    perturbed_evidence = fn(evidence)
                else:
                    perturbed_evidence = fn(evidence, rng)

            # Skip if perturbation had no effect (e.g. single-sentence evidence)
            if perturbed_claim == claim and perturbed_evidence == evidence:
                continue

            rows.append({
                "claim":             perturbed_claim,
                "evidence":          perturbed_evidence,
                "label":             label,
                "original_claim":    claim,
                "original_evidence": evidence,
                "perturbation_type": ptype,
            })

    out_df = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"Wrote {len(out_df)} perturbed rows to {args.output}")
    print("Perturbation breakdown:")
    print(out_df["perturbation_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
