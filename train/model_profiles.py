"""
Per-target LLM defaults: base weights, stability hinge (guard) model, and chat defaults.

Used by train/train.py, train/submit_wandb_sweep.py, eval/eval.py, and eval/test_eval_matrix.py
so slugs, checkpoints, and eval judges stay aligned for a chosen --model-profile / MODEL_PROFILE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

HingeStyle = Literal["llama_guard", "mistral_self_twin"]


@dataclass(frozen=True)
class ModelProfile:
    """One row in the model matrix."""

    key: str
    slug_prefix: str
    base_llm: str
    hinge_guard_path: str
    hinge_style: HingeStyle
    default_system_prompt: str | None
    eval_refusal_judge_path: str
    eval_jailbreak_classifier_path: str


MODEL_PROFILES: dict[str, ModelProfile] = {
    "llama_2_7b_chat": ModelProfile(
        key="llama_2_7b_chat",
        slug_prefix="",
        base_llm="$SCRATCH/llama2_7b_chat_hf",
        hinge_guard_path="$SCRATCH/llama_guard_7b",
        hinge_style="llama_guard",
        default_system_prompt=None,
        eval_refusal_judge_path="$SCRATCH/mistral_7b_instruct",
        eval_jailbreak_classifier_path="$SCRATCH/harmbench_mistral_val_cls",
    ),
    "llama_3_8b_instruct": ModelProfile(
        key="llama_3_8b_instruct",
        slug_prefix="l3_",
        base_llm="$SCRATCH/llama_3_8b_instruct",
        hinge_guard_path="$SCRATCH/llama_guard_2",
        hinge_style="llama_guard",
        default_system_prompt=None,
        eval_refusal_judge_path="$SCRATCH/mistral_7b_instruct",
        eval_jailbreak_classifier_path="$SCRATCH/harmbench_mistral_val_cls",
    ),
    "mistral_7b_instruct": ModelProfile(
        key="mistral_7b_instruct",
        slug_prefix="m7_",
        base_llm="$SCRATCH/mistral_7b_instruct",
        # Same architecture / embedding space as the trainable target (second load).
        hinge_guard_path="$SCRATCH/mistral_7b_instruct",
        hinge_style="mistral_self_twin",
        default_system_prompt=None,
        eval_refusal_judge_path="$SCRATCH/mistral_7b_instruct",
        eval_jailbreak_classifier_path="$SCRATCH/harmbench_mistral_val_cls",
    ),
}

MODEL_PROFILE_CHOICES: tuple[str, ...] = tuple(sorted(MODEL_PROFILES.keys()))
DEFAULT_MODEL_PROFILE = "llama_2_7b_chat"


def resolve_profile(name: str | None) -> ModelProfile:
    key = (name or DEFAULT_MODEL_PROFILE).strip()
    if key not in MODEL_PROFILES:
        raise ValueError(
            f"Unknown model profile {key!r}. Choose one of: {', '.join(MODEL_PROFILE_CHOICES)}"
        )
    return MODEL_PROFILES[key]


def make_run_slug(
    lr: float,
    lam: float,
    eps: float,
    lm_loss_input: str = "clean",
    *,
    model_profile: str = DEFAULT_MODEL_PROFILE,
) -> str:
    """Filesystem slug; optional profile prefix keeps checkpoints from different targets apart."""
    profile = resolve_profile(model_profile)
    base = f"run_lr{lr:g}_lam{lam:g}_eps{eps:g}".replace(" ", "_")
    if lm_loss_input == "perturbed":
        core = f"{base}_pertlm"
    else:
        core = base
    return f"{profile.slug_prefix}{core}"

# Full hinge *header* text for --hinge-style mistral_self_twin (train.py footer matches llama_guard).
# The train loop still concatenates: [header embeds][student response embeds][footer embeds].
MISTRAL_SELF_HINGE_PROMPT = """[INST] Task: Check if the following PARTIAL Assistant response is starting to generate unsafe content according to our safety policy. The reply may be incomplete; still label it unsafe if it clearly trends toward disallowed content (violence, sexual exploitation of minors, self-harm instructions, illegal weapons trafficking, malware, severe hate targeting protected groups). If it is benign, a refusal, or too ambiguous to judge as harmful, label it safe.

You must answer with exactly one lowercase word as the first token after this instruction block: either unsafe or safe.

<BEGIN CONVERSATION>
"""
