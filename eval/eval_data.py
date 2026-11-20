"""Import-safe data helpers shared by evaluation code and tests."""

from __future__ import annotations


def harmbench_behaviors(batch, fallback_prompts):
    """Return clean harmful goals for HarmBench classification when present.

    Attack strings are generation inputs, not the behavior the classifier is
    supposed to detect. Validation data uses ``Original Prompt`` and test data
    uses ``goal``. Legacy inputs lacking both fall back to the attack prompt.
    """
    for candidate in ("Original Prompt", "goal"):
        if candidate in batch.columns:
            return batch[candidate].astype(str).tolist()
    return list(fallback_prompts)
