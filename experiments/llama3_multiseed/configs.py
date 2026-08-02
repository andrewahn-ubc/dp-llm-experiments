"""Shared config for Llama-3 multi-seed seen/heldout table experiments.

Train seeds 1 and 2 only; seed 0 numbers come from ``seed0_from_image.csv``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from train.model_profiles import make_run_slug

LR = 2e-5
MODEL_PROFILE = "llama_3_8b_instruct"
EPOCH = 2  # --total-epochs 1 + --training-half both → evaluate *_epoch2
SEEDS = (1, 2)
FAMILIES = ("gcg", "autodan", "pair")
BENCHMARKS = ("advbench", "harmbench", "jailbreakbench")


@dataclass(frozen=True)
class TrainConfig:
    """One training job (seen or held-out adapter)."""

    config_id: str
    role: Literal["seen", "heldout"]
    lam: float
    eps: float
    lm_loss_input: Literal["clean", "perturbed"]
    heldout_family: str | None = None  # required when role == heldout

    def base_slug(self, seed: int) -> str:
        return make_run_slug(
            LR,
            self.lam,
            self.eps,
            self.lm_loss_input,
            model_profile=MODEL_PROFILE,
            seed=seed,
        )

    def run_slug(self, seed: int) -> str:
        base = self.base_slug(seed)
        if self.role == "heldout":
            assert self.heldout_family
            return f"heldout_{self.heldout_family}_{base}"
        return base

    def finetuned_base(self, checkpoint_root: str, seed: int) -> str:
        return f"{checkpoint_root.rstrip('/')}/{self.run_slug(seed)}_finetuned_llm"

    def ckpt_path(self, checkpoint_root: str, seed: int, epoch: int = EPOCH) -> str:
        return f"{self.finetuned_base(checkpoint_root, seed)}_epoch{epoch}"


# 10 model configs × 2 seeds = 20 train/eval array tasks.
TRAIN_CONFIGS: tuple[TrainConfig, ...] = (
    # Seen (3)
    TrainConfig("seen_dcl_1_-1", "seen", 1.0, -1.0, "clean"),
    TrainConfig("seen_dcl_1_-0.5", "seen", 1.0, -0.5, "clean"),
    TrainConfig("seen_advsft", "seen", 0.0, 0.0, "perturbed"),
    # Heldout DCL (4)
    TrainConfig("ho_gcg_1_-0.5", "heldout", 1.0, -0.5, "clean", "gcg"),
    TrainConfig("ho_gcg_3_1", "heldout", 3.0, 1.0, "clean", "gcg"),
    TrainConfig("ho_autodan_1_-1", "heldout", 1.0, -1.0, "clean", "autodan"),
    TrainConfig("ho_pair_30_1", "heldout", 30.0, 1.0, "clean", "pair"),
    # Heldout Adv-SFT (3)
    TrainConfig("ho_gcg_advsft", "heldout", 0.0, 0.0, "perturbed", "gcg"),
    TrainConfig("ho_autodan_advsft", "heldout", 0.0, 0.0, "perturbed", "autodan"),
    TrainConfig("ho_pair_advsft", "heldout", 0.0, 0.0, "perturbed", "pair"),
)

N_CONFIGS = len(TRAIN_CONFIGS)
N_TRAIN_JOBS = N_CONFIGS * len(SEEDS)  # 20


def array_index_to_seed_config(task_id: int) -> tuple[int, TrainConfig]:
    """Map SLURM_ARRAY_TASK_ID ∈ [0, 19] → (seed, config)."""
    if task_id < 0 or task_id >= N_TRAIN_JOBS:
        raise IndexError(f"task_id {task_id} out of range [0, {N_TRAIN_JOBS - 1}]")
    seed_idx, cfg_idx = divmod(task_id, N_CONFIGS)
    return SEEDS[seed_idx], TRAIN_CONFIGS[cfg_idx]


@dataclass(frozen=True)
class TableCell:
    """One ASR/FRR cell in the seen or heldout family table."""

    mode: Literal["seen", "heldout"]
    family: str  # gcg | autodan | pair | advsft | all
    benchmark: str  # advbench | harmbench | jailbreakbench | mean
    # Which trained adapter fills this cell (None for derived "all" rows).
    source_config_id: str | None
    # For heldout family rows, which family column to read from harmful CSV.
    eval_family: str | None = None


def _cell(
    mode: str,
    family: str,
    bench: str,
    source: str | None,
    eval_family: str | None = None,
) -> TableCell:
    return TableCell(mode, family, bench, source, eval_family or (family if family in FAMILIES else None))


# Cell → checkpoint map matching the image (plus Adv-SFT rows).
TABLE_CELLS: tuple[TableCell, ...] = (
    # ----- Seen DCL families -----
    _cell("seen", "gcg", "advbench", "seen_dcl_1_-1", "gcg"),
    _cell("seen", "gcg", "harmbench", "seen_dcl_1_-0.5", "gcg"),
    _cell("seen", "gcg", "jailbreakbench", "seen_dcl_1_-0.5", "gcg"),
    _cell("seen", "autodan", "advbench", "seen_dcl_1_-1", "autodan"),
    _cell("seen", "autodan", "harmbench", "seen_dcl_1_-1", "autodan"),
    _cell("seen", "autodan", "jailbreakbench", "seen_dcl_1_-1", "autodan"),
    _cell("seen", "pair", "advbench", "seen_dcl_1_-0.5", "pair"),
    _cell("seen", "pair", "harmbench", "seen_dcl_1_-0.5", "pair"),
    _cell("seen", "pair", "jailbreakbench", "seen_dcl_1_-0.5", "pair"),
    # Seen Adv-SFT (same adapter for all benches; FRR shared)
    _cell("seen", "advsft", "advbench", "seen_advsft", "all"),
    _cell("seen", "advsft", "harmbench", "seen_advsft", "all"),
    _cell("seen", "advsft", "jailbreakbench", "seen_advsft", "all"),
    # ----- Heldout DCL -----
    _cell("heldout", "gcg", "advbench", "ho_gcg_1_-0.5", "gcg"),
    _cell("heldout", "gcg", "harmbench", "ho_gcg_3_1", "gcg"),
    _cell("heldout", "gcg", "jailbreakbench", "ho_gcg_3_1", "gcg"),
    _cell("heldout", "autodan", "advbench", "ho_autodan_1_-1", "autodan"),
    _cell("heldout", "autodan", "harmbench", "ho_autodan_1_-1", "autodan"),
    _cell("heldout", "autodan", "jailbreakbench", "ho_autodan_1_-1", "autodan"),
    _cell("heldout", "pair", "advbench", "ho_pair_30_1", "pair"),
    _cell("heldout", "pair", "harmbench", "ho_pair_30_1", "pair"),
    _cell("heldout", "pair", "jailbreakbench", "ho_pair_30_1", "pair"),
    # Heldout Adv-SFT per family
    _cell("heldout", "advsft_gcg", "advbench", "ho_gcg_advsft", "gcg"),
    _cell("heldout", "advsft_gcg", "harmbench", "ho_gcg_advsft", "gcg"),
    _cell("heldout", "advsft_gcg", "jailbreakbench", "ho_gcg_advsft", "gcg"),
    _cell("heldout", "advsft_autodan", "advbench", "ho_autodan_advsft", "autodan"),
    _cell("heldout", "advsft_autodan", "harmbench", "ho_autodan_advsft", "autodan"),
    _cell("heldout", "advsft_autodan", "jailbreakbench", "ho_autodan_advsft", "autodan"),
    _cell("heldout", "advsft_pair", "advbench", "ho_pair_advsft", "pair"),
    _cell("heldout", "advsft_pair", "harmbench", "ho_pair_advsft", "pair"),
    _cell("heldout", "advsft_pair", "jailbreakbench", "ho_pair_advsft", "pair"),
)

CONFIG_BY_ID = {c.config_id: c for c in TRAIN_CONFIGS}


def list_train_jobs() -> None:
    print(f"n_jobs={N_TRAIN_JOBS}")
    for i in range(N_TRAIN_JOBS):
        seed, cfg = array_index_to_seed_config(i)
        print(f"{i}\tseed={seed}\t{cfg.config_id}\t{cfg.run_slug(seed)}")


if __name__ == "__main__":
    list_train_jobs()
