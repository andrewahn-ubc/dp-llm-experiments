"""Deterministic 100-point validation design for the Llama-2 experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass


LEARNING_RATES = (5e-6, 1e-5, 2e-5, 4e-5)
BENIGN_WEIGHTS = (0.0, 0.1, 0.3, 1.0, 3.0)
REGULARIZED_LAMBDAS = (0.1, 1.0, 3.0, 10.0, 30.0)
REGULARIZED_EPSILONS = (-1.0, -0.5, 0.0, 0.5)


@dataclass(frozen=True)
class SweepConfig:
    config_id: int
    learning_rate: float
    lambda_val: float
    epsilon: float
    benign_lm_weight: float
    rollout_length: int = 3
    stability_loss: str = "squared_hinge"

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def slug(self) -> str:
        return f"cfg{self.config_id:03d}"


def sweep_configs() -> list[SweepConfig]:
    """Return exactly 100 non-duplicated configurations.

    Twenty cells are lambda=0 controls: four learning rates crossed with all
    five benign weights (epsilon is inert and fixed to zero). The remaining 80
    cells cross four learning rates, five positive lambdas and four epsilons.
    Benign weight is assigned as a balanced fractional-factorial dimension so
    every value occurs exactly 16 times without spending 400 regularized cells.
    """
    rows: list[SweepConfig] = []

    for lr in LEARNING_RATES:
        for beta in BENIGN_WEIGHTS:
            rows.append(
                SweepConfig(
                    config_id=len(rows),
                    learning_rate=lr,
                    lambda_val=0.0,
                    epsilon=0.0,
                    benign_lm_weight=beta,
                )
            )

    for lr_i, lr in enumerate(LEARNING_RATES):
        for lam_i, lam in enumerate(REGULARIZED_LAMBDAS):
            for eps_i, eps in enumerate(REGULARIZED_EPSILONS):
                beta_i = (lr_i + 2 * lam_i + 3 * eps_i) % len(BENIGN_WEIGHTS)
                rows.append(
                    SweepConfig(
                        config_id=len(rows),
                        learning_rate=lr,
                        lambda_val=lam,
                        epsilon=eps,
                        benign_lm_weight=BENIGN_WEIGHTS[beta_i],
                    )
                )

    assert len(rows) == 100
    signatures = {
        (r.learning_rate, r.lambda_val, r.epsilon, r.benign_lm_weight)
        for r in rows
    }
    assert len(signatures) == 100
    return rows


def get_config(config_id: int) -> SweepConfig:
    rows = sweep_configs()
    if not 0 <= config_id < len(rows):
        raise ValueError(f"config_id must be in [0, {len(rows) - 1}], got {config_id}")
    return rows[config_id]
