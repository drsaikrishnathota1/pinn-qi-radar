"""Ablation experiments for PINN-QI Radar."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .policy_optimizer import PolicyTrainConfig, train_qi_policy
from .train import TrainConfig, train_pinn_surrogate


@dataclass
class AblationConfig:
    n_samples: int = 30000
    pinn_epochs: int = 40
    policy_epochs: int = 40
    batch_size: int = 1024
    seed: int = 42
    device: str = "auto"


def run_ablation(config: AblationConfig) -> pd.DataFrame:
    """Run minimal but useful ablations for manuscript evidence.

    Ablations:
    - PINN with physics loss disabled
    - PINN with physics loss enabled
    - QI policy trained with oracle supervision
    """

    rows = []

    no_phys = train_pinn_surrogate(
        TrainConfig(
            n_samples=config.n_samples,
            epochs=config.pinn_epochs,
            batch_size=config.batch_size,
            physics_weight=0.0,
            seed=config.seed,
            device=config.device,
        )
    )
    for _, row in no_phys["metrics"].iterrows():
        rows.append(
            {
                "experiment": "pinn_without_physics_loss",
                "target": row["target"],
                "mae": row["mae"],
                "rmse": row["rmse"],
                "r2": row["r2"],
            }
        )

    with_phys = train_pinn_surrogate(
        TrainConfig(
            n_samples=config.n_samples,
            epochs=config.pinn_epochs,
            batch_size=config.batch_size,
            physics_weight=0.05,
            seed=config.seed,
            device=config.device,
        )
    )
    for _, row in with_phys["metrics"].iterrows():
        rows.append(
            {
                "experiment": "pinn_with_physics_loss",
                "target": row["target"],
                "mae": row["mae"],
                "rmse": row["rmse"],
                "r2": row["r2"],
            }
        )

    policy = train_qi_policy(
        PolicyTrainConfig(
            n_samples=max(5000, config.n_samples // 2),
            epochs=config.policy_epochs,
            batch_size=config.batch_size,
            seed=config.seed,
            device=config.device,
        )
    )
    policy_metrics = policy["metrics"].iloc[0].to_dict()
    rows.append(
        {
            "experiment": "neural_qi_policy",
            "target": "policy_vs_grid_oracle",
            "mae": policy_metrics["ns_mae"],
            "rmse": policy_metrics["m_mae"],
            "r2": policy_metrics["policy_gain_db_vs_fixed_mean"],
        }
    )

    out = pd.DataFrame(rows)
    Path("tables").mkdir(exist_ok=True)
    out.to_csv("tables/ablation_summary.csv", index=False)
    return out
