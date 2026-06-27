"""Neural QI policy optimizer.

This module adds a real trainable AI component for selecting QI operating
parameters. The policy learns to imitate a resource-constrained grid-search
oracle and can then infer QI parameters in a single forward pass.

This strengthens the paper claim from "adaptive heuristic optimization" to
"AI-based QI parameter policy learning under plasma-channel conditions."
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .optimizer import (
    OptimizerConfig,
    QIParameterBounds,
    grid_search_strategy,
    normalized_resource_cost,
)
from .plasma_channel import PlasmaChannelConfig, generate_plasma_dataset
from .qi_bounds import quantum_illumination_error_bound, db_gain


@dataclass(frozen=True)
class PolicyFeatureSpec:
    input_columns: tuple[str, ...] = (
        "attenuation_db",
        "effective_reflectivity_kappa",
        "effective_noise_nb",
        "phase_distortion_rad",
        "radar_frequency_ghz",
        "electron_density_m3",
        "collision_frequency_hz",
        "sheath_thickness_m",
    )
    target_columns: tuple[str, ...] = ("oracle_ns_norm", "oracle_m_norm")


@dataclass
class PolicyTrainConfig:
    n_samples: int = 50000
    epochs: int = 60
    batch_size: int = 1024
    learning_rate: float = 1e-3
    hidden_dim: int = 128
    depth: int = 4
    dropout: float = 0.02
    test_size: float = 0.2
    seed: int = 42
    device: str = "auto"
    grid_size: int = 35
    budget: float = 0.65
    output_dir: str = "results"
    table_dir: str = "tables"
    model_dir: str = "models"


class QIPolicyNet(nn.Module):
    """Policy network that predicts normalized Ns and M in [0, 1]."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, depth: int = 4, dropout: float = 0.02):
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be at least 2")
        layers: list[nn.Module] = []
        last = input_dim
        for _ in range(depth - 1):
            layers.append(nn.Linear(last, hidden_dim))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last = hidden_dim
        layers.append(nn.Linear(last, 2))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def ns_m_to_norm(ns: float, m: float, bounds: QIParameterBounds) -> tuple[float, float]:
    ns_norm = (ns - bounds.ns_min) / max(bounds.ns_max - bounds.ns_min, 1e-12)
    m_norm = (m - bounds.m_min) / max(bounds.m_max - bounds.m_min, 1e-12)
    return float(np.clip(ns_norm, 0.0, 1.0)), float(np.clip(m_norm, 0.0, 1.0))


def norm_to_ns_m(norm: np.ndarray, bounds: QIParameterBounds) -> tuple[np.ndarray, np.ndarray]:
    norm = np.asarray(norm, dtype=float)
    ns_norm = np.clip(norm[:, 0], 0.0, 1.0)
    m_norm = np.clip(norm[:, 1], 0.0, 1.0)
    ns = bounds.ns_min + ns_norm * (bounds.ns_max - bounds.ns_min)
    m = bounds.m_min + m_norm * (bounds.m_max - bounds.m_min)
    return ns, m


def project_to_budget(ns: np.ndarray, m: np.ndarray, bounds: QIParameterBounds) -> tuple[np.ndarray, np.ndarray]:
    """Project predicted Ns/M to the feasible resource budget approximately."""
    ns = np.asarray(ns, dtype=float).copy()
    m = np.asarray(m, dtype=float).copy()

    for i in range(len(ns)):
        cost = float(normalized_resource_cost(ns[i], m[i], bounds))
        if cost <= bounds.budget:
            continue

        ns_norm = (ns[i] - bounds.ns_min) / max(bounds.ns_max - bounds.ns_min, 1e-12)
        m_norm = (m[i] - bounds.m_min) / max(bounds.m_max - bounds.m_min, 1e-12)
        if cost > 1e-12:
            scale = bounds.budget / cost
            ns_norm *= scale
            m_norm *= scale
        ns[i] = bounds.ns_min + np.clip(ns_norm, 0.0, 1.0) * (bounds.ns_max - bounds.ns_min)
        m[i] = bounds.m_min + np.clip(m_norm, 0.0, 1.0) * (bounds.m_max - bounds.m_min)

    return ns, m


def build_oracle_policy_dataset(
    n_samples: int,
    seed: int,
    bounds: QIParameterBounds,
    optimizer_config: OptimizerConfig,
) -> pd.DataFrame:
    """Generate plasma samples and grid-search oracle labels."""
    df = generate_plasma_dataset(PlasmaChannelConfig(n_samples=n_samples, seed=seed))
    oracle_ns = []
    oracle_m = []
    oracle_pe = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building grid-search oracle"):
        ns, m, pe = grid_search_strategy(
            float(row["effective_reflectivity_kappa"]),
            float(row["effective_noise_nb"]),
            bounds,
            optimizer_config,
        )
        oracle_ns.append(ns)
        oracle_m.append(m)
        oracle_pe.append(pe)

    ns_norm, m_norm = zip(*(ns_m_to_norm(ns, m, bounds) for ns, m in zip(oracle_ns, oracle_m)))
    df = df.copy()
    df["oracle_ns"] = oracle_ns
    df["oracle_m"] = oracle_m
    df["oracle_pe"] = oracle_pe
    df["oracle_ns_norm"] = ns_norm
    df["oracle_m_norm"] = m_norm
    return df


def train_qi_policy(config: PolicyTrainConfig) -> dict[str, object]:
    """Train neural QI policy against a grid-search oracle."""

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    output_dir = Path(config.output_dir)
    table_dir = Path(config.table_dir)
    model_dir = Path(config.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    bounds = QIParameterBounds(budget=config.budget)
    oracle_config = OptimizerConfig(grid_size=config.grid_size, seed=config.seed)

    spec = PolicyFeatureSpec()
    df = build_oracle_policy_dataset(config.n_samples, config.seed, bounds, oracle_config)

    x = df.loc[:, spec.input_columns].to_numpy(dtype=np.float32)
    y = df.loc[:, spec.target_columns].to_numpy(dtype=np.float32)

    x_train, x_test, y_train, y_test, df_train, df_test = train_test_split(
        x, y, df, test_size=config.test_size, random_state=config.seed, shuffle=True
    )

    x_scaler = StandardScaler()
    x_train_s = x_scaler.fit_transform(x_train).astype(np.float32)
    x_test_s = x_scaler.transform(x_test).astype(np.float32)

    device = resolve_device(config.device)
    model = QIPolicyNet(
        input_dim=len(spec.input_columns),
        hidden_dim=config.hidden_dim,
        depth=config.depth,
        dropout=config.dropout,
    ).to(device)

    loader = DataLoader(
        TensorDataset(torch.tensor(x_train_s), torch.tensor(y_train)),
        batch_size=config.batch_size,
        shuffle=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()

    history = []
    for epoch in tqdm(range(1, config.epochs + 1), desc="Training QI policy"):
        model.train()
        total = 0.0
        seen = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

            total += float(loss.detach().cpu()) * xb.shape[0]
            seen += xb.shape[0]

        history.append({"epoch": epoch, "policy_loss": total / max(seen, 1)})

    model.eval()
    with torch.no_grad():
        pred_norm = model(torch.tensor(x_test_s, dtype=torch.float32, device=device)).cpu().numpy()

    pred_ns, pred_m = norm_to_ns_m(pred_norm, bounds)
    pred_ns, pred_m = project_to_budget(pred_ns, pred_m, bounds)

    df_test = df_test.reset_index(drop=True)
    policy_pe = quantum_illumination_error_bound(
        pred_ns,
        pred_m,
        df_test["effective_reflectivity_kappa"].to_numpy(dtype=float),
        df_test["effective_noise_nb"].to_numpy(dtype=float),
    )

    oracle_pe = df_test["oracle_pe"].to_numpy(dtype=float)
    fixed_pe = quantum_illumination_error_bound(
        0.02,
        1.0e6,
        df_test["effective_reflectivity_kappa"].to_numpy(dtype=float),
        df_test["effective_noise_nb"].to_numpy(dtype=float),
    )

    metrics = {
        "ns_mae": float(mean_absolute_error(df_test["oracle_ns"], pred_ns)),
        "m_mae": float(mean_absolute_error(df_test["oracle_m"], pred_m)),
        "ns_r2": float(r2_score(df_test["oracle_ns"], pred_ns)),
        "m_r2": float(r2_score(df_test["oracle_m"], pred_m)),
        "policy_pe_mean": float(np.mean(policy_pe)),
        "oracle_pe_mean": float(np.mean(oracle_pe)),
        "fixed_pe_mean": float(np.mean(fixed_pe)),
        "policy_gain_db_vs_fixed_mean": float(np.mean(db_gain(fixed_pe, policy_pe))),
        "oracle_gap_percent": float(100.0 * (np.mean(policy_pe) - np.mean(oracle_pe)) / max(np.mean(oracle_pe), 1e-12)),
        "budget_violation_rate": float(np.mean(normalized_resource_cost(pred_ns, pred_m, bounds) > bounds.budget + 1e-9)),
    }

    history_df = pd.DataFrame(history)
    metrics_df = pd.DataFrame([metrics])
    pred_df = pd.DataFrame(
        {
            "plasma_severity": df_test["plasma_severity"],
            "pred_ns": pred_ns,
            "pred_m": pred_m,
            "policy_pe": policy_pe,
            "oracle_ns": df_test["oracle_ns"],
            "oracle_m": df_test["oracle_m"],
            "oracle_pe": oracle_pe,
            "fixed_pe": fixed_pe,
            "policy_gain_db_vs_fixed": db_gain(fixed_pe, policy_pe),
            "resource_cost": normalized_resource_cost(pred_ns, pred_m, bounds),
        }
    )

    history_path = output_dir / "qi_policy_training_log.csv"
    pred_path = output_dir / "qi_policy_predictions.csv"
    metrics_path = table_dir / "qi_policy_metrics.csv"
    model_path = model_dir / "qi_policy_net.pt"

    history_df.to_csv(history_path, index=False)
    pred_df.to_csv(pred_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "input_columns": spec.input_columns,
            "target_columns": spec.target_columns,
            "x_scaler_mean": x_scaler.mean_,
            "x_scaler_scale": x_scaler.scale_,
        },
        model_path,
    )

    return {
        "model": model,
        "history": history_df,
        "metrics": metrics_df,
        "predictions": pred_df,
        "history_path": str(history_path),
        "prediction_path": str(pred_path),
        "metrics_path": str(metrics_path),
        "model_path": str(model_path),
        "device": str(device),
    }
