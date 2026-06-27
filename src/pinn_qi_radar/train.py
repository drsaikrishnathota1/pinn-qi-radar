"""Training utilities for the plasma-sheath PINN surrogate."""

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

from .plasma_channel import PlasmaChannelConfig, generate_plasma_dataset
from .pinn_model import PINNFeatureSpec, PlasmaPINN, monotonic_physics_residual


@dataclass
class TrainConfig:
    n_samples: int = 50000
    epochs: int = 50
    batch_size: int = 512
    learning_rate: float = 1e-3
    hidden_dim: int = 128
    depth: int = 4
    dropout: float = 0.0
    physics_weight: float = 0.05
    test_size: float = 0.2
    seed: int = 42
    device: str = "auto"
    output_dir: str = "results"
    table_dir: str = "tables"
    model_dir: str = "models"


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def prepare_training_data(
    df: pd.DataFrame,
    spec: PINNFeatureSpec | None = None,
    test_size: float = 0.2,
    seed: int = 42,
):
    """Split dataframe and standardize input/output columns."""
    spec = spec or PINNFeatureSpec()

    x = df.loc[:, spec.input_columns].to_numpy(dtype=np.float32)
    y = df.loc[:, spec.target_columns].to_numpy(dtype=np.float32)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_train_s = x_scaler.fit_transform(x_train).astype(np.float32)
    x_test_s = x_scaler.transform(x_test).astype(np.float32)
    y_train_s = y_scaler.fit_transform(y_train).astype(np.float32)
    y_test_s = y_scaler.transform(y_test).astype(np.float32)

    return x_train_s, x_test_s, y_train_s, y_test_s, x_scaler, y_scaler


def train_pinn_surrogate(config: TrainConfig) -> dict[str, object]:
    """Train the PINN surrogate and save metrics/artifacts."""

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    output_dir = Path(config.output_dir)
    table_dir = Path(config.table_dir)
    model_dir = Path(config.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    spec = PINNFeatureSpec()
    df = generate_plasma_dataset(
        PlasmaChannelConfig(n_samples=config.n_samples, seed=config.seed)
    )

    x_train, x_test, y_train, y_test, x_scaler, y_scaler = prepare_training_data(
        df, spec=spec, test_size=config.test_size, seed=config.seed
    )

    device = resolve_device(config.device)

    train_ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
    )

    model = PlasmaPINN(
        input_dim=len(spec.input_columns),
        output_dim=len(spec.target_columns),
        hidden_dim=config.hidden_dim,
        depth=config.depth,
        dropout=config.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    mse_loss = nn.MSELoss()

    history: list[dict[str, float]] = []

    for epoch in tqdm(range(1, config.epochs + 1), desc="Training PINN"):
        model.train()
        total_loss = 0.0
        total_data_loss = 0.0
        total_physics_loss = 0.0
        n_seen = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            xb_physics = xb.detach().clone().requires_grad_(True)

            optimizer.zero_grad(set_to_none=True)

            pred = model(xb_physics)
            data_loss = mse_loss(pred, yb)
            physics_loss = monotonic_physics_residual(model, xb_physics)
            loss = data_loss + config.physics_weight * physics_loss

            loss.backward()
            optimizer.step()

            batch_n = xb.shape[0]
            total_loss += float(loss.detach().cpu()) * batch_n
            total_data_loss += float(data_loss.detach().cpu()) * batch_n
            total_physics_loss += float(physics_loss.detach().cpu()) * batch_n
            n_seen += batch_n

        history.append(
            {
                "epoch": epoch,
                "loss": total_loss / n_seen,
                "data_loss": total_data_loss / n_seen,
                "physics_loss": total_physics_loss / n_seen,
            }
        )

    model.eval()
    with torch.no_grad():
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32, device=device)
        pred_test_s = model(x_test_tensor).detach().cpu().numpy()

    pred_test = y_scaler.inverse_transform(pred_test_s)
    y_test_raw = y_scaler.inverse_transform(y_test)

    metrics = []
    for i, col in enumerate(spec.target_columns):
        y_true = y_test_raw[:, i]
        y_pred = pred_test[:, i]
        metrics.append(
            {
                "target": col,
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "r2": float(r2_score(y_true, y_pred)),
            }
        )

    history_df = pd.DataFrame(history)
    metrics_df = pd.DataFrame(metrics)

    history_path = output_dir / "pinn_training_log.csv"
    metrics_path = table_dir / "pinn_prediction_metrics.csv"
    model_path = model_dir / "plasma_pinn_surrogate.pt"

    history_df.to_csv(history_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "input_columns": spec.input_columns,
            "target_columns": spec.target_columns,
            "x_scaler_mean": x_scaler.mean_,
            "x_scaler_scale": x_scaler.scale_,
            "y_scaler_mean": y_scaler.mean_,
            "y_scaler_scale": y_scaler.scale_,
        },
        model_path,
    )

    return {
        "model": model,
        "history": history_df,
        "metrics": metrics_df,
        "history_path": str(history_path),
        "metrics_path": str(metrics_path),
        "model_path": str(model_path),
        "device": str(device),
    }
