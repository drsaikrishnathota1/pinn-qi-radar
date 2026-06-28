#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from pinn_qi_radar.pinn_model import PINNFeatureSpec, PlasmaPINN, count_parameters, monotonic_physics_residual
from pinn_qi_radar.plasma_channel import PlasmaChannelConfig, generate_plasma_dataset
from pinn_qi_radar.train import resolve_device


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--n-samples", type=int, default=100000)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--physics-weight", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--rf-estimators", type=int, default=120)
    p.add_argument("--rf-max-depth", type=int, default=24)
    return p.parse_args()


def metrics(model, y_true, y_pred, targets):
    rows = []
    for i, t in enumerate(targets):
        rows.append({
            "model": model,
            "target": t,
            "mae": mean_absolute_error(y_true[:, i], y_pred[:, i]),
            "rmse": np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])),
            "r2": r2_score(y_true[:, i], y_pred[:, i]),
        })
    return rows


def train_net(name, xtr, ytr, xte, y_scaler, spec, args, physics_weight):
    device = resolve_device(args.device)
    model = PlasmaPINN(
        input_dim=len(spec.input_columns),
        output_dim=len(spec.target_columns),
        hidden_dim=args.hidden_dim,
        depth=args.depth,
    ).to(device)

    loader = DataLoader(
        TensorDataset(torch.tensor(xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32)),
        batch_size=args.batch_size,
        shuffle=True,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.MSELoss()

    start = time.perf_counter()
    for _ in tqdm(range(args.epochs), desc=f"Training {name}"):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            xb2 = xb.detach().clone().requires_grad_(physics_weight > 0)
            opt.zero_grad(set_to_none=True)
            pred = model(xb2)
            data_loss = loss_fn(pred, yb)
            phys_loss = monotonic_physics_residual(model, xb2) if physics_weight > 0 else torch.tensor(0.0, device=device)
            loss = data_loss + physics_weight * phys_loss
            loss.backward()
            opt.step()
    train_time = time.perf_counter() - start

    start = time.perf_counter()
    model.eval()
    with torch.no_grad():
        pred_s = model(torch.tensor(xte, dtype=torch.float32, device=device)).cpu().numpy()
    infer_time = time.perf_counter() - start

    return model, y_scaler.inverse_transform(pred_s), train_time, infer_time


def main():
    args = parse_args()
    if args.quick:
        args.n_samples = 5000
        args.epochs = 5
        args.batch_size = 512
        args.rf_estimators = 30
        args.rf_max_depth = 12

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    Path("tables").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    spec = PINNFeatureSpec()
    df = generate_plasma_dataset(PlasmaChannelConfig(n_samples=args.n_samples, seed=args.seed))

    x = df.loc[:, spec.input_columns].to_numpy(np.float32)
    y = df.loc[:, spec.target_columns].to_numpy(np.float32)

    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.2, random_state=args.seed, shuffle=True)

    xs = StandardScaler()
    ys = StandardScaler()
    xtr_s = xs.fit_transform(xtr).astype(np.float32)
    xte_s = xs.transform(xte).astype(np.float32)
    ytr_s = ys.fit_transform(ytr).astype(np.float32)
    yte_raw = yte

    all_rows = []
    summary = []

    start = time.perf_counter()
    lr = LinearRegression().fit(xtr_s, ytr_s)
    train_time = time.perf_counter() - start
    start = time.perf_counter()
    pred = ys.inverse_transform(lr.predict(xte_s))
    infer_time = time.perf_counter() - start
    rows = metrics("Linear Regression", yte_raw, pred, spec.target_columns)
    all_rows.extend(rows)
    d = pd.DataFrame(rows)
    summary.append({"model": "Linear Regression", "mean_r2": d.r2.mean(), "min_r2": d.r2.min(), "mean_rmse": d.rmse.mean(), "train_seconds": train_time, "inference_seconds": infer_time, "parameters": "closed_form"})

    start = time.perf_counter()
    rf = MultiOutputRegressor(RandomForestRegressor(n_estimators=args.rf_estimators, max_depth=args.rf_max_depth, random_state=args.seed, n_jobs=-1))
    rf.fit(xtr_s, ytr_s)
    train_time = time.perf_counter() - start
    start = time.perf_counter()
    pred = ys.inverse_transform(rf.predict(xte_s))
    infer_time = time.perf_counter() - start
    rows = metrics("Random Forest", yte_raw, pred, spec.target_columns)
    all_rows.extend(rows)
    d = pd.DataFrame(rows)
    summary.append({"model": "Random Forest", "mean_r2": d.r2.mean(), "min_r2": d.r2.min(), "mean_rmse": d.rmse.mean(), "train_seconds": train_time, "inference_seconds": infer_time, "parameters": f"{args.rf_estimators} trees"})

    plain, pred, train_time, infer_time = train_net("Plain MLP", xtr_s, ytr_s, xte_s, ys, spec, args, 0.0)
    rows = metrics("Plain MLP", yte_raw, pred, spec.target_columns)
    all_rows.extend(rows)
    d = pd.DataFrame(rows)
    summary.append({"model": "Plain MLP", "mean_r2": d.r2.mean(), "min_r2": d.r2.min(), "mean_rmse": d.rmse.mean(), "train_seconds": train_time, "inference_seconds": infer_time, "parameters": count_parameters(plain)})

    pinn, pred, train_time, infer_time = train_net("PINN", xtr_s, ytr_s, xte_s, ys, spec, args, args.physics_weight)
    rows = metrics("PINN", yte_raw, pred, spec.target_columns)
    all_rows.extend(rows)
    d = pd.DataFrame(rows)
    summary.append({"model": "PINN", "mean_r2": d.r2.mean(), "min_r2": d.r2.min(), "mean_rmse": d.rmse.mean(), "train_seconds": train_time, "inference_seconds": infer_time, "parameters": count_parameters(pinn)})

    target_df = pd.DataFrame(all_rows)
    summary_df = pd.DataFrame(summary)

    target_df.to_csv("tables/surrogate_comparison_target_metrics.csv", index=False)
    summary_df.to_csv("tables/surrogate_comparison_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(summary_df["model"], summary_df["min_r2"])
    ax.set_ylabel("Minimum R² across outputs")
    ax.set_title("Surrogate comparison: worst-target R²")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig("figures/fig_surrogate_comparison_r2.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(summary_df["model"], summary_df["mean_rmse"])
    ax.set_ylabel("Mean RMSE across outputs")
    ax.set_title("Surrogate comparison: mean RMSE")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig("figures/fig_surrogate_comparison_rmse.png", dpi=300)
    plt.close(fig)

    print("\nSurrogate comparison summary:")
    print(summary_df.to_string(index=False))
    print("\nTarget-level metrics:")
    print(target_df.to_string(index=False))


if __name__ == "__main__":
    main()
