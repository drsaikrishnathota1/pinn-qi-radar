"""Manuscript figure generation for PINN-QI Radar."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def _ensure_figures_dir() -> Path:
    path = Path("figures")
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_pinn_training_loss(log_path: str = "results/pinn_training_log.csv") -> str | None:
    path = Path(log_path)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    fig_dir = _ensure_figures_dir()
    out = fig_dir / "fig_pinn_training_loss.png"

    plt.figure()
    plt.plot(df["epoch"], df["loss"], label="total loss")
    if "data_loss" in df:
        plt.plot(df["epoch"], df["data_loss"], label="data loss")
    if "physics_loss" in df:
        plt.plot(df["epoch"], df["physics_loss"], label="physics loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("PINN plasma-channel surrogate training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    return str(out)


def plot_optimizer_comparison(summary_path: str = "tables/qi_optimizer_summary.csv") -> str | None:
    path = Path(summary_path)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    fig_dir = _ensure_figures_dir()
    out = fig_dir / "fig_qi_optimizer_comparison.png"

    pivot = df.pivot(index="plasma_severity", columns="strategy", values="qi_pe_mean")
    pivot = pivot.reindex(["low", "medium", "high"])
    pivot.plot(kind="bar")
    plt.xlabel("Plasma severity")
    plt.ylabel("Mean QI error bound")
    plt.title("Resource-constrained QI optimizer comparison")
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    return str(out)


def plot_runtime_comparison(summary_path: str = "tables/qi_optimizer_summary.csv") -> str | None:
    path = Path(summary_path)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    fig_dir = _ensure_figures_dir()
    out = fig_dir / "fig_optimizer_runtime.png"

    pivot = df.pivot(index="plasma_severity", columns="strategy", values="runtime_ms_mean")
    pivot = pivot.reindex(["low", "medium", "high"])
    pivot.plot(kind="bar")
    plt.xlabel("Plasma severity")
    plt.ylabel("Runtime per sample (ms)")
    plt.title("Optimizer runtime comparison")
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    return str(out)


def plot_policy_gain(pred_path: str = "results/qi_policy_predictions.csv") -> str | None:
    path = Path(pred_path)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    fig_dir = _ensure_figures_dir()
    out = fig_dir / "fig_neural_policy_gain.png"

    grouped = df.groupby("plasma_severity")["policy_gain_db_vs_fixed"].mean().reindex(["low", "medium", "high"])
    grouped.plot(kind="bar")
    plt.xlabel("Plasma severity")
    plt.ylabel("Mean gain vs fixed QI (dB)")
    plt.title("Neural QI policy gain under plasma severity")
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    return str(out)


def generate_all_figures() -> list[str]:
    outputs = [
        plot_pinn_training_loss(),
        plot_optimizer_comparison(),
        plot_runtime_comparison(),
        plot_policy_gain(),
    ]
    return [x for x in outputs if x is not None]
