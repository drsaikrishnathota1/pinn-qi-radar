#!/usr/bin/env python3
"""Final ablation study for PINN-QI Radar.

This experiment answers the reviewer question:
"Which component actually contributes to the improvement?"

It compares:
 - fixed_qi
 - random_search_qi
 - grid_search_qi
 - adaptive_pinn_qi
 - neural_policy_qi

Outputs:
 - tables/final_ablation_summary.csv
 - tables/final_ablation_policy_metrics.csv
 - figures/fig_final_ablation_pe.png
 - figures/fig_final_ablation_runtime.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pinn_qi_radar.optimizer import (
    OptimizerConfig,
    QIParameterBounds,
    optimize_dataframe,
    summarize_optimization_results,
)
from pinn_qi_radar.plasma_channel import PlasmaChannelConfig, generate_plasma_dataset
from pinn_qi_radar.policy_optimizer import PolicyTrainConfig, train_qi_policy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--n-samples", type=int, default=50000)
    p.add_argument("--policy-samples", type=int, default=100000)
    p.add_argument("--policy-epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--grid-size", type=int, default=35)
    p.add_argument("--random-candidates", type=int, default=128)
    p.add_argument("--adaptive-candidates", type=int, default=24)
    return p.parse_args()


def plot_grouped_bar(df, value_col, ylabel, title, out_path):
    severity_order = ["low", "medium", "high"]
    strategy_order = [
        "fixed_qi",
        "random_search_qi",
        "grid_search_qi",
        "adaptive_pinn_qi",
        "neural_policy_qi",
    ]

    available_sev = [s for s in severity_order if s in set(df["plasma_severity"])]
    available_strat = [s for s in strategy_order if s in set(df["strategy"])]

    pivot = (
        df.pivot_table(
            index="plasma_severity",
            columns="strategy",
            values=value_col,
            aggfunc="mean",
        )
        .reindex(index=available_sev, columns=available_strat)
    )

    x = np.arange(len(pivot.index))
    width = 0.8 / max(len(pivot.columns), 1)

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for i, col in enumerate(pivot.columns):
        ax.bar(x + i * width - 0.4 + width / 2, pivot[col].to_numpy(), width, label=col)

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def add_neural_policy_to_ablation(summary_df, policy_metrics_df):
    """Add policy-level row to ablation summary.

    The neural policy metrics are global over its held-out test split, not severity-grouped.
    To keep the ablation table simple, we add it under plasma_severity='overall'.
    """
    m = policy_metrics_df.iloc[0].to_dict()

    row = {
        "plasma_severity": "overall",
        "strategy": "neural_policy_qi",
        "samples": np.nan,
        "qi_pe_mean": float(m["policy_pe_mean"]),
        "qi_pe_median": np.nan,
        "gain_db_vs_fixed_qi_mean": float(m["policy_gain_db_vs_fixed_mean"]),
        "runtime_ms_mean": np.nan,
        "resource_cost_mean": np.nan,
        "ns_mean": np.nan,
        "m_mean": np.nan,
        "oracle_pe_mean": float(m["oracle_pe_mean"]),
        "fixed_pe_mean": float(m["fixed_pe_mean"]),
        "oracle_gap_percent": float(m["oracle_gap_percent"]),
        "budget_violation_rate": float(m["budget_violation_rate"]),
    }

    for col in row:
        if col not in summary_df.columns:
            summary_df[col] = np.nan

    return pd.concat([summary_df, pd.DataFrame([row])], ignore_index=True)


def main():
    args = parse_args()

    if args.quick:
        args.n_samples = 2000
        args.policy_samples = 3000
        args.policy_epochs = 3
        args.batch_size = 512
        args.grid_size = 15
        args.random_candidates = 32
        args.adaptive_candidates = 12

    Path("tables").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    bounds = QIParameterBounds()
    opt_config = OptimizerConfig(
        random_candidates=args.random_candidates,
        grid_size=args.grid_size,
        adaptive_candidates=args.adaptive_candidates,
        seed=args.seed,
    )

    print("Generating ablation plasma dataset...")
    df = generate_plasma_dataset(PlasmaChannelConfig(n_samples=args.n_samples, seed=args.seed))

    print("Running fixed/random/grid/adaptive ablation...")
    start = time.perf_counter()
    opt_results = optimize_dataframe(df, bounds=bounds, config=opt_config)
    opt_seconds = time.perf_counter() - start

    opt_results.to_csv("results/final_ablation_raw_optimizer_results.csv", index=False)
    opt_summary = summarize_optimization_results(opt_results)

    print("Training/evaluating neural policy for ablation...")
    policy_result = train_qi_policy(
        PolicyTrainConfig(
            n_samples=args.policy_samples,
            epochs=args.policy_epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            device=args.device,
            grid_size=args.grid_size,
            budget=bounds.budget,
        )
    )
    policy_metrics = policy_result["metrics"]
    policy_metrics.to_csv("tables/final_ablation_policy_metrics.csv", index=False)

    final_summary = add_neural_policy_to_ablation(opt_summary, policy_metrics)
    final_summary["ablation_optimizer_seconds"] = opt_seconds
    final_summary.to_csv("tables/final_ablation_summary.csv", index=False)

    # Figures are for severity-specific optimizer strategies only.
    plot_grouped_bar(
        opt_summary,
        "qi_pe_mean",
        "Mean QI error bound Pe",
        "Final ablation: error-bound comparison",
        "figures/fig_final_ablation_pe.png",
    )

    plot_grouped_bar(
        opt_summary,
        "runtime_ms_mean",
        "Mean runtime per sample (ms)",
        "Final ablation: runtime comparison",
        "figures/fig_final_ablation_runtime.png",
    )

    print("\nFinal ablation summary:")
    print(final_summary.to_string(index=False))

    print("\nSaved:")
    print(" - tables/final_ablation_summary.csv")
    print(" - tables/final_ablation_policy_metrics.csv")
    print(" - results/final_ablation_raw_optimizer_results.csv")
    print(" - figures/fig_final_ablation_pe.png")
    print(" - figures/fig_final_ablation_runtime.png")


if __name__ == "__main__":
    main()
