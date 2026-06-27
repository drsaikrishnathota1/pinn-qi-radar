#!/usr/bin/env python3
"""Run very-low to extreme plasma stress-test analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from pinn_qi_radar.optimizer import (
    OptimizerConfig,
    QIParameterBounds,
    optimize_dataframe,
    summarize_optimization_results,
)
from pinn_qi_radar.qi_bounds import add_baseline_bounds, summarize_bounds
from pinn_qi_radar.stress_plasma import StressPlasmaConfig, generate_stress_plasma_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=25000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    Path("results").mkdir(exist_ok=True)
    Path("tables").mkdir(exist_ok=True)

    df = generate_stress_plasma_dataset(
        StressPlasmaConfig(n_samples=args.n_samples, seed=args.seed)
    )

    df_with_bounds = add_baseline_bounds(df)
    baseline_summary = summarize_bounds(df_with_bounds)

    optimized = optimize_dataframe(
        df,
        bounds=QIParameterBounds(),
        config=OptimizerConfig(seed=args.seed),
    )
    opt_summary = summarize_optimization_results(optimized)

    df_with_bounds.to_csv("results/stress_plasma_dataset.csv", index=False)
    baseline_summary.to_csv("tables/stress_baseline_summary.csv", index=False)
    opt_summary.to_csv("tables/stress_optimizer_summary.csv", index=False)

    print("Stress-test baseline summary:")
    print(baseline_summary.to_string(index=False))
    print()
    print("Stress-test optimizer summary:")
    print(opt_summary.to_string(index=False))
    print()
    print("Saved:")
    print(" - tables/stress_baseline_summary.csv")
    print(" - tables/stress_optimizer_summary.csv")


if __name__ == "__main__":
    main()
