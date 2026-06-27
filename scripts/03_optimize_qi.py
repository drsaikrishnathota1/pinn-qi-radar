#!/usr/bin/env python3
"""Run QI parameter optimization baselines.

Example:
    python3 scripts/03_optimize_qi.py --n-samples 3000

RunPod larger experiment:
    python3 scripts/03_optimize_qi.py --n-samples 20000 --random-candidates 256 --grid-size 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pinn_qi_radar.plasma_channel import PlasmaChannelConfig, generate_plasma_dataset
from pinn_qi_radar.optimizer import (
    OptimizerConfig,
    QIParameterBounds,
    optimize_dataframe,
    summarize_optimization_results,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-candidates", type=int, default=128)
    parser.add_argument("--grid-size", type=int, default=35)
    parser.add_argument("--adaptive-candidates", type=int, default=24)
    parser.add_argument("--budget", type=float, default=0.65)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results_dir = Path("results")
    tables_dir = Path("tables")
    results_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    df = generate_plasma_dataset(
        PlasmaChannelConfig(n_samples=args.n_samples, seed=args.seed)
    )

    bounds = QIParameterBounds(budget=args.budget)
    config = OptimizerConfig(
        random_candidates=args.random_candidates,
        grid_size=args.grid_size,
        adaptive_candidates=args.adaptive_candidates,
        seed=args.seed,
    )

    optimized = optimize_dataframe(df, bounds=bounds, config=config)
    summary = summarize_optimization_results(optimized)

    results_path = results_dir / "qi_optimizer_results.csv"
    summary_path = tables_dir / "qi_optimizer_summary.csv"

    optimized.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"Generated optimization results for {args.n_samples:,} plasma-channel samples")
    print(f"Detailed results saved to: {results_path}")
    print(f"Summary table saved to: {summary_path}")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
