"""Generate plasma-channel dataset and baseline detection-bound tables."""

from __future__ import annotations

import argparse

from pinn_qi_radar.dataset import build_baseline_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fixed-ns", type=float, default=0.02)
    parser.add_argument("--fixed-m", type=float, default=1e6)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--tables-dir", type=str, default="tables")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df, summary = build_baseline_dataset(
        n_samples=args.n_samples,
        seed=args.seed,
        fixed_ns=args.fixed_ns,
        fixed_m=args.fixed_m,
        results_dir=args.results_dir,
        tables_dir=args.tables_dir,
    )
    print(f"Generated {len(df):,} plasma-channel samples")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
