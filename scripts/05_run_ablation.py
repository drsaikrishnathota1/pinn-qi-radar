#!/usr/bin/env python3
"""Run ablation study."""

from __future__ import annotations

import argparse

from pinn_qi_radar.ablation import AblationConfig, run_ablation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=30000)
    parser.add_argument("--pinn-epochs", type=int, default=40)
    parser.add_argument("--policy-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_ablation(
        AblationConfig(
            n_samples=args.n_samples,
            pinn_epochs=args.pinn_epochs,
            policy_epochs=args.policy_epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            device=args.device,
        )
    )
    print("Ablation summary saved to: tables/ablation_summary.csv")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
