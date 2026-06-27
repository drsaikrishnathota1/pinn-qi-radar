#!/usr/bin/env python3
"""Train neural QI policy optimizer.

Local quick test:
    python3 scripts/04_train_qi_policy.py --n-samples 5000 --epochs 5 --grid-size 20

RunPod stronger run:
    python3 scripts/04_train_qi_policy.py --n-samples 100000 --epochs 80 --grid-size 45 --batch-size 2048 --device auto
"""

from __future__ import annotations

import argparse

from pinn_qi_radar.policy_optimizer import PolicyTrainConfig, train_qi_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=30000)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.02)
    parser.add_argument("--grid-size", type=int, default=35)
    parser.add_argument("--budget", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_qi_policy(
        PolicyTrainConfig(
            n_samples=args.n_samples,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            hidden_dim=args.hidden_dim,
            depth=args.depth,
            dropout=args.dropout,
            grid_size=args.grid_size,
            budget=args.budget,
            seed=args.seed,
            device=args.device,
        )
    )
    print(f"Device: {result['device']}")
    print(f"Training log saved to: {result['history_path']}")
    print(f"Predictions saved to: {result['prediction_path']}")
    print(f"Metrics saved to: {result['metrics_path']}")
    print(f"Model saved to: {result['model_path']}")
    print()
    print(result["metrics"].to_string(index=False))


if __name__ == "__main__":
    main()
