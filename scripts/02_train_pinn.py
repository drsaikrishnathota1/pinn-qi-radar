#!/usr/bin/env python3
"""Train the plasma-sheath PINN surrogate.

Example:
    python3 scripts/02_train_pinn.py --n-samples 50000 --epochs 50

RunPod quick GPU test:
    python3 scripts/02_train_pinn.py --n-samples 200000 --epochs 100 --device auto
"""

from __future__ import annotations

import argparse

from pinn_qi_radar.train import TrainConfig, train_pinn_surrogate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=50000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--physics-weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = TrainConfig(
        n_samples=args.n_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        dropout=args.dropout,
        physics_weight=args.physics_weight,
        seed=args.seed,
        device=args.device,
    )

    result = train_pinn_surrogate(config)

    print(f"Device: {result['device']}")
    print(f"Training log saved to: {result['history_path']}")
    print(f"Prediction metrics saved to: {result['metrics_path']}")
    print(f"Model saved to: {result['model_path']}")
    print()
    print(result["metrics"].to_string(index=False))


if __name__ == "__main__":
    main()
