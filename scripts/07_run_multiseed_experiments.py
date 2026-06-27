#!/usr/bin/env python3
"""Run multi-seed robustness experiments.

This script is intended for RunPod final validation. It repeats key experiments
across multiple random seeds and saves combined CSV files so the paper can report
mean ± standard deviation instead of one lucky seed.

Example quick:
    python3 scripts/07_run_multiseed_experiments.py --quick

RunPod:
    python3 scripts/07_run_multiseed_experiments.py \
      --seeds 42 43 44 45 46 \
      --n-samples 100000 \
      --pinn-epochs 80 \
      --optimizer-samples 10000 \
      --policy-samples 50000 \
      --policy-epochs 60 \
      --batch-size 2048 \
      --device auto
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pinn_qi_radar.optimizer import OptimizerConfig, QIParameterBounds, optimize_dataframe, summarize_optimization_results
from pinn_qi_radar.plasma_channel import PlasmaChannelConfig, generate_plasma_dataset
from pinn_qi_radar.policy_optimizer import PolicyTrainConfig, train_qi_policy
from pinn_qi_radar.train import TrainConfig, train_pinn_surrogate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--n-samples", type=int, default=100000)
    parser.add_argument("--pinn-epochs", type=int, default=80)
    parser.add_argument("--optimizer-samples", type=int, default=10000)
    parser.add_argument("--policy-samples", type=int, default=50000)
    parser.add_argument("--policy-epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Path("results").mkdir(exist_ok=True)
    Path("tables").mkdir(exist_ok=True)

    if args.quick:
        seeds = [42, 43]
        n_samples = 4000
        pinn_epochs = 4
        optimizer_samples = 800
        policy_samples = 2000
        policy_epochs = 4
        batch_size = 512
    else:
        seeds = args.seeds
        n_samples = args.n_samples
        pinn_epochs = args.pinn_epochs
        optimizer_samples = args.optimizer_samples
        policy_samples = args.policy_samples
        policy_epochs = args.policy_epochs
        batch_size = args.batch_size

    all_pinn = []
    all_opt = []
    all_policy = []

    for seed in seeds:
        print(f"\n===== MULTI-SEED RUN: seed={seed} =====")

        pinn = train_pinn_surrogate(
            TrainConfig(
                n_samples=n_samples,
                epochs=pinn_epochs,
                batch_size=batch_size,
                seed=seed,
                device=args.device,
            )
        )
        pinn_metrics = pinn["metrics"].copy()
        pinn_metrics["seed"] = seed
        all_pinn.append(pinn_metrics)

        opt_df = generate_plasma_dataset(PlasmaChannelConfig(n_samples=optimizer_samples, seed=seed))
        opt_results = optimize_dataframe(
            opt_df,
            bounds=QIParameterBounds(),
            config=OptimizerConfig(seed=seed),
        )
        opt_summary = summarize_optimization_results(opt_results)
        opt_summary["seed"] = seed
        all_opt.append(opt_summary)

        policy = train_qi_policy(
            PolicyTrainConfig(
                n_samples=policy_samples,
                epochs=policy_epochs,
                batch_size=batch_size,
                seed=seed,
                device=args.device,
            )
        )
        policy_metrics = policy["metrics"].copy()
        policy_metrics["seed"] = seed
        all_policy.append(policy_metrics)

    pinn_all = pd.concat(all_pinn, ignore_index=True)
    opt_all = pd.concat(all_opt, ignore_index=True)
    policy_all = pd.concat(all_policy, ignore_index=True)

    pinn_all.to_csv("results/multiseed_pinn_metrics_raw.csv", index=False)
    opt_all.to_csv("results/multiseed_optimizer_summary_raw.csv", index=False)
    policy_all.to_csv("results/multiseed_policy_metrics_raw.csv", index=False)

    pinn_agg = (
        pinn_all.groupby("target")
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
        )
        .reset_index()
    )

    opt_agg = (
        opt_all.groupby(["plasma_severity", "strategy"])
        .agg(
            qi_pe_mean=("qi_pe_mean", "mean"),
            qi_pe_std=("qi_pe_mean", "std"),
            gain_db_mean=("gain_db_vs_fixed_qi_mean", "mean"),
            gain_db_std=("gain_db_vs_fixed_qi_mean", "std"),
            runtime_ms_mean=("runtime_ms_mean", "mean"),
            runtime_ms_std=("runtime_ms_mean", "std"),
        )
        .reset_index()
    )

    policy_agg = (
        policy_all.agg(["mean", "std"])
        .reset_index()
        .rename(columns={"index": "statistic"})
    )

    pinn_agg.to_csv("tables/multiseed_pinn_metrics_mean_std.csv", index=False)
    opt_agg.to_csv("tables/multiseed_optimizer_mean_std.csv", index=False)
    policy_agg.to_csv("tables/multiseed_policy_mean_std.csv", index=False)

    print("\nMulti-seed results saved:")
    print(" - tables/multiseed_pinn_metrics_mean_std.csv")
    print(" - tables/multiseed_optimizer_mean_std.csv")
    print(" - tables/multiseed_policy_mean_std.csv")


if __name__ == "__main__":
    main()
