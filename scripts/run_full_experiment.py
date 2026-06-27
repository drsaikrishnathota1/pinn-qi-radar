#!/usr/bin/env python3
"""One-command RunPod experiment pipeline.

This script runs the complete computational pipeline:
1. plasma + QI baseline bounds
2. PINN plasma-channel surrogate training
3. resource-constrained QI optimizer comparison
4. neural QI policy training
5. optional stress and multi-seed robustness
6. figure and final table generation

Local quick run:
    python3 scripts/run_full_experiment.py --quick

RunPod strong run:
    python3 scripts/run_full_experiment.py --n-samples 200000 --pinn-epochs 100 --optimizer-samples 20000 --policy-samples 100000 --policy-epochs 80 --batch-size 2048 --device auto --stress-test --final-tables
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print()
    print("=" * 90)
    print("RUNNING:", " ".join(cmd))
    print("=" * 90)
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--n-samples", type=int, default=50000)
    parser.add_argument("--pinn-epochs", type=int, default=50)
    parser.add_argument("--optimizer-samples", type=int, default=5000)
    parser.add_argument("--policy-samples", type=int, default=30000)
    parser.add_argument("--policy-epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--stress-test", action="store_true")
    parser.add_argument("--stress-samples", type=int, default=25000)
    parser.add_argument("--final-tables", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Path("results").mkdir(exist_ok=True)
    Path("tables").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    if args.quick:
        n_samples = 5000
        pinn_epochs = 5
        optimizer_samples = 1000
        policy_samples = 3000
        policy_epochs = 5
        batch_size = 512
        stress_samples = 1000
    else:
        n_samples = args.n_samples
        pinn_epochs = args.pinn_epochs
        optimizer_samples = args.optimizer_samples
        policy_samples = args.policy_samples
        policy_epochs = args.policy_epochs
        batch_size = args.batch_size
        stress_samples = args.stress_samples

    py = sys.executable

    run([py, "scripts/01_generate_baseline_bounds.py", "--n-samples", str(n_samples)])
    run([py, "scripts/02_train_pinn.py", "--n-samples", str(n_samples), "--epochs", str(pinn_epochs), "--batch-size", str(batch_size), "--device", args.device])
    run([py, "scripts/03_optimize_qi.py", "--n-samples", str(optimizer_samples)])
    run([py, "scripts/04_train_qi_policy.py", "--n-samples", str(policy_samples), "--epochs", str(policy_epochs), "--batch-size", str(batch_size), "--device", args.device])

    if args.stress_test or args.quick:
        run([py, "scripts/08_run_stress_test.py", "--n-samples", str(stress_samples)])

    run([py, "scripts/06_generate_figures.py"])

    if args.final_tables or args.quick:
        run([py, "scripts/09_generate_final_report_tables.py"])

    print()
    print("Full experiment completed.")
    print("Key outputs:")
    print(" - tables/baseline_bounds_summary.csv")
    print(" - tables/pinn_prediction_metrics.csv")
    print(" - tables/qi_optimizer_summary.csv")
    print(" - tables/qi_policy_metrics.csv")
    print(" - tables/stress_baseline_summary.csv")
    print(" - tables/stress_optimizer_summary.csv")
    print(" - tables/final_manuscript_summary.csv")
    print(" - tables/final_key_claims.csv")
    print(" - figures/*.png")


if __name__ == "__main__":
    main()
