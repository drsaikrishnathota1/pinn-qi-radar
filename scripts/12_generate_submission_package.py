#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import pandas as pd


def read_csv(path):
    p = Path(path)
    if not p.exists():
        print(f"Missing optional file: {path}")
        return None
    return pd.read_csv(p)


def main():
    Path("tables").mkdir(exist_ok=True)

    rows = []

    pinn = read_csv("tables/pinn_prediction_metrics.csv")
    if pinn is not None and "r2" in pinn.columns:
        rows.append({
            "claim": "PINN surrogate minimum R2",
            "value": float(pinn["r2"].min()),
            "source_file": "tables/pinn_prediction_metrics.csv",
        })

    policy = read_csv("tables/qi_policy_metrics.csv")
    if policy is not None and len(policy) > 0:
        m = policy.iloc[0]
        for col in [
            "policy_pe_mean",
            "oracle_pe_mean",
            "fixed_pe_mean",
            "policy_gain_db_vs_fixed_mean",
            "oracle_gap_percent",
            "budget_violation_rate",
        ]:
            if col in policy.columns:
                rows.append({
                    "claim": col,
                    "value": float(m[col]),
                    "source_file": "tables/qi_policy_metrics.csv",
                })

    ablation = read_csv("tables/final_ablation_summary.csv")
    if ablation is not None:
        adaptive = ablation[ablation["strategy"] == "adaptive_pinn_qi"]
        if len(adaptive) > 0:
            rows.append({
                "claim": "adaptive_pinn_qi_mean_pe",
                "value": float(adaptive["qi_pe_mean"].mean()),
                "source_file": "tables/final_ablation_summary.csv",
            })

        neural = ablation[ablation["strategy"] == "neural_policy_qi"]
        if len(neural) > 0:
            rows.append({
                "claim": "neural_policy_qi_overall_pe",
                "value": float(neural.iloc[0]["qi_pe_mean"]),
                "source_file": "tables/final_ablation_summary.csv",
            })

    surrogate = read_csv("tables/surrogate_comparison_summary.csv")
    if surrogate is not None:
        for _, r in surrogate.iterrows():
            rows.append({
                "claim": f"surrogate_{r['model']}_min_r2",
                "value": float(r["min_r2"]),
                "source_file": "tables/surrogate_comparison_summary.csv",
            })

    pd.DataFrame(rows).to_csv("tables/submission_key_results.csv", index=False)

    reproducibility = pd.DataFrame([
        {"item": "repository", "value": "https://github.com/drsaikrishnathota1/pinn-qi-radar"},
        {"item": "main_samples", "value": "200000"},
        {"item": "pinn_epochs_main", "value": "100"},
        {"item": "policy_samples_main", "value": "100000"},
        {"item": "policy_epochs_main", "value": "80"},
        {"item": "multiseed_runs", "value": "5"},
        {"item": "seeds", "value": "42,43,44,45,46"},
        {"item": "stress_test_samples", "value": "50000"},
        {"item": "surrogate_comparison_samples_target", "value": "200000"},
        {"item": "ablation_samples_target", "value": "50000"},
        {"item": "ablation_policy_samples_target", "value": "100000"},
        {"item": "scope", "value": "Simulation-level plasma-channel and QI error-bound analysis only"},
    ])
    reproducibility.to_csv("tables/reproducibility_configuration.csv", index=False)

    figures = [
        ("fig_pinn_training_loss.png", "PINN training convergence"),
        ("fig_surrogate_comparison_r2.png", "Surrogate baseline comparison"),
        ("fig_qi_optimizer_comparison.png", "QI optimizer comparison"),
        ("fig_optimizer_runtime.png", "Optimizer runtime comparison"),
        ("fig_neural_policy_gain.png", "Neural policy gain"),
        ("fig_stress_severity_comparison.png", "Stress-test severity comparison"),
        ("fig_multiseed_policy_robustness.png", "Multi-seed robustness"),
        ("fig_final_ablation_pe.png", "Final ablation Pe comparison"),
        ("fig_final_ablation_runtime.png", "Final ablation runtime comparison"),
    ]

    pd.DataFrame([
        {
            "figure_file": f,
            "recommended_caption": c,
            "exists": (Path("figures") / f).exists(),
        }
        for f, c in figures
    ]).to_csv("tables/final_figure_index.csv", index=False)

    print("Saved:")
    print(" - tables/submission_key_results.csv")
    print(" - tables/reproducibility_configuration.csv")
    print(" - tables/final_figure_index.csv")


if __name__ == "__main__":
    main()
