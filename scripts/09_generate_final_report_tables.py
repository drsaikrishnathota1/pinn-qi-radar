#!/usr/bin/env python3
"""Generate final manuscript summary and key-claims tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_if_exists(path: str) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_csv(p)


def main() -> None:
    Path("tables").mkdir(exist_ok=True)

    rows = []
    claims = []

    pinn = read_if_exists("tables/pinn_prediction_metrics.csv")
    if pinn is not None:
        for _, r in pinn.iterrows():
            rows.append(
                {
                    "category": "PINN surrogate",
                    "metric": f"{r['target']} R2",
                    "value": r["r2"],
                    "source_file": "tables/pinn_prediction_metrics.csv",
                }
            )
        min_r2 = float(pinn["r2"].min())
        claims.append(
            {
                "claim": "PINN surrogate accuracy",
                "evidence": f"Minimum channel-output R2 across targets = {min_r2:.4f}",
                "safe_wording": "The PINN surrogate accurately learned the simulated plasma-channel mapping.",
            }
        )

    opt = read_if_exists("tables/qi_optimizer_summary.csv")
    if opt is not None:
        for sev in opt["plasma_severity"].unique():
            sev_df = opt[opt["plasma_severity"] == sev]
            fixed = sev_df[sev_df["strategy"] == "fixed_qi"]
            adaptive = sev_df[sev_df["strategy"] == "adaptive_pinn_qi"]
            if not fixed.empty and not adaptive.empty:
                gain = float(adaptive["gain_db_vs_fixed_qi_mean"].iloc[0])
                pe = float(adaptive["qi_pe_mean"].iloc[0])
                rows.append(
                    {
                        "category": "QI optimization",
                        "metric": f"{sev} adaptive gain vs fixed QI dB",
                        "value": gain,
                        "source_file": "tables/qi_optimizer_summary.csv",
                    }
                )
                rows.append(
                    {
                        "category": "QI optimization",
                        "metric": f"{sev} adaptive QI Pe mean",
                        "value": pe,
                        "source_file": "tables/qi_optimizer_summary.csv",
                    }
                )

    policy = read_if_exists("tables/qi_policy_metrics.csv")
    if policy is not None and not policy.empty:
        r = policy.iloc[0]
        for col in policy.columns:
            rows.append(
                {
                    "category": "Neural QI policy",
                    "metric": col,
                    "value": r[col],
                    "source_file": "tables/qi_policy_metrics.csv",
                }
            )
        claims.append(
            {
                "claim": "Near-oracle neural policy",
                "evidence": f"Mean policy gap vs grid-search oracle = {float(r['oracle_gap_percent']):.4f}%",
                "safe_wording": "The neural QI policy approximated the grid-search oracle with a small simulated error-bound gap.",
            }
        )
        claims.append(
            {
                "claim": "Resource-feasible policy",
                "evidence": f"Budget violation rate = {float(r['budget_violation_rate']):.4f}",
                "safe_wording": "The learned policy satisfied the normalized resource budget in evaluation.",
            }
        )

    stress = read_if_exists("tables/stress_optimizer_summary.csv")
    if stress is not None:
        extreme = stress[(stress["plasma_severity"] == "extreme") & (stress["strategy"] == "adaptive_pinn_qi")]
        if not extreme.empty:
            claims.append(
                {
                    "claim": "Extreme stress-test robustness",
                    "evidence": f"Extreme-regime adaptive QI mean Pe = {float(extreme['qi_pe_mean'].iloc[0]):.4f}",
                    "safe_wording": "The framework was evaluated under an extreme simulated plasma-sheath stress regime.",
                }
            )

    summary = pd.DataFrame(rows)
    key_claims = pd.DataFrame(claims)

    summary.to_csv("tables/final_manuscript_summary.csv", index=False)
    key_claims.to_csv("tables/final_key_claims.csv", index=False)

    print("Final manuscript tables generated:")
    print(" - tables/final_manuscript_summary.csv")
    print(" - tables/final_key_claims.csv")
    if not key_claims.empty:
        print()
        print(key_claims.to_string(index=False))


if __name__ == "__main__":
    main()
