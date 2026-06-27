"""Dataset generation helpers for PINN-QI Radar."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .plasma_channel import PlasmaChannelConfig, generate_plasma_dataset
from .qi_bounds import add_baseline_bounds, summarize_bounds


def build_baseline_dataset(
    n_samples: int = 50_000,
    seed: int = 42,
    fixed_ns: float = 0.02,
    fixed_m: float = 1e6,
    results_dir: str | Path = "results",
    tables_dir: str | Path = "tables",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate plasma-channel data and baseline QI/classical bounds."""
    results_path = Path(results_dir)
    tables_path = Path(tables_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    tables_path.mkdir(parents=True, exist_ok=True)

    df = generate_plasma_dataset(PlasmaChannelConfig(n_samples=n_samples, seed=seed))
    df = add_baseline_bounds(df, fixed_ns=fixed_ns, fixed_m=fixed_m)
    summary = summarize_bounds(df)

    df.to_csv(results_path / "plasma_qi_baseline_dataset.csv", index=False)
    summary.to_csv(tables_path / "baseline_bounds_summary.csv", index=False)
    return df, summary
