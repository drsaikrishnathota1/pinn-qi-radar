"""Quantum-illumination parameter optimization utilities.

This module compares four strategies:

1. fixed QI operating point
2. random-search QI under a normalized resource budget
3. grid-search QI under the same resource budget
4. adaptive surrogate-guided QI search under the same resource budget

The optimizer minimizes the simulated quantum-illumination error bound while
respecting a compact resource budget over signal photon number Ns and mode-pair
count M. This prevents the trivial solution of always choosing the largest
possible Ns and M.

The implementation is simulation-level and intended for computational analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np
import pandas as pd

from .qi_bounds import db_gain, quantum_illumination_error_bound


@dataclass(frozen=True)
class QIParameterBounds:
    """Search bounds and budget for QI operating parameters."""

    ns_min: float = 0.005
    ns_max: float = 0.08
    m_min: float = 2.0e5
    m_max: float = 5.0e6
    budget: float = 0.65
    ns_weight: float = 0.55
    m_weight: float = 0.45


@dataclass(frozen=True)
class OptimizerConfig:
    """Configuration for baseline and proposed optimizers."""

    fixed_ns: float = 0.02
    fixed_m: float = 1.0e6
    random_candidates: int = 128
    grid_size: int = 35
    adaptive_candidates: int = 24
    seed: int = 42


def normalized_resource_cost(
    ns: np.ndarray | float,
    m: np.ndarray | float,
    bounds: QIParameterBounds | None = None,
) -> np.ndarray:
    """Compute normalized resource cost for Ns and M."""
    b = bounds or QIParameterBounds()
    ns_arr = np.asarray(ns, dtype=float)
    m_arr = np.asarray(m, dtype=float)

    ns_norm = (ns_arr - b.ns_min) / max(b.ns_max - b.ns_min, 1e-12)
    m_norm = (m_arr - b.m_min) / max(b.m_max - b.m_min, 1e-12)
    cost = b.ns_weight * ns_norm + b.m_weight * m_norm
    return np.clip(cost, 0.0, 10.0)


def is_budget_feasible(
    ns: np.ndarray | float,
    m: np.ndarray | float,
    bounds: QIParameterBounds | None = None,
) -> np.ndarray:
    """Return whether candidate parameters satisfy the normalized budget."""
    b = bounds or QIParameterBounds()
    return normalized_resource_cost(ns, m, b) <= b.budget + 1e-12


def _best_candidate(
    ns_candidates: np.ndarray,
    m_candidates: np.ndarray,
    kappa: float,
    nb: float,
) -> tuple[float, float, float]:
    """Return Ns, M, and Pe for the best candidate set."""
    pe = quantum_illumination_error_bound(ns_candidates, m_candidates, kappa, nb)
    idx = int(np.argmin(pe))
    return float(ns_candidates[idx]), float(m_candidates[idx]), float(pe[idx])


def fixed_qi_strategy(
    kappa: float,
    nb: float,
    bounds: QIParameterBounds | None = None,
    config: OptimizerConfig | None = None,
) -> tuple[float, float, float]:
    """Evaluate fixed QI parameters."""
    b = bounds or QIParameterBounds()
    c = config or OptimizerConfig()
    ns = float(np.clip(c.fixed_ns, b.ns_min, b.ns_max))
    m = float(np.clip(c.fixed_m, b.m_min, b.m_max))
    if not bool(is_budget_feasible(ns, m, b)):
        # Project to a conservative feasible operating point.
        ns = b.ns_min + 0.35 * (b.ns_max - b.ns_min)
        m = b.m_min + 0.35 * (b.m_max - b.m_min)
    pe = float(quantum_illumination_error_bound(ns, m, kappa, nb))
    return ns, m, pe


def random_search_strategy(
    kappa: float,
    nb: float,
    rng: np.random.Generator,
    bounds: QIParameterBounds | None = None,
    config: OptimizerConfig | None = None,
) -> tuple[float, float, float]:
    """Random-search optimizer under a resource budget."""
    b = bounds or QIParameterBounds()
    c = config or OptimizerConfig()

    ns = rng.uniform(b.ns_min, b.ns_max, c.random_candidates * 4)
    m = rng.uniform(b.m_min, b.m_max, c.random_candidates * 4)
    feasible = is_budget_feasible(ns, m, b)
    ns = ns[feasible][: c.random_candidates]
    m = m[feasible][: c.random_candidates]

    if len(ns) == 0:
        ns = np.array([b.ns_min])
        m = np.array([b.m_min])

    return _best_candidate(ns, m, kappa, nb)


def grid_search_strategy(
    kappa: float,
    nb: float,
    bounds: QIParameterBounds | None = None,
    config: OptimizerConfig | None = None,
) -> tuple[float, float, float]:
    """Grid-search optimizer under a resource budget."""
    b = bounds or QIParameterBounds()
    c = config or OptimizerConfig()

    ns_grid = np.linspace(b.ns_min, b.ns_max, c.grid_size)
    m_grid = np.linspace(b.m_min, b.m_max, c.grid_size)
    ns_mesh, m_mesh = np.meshgrid(ns_grid, m_grid, indexing="ij")
    ns_flat = ns_mesh.ravel()
    m_flat = m_mesh.ravel()
    feasible = is_budget_feasible(ns_flat, m_flat, b)

    return _best_candidate(ns_flat[feasible], m_flat[feasible], kappa, nb)


def adaptive_surrogate_guided_strategy(
    kappa: float,
    nb: float,
    bounds: QIParameterBounds | None = None,
    config: OptimizerConfig | None = None,
) -> tuple[float, float, float]:
    """Fast adaptive QI optimizer under a resource budget.

    The QI exponent is proportional to M * Ns * kappa / (NB + 1). For a fixed
    channel, reducing Pe means increasing M * Ns while respecting the resource
    budget. This strategy searches a compact set of candidates on and near the
    feasible budget frontier, so it is much faster than dense grid search while
    remaining deterministic and channel-aware.
    """
    b = bounds or QIParameterBounds()
    c = config or OptimizerConfig()

    # Candidate Ns ratios are denser near the high-resource frontier.
    ratios = np.linspace(0.05, b.budget, c.adaptive_candidates)
    ratios = np.unique(np.concatenate([ratios, np.linspace(0.45, b.budget, 10)]))

    ns_candidates = []
    m_candidates = []

    for ns_cost_fraction in ratios:
        # Split total budget between Ns and M.
        m_cost_fraction = b.budget - ns_cost_fraction
        ns_norm = ns_cost_fraction / max(b.ns_weight, 1e-12)
        m_norm = m_cost_fraction / max(b.m_weight, 1e-12)

        if ns_norm < 0 or m_norm < 0:
            continue

        ns = b.ns_min + np.clip(ns_norm, 0.0, 1.0) * (b.ns_max - b.ns_min)
        m = b.m_min + np.clip(m_norm, 0.0, 1.0) * (b.m_max - b.m_min)

        ns_candidates.append(ns)
        m_candidates.append(m)

    # Add a few conservative fallback points.
    for r in (0.25, 0.4, 0.55, 0.65):
        ns_candidates.append(b.ns_min + r * (b.ns_max - b.ns_min))
        m_candidates.append(b.m_min + r * (b.m_max - b.m_min))

    ns_arr = np.asarray(ns_candidates, dtype=float)
    m_arr = np.asarray(m_candidates, dtype=float)
    feasible = is_budget_feasible(ns_arr, m_arr, b)

    return _best_candidate(ns_arr[feasible], m_arr[feasible], kappa, nb)


def _time_strategy(
    strategy_fn: Callable[[], tuple[float, float, float]]
) -> tuple[float, float, float, float]:
    start = perf_counter()
    ns, m, pe = strategy_fn()
    elapsed_ms = (perf_counter() - start) * 1000.0
    return ns, m, pe, elapsed_ms


def optimize_dataframe(
    df: pd.DataFrame,
    bounds: QIParameterBounds | None = None,
    config: OptimizerConfig | None = None,
) -> pd.DataFrame:
    """Run all QI optimization strategies on a plasma-channel dataframe."""
    b = bounds or QIParameterBounds()
    c = config or OptimizerConfig()
    rng = np.random.default_rng(c.seed)

    rows: list[dict[str, float | str]] = []

    for idx, row in df.reset_index(drop=True).iterrows():
        kappa = float(row["effective_reflectivity_kappa"])
        nb = float(row["effective_noise_nb"])

        fixed_ns, fixed_m, fixed_pe, fixed_ms = _time_strategy(
            lambda: fixed_qi_strategy(kappa, nb, b, c)
        )
        rand_ns, rand_m, rand_pe, rand_ms = _time_strategy(
            lambda: random_search_strategy(kappa, nb, rng, b, c)
        )
        grid_ns, grid_m, grid_pe, grid_ms = _time_strategy(
            lambda: grid_search_strategy(kappa, nb, b, c)
        )
        adapt_ns, adapt_m, adapt_pe, adapt_ms = _time_strategy(
            lambda: adaptive_surrogate_guided_strategy(kappa, nb, b, c)
        )

        strategy_data = [
            ("fixed_qi", fixed_ns, fixed_m, fixed_pe, fixed_ms),
            ("random_search_qi", rand_ns, rand_m, rand_pe, rand_ms),
            ("grid_search_qi", grid_ns, grid_m, grid_pe, grid_ms),
            ("adaptive_pinn_qi", adapt_ns, adapt_m, adapt_pe, adapt_ms),
        ]

        for strategy, ns, m, pe, runtime_ms in strategy_data:
            rows.append(
                {
                    "sample_id": idx,
                    "plasma_severity": row["plasma_severity"],
                    "strategy": strategy,
                    "optimized_ns": ns,
                    "optimized_m": m,
                    "resource_cost": float(normalized_resource_cost(ns, m, b)),
                    "qi_pe": pe,
                    "gain_db_vs_fixed_qi": float(db_gain(fixed_pe, pe)),
                    "runtime_ms": runtime_ms,
                    "attenuation_db": float(row["attenuation_db"]),
                    "effective_reflectivity_kappa": kappa,
                    "effective_noise_nb": nb,
                }
            )

    return pd.DataFrame(rows)


def summarize_optimization_results(results: pd.DataFrame) -> pd.DataFrame:
    """Create manuscript-ready optimizer comparison summary."""
    required = ["plasma_severity", "strategy", "qi_pe", "gain_db_vs_fixed_qi", "runtime_ms"]
    missing = [col for col in required if col not in results.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    summary = (
        results.groupby(["plasma_severity", "strategy"])
        .agg(
            samples=("sample_id", "nunique"),
            qi_pe_mean=("qi_pe", "mean"),
            qi_pe_median=("qi_pe", "median"),
            gain_db_vs_fixed_qi_mean=("gain_db_vs_fixed_qi", "mean"),
            runtime_ms_mean=("runtime_ms", "mean"),
            resource_cost_mean=("resource_cost", "mean"),
            ns_mean=("optimized_ns", "mean"),
            m_mean=("optimized_m", "mean"),
        )
        .reset_index()
    )

    severity_order = {"low": 0, "medium": 1, "high": 2}
    strategy_order = {
        "fixed_qi": 0,
        "random_search_qi": 1,
        "grid_search_qi": 2,
        "adaptive_pinn_qi": 3,
    }
    summary["_severity_order"] = summary["plasma_severity"].map(severity_order)
    summary["_strategy_order"] = summary["strategy"].map(strategy_order)
    return (
        summary.sort_values(["_severity_order", "_strategy_order"])
        .drop(columns=["_severity_order", "_strategy_order"])
        .reset_index(drop=True)
    )
