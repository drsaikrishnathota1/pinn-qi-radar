import numpy as np
import pandas as pd

from pinn_qi_radar.optimizer import (
    OptimizerConfig,
    QIParameterBounds,
    adaptive_surrogate_guided_strategy,
    fixed_qi_strategy,
    grid_search_strategy,
    is_budget_feasible,
    normalized_resource_cost,
    optimize_dataframe,
    random_search_strategy,
    summarize_optimization_results,
)


def test_resource_cost_and_budget_feasibility():
    bounds = QIParameterBounds()
    assert normalized_resource_cost(bounds.ns_min, bounds.m_min, bounds) == 0.0
    assert bool(is_budget_feasible(bounds.ns_min, bounds.m_min, bounds))
    assert not bool(is_budget_feasible(bounds.ns_max, bounds.m_max, bounds))


def test_optimizer_strategies_return_feasible_values():
    bounds = QIParameterBounds()
    config = OptimizerConfig(random_candidates=16, grid_size=10, adaptive_candidates=8)
    rng = np.random.default_rng(42)
    kappa = 0.003
    nb = 400.0

    strategies = [
        fixed_qi_strategy(kappa, nb, bounds, config),
        random_search_strategy(kappa, nb, rng, bounds, config),
        grid_search_strategy(kappa, nb, bounds, config),
        adaptive_surrogate_guided_strategy(kappa, nb, bounds, config),
    ]

    for ns, m, pe in strategies:
        assert bounds.ns_min <= ns <= bounds.ns_max
        assert bounds.m_min <= m <= bounds.m_max
        assert bool(is_budget_feasible(ns, m, bounds))
        assert 0.0 < pe <= 0.5


def test_optimize_dataframe_and_summary():
    df = pd.DataFrame(
        {
            "plasma_severity": ["low", "medium", "high"],
            "attenuation_db": [1.0, 3.0, 8.0],
            "effective_reflectivity_kappa": [0.005, 0.003, 0.001],
            "effective_noise_nb": [200.0, 500.0, 1500.0],
        }
    )
    bounds = QIParameterBounds()
    config = OptimizerConfig(random_candidates=16, grid_size=10, adaptive_candidates=8)

    results = optimize_dataframe(df, bounds=bounds, config=config)
    assert set(results["strategy"]) == {
        "fixed_qi",
        "random_search_qi",
        "grid_search_qi",
        "adaptive_pinn_qi",
    }
    assert len(results) == 12

    summary = summarize_optimization_results(results)
    assert not summary.empty
    assert {"qi_pe_mean", "runtime_ms_mean", "gain_db_vs_fixed_qi_mean"}.issubset(summary.columns)
