import numpy as np

from pinn_qi_radar.policy_optimizer import (
    QIPolicyNet,
    ns_m_to_norm,
    norm_to_ns_m,
    project_to_budget,
)
from pinn_qi_radar.optimizer import QIParameterBounds, normalized_resource_cost


def test_policy_forward_shape():
    import torch

    model = QIPolicyNet(input_dim=8, hidden_dim=16, depth=3)
    x = torch.randn(5, 8)
    y = model(x)
    assert y.shape == (5, 2)
    assert float(y.min()) >= 0.0
    assert float(y.max()) <= 1.0


def test_norm_round_trip():
    bounds = QIParameterBounds()
    ns, m = 0.03, 1.5e6
    ns_norm, m_norm = ns_m_to_norm(ns, m, bounds)
    ns2, m2 = norm_to_ns_m(np.array([[ns_norm, m_norm]]), bounds)
    assert abs(ns2[0] - ns) < 1e-9
    assert abs(m2[0] - m) < 1e-3


def test_project_to_budget():
    bounds = QIParameterBounds(budget=0.65)
    ns = np.array([bounds.ns_max])
    m = np.array([bounds.m_max])
    ns2, m2 = project_to_budget(ns, m, bounds)
    assert normalized_resource_cost(ns2[0], m2[0], bounds) <= bounds.budget + 1e-9
