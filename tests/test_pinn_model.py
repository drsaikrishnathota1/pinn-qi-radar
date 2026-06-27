import torch

from pinn_qi_radar.pinn_model import PlasmaPINN, count_parameters, monotonic_physics_residual


def test_plasma_pinn_forward_shape():
    model = PlasmaPINN(input_dim=6, output_dim=4, hidden_dim=16, depth=3)
    x = torch.randn(8, 6)
    y = model(x)
    assert y.shape == (8, 4)


def test_plasma_pinn_has_trainable_parameters():
    model = PlasmaPINN(input_dim=6, output_dim=4, hidden_dim=16, depth=3)
    assert count_parameters(model) > 0


def test_physics_residual_is_non_negative_scalar():
    model = PlasmaPINN(input_dim=6, output_dim=4, hidden_dim=16, depth=3)
    x = torch.randn(8, 6, requires_grad=True)
    residual = monotonic_physics_residual(model, x)
    assert residual.ndim == 0
    assert float(residual.detach()) >= 0.0
