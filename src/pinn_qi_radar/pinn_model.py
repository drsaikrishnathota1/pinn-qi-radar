"""Physics-informed neural surrogate for plasma-sheath radar channels.

The model learns a normalized mapping from controllable/measured plasma inputs
to normalized channel outputs. A soft physics residual encourages monotonic
behavior that is expected in this simplified simulation setting:

- attenuation should generally increase with electron density
- effective reflectivity should generally decrease with electron density
- effective noise should generally increase with collision frequency

This is a simulation-level surrogate, not a validated real radar model.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class PINNFeatureSpec:
    """Column specification for the plasma-channel PINN surrogate."""

    input_columns: tuple[str, ...] = (
        "radar_frequency_ghz",
        "electron_density_m3",
        "collision_frequency_hz",
        "sheath_thickness_m",
        "base_reflectivity",
        "thermal_noise_nb",
    )
    target_columns: tuple[str, ...] = (
        "attenuation_db",
        "effective_reflectivity_kappa",
        "effective_noise_nb",
        "phase_distortion_rad",
    )


class PlasmaPINN(nn.Module):
    """Small MLP surrogate for plasma-sheath channel outputs."""

    def __init__(
        self,
        input_dim: int = 6,
        output_dim: int = 4,
        hidden_dim: int = 128,
        depth: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be at least 2")
        layers: list[nn.Module] = []
        last_dim = input_dim
        for _ in range(depth - 1):
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return normalized channel-output predictions."""
        return self.net(x)


def gradient_wrt_feature(
    y: torch.Tensor,
    x: torch.Tensor,
    feature_index: int,
) -> torch.Tensor:
    """Compute dy/dx_feature for a scalar batch output y."""
    grad = torch.autograd.grad(
        y.sum(),
        x,
        create_graph=True,
        retain_graph=True,
        allow_unused=False,
    )[0]
    return grad[:, feature_index]


def monotonic_physics_residual(
    model: nn.Module,
    x: torch.Tensor,
    electron_density_index: int = 1,
    collision_frequency_index: int = 2,
    attenuation_target_index: int = 0,
    kappa_target_index: int = 1,
    noise_target_index: int = 2,
) -> torch.Tensor:
    """Soft monotonic residual for the simplified plasma-channel surrogate.

    The residual penalizes violations of expected local trends in normalized
    coordinates. Because all variables are standardized, this is not a hard
    law; it is a stabilizing physics-informed regularizer.
    """

    if not x.requires_grad:
        x = x.detach().clone().requires_grad_(True)

    pred = model(x)

    d_attn_d_ne = gradient_wrt_feature(
        pred[:, attenuation_target_index], x, electron_density_index
    )
    d_kappa_d_ne = gradient_wrt_feature(
        pred[:, kappa_target_index], x, electron_density_index
    )
    d_noise_d_collision = gradient_wrt_feature(
        pred[:, noise_target_index], x, collision_frequency_index
    )

    residual = (
        torch.relu(-d_attn_d_ne).mean()
        + torch.relu(d_kappa_d_ne).mean()
        + torch.relu(-d_noise_d_collision).mean()
    )
    return residual


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
