"""PINN-QI Radar research package."""

from .dataset import build_baseline_dataset
from .plasma_channel import PlasmaChannelConfig, generate_plasma_dataset, plasma_frequency_hz
from .qi_bounds import (
    add_baseline_bounds,
    classical_error_bound,
    db_gain,
    quantum_illumination_error_bound,
    summarize_bounds,
)

__all__ = [
    "PlasmaChannelConfig",
    "generate_plasma_dataset",
    "plasma_frequency_hz",
    "quantum_illumination_error_bound",
    "classical_error_bound",
    "db_gain",
    "add_baseline_bounds",
    "summarize_bounds",
    "build_baseline_dataset",
]
