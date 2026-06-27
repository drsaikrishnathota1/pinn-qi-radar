"""Plasma-sheath radar channel model for PINN-QI Radar.

The model is intentionally simulation-level and lightweight for a short
communication. It converts hypersonic plasma-sheath condition proxies into
channel quantities used by the quantum-illumination bound layer.

This is not a validated operational radar model. It is a reproducible,
physics-motivated surrogate for computational experiments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EPS0 = 8.8541878128e-12
E_CHARGE = 1.602176634e-19
E_MASS = 9.1093837015e-31
PI = np.pi


@dataclass(frozen=True)
class PlasmaChannelConfig:
    """Parameter ranges for synthetic plasma-sheath scenarios."""

    n_samples: int = 50_000
    seed: int = 42
    radar_frequency_min_ghz: float = 1.0
    radar_frequency_max_ghz: float = 18.0
    electron_density_min_m3: float = 1e15
    electron_density_max_m3: float = 5e18
    collision_frequency_min_hz: float = 1e8
    collision_frequency_max_hz: float = 8e10
    sheath_thickness_min_m: float = 0.01
    sheath_thickness_max_m: float = 0.25
    base_reflectivity_min: float = 1e-5
    base_reflectivity_max: float = 5e-2
    thermal_noise_min: float = 10.0
    thermal_noise_max: float = 1_000.0


def plasma_frequency_hz(electron_density_m3: np.ndarray) -> np.ndarray:
    """Compute electron plasma frequency in Hz from electron density."""
    omega_p = np.sqrt(electron_density_m3 * E_CHARGE**2 / (EPS0 * E_MASS))
    return omega_p / (2.0 * PI)


def classify_severity(electron_density_m3: np.ndarray) -> np.ndarray:
    """Map electron density to low/medium/high plasma-severity labels."""
    return np.select(
        [electron_density_m3 < 8e16, electron_density_m3 < 8e17],
        ["low", "medium"],
        default="high",
    )


def compute_channel_outputs(
    radar_frequency_ghz: np.ndarray,
    electron_density_m3: np.ndarray,
    collision_frequency_hz: np.ndarray,
    sheath_thickness_m: np.ndarray,
    base_reflectivity: np.ndarray,
    thermal_noise_nb: np.ndarray,
) -> pd.DataFrame:
    """Compute effective channel outputs for modeled plasma-sheath conditions."""
    radar_frequency_hz = radar_frequency_ghz * 1e9
    plasma_frequency = plasma_frequency_hz(electron_density_m3)
    frequency_ratio = plasma_frequency / np.maximum(radar_frequency_hz, 1.0)
    collision_ratio = collision_frequency_hz / np.maximum(radar_frequency_hz, 1.0)

    # Physics-motivated attenuation proxy: stronger below/near plasma frequency,
    # stronger with collision frequency, and stronger for thicker sheaths.
    attenuation_db = (
        4.0 * np.log10(1.0 + frequency_ratio**2)
        + 2.5 * np.log10(1.0 + collision_ratio)
        + 18.0 * sheath_thickness_m * np.clip(frequency_ratio, 0.0, 8.0)
    )
    attenuation_db = np.clip(attenuation_db, 0.0, 80.0)
    attenuation_linear = 10.0 ** (-attenuation_db / 10.0)

    effective_reflectivity = np.clip(base_reflectivity * attenuation_linear, 1e-12, 1.0)

    # Phase distortion proxy increases near cutoff-like conditions and with collisions.
    phase_distortion_rad = np.clip(
        0.15 * frequency_ratio + 0.08 * collision_ratio + 2.0 * sheath_thickness_m,
        0.0,
        4.0,
    )

    # Effective noise includes thermal noise and plasma-induced excess noise proxy.
    plasma_excess_noise = 1.0 + 0.65 * frequency_ratio**2 + 0.10 * collision_ratio
    effective_noise_nb = np.clip(thermal_noise_nb * plasma_excess_noise, 1.0, 1e6)

    return pd.DataFrame(
        {
            "radar_frequency_ghz": radar_frequency_ghz,
            "electron_density_m3": electron_density_m3,
            "collision_frequency_hz": collision_frequency_hz,
            "sheath_thickness_m": sheath_thickness_m,
            "base_reflectivity": base_reflectivity,
            "thermal_noise_nb": thermal_noise_nb,
            "plasma_frequency_hz": plasma_frequency,
            "frequency_ratio": frequency_ratio,
            "collision_ratio": collision_ratio,
            "attenuation_db": attenuation_db,
            "effective_reflectivity_kappa": effective_reflectivity,
            "phase_distortion_rad": phase_distortion_rad,
            "effective_noise_nb": effective_noise_nb,
            "plasma_severity": classify_severity(electron_density_m3),
        }
    )


def generate_plasma_dataset(config: PlasmaChannelConfig | None = None) -> pd.DataFrame:
    """Generate a synthetic plasma-channel dataset for computational analysis."""
    cfg = config or PlasmaChannelConfig()
    if cfg.n_samples <= 0:
        raise ValueError("n_samples must be positive")
    rng = np.random.default_rng(cfg.seed)

    radar_frequency_ghz = rng.uniform(
        cfg.radar_frequency_min_ghz, cfg.radar_frequency_max_ghz, cfg.n_samples
    )
    electron_density_m3 = 10.0 ** rng.uniform(
        np.log10(cfg.electron_density_min_m3),
        np.log10(cfg.electron_density_max_m3),
        cfg.n_samples,
    )
    collision_frequency_hz = 10.0 ** rng.uniform(
        np.log10(cfg.collision_frequency_min_hz),
        np.log10(cfg.collision_frequency_max_hz),
        cfg.n_samples,
    )
    sheath_thickness_m = rng.uniform(
        cfg.sheath_thickness_min_m, cfg.sheath_thickness_max_m, cfg.n_samples
    )
    base_reflectivity = 10.0 ** rng.uniform(
        np.log10(cfg.base_reflectivity_min),
        np.log10(cfg.base_reflectivity_max),
        cfg.n_samples,
    )
    thermal_noise_nb = 10.0 ** rng.uniform(
        np.log10(cfg.thermal_noise_min),
        np.log10(cfg.thermal_noise_max),
        cfg.n_samples,
    )

    return compute_channel_outputs(
        radar_frequency_ghz=radar_frequency_ghz,
        electron_density_m3=electron_density_m3,
        collision_frequency_hz=collision_frequency_hz,
        sheath_thickness_m=sheath_thickness_m,
        base_reflectivity=base_reflectivity,
        thermal_noise_nb=thermal_noise_nb,
    )
