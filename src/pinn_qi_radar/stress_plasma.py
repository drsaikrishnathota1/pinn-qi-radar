"""Stress-test plasma scenario generation.

This module extends the original low/medium/high plasma-channel simulation with
very_low and extreme severity regimes for robustness testing. It keeps the
original generator untouched, so previous scripts remain compatible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .plasma_channel import compute_channel_outputs


@dataclass(frozen=True)
class StressPlasmaConfig:
    n_samples: int = 50000
    seed: int = 42


def _severity_inputs(rng: np.random.Generator, severity: str, n: int) -> pd.DataFrame:
    if severity == "very_low":
        electron_density = rng.uniform(1e15, 8e15, n)
        collision_frequency = rng.uniform(1e7, 8e8, n)
        sheath_thickness = rng.uniform(0.005, 0.025, n)
        thermal_noise = rng.uniform(30, 120, n)
    elif severity == "low":
        electron_density = rng.uniform(8e15, 4e16, n)
        collision_frequency = rng.uniform(8e8, 4e9, n)
        sheath_thickness = rng.uniform(0.02, 0.08, n)
        thermal_noise = rng.uniform(100, 350, n)
    elif severity == "medium":
        electron_density = rng.uniform(4e16, 1.5e17, n)
        collision_frequency = rng.uniform(4e9, 1.2e10, n)
        sheath_thickness = rng.uniform(0.06, 0.18, n)
        thermal_noise = rng.uniform(250, 900, n)
    elif severity == "high":
        electron_density = rng.uniform(1.5e17, 5e17, n)
        collision_frequency = rng.uniform(1.2e10, 3.5e10, n)
        sheath_thickness = rng.uniform(0.14, 0.35, n)
        thermal_noise = rng.uniform(800, 2500, n)
    elif severity == "extreme":
        electron_density = rng.uniform(5e17, 1.2e18, n)
        collision_frequency = rng.uniform(3.5e10, 8e10, n)
        sheath_thickness = rng.uniform(0.30, 0.70, n)
        thermal_noise = rng.uniform(2000, 8000, n)
    else:
        raise ValueError(f"Unknown severity: {severity}")

    return pd.DataFrame(
        {
            "radar_frequency_ghz": rng.uniform(8.0, 18.0, n),
            "electron_density_m3": electron_density,
            "collision_frequency_hz": collision_frequency,
            "sheath_thickness_m": sheath_thickness,
            "base_reflectivity": rng.uniform(0.004, 0.012, n),
            "thermal_noise_nb": thermal_noise,
            "stress_plasma_severity": severity,
        }
    )


def generate_stress_plasma_dataset(config: StressPlasmaConfig) -> pd.DataFrame:
    severities = ["very_low", "low", "medium", "high", "extreme"]
    rng = np.random.default_rng(config.seed)

    counts = np.full(len(severities), config.n_samples // len(severities), dtype=int)
    counts[: config.n_samples % len(severities)] += 1

    frames = []
    for severity, count in zip(severities, counts):
        inputs = _severity_inputs(rng, severity, int(count))
        channel = compute_channel_outputs(
            radar_frequency_ghz=inputs["radar_frequency_ghz"].to_numpy(),
            electron_density_m3=inputs["electron_density_m3"].to_numpy(),
            collision_frequency_hz=inputs["collision_frequency_hz"].to_numpy(),
            sheath_thickness_m=inputs["sheath_thickness_m"].to_numpy(),
            base_reflectivity=inputs["base_reflectivity"].to_numpy(),
            thermal_noise_nb=inputs["thermal_noise_nb"].to_numpy(),
        )
        channel["plasma_severity"] = severity
        frames.append(channel)

    return (
        pd.concat(frames, ignore_index=True)
        .sample(frac=1.0, random_state=config.seed)
        .reset_index(drop=True)
    )
