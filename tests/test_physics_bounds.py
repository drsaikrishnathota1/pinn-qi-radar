from __future__ import annotations

import numpy as np

from pinn_qi_radar.plasma_channel import PlasmaChannelConfig, generate_plasma_dataset, plasma_frequency_hz
from pinn_qi_radar.qi_bounds import add_baseline_bounds, classical_error_bound, quantum_illumination_error_bound


def test_plasma_frequency_increases_with_density() -> None:
    densities = np.array([1e15, 1e16, 1e17])
    freqs = plasma_frequency_hz(densities)
    assert np.all(np.diff(freqs) > 0)


def test_dataset_contains_channel_outputs() -> None:
    df = generate_plasma_dataset(PlasmaChannelConfig(n_samples=250, seed=7))
    expected = {
        "effective_reflectivity_kappa",
        "effective_noise_nb",
        "attenuation_db",
        "plasma_severity",
    }
    assert expected.issubset(df.columns)
    assert len(df) == 250
    assert df["effective_reflectivity_kappa"].between(1e-12, 1.0).all()


def test_qi_bound_is_not_worse_than_classical_proxy() -> None:
    kappa = np.array([1e-5, 1e-4, 1e-3])
    nb = np.array([100.0, 100.0, 100.0])
    ns = 0.02
    m = 1e6
    qi = quantum_illumination_error_bound(ns, m, kappa, nb)
    classical = classical_error_bound(ns, m, kappa, nb)
    assert np.all(qi <= classical)


def test_baseline_bounds_are_added() -> None:
    df = generate_plasma_dataset(PlasmaChannelConfig(n_samples=100, seed=11))
    out = add_baseline_bounds(df)
    assert "classical_pe" in out.columns
    assert "fixed_qi_pe" in out.columns
    assert out["classical_pe"].between(0, 0.5).all()
    assert out["fixed_qi_pe"].between(0, 0.5).all()
