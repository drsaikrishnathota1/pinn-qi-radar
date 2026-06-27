"""Quantum-illumination and classical detection-bound utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def quantum_illumination_error_bound(
    signal_photon_number_ns: np.ndarray | float,
    mode_pairs_m: np.ndarray | float,
    effective_reflectivity_kappa: np.ndarray | float,
    effective_noise_nb: np.ndarray | float,
) -> np.ndarray:
    """Approximate QI error-probability bound.

    The expression is a compact simulation-level bound proxy for comparing
    parameter choices under high-loss/high-noise conditions:

        Pe_QI = 0.5 * exp(-M * kappa * Ns / (NB + 1))

    It is intended for relative computational comparison, not hardware claims.
    """
    ns = np.asarray(signal_photon_number_ns, dtype=float)
    m = np.asarray(mode_pairs_m, dtype=float)
    kappa = np.asarray(effective_reflectivity_kappa, dtype=float)
    nb = np.asarray(effective_noise_nb, dtype=float)
    exponent = -m * np.clip(kappa, 1e-12, None) * np.clip(ns, 1e-12, None) / (nb + 1.0)
    return np.clip(0.5 * np.exp(exponent), 1e-300, 0.5)


def classical_error_bound(
    signal_photon_number_ns: np.ndarray | float,
    pulses_m: np.ndarray | float,
    effective_reflectivity_kappa: np.ndarray | float,
    effective_noise_nb: np.ndarray | float,
) -> np.ndarray:
    """Conservative classical coherent-radar error-bound proxy."""
    ns = np.asarray(signal_photon_number_ns, dtype=float)
    m = np.asarray(pulses_m, dtype=float)
    kappa = np.asarray(effective_reflectivity_kappa, dtype=float)
    nb = np.asarray(effective_noise_nb, dtype=float)
    exponent = -0.25 * m * np.clip(kappa, 1e-12, None) * np.clip(ns, 1e-12, None) / (nb + 1.0)
    return np.clip(0.5 * np.exp(exponent), 1e-300, 0.5)


def db_gain(reference_error: np.ndarray | float, proposed_error: np.ndarray | float) -> np.ndarray:
    """Compute positive dB gain when proposed error is lower than reference."""
    ref = np.clip(np.asarray(reference_error, dtype=float), 1e-300, None)
    prop = np.clip(np.asarray(proposed_error, dtype=float), 1e-300, None)
    return 10.0 * np.log10(ref / prop)


def add_baseline_bounds(
    df: pd.DataFrame,
    fixed_ns: float = 0.02,
    fixed_m: float = 1e6,
) -> pd.DataFrame:
    """Add classical and fixed-QI detection-bound columns to a channel dataset."""
    out = df.copy()
    out["fixed_ns"] = fixed_ns
    out["fixed_m"] = fixed_m
    out["classical_pe"] = classical_error_bound(
        fixed_ns, fixed_m, out["effective_reflectivity_kappa"], out["effective_noise_nb"]
    )
    out["fixed_qi_pe"] = quantum_illumination_error_bound(
        fixed_ns, fixed_m, out["effective_reflectivity_kappa"], out["effective_noise_nb"]
    )
    out["fixed_qi_gain_db_vs_classical"] = db_gain(out["classical_pe"], out["fixed_qi_pe"])
    return out


def summarize_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """Create a manuscript-ready summary by plasma severity."""
    required = ["plasma_severity", "classical_pe", "fixed_qi_pe", "fixed_qi_gain_db_vs_classical"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return (
        df.groupby("plasma_severity")
        .agg(
            samples=("plasma_severity", "size"),
            attenuation_db_mean=("attenuation_db", "mean"),
            kappa_mean=("effective_reflectivity_kappa", "mean"),
            noise_nb_mean=("effective_noise_nb", "mean"),
            classical_pe_mean=("classical_pe", "mean"),
            fixed_qi_pe_mean=("fixed_qi_pe", "mean"),
            fixed_qi_gain_db_mean=("fixed_qi_gain_db_vs_classical", "mean"),
        )
        .reset_index()
        .sort_values("plasma_severity")
    )
