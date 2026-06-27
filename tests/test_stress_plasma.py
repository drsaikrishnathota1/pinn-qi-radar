from pinn_qi_radar.stress_plasma import StressPlasmaConfig, generate_stress_plasma_dataset


def test_stress_plasma_dataset_has_all_regimes():
    df = generate_stress_plasma_dataset(StressPlasmaConfig(n_samples=50, seed=1))
    assert len(df) == 50
    assert set(df["plasma_severity"]) == {"very_low", "low", "medium", "high", "extreme"}
    assert "effective_reflectivity_kappa" in df.columns
    assert "effective_noise_nb" in df.columns
