from pinn_qi_radar.plotting import generate_all_figures


def test_generate_all_figures_handles_missing_inputs():
    figs = generate_all_figures()
    assert isinstance(figs, list)
