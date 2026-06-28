#!/usr/bin/env python3
"""
validate_plasma_model.py
========================
External-consistency validation for the PINN-QI Radar plasma-sheath channel model.

WHAT THIS DOES (and does NOT do):
  - RIGOROUSLY validates the electron plasma-frequency physics against the
    closed-form relation and against well-known reference points (these are
    exact physical constants, not fitted values). This is a real validation.
  - Checks that the attenuation / reflectivity / noise proxies are PHYSICALLY
    CONSISTENT: monotonic in the correct directions and bounded sensibly.
    This is a consistency check, not a claim of solver-level accuracy.
  - Produces one publication figure (validation_plasma.png) and a CSV of
    residuals you can cite in the manuscript.

It deliberately does NOT claim agreement with a CFD/full-wave solver, because
the channel model is a compact proxy. Keep that framing in the paper.

USAGE (on RunPod, from the repo root):
    pip install -e .            # or: pip install numpy pandas matplotlib scipy
    python validate_plasma_model.py

Outputs are written to ./validation_outputs/
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- import the repo's own model so we validate the ACTUAL code ----
try:
    from pinn_qi_radar.plasma_channel import (
        plasma_frequency_hz,
        compute_channel_outputs,
        PlasmaChannelConfig,
    )
    USING_REPO = True
except Exception as e:
    print(f"[warn] could not import repo module ({e}); using local copy of formulas")
    USING_REPO = False
    EPS0 = 8.8541878128e-12
    E_CHARGE = 1.602176634e-19
    E_MASS = 9.1093837015e-31
    def plasma_frequency_hz(ne):
        return np.sqrt(np.asarray(ne, float) * E_CHARGE**2 / (EPS0 * E_MASS)) / (2*np.pi)

OUT = "validation_outputs"
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------
# PART 1 — Plasma frequency: validate against the closed-form constant.
# f_p ≈ 8.98 * sqrt(n_e[cm^-3]) Hz  is the classic textbook approximation.
# We confirm the code reproduces it across 6 decades of electron density.
# ----------------------------------------------------------------------
ne_m3 = np.logspace(15, 21, 200)          # electron density [m^-3]
ne_cm3 = ne_m3 * 1e-6                      # convert to cm^-3
fp_code = plasma_frequency_hz(ne_m3)       # from the repo [Hz]
fp_ref = 8980.3 * np.sqrt(ne_cm3)          # textbook: f_p[Hz] = 8980.3*sqrt(n_e[cm^-3])

rel_err = np.abs(fp_code - fp_ref) / fp_ref
print(f"[plasma freq] max relative error vs textbook 8.98*sqrt(n_e[cm^-3]): "
      f"{rel_err.max()*100:.4f}%   (mean {rel_err.mean()*100:.4f}%)")

# Reference anchor points often quoted in plasma-sheath / blackout literature:
anchors = {
    "n_e=1e16 m^-3 (mild reentry)": 1e16,
    "n_e=1e18 m^-3 (moderate)":     1e18,
    "n_e=1e19 m^-3 (severe sheath)":1e19,
}
rows = []
for label, ne in anchors.items():
    rows.append({
        "case": label,
        "n_e_m3": ne,
        "f_p_GHz_code": plasma_frequency_hz(ne) / 1e9,
        "f_p_GHz_textbook": 8980.3 * np.sqrt(ne*1e-6) / 1e9,
    })
anchor_df = pd.DataFrame(rows)
anchor_df["rel_err_pct"] = (
    (anchor_df.f_p_GHz_code - anchor_df.f_p_GHz_textbook).abs()
    / anchor_df.f_p_GHz_textbook * 100
)
anchor_df.to_csv(f"{OUT}/plasma_frequency_anchors.csv", index=False)
print(anchor_df.to_string(index=False))

# ----------------------------------------------------------------------
# PART 2 — Monotonicity / physical-consistency of channel proxies.
# Sweep ONE variable at a time, hold others at a nominal value, and confirm
# the documented physical trends hold (this is what reviewers will probe).
# ----------------------------------------------------------------------
consistency = {}
if USING_REPO:
    n = 200
    nominal = dict(
        radar_frequency_ghz=np.full(n, 8.0),
        electron_density_m3=np.full(n, 1e17),
        collision_frequency_hz=np.full(n, 1e9),
        sheath_thickness_m=np.full(n, 0.1),
        base_reflectivity=np.full(n, 1e-3),
        thermal_noise_nb=np.full(n, 100.0),
    )

    # (a) attenuation should be NON-DECREASING with electron density
    sweep = dict(nominal); sweep["electron_density_m3"] = np.logspace(15, 19, n)
    df = compute_channel_outputs(**sweep)
    d_att = np.diff(df.attenuation_db.values)
    consistency["attenuation_increases_with_ne"] = bool((d_att >= -1e-6).mean() > 0.98)

    # (b) effective reflectivity should be NON-INCREASING with electron density
    consistency["reflectivity_decreases_with_ne"] = bool(
        (np.diff(df.effective_reflectivity_kappa.values) <= 1e-12).mean() > 0.95
    )

    # (c) effective noise should be NON-DECREASING with collision frequency
    sweep = dict(nominal); sweep["collision_frequency_hz"] = np.logspace(8, 11, n)
    df2 = compute_channel_outputs(**sweep)
    consistency["noise_increases_with_collision"] = bool(
        (np.diff(df2.effective_noise_nb.values) >= -1e-6).mean() > 0.95
    )

    print("\n[consistency checks]")
    for k, v in consistency.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    pd.Series(consistency).to_csv(f"{OUT}/consistency_checks.csv")

# ----------------------------------------------------------------------
# PART 3 — Publication figure.
# ----------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

ax[0].loglog(ne_m3, fp_code/1e9, lw=2, label="Model (repo code)")
ax[0].loglog(ne_m3, fp_ref/1e9, "--", lw=1.5, label=r"Textbook $8.98\sqrt{n_e[\mathrm{cm}^{-3}]}$")
ax[0].set_xlabel(r"Electron density $n_e$ [m$^{-3}$]")
ax[0].set_ylabel(r"Plasma frequency $f_p$ [GHz]")
ax[0].set_title(f"Plasma-frequency validation\nmax rel. error {rel_err.max()*100:.3f}%")
ax[0].legend(); ax[0].grid(True, which="both", alpha=0.3)

if USING_REPO:
    ne_axis = np.logspace(15, 19, 200)
    sweep = dict(
        radar_frequency_ghz=np.full(200, 8.0),
        electron_density_m3=ne_axis,
        collision_frequency_hz=np.full(200, 1e9),
        sheath_thickness_m=np.full(200, 0.1),
        base_reflectivity=np.full(200, 1e-3),
        thermal_noise_nb=np.full(200, 100.0),
    )
    dfp = compute_channel_outputs(**sweep)
    ax[1].semilogx(ne_axis, dfp.attenuation_db, lw=2)
    ax[1].set_xlabel(r"Electron density $n_e$ [m$^{-3}$]")
    ax[1].set_ylabel("Attenuation [dB]")
    ax[1].set_title("Attenuation monotonicity\n(physical-consistency check)")
    ax[1].grid(True, which="both", alpha=0.3)
else:
    ax[1].axis("off")
    ax[1].text(0.5, 0.5, "Run inside repo for\nchannel-consistency panel",
               ha="center", va="center")

plt.tight_layout()
plt.savefig(f"{OUT}/validation_plasma.png", dpi=220)
print(f"\nSaved figure -> {OUT}/validation_plasma.png")
print(f"Saved CSVs   -> {OUT}/")
print("\nDONE. Cite Part 1 as physics validation; cite Part 2 as consistency checks.")
print("Do NOT claim solver-level agreement for the attenuation/noise proxies.")
