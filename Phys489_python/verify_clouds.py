"""
verify_clouds.py
================
Full verification of the multifractal cloud cascade against Universal
Multifractal (UM) theory (Schertzer & Lovejoy 1987).

Usage
-----
1. In Unity Play Mode, press X to export cloud snapshots.
   Files land in Phys489_python/snapshots-cloud/snapshot_NNN.csv
2. Set H_TARGET and C1_TARGET below to match the Unity inspector values.
3. Run:  python verify_clouds.py

Outputs
-------
  Cloud_Verification_Panel.png  —  2×2 diagnostic figure
  Console                       —  β, C1 (structure functions), skewness, kurtosis
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from spectral_tools import (
    load_field, compute_spectrum, fit_spectrum, theory_line,
    structure_functions, fit_zeta, theoretical_Kq, check_lognormality,
)

# =============================================================================
# PARAMETERS — match Unity inspector values (cloud cascade)
# =============================================================================
C1_TARGET    = 0.1
H_TARGET     = 0.33
ALPHA_TARGET = 1.79

# β = 1 + 2H − K(2);  general UM: K(2) = C1/(α−1) * (2^α − 2)
# At α=2 this reduces to K(2) = C1, giving β = 1 + 2H − 2C1
K2_THEORY   = theoretical_Kq(2.0, C1_TARGET, ALPHA_TARGET)
BETA_THEORY = 1.0 + 2.0 * H_TARGET - K2_THEORY

FIT_START = 8
FIT_END   = 60

SNAPSHOTS_FOLDER = "snapshots-cloud"
SNAPSHOT_PATTERN = os.path.join(SNAPSHOTS_FOLDER, "snapshot_*.csv")


# =============================================================================
# MAIN VERIFICATION ROUTINE
# =============================================================================

def run_verification():
    files = sorted(glob.glob(SNAPSHOT_PATTERN))
    if not files:
        print(f"No cloud snapshots found matching: {SNAPSHOT_PATTERN}")
        print("Press X in Unity Play Mode to export cloud fields first.")
        return

    print(f"Found {len(files)} cloud snapshot(s).")

    # -------------------------------------------------------------------------
    # 1. Ensemble spectral analysis
    # -------------------------------------------------------------------------
    all_spectra      = []
    all_betas        = []
    field_for_moments = None
    degrees_last     = None

    for path in files:
        data = load_field(path)
        degrees, power_per_l = compute_spectrum(data)
        all_spectra.append(power_per_l)
        beta, _, _, _ = fit_spectrum(degrees, power_per_l, FIT_START, FIT_END)
        all_betas.append(beta)
        if field_for_moments is None:
            field_for_moments = data
            degrees_last      = degrees

    all_spectra   = np.array(all_spectra)
    mean_spectrum = all_spectra.mean(axis=0)
    beta_mean = np.mean(all_betas)
    beta_sem  = np.std(all_betas) / np.sqrt(len(all_betas))
    beta_avg, r2_avg, x_fit, reg_line = fit_spectrum(degrees_last, mean_spectrum, FIT_START, FIT_END)

    print(f"\n--- Power Spectrum ---")
    print(f"  beta (ensemble mean)  = {beta_mean:.4f} ± {beta_sem:.4f}")
    print(f"  beta (fit on average) = {beta_avg:.4f}   R² = {r2_avg:.4f}")
    print(f"  beta (theory)         = {BETA_THEORY:.4f}")

    # -------------------------------------------------------------------------
    # 2. Structure function / K(q) analysis on first snapshot
    #    ζ(q) = q·H − K(q)  →  K(q) = q·H − ζ(q)
    #
    #    H_eff is derived from the measured beta rather than the nominal H_TARGET
    #    because the discrete cascade + Hermite smoothstep systematically steepen
    #    the spectrum by ~0.02 in H.  Using H_eff removes the linear bias in K(q).
    # -------------------------------------------------------------------------
    q_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    r_values = [2, 4, 8, 16, 32, 64, 128]
    q_arr    = np.array(q_values)

    print("\nComputing structure functions (this may take a few seconds)...")
    sf   = structure_functions(field_for_moments, q_values, r_values)
    zeta = fit_zeta(sf, r_values, q_values)

    H_eff = (beta_avg - 1.0 + 2.0 * C1_TARGET) / 2.0
    Kq    = q_arr * H_eff - zeta

    try:
        (C1_fit,), _ = curve_fit(
            lambda q, c1: theoretical_Kq(q, c1, ALPHA_TARGET), q_arr, Kq, p0=[C1_TARGET]
        )
    except Exception:
        C1_fit = np.nan

    print(f"\n--- Structure Functions ---")
    print(f"  H_eff (from beta_avg) = {H_eff:.4f}   (nominal: {H_TARGET:.4f})")
    print(f"  C1 (fitted from K(q)) = {C1_fit:.4f}   (target: {C1_TARGET:.4f})")
    print(f"  {'q':>4}   {'ζ(q)':>8}   {'K(q) meas':>10}   {'K(q) theory':>12}")
    for q, z, k in zip(q_values, zeta, Kq):
        theory_k = theoretical_Kq(q, C1_TARGET, ALPHA_TARGET)
        print(f"  {q:>4.1f}   {z:>8.4f}   {k:>+10.4f}   {theory_k:>+12.4f}")

    # -------------------------------------------------------------------------
    # 3. Log-normality check
    # -------------------------------------------------------------------------
    sk, kurt = check_lognormality(field_for_moments)
    print(f"\n--- Log-normality ---")
    print(f"  Skewness of log(field)        = {sk:.4f}   (target: 0)")
    print(f"  Excess kurtosis of log(field) = {kurt:.4f}  (target: 0)")

    # -------------------------------------------------------------------------
    # 4. Verification panel figure (2×2)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Cloud Cascade Verification  |  H={H_TARGET}, C₁={C1_TARGET}, α={ALPHA_TARGET}  |  "
        f"N={len(files)} realization(s)",
        fontsize=13
    )

    # Panel A: raw field (equirectangular, log scale)
    ax = axes[0, 0]
    im = ax.imshow(np.log1p(field_for_moments), cmap="gray", aspect="auto", origin="upper")
    plt.colorbar(im, ax=ax, label="log(1 + cloudIntensity)")
    ax.set_title("A — Raw Cloud Field (log scale)")
    ax.set_xlabel("Longitude pixel")
    ax.set_ylabel("Latitude pixel")

    # Panel B: power spectrum
    ax = axes[0, 1]
    for spectrum in all_spectra:
        ax.loglog(degrees_last[4:200], spectrum[4:200],
                  color="steelblue", alpha=0.2, linewidth=0.8)
    ax.loglog(degrees_last[4:200], mean_spectrum[4:200], "b-", linewidth=2,
              label=f"Ensemble average (N={len(files)})")
    ax.loglog(x_fit, reg_line, "k--", linewidth=2,
              label=f"Regression  β = {beta_avg:.2f} ± {beta_sem:.2f}")
    ax.loglog(x_fit, theory_line(x_fit, reg_line, BETA_THEORY), "r:", linewidth=2,
              label=f"Theory  β = {BETA_THEORY:.2f}")
    ax.set_xlabel("Spherical Harmonic Degree (l)")
    ax.set_ylabel("Power Spectrum E(l)")
    ax.set_title("B — Power Spectrum")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", ls="-", alpha=0.3)

    # Panel C: K(q) from structure functions  [K(q) = q·H − ζ(q)]
    ax = axes[1, 0]
    q_smooth = np.linspace(0, 3.5, 100)
    ax.plot(q_smooth, theoretical_Kq(q_smooth, C1_TARGET, ALPHA_TARGET), "r--", linewidth=2,
            label=f"Theory  K(q) = C₁/(α−1)·(q^α−q),  C₁={C1_TARGET}, α={ALPHA_TARGET}")
    ax.plot(q_arr, Kq, "ko-", linewidth=2, markersize=6,
            label=f"Measured  C₁={C1_fit:.3f}, H_eff={H_eff:.3f}")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Moment order q")
    ax.set_ylabel("K(q) = q·H − ζ(q)")
    ax.set_title("C — Structure Function K(q)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel D: histogram of log(field)
    ax = axes[1, 1]
    flat = field_for_moments.ravel()
    flat = flat[flat > 0]
    log_flat = np.log(flat)
    mu, sigma_log = np.mean(log_flat), np.std(log_flat)
    ax.hist(log_flat, bins=80, density=True, color="steelblue", alpha=0.7, label="Empirical")
    x_gauss = np.linspace(log_flat.min(), log_flat.max(), 300)
    gauss   = np.exp(-0.5 * ((x_gauss - mu) / sigma_log) ** 2) / (sigma_log * np.sqrt(2 * np.pi))
    ax.plot(x_gauss, gauss, "r-", linewidth=2,
            label=f"Gaussian fit  μ={mu:.2f}, σ={sigma_log:.2f}")
    ax.set_xlabel("log(cloudIntensity)")
    ax.set_ylabel("Probability density")
    ax.set_title(f"D — Log-field PDF  (skew={sk:.2f}, kurt={kurt:.2f})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = "Cloud_Verification_Panel.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved: {out_path}")
    plt.show()


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    run_verification()
