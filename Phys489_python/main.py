import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from spectral_tools import load_field, compute_spectrum, fit_spectrum, theory_line

# =============================================================================
# PARAMETERS — edit these to match Unity inspector values (cloud cascade)
# =============================================================================

H_TARGET  = 0.5
C1_TARGET = 0.1

# Fitting range (spherical harmonic degrees)
FIT_START = 8
FIT_END   = 60

SNAPSHOTS_FOLDER = "snapshots-cloud"
SNAPSHOT_PATTERN = "snapshot_*.csv"

BETA_THEORY = 1.0 + 2.0 * H_TARGET - 2.0 * C1_TARGET


# =============================================================================
# PLOT SINGLE SNAPSHOT
# =============================================================================

def plot_single(csv_path=None, fit_start=FIT_START, fit_end=FIT_END):
    """Analyse and plot one snapshot. Defaults to the first file in the snapshots folder."""
    if csv_path is None:
        files = sorted(glob.glob(os.path.join(SNAPSHOTS_FOLDER, SNAPSHOT_PATTERN)))
        if not files:
            print(f"No snapshots found in '{SNAPSHOTS_FOLDER}/'.")
            return
        csv_path = files[0]
    print(f"Analysing single snapshot: {csv_path}")
    data = load_field(csv_path)
    degrees, power_per_l = compute_spectrum(data)
    beta, r2, x_fit, reg_line = fit_spectrum(degrees, power_per_l, fit_start, fit_end)
    print(f"  Measured beta = {beta:.4f}   R² = {r2:.4f}   Theory beta = {BETA_THEORY:.4f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(degrees[4:200], power_per_l[4:200], 'b-', alpha=0.6, label='Spectrum')
    ax.loglog(x_fit, reg_line, 'k--', linewidth=2,
              label=f'Regression (l={fit_start}–{fit_end})  $\\beta$ = {beta:.2f}')
    ax.loglog(x_fit, theory_line(x_fit, reg_line, BETA_THEORY), 'r:', linewidth=2,
              label=f'Theory ($\\beta$ = {BETA_THEORY:.2f})')
    ax.set_xlabel('Spherical Harmonic Degree (l)')
    ax.set_ylabel('Power Spectrum E(l)')
    ax.set_title(f'Single Snapshot — Measured $\\beta={beta:.2f}$ vs Theory $\\beta={BETA_THEORY:.2f}$')
    ax.legend()
    ax.grid(True, which="both", ls="-", alpha=0.4)
    plt.tight_layout()
    plt.savefig("Spectrum_Single.png")
    plt.show()


# =============================================================================
# PLOT AVERAGE OVER N SNAPSHOTS
# =============================================================================

def plot_average(snapshots_folder=SNAPSHOTS_FOLDER, fit_start=FIT_START, fit_end=FIT_END):
    """
    Load all snapshot_*.csv files, compute their spectra, fit each one,
    then plot individual traces (faint) and the ensemble average (bold).
    """
    csv_files = sorted(glob.glob(os.path.join(snapshots_folder, SNAPSHOT_PATTERN)))
    if not csv_files:
        print(f"No snapshots found in '{snapshots_folder}/'.")
        return

    print(f"Analysing {len(csv_files)} snapshots...")

    all_spectra = []
    all_betas   = []

    for path in csv_files:
        data = load_field(path)
        degrees, power_per_l = compute_spectrum(data)
        all_spectra.append(power_per_l)
        beta, _, _, _ = fit_spectrum(degrees, power_per_l, fit_start, fit_end)
        all_betas.append(beta)

    all_spectra   = np.array(all_spectra)
    mean_spectrum = all_spectra.mean(axis=0)
    beta_std = np.std(all_betas)
    beta_sem = beta_std / np.sqrt(len(csv_files))
    print(f"  beta = {np.mean(all_betas):.4f} ± {beta_sem:.4f} (SEM)   sigma = {beta_std:.4f} (realization spread)   Theory = {BETA_THEORY:.4f}")

    _, _, x_fit, reg_line = fit_spectrum(degrees, mean_spectrum, fit_start, fit_end)
    beta_avg, r2_avg, _, _ = fit_spectrum(degrees, mean_spectrum, fit_start, fit_end)

    fig, ax = plt.subplots(figsize=(10, 6))

    for spectrum in all_spectra:
        ax.loglog(degrees[4:200], spectrum[4:200], color='steelblue', alpha=0.15, linewidth=0.8)

    ax.loglog(degrees[4:200], mean_spectrum[4:200], 'b-', linewidth=2,
              label=f'Ensemble average (N={len(csv_files)})')
    ax.loglog(x_fit, reg_line, 'k--', linewidth=2,
              label=f'Regression on mean (l={fit_start}–{fit_end})  $\\beta$ = {beta_avg:.2f} $\\pm$ {beta_sem:.2f}')
    ax.loglog(x_fit, theory_line(x_fit, reg_line, BETA_THEORY), 'r:', linewidth=2,
              label=f'Theory  $\\beta$ = {BETA_THEORY:.2f}')

    ax.set_xlabel('Spherical Harmonic Degree (l)')
    ax.set_ylabel('Power Spectrum E(l)')
    ax.set_title(
        f'Spherical Power Spectrum of a Log-Normal Multiplicative Cascade (H={H_TARGET}, C$_1$={C1_TARGET})\n'
        f'Ensemble Average over N={len(csv_files)} Independent Realizations'
    )
    ax.legend()
    ax.grid(True, which="both", ls="-", alpha=0.4)
    plt.tight_layout()
    plt.savefig("Spectrum_Average.png")
    plt.show()


# =============================================================================
# ENTRY POINT — choose mode here
# =============================================================================

if __name__ == "__main__":
    # Analyse a single snapshot:
    # plot_single()

    # Analyse all snapshots and plot the ensemble average:
    plot_average()
