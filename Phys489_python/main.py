import os
import glob
import numpy as np
import pyshtools as pysh
import matplotlib.pyplot as plt
from scipy.stats import linregress

# =============================================================================
# PARAMETERS — match these to the Unity Inspector values before running
# =============================================================================
H_target  = 0.5
C1_target = 0.1

# Theoretical spectral slope: beta = 1 + 2H - 2C1
BETA_THEORY = 1.0 + 2.0 * H_target - 2.0 * C1_target

# Fitting range: skip the planetary-scale end (l<fit_start) and the
# resolution roll-off (l>fit_end). Adjust if resolution changes.
FIT_START = 8
FIT_END   = 60

# Paths
SNAPSHOTS_FOLDER = "snapshots"


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def compute_spectrum(csv_path):
    """Load a field CSV and return (degrees, power_per_l) via spherical harmonics."""
    data        = np.loadtxt(csv_path, delimiter=",")
    grid        = pysh.SHGrid.from_array(data)
    coeffs      = grid.expand()
    power_per_l = coeffs.spectrum()
    degrees     = np.arange(len(power_per_l))
    return degrees, power_per_l


def fit_spectrum(degrees, power_per_l, fit_start=FIT_START, fit_end=FIT_END):
    """
    Log-log linear regression over degrees[fit_start:fit_end].
    Returns (beta_measured, r_squared, x_fit, regression_line).
    """
    x_fit = degrees[fit_start:fit_end]
    y_fit = power_per_l[fit_start:fit_end]

    slope, intercept, r_value, _, _ = linregress(np.log(x_fit), np.log(y_fit))

    beta_measured  = -slope
    reg_line       = np.exp(intercept) * x_fit ** slope
    return beta_measured, r_value ** 2, x_fit, reg_line


def _add_theory_line(ax, x_fit, reg_line, beta_theory):
    """Overlay the theoretical slope anchored to the midpoint of the regression line."""
    mid_x        = x_fit[len(x_fit) // 2]
    mid_y        = reg_line[len(reg_line) // 2]
    theory_line  = mid_y * (x_fit / mid_x) ** (-beta_theory)
    ax.loglog(x_fit, theory_line, 'r:', linewidth=2,
              label=f'Theory ($\\beta$ = {beta_theory:.2f})')


# =============================================================================
# PLOT SINGLE SNAPSHOT
# =============================================================================

def plot_single(csv_path=None, fit_start=FIT_START, fit_end=FIT_END):
    """Analyse and plot one snapshot. Defaults to the first file in the snapshots folder."""
    if csv_path is None:
        files = sorted(glob.glob(os.path.join(SNAPSHOTS_FOLDER, "snapshot_*.csv")))
        if not files:
            print(f"No snapshots found in '{SNAPSHOTS_FOLDER}/'.")
            return
        csv_path = files[0]
    print(f"Analysing single snapshot: {csv_path}")
    degrees, power_per_l = compute_spectrum(csv_path)
    beta, r2, x_fit, reg_line = fit_spectrum(degrees, power_per_l, fit_start, fit_end)
    print(f"  Measured beta = {beta:.4f}   R² = {r2:.4f}   Theory beta = {BETA_THEORY:.4f}")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.loglog(degrees[4:200], power_per_l[4:200], 'b-', alpha=0.6, label='Spectrum')
    ax.loglog(x_fit, reg_line, 'k--', linewidth=2,
              label=f'Regression (l={fit_start}–{fit_end})  $\\beta$ = {beta:.2f}')
    _add_theory_line(ax, x_fit, reg_line, BETA_THEORY)

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
    csv_files = sorted(glob.glob(os.path.join(snapshots_folder, "snapshot_*.csv")))
    if not csv_files:
        print(f"No snapshots found in '{snapshots_folder}/'.")
        return

    print(f"Analysing {len(csv_files)} snapshots...")

    all_spectra = []
    all_betas   = []

    for path in csv_files:
        degrees, power_per_l = compute_spectrum(path)
        all_spectra.append(power_per_l)
        beta, _, _, _ = fit_spectrum(degrees, power_per_l, fit_start, fit_end)
        all_betas.append(beta)

    all_spectra  = np.array(all_spectra)   # shape: (N, n_degrees)
    mean_spectrum = all_spectra.mean(axis=0)
    beta_std     = np.std(all_betas)
    beta_sem     = beta_std / np.sqrt(len(csv_files))  # standard error on the ensemble mean
    print(f"  beta = {np.mean(all_betas):.4f} ± {beta_sem:.4f} (SEM)   sigma = {beta_std:.4f} (realization spread)   Theory = {BETA_THEORY:.4f}")

    _, _, x_fit, reg_line = fit_spectrum(degrees, mean_spectrum, fit_start, fit_end)
    beta_avg, r2_avg, _, _ = fit_spectrum(degrees, mean_spectrum, fit_start, fit_end)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Individual realizations — faint traces
    for spectrum in all_spectra:
        ax.loglog(degrees[4:200], spectrum[4:200], color='steelblue', alpha=0.15, linewidth=0.8)

    # Ensemble average
    ax.loglog(degrees[4:200], mean_spectrum[4:200], 'b-', linewidth=2,
              label=f'Ensemble average (N={len(csv_files)})')

    # Regression on the average
    ax.loglog(x_fit, reg_line, 'k--', linewidth=2,
              label=f'Regression on mean (l={fit_start}–{fit_end})  $\\beta$ = {beta_avg:.2f} $\\pm$ {beta_sem:.2f}')

    _add_theory_line(ax, x_fit, reg_line, BETA_THEORY)

    ax.set_xlabel('Spherical Harmonic Degree (l)')
    ax.set_ylabel('Power Spectrum E(l)')
    ax.set_title(
        f'Spherical Power Spectrum of a Log-Normal Multiplicative Cascade (H={H_target}, C$_1$={C1_target})\n'
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