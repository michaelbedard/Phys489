# verify_anisotropy.py
# Verifies space-time anisotropy of the OU-evolved FIF cloud simulation.
#
# Physics law to verify:
#   tau(k) = tau0 * |k|^{-z}
#
# Method (Option A — autocorrelation):
#   1. Load N_t consecutive frames from snapshots-timeseries/
#   2. For each frame, take the 2D FFT of log(field) to get Fourier modes F(k, t)
#   3. For each ring of wavenumbers |k| ~ r, compute the normalized temporal
#      autocorrelation C(r, delta_t) = <Re[F*(k,t) F(k,t+delta_t)]> / <|F(k,t)|^2>
#   4. Fit C(r, delta_t) = exp(-delta_t / tau(r)) to extract tau per ring
#   5. Log-log regression of tau vs r -> slope should be -z
#
# Usage:
#   cd C:\Users\courr\Phys489\Phys489_python
#   .venv\Scripts\activate
#   python verify_anisotropy.py
#
# Parameters to set below: match Inspector values used during capture.

import os
import glob
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# USER PARAMETERS — match Inspector at capture time
# ---------------------------------------------------------------------------
Z_THEORY    = 0.5    # Inspector z (dynamical scaling exponent)
TAU0_THEORY = 30.0    # Inspector tau0 (seconds)
DT_EXPORT   = 0.1    # Inspector timeSeriesInterval (seconds between frames)
WIND_VX     = 0.05   # Inspector Wind Velocity X (UV-space cycles/s)
WIND_VY     = 0.05   # Inspector Wind Velocity Y (UV-space cycles/s)

RING_MIN   = 5     # minimum |k| ring to include in the fit (avoid DC boundary)
RING_MAX   = 150   # maximum |k| ring (avoid resolution limit)
MAX_LAG    = 80    # maximum lag index to use for autocorrelation fit
MIN_MODES  = 8     # minimum modes per ring to include ring in fit

SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "snapshots-timeseries")
# ---------------------------------------------------------------------------

def load_frames(folder):
    paths = sorted(glob.glob(os.path.join(folder, "frame_*.csv")))
    if len(paths) == 0:
        raise FileNotFoundError(f"No frame_NNN.csv files found in {folder}")
    print(f"Loading {len(paths)} frames from {folder} ...")
    frames = []
    for p in paths:
        f = np.loadtxt(p, delimiter=",", dtype=np.float32)
        frames.append(f)
    return np.stack(frames, axis=0)  # shape (N_t, N, N)


def compute_fourier_series(frames):
    """
    Take the 2D FFT of log(field) for each frame.
    Using log(field) recovers the log-field (the Gaussian-like variable before exp()).
    Returns complex array of shape (N_t, N, N).
    """
    log_frames = np.log(np.clip(frames, 1e-10, None))
    # Subtract spatial mean per frame to remove DC drift
    log_frames -= log_frames.mean(axis=(1, 2), keepdims=True)
    print("Computing 2D FFT for each frame ...")
    F = np.fft.fft2(log_frames, axes=(1, 2))  # (N_t, N, N)
    return F


def deadvect(F, dt, wind_vx, wind_vy):
    """
    Strip the wind-advection phase from each Fourier mode.

    PhaseShift applies: F_working(k,t) = F_base(k,t) * exp(i*2π*(kx*vx+ky*vy)*t)
    Taking the autocorrelation of F_working gives:
        C(Δt) = C_base(Δt) * cos(2π*(kx*vx+ky*vy)*Δt)
    which oscillates and hides the true OU decorrelation time.

    This function multiplies by exp(-i*2π*(kx*vx+ky*vy)*t) to recover F_base,
    so that the autocorrelation of the output decays as a pure exponential.
    """
    N_t, N, _ = F.shape
    ix = np.arange(N)
    kx = np.where(ix < N // 2, ix, ix - N).astype(float)
    ky = np.where(ix < N // 2, ix, ix - N).astype(float)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')       # (N, N)
    t_arr   = np.arange(N_t) * dt                      # (N_t,)
    # advection phase per mode per frame: shape (N_t, N, N)
    phase = 2.0 * np.pi * (KX[None] * wind_vx + KY[None] * wind_vy) * t_arr[:, None, None]
    return F * np.exp(-1j * phase)


def build_ring_map(N):
    """
    For each pixel (ix, iy) in an N x N FFT grid, compute the integer ring index
    r = round(|k|) where k = (kx, ky) in signed-frequency convention.
    Returns array of shape (N, N).
    """
    ix = np.arange(N)
    iy = np.arange(N)
    kx = np.where(ix < N // 2, ix, ix - N).astype(float)
    ky = np.where(iy < N // 2, iy, iy - N).astype(float)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    return np.round(np.sqrt(KX**2 + KY**2)).astype(int)


def normalized_autocorrelation(signal, max_lag):
    """
    Compute the normalized temporal autocorrelation of a 1D complex signal X(t):
        C(delta_t) = <Re[X*(t) X(t+delta_t)]> / <|X(t)|^2>
    Returns array of length max_lag+1 (lags 0..max_lag).
    """
    N_t = len(signal)
    norm = np.mean(np.abs(signal)**2)
    if norm < 1e-30:
        return np.zeros(max_lag + 1)
    C = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        if N_t - lag <= 0:
            break
        C[lag] = np.mean(np.real(np.conj(signal[:N_t - lag]) * signal[lag:N_t])) / norm
    return C


def fit_tau(C, dt):
    """
    Fit C(delta_t) = exp(-delta_t / tau) to the autocorrelation array C.
    Returns (tau, fit_quality_r2).
    """
    lags = np.arange(len(C)) * dt
    # Only fit where C > exp(-3) to avoid fitting the noise floor
    valid = C > np.exp(-3)
    if valid.sum() < 3:
        return np.nan, 0.0
    try:
        popt, _ = curve_fit(lambda t, tau: np.exp(-t / tau),
                             lags[valid], C[valid],
                             p0=[lags[valid].max() / 2],
                             bounds=(1e-6, np.inf),
                             maxfev=2000)
        tau = popt[0]
        resid = C[valid] - np.exp(-lags[valid] / tau)
        ss_res = np.sum(resid**2)
        ss_tot = np.sum((C[valid] - C[valid].mean())**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return tau, r2
    except Exception:
        return np.nan, 0.0


def main():
    frames = load_frames(SNAPSHOT_DIR)
    N_t, N, _ = frames.shape
    print(f"  N_t={N_t} frames, N={N}, dt={DT_EXPORT}s, total={N_t*DT_EXPORT:.1f}s")

    F = compute_fourier_series(frames)              # (N_t, N, N) complex
    F = deadvect(F, DT_EXPORT, WIND_VX, WIND_VY)  # strip wind phase → recover F_base
    ring_map = build_ring_map(N)                   # (N, N) int

    max_ring = int(ring_map.max())
    rings    = np.arange(RING_MIN, min(RING_MAX, max_ring) + 1)

    print(f"Computing autocorrelations for rings {RING_MIN}..{rings[-1]} ...")
    tau_list   = []
    ring_list  = []
    r2_list    = []

    for r in rings:
        mask = ring_map == r              # shape (N, N) bool
        n_modes = mask.sum()
        if n_modes < MIN_MODES:
            continue

        # Aggregate autocorrelation over all modes in this ring
        C_sum = np.zeros(MAX_LAG + 1)
        count = 0
        for ix in range(N):
            for iy in range(N):
                if mask[ix, iy]:
                    sig = F[:, ix, iy]       # (N_t,) complex time series for mode (ix,iy)
                    C_sum += normalized_autocorrelation(sig, MAX_LAG)
                    count += 1
        if count == 0:
            continue
        C_avg = C_sum / count

        tau, r2 = fit_tau(C_avg, DT_EXPORT)
        if np.isnan(tau) or tau <= 0:
            continue

        tau_list.append(tau)
        ring_list.append(r)
        r2_list.append(r2)

    if len(tau_list) < 5:
        print("ERROR: Too few valid rings. Check that timeSeriesInterval and timeSeriesFrames"
              " give enough temporal coverage relative to tau0.")
        return

    tau_arr  = np.array(tau_list)
    ring_arr = np.array(ring_list, dtype=float)
    r2_arr   = np.array(r2_list)

    # Log-log linear regression: log(tau) = -z * log(|k|) + log(tau0)
    log_k   = np.log(ring_arr)
    log_tau = np.log(tau_arr)
    slope, intercept, r_value, p_value, std_err = linregress(log_k, log_tau)
    z_measured   = -slope
    tau0_measured = np.exp(intercept)

    print(f"\n=== Space-Time Anisotropy Results ===")
    print(f"  z  measured : {z_measured:.3f}   (theory: {Z_THEORY})")
    print(f"  tau0 measured: {tau0_measured:.2f}s  (theory: {TAU0_THEORY}s)")
    print(f"  R² of log-log fit: {r_value**2:.4f}")
    print(f"  Mean R² of individual tau fits: {r2_arr.mean():.3f}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: tau(k) log-log plot
    ax = axes[0]
    sc = ax.scatter(ring_arr, tau_arr, c=r2_arr, cmap='viridis', vmin=0, vmax=1,
                    s=20, alpha=0.8, label='Measured τ(k)')
    plt.colorbar(sc, ax=ax, label='R² of exp fit')
    k_fit = np.logspace(np.log10(ring_arr.min()), np.log10(ring_arr.max()), 200)
    ax.plot(k_fit, tau0_measured * k_fit**(-z_measured), 'r-',
            label=f'Fit: τ = {tau0_measured:.2f}|k|^(−{z_measured:.3f})', lw=2)
    ax.plot(k_fit, TAU0_THEORY * k_fit**(-Z_THEORY), 'k--',
            label=f'Theory: τ = {TAU0_THEORY}|k|^(−{Z_THEORY})', lw=1.5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Wavenumber |k|')
    ax.set_ylabel('Correlation time τ (s)')
    ax.set_title(f'Space-Time Anisotropy: τ(k) = τ₀·|k|^(−z)\nz={z_measured:.3f} (theory {Z_THEORY}), R²={r_value**2:.3f}')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)

    # Right: example autocorrelations for a few representative rings
    ax2 = axes[1]
    sample_rings = [int(r) for r in np.percentile(ring_arr, [10, 30, 60, 85])]
    lags_t = np.arange(MAX_LAG + 1) * DT_EXPORT
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(sample_rings)))
    for sr, col in zip(sample_rings, colors):
        idx = np.argmin(np.abs(ring_arr - sr))
        r_actual = int(ring_arr[idx])
        mask = ring_map == r_actual
        C_sum = np.zeros(MAX_LAG + 1)
        count = 0
        for ix in range(N):
            for iy in range(N):
                if mask[ix, iy]:
                    sig = F[:, ix, iy]
                    C_sum += normalized_autocorrelation(sig, MAX_LAG)
                    count += 1
        if count == 0:
            continue
        C_avg = C_sum / count
        tau_r = tau_arr[idx]
        ax2.plot(lags_t, C_avg, color=col, alpha=0.7, label=f'|k|={r_actual}, τ={tau_r:.2f}s')
        ax2.plot(lags_t, np.exp(-lags_t / tau_r), color=col, ls='--', lw=1.5)
    ax2.axhline(0, color='k', lw=0.5)
    ax2.set_xlabel('Time lag Δt (s)')
    ax2.set_ylabel('Normalized autocorrelation C(Δt)')
    ax2.set_title('Autocorrelations (solid) vs exp fit (dashed)\nfor representative k-rings')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "Anisotropy_Verification.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
