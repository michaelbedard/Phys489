"""
spectral_tools.py
=================
Shared spectral and multifractal analysis primitives for the PHYS 489 UM
verification pipeline.  Contains no top-level side effects and defines no
simulation-specific parameters — all UM constants (H, C1, α, fit ranges)
are supplied by the calling script.
"""

import numpy as np
import pyshtools as pysh
from scipy.stats   import linregress, skew, kurtosis
from scipy.optimize import curve_fit


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_field(csv_path: str) -> np.ndarray:
    """Load a raw field CSV and return a 2-D float array."""
    return np.loadtxt(csv_path, delimiter=",")


# ---------------------------------------------------------------------------
# SPECTRAL ANALYSIS
# ---------------------------------------------------------------------------

def compute_spectrum(data: np.ndarray):
    """
    Expand a 2-D equirectangular field into spherical harmonics.

    Parameters
    ----------
    data : np.ndarray  shape (nlat, nlon) — raw scalar field

    Returns
    -------
    degrees      : np.ndarray  [0, 1, …, lmax]
    power_per_l  : np.ndarray  isotropic power E(l)
    """
    grid        = pysh.SHGrid.from_array(data)
    coeffs      = grid.expand()
    power_per_l = coeffs.spectrum()
    degrees     = np.arange(len(power_per_l))
    return degrees, power_per_l


def fit_spectrum(degrees, power_per_l, fit_start: int, fit_end: int):
    """
    Log-log linear regression over degrees[fit_start:fit_end].

    Returns
    -------
    beta_measured : float        −slope of log E(l) vs log l
    r_squared     : float
    x_fit         : np.ndarray   l values used for the fit
    reg_line      : np.ndarray   regression values at x_fit
    """
    x = degrees[fit_start:fit_end]
    y = power_per_l[fit_start:fit_end]
    slope, intercept, r, _, _ = linregress(np.log(x), np.log(y))
    return -slope, r**2, x, np.exp(intercept) * x**slope


def theory_line(x_fit, reg_line, beta: float):
    """
    Theoretical power-law slope anchored to the midpoint of the regression line.

    Returns array of theoretical E(l) values at x_fit positions.
    """
    mid = len(x_fit) // 2
    return reg_line[mid] * (x_fit / x_fit[mid]) ** (-beta)


# ---------------------------------------------------------------------------
# STRUCTURE FUNCTIONS  (verifies C1 and α on the dressed field)
#
#   S_q(r) = <|f(x+r) − f(x)|^q>  ~  r^{ζ(q)}
#   ζ(q)   = q·H − K(q)
#   K(q)   = q·H − ζ(q)
# ---------------------------------------------------------------------------

def structure_functions(data: np.ndarray, q_values, r_values,
                        n_pairs: int = 50000, equatorial_belt: bool = False):
    """
    Compute S_q(r) = <|f(x+r) - f(x)|^q> for horizontal pixel lags r.
    Uses random-pair sampling to keep computation tractable.

    Parameters
    ----------
    equatorial_belt : bool
        If True, restrict sampling to rows within ±45° latitude (middle half of
        the grid).  This avoids polar bias on equirectangular projections: near
        the poles a lag of r pixels covers ~r·sin(θ)→0 physical distance,
        mixing scales and biasing ζ(q) estimates.  Default False (full grid).

    Returns
    -------
    dict : q -> np.ndarray of S_q values (one per entry in r_values)
    """
    h, w = data.shape
    rng  = np.random.default_rng(42)
    results = {q: [] for q in q_values}

    y_min = h // 4 if equatorial_belt else 0
    y_max = 3 * h // 4 if equatorial_belt else h

    for r in r_values:
        xs   = rng.integers(0, w - r, size=n_pairs)
        ys   = rng.integers(y_min, y_max, size=n_pairs)
        diff = np.abs(data[ys, xs + r] - data[ys, xs])
        for q in q_values:
            results[q].append(np.mean(diff ** q))

    return {q: np.array(v) for q, v in results.items()}


def fit_zeta(sf_results, r_values, q_values):
    """
    Fit ζ(q) from slopes of log S_q(r) vs log r.

    Returns array of ζ(q) values (one per q in q_values).
    """
    log_r = np.log(np.array(r_values, dtype=float))
    zeta  = []
    for q in q_values:
        slope, _, _, _, _ = linregress(log_r, np.log(sf_results[q]))
        zeta.append(slope)
    return np.array(zeta)


def theoretical_Kq(q, C1: float, alpha: float):
    """
    Theoretical K(q) for a Lévy-stable UM cascade:
        K(q) = C1 / (α−1) * (q^α − q)
    Reduces to C1*(q²−q)/2 at α=2 (log-normal limit).
    """
    return C1 / (alpha - 1.0) * (q**alpha - q)


# ---------------------------------------------------------------------------
# LOG-NORMALITY CHECK
# ---------------------------------------------------------------------------

def check_lognormality(data: np.ndarray):
    """
    Returns (skewness, excess_kurtosis) of log(field) for positive pixels.
    Both should be near 0 for a perfect log-normal field.
    """
    flat     = data.ravel()
    flat     = flat[flat > 0]
    log_flat = np.log(flat)
    return skew(log_flat), kurtosis(log_flat, fisher=True)
