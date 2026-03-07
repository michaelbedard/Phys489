import numpy as np
import pyshtools as pysh
import matplotlib.pyplot as plt
from scipy.stats import linregress

# --- PARAMETERS ---
H_target = 0.5
C1_target = 0.1

# Theoretical Beta for Multiplicative Cascade
# beta = 1 + 2H - 2C1
beta_theory = 1.0 + 2.0 * H_target - 2.0 * C1_target
print(f"Theoretical Slope: {beta_theory}")

# --- 1. LOAD DATA ---
print("Loading CSV...")
data = np.loadtxt("AtmosphereData.csv", delimiter=",")
grid = pysh.SHGrid.from_array(data)
coeffs = grid.expand()
power_per_l = coeffs.spectrum()
degrees = np.arange(len(power_per_l))

# --- 2. DEFINE FITTING RANGE ---
# CRITICAL: We only fit the 'straight' part of the graph.
# We skip the first few (earth size) and the last few (resolution blur).
fit_start = 8
fit_end = 60    # Stop before the "drop off" seen in your plot around l=100

x_fit = degrees[fit_start:fit_end]
y_fit = power_per_l[fit_start:fit_end]

# --- 3. LINEAR REGRESSION (Log-Log) ---
# We fit log(y) = slope * log(x) + intercept
log_x = np.log(x_fit)
log_y = np.log(y_fit)

slope, intercept, r_value, p_value, std_err = linregress(log_x, log_y)

# The 'slope' here is negative (e.g., -1.7).
# We usually talk about beta as positive, so beta_measured = -slope.
beta_measured = -slope

print(f"Measured Slope (Beta): {beta_measured:.4f}")
print(f"R-squared: {r_value**2:.4f}")

# --- 4. PLOTTING ---
plt.figure(figsize=(10, 6))

# Plot All Data
start_plot = 4
end_plot = 200
plt.loglog(degrees[start_plot:end_plot], power_per_l[start_plot:end_plot],
           'b-', alpha=0.5, label='Raw Spectrum Data')

# Plot the Regression Line
# y = exp(intercept) * x^(slope)
reg_line = np.exp(intercept) * x_fit**(slope)
plt.loglog(x_fit, reg_line, 'k--', linewidth=2,
           label=f'Regression (Fit l={fit_start}-{fit_end})\nSlope $\\beta$ = {beta_measured:.2f}')

# Plot Theoretical Slope (Shifted to match regression height for comparison)
# We anchor it to the middle of the regression line
mid_x = x_fit[len(x_fit)//2]
mid_y = np.exp(intercept) * mid_x**(slope)
theory_line = mid_y * (x_fit / mid_x)**(-beta_theory)

plt.loglog(x_fit, theory_line, 'r:', linewidth=2,
           label=f'Theory ($\\beta$ = {beta_theory:.2f})')

plt.xlabel('Spherical Harmonic Degree (l)')
plt.ylabel('Power Spectrum E(l)')
plt.title(f'Verification: Measured $\\beta={beta_measured:.2f}$ vs Theory $\\beta={beta_theory:.2f}$')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.5)

plt.savefig("Spectrum_Regression.png")
plt.show()