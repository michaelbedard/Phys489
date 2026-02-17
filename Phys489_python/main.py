import numpy as np
import pyshtools as pysh
import matplotlib.pyplot as plt

# 1. Load the Data
print("Loading CSV data... (This might take a moment)")
# Load the matrix we exported from Unity
data = np.loadtxt("AtmosphereData.csv", delimiter=",")

print(f"Data Loaded. Shape: {data.shape}")

# 2. Convert to Spherical Harmonic Coefficients
# We tell pyshtools this data is on a regular grid (Equirectangular)
grid = pysh.SHGrid.from_array(data)

# Expand into spherical harmonics
# This calculates the coefficients for every frequency 'l' (degree)
coeffs = grid.expand()

# 3. Calculate Power Spectrum
# spectrum() returns the power per degree 'l'
power_per_l = coeffs.spectrum()
degrees = np.arange(len(power_per_l))

# 4. Plotting (The Verification)
plt.figure(figsize=(10, 6))

# We skip l=0 (mean value) as it dominates the plot
start_l = 5
end_l = 200 # We stop before the noise floor/aliasing at high l

x = degrees[start_l:end_l]
y = power_per_l[start_l:end_l]

# Plot the Data
plt.loglog(x, y, label='Simulation Spectrum', color='blue', linewidth=2)

# 5. Plot the Theoretical Slope
# Theory: E(l) ~ l ^ -beta
# beta = 1 + 2H
H_target = 0.4
beta = 1.0 + 2.0 * H_target

# Create a reference line that starts at the same height as our data
# y = C * x^-beta  =>  log(y) = -beta * log(x) + C
reference_line = y[0] * (x / x[0])**(-beta)

plt.loglog(x, reference_line, 'r--', label=f'Theory (H={H_target}, $\\beta$={beta})')

plt.xlabel('Spherical Harmonic Degree (l)')
plt.ylabel('Power Spectrum E(l)')
plt.title(f'Multifractal Verification (Target H={H_target})')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.5)

plt.savefig("Spectrum_Verification.png")
plt.show()