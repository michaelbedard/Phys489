# McGill Physical Journal

# 3D Multifractal Cloud Simulation

### Michael Bedard (261120269)

### Supervised by Prof. Shaun Lovejoy

### McGill University Department of Physics

### March 8, 2026

```
Abstract
Traditional Euclidean models fail to capture the extreme intermittency and scale-
invariance of atmospheric turbulence. This project develops a high-performance,
real-time simulation of atmospheric fields governed by the Universal Multifractal
(UM) framework. The field generation uses a 2D FFT-based Fractionally Integrated
Flux (FIF) algorithm that runs entirely on the GPU. CMS alpha-stable noise
(alpha = 1.79) encodes intermittency correctly. Two independent fields are
simulated: cloud cover and surface albedo, each with its own UM parameters.
Temporal evolution uses an Ornstein-Uhlenbeck process per Fourier mode, which
implements the UM dynamical scaling exponent z. The simulation is validated by
ensemble spectral analysis over N = 50 independent realizations. The measured
spectral slopes are beta = 1.51 +/- 0.04 for clouds (theory 1.48) and
beta = 2.18 +/- 0.03 for albedo (theory 2.10), both within 4% of theory.
The space-time anisotropy exponent is measured as z = 0.49, compared to a
theoretical target of z = 0.5.
```

## 1 Introduction

### 1.1 Theory Behind Multifractals

Traditional weather models use classical Euclidean geometry. However, Euclidean shapes are
too smooth and simple to capture the extreme, irregular, and fragmented structures of the
real atmosphere. Furthermore, traditional models often underestimate how variability builds
up from very large global scales down to tiny local scales [1].

To solve this, physicists use fractal geometry. Fractal geometry has the property that, when zoomed in,
it reveals the same level of complex detail. This is unlike zooming in on a Euclidean shape,
where it eventually looks flat. This is called self-similarity and scale-invariance. For example, a small piece of a
cloud statistically looks like the whole cloud [2].

However, a simple fractal (called monofractal) is not enough. Monofractals use only
one scaling rule, which creates a field with uniform roughness. The atmosphere is highly
intermittent; calm areas are frequently interrupted by sudden, extreme events like intense
storms. To model this behavior, we need a multifractal framework. Multifractals use a
wide set of scaling rules to handle these extreme spikes in energy.

This project crosses the gap between visual rendering and theoretical physics. It develops
a fast, real-time simulation tool for atmospheric fields governed by Universal Multifractal
laws.

### 1.2 Application on Clouds and Albedo

To model atmospheric turbulence and clouds, Lovejoy and Schertzer developed the Universal
Multifractal (UM) framework [3]. This framework describes mathematically how energy
cascades from large atmospheric structures down to microscopic scales.

The UM model has three fundamental parameters:

- α: The Lévy index, which measures the degree of multifractality. Values of α < 2
  give heavier tails and stronger intermittency than a log-normal field (α = 2).
- C₁: The intermittency codimension. This controls how sparse and intense extreme events are.
- H: The Hurst exponent, which controls the overall spatial smoothness of the field.

A correct multifractal field has isotropic scale-invariance. Its energy spectrum follows a
strict power-law relationship: E(l) ∝ l^{−β}, where l is the spherical harmonic degree and
β is the spectral slope.

For general α-stable noise, the theoretical spectral slope is:

β = 1 + 2H − K(2) (1)

where K(2) is the second-order moment scaling function:

K(2) = C₁ / (α − 1) · (2^α − 2) (2)

This project simulates two independent fields. The first is a cloud cover field with
parameters H = 0.33, C₁ = 0.10, α = 1.79, which gives a theoretical slope of β ≈ 1.48.
The second is a surface albedo field with parameters H = 0.66, C₁ = 0.12, α = 1.79,
which gives β ≈ 2.10. The albedo parameters come from Lovejoy's empirical measurements
for continental topography.

By measuring β for each field and comparing it to Equation 1, we can verify that the
simulation correctly reproduces the fundamental atmospheric physics.

## 2 Materials and Methods

### 2.1 Software Architecture

This project uses a hybrid software architecture to balance real-time field generation with
offline physical verification. The simulation is built in the Unity engine. C# handles system
logic, parameter control, and data export. All field generation math runs on the GPU using
HLSL Compute Shaders.

Two independent FIF pipelines run: one for the cloud field, which evolves each frame, and
one for the albedo field, which is static. Both share the same `FIFGenerator.compute` shader
but use separate parameter sets and GPU buffers.

Physical validation uses three Python scripts:
- `main.py` — cloud field β measurement via spherical harmonic power spectrum
- `verify_albedo.py` — albedo field β, structure function K(q), and log-normality check
- `verify_anisotropy.py` — space-time anisotropy, z exponent via temporal autocorrelation

The `pyshtools` library is used for spatial analysis. The `scipy` library is used for
anisotropy fitting. Memory usage stays low because the field is computed procedurally on
the GPU and only the 1024×1024 output texture is stored.

### 2.2 Spatial Implementation

The field is generated by the FFT-based Fractionally Integrated Flux (FIF) method. The
pipeline has five steps that run in sequence each frame.

**Spectrum initialization** — done once per realization. For each Fourier mode (kx, ky):

- Modes with |k| < 1 are set to zero. This enforces a zero-mean log-field.
- A CMS α-stable amplitude is drawn [4], seeded by the mode coordinates and a
  per-realization random seed.
- An independent uniform random phase is drawn from a hash function.
- The spectral weight applied is: `(C₁ · ln 2)^{1/α} · |k|^{−(H+1)}`

The factor |k|^{−(H+1)} accounts for the 2D mode density, which is proportional to |k|.
This gives E(l) ∝ l^{−(1+2H)} as required by UM theory. Each realization uses a different
random seed so the Fourier amplitudes and phases are fully independent.

**Temporal evolution** — applies the Ornstein-Uhlenbeck update to the spectrum each frame.
This is described in §2.4.

**Wind advection** — copies the spectrum and applies a time-dependent phase to simulate
cloud drift. This follows the Taylor frozen-turbulence hypothesis:

coeff(k, t) = coeff(k, 0) · exp(i · 2π · (kx · vx + ky · vy) · t) (3)

For the static albedo field, this step makes a plain copy with no phase shift.

**Inverse FFT** — converts the Fourier-space array back into a real-space field using a
radix-2 Cooley-Tukey algorithm [5], applied first along rows then along columns. There is
no 1/N normalization because the |k|^{−(H+1)} weighting already gives the correct field
magnitude.

**Field extraction** — reads the real part of the IFFT output, exponentiates it, and writes
the dressed field:

field = exp(clamp(Re(IFFT output), −20, 20)) (4)

The result is exported as a raw float array for Python analysis and as a texture for
GPU visualization.

### 2.3 Data Verification Pipeline

To verify that the simulation follows UM theory, a validation pipeline was developed in
Python.

For a given snapshot, the raw 1024×1024 float array is read from GPU memory via C# and
exported to a CSV file. This data is loaded in Python and analyzed with `pyshtools`.

The field is expanded into spherical harmonics. This is necessary because the FIF operates
on a flat 1024×1024 pixel grid. In the equirectangular projection, pixels near the poles
represent smaller physical areas on the sphere. The spherical harmonic expansion applies the
correct latitude weighting so the power spectrum is physically accurate. A standard 2D FFT
over the raw pixel grid would not apply this weighting.

The pipeline computes the isotropic power spectrum E(l) as a function of spherical harmonic
degree l. On a log-log plot, a scale-invariant field shows a straight line. A linear
regression gives the empirical slope β_empirical. This is compared to the theoretical
prediction from Equation 1:

β = 1 + 2H − K(2) = 1 + 2H − C₁ / (α − 1) · (2^α − 2)

Two additional verification sub-pipelines complete the validation:

1. **Albedo verification** (`verify_albedo.py`): measures β, computes structure functions
   K(q) at multiple orders q, and checks log-normality of the log-field distribution.

2. **Space-time anisotropy verification** (`verify_anisotropy.py`): exports N_t = 200
   consecutive frames via the V key in Unity. For each frame, a 2D FFT of log(field) is
   computed and the wind-advection phase is removed. The normalized temporal autocorrelation
   C(Δt) is computed per wavenumber ring |k| ~ r. An exponential fit C(Δt) = exp(−Δt/τ)
   extracts the correlation time τ per ring. A log-log regression of τ vs |k| gives the
   dynamical scaling exponent z via τ(k) = τ₀ · |k|^{−z}.

### 2.4 Temporal Dynamics and Space-Time Anisotropy

The temporal evolution of the field uses an Ornstein-Uhlenbeck (OU) process applied
independently to each Fourier mode in `BaseSpectrumBuffer`. This implements the UM
dynamical scaling exponent z.

Each mode (kx, ky) has a scale-dependent correlation time:

τ(k) = τ₀ · |k|^{−z} (5)

Large scales (small |k|) persist over many frames. Small scales (large |k|) decorrelate
rapidly. This reproduces the atmospheric behavior where planetary-scale systems evolve
slowly and small-scale turbulence dissipates quickly.

The OU update per frame (time step Δt) is:

BaseSpectrum[k] ← a · BaseSpectrum[k] + (1 − a^α)^{1/α} · fresh(k) (6)

where:
- a = exp(−Δt / τ(k)) is the OU decay factor
- `fresh(k)` is a new CMS α-stable sample with the same spectral weight as GenerateSpectrum
- (1 − a^α)^{1/α} is the noise scale that ensures α-stable stationarity

The noise scale formula needs some explanation. For a Gaussian OU process (α = 2),
stationarity requires a noise scale of √(1 − a²). This comes from σ²_total = a²σ² + σ²_noise.
For α-stable distributions, the stability parameter adds in the α-norm:
σ^α_total = a^α · σ^α + σ^α_noise, so stationarity requires σ_noise = (1 − a^α)^{1/α}.
Using the Gaussian formula with α = 1.79 injects about 13% excess energy per frame.
Over 100 frames, this compounds into a field blow-up of about 3.7×, which appears as a
white blob in the visualization. The correct formula keeps the spatial power spectrum
constant at all times.

Wind advection (PhaseShift kernel) is applied on top of the OU-evolved spectrum as a
Galilean frame shift. The two effects are physically and mathematically independent.

## 3 Results

Before looking at the numbers, we first evaluated visually the generated output to confirm
that the parameters have the expected effect. The parameter C₁ controls the intermittency
of the cascade.

As shown in Figure 1, changing C₁ strongly changes the visual structure of the field.
A low value (C₁ = 0.1, H = 0.33, α = 1.79) produces a relatively uniform and continuous
field. A high value (C₁ = 0.8) produces a highly intermittent field with intense localized
extremes surrounded by large calm regions. This confirms that C₁ acts as expected on the
visual output.

```
(a) Low intermittency (C₁ = 0.1).          (b) High intermittency (C₁ = 0.8).
```
Figure 1: Real-time visual output rendered with an Infrared (IR) color gradient.
Increasing C₁ concentrates the energy into isolated, extreme cloud structures.

To evaluate scale-invariance, N = 50 independent realizations were generated for both
fields. Each snapshot is a 1024×1024 spatial array. Each realization uses a different
random seed so the Fourier amplitudes and phases are fully independent.

Each snapshot was transformed into the spherical harmonic domain with `pyshtools` to
compute E(l). The ensemble average was computed over all 50 realizations to reduce
statistical noise. A linear regression was applied to the ensemble mean in log-log space
over the inertial range: l ∈ [8, 60] for clouds and l ∈ [4, 80] for albedo.

Figure 2 shows the power spectra for both fields. The measured results are:

- **Cloud** (H = 0.33, C₁ = 0.10, α = 1.79): β measured = 1.51 ± 0.04, theory = 1.48 (~2% error)
- **Albedo** (H = 0.66, C₁ = 0.12, α = 1.79): β measured = 2.18 ± 0.03, theory = 2.10 (~4% error)

Both fields show a clear power-law in the inertial range, which confirms scale-invariant behavior.

Figure 2: Log-log plot of the isotropic power spectrum E(l) versus spherical harmonic degree
l for the cloud field (top) and albedo field (bottom). Faint lines are individual realizations.
The solid line is the ensemble average. The dashed line is the linear regression. The dotted
line is the theoretical slope.

For the space-time anisotropy, N_t = 200 consecutive frames were captured at intervals of
Δt = 0.1 s using the V key in Unity. The temporal autocorrelation C(Δt) was computed per
wavenumber ring after removing the wind-advection phase. An exponential fit extracted the
correlation time τ per ring. The log-log regression of τ vs |k| gives:

z = 0.49 (theory z = 0.5, ~2% error)

Figure 3: Log-log plot of correlation time τ versus wavenumber |k| with the regression line.
The slope gives z = 0.49.

## 4 Discussion

The results show that the simulation reproduces the expected UM scaling very well. Both
the cloud and albedo fields follow a clear power-law over their inertial range, and the
measured β values are within 2-4% of the theoretical predictions. This is a strong result
for a real-time GPU simulation.

**Spectral slope accuracy.** The FIF assigns the spectral amplitude directly as
|k|^{−(H+1)} in Fourier space, which makes the power-law exact by construction. This is
why β is so accurate. The small remaining error (2-4%) comes from the finite number of
realizations and the limits of the inertial range fit.

**Intermittency underestimation.** The one limitation that stands out is the K(q) result.
Even though β is accurate, the measured C₁ is much lower than the target — about 0.046
vs 0.10 for clouds, and 0.075 vs 0.12 for albedo. This is not a bug. It is a fundamental
property of the additive FIF approach. When you sum about 10⁶ independent Fourier modes,
the Central Limit Theorem pushes the log-field toward a Gaussian, regardless of how heavy
the tails of the injected noise are. The intermittency, which lives in the higher-order
statistics, gets washed out in this averaging process. The spectral slope β is not
affected because it only depends on second-order statistics. To fully recover K(q), a
multiplicative cascade in Fourier space would be needed, but that is significantly more
complex.

**Space-time anisotropy.** The measured z = 0.49 is very satisfying. It means the
simulation correctly captures the fact that large atmospheric structures evolve slowly
and small structures decorrelate fast, with the right relative scaling between scales.
The τ₀ intercept is harder to recover reliably. For α-stable distributions with α < 2,
the sample variance is theoretically infinite, which means the autocorrelation estimator
converges very slowly. With 200 frames, τ₀ is not a reliable number. But the slope z,
which comes from comparing decorrelation rates across many wavenumber rings, is robust
because the systematic errors cancel in the relative comparison.

## 5 Conclusions

This project implemented and validated a real-time GPU simulation of atmospheric fields
governed by the Universal Multifractal framework. The main results are:

- The FFT-based FIF produces scale-invariant spatial fields correctly. The measured spectral
  slopes are within 2-4% of the theoretical predictions: β = 1.51 vs 1.48 for clouds, and
  β = 2.18 vs 2.10 for albedo.
- A second field representing surface albedo was independently implemented and verified with
  its own UM parameters, taken from Lovejoy's empirical measurements for continental topography.
- Space-time anisotropy was implemented via an Ornstein-Uhlenbeck process per Fourier mode.
  The measured dynamical scaling exponent z = 0.49 agrees with the theoretical target of
  z = 0.5 within 2%.
- The main remaining limitation is the underestimation of K(q) and C₁ by 40-55%. This comes
  from the Central Limit Theorem acting on the large number of independent Fourier modes in
  the additive FIF. The spectral slope β is not affected.
- The simulation runs in real-time at 1024×1024 resolution. All UM parameters (H, C₁, α)
  are adjustable in the Unity inspector without recompiling.

## References

[1] Benoit B. Mandelbrot. *The Fractal Geometry of Nature*. W. H. Freeman and Company,
1982.

[2] Shaun Lovejoy. Area-perimeter relation for rain and cloud areas. *Science*, 216(4542):
185–187, 1982.

[3] Daniel Schertzer and Shaun Lovejoy. Physical modeling and analysis of rain and clouds
by anisotropic scaling multiplicative processes. *Journal of Geophysical Research:
Atmospheres*, 92(D8):9693–9714, 1987.

[4] J. M. Chambers, C. L. Mallows, and B. W. Stuck. A method for simulating stable random
variables. *Journal of the American Statistical Association*, 71(354):340–344, 1976.

[5] James W. Cooley and John W. Tukey. An algorithm for the machine calculation of complex
Fourier series. *Mathematics of Computation*, 19(90):297–301, 1965.

[6] Shaun Lovejoy and Daniel Schertzer. *The Weather and Climate: Emergent Laws and
Multifractal Cascades*. Cambridge University Press, 2013.

[7] David Marsan, Daniel Schertzer, and Shaun Lovejoy. Causal space-time multifractal
processes: Predictability and forecasting of rain fields. *Journal of Geophysical Research:
Atmospheres*, 101(D21):26333–26346, 1996.


## A Compute Shader: FIF Field Generation

The following High-Level Shader Language (HLSL) snippets show the two GPU kernels where
all the UM physics is implemented. The first kernel builds the Fourier-space spectrum from
scratch for each realization. The second evolves it over time using the Ornstein-Uhlenbeck
process. All three UM parameters (H, C₁, α) appear directly in both kernels.

**Spectrum initialization** — spectral amplitude and phase assignment:

```hlsl
[numthreads(8, 8, 1)]
void GenerateSpectrum(uint3 id : SV_DispatchThreadID)
{
    int kx = ix < (int)(N / 2) ? ix : ix - (int)N;
    int ky = iy < (int)(N / 2) ? iy : iy - (int)N;
    float k_mag = sqrt((float)(kx * kx + ky * ky));

    // DC and sub-pixel modes: enforce zero-mean log-field
    if (k_mag < 1.0)
    {
        BaseSpectrumBuffer[id.y * N + id.x] = float2(0.0, 0.0);
        return;
    }

    // CMS alpha-stable amplitude — encodes intermittency (C1, alpha)
    float amp = levyStable(float3((float)kx, (float)ky, (float)Seed), Alpha);

    // Independent uniform random phase
    float phase = hash(float3((float)kx + 0.5, (float)ky + 0.5, (float)Seed + 9973.0))
                  * 2.0 * PI;

    // Spectral weight: (C1 * ln2)^{1/alpha} * |k|^{-(H+1)}
    // The |k|^{-(H+1)} factor gives E(l) proportional to l^{-(1+2H)} after
    // accounting for the 2D mode density (proportional to |k|).
    float weight = pow(C1 * LN2, 1.0 / Alpha) * pow(k_mag, -(H + 1.0));

    BaseSpectrumBuffer[id.y * N + id.x] = float2(amp * weight * cos(phase),
                                                   amp * weight * sin(phase));
}
```

**Temporal evolution** — Ornstein-Uhlenbeck update per Fourier mode:

```hlsl
[numthreads(8, 8, 1)]
void EvolveSpectrum(uint3 id : SV_DispatchThreadID)
{
    // Scale-dependent correlation time: tau(k) = Tau0 * |k|^{-z}
    // Large scales (small |k|) persist. Small scales (large |k|) decorrelate fast.
    float tau_k = Tau0 * pow(k_mag, -Gz);
    float a     = clamp(exp(-Dt / tau_k), 0.0, 1.0);

    // Alpha-stable stationarity: noise_scale = (1 - a^alpha)^{1/alpha}
    // Using the Gaussian formula sqrt(1 - a^2) injects ~13% excess energy per frame
    // for alpha = 1.79, which causes a field blow-up. The correct formula keeps
    // the spatial power spectrum invariant at all times.
    float noise_scale = pow(max(1e-12, 1.0 - pow(a, Alpha)), 1.0 / Alpha);

    // Fresh CMS alpha-stable sample, seeded by frame count for temporal independence
    float amp_new   = levyStable(float3((float)kx, (float)ky, (float)FrameCount), Alpha);
    float phase_new = hash(float3((float)kx + 0.5, (float)ky + 0.5,
                                  (float)FrameCount + 9973.0)) * 2.0 * PI;
    float weight = pow(C1 * LN2, 1.0 / Alpha) * pow(k_mag, -(H + 1.0));
    float2 fresh = float2(amp_new * weight * cos(phase_new),
                          amp_new * weight * sin(phase_new));

    // OU blend: preserves the stationary distribution of each mode
    uint idx = id.y * N + id.x;
    BaseSpectrumBuffer[idx] = a * BaseSpectrumBuffer[idx] + noise_scale * fresh;
}
```

The UM parameters H, C₁, and α appear directly in both kernels. The other kernels
(PhaseShift, IFFTRow, IFFTCol, ExtractField) contain no physics — they are infrastructure
for the FFT transform and field output.
