# PHYS 489 — Spatiotemporal Multifractal Planetary Temperature Simulation

## Rules for Claude

- **Never edit any file without explicit user approval first.** Always propose changes and wait for confirmation before touching code, shaders, or scripts.
- **Never compromise the physical meaning of the simulation code.** This is an academic project graded on its physics. All formulas, variable names, constants, and comments must remain physically meaningful and traceable to UM theory. Do not introduce heuristic hacks, magic numbers, or purely aesthetic shortcuts into the physics pipeline without clearly labelling them as visualization-only and getting approval.

## Project Overview

McGill University PHYS 489 research project (Student: Michael Bedard, Supervisor: Prof. Shaun Lovejoy).

Goal: Real-time interactive simulation of a 2D spherical atmospheric field governed by Universal Multifractal (UM) laws, visualized as geostationary Infrared (IR) satellite imagery.

## Monorepo Structure

```
C:\Users\courr\Phys489\
├── Phys489_unity/          ← THIS REPO (Unity C#/HLSL)
│   └── Assets/
│       ├── SimulationController.cs   — MonoBehaviour; drives compute shader each frame
│       ├── DataExporter.cs           — Press X to export RenderTexture → CSV
│       ├── CascadeShader.compute     — HLSL multiplicative cascade on GPU
│       └── PlanetMat.mat             — Material applied to sphere mesh
└── Phys489_python/         ← SIBLING directory (spectral verification)
    ├── main.py                       — Loads CSV, runs spherical harmonic power spectrum
    ├── AtmosphereData.csv            — Export target from Unity (written by DataExporter)
    ├── Spectrum_Regression.png
    └── Spectrum_Verification.png
```

## Key Physics Concepts

### Universal Multifractal (UM) Parameters
- **H** (0–1): Hurst exponent — controls spectral slope / smoothness of the field. Higher H = smoother.
- **C1** (0–1): Intermittency codimension — controls sparseness/clustering of extreme events. Higher C1 = spikier.
- **alpha (α)**: Levy index — multifractality index (not yet exposed as a shader parameter; currently using log-normal approximation α=2).

### Theoretical spectral slope
```
β = 1 + 2H - 2C1      (for the multiplicative cascade implemented here)
```
The Python analysis fits E(l) ∝ l^(−β) in log-log space over spherical harmonic degrees l ∈ [8, 60] to verify scale invariance.

### Cascade Implementation (CascadeShader.compute)
- Maps UV → spherical coordinates (θ, φ) → unit sphere position
- 10-octave multiplicative log-normal cascade: accumulates `(γ_i · σ) − C1·ln2` per octave weighted by `scale^(−H)`
- Exponentiates log-sum to produce the dressed flux field `exp(log_field)`
- Gaussian noise uses deterministic Box-Muller (hash-based) — same seed always gives same field
- Time-driven drift `float3(sin(t·0.2), cos(t·0.2), t·0.1)` animates the field on the sphere
- Visualization: `displayValue = saturate(fieldIntensity · 0.2)` → sampled into IR gradient LUT
- Composited with `EarthTex` using `alpha = saturate(fieldIntensity − 0.2)`

## Data Export Workflow

1. In Unity Play Mode, press **X** to trigger `DataExporter.ExportToCsv()`
2. **Before exporting**: in `CascadeShader.compute`, uncomment the raw export line and comment the visual line:
   ```hlsl
   Result[id.xy] = float4(fieldIntensity, 0, 0, 1);  // FOR EXPORT
   // Result[id.xy] = lerp(earthColor, cloudColor, alpha);  // FOR VISUALS
   ```
3. CSV is written to `../Phys489_python/AtmosphereData.csv` (relative to `Assets/`)
4. Run `main.py` in the Python venv to verify the power spectrum

## Python Analysis Setup

```bash
cd C:\Users\courr\Phys489\Phys489_python
.venv\Scripts\activate
python main.py
```

Dependencies: `numpy`, `pyshtools`, `matplotlib`, `scipy`

Set `H_target` and `C1_target` in `main.py` to match the Unity inspector values before running.

## Unity Architecture

- **SimulationController.cs**: MonoBehaviour on the planet sphere GameObject. Owns the `RenderTexture`, `ComputeShader`, and IR gradient `Texture2D`. Calls `Dispatch()` every frame.
- **DataExporter.cs**: Separate MonoBehaviour; holds a reference to `SimulationController`. Reads `RenderTexture` back to CPU via `ReadPixels` on keypress.
- **CascadeShader.compute**: Single kernel `CSMain`, dispatched as `(resolution/8, resolution/8, 1)` thread groups with `[numthreads(8,8,1)]`.
- **PlanetMat.mat**: Standard material; `mainTexture` is set at runtime to the `RenderTexture`.

## Important Shader Notes

- `GradientTex` is a 256×1 `Texture2D` baked from the Unity `Gradient` in `SimulationController`
- `EarthTex` is optional; falls back to `Texture2D.blackTexture` if unassigned
- The shader uses `SamplerState _LinearClamp` — Unity auto-resolves this by name convention
- Shader parameters `C1`, `H`, `Time` are set via `cascadeShader.SetFloat()` every frame
- `InitGradient()` is called every `Update()` frame (useful for live color editing; disable for release)

## Current Limitations / Known Issues

- **α parameter not implemented**: The cascade uses the log-normal approximation (α=2). True α-stable Levy noise would require a different random variable generator.
- **Export mode requires manual shader edit**: Must comment/uncomment lines in `.compute` file to switch between visual and raw-data output modes.
- **No surface albedo layer yet**: The proposal describes compositing a static albedo/topography field; currently only the dynamic cloud field is generated.
- **Spectral fitting range is hardcoded** in `main.py` (`fit_start=8`, `fit_end=60`); adjust if resolution changes.