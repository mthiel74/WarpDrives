# Wolfram Community post — `warpdrives.nb`

Self-contained build of a Wolfram Community post about Alcubierre / Bobrick–Martire / Lentz warp drives.  Same style as [contiguous-cartograms](https://github.com/mthiel74/Contiguous-Cartograms) and [TrafficJAms](https://github.com/mthiel74/TrafficJAms).

## What to upload to Wolfram Community (just two files)

| File | What it is | Size |
|------|------------|------|
| `warpdrives.nb` | The notebook itself.  Drop this into the Community post. | **17 MB** |
| `WarpBubbleSim.wl` | A single, self-contained Wolfram Language package containing all the GR machinery (shape functions, the six warp metrics, Christoffel/Riemann/Einstein tensors, ADM tools, energy-condition checkers, geodesics, plotting helpers).  Attach it to the post so readers can re-evaluate the Input cells locally. | ~30 KB |

Readers download both, place them in the same folder, and the Input cell on page 1 of the notebook is

```mathematica
SetDirectory[NotebookDirectory[]];
Get["WarpBubbleSim.wl"]
```

That's it — every public symbol in the notebook (`AlcubierreMetric`, `EinsteinTensor`, `CheckAllConditions`, `IntegrateGeodesic`, `PlotMetricComparison`, …) is then available without any further imports.

## Other files in this folder (build-time, not needed for the post)

The remaining files are how the notebook is *produced*; they are not required for anyone *reading* the post.

| File | Role |
|---|---|
| `build_assets.wls` | Renders 5 PNGs and 4 GIF animations into `results/`.  Pure Wolfram. |
| `build_hero_assets.wls` | Renders the cover spacecraft animations (3D + top-down) and `hero_still.png`.  Pure Wolfram, ports `generate_hero_animation.py` from the Python project. |
| `build_notebook.wls` | Assembles `warpdrives.nb` from the rendered assets, native math typesetting, and runnable Input/Output cell pairs. |
| `results/` | Pre-rendered PNGs + GIFs that get embedded into the notebook. |

## Rebuilding from scratch

```bash
cd Wolfram/community
wolframscript -file build_assets.wls          # ~3 min: 5 PNGs + 4 GIFs
wolframscript -file build_hero_assets.wls     # ~2 min: cover spacecraft animations
wolframscript -file build_notebook.wls        # ~10 s: assembles warpdrives.nb
```

## Notebook structure

15 sections, ~93 cells, **5 embedded animations** at full quality plus 6 stills:

| # | Section | Visual |
|---|---|---|
| 0 | Cover | `hero_still.png` (single 720 px frame) |
| 1 | What is a warp drive? | shape-function plot |
| 2 | The metric in ADM form | runnable code only |
| 3 | Einstein tensor in three lines | analytic ρ density-plot + 3D surface |
| 4 | Where space contracts/expands | expansion-scalar density plot |
| 5 | The bubble in motion | **animation**: bubble flying through space |
| 5b | **Putting a spacecraft inside the bubble** | **two animations**: 3D hero (cyan grid + bubble + ring + ship + engine wake + starfield + orbiting camera) and top-down (red ρ backdrop + cyan distorted grid + ship silhouette + dashed bubble + annotations) |
| 6 | The grid being warped | **animation**: grid distortion |
| 7 | Geodesics | **animation**: test particles swept up by bubble |
| 8 | Six warp drives in one figure | side-by-side comparison panel |
| 9 | Energy conditions | runnable code only |
| 10 | Implementation notes | text only |
| 11 | Take-aways | text only |
| 12 | References | text only |

## How the animations are embedded

Each GIF is loaded with `Import[..., {"GIF", "ImageList"}]`, subsampled, and wrapped in `AnimatedImage[..., AnimationRunning -> True, AnimationRepetitions -> Infinity]`.  The Wolfram Community viewer preserves these as looping `<img>` tags so the videos play directly in the browser without anyone having to evaluate the notebook.

## Package API summary

`WarpBubbleSim.wl` exports 65 public symbols across nine logical sections:

| Section | Public symbols |
|---|---|
| Shape functions | `TanhShape`, `GaussianShape`, `CompactPolynomialShape`, `SmoothStepShape`, `SechShape`, `ShapeFunction` |
| Metrics | `AlcubierreMetric`, `NatarioMetric`, `VanDenBroeckMetric`, `WhiteToroidalMetric`, `BobrickMartireMetric`, `LentzMetric`, `MetricRegistry`, `ADMToMetric`, `BubbleCenter`, `RFromCenter` |
| Tensors | `MetricInverse`, `ChristoffelSymbols`, `RiemannTensor`, `RiemannTensorLower`, `RicciTensor`, `RicciScalar`, `EinsteinTensor`, `WeylTensor`, `AllTensors`, `NumericTensors` |
| ADM | `MetricToADM`, `ADMInverse`, `ExtrinsicCurvatureFlat`, `ExpansionScalar`, `ShearTensor`, `ShiftDivergence`, `NormalVector`, `EulerianVelocity` |
| Energy | `StressEnergyTensor`, `EnergyDensity`, `EulerianEnergyDensity`, `MomentumDensity`, `PressureDecomposition`, `EnergyFlux`, `DecomposeStressEnergy`, `AlcubierreAnalyticEnergyDensity`, `AlcubierreExpansionScalar` |
| Conditions | `TimelikeVelocities`, `NullVectors`, `CheckWEC`, `CheckNEC`, `CheckSEC`, `CheckDEC`, `CheckAllConditions` |
| Invariants | `KretschmannScalar`, `WeylSquared`, `RicciSquared`, `EulerDensity4D` |
| Geodesics | `NormalizeVelocity`, `CreateInitialVelocity`, `IntegrateGeodesic`, `IntegrateGeodesicBundle` |
| Visualize | `PlotShapeFunctions`, `PlotEnergyDensity2D`, `PlotExpansionScalar2D`, `PlotShiftField`, `PlotMetricComparison`, `PlotGridDistortion` |

All symbols live in the single `WarpBubbleSim`` context.
