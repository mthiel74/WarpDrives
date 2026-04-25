# WarpBubbleSim — Wolfram Language port

This directory contains a Wolfram Language port of the Python
`warpbubblesim` package. The Python sources at `../warpbubblesim/` remain
the reference implementation; the Wolfram packages here are an
idiomatic translation that takes advantage of symbolic differentiation,
`NDSolve`, and Wolfram's built-in tensor operations.

## Layout

| File | Wolfram package | Translates from |
|------|-----------------|-----------------|
| `WarpBubbleSim.wl` | `WarpBubbleSim`` (loader) | — |
| `ShapeFunctions.wl` | `WarpBubbleSim`ShapeFunctions`` | `metrics/base.py` (shape funcs) |
| `Metrics.wl` | `WarpBubbleSim`Metrics`` | `metrics/{alcubierre,natario,vdbroek,white_toroidal,bobrick_martire,lentz}.py` |
| `Tensors.wl` | `WarpBubbleSim`Tensors`` | `gr/tensors.py` |
| `ADM.wl` | `WarpBubbleSim`ADM`` | `gr/adm.py` |
| `Energy.wl` | `WarpBubbleSim`Energy`` | `gr/energy.py` |
| `Conditions.wl` | `WarpBubbleSim`Conditions`` | `gr/conditions.py` |
| `Invariants.wl` | `WarpBubbleSim`Invariants`` | `gr/invariants.py` |
| `Geodesics.wl` | `WarpBubbleSim`Geodesics`` | `gr/geodesics.py` |
| `Visualize.wl` | `WarpBubbleSim`Visualize`` | `viz/fields2d.py` |
| `Tests.wls` | — | `tests/test_minkowski_limit.py` |
| `examples/Demo.wls` | — | `notebooks/00_quickstart.ipynb` |
| `examples/Geodesics.wls` | — | `notebooks/01_alcubierre.ipynb` |

## Conventions

* Metric signature `(-, +, +, +)`.
* Coordinate ordering `{t, x, y, z}` (1-indexed in Wolfram, so `g[[1,1]]` is `g_{tt}`).
* Geometric units `G = c = 1`.
* Christoffel symbols and curvature tensors are derived **symbolically** with `D[]`,
  not finite differences. This means a single call returns expressions valid at any
  point in the spacetime, with the trade-off that `Simplify` can be expensive for
  the more complicated metrics.

## Quick start

```mathematica
Get["/path/to/WarpDrives/Wolfram/WarpBubbleSim.wl"]

(* Symbolic Alcubierre metric with v=1, R=1, sigma=8 *)
g = AlcubierreMetric[{1, 1, 8}, {t, x, y, z}];
MatrixForm @ g

(* Einstein tensor at a wall point *)
N @ Chop @ (EinsteinTensor[g, {t, x, y, z}] /. {t -> 0, x -> 1.0, y -> 0.3, z -> 0})

(* Closed-form Eulerian energy density *)
AlcubierreAnalyticEnergyDensity[{1, 1, 8}, {0, 1.0, 0.3, 0}] // N

(* Energy conditions *)
CheckAllConditions[g, {t, x, y, z}] /. {t -> 0, x -> 1.0, y -> 0.3, z -> 0}

(* Geodesic of a static observer being swept up by the bubble *)
sol = IntegrateGeodesic[
  Function[{tt, xx, yy, zz}, AlcubierreMetric[{1, 1, 8}, {tt, xx, yy, zz}]],
  {0., -3., 0.3, 0.},  (* start (t,x,y,z) *)
  {1., 0., 0., 0.},    (* initial 4-velocity *)
  {0, 8}];             (* affine parameter range *)
ParametricPlot[Evaluate@sol["Coords"][l][[{1, 2}]], {l, 0, 8}]
```

Run the demo and tests from a shell:

```bash
cd Wolfram
wolframscript -file Tests.wls
wolframscript -file examples/Demo.wls
```

## Differences from the Python port

* **Symbolic vs numerical** — most tensors are computed once symbolically, then
  evaluated at points. The Python port uses finite differences for everything.
* **Plotting** — the Wolfram `Visualize` package only ports a small subset
  (shape functions, Alcubierre energy density and expansion scalar, metric
  comparison grid, grid distortion preview). The animation/3-D pipelines from
  `viz/animations.py` and `viz/fields3d.py` are not ported; use the Python
  notebooks for those, or extend `Visualize.wl` with `Animate` / `ListPointPlot3D`.
* **Energy-condition sampling** — uses `RandomVariate[NormalDistribution[]]` to
  draw test observers, the same idea as the Python `np.random.randn` approach.

## Status

Tests cover the Minkowski limit, the determinant invariant of the Alcubierre
metric, the ADM round-trip, shape-function limits, and the sign of the wall
energy density. They all pass under WolframScript 1.13.0 / Mathematica 14.x on
macOS arm64.
