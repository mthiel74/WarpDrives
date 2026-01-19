You are Claude Code. Create a complete, runnable GitHub repository that simulates and visualizes multiple GR “warp bubble” spacetimes (Alcubierre + major variants) and produces animations of spacetime distortion, geodesics, curvature, and stress–energy/energy-condition diagnostics.

Repository name: WarpBubbleSim

High-level goals
1) Implement a modular GR simulation engine for user-defined 3+1 (ADM) metrics and full 4D metrics.
2) Provide metric “plugins” for the main warp-drive families:
   A. Alcubierre (1994) warp bubble (classic ADM shift construction).
   B. Natário (2002) class: divergence-free shift vector examples (“expansion-free” flow).
   C. Van Den Broeck (1999) “pocket” modification of Alcubierre (energy reduction via geometry).
   D. White-style toroidal/modified shaping (toroidal energy distribution heuristic; clearly label assumptions).
   E. Bobrick & Martire (2021) “physical warp drives” class (at minimum: one explicit subluminal positive-energy example from their paper; plus their general classification utilities).
   F. Lentz (2021) soliton warp drive: implement one explicit configuration from the paper in a reproducible parameterization. If any detail is ambiguous, explicitly mark it in code and docs as [ASSUMPTION] and provide a switchable placeholder with citations to where it should be corrected.

3) For each metric: compute
   - metric g_{μν}(t,x,y,z) and inverse
   - Christoffel Γ^μ_{αβ}
   - Riemann, Ricci, scalar curvature R
   - Einstein tensor G_{μν}
   - curvature invariants (e.g., Kretschmann K = R_{αβγδ}R^{αβγδ})
   - implied stress-energy T_{μν} = (1/8π) G_{μν} (units G=c=1, signature (-,+,+,+) unless otherwise stated)
   - energy density as measured by chosen observers (Eulerian observers in ADM, and/or ship comoving observers)
   - energy condition checks: WEC, NEC, SEC, DEC (sampled over a grid)

4) Integrate geodesics (timelike and null) through each spacetime and visualize:
   - worldlines in (t,x) diagrams for 2D slices
   - 3D trajectories in space for fixed initial conditions
   - bundles of null rays to show lensing / horizon-like behavior
   - causal cones (local light cones) along selected worldlines

5) Visualize “space distortion in front/behind” the craft:
   - Use the ADM picture: show expansion/contraction patterns via extrinsic curvature K_{ij} and its trace K, and/or expansion scalar of Eulerian congruence.
   - Provide a clear visualization method: draw a coordinate grid and map it to proper distances on constant-t hypersurfaces using the induced 3-metric γ_{ij}.
   - Provide heatmaps/isosurfaces of |K|, energy density ρ, and invariants.

6) Produce exportable animations:
   - mp4 and gif outputs
   - reproducible via CLI commands
   - parameter sweeps (bubble speed, radius, wall thickness) producing a folder of outputs

Constraints / engineering requirements
- Language: Python 3.11+.
- Use a modern scientific stack:
  - numpy, scipy
  - matplotlib (for 2D and animations)
  - pyvista or plotly for 3D (choose one and keep it consistent)
  - jax OR autograd/sympy for derivatives. Prefer JAX for automatic differentiation + speed, with a fallback finite-difference backend.
- Avoid “toy” stubs: implement complete working code with tests and examples.
- Provide good performance: support at least 128×128×1 2D grids at interactive speeds; optionally accelerate with numba/jax jit.
- Reproducibility:
  - pyproject.toml with pinned minimal dependencies
  - `make setup`, `make test`, `make demo`
  - GitHub Actions CI running tests and a short demo render (small resolution) on push.

Repository structure (create exactly)
WarpBubbleSim/
  README.md
  LICENSE
  pyproject.toml
  Makefile
  docs/
    theory.md
    metrics.md
    numerics.md
    visualizations.md
    references.bib
  notebooks/
    00_quickstart.ipynb
    01_alcubierre.ipynb
    02_natario.ipynb
    03_vdbroek.ipynb
    04_white_toroidal.ipynb
    05_bobrick_martire.ipynb
    06_lentz.ipynb
  warpbubblesim/
    __init__.py
    config.py
    utils/
      units.py
      grids.py
      io.py
    gr/
      signature.py
      adm.py
      tensors.py
      invariants.py
      energy.py
      conditions.py
      observers.py
      geodesics.py
      raybundle.py
    metrics/
      base.py
      alcubierre.py
      natario.py
      vdbroek.py
      white_toroidal.py
      bobrick_martire.py
      lentz.py
    viz/
      fields2d.py
      fields3d.py
      spacetime_diagrams.py
      animations.py
    cli/
      __init__.py
      main.py
  tests/
    test_minkowski_limit.py
    test_alcubierre_energy_density_sign.py
    test_christoffel_symmetry.py
    test_geodesic_normalization.py
    test_energy_conditions.py

Scientific correctness requirements
- Use explicit conventions everywhere:
  - metric signature
  - units (G=c=1 default; include conversion helpers)
  - index ordering (t,x,y,z) = (0,1,2,3)
- Validate against known limits:
  1) Minkowski: v=0 or f=0 ⇒ flat space: Riemann=0, Einstein=0.
  2) Alcubierre: reproduce the known negative-energy density pattern concentrated in the bubble wall (for Eulerian observers).
- Provide doc sections with the actual equations used (not just prose). Put equations in docs/theory.md with LaTeX.

Metric implementation details (minimum viable equations to implement)
A) Alcubierre metric (ADM form)
- Lapse α = 1
- Spatial metric γ_{ij} = δ_{ij}
- Shift β^i = (-v_s(t) f(r_s), 0, 0)  [choose sign consistent with ds^2 = -dt^2 + (dx - v f dt)^2 + dy^2 + dz^2]
- Bubble center x_s(t) with v_s(t)=dx_s/dt
- r_s = sqrt((x - x_s(t))^2 + y^2 + z^2)
- Provide multiple shape functions:
  - Alcubierre smooth top-hat (tanh form)
  - Gaussian wall
  - compact-support polynomial (C^2)

B) Natário class (divergence-free shift)
- Implement at least one explicit divergence-free β field that yields a moving bubble-like region.
- Enforce ∇·β = 0 on spatial slice (numerically check and report error).
- Keep α=1 and γ_{ij}=δ_{ij} initially for clarity, unless the paper requires otherwise.

C) Van Den Broeck pocket geometry
- Implement the “pocket inside bubble” mapping per the paper: a small internal volume with a large external radius.
- If the exact formula is complex, implement it with a clearly documented parameterization and cite the equation numbers.

D) White toroidal shaping (heuristic)
- Implement a toroidal modulation of the Alcubierre shape:
  - Use cylindrical coordinates (ρ = sqrt(y^2+z^2)) and define a torus radius ρ0.
  - Define f as a function of (x-x_s, ρ-ρ0) to form a torus-like wall.
- Clearly label this as a heuristic/engineering variant unless the exact published metric is available; cite sources.

E) Bobrick & Martire
- Implement:
  1) a generic “warp shell” builder utility
  2) at least one explicit “positive-energy subluminal” example metric from their paper
- Provide a switch to compute required shell stress-energy and show WEC/NEC compliance (numerically sampled).

F) Lentz soliton
- Implement one explicit configuration from the paper using Einstein–Maxwell–plasma theory assumptions.
- If fields (EM/plasma) are part of the construction, include them as separate modules and show how they source T_{μν}.
- If the paper provides metric functions in a particular coordinate system, implement that system and provide conversion utilities for visualization.

Core GR numerics
- tensors.py must compute Γ^μ_{αβ} from g_{μν} using:
  Γ^μ_{αβ} = 1/2 g^{μν}(∂_α g_{βν} + ∂_β g_{αν} - ∂_ν g_{αβ})
- Riemann:
  R^ρ_{ σμν} = ∂_μ Γ^ρ_{νσ} - ∂_ν Γ^ρ_{μσ} + Γ^ρ_{μλ}Γ^λ_{νσ} - Γ^ρ_{νλ}Γ^λ_{μσ}
- Ricci R_{σν} = R^ρ_{ σρν}, scalar R = g^{σν}R_{σν}
- Einstein G_{μν} = R_{μν} - 1/2 g_{μν} R
- Provide both AD (JAX) and finite-difference derivative backends with the same API.

Geodesic integration
- Implement first-order system with state (x^μ, u^μ):
  dx^μ/dλ = u^μ
  du^μ/dλ = - Γ^μ_{αβ} u^α u^β
- For timelike geodesics, enforce normalization g_{μν}u^μ u^ν = -1 at initialization and monitor drift.
- Provide optional projection step to renormalize u^μ periodically (documented as a numerical technique).
- Use scipy.integrate.solve_ivp with RK45 and optionally DOP853.

Visualization deliverables (must all exist)
1) 2D slice viewer: x–y plane at z=0 with time t fixed
   - show f(r), energy density ρ, trace K, Kretschmann
   - overlay geodesics
2) Spacetime diagram: plot x(t) for multiple worldlines and null rays; show bubble center x_s(t).
3) 3D isosurfaces: energy density and curvature invariants around the bubble.
4) “Grid distortion” animation: show a square grid and how proper distances change, using γ_{ij}.

CLI interface
- `python -m warpbubblesim.cli.main --help` must work.
- Commands:
  - `warpsim list-metrics`
  - `warpsim render --metric alcubierre --scenario scenarios/alcubierre_demo.yaml`
  - `warpsim geodesics --metric alcubierre --output out/alcubierre_geodesics.mp4`
  - `warpsim sweep --metric alcubierre --param v0 --values 0.1,0.5,1.0,2.0`
- Provide example YAML scenarios in a `scenarios/` folder (create it).

Documentation requirements
- README.md: quickstart, 3 command examples, and explanation of what outputs are produced.
- docs/theory.md: include the explicit equations and conventions used; include a section per metric with citations.
- docs/references.bib: include BibTeX entries for:
  - Alcubierre 1994
  - Natário 2002
  - Van Den Broeck 1999
  - White (AIAA “Warp Field Mechanics 101” or closest citable public version)
  - Bobrick & Martire 2021
  - Lentz 2021
  - Any critique papers used for horizons/radiation/causality issues (optional but recommended)

Testing
- Implement tests that run fast (<30s in CI):
  - Minkowski limit: random points yield near-zero curvature.
  - Alcubierre energy density: show negative region exists in wall for standard parameters (sign test, not exact magnitude).
  - Christoffel symmetry in lower indices.
  - Geodesic normalization drift bounded.
  - Energy conditions evaluation returns expected booleans for Minkowski.

Output examples (must be produced by `make demo`)
- out/alcubierre_fields.png
- out/alcubierre_geodesics.mp4
- out/alcubierre_grid_distortion.gif
- out/natario_fields.png
- out/bobrick_martire_conditions.png
- out/lentz_fields.png  (even low-res)

Important: scientific labeling
- If you must approximate or invent a parameterization (especially for “White toroidal” or any unclear part of Lentz), label it explicitly in docs and code as [ASSUMPTION] and explain what exact equation from the literature should replace it.
- Do not claim physical feasibility. Keep the project as “mathematical spacetime simulation/visualization”.

Implementation plan
1) Set up project skeleton + CI + CLI.
2) Implement GR tensor engine with AD backend + finite-difference fallback.
3) Implement Alcubierre metric plugin + validations + first visuals.
4) Add geodesic integrator + null ray bundles.
5) Add remaining metric plugins incrementally with notebooks.
6) Finalize docs + demos.

Deliver the full repository content: all files created, complete code, no placeholders. Use clear comments and minimal but sufficient abstractions.
