# WarpDrives

A Python library for simulating and visualizing General Relativistic warp bubble spacetimes.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11+-green.svg)

## Overview

WarpDrives provides tools for exploring the mathematics of warp drive spacetimes, including:

- **Alcubierre (1994)**: Classic warp bubble with shift vector construction
- **Natário (2002)**: Divergence-free shift vector (expansion-free)
- **Van Den Broeck (1999)**: Pocket modification for energy reduction
- **White toroidal**: Heuristic toroidal energy distribution [ASSUMPTION]
- **Bobrick & Martire (2021)**: Physical warp drives classification
- **Lentz (2021)**: Soliton warp drive [ASSUMPTION]

**Note**: This is a mathematical simulation tool. No claims about physical feasibility are made.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/WarpDrives.git
cd WarpDrives

# Install with pip
pip install -e .

# Or with development dependencies
pip install -e ".[dev,notebooks]"
```

## Quick Start

### Python API

```python
from warpbubblesim.metrics import AlcubierreMetric
from warpbubblesim.gr import compute_einstein, compute_energy_density
from warpbubblesim.viz import plot_energy_density
import numpy as np

# Create an Alcubierre warp bubble
metric = AlcubierreMetric(v0=1.0, R=1.0, sigma=8.0)

# Get the metric tensor at a point
g = metric.metric(t=0, x=0, y=0, z=0)
print("Metric at center:", g)

# Compute energy density
metric_func = metric.get_metric_func()
coords = np.array([0, 1, 0.5, 0])  # In the wall region
rho = compute_energy_density(metric_func, coords)
print(f"Energy density: {rho:.4e}")  # Will be negative!

# Visualize
fig, ax = plot_energy_density(metric)
fig.savefig("energy_density.png")
```

### Command Line

```bash
# List available metrics
warpsim list-metrics

# Render field visualizations
warpsim render --metric alcubierre --scenario scenarios/alcubierre_demo.yaml --output out/

# Create geodesic animation
warpsim geodesics --metric alcubierre --output out/geodesics.mp4

# Parameter sweep
warpsim sweep --metric alcubierre --param v0 --values 0.1,0.5,1.0,2.0
```

## Examples

### 1. Compare Different Metrics

```python
from warpbubblesim.metrics import (
    AlcubierreMetric, NatarioMetric, BobrickMartireMetric
)
from warpbubblesim.viz import plot_multiple_fields

metrics = [
    AlcubierreMetric(v0=1.0),
    NatarioMetric(v0=1.0),
    BobrickMartireMetric(v0=0.5),
]

for metric in metrics:
    fig = plot_multiple_fields(metric, save_path=f"out/{metric.citation}.png")
```

### 2. Integrate Geodesics

```python
from warpbubblesim.metrics import AlcubierreMetric
from warpbubblesim.gr.geodesics import integrate_geodesic
import numpy as np

metric = AlcubierreMetric(v0=1.0)
metric_func = metric.get_metric_func()

# Initial conditions
x0 = np.array([0.0, -3.0, 0.0, 0.0])  # Start outside bubble
u0 = np.array([1.0, 0.0, 0.0, 0.0])   # Initially at rest

# Integrate
result = integrate_geodesic(metric_func, x0, u0, (0, 10))
print(f"Final position: {result['coords'][-1]}")
```

### 3. Check Energy Conditions

```python
from warpbubblesim.metrics import AlcubierreMetric
from warpbubblesim.gr.conditions import check_energy_conditions
import numpy as np

metric = AlcubierreMetric(v0=1.0, R=1.0)
metric_func = metric.get_metric_func()

# Check in the bubble wall
coords = np.array([0, 1.0, 0.3, 0])
conditions = check_energy_conditions(metric_func, coords)

for name, (satisfied, value) in conditions.items():
    status = "✓" if satisfied else "✗"
    print(f"{name}: {status} (value: {value:.2e})")
```

## Output Examples

Running `make demo` produces:

- `out/alcubierre_fields.png` - Combined field visualization
- `out/alcubierre_geodesics.mp4` - Animated geodesics
- `out/alcubierre_grid_distortion.gif` - Grid distortion animation
- `out/natario_fields.png` - Natário metric fields
- `out/bobrick_martire_conditions.png` - Energy condition map
- `out/lentz_fields.png` - Lentz soliton fields

## Conventions

- **Metric signature**: (-,+,+,+)
- **Index ordering**: (t,x,y,z) = (0,1,2,3)
- **Units**: G = c = 1 (geometric units)
- **Einstein equations**: G_{μν} = 8π T_{μν}, so T_{μν} = G_{μν}/(8π)

## Documentation

- [Theory](docs/theory.md) - Mathematical foundations and equations
- [Metrics](docs/metrics.md) - Detailed metric descriptions
- [Numerics](docs/numerics.md) - Numerical methods
- [Visualizations](docs/visualizations.md) - Visualization guide

## Testing

```bash
# Run all tests
make test

# Run specific test file
pytest tests/test_minkowski_limit.py -v
```

## Project Structure

```
WarpDrives/
├── warpbubblesim/
│   ├── gr/          # GR tensor computations
│   ├── metrics/     # Warp drive metric implementations
│   ├── viz/         # Visualization tools
│   └── cli/         # Command-line interface
├── tests/           # Test suite
├── notebooks/       # Jupyter notebooks
├── scenarios/       # YAML configuration files
└── docs/            # Documentation
```

## Citation

If you use this code in research, please cite:

```bibtex
@software{warpdrives,
  title = {WarpDrives: GR Warp Bubble Spacetime Simulator},
  year = {2024},
  url = {https://github.com/yourusername/WarpDrives}
}
```

And the original papers for each metric (see [docs/references.bib](docs/references.bib)).

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

This project implements metrics from:
- Alcubierre (1994)
- Natário (2002)
- Van Den Broeck (1999)
- Bobrick & Martire (2021)
- Lentz (2021)
- White (AIAA papers)

See the documentation for full references.
