# Metrics Reference

This document provides detailed descriptions of all implemented warp drive metrics.

## Alcubierre Metric

**Class**: `AlcubierreMetric`
**Citation**: Alcubierre (1994)

The classic warp drive metric that started the field.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `v0` | 1.0 | Bubble velocity (units of c) |
| `R` | 1.0 | Bubble radius |
| `sigma` | 8.0 | Wall steepness (higher = sharper) |
| `x0` | 0.0 | Initial bubble center position |
| `shape` | 'tanh' | Shape function type |

### Shape Function Options
- `'tanh'`: Alcubierre's original (smooth, non-compact)
- `'gaussian'`: Gaussian profile (smooth, non-compact)
- `'polynomial'`: Compact support C² polynomial
- `'smoothstep'`: Smooth step function

### Example
```python
from warpbubblesim.metrics import AlcubierreMetric

# Standard superluminal bubble
metric = AlcubierreMetric(v0=2.0, R=1.0, sigma=8.0)

# Subluminal bubble
metric = AlcubierreMetric(v0=0.5, R=1.0, sigma=8.0)

# With different shape function
metric = AlcubierreMetric(v0=1.0, shape='gaussian')
```

### Special Methods
- `eulerian_energy_density_analytic(t, x, y, z)`: Analytic energy density formula
- `expansion_scalar_analytic(t, x, y, z)`: Analytic expansion scalar
- `total_energy_estimate()`: Rough total energy estimate

---

## Natário Metric

**Class**: `NatarioMetric`
**Citation**: Natário (2002)

Expansion-free warp drive with divergence-free shift vector.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `v0` | 1.0 | Bubble velocity |
| `R` | 1.0 | Bubble radius |
| `sigma` | 8.0 | Wall steepness |
| `x0` | 0.0 | Initial position |

### Key Properties
- $\nabla \cdot \vec{\beta} = 0$ (divergence-free)
- No expansion/contraction regions
- Still requires negative energy

### Verification
```python
metric = NatarioMetric(v0=1.0)
div = metric.verify_divergence_free(t=0, x=1, y=0.5, z=0)
print(f"Divergence: {div}")  # Should be near zero
```

---

## Van Den Broeck Metric

**Class**: `VanDenBroeckMetric`
**Citation**: Van Den Broeck (1999)

Pocket modification to reduce total energy requirements.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `v0` | 1.0 | Bubble velocity |
| `R_ext` | 0.1 | External radius (small) |
| `R_int` | 1.0 | Internal radius (large) |
| `B_int` | 10.0 | Internal expansion factor |
| `sigma` | 8.0 | Wall steepness |
| `sigma_B` | 5.0 | B transition steepness |

### Key Properties
- Large internal proper volume
- Small external appearance
- Energy scales with $R_{ext}$, not $R_{int}$

### Example
```python
metric = VanDenBroeckMetric(
    v0=1.0,
    R_ext=0.01,   # Microscopic external radius
    R_int=10.0,   # Large internal radius
    B_int=1000.0  # Large expansion
)

print(f"Internal volume: {metric.internal_volume()}")
print(f"Energy reduction: {metric.energy_reduction_factor()}")
```

---

## White Toroidal Metric

**Class**: `WhiteToroidalMetric`
**Citation**: White (2011) AIAA [HEURISTIC]

**⚠️ ASSUMPTION**: This is an interpretation of White's AIAA presentations. The exact published metric form may differ.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `v0` | 1.0 | Bubble velocity |
| `R_major` | 2.0 | Torus major radius |
| `R_minor` | 0.5 | Torus minor radius (tube) |
| `sigma` | 8.0 | Transition sharpness |

### Key Properties
- Toroidal energy distribution (donut-shaped)
- Energy concentrated in torus tube
- Potentially reduces total energy requirements

### Example
```python
metric = WhiteToroidalMetric(
    v0=1.0,
    R_major=2.0,  # Distance from center to tube
    R_minor=0.5   # Tube radius
)

params = metric.torus_parameters()
print(f"Volume: {params['volume']}")
print(f"Aspect ratio: {params['aspect_ratio']}")
```

---

## Bobrick-Martire Metric

**Class**: `BobrickMartireMetric`
**Citation**: Bobrick & Martire (2021)

Physical warp drives with positive-energy configurations.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `v0` | 0.5 | Bubble velocity (subluminal!) |
| `R_inner` | 1.0 | Inner shell radius |
| `R_outer` | 2.0 | Outer shell radius |
| `shell_amplitude` | 0.1 | Metric perturbation amplitude |
| `positive_energy` | True | Use positive-energy configuration |
| `sigma` | 5.0 | Transition sharpness |

### Key Requirement
**Subluminal velocity required** (`v0 < 1`) for positive-energy configurations.

### Example
```python
# Positive-energy subluminal warp drive
metric = BobrickMartireMetric(
    v0=0.1,  # Very subluminal
    positive_energy=True
)

print(f"Energy condition: {metric.energy_condition_type()}")

# Check energy requirements
energy = metric.energy_estimate()
print(f"Velocity: {energy['velocity']}")
print(f"Positive energy: {energy['positive_energy']}")
```

### Explicit Subluminal Example
```python
from warpbubblesim.metrics.bobrick_martire import BobrickMartireSubluminal

metric = BobrickMartireSubluminal(v0=0.1)
result = metric.verify_positive_energy()
print(f"All positive: {result['all_positive']}")
```

---

## Lentz Soliton Metric

**Class**: `LentzMetric`
**Citation**: Lentz (2021)

**⚠️ ASSUMPTION**: This implementation is an interpretation of Lentz's paper. Specific parameterizations may differ from the original.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `v0` | 0.5 | Soliton velocity |
| `R` | 1.0 | Soliton width |
| `sigma` | 5.0 | Transition sharpness |
| `amplitude` | 0.1 | Perturbation amplitude |
| `hyperbolic_param` | 0.5 | Hyperbolic relation parameter |

### Key Properties
- Soliton-like profile (sech function)
- Designed for Einstein-Maxwell-plasma sourcing
- Claims positive energy through EM field sources

### Example
```python
metric = LentzMetric(v0=0.5, R=1.0)

# Estimate required EM fields
em_info = metric.em_field_estimate(t=0, x=1, y=0, z=0)
print(f"Required B field: {em_info['magnetic_field_estimate']}")
```

### Field Sourcing Analysis
```python
from warpbubblesim.metrics.lentz import LentzFieldSourcing

metric = LentzMetric()
sourcing = LentzFieldSourcing(metric)

analysis = sourcing.energy_condition_analysis()
print(f"Min energy density: {analysis['min_energy_density']}")
print(f"All positive: {analysis['all_positive']}")
```

---

## Using the Registry

```python
from warpbubblesim.metrics import get_metric, list_metrics

# List all available
print(list_metrics())

# Get by name
metric = get_metric('alcubierre', v0=1.0, R=2.0)
metric = get_metric('natario', v0=0.5)
metric = get_metric('bobrick_martire', v0=0.3, positive_energy=True)
```

---

## Common Interface

All metrics inherit from `WarpMetric` and provide:

```python
# ADM variables
alpha = metric.lapse(t, x, y, z)      # Scalar
beta = metric.shift(t, x, y, z)        # 3-vector
gamma = metric.spatial_metric(t, x, y, z)  # 3x3 matrix

# Full 4-metric
g = metric.metric(t, x, y, z)          # 4x4 matrix
g_inv = metric.metric_inverse(t, x, y, z)  # 4x4 matrix

# Shape function
f = metric.shape_function(r)

# Bubble dynamics
x_s = metric.bubble_center(t)
v_s = metric.bubble_velocity(t)
r_s = metric.r_from_center(t, x, y, z)

# Callable interface
g = metric(t, x, y, z)

# Get functions for GR module
metric_func = metric.get_metric_func()  # (t,x,y,z) -> g
shift_func = metric.get_shift_func()    # (t,x,y,z) -> beta

# Metadata
info = metric.info()
print(f"Name: {metric.name}")
print(f"Citation: {metric.citation}")
```
