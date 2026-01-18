# Numerical Methods

This document describes the numerical methods used in WarpBubbleSim.

## Derivative Computation

### Finite Differences

The default method for computing derivatives uses central finite differences:

$$\frac{\partial f}{\partial x} \approx \frac{f(x+h) - f(x-h)}{2h}$$

Default step size: `h = 1e-6`

**Pros**:
- Simple and robust
- No additional dependencies
- Works with any metric function

**Cons**:
- Slower than AD
- Accumulates errors for higher derivatives

### JAX Automatic Differentiation

When JAX is available, automatic differentiation provides:
- Exact derivatives (to machine precision)
- Faster computation through JIT compilation
- Better scaling for complex metrics

```python
from warpbubblesim.gr.tensors import compute_christoffel

# Using finite differences (default)
gamma = compute_christoffel(metric_func, coords, backend='finite_difference')

# Using JAX AD
gamma = compute_christoffel(metric_func, coords, backend='jax')
```

## Tensor Computations

### Christoffel Symbols

Computed from the Christoffel formula:
$$\Gamma^\mu_{\alpha\beta} = \frac{1}{2}g^{\mu\nu}(\partial_\alpha g_{\beta\nu} + \partial_\beta g_{\alpha\nu} - \partial_\nu g_{\alpha\beta})$$

**Algorithm**:
1. Compute metric $g_{\mu\nu}$ at point
2. Compute inverse metric $g^{\mu\nu}$
3. Compute all first derivatives $\partial_\alpha g_{\beta\nu}$
4. Contract using formula

**Complexity**: O(1) for fixed dimensions (4D)

### Riemann Tensor

The Riemann tensor requires derivatives of Christoffel symbols:
$$R^\rho_{\sigma\mu\nu} = \partial_\mu\Gamma^\rho_{\nu\sigma} - \partial_\nu\Gamma^\rho_{\mu\sigma} + \Gamma^\rho_{\mu\lambda}\Gamma^\lambda_{\nu\sigma} - \Gamma^\rho_{\nu\lambda}\Gamma^\lambda_{\mu\sigma}$$

**Algorithm**:
1. Compute Christoffel at point
2. Compute Christoffel at neighboring points for derivatives
3. Apply formula

**Numerical Considerations**:
- Uses larger step size for Christoffel derivatives to reduce noise
- Total of 9 metric evaluations per Riemann computation

## Geodesic Integration

### ODE System

The geodesic equations are written as a first-order system:
$$\frac{dx^\mu}{d\lambda} = u^\mu$$
$$\frac{du^\mu}{d\lambda} = -\Gamma^\mu_{\alpha\beta}u^\alpha u^\beta$$

State vector: $[x^0, x^1, x^2, x^3, u^0, u^1, u^2, u^3]$

### Integration Methods

Uses SciPy's `solve_ivp` with:

**RK45** (default):
- 5th order Runge-Kutta with 4th order error estimate
- Adaptive step size
- Good for most cases

**DOP853**:
- 8th order method
- Better for high-precision requirements
- More expensive per step

```python
from warpbubblesim.gr.geodesics import integrate_geodesic

# Standard integration
result = integrate_geodesic(metric_func, x0, u0, (0, 10))

# High precision
result = integrate_geodesic(
    metric_func, x0, u0, (0, 10),
    method='DOP853',
    rtol=1e-10,
    atol=1e-12
)
```

### Normalization Preservation

For timelike geodesics, $g_{\mu\nu}u^\mu u^\nu = -1$ should be preserved.

**Monitoring**:
- Normalization is checked at initialization
- Final normalization drift is reported in results

**Renormalization** (optional):
- Periodically project velocity back to constraint surface
- Use when high precision is needed over long integration

```python
result = integrate_geodesic(
    metric_func, x0, u0, (0, 100),
    renormalize=True,
    renorm_interval=100  # Renormalize every 100 steps
)
```

## Energy Condition Checks

Energy conditions are checked by sampling random test vectors.

### WEC (Weak Energy Condition)

Samples random timelike vectors $u^\mu$ and checks $T_{\mu\nu}u^\mu u^\nu \geq 0$.

```python
# Default: 10 random samples
satisfied, min_value = check_wec(metric_func, coords, n_samples=10)

# More thorough: 50 samples
satisfied, min_value = check_wec(metric_func, coords, n_samples=50)
```

### NEC (Null Energy Condition)

Samples random null vectors uniformly on the light cone.

### Statistical Reliability

- More samples → more reliable results
- `n_samples=10` sufficient for most cases
- `n_samples=50+` for publication-quality results

## Grid Computations

### 2D Slices

For field visualization, compute values on a 2D grid:

```python
from warpbubblesim.utils.grids import create_grid_2d

X, Y = create_grid_2d((-5, 5), (-5, 5), 128, 128)
# X, Y are 128×128 arrays
```

**Performance**:
- 128×128 grid: ~16,000 points
- With Christoffel: ~9 metric evaluations per point
- Total: ~150,000 metric evaluations

### Vectorization

The code uses loops for clarity, but can be optimized with:
- NumPy vectorization
- JAX vmap
- Numba JIT compilation

```python
# Future optimization example
import jax.numpy as jnp
from jax import vmap

# Vectorized Christoffel computation
vectorized_christoffel = vmap(
    lambda coords: compute_christoffel(metric_func, coords)
)
```

## Performance Tips

### 1. Use Lower Resolution for Exploration

```python
# Quick exploration: 64×64
plot_energy_density(metric, nx=64, ny=64)

# Final visualization: 128×128
plot_energy_density(metric, nx=128, ny=128)
```

### 2. Use Analytic Formulas When Available

```python
# Slow: numerical computation
from warpbubblesim.gr.energy import compute_energy_density
rho = compute_energy_density(metric_func, coords)

# Fast: analytic formula (Alcubierre only)
rho = metric.eulerian_energy_density_analytic(t, x, y, z)
```

### 3. Cache Metric Evaluations

For intensive computations, consider caching:

```python
from functools import lru_cache

@lru_cache(maxsize=10000)
def cached_metric(t, x, y, z):
    return tuple(metric.metric(t, x, y, z).flatten())
```

### 4. Use JAX Backend

When available, JAX provides significant speedups:

```python
# Check if JAX is available
from warpbubblesim.gr.tensors import JAX_AVAILABLE
if JAX_AVAILABLE:
    backend = 'jax'
else:
    backend = 'finite_difference'
```

## Numerical Stability

### Avoiding Singularities

The code includes protections for:
- Division by zero at $r_s = 0$
- Metric degeneracy (det(g) = 0)
- Overflow in exponential functions

```python
# Example: safe tanh evaluation
arg = np.clip(sigma * r, -20, 20)  # Prevent overflow
result = np.tanh(arg)
```

### Step Size Selection

For finite differences:
- Too small: roundoff error dominates
- Too large: truncation error dominates
- Default `h=1e-6` balances these for double precision

For geodesic integration:
- `max_step` prevents jumping over bubble features
- Adaptive stepping handles varying curvature
