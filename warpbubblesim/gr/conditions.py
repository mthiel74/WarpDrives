"""
Energy condition checks for stress-energy tensors.

The classical energy conditions are:
- WEC (Weak Energy Condition): T_{μν} u^μ u^ν ≥ 0 for all timelike u^μ
- NEC (Null Energy Condition): T_{μν} k^μ k^ν ≥ 0 for all null k^μ
- SEC (Strong Energy Condition): (T_{μν} - (1/2)T g_{μν}) u^μ u^ν ≥ 0 for timelike u^μ
- DEC (Dominant Energy Condition): WEC and T^μ_ν u^ν is non-spacelike for timelike u^μ

For warp drive spacetimes, these conditions are typically violated,
indicating the need for "exotic matter" with negative energy density.
"""

import numpy as np
from typing import Callable, Dict, List, Tuple, Optional
from warpbubblesim.gr.tensors import compute_metric_inverse, BackendType
from warpbubblesim.gr.energy import compute_stress_energy


def check_wec(
    metric_func: Callable,
    coords: np.ndarray,
    backend: BackendType = "finite_difference",
    h: float = 1e-6,
    n_samples: int = 10
) -> Tuple[bool, float]:
    """
    Check Weak Energy Condition (WEC).

    WEC requires T_{μν} u^μ u^ν ≥ 0 for all timelike vectors u^μ.
    This is equivalent to ρ ≥ 0 and ρ + p_i ≥ 0 for all principal pressures.

    Parameters
    ----------
    metric_func : callable
        Function (t, x, y, z) -> g_{μν}.
    coords : np.ndarray
        Coordinates [t, x, y, z].
    backend : str
        Derivative backend.
    h : float
        Step size for finite differences.
    n_samples : int
        Number of random timelike vectors to test.

    Returns
    -------
    tuple
        (satisfied, min_value) where satisfied is bool and min_value
        is the minimum T_{μν} u^μ u^ν found.
    """
    g = metric_func(*coords)
    g_inv = compute_metric_inverse(g)
    T = compute_stress_energy(metric_func, coords, backend, h)

    min_val = np.inf

    # Test with several timelike vectors
    for _ in range(n_samples):
        # Generate random spatial direction
        spatial = np.random.randn(3)
        spatial_norm = np.sqrt(np.dot(spatial, spatial))

        # Create timelike vector with |v| < 1
        v_mag = 0.9 * np.random.random()  # 3-velocity magnitude
        if spatial_norm > 0:
            v = v_mag * spatial / spatial_norm
        else:
            v = np.zeros(3)

        # Construct 4-velocity (approximately, for flat spatial metric)
        # u^0 = γ, u^i = γ v^i where γ = 1/√(1-v²)
        gamma = 1.0 / np.sqrt(1 - v_mag**2)
        u = np.array([gamma, gamma * v[0], gamma * v[1], gamma * v[2]])

        # Verify timelike: g_{μν} u^μ u^ν < 0
        norm_sq = np.einsum('mn,m,n->', g, u, u)
        if norm_sq >= 0:
            # Adjust to ensure timelike
            u[0] = np.sqrt(1 + np.einsum('ij,i,j->', g[1:, 1:], u[1:], u[1:]))

        # Compute T_{μν} u^μ u^ν
        val = np.einsum('mn,m,n->', T, u, u)
        min_val = min(min_val, val)

    # Also test the coordinate time direction
    u_static = np.array([1.0, 0.0, 0.0, 0.0])
    val_static = np.einsum('mn,m,n->', T, u_static, u_static)
    min_val = min(min_val, val_static)

    return min_val >= -1e-10, float(min_val)


def check_nec(
    metric_func: Callable,
    coords: np.ndarray,
    backend: BackendType = "finite_difference",
    h: float = 1e-6,
    n_samples: int = 20
) -> Tuple[bool, float]:
    """
    Check Null Energy Condition (NEC).

    NEC requires T_{μν} k^μ k^ν ≥ 0 for all null vectors k^μ.
    This is equivalent to ρ + p_i ≥ 0 for all principal pressures.

    Parameters
    ----------
    metric_func : callable
        Function (t, x, y, z) -> g_{μν}.
    coords : np.ndarray
        Coordinates [t, x, y, z].
    backend : str
        Derivative backend.
    h : float
        Step size for finite differences.
    n_samples : int
        Number of random null vectors to test.

    Returns
    -------
    tuple
        (satisfied, min_value) where satisfied is bool and min_value
        is the minimum T_{μν} k^μ k^ν found.
    """
    g = metric_func(*coords)
    T = compute_stress_energy(metric_func, coords, backend, h)

    min_val = np.inf

    # Test with null vectors in various directions
    for _ in range(n_samples):
        # Random direction on unit sphere
        theta = np.arccos(2 * np.random.random() - 1)
        phi = 2 * np.pi * np.random.random()

        # Spatial direction
        n_spatial = np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])

        # For approximately flat spatial metric, null vector k^μ = (1, n^i)
        k = np.array([1.0, n_spatial[0], n_spatial[1], n_spatial[2]])

        # Compute T_{μν} k^μ k^ν
        val = np.einsum('mn,m,n->', T, k, k)
        min_val = min(min_val, val)

    return min_val >= -1e-10, float(min_val)


def check_sec(
    metric_func: Callable,
    coords: np.ndarray,
    backend: BackendType = "finite_difference",
    h: float = 1e-6,
    n_samples: int = 10
) -> Tuple[bool, float]:
    """
    Check Strong Energy Condition (SEC).

    SEC requires (T_{μν} - (1/2) T g_{μν}) u^μ u^ν ≥ 0 for timelike u^μ.
    Equivalently: ρ + Σp_i ≥ 0 and ρ + p_i ≥ 0.

    Parameters
    ----------
    metric_func : callable
        Function (t, x, y, z) -> g_{μν}.
    coords : np.ndarray
        Coordinates [t, x, y, z].
    backend : str
        Derivative backend.
    h : float
        Step size for finite differences.
    n_samples : int
        Number of random timelike vectors to test.

    Returns
    -------
    tuple
        (satisfied, min_value)
    """
    g = metric_func(*coords)
    g_inv = compute_metric_inverse(g)
    T = compute_stress_energy(metric_func, coords, backend, h)

    # Trace of T
    T_trace = np.einsum('mn,mn->', g_inv, T)

    # Modified tensor: T_{μν} - (1/2) T g_{μν}
    T_sec = T - 0.5 * T_trace * g

    min_val = np.inf

    # Test with timelike vectors
    for _ in range(n_samples):
        spatial = np.random.randn(3)
        spatial_norm = np.sqrt(np.dot(spatial, spatial))

        v_mag = 0.9 * np.random.random()
        if spatial_norm > 0:
            v = v_mag * spatial / spatial_norm
        else:
            v = np.zeros(3)

        gamma = 1.0 / np.sqrt(1 - v_mag**2)
        u = np.array([gamma, gamma * v[0], gamma * v[1], gamma * v[2]])

        val = np.einsum('mn,m,n->', T_sec, u, u)
        min_val = min(min_val, val)

    # Static observer
    u_static = np.array([1.0, 0.0, 0.0, 0.0])
    val_static = np.einsum('mn,m,n->', T_sec, u_static, u_static)
    min_val = min(min_val, val_static)

    return min_val >= -1e-10, float(min_val)


def check_dec(
    metric_func: Callable,
    coords: np.ndarray,
    backend: BackendType = "finite_difference",
    h: float = 1e-6,
    n_samples: int = 10
) -> Tuple[bool, float]:
    """
    Check Dominant Energy Condition (DEC).

    DEC requires:
    1. WEC is satisfied
    2. For any timelike u^μ, the vector -T^μ_ν u^ν is non-spacelike

    The second condition ensures energy flux doesn't exceed light speed.

    Parameters
    ----------
    metric_func : callable
        Function (t, x, y, z) -> g_{μν}.
    coords : np.ndarray
        Coordinates [t, x, y, z].
    backend : str
        Derivative backend.
    h : float
        Step size for finite differences.
    n_samples : int
        Number of random timelike vectors to test.

    Returns
    -------
    tuple
        (satisfied, violation_measure) where violation_measure is
        the most positive g_{μν} J^μ J^ν found (should be ≤ 0).
    """
    g = metric_func(*coords)
    g_inv = compute_metric_inverse(g)
    T = compute_stress_energy(metric_func, coords, backend, h)

    # First check WEC
    wec_ok, wec_val = check_wec(metric_func, coords, backend, h, n_samples)

    if not wec_ok:
        return False, wec_val

    # T^μ_ν = g^{μρ} T_{ρν}
    T_mixed = np.einsum('mr,rn->mn', g_inv, T)

    max_norm = -np.inf

    for _ in range(n_samples):
        spatial = np.random.randn(3)
        spatial_norm = np.sqrt(np.dot(spatial, spatial))

        v_mag = 0.9 * np.random.random()
        if spatial_norm > 0:
            v = v_mag * spatial / spatial_norm
        else:
            v = np.zeros(3)

        gamma = 1.0 / np.sqrt(1 - v_mag**2)
        u = np.array([gamma, gamma * v[0], gamma * v[1], gamma * v[2]])

        # J^μ = -T^μ_ν u^ν (energy-momentum current)
        J = -np.einsum('mn,n->m', T_mixed, u)

        # Check if J is non-spacelike: g_{μν} J^μ J^ν ≤ 0
        norm_sq = np.einsum('mn,m,n->', g, J, J)
        max_norm = max(max_norm, norm_sq)

    # Static observer
    u_static = np.array([1.0, 0.0, 0.0, 0.0])
    J_static = -np.einsum('mn,n->m', T_mixed, u_static)
    norm_static = np.einsum('mn,m,n->', g, J_static, J_static)
    max_norm = max(max_norm, norm_static)

    return max_norm <= 1e-10, float(max_norm)


def check_energy_conditions(
    metric_func: Callable,
    coords: np.ndarray,
    backend: BackendType = "finite_difference",
    h: float = 1e-6,
    n_samples: int = 10
) -> Dict[str, Tuple[bool, float]]:
    """
    Check all classical energy conditions at a point.

    Parameters
    ----------
    metric_func : callable
        Function (t, x, y, z) -> g_{μν}.
    coords : np.ndarray
        Coordinates [t, x, y, z].
    backend : str
        Derivative backend.
    h : float
        Step size for finite differences.
    n_samples : int
        Number of random vectors to test.

    Returns
    -------
    dict
        Dictionary with keys 'WEC', 'NEC', 'SEC', 'DEC', each mapping to
        (satisfied, critical_value) tuple.
    """
    return {
        'WEC': check_wec(metric_func, coords, backend, h, n_samples),
        'NEC': check_nec(metric_func, coords, backend, h, n_samples),
        'SEC': check_sec(metric_func, coords, backend, h, n_samples),
        'DEC': check_dec(metric_func, coords, backend, h, n_samples),
    }


def check_conditions_on_grid(
    metric_func: Callable,
    coords_array: np.ndarray,
    backend: BackendType = "finite_difference",
    h: float = 1e-6
) -> Dict[str, np.ndarray]:
    """
    Check energy conditions on a grid of points.

    Returns arrays of boolean values for each condition.

    Parameters
    ----------
    metric_func : callable
        Function (t, x, y, z) -> g_{μν}.
    coords_array : np.ndarray
        Array of coordinates, shape (..., 4).
    backend : str
        Derivative backend.
    h : float
        Step size for finite differences.

    Returns
    -------
    dict
        Dictionary with arrays of booleans for each condition.
    """
    original_shape = coords_array.shape[:-1]
    coords_flat = coords_array.reshape(-1, 4)
    n_points = coords_flat.shape[0]

    results = {
        'WEC': np.zeros(n_points, dtype=bool),
        'NEC': np.zeros(n_points, dtype=bool),
        'SEC': np.zeros(n_points, dtype=bool),
        'DEC': np.zeros(n_points, dtype=bool),
        'WEC_value': np.zeros(n_points),
        'NEC_value': np.zeros(n_points),
    }

    for i in range(n_points):
        conds = check_energy_conditions(metric_func, coords_flat[i], backend, h, 5)
        for key in ['WEC', 'NEC', 'SEC', 'DEC']:
            results[key][i] = conds[key][0]
        results['WEC_value'][i] = conds['WEC'][1]
        results['NEC_value'][i] = conds['NEC'][1]

    # Reshape back
    for key in results:
        results[key] = results[key].reshape(original_shape)

    return results


def summarize_energy_conditions(
    metric_func: Callable,
    coords_array: np.ndarray,
    backend: BackendType = "finite_difference",
    h: float = 1e-6
) -> Dict[str, any]:
    """
    Summarize energy condition violations over a region.

    Parameters
    ----------
    metric_func : callable
        Function (t, x, y, z) -> g_{μν}.
    coords_array : np.ndarray
        Array of coordinates, shape (..., 4).
    backend : str
        Derivative backend.
    h : float
        Step size for finite differences.

    Returns
    -------
    dict
        Summary with violation fractions and extremal values.
    """
    results = check_conditions_on_grid(metric_func, coords_array, backend, h)

    n_total = results['WEC'].size

    summary = {
        'n_points': n_total,
        'WEC_violation_fraction': 1.0 - results['WEC'].sum() / n_total,
        'NEC_violation_fraction': 1.0 - results['NEC'].sum() / n_total,
        'SEC_violation_fraction': 1.0 - results['SEC'].sum() / n_total,
        'DEC_violation_fraction': 1.0 - results['DEC'].sum() / n_total,
        'min_WEC_value': float(results['WEC_value'].min()),
        'max_WEC_value': float(results['WEC_value'].max()),
        'min_NEC_value': float(results['NEC_value'].min()),
        'max_NEC_value': float(results['NEC_value'].max()),
    }

    return summary
