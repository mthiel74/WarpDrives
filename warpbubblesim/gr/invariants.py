"""
Curvature invariants for General Relativity.

Provides computation of scalar curvature invariants that are
coordinate-independent measures of spacetime curvature.

Key invariants:
- Kretschmann scalar: K = R_{αβγδ} R^{αβγδ}
- Chern-Pontryagin: *R R = R_{αβγδ} *R^{αβγδ}
- Weyl squared: C_{αβγδ} C^{αβγδ}
- Euler density (4D): E_4
"""

import numpy as np
from typing import Callable, Optional
from warpbubblesim.gr.tensors import (
    compute_riemann,
    compute_riemann_all_lower,
    compute_ricci,
    compute_ricci_scalar,
    compute_weyl,
    compute_metric_inverse,
    BackendType,
)


def compute_kretschmann(
    metric_func: Callable,
    coords: np.ndarray,
    backend: BackendType = "finite_difference",
    h: float = 1e-6
) -> float:
    """
    Compute Kretschmann scalar K = R_{αβγδ} R^{αβγδ}.

    This is a coordinate-independent measure of spacetime curvature.
    For Schwarzschild: K = 48 M² / r⁶

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

    Returns
    -------
    float
        Kretschmann scalar.
    """
    g = metric_func(*coords)
    g_inv = compute_metric_inverse(g)

    # Get Riemann with all lower indices
    riemann_lower = compute_riemann_all_lower(metric_func, coords, backend, h)

    # Raise all indices to get R^{αβγδ}
    # R^{αβγδ} = g^{αμ} g^{βν} g^{γρ} g^{δσ} R_{μνρσ}
    riemann_upper = np.einsum(
        'am,bn,cr,ds,mnrs->abcd',
        g_inv, g_inv, g_inv, g_inv, riemann_lower
    )

    # Contract: K = R_{αβγδ} R^{αβγδ}
    kretschmann = np.einsum('abcd,abcd->', riemann_lower, riemann_upper)

    return float(kretschmann)


def compute_chern_pontryagin(
    metric_func: Callable,
    coords: np.ndarray,
    backend: BackendType = "finite_difference",
    h: float = 1e-6
) -> float:
    """
    Compute Chern-Pontryagin scalar *RR = R_{αβγδ} *R^{αβγδ}.

    This is a pseudo-scalar that measures the "handedness" of curvature.
    Vanishes for parity-symmetric spacetimes.

    *R^{αβγδ} = (1/2) ε^{αβμν} R^{γδ}_{  μν}

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

    Returns
    -------
    float
        Chern-Pontryagin scalar.
    """
    g = metric_func(*coords)
    g_inv = compute_metric_inverse(g)
    det_g = np.linalg.det(g)

    riemann_lower = compute_riemann_all_lower(metric_func, coords, backend, h)
    riemann_mixed = compute_riemann(metric_func, coords, backend, h)

    # Levi-Civita tensor density
    epsilon = np.zeros((4, 4, 4, 4))
    for a in range(4):
        for b in range(4):
            for c in range(4):
                for d in range(4):
                    if len({a, b, c, d}) == 4:  # All different
                        # Count permutations
                        perm = [a, b, c, d]
                        sign = 1
                        for i in range(4):
                            for j in range(i + 1, 4):
                                if perm[i] > perm[j]:
                                    sign *= -1
                        epsilon[a, b, c, d] = sign

    # Epsilon tensor with metric factor: ε^{αβγδ} = ε_{αβγδ} / sqrt(-det(g))
    sqrt_neg_det = np.sqrt(np.abs(det_g))
    epsilon_up = epsilon / sqrt_neg_det

    # Dual Riemann: *R_{αβγδ} = (1/2) ε_{αβμν} R^{μν}_{  γδ}
    # First raise indices on Riemann for the contraction
    riemann_mixed_raised = np.einsum('am,bn,mncd->abcd', g_inv, g_inv, riemann_lower)

    dual_riemann = 0.5 * np.einsum('abmn,mncd->abcd', epsilon, riemann_mixed_raised)

    # Contract R_{αβγδ} * (*R^{αβγδ})
    dual_raised = np.einsum('am,bn,cr,ds,mnrs->abcd', g_inv, g_inv, g_inv, g_inv, dual_riemann)

    chern_pontryagin = np.einsum('abcd,abcd->', riemann_lower, dual_raised)

    return float(chern_pontryagin)


def compute_weyl_squared(
    metric_func: Callable,
    coords: np.ndarray,
    backend: BackendType = "finite_difference",
    h: float = 1e-6
) -> float:
    """
    Compute Weyl squared C_{αβγδ} C^{αβγδ}.

    The Weyl tensor measures the "vacuum" part of curvature
    (tidal deformations without volume change).

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

    Returns
    -------
    float
        Weyl squared scalar.
    """
    g = metric_func(*coords)
    g_inv = compute_metric_inverse(g)

    weyl = compute_weyl(metric_func, coords, backend, h)

    # Raise all indices
    weyl_upper = np.einsum(
        'am,bn,cr,ds,mnrs->abcd',
        g_inv, g_inv, g_inv, g_inv, weyl
    )

    # Contract
    weyl_squared = np.einsum('abcd,abcd->', weyl, weyl_upper)

    return float(weyl_squared)


def compute_ricci_squared(
    metric_func: Callable,
    coords: np.ndarray,
    backend: BackendType = "finite_difference",
    h: float = 1e-6
) -> float:
    """
    Compute Ricci squared R_{μν} R^{μν}.

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

    Returns
    -------
    float
        Ricci squared scalar.
    """
    g = metric_func(*coords)
    g_inv = compute_metric_inverse(g)
    ricci = compute_ricci(metric_func, coords, backend, h)

    # Raise indices
    ricci_upper = np.einsum('am,bn,mn->ab', g_inv, g_inv, ricci)

    # Contract
    ricci_squared = np.einsum('ab,ab->', ricci, ricci_upper)

    return float(ricci_squared)


def compute_euler_density(
    metric_func: Callable,
    coords: np.ndarray,
    backend: BackendType = "finite_difference",
    h: float = 1e-6
) -> float:
    """
    Compute Euler density (Gauss-Bonnet invariant) in 4D.

    E_4 = R_{αβγδ} R^{αβγδ} - 4 R_{μν} R^{μν} + R²

    This is a topological invariant (total derivative in 4D).

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

    Returns
    -------
    float
        Euler density.
    """
    kretschmann = compute_kretschmann(metric_func, coords, backend, h)
    ricci_sq = compute_ricci_squared(metric_func, coords, backend, h)
    R = compute_ricci_scalar(metric_func, coords, backend, h)

    return kretschmann - 4 * ricci_sq + R**2


def compute_all_invariants(
    metric_func: Callable,
    coords: np.ndarray,
    backend: BackendType = "finite_difference",
    h: float = 1e-6
) -> dict:
    """
    Compute all curvature invariants at a point.

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

    Returns
    -------
    dict
        Dictionary of invariant values.
    """
    return {
        'ricci_scalar': compute_ricci_scalar(metric_func, coords, backend, h),
        'kretschmann': compute_kretschmann(metric_func, coords, backend, h),
        'ricci_squared': compute_ricci_squared(metric_func, coords, backend, h),
        'weyl_squared': compute_weyl_squared(metric_func, coords, backend, h),
        'chern_pontryagin': compute_chern_pontryagin(metric_func, coords, backend, h),
        'euler_density': compute_euler_density(metric_func, coords, backend, h),
    }


def vectorized_kretschmann(
    metric_func: Callable,
    coords_array: np.ndarray,
    backend: BackendType = "finite_difference",
    h: float = 1e-6
) -> np.ndarray:
    """
    Compute Kretschmann scalar at multiple points.

    Parameters
    ----------
    metric_func : callable
        Function (t, x, y, z) -> g_{μν}.
    coords_array : np.ndarray
        Array of coordinates, shape (N, 4) or grid shape (..., 4).
    backend : str
        Derivative backend.
    h : float
        Step size for finite differences.

    Returns
    -------
    np.ndarray
        Kretschmann values at each point.
    """
    original_shape = coords_array.shape[:-1]
    coords_flat = coords_array.reshape(-1, 4)

    result = np.zeros(coords_flat.shape[0])
    for i in range(coords_flat.shape[0]):
        result[i] = compute_kretschmann(metric_func, coords_flat[i], backend, h)

    return result.reshape(original_shape)
