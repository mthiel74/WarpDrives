"""
Stress-energy tensor and energy density computations.

Given a metric, we compute the Einstein tensor G_{μν} and then
the implied stress-energy tensor via:

    T_{μν} = G_{μν} / (8π)

This is the matter content required by Einstein's equations
to produce the given spacetime geometry.

Units: G = c = 1
"""

import numpy as np
from typing import Callable, Tuple, Optional
from warpbubblesim.gr.tensors import (
    compute_einstein,
    compute_metric_inverse,
    BackendType,
)
from warpbubblesim.gr.adm import compute_eulerian_velocity


def compute_stress_energy(
    metric_func: Callable,
    coords: np.ndarray,
    backend: BackendType = "finite_difference",
    h: float = 1e-6
) -> np.ndarray:
    """
    Compute stress-energy tensor T_{μν} from Einstein tensor.

    T_{μν} = G_{μν} / (8π)

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
    np.ndarray
        Stress-energy tensor T_{μν}, shape (4, 4).
    """
    G = compute_einstein(metric_func, coords, backend, h)
    return G / (8.0 * np.pi)


def compute_energy_density(
    metric_func: Callable,
    coords: np.ndarray,
    observer_velocity: Optional[np.ndarray] = None,
    backend: BackendType = "finite_difference",
    h: float = 1e-6
) -> float:
    """
    Compute energy density as measured by an observer.

    ρ = T_{μν} u^μ u^ν

    where u^μ is the observer's 4-velocity.

    Parameters
    ----------
    metric_func : callable
        Function (t, x, y, z) -> g_{μν}.
    coords : np.ndarray
        Coordinates [t, x, y, z].
    observer_velocity : np.ndarray, optional
        Observer 4-velocity u^μ. If None, uses Eulerian observer.
    backend : str
        Derivative backend.
    h : float
        Step size for finite differences.

    Returns
    -------
    float
        Energy density ρ.
    """
    T = compute_stress_energy(metric_func, coords, backend, h)

    if observer_velocity is None:
        # Use Eulerian observer (comoving with spatial coords)
        # For ADM with α=1, u^μ = (1, 0, 0, 0)
        u = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        u = observer_velocity

    # ρ = T_{μν} u^μ u^ν
    rho = np.einsum('mn,m,n->', T, u, u)

    return float(rho)


def compute_energy_density_eulerian(
    metric_func: Callable,
    shift_func: Callable,
    coords: np.ndarray,
    backend: BackendType = "finite_difference",
    h: float = 1e-6
) -> float:
    """
    Compute energy density for Eulerian observer in ADM formalism.

    The Eulerian observer has 4-velocity n^μ (the normal to hypersurfaces).
    For α=1: n^μ = (1, -β^x, -β^y, -β^z)

    ρ_Eulerian = T_{μν} n^μ n^ν

    Parameters
    ----------
    metric_func : callable
        Function (t, x, y, z) -> g_{μν}.
    shift_func : callable
        Function (t, x, y, z) -> β^i (shift vector).
    coords : np.ndarray
        Coordinates [t, x, y, z].
    backend : str
        Derivative backend.
    h : float
        Step size for finite differences.

    Returns
    -------
    float
        Energy density as measured by Eulerian observer.
    """
    T = compute_stress_energy(metric_func, coords, backend, h)
    beta = shift_func(*coords)

    # Eulerian observer velocity (with α=1)
    n = compute_eulerian_velocity(beta)

    # ρ = T_{μν} n^μ n^ν
    rho = np.einsum('mn,m,n->', T, n, n)

    return float(rho)


def compute_momentum_density(
    metric_func: Callable,
    coords: np.ndarray,
    observer_velocity: Optional[np.ndarray] = None,
    backend: BackendType = "finite_difference",
    h: float = 1e-6
) -> np.ndarray:
    """
    Compute momentum density as measured by an observer.

    j^μ = -T^μ_ν u^ν + ρ u^μ

    (the spatial part gives momentum density)

    Parameters
    ----------
    metric_func : callable
        Function (t, x, y, z) -> g_{μν}.
    coords : np.ndarray
        Coordinates [t, x, y, z].
    observer_velocity : np.ndarray, optional
        Observer 4-velocity u^μ. If None, uses static observer.
    backend : str
        Derivative backend.
    h : float
        Step size for finite differences.

    Returns
    -------
    np.ndarray
        Momentum density j^μ, shape (4,).
    """
    g = metric_func(*coords)
    g_inv = compute_metric_inverse(g)
    T = compute_stress_energy(metric_func, coords, backend, h)

    if observer_velocity is None:
        u = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        u = observer_velocity

    # Raise first index of T: T^μ_ν = g^{μρ} T_{ρν}
    T_mixed = np.einsum('mr,rn->mn', g_inv, T)

    # j^μ = -T^μ_ν u^ν
    j = -np.einsum('mn,n->m', T_mixed, u)

    return j


def compute_pressure(
    metric_func: Callable,
    coords: np.ndarray,
    backend: BackendType = "finite_difference",
    h: float = 1e-6
) -> Tuple[float, np.ndarray]:
    """
    Compute isotropic pressure and principal pressures.

    For a perfect fluid: T_{μν} = (ρ + p) u_μ u_ν + p g_{μν}

    Returns both the isotropic pressure (trace/3) and
    principal pressures from eigenvalue decomposition.

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
    tuple
        (isotropic_pressure, principal_pressures)
        - isotropic_pressure: float, p = (1/3) T^i_i (spatial trace)
        - principal_pressures: np.ndarray, eigenvalues of T^i_j
    """
    g = metric_func(*coords)
    g_inv = compute_metric_inverse(g)
    T = compute_stress_energy(metric_func, coords, backend, h)

    # T^μ_ν = g^{μρ} T_{ρν}
    T_mixed = np.einsum('mr,rn->mn', g_inv, T)

    # Spatial part T^i_j
    T_spatial = T_mixed[1:, 1:]

    # Isotropic pressure: p = (1/3) Tr(T^i_j)
    p_iso = np.trace(T_spatial) / 3.0

    # Principal pressures (eigenvalues)
    principal = np.linalg.eigvalsh(T_spatial)

    return float(p_iso), principal


def compute_energy_flux(
    metric_func: Callable,
    coords: np.ndarray,
    backend: BackendType = "finite_difference",
    h: float = 1e-6
) -> np.ndarray:
    """
    Compute energy flux vector S^i = T^{0i}.

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
    np.ndarray
        Energy flux S^i, shape (3,).
    """
    g = metric_func(*coords)
    g_inv = compute_metric_inverse(g)
    T = compute_stress_energy(metric_func, coords, backend, h)

    # T^{μν} = g^{μρ} g^{νσ} T_{ρσ}
    T_upper = np.einsum('mr,ns,rs->mn', g_inv, g_inv, T)

    # S^i = T^{0i}
    return T_upper[0, 1:]


def decompose_stress_energy(
    metric_func: Callable,
    coords: np.ndarray,
    observer_velocity: Optional[np.ndarray] = None,
    backend: BackendType = "finite_difference",
    h: float = 1e-6
) -> dict:
    """
    Decompose stress-energy tensor relative to an observer.

    Returns:
    - ρ: energy density
    - p: isotropic pressure
    - q^μ: heat flux
    - π_{μν}: anisotropic stress

    Parameters
    ----------
    metric_func : callable
        Function (t, x, y, z) -> g_{μν}.
    coords : np.ndarray
        Coordinates [t, x, y, z].
    observer_velocity : np.ndarray, optional
        Observer 4-velocity u^μ.
    backend : str
        Derivative backend.
    h : float
        Step size for finite differences.

    Returns
    -------
    dict
        Dictionary with 'rho', 'pressure', 'heat_flux', 'anisotropic_stress'.
    """
    g = metric_func(*coords)
    g_inv = compute_metric_inverse(g)
    T = compute_stress_energy(metric_func, coords, backend, h)

    if observer_velocity is None:
        u = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        u = observer_velocity

    # Lower u: u_μ = g_{μν} u^ν
    u_lower = g @ u

    # Projection tensor: h_{μν} = g_{μν} + u_μ u_ν
    h_proj = g + np.outer(u_lower, u_lower)

    # Energy density: ρ = T_{μν} u^μ u^ν
    rho = np.einsum('mn,m,n->', T, u, u)

    # Pressure: p = (1/3) h^{μν} T_{μν}
    h_inv = g_inv + np.outer(u, u)  # h^{μν}
    p = np.einsum('mn,mn->', h_inv, T) / 3.0

    # Heat flux: q_μ = -h_μ^ρ T_{ρσ} u^σ
    h_mixed = np.einsum('mr,rn->mn', g, h_inv)  # h_μ^ρ = g_{μν} h^{νρ}
    # Actually: h_μ^ν = δ_μ^ν + u_μ u^ν
    h_mixed = np.eye(4) + np.outer(u_lower, u)
    q = -np.einsum('mr,rs,s->m', h_mixed, T, u)

    # Anisotropic stress: π_{μν} = h_μ^ρ h_ν^σ T_{ρσ} - p h_{μν}
    pi = np.einsum('mr,ns,rs->mn', h_mixed, h_mixed, T) - p * h_proj

    return {
        'rho': float(rho),
        'pressure': float(p),
        'heat_flux': q,
        'anisotropic_stress': pi,
    }


def vectorized_energy_density(
    metric_func: Callable,
    coords_array: np.ndarray,
    observer_velocity: Optional[np.ndarray] = None,
    backend: BackendType = "finite_difference",
    h: float = 1e-6
) -> np.ndarray:
    """
    Compute energy density at multiple points.

    Parameters
    ----------
    metric_func : callable
        Function (t, x, y, z) -> g_{μν}.
    coords_array : np.ndarray
        Array of coordinates, shape (N, 4) or grid shape (..., 4).
    observer_velocity : np.ndarray, optional
        Observer 4-velocity.
    backend : str
        Derivative backend.
    h : float
        Step size for finite differences.

    Returns
    -------
    np.ndarray
        Energy density at each point.
    """
    original_shape = coords_array.shape[:-1]
    coords_flat = coords_array.reshape(-1, 4)

    result = np.zeros(coords_flat.shape[0])
    for i in range(coords_flat.shape[0]):
        result[i] = compute_energy_density(
            metric_func, coords_flat[i], observer_velocity, backend, h
        )

    return result.reshape(original_shape)
