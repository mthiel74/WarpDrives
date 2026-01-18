"""
ADM (Arnowitt-Deser-Misner) 3+1 decomposition utilities.

The ADM formalism decomposes spacetime into spatial slices with:
- Lapse function α: relates proper time to coordinate time
- Shift vector β^i: relates spatial coordinates between slices
- Spatial metric γ_{ij}: metric on spatial hypersurfaces

The 4-metric in ADM form:
ds² = -α² dt² + γ_{ij}(dx^i + β^i dt)(dx^j + β^j dt)

Conventions:
- Latin indices (i,j,k) run 1,2,3 (spatial)
- Greek indices (μ,ν) run 0,1,2,3 (spacetime)
"""

import numpy as np
from typing import Callable, Tuple, Optional
from warpbubblesim.gr.tensors import _finite_diff_derivative, compute_christoffel


def adm_to_4metric(
    lapse: float,
    shift: np.ndarray,
    spatial_metric: np.ndarray
) -> np.ndarray:
    """
    Construct 4-metric from ADM variables.

    g_{μν} is constructed as:
    g_{00} = -α² + β_i β^i = -α² + γ_{ij} β^i β^j
    g_{0i} = g_{i0} = β_i = γ_{ij} β^j
    g_{ij} = γ_{ij}

    Parameters
    ----------
    lapse : float
        Lapse function α.
    shift : np.ndarray
        Shift vector β^i, shape (3,).
    spatial_metric : np.ndarray
        Spatial metric γ_{ij}, shape (3, 3).

    Returns
    -------
    np.ndarray
        4-metric g_{μν}, shape (4, 4).
    """
    g = np.zeros((4, 4))

    # Lower the shift index: β_i = γ_{ij} β^j
    beta_lower = spatial_metric @ shift

    # g_{00} = -α² + β_i β^i
    g[0, 0] = -lapse**2 + np.dot(beta_lower, shift)

    # g_{0i} = g_{i0} = β_i
    g[0, 1:] = beta_lower
    g[1:, 0] = beta_lower

    # g_{ij} = γ_{ij}
    g[1:, 1:] = spatial_metric

    return g


def metric_to_adm(metric: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Extract ADM variables from 4-metric.

    Parameters
    ----------
    metric : np.ndarray
        4-metric g_{μν}, shape (4, 4).

    Returns
    -------
    tuple
        (lapse, shift, spatial_metric) where:
        - lapse: float, lapse function α
        - shift: np.ndarray, shift vector β^i, shape (3,)
        - spatial_metric: np.ndarray, γ_{ij}, shape (3, 3)
    """
    # Extract spatial metric
    gamma = metric[1:, 1:].copy()

    # Inverse spatial metric
    gamma_inv = np.linalg.inv(gamma)

    # Extract β_i from g_{0i}
    beta_lower = metric[0, 1:]

    # Raise index: β^i = γ^{ij} β_j
    shift = gamma_inv @ beta_lower

    # Compute lapse from g_{00} = -α² + β_i β^i
    beta_squared = np.dot(beta_lower, shift)
    lapse_squared = -metric[0, 0] + beta_squared
    lapse = np.sqrt(max(lapse_squared, 0.0))  # Protect against numerical issues

    return lapse, shift, gamma


def compute_adm_inverse(
    lapse: float,
    shift: np.ndarray,
    spatial_metric: np.ndarray
) -> np.ndarray:
    """
    Compute inverse 4-metric from ADM variables.

    g^{μν} is:
    g^{00} = -1/α²
    g^{0i} = g^{i0} = β^i/α²
    g^{ij} = γ^{ij} - β^i β^j/α²

    Parameters
    ----------
    lapse : float
        Lapse function α.
    shift : np.ndarray
        Shift vector β^i, shape (3,).
    spatial_metric : np.ndarray
        Spatial metric γ_{ij}, shape (3, 3).

    Returns
    -------
    np.ndarray
        Inverse 4-metric g^{μν}, shape (4, 4).
    """
    g_inv = np.zeros((4, 4))
    gamma_inv = np.linalg.inv(spatial_metric)

    alpha_sq = lapse**2

    # g^{00} = -1/α²
    g_inv[0, 0] = -1.0 / alpha_sq

    # g^{0i} = g^{i0} = β^i/α²
    g_inv[0, 1:] = shift / alpha_sq
    g_inv[1:, 0] = shift / alpha_sq

    # g^{ij} = γ^{ij} - β^i β^j/α²
    g_inv[1:, 1:] = gamma_inv - np.outer(shift, shift) / alpha_sq

    return g_inv


def compute_extrinsic_curvature(
    lapse_func: Callable,
    shift_func: Callable,
    spatial_metric_func: Callable,
    coords: np.ndarray,
    h: float = 1e-6
) -> np.ndarray:
    """
    Compute extrinsic curvature K_{ij} of spatial slice.

    K_{ij} = (1/2α) (D_i β_j + D_j β_i - ∂_t γ_{ij})

    where D_i is the covariant derivative with respect to γ_{ij}.

    For static shift (∂_t γ = 0 and time-independent shift):
    K_{ij} = (1/2α) (D_i β_j + D_j β_i)

    Parameters
    ----------
    lapse_func : callable
        Function (t, x, y, z) -> α.
    shift_func : callable
        Function (t, x, y, z) -> β^i (3-vector).
    spatial_metric_func : callable
        Function (t, x, y, z) -> γ_{ij} (3x3 matrix).
    coords : np.ndarray
        Coordinates [t, x, y, z].
    h : float
        Step size for finite differences.

    Returns
    -------
    np.ndarray
        Extrinsic curvature K_{ij}, shape (3, 3).
    """
    t, x, y, z = coords

    alpha = lapse_func(t, x, y, z)
    beta = shift_func(t, x, y, z)
    gamma = spatial_metric_func(t, x, y, z)
    gamma_inv = np.linalg.inv(gamma)

    # Lower shift index
    beta_lower = gamma @ beta

    # Compute spatial Christoffel symbols (3D)
    # Γ^k_{ij} = (1/2) γ^{kl} (∂_i γ_{jl} + ∂_j γ_{il} - ∂_l γ_{ij})
    dgamma = np.zeros((3, 3, 3))  # dgamma[l, i, j] = ∂_l γ_{ij}

    for l in range(3):
        coords_plus = coords.copy()
        coords_minus = coords.copy()
        coords_plus[l + 1] += h  # +1 because coords[0] is t
        coords_minus[l + 1] -= h

        gamma_plus = spatial_metric_func(*coords_plus)
        gamma_minus = spatial_metric_func(*coords_minus)

        dgamma[l] = (gamma_plus - gamma_minus) / (2 * h)

    gamma3 = np.zeros((3, 3, 3))  # 3D Christoffel
    for k in range(3):
        for i in range(3):
            for j in range(3):
                for l in range(3):
                    gamma3[k, i, j] += 0.5 * gamma_inv[k, l] * (
                        dgamma[i, j, l] + dgamma[j, i, l] - dgamma[l, i, j]
                    )

    # Compute covariant derivatives of β_j
    # D_i β_j = ∂_i β_j - Γ^k_{ij} β_k
    dbeta_lower = np.zeros((3, 3))  # dbeta_lower[i, j] = ∂_i β_j

    for i in range(3):
        coords_plus = coords.copy()
        coords_minus = coords.copy()
        coords_plus[i + 1] += h
        coords_minus[i + 1] -= h

        beta_plus = shift_func(*coords_plus)
        beta_minus = shift_func(*coords_minus)
        gamma_plus = spatial_metric_func(*coords_plus)
        gamma_minus = spatial_metric_func(*coords_minus)

        beta_lower_plus = gamma_plus @ beta_plus
        beta_lower_minus = gamma_minus @ beta_minus

        dbeta_lower[i] = (beta_lower_plus - beta_lower_minus) / (2 * h)

    # D_i β_j
    D_beta = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            D_beta[i, j] = dbeta_lower[i, j]
            for k in range(3):
                D_beta[i, j] -= gamma3[k, i, j] * beta_lower[k]

    # Compute time derivative of spatial metric (if needed)
    coords_t_plus = coords.copy()
    coords_t_minus = coords.copy()
    coords_t_plus[0] += h
    coords_t_minus[0] -= h

    gamma_t_plus = spatial_metric_func(*coords_t_plus)
    gamma_t_minus = spatial_metric_func(*coords_t_minus)

    dgamma_dt = (gamma_t_plus - gamma_t_minus) / (2 * h)

    # K_{ij} = (1/2α) (D_i β_j + D_j β_i - ∂_t γ_{ij})
    K = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            K[i, j] = (D_beta[i, j] + D_beta[j, i] - dgamma_dt[i, j]) / (2 * alpha)

    return K


def compute_extrinsic_curvature_from_shift(
    shift_func: Callable,
    spatial_metric: np.ndarray,
    coords: np.ndarray,
    lapse: float = 1.0,
    h: float = 1e-6
) -> np.ndarray:
    """
    Compute extrinsic curvature for flat spatial metric (γ_{ij} = δ_{ij}).

    Simplified version for Alcubierre-type metrics where:
    - α = 1
    - γ_{ij} = δ_{ij}
    - ∂_t γ_{ij} = 0

    Then K_{ij} = (1/2) (∂_i β_j + ∂_j β_i)

    Parameters
    ----------
    shift_func : callable
        Function (t, x, y, z) -> β^i (3-vector).
    spatial_metric : np.ndarray
        Spatial metric (should be δ_{ij} for this simplified form).
    coords : np.ndarray
        Coordinates [t, x, y, z].
    lapse : float
        Lapse function (default 1).
    h : float
        Step size for finite differences.

    Returns
    -------
    np.ndarray
        Extrinsic curvature K_{ij}, shape (3, 3).
    """
    # For flat spatial metric with α=1, β_i = β^i
    dbeta = np.zeros((3, 3))  # dbeta[i, j] = ∂_i β_j

    for i in range(3):
        coords_plus = coords.copy()
        coords_minus = coords.copy()
        coords_plus[i + 1] += h
        coords_minus[i + 1] -= h

        beta_plus = shift_func(*coords_plus)
        beta_minus = shift_func(*coords_minus)

        dbeta[i] = (beta_plus - beta_minus) / (2 * h)

    # K_{ij} = (1/2α) (∂_i β_j + ∂_j β_i)
    K = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            K[i, j] = (dbeta[i, j] + dbeta[j, i]) / (2 * lapse)

    return K


def compute_expansion_scalar(K: np.ndarray, spatial_metric_inv: np.ndarray) -> float:
    """
    Compute expansion scalar (trace of extrinsic curvature).

    K = γ^{ij} K_{ij}

    Parameters
    ----------
    K : np.ndarray
        Extrinsic curvature K_{ij}, shape (3, 3).
    spatial_metric_inv : np.ndarray
        Inverse spatial metric γ^{ij}, shape (3, 3).

    Returns
    -------
    float
        Expansion scalar K.
    """
    return np.einsum('ij,ij->', spatial_metric_inv, K)


def compute_shear_tensor(
    K: np.ndarray,
    spatial_metric: np.ndarray,
    spatial_metric_inv: np.ndarray
) -> np.ndarray:
    """
    Compute shear tensor σ_{ij} = K_{ij} - (1/3) γ_{ij} K.

    Parameters
    ----------
    K : np.ndarray
        Extrinsic curvature K_{ij}, shape (3, 3).
    spatial_metric : np.ndarray
        Spatial metric γ_{ij}, shape (3, 3).
    spatial_metric_inv : np.ndarray
        Inverse spatial metric γ^{ij}, shape (3, 3).

    Returns
    -------
    np.ndarray
        Shear tensor σ_{ij}, shape (3, 3).
    """
    trace_K = compute_expansion_scalar(K, spatial_metric_inv)
    return K - (1.0 / 3.0) * spatial_metric * trace_K


def compute_shift_divergence(
    shift_func: Callable,
    coords: np.ndarray,
    h: float = 1e-6
) -> float:
    """
    Compute divergence of shift vector ∇·β = ∂_i β^i.

    For flat spatial metric.

    Parameters
    ----------
    shift_func : callable
        Function (t, x, y, z) -> β^i (3-vector).
    coords : np.ndarray
        Coordinates [t, x, y, z].
    h : float
        Step size for finite differences.

    Returns
    -------
    float
        Divergence ∂_i β^i.
    """
    div = 0.0

    for i in range(3):
        coords_plus = coords.copy()
        coords_minus = coords.copy()
        coords_plus[i + 1] += h
        coords_minus[i + 1] -= h

        beta_plus = shift_func(*coords_plus)
        beta_minus = shift_func(*coords_minus)

        div += (beta_plus[i] - beta_minus[i]) / (2 * h)

    return div


def compute_normal_vector(lapse: float, shift: np.ndarray) -> np.ndarray:
    """
    Compute the unit normal to spatial hypersurface.

    n^μ = (1/α, -β^i/α) = (1/α)(1, -β^i)

    Parameters
    ----------
    lapse : float
        Lapse function α.
    shift : np.ndarray
        Shift vector β^i, shape (3,).

    Returns
    -------
    np.ndarray
        Unit normal n^μ, shape (4,).
    """
    n = np.zeros(4)
    n[0] = 1.0 / lapse
    n[1:] = -shift / lapse
    return n


def compute_eulerian_velocity(shift: np.ndarray) -> np.ndarray:
    """
    Compute 4-velocity of Eulerian observer.

    For Eulerian observers (zero 3-velocity in spatial coords):
    u^μ = n^μ where n^μ is the unit normal.

    With α=1: u^μ = (1, -β^i)

    Parameters
    ----------
    shift : np.ndarray
        Shift vector β^i, shape (3,).

    Returns
    -------
    np.ndarray
        4-velocity u^μ, shape (4,).
    """
    u = np.zeros(4)
    u[0] = 1.0
    u[1:] = -shift
    return u
