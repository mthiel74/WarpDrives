"""
Core tensor computations for General Relativity.

Implements Christoffel symbols, Riemann tensor, Ricci tensor, and Einstein tensor
using both JAX automatic differentiation and finite-difference backends.

Conventions:
- Metric signature: (-,+,+,+)
- Index ordering: (t,x,y,z) = (0,1,2,3)
- Christoffel: Γ^μ_{αβ} = (1/2) g^{μν} (∂_α g_{βν} + ∂_β g_{αν} - ∂_ν g_{αβ})
- Riemann: R^ρ_{σμν} = ∂_μ Γ^ρ_{νσ} - ∂_ν Γ^ρ_{μσ} + Γ^ρ_{μλ} Γ^λ_{νσ} - Γ^ρ_{νλ} Γ^λ_{μσ}
"""

import numpy as np
from typing import Callable, Literal, Optional
from functools import lru_cache

# Try to import JAX, fall back gracefully
try:
    import jax
    import jax.numpy as jnp
    from jax import jacfwd
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = np


BackendType = Literal["jax", "finite_difference"]


def compute_metric_inverse(metric: np.ndarray) -> np.ndarray:
    """
    Compute the inverse metric g^{μν}.

    Parameters
    ----------
    metric : np.ndarray
        Metric tensor g_{μν}, shape (4, 4) or (..., 4, 4) for batch.

    Returns
    -------
    np.ndarray
        Inverse metric g^{μν}, same shape as input.
    """
    return np.linalg.inv(metric)


def _finite_diff_derivative(
    metric_func: Callable,
    coords: np.ndarray,
    direction: int,
    h: float = 1e-6
) -> np.ndarray:
    """
    Compute partial derivative of metric using finite differences.

    Parameters
    ----------
    metric_func : callable
        Function (t, x, y, z) -> g_{μν} (4x4 array).
    coords : np.ndarray
        Coordinates [t, x, y, z].
    direction : int
        Direction for derivative (0=t, 1=x, 2=y, 3=z).
    h : float
        Step size for finite difference.

    Returns
    -------
    np.ndarray
        ∂_direction g_{μν}, shape (4, 4).
    """
    coords_plus = coords.copy()
    coords_minus = coords.copy()
    coords_plus[direction] += h
    coords_minus[direction] -= h

    g_plus = metric_func(*coords_plus)
    g_minus = metric_func(*coords_minus)

    return (g_plus - g_minus) / (2 * h)


def _compute_metric_derivatives_fd(
    metric_func: Callable,
    coords: np.ndarray,
    h: float = 1e-6
) -> np.ndarray:
    """
    Compute all first derivatives of metric using finite differences.

    Returns ∂_α g_{μν} as array of shape (4, 4, 4) where first index is α.
    """
    dg = np.zeros((4, 4, 4))
    for alpha in range(4):
        dg[alpha] = _finite_diff_derivative(metric_func, coords, alpha, h)
    return dg


def compute_christoffel(
    metric_func: Callable,
    coords: np.ndarray,
    backend: BackendType = "finite_difference",
    h: float = 1e-6
) -> np.ndarray:
    """
    Compute Christoffel symbols Γ^μ_{αβ} at given coordinates.

    Γ^μ_{αβ} = (1/2) g^{μν} (∂_α g_{βν} + ∂_β g_{αν} - ∂_ν g_{αβ})

    Parameters
    ----------
    metric_func : callable
        Function (t, x, y, z) -> g_{μν} (4x4 array).
    coords : np.ndarray
        Coordinates [t, x, y, z].
    backend : str
        'jax' for automatic differentiation, 'finite_difference' for numerical.
    h : float
        Step size for finite differences.

    Returns
    -------
    np.ndarray
        Christoffel symbols Γ^μ_{αβ}, shape (4, 4, 4).
        First index is upper, last two are lower symmetric.
    """
    coords = np.asarray(coords, dtype=np.float64)

    # Get metric and inverse at the point
    g = metric_func(*coords)
    g_inv = compute_metric_inverse(g)

    # Compute derivatives ∂_α g_{μν}
    if backend == "jax" and JAX_AVAILABLE:
        # Use JAX automatic differentiation
        def g_flat(x):
            return jnp.array(metric_func(*x))

        dg_func = jacfwd(g_flat)
        dg = np.array(dg_func(jnp.array(coords)))
        # dg has shape (4, 4, 4) with dg[μ, ν, α] = ∂_α g_{μν}
        # We need dg[α, μ, ν] = ∂_α g_{μν}
        dg = np.transpose(dg, (2, 0, 1))
    else:
        # Use finite differences
        dg = _compute_metric_derivatives_fd(metric_func, coords, h)

    # Compute Christoffel symbols
    # Γ^μ_{αβ} = (1/2) g^{μν} (∂_α g_{βν} + ∂_β g_{αν} - ∂_ν g_{αβ})
    christoffel = np.zeros((4, 4, 4))

    for mu in range(4):
        for alpha in range(4):
            for beta in range(4):
                for nu in range(4):
                    christoffel[mu, alpha, beta] += 0.5 * g_inv[mu, nu] * (
                        dg[alpha, beta, nu] + dg[beta, alpha, nu] - dg[nu, alpha, beta]
                    )

    return christoffel


def _compute_christoffel_derivatives_fd(
    metric_func: Callable,
    coords: np.ndarray,
    h: float = 1e-6,
    backend: BackendType = "finite_difference"
) -> np.ndarray:
    """
    Compute derivatives of Christoffel symbols ∂_μ Γ^ρ_{νσ}.

    Returns array of shape (4, 4, 4, 4) where first index is derivative direction.
    """
    dgamma = np.zeros((4, 4, 4, 4))

    for mu in range(4):
        coords_plus = coords.copy()
        coords_minus = coords.copy()
        coords_plus[mu] += h
        coords_minus[mu] -= h

        gamma_plus = compute_christoffel(metric_func, coords_plus, backend, h)
        gamma_minus = compute_christoffel(metric_func, coords_minus, backend, h)

        dgamma[mu] = (gamma_plus - gamma_minus) / (2 * h)

    return dgamma


def compute_riemann(
    metric_func: Callable,
    coords: np.ndarray,
    backend: BackendType = "finite_difference",
    h: float = 1e-6
) -> np.ndarray:
    """
    Compute Riemann curvature tensor R^ρ_{σμν} at given coordinates.

    R^ρ_{σμν} = ∂_μ Γ^ρ_{νσ} - ∂_ν Γ^ρ_{μσ} + Γ^ρ_{μλ} Γ^λ_{νσ} - Γ^ρ_{νλ} Γ^λ_{μσ}

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
        Riemann tensor R^ρ_{σμν}, shape (4, 4, 4, 4).
    """
    coords = np.asarray(coords, dtype=np.float64)

    # Get Christoffel symbols
    gamma = compute_christoffel(metric_func, coords, backend, h)

    # Get derivatives of Christoffel symbols
    dgamma = _compute_christoffel_derivatives_fd(metric_func, coords, h, backend)

    # Compute Riemann tensor
    riemann = np.zeros((4, 4, 4, 4))

    for rho in range(4):
        for sigma in range(4):
            for mu in range(4):
                for nu in range(4):
                    # ∂_μ Γ^ρ_{νσ} - ∂_ν Γ^ρ_{μσ}
                    riemann[rho, sigma, mu, nu] = (
                        dgamma[mu, rho, nu, sigma] - dgamma[nu, rho, mu, sigma]
                    )

                    # + Γ^ρ_{μλ} Γ^λ_{νσ} - Γ^ρ_{νλ} Γ^λ_{μσ}
                    for lam in range(4):
                        riemann[rho, sigma, mu, nu] += (
                            gamma[rho, mu, lam] * gamma[lam, nu, sigma]
                            - gamma[rho, nu, lam] * gamma[lam, mu, sigma]
                        )

    return riemann


def compute_riemann_all_lower(
    metric_func: Callable,
    coords: np.ndarray,
    backend: BackendType = "finite_difference",
    h: float = 1e-6
) -> np.ndarray:
    """
    Compute Riemann tensor with all indices lowered: R_{ρσμν}.

    R_{ρσμν} = g_{ρλ} R^λ_{σμν}

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
        Riemann tensor R_{ρσμν}, shape (4, 4, 4, 4).
    """
    g = metric_func(*coords)
    riemann_mixed = compute_riemann(metric_func, coords, backend, h)

    # Lower the first index
    riemann_lower = np.einsum('rl,lsmn->rsmn', g, riemann_mixed)

    return riemann_lower


def compute_ricci(
    metric_func: Callable,
    coords: np.ndarray,
    backend: BackendType = "finite_difference",
    h: float = 1e-6
) -> np.ndarray:
    """
    Compute Ricci tensor R_{μν} at given coordinates.

    R_{μν} = R^ρ_{μρν}

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
        Ricci tensor R_{μν}, shape (4, 4).
    """
    riemann = compute_riemann(metric_func, coords, backend, h)

    # Contract: R_{μν} = R^ρ_{μρν}
    ricci = np.einsum('rmrn->mn', riemann)

    return ricci


def compute_ricci_scalar(
    metric_func: Callable,
    coords: np.ndarray,
    backend: BackendType = "finite_difference",
    h: float = 1e-6
) -> float:
    """
    Compute Ricci scalar R at given coordinates.

    R = g^{μν} R_{μν}

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
        Ricci scalar R.
    """
    g = metric_func(*coords)
    g_inv = compute_metric_inverse(g)
    ricci = compute_ricci(metric_func, coords, backend, h)

    # Contract: R = g^{μν} R_{μν}
    return float(np.einsum('mn,mn->', g_inv, ricci))


def compute_einstein(
    metric_func: Callable,
    coords: np.ndarray,
    backend: BackendType = "finite_difference",
    h: float = 1e-6
) -> np.ndarray:
    """
    Compute Einstein tensor G_{μν} at given coordinates.

    G_{μν} = R_{μν} - (1/2) g_{μν} R

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
        Einstein tensor G_{μν}, shape (4, 4).
    """
    g = metric_func(*coords)
    ricci = compute_ricci(metric_func, coords, backend, h)
    g_inv = compute_metric_inverse(g)

    # Compute Ricci scalar
    R = float(np.einsum('mn,mn->', g_inv, ricci))

    # Einstein tensor
    einstein = ricci - 0.5 * g * R

    return einstein


def compute_weyl(
    metric_func: Callable,
    coords: np.ndarray,
    backend: BackendType = "finite_difference",
    h: float = 1e-6
) -> np.ndarray:
    """
    Compute Weyl tensor C_{ρσμν} at given coordinates.

    In 4D: C_{ρσμν} = R_{ρσμν} - (g_{ρμ}R_{σν} - g_{ρν}R_{σμ} + g_{σν}R_{ρμ} - g_{σμ}R_{ρν})/2
                     + R(g_{ρμ}g_{σν} - g_{ρν}g_{σμ})/6

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
        Weyl tensor C_{ρσμν}, shape (4, 4, 4, 4).
    """
    g = metric_func(*coords)
    riemann = compute_riemann_all_lower(metric_func, coords, backend, h)
    ricci = compute_ricci(metric_func, coords, backend, h)
    g_inv = compute_metric_inverse(g)
    R = float(np.einsum('mn,mn->', g_inv, ricci))

    # Build Weyl tensor
    weyl = np.zeros((4, 4, 4, 4))

    for rho in range(4):
        for sigma in range(4):
            for mu in range(4):
                for nu in range(4):
                    weyl[rho, sigma, mu, nu] = riemann[rho, sigma, mu, nu]

                    # Ricci contribution
                    weyl[rho, sigma, mu, nu] -= (
                        g[rho, mu] * ricci[sigma, nu]
                        - g[rho, nu] * ricci[sigma, mu]
                        + g[sigma, nu] * ricci[rho, mu]
                        - g[sigma, mu] * ricci[rho, nu]
                    ) / 2.0

                    # Scalar curvature contribution
                    weyl[rho, sigma, mu, nu] += R * (
                        g[rho, mu] * g[sigma, nu]
                        - g[rho, nu] * g[sigma, mu]
                    ) / 6.0

    return weyl


def compute_all_tensors(
    metric_func: Callable,
    coords: np.ndarray,
    backend: BackendType = "finite_difference",
    h: float = 1e-6
) -> dict:
    """
    Compute all standard GR tensors at given coordinates.

    Returns a dictionary with:
    - 'metric': g_{μν}
    - 'metric_inverse': g^{μν}
    - 'christoffel': Γ^μ_{αβ}
    - 'riemann': R^ρ_{σμν}
    - 'riemann_lower': R_{ρσμν}
    - 'ricci': R_{μν}
    - 'ricci_scalar': R
    - 'einstein': G_{μν}
    - 'weyl': C_{ρσμν}

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
        Dictionary of tensors.
    """
    g = metric_func(*coords)
    g_inv = compute_metric_inverse(g)
    christoffel = compute_christoffel(metric_func, coords, backend, h)
    riemann = compute_riemann(metric_func, coords, backend, h)
    riemann_lower = np.einsum('rl,lsmn->rsmn', g, riemann)
    ricci = np.einsum('rmrn->mn', riemann)
    R = float(np.einsum('mn,mn->', g_inv, ricci))
    einstein = ricci - 0.5 * g * R
    weyl = compute_weyl(metric_func, coords, backend, h)

    return {
        'metric': g,
        'metric_inverse': g_inv,
        'christoffel': christoffel,
        'riemann': riemann,
        'riemann_lower': riemann_lower,
        'ricci': ricci,
        'ricci_scalar': R,
        'einstein': einstein,
        'weyl': weyl,
    }


def vectorized_christoffel(
    metric_func: Callable,
    coords_array: np.ndarray,
    backend: BackendType = "finite_difference",
    h: float = 1e-6
) -> np.ndarray:
    """
    Compute Christoffel symbols at multiple points.

    Parameters
    ----------
    metric_func : callable
        Function (t, x, y, z) -> g_{μν}.
    coords_array : np.ndarray
        Array of coordinates, shape (N, 4).
    backend : str
        Derivative backend.
    h : float
        Step size for finite differences.

    Returns
    -------
    np.ndarray
        Christoffel symbols at each point, shape (N, 4, 4, 4).
    """
    N = coords_array.shape[0]
    result = np.zeros((N, 4, 4, 4))

    for i in range(N):
        result[i] = compute_christoffel(metric_func, coords_array[i], backend, h)

    return result
