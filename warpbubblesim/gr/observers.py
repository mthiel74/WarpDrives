"""
Observer utilities for WarpBubbleSim.

Different observers measure different physical quantities.
This module provides tools for computing observables in various frames:
- Eulerian (coordinate) observers
- Ship-comoving observers
- Arbitrary timelike observers
"""

import numpy as np
from typing import Callable, Tuple, Optional
from warpbubblesim.gr.tensors import compute_metric_inverse, BackendType
from warpbubblesim.gr.energy import compute_stress_energy


def eulerian_observer(
    metric_func: Callable,
    coords: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct Eulerian (coordinate-stationary) observer.

    The Eulerian observer has zero spatial velocity in the coordinates.
    Their 4-velocity is the unit normal to constant-t hypersurfaces.

    For ADM metric with α=1, γ_{ij}=δ_{ij}:
    u^μ = (1, -β^x, -β^y, -β^z) / ||...||

    For general metric, we normalize u^μ = (1, 0, 0, 0) / ||...||

    Parameters
    ----------
    metric_func : callable
        Function (t, x, y, z) -> g_{μν}.
    coords : np.ndarray
        Coordinates [t, x, y, z].

    Returns
    -------
    tuple
        (u_upper, u_lower) - contravariant and covariant 4-velocity.
    """
    g = metric_func(*coords)

    # Start with coordinate-stationary: u^μ ∝ (1, 0, 0, 0)
    u_raw = np.array([1.0, 0.0, 0.0, 0.0])

    # Normalize: g_{μν} u^μ u^ν = -1
    norm_sq = np.einsum('mn,m,n->', g, u_raw, u_raw)

    # For signature (-,+,+,+), we need norm_sq < 0
    if norm_sq >= 0:
        raise ValueError("Coordinate time direction is not timelike at this point")

    u_upper = u_raw / np.sqrt(-norm_sq)

    # Lower index
    u_lower = g @ u_upper

    return u_upper, u_lower


def ship_comoving_observer(
    metric_func: Callable,
    coords: np.ndarray,
    ship_velocity: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct observer comoving with a "ship" at specified 3-velocity.

    Parameters
    ----------
    metric_func : callable
        Function (t, x, y, z) -> g_{μν}.
    coords : np.ndarray
        Coordinates [t, x, y, z].
    ship_velocity : np.ndarray
        3-velocity v^i of the ship in coordinates, shape (3,).

    Returns
    -------
    tuple
        (u_upper, u_lower) - 4-velocity of ship-comoving observer.
    """
    g = metric_func(*coords)

    # Construct 4-velocity: u^μ = γ (1, v^i) where γ normalizes
    u_raw = np.array([1.0, ship_velocity[0], ship_velocity[1], ship_velocity[2]])

    # Normalize
    norm_sq = np.einsum('mn,m,n->', g, u_raw, u_raw)

    if norm_sq >= 0:
        raise ValueError("Specified velocity is not subluminal at this point")

    u_upper = u_raw / np.sqrt(-norm_sq)
    u_lower = g @ u_upper

    return u_upper, u_lower


def warp_bubble_center_observer(
    metric_func: Callable,
    shift_func: Callable,
    coords: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct observer at center of warp bubble.

    At the bubble center (where shift β ≈ (v_s, 0, 0)),
    an observer comoving with the bubble has 4-velocity:
    u^μ = (1, 0, 0, 0) / ||...||

    This observer experiences flat spacetime locally (equivalence principle).

    Parameters
    ----------
    metric_func : callable
        Function (t, x, y, z) -> g_{μν}.
    shift_func : callable
        Function (t, x, y, z) -> β^i.
    coords : np.ndarray
        Coordinates [t, x, y, z].

    Returns
    -------
    tuple
        (u_upper, u_lower) - 4-velocity of bubble-center observer.
    """
    # At bubble center, the comoving observer is the Eulerian observer
    return eulerian_observer(metric_func, coords)


def compute_proper_time_rate(
    metric_func: Callable,
    coords: np.ndarray,
    coordinate_velocity: Optional[np.ndarray] = None
) -> float:
    """
    Compute rate of proper time vs coordinate time.

    dτ/dt = √(-g_{μν} (dx^μ/dt)(dx^ν/dt))

    For an observer at rest in coordinates: dτ/dt = √(-g_{00})

    Parameters
    ----------
    metric_func : callable
        Function (t, x, y, z) -> g_{μν}.
    coords : np.ndarray
        Coordinates [t, x, y, z].
    coordinate_velocity : np.ndarray, optional
        Coordinate 3-velocity dx^i/dt. If None, assumes stationary.

    Returns
    -------
    float
        dτ/dt
    """
    g = metric_func(*coords)

    if coordinate_velocity is None:
        # Stationary observer
        return np.sqrt(-g[0, 0])
    else:
        # Moving observer: dτ/dt = √(-g_{μν} dx^μ/dt dx^ν/dt)
        # with dx^0/dt = 1
        v4 = np.array([1.0, coordinate_velocity[0], coordinate_velocity[1], coordinate_velocity[2]])
        norm_sq = np.einsum('mn,m,n->', g, v4, v4)
        return np.sqrt(-norm_sq)


def compute_local_speed_of_light(
    metric_func: Callable,
    coords: np.ndarray,
    direction: np.ndarray
) -> float:
    """
    Compute local coordinate speed of light in a given direction.

    For null geodesics: g_{μν} dx^μ dx^ν = 0
    Solving for dx/dt in direction n^i:
    g_{00} + 2 g_{0i} v n^i + g_{ij} v² n^i n^j = 0

    Parameters
    ----------
    metric_func : callable
        Function (t, x, y, z) -> g_{μν}.
    coords : np.ndarray
        Coordinates [t, x, y, z].
    direction : np.ndarray
        Unit 3-vector indicating direction, shape (3,).

    Returns
    -------
    float
        Coordinate speed of light |dx/dt| in given direction.
    """
    g = metric_func(*coords)
    n = direction / np.linalg.norm(direction)

    # Coefficients of quadratic: a v² + b v + c = 0
    a = np.einsum('ij,i,j->', g[1:, 1:], n, n)
    b = 2 * np.einsum('i,i->', g[0, 1:], n)
    c = g[0, 0]

    # Solve quadratic
    discriminant = b**2 - 4*a*c

    if discriminant < 0:
        raise ValueError("No real solution - direction may not be valid")

    v1 = (-b + np.sqrt(discriminant)) / (2*a)
    v2 = (-b - np.sqrt(discriminant)) / (2*a)

    # Return the positive root (forward light cone)
    if v1 > 0:
        return v1
    return v2


def compute_redshift(
    metric_func: Callable,
    emitter_coords: np.ndarray,
    receiver_coords: np.ndarray,
    emitter_velocity: Optional[np.ndarray] = None,
    receiver_velocity: Optional[np.ndarray] = None
) -> float:
    """
    Compute gravitational + kinematic redshift between two observers.

    z = (λ_received - λ_emitted) / λ_emitted = ν_e / ν_r - 1

    For stationary observers in a static metric:
    1 + z = √(-g_{00}(receiver)) / √(-g_{00}(emitter))

    Parameters
    ----------
    metric_func : callable
        Function (t, x, y, z) -> g_{μν}.
    emitter_coords : np.ndarray
        Coordinates of emitter.
    receiver_coords : np.ndarray
        Coordinates of receiver.
    emitter_velocity : np.ndarray, optional
        3-velocity of emitter.
    receiver_velocity : np.ndarray, optional
        3-velocity of receiver.

    Returns
    -------
    float
        Redshift z (positive for redshift, negative for blueshift).
    """
    # Proper time rates
    dtau_dt_e = compute_proper_time_rate(metric_func, emitter_coords, emitter_velocity)
    dtau_dt_r = compute_proper_time_rate(metric_func, receiver_coords, receiver_velocity)

    # Redshift: 1 + z = (dτ/dt)_receiver / (dτ/dt)_emitter
    return dtau_dt_r / dtau_dt_e - 1


def project_to_observer_frame(
    tensor: np.ndarray,
    observer_velocity: np.ndarray,
    metric: np.ndarray
) -> np.ndarray:
    """
    Project a tensor into an observer's local frame.

    Uses the projection tensor h_{μν} = g_{μν} + u_μ u_ν.

    Parameters
    ----------
    tensor : np.ndarray
        Tensor to project (rank 2).
    observer_velocity : np.ndarray
        Observer 4-velocity u^μ.
    metric : np.ndarray
        Metric tensor g_{μν}.

    Returns
    -------
    np.ndarray
        Projected tensor.
    """
    u_lower = metric @ observer_velocity
    h = metric + np.outer(u_lower, u_lower)

    # Project both indices
    return np.einsum('ma,nb,mn->ab', h, h, tensor)


def compute_tidal_acceleration(
    metric_func: Callable,
    coords: np.ndarray,
    observer_velocity: np.ndarray,
    separation: np.ndarray,
    backend: BackendType = "finite_difference",
    h: float = 1e-6
) -> np.ndarray:
    """
    Compute tidal acceleration between nearby geodesics.

    Uses geodesic deviation equation:
    D²ξ^μ/dτ² = -R^μ_{νρσ} u^ν u^ρ ξ^σ

    Parameters
    ----------
    metric_func : callable
        Function (t, x, y, z) -> g_{μν}.
    coords : np.ndarray
        Coordinates [t, x, y, z].
    observer_velocity : np.ndarray
        Observer 4-velocity u^μ.
    separation : np.ndarray
        Separation 4-vector ξ^μ.
    backend : str
        Derivative backend.
    h : float
        Step size for finite differences.

    Returns
    -------
    np.ndarray
        Tidal acceleration a^μ.
    """
    from warpbubblesim.gr.tensors import compute_riemann

    riemann = compute_riemann(metric_func, coords, backend, h)

    # a^μ = -R^μ_{νρσ} u^ν u^ρ ξ^σ
    a = -np.einsum('mnrs,n,r,s->m', riemann, observer_velocity,
                   observer_velocity, separation)

    return a
