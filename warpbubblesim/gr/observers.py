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
from warpbubblesim.gr.adm import metric_to_adm


def eulerian_observer(
    metric_func: Callable,
    coords: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct the Eulerian (slice-normal) observer.

    The Eulerian observer's worldline is orthogonal to the constant-t
    hypersurfaces.  Its 4-velocity is the future-directed unit normal:

        n^μ = (1/α)(1, -β^i)

    where α is the lapse and β^i is the shift, both extracted from the
    metric via metric_to_adm.  For α = 1, β = 0 the Eulerian observer
    coincides with the coordinate-static observer (1, 0, 0, 0); for any
    warp metric (β ≠ 0 in the wall) they differ materially.

    Use static_observer() if you specifically want the coordinate-time
    direction normalised — that observer is NOT the Eulerian one in any
    metric with non-zero shift.

    Parameters
    ----------
    metric_func : callable
        Function (t, x, y, z) -> g_{μν}.
    coords : np.ndarray
        Coordinates [t, x, y, z].

    Returns
    -------
    tuple
        (u_upper, u_lower) - contravariant and covariant 4-velocity
        of the slice-normal observer; satisfies g_{μν} u^μ u^ν = -1.
    """
    g = metric_func(*coords)
    lapse, shift, _ = metric_to_adm(g)

    if lapse <= 0.0:
        raise ValueError(
            f"Eulerian observer is undefined where the metric is "
            f"non-Lorentzian (lapse = {lapse:.3e})"
        )

    u_upper = np.empty(4)
    u_upper[0] = 1.0 / lapse
    u_upper[1:] = -np.asarray(shift) / lapse

    # Lower index for callers that need it.
    u_lower = g @ u_upper

    return u_upper, u_lower


def static_observer(
    metric_func: Callable,
    coords: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct the coordinate-static observer u^μ ∝ (1, 0, 0, 0).

    This observer holds a fixed spatial coordinate position, not a
    fixed slice-normal direction.  In a Minkowski metric they coincide
    with the Eulerian observer; in a warp metric (β ≠ 0) they do not.

    Use eulerian_observer() for the slice-normal observer that
    measures the physically meaningful "Eulerian energy density".
    """
    g = metric_func(*coords)
    u_raw = np.array([1.0, 0.0, 0.0, 0.0])
    norm_sq = np.einsum('mn,m,n->', g, u_raw, u_raw)
    if norm_sq >= 0:
        raise ValueError(
            "Coordinate time direction is not timelike at this point"
        )
    u_upper = u_raw / np.sqrt(-norm_sq)
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
    shift_func: Optional[Callable] = None,
    coords: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct an observer comoving with the warp bubble.

    At any point inside the bubble's flat-interior region, the
    coordinate-static observer is the Eulerian (slice-normal) one
    -- the bubble carries this observer along with itself, hence the
    "comoving" label.  The shift information needed to build n^μ is
    extracted from the metric directly via metric_to_adm; the
    `shift_func` argument is retained only for API back-compat and
    is no longer used.

    Parameters
    ----------
    metric_func : callable
        Function (t, x, y, z) -> g_{μν}.
    coords : np.ndarray
        Coordinates [t, x, y, z].
    shift_func : callable, optional
        Unused.  Kept for backward-compatibility with callers that
        pass it positionally; will be removed in a future release.

    Returns
    -------
    tuple
        (u_upper, u_lower) - 4-velocity of the bubble-comoving
        (Eulerian) observer.
    """
    if shift_func is not None:
        import warnings
        warnings.warn(
            "warp_bubble_center_observer no longer uses shift_func; "
            "lapse and shift are extracted from metric_func via "
            "metric_to_adm.  The shift_func argument is deprecated "
            "and will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
    if coords is None:
        raise TypeError(
            "warp_bubble_center_observer: coords is required"
        )
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
    receiver_velocity: Optional[np.ndarray] = None,
) -> float:
    """
    Static-observer gravitational redshift between two coordinate points.

    For two observers held at fixed spatial coordinates in a stationary
    metric (Killing time symmetry), the ratio of their proper-time
    rates is

        1 + z = √(-g_{00}(receiver)) / √(-g_{00}(emitter)),

    which reduces to the familiar Schwarzschild gravitational redshift.

    Limitations
    -----------
    This function does NOT compute the full gravitational + kinematic
    redshift: the correct general formula

        1 + z = (k_μ u^μ)_emit / (k_μ u^μ)_recv

    requires parallel-transporting the photon 4-momentum k^μ along the
    null geodesic that connects the two events.  That construction is
    not implemented here and is non-trivial in any metric where the
    photon path is not radial.

    To prevent silent wrong answers, this function raises
    ``NotImplementedError`` if either ``emitter_velocity`` or
    ``receiver_velocity`` is given (those would require the kinematic
    contribution).  Use it only for two coordinate-static observers in
    a stationary spacetime; for warp metrics, expect at most a coarse
    qualitative comparison.

    Parameters
    ----------
    metric_func : callable
        Function (t, x, y, z) -> g_{μν}.
    emitter_coords, receiver_coords : np.ndarray
        Coordinates of emitter / receiver.  Currently must be stationary
        observers (no velocity argument).

    Returns
    -------
    float
        Redshift z (positive for redshift, negative for blueshift).

    Raises
    ------
    NotImplementedError
        If a non-trivial emitter or receiver velocity is supplied.
    """
    if emitter_velocity is not None or receiver_velocity is not None:
        raise NotImplementedError(
            "compute_redshift currently only supports two coordinate-"
            "static observers in a stationary metric.  General "
            "gravitational+kinematic redshift requires parallel-"
            "transporting the photon 4-momentum along the connecting "
            "null geodesic, which is not implemented in this module.  "
            "If you need it, integrate the photon explicitly with "
            "integrate_null_geodesic and contract the transported "
            "k_μ with each observer's 4-velocity."
        )

    # Static-observer gravitational redshift only.
    g_e = metric_func(*emitter_coords)
    g_r = metric_func(*receiver_coords)
    g00_e = float(g_e[0, 0])
    g00_r = float(g_r[0, 0])
    if g00_e >= 0.0 or g00_r >= 0.0:
        raise ValueError(
            "compute_redshift: g_{00} is non-negative at one of the "
            "endpoints -- the coordinate-time direction is not "
            "timelike there, so the static observer is undefined."
        )
    return float(np.sqrt(-g00_r) / np.sqrt(-g00_e) - 1.0)


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

    # a^μ = -R^μ_{νρσ} u^ν ξ^ρ u^σ  (MTW eq. 11.10).
    # The separation ξ goes in the third Riemann slot (between two
    # 4-velocities), not the fourth -- Riemann is antisymmetric in the
    # last pair, so swapping the last two arrays in einsum flips the
    # sign of the result.  An earlier version used (u, u, ξ) and
    # returned -1 × the correct tidal acceleration.
    a = -np.einsum('mnrs,n,r,s->m', riemann, observer_velocity,
                   separation, observer_velocity)

    return a
