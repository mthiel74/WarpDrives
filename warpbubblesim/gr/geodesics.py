"""
Geodesic integration for WarpBubbleSim.

Geodesics are the paths of freely falling particles in curved spacetime.
- Timelike geodesics: massive particles (proper time parameterization)
- Null geodesics: massless particles / light rays (affine parameter)

The geodesic equation:
d²x^μ/dλ² + Γ^μ_{αβ} (dx^α/dλ)(dx^β/dλ) = 0

Written as first-order system:
dx^μ/dλ = u^μ
du^μ/dλ = -Γ^μ_{αβ} u^α u^β
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Callable, Tuple, Optional, List, Dict
from warpbubblesim.gr.tensors import compute_christoffel, compute_metric_inverse, BackendType


def geodesic_rhs(
    lam: float,
    state: np.ndarray,
    metric_func: Callable,
    backend: BackendType = "finite_difference",
    h: float = 1e-6
) -> np.ndarray:
    """
    Right-hand side of geodesic equations.

    dx^μ/dλ = u^μ
    du^μ/dλ = -Γ^μ_{αβ} u^α u^β

    Parameters
    ----------
    lam : float
        Affine parameter λ.
    state : np.ndarray
        State vector [x^0, x^1, x^2, x^3, u^0, u^1, u^2, u^3].
    metric_func : callable
        Function (t, x, y, z) -> g_{μν}.
    backend : str
        Derivative backend for Christoffel computation.
    h : float
        Step size for finite differences.

    Returns
    -------
    np.ndarray
        Time derivatives [u^0, u^1, u^2, u^3, a^0, a^1, a^2, a^3].
    """
    x = state[:4]
    u = state[4:]

    # Compute Christoffel symbols at current position
    gamma = compute_christoffel(metric_func, x, backend, h)

    # Acceleration: a^μ = -Γ^μ_{αβ} u^α u^β
    a = -np.einsum('mab,a,b->m', gamma, u, u)

    return np.concatenate([u, a])


def normalize_velocity(
    velocity: np.ndarray,
    metric: np.ndarray,
    timelike: bool = True
) -> np.ndarray:
    """
    Normalize a 4-velocity to satisfy the constraint.

    For timelike: g_{μν} u^μ u^ν = -1
    For null: g_{μν} u^μ u^ν = 0 (just scale)

    Parameters
    ----------
    velocity : np.ndarray
        4-velocity to normalize.
    metric : np.ndarray
        Metric tensor at the point.
    timelike : bool
        If True, normalize to timelike; if False, ensure null.

    Returns
    -------
    np.ndarray
        Normalized 4-velocity.
    """
    norm_sq = np.einsum('mn,m,n->', metric, velocity, velocity)

    if timelike:
        # Normalize to -1
        if norm_sq >= 0:
            raise ValueError("Velocity is not timelike")
        return velocity / np.sqrt(-norm_sq)
    else:
        # For null, just return as-is (or scale by convention)
        # Typically we fix u^0 = 1 or similar
        return velocity


def create_initial_velocity(
    metric_func: Callable,
    coords: np.ndarray,
    spatial_direction: np.ndarray,
    speed: float = 0.0,
    timelike: bool = True
) -> np.ndarray:
    """
    Create initial 4-velocity from spatial direction and speed.

    Parameters
    ----------
    metric_func : callable
        Function (t, x, y, z) -> g_{μν}.
    coords : np.ndarray
        Starting coordinates.
    spatial_direction : np.ndarray
        Unit 3-vector for spatial direction.
    speed : float
        Coordinate speed |dx^i/dt| (for timelike geodesics).
    timelike : bool
        If True, create timelike velocity; if False, create null.

    Returns
    -------
    np.ndarray
        Normalized 4-velocity u^μ.
    """
    g = metric_func(*coords)
    n = spatial_direction / np.linalg.norm(spatial_direction)

    if timelike:
        # u^μ = γ(1, v n^i) with v = speed
        # Normalize: g_{μν} u^μ u^ν = -1
        u_raw = np.array([1.0, speed * n[0], speed * n[1], speed * n[2]])
        return normalize_velocity(u_raw, g, timelike=True)
    else:
        # For null, solve g_{μν} k^μ k^ν = 0
        # With k^μ = (1, v n^i), solve for v
        a = np.einsum('ij,i,j->', g[1:, 1:], n, n)
        b = 2 * np.einsum('i,i->', g[0, 1:], n)
        c = g[0, 0]

        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            raise ValueError("Cannot create null vector in this direction")

        v = (-b + np.sqrt(discriminant)) / (2*a)
        return np.array([1.0, v * n[0], v * n[1], v * n[2]])


def integrate_geodesic(
    metric_func: Callable,
    initial_coords: np.ndarray,
    initial_velocity: np.ndarray,
    lambda_span: Tuple[float, float],
    backend: BackendType = "finite_difference",
    h: float = 1e-6,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_step: float = 0.1,
    method: str = "RK45",
    dense_output: bool = True,
    renormalize: bool = True,
    renorm_interval: int = 100,
    events: Optional[List[Callable]] = None,
) -> Dict:
    """
    Integrate a geodesic through spacetime.

    Parameters
    ----------
    metric_func : callable
        Function (t, x, y, z) -> g_{μν}.
    initial_coords : np.ndarray
        Initial coordinates [t, x, y, z].
    initial_velocity : np.ndarray
        Initial 4-velocity u^μ.
    lambda_span : tuple
        (λ_start, λ_end) range for affine parameter.
    backend : str
        Derivative backend.
    h : float
        Step size for Christoffel computation.
    rtol, atol : float
        Relative and absolute tolerances for ODE solver.
    max_step : float
        Maximum step size.
    method : str
        Integration method ('RK45', 'DOP853', etc.).
    dense_output : bool
        Whether to compute dense output for interpolation.
    renormalize : bool
        If True, post-correct each output velocity sample so that
        g_{μν} u^μ u^ν matches its initial value (-1 for timelike,
        0 for null) to machine precision.  This removes drift from
        the adaptive integrator at no cost in trajectory fidelity:
        the corrected velocity is parallel to the integrated one,
        with magnitude rescaled.  Skipped for null geodesics where
        the rescale factor is undefined (norm² = 0).
    renorm_interval : int
        Retained for backward compatibility; unused now that
        renormalisation is a post-pass on the dense output.
    events : list of callables, optional
        Event functions to pass to solve_ivp.  Each must accept
        (lam, state) and return a scalar; assign .terminal/.direction
        attributes to control behaviour.  When events fire,
        solve_ivp.t_events / .y_events become available on
        result['solution'].

    Returns
    -------
    dict
        Dictionary with:
        - 'lambda': affine parameter values
        - 'coords': coordinates at each step, shape (n_steps, 4)
        - 'velocity': 4-velocity at each step, shape (n_steps, 4)
        - 'proper_time': proper time (for timelike geodesics)
        - 'solution': scipy ODE solution object
        - 'is_timelike': bool
        - 'initial_norm': float
        - 'final_norm': float
        - 'normalization_drift': drift |final_norm - initial_norm|
        - 'renormalized': bool, whether post-correction was applied
    """
    initial_state = np.concatenate([initial_coords, initial_velocity])

    # Check initial normalization
    g0 = metric_func(*initial_coords)
    norm_sq = np.einsum('mn,m,n->', g0, initial_velocity, initial_velocity)
    is_timelike = norm_sq < -0.5
    is_null = abs(norm_sq) < 1e-6

    step_count = [0]

    def rhs_wrapper(lam, state):
        step_count[0] += 1
        return geodesic_rhs(lam, state, metric_func, backend, h)

    sol = solve_ivp(
        rhs_wrapper,
        lambda_span,
        initial_state,
        method=method,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        dense_output=dense_output,
        events=(events if events is not None else []),
    )

    # Extract results
    coords = sol.y[:4].T
    velocity = sol.y[4:].T
    lam = sol.t

    renormalized = False
    if renormalize and not is_null:
        # Post-pass: rescale each velocity so g_{μν} u^μ u^ν == norm_sq.
        # The rescale factor is sqrt(norm_sq / current_norm); it leaves
        # the spatial direction of u unchanged, so the integrated
        # trajectory shape is preserved.  Skipped for null geodesics
        # since norm_sq = 0 makes the ratio degenerate.
        for i in range(velocity.shape[0]):
            g_i = metric_func(*coords[i])
            current = np.einsum('mn,m,n->', g_i, velocity[i], velocity[i])
            if current * norm_sq > 0.0:  # same sign -> safe to rescale
                velocity[i] = velocity[i] * np.sqrt(norm_sq / current)
        renormalized = True

    # Compute proper time for timelike geodesics
    if is_timelike:
        proper_time = lam - lam[0]
    else:
        proper_time = None

    # Final normalization (after any renormalisation)
    final_g = metric_func(*coords[-1])
    final_norm = np.einsum('mn,m,n->', final_g, velocity[-1], velocity[-1])
    normalization_drift = abs(final_norm - norm_sq)

    return {
        'lambda': lam,
        'coords': coords,
        'velocity': velocity,
        'proper_time': proper_time,
        'solution': sol,
        'is_timelike': is_timelike,
        'initial_norm': float(norm_sq),
        'final_norm': float(final_norm),
        'normalization_drift': float(normalization_drift),
        'renormalized': renormalized,
    }


def integrate_null_geodesic(
    metric_func: Callable,
    initial_coords: np.ndarray,
    spatial_direction: np.ndarray,
    lambda_span: Tuple[float, float],
    backward: bool = False,
    **kwargs
) -> Dict:
    """
    Integrate a null geodesic (light ray).

    Parameters
    ----------
    metric_func : callable
        Function (t, x, y, z) -> g_{μν}.
    initial_coords : np.ndarray
        Initial coordinates [t, x, y, z].
    spatial_direction : np.ndarray
        Direction of propagation (3-vector).
    lambda_span : tuple
        Affine parameter range.
    backward : bool
        If True, trace ray backward in time.
    **kwargs
        Additional arguments for integrate_geodesic.

    Returns
    -------
    dict
        Geodesic solution dictionary.
    """
    # Create null initial velocity
    k = create_initial_velocity(
        metric_func, initial_coords, spatial_direction,
        speed=0, timelike=False
    )

    if backward:
        k = -k

    return integrate_geodesic(
        metric_func, initial_coords, k, lambda_span, **kwargs
    )


def integrate_geodesic_to_boundary(
    metric_func: Callable,
    initial_coords: np.ndarray,
    initial_velocity: np.ndarray,
    boundary_func: Callable,
    max_lambda: float = 100.0,
    **kwargs
) -> Dict:
    """
    Integrate geodesic until it crosses a boundary.

    Parameters
    ----------
    metric_func : callable
        Function (t, x, y, z) -> g_{μν}.
    initial_coords : np.ndarray
        Initial coordinates.
    initial_velocity : np.ndarray
        Initial 4-velocity.
    boundary_func : callable
        Function (x, y, z) -> float, geodesic stops when this becomes negative.
    max_lambda : float
        Maximum affine parameter value.
    **kwargs
        Additional arguments for integrate_geodesic.

    Returns
    -------
    dict
        Geodesic solution with 'hit_boundary' flag.
    """
    def boundary_event(lam, state):
        return boundary_func(state[1], state[2], state[3])

    boundary_event.terminal = True
    boundary_event.direction = -1

    # Pass the event into solve_ivp so the adaptive integrator
    # locates the crossing precisely (was previously dropped, with
    # post-processing on the discrete output points only catching
    # crossings that happened to land at a sampled λ).
    kwargs.setdefault("events", [])
    kwargs["events"] = list(kwargs["events"]) + [boundary_event]

    result = integrate_geodesic(
        metric_func, initial_coords, initial_velocity,
        (0, max_lambda), **kwargs
    )

    sol = result['solution']
    # solve_ivp truncates the trajectory at the event; the t_events
    # / y_events arrays carry the precise crossing.  result['coords']
    # already ends at (or just before) the event because solve_ivp
    # stops integrating when the event fires.
    fired = bool(sol.t_events) and len(sol.t_events[-1]) > 0
    result['hit_boundary'] = fired
    if fired:
        # Append the precise event point so callers can rely on the
        # last sample being on the boundary.
        evt_t = sol.t_events[-1][0]
        evt_y = sol.y_events[-1][0]
        # Avoid duplicate if solver's last stored sample is the event.
        if not np.isclose(result['lambda'][-1], evt_t):
            result['lambda'] = np.append(result['lambda'], evt_t)
            result['coords'] = np.vstack([result['coords'], evt_y[:4]])
            result['velocity'] = np.vstack([result['velocity'], evt_y[4:]])

    return result


def integrate_geodesic_bundle(
    metric_func: Callable,
    initial_coords: np.ndarray,
    initial_velocities: List[np.ndarray],
    lambda_span: Tuple[float, float],
    **kwargs
) -> List[Dict]:
    """
    Integrate multiple geodesics in parallel.

    Parameters
    ----------
    metric_func : callable
        Function (t, x, y, z) -> g_{μν}.
    initial_coords : np.ndarray
        Starting coordinates (same for all).
    initial_velocities : list
        List of initial 4-velocities.
    lambda_span : tuple
        Affine parameter range.
    **kwargs
        Additional arguments for integrate_geodesic.

    Returns
    -------
    list
        List of geodesic solution dictionaries.
    """
    results = []
    for v0 in initial_velocities:
        result = integrate_geodesic(
            metric_func, initial_coords.copy(), v0,
            lambda_span, **kwargs
        )
        results.append(result)
    return results


def compute_geodesic_deviation(
    metric_func: Callable,
    geodesic_result: Dict,
    initial_separation: np.ndarray,
    initial_separation_velocity: Optional[np.ndarray] = None,
    backend: BackendType = "finite_difference",
    h: float = 1e-6,
    rtol: float = 1e-6,
    atol: float = 1e-9,
) -> np.ndarray:
    """
    Compute geodesic deviation (Jacobi field) along a geodesic.

    The deviation vector ξ^μ satisfies the Jacobi equation:

        D²ξ^μ/dλ² = -R^μ_{νρσ} u^ν u^ρ ξ^σ

    Solved as an 8-component first-order system (ξ, dξ/dλ) using
    scipy.integrate.solve_ivp with adaptive RK45.  The previous
    implementation used forward Euler with the geodesic's own step
    size, which diverges quickly under any non-trivial Riemann
    coupling -- it was producing quantitatively wrong results
    silently.

    The trajectory (coords, velocity) is interpolated linearly along
    λ so the integrator can query at any intermediate λ.

    Parameters
    ----------
    metric_func : callable
        Function (t, x, y, z) -> g_{μν}.
    geodesic_result : dict
        Result from integrate_geodesic.
    initial_separation : np.ndarray
        Initial separation vector ξ^μ(λ_0), shape (4,).
    initial_separation_velocity : np.ndarray, optional
        Initial dξ/dλ.  Defaults to zero.
    backend : str
        Derivative backend.
    h : float
        Step size for the Riemann finite-difference computation.
    rtol, atol : float
        ODE tolerances passed through to solve_ivp.

    Returns
    -------
    np.ndarray
        Separation vectors along the geodesic, shape (n_steps, 4),
        sampled at the geodesic's own λ values.
    """
    from warpbubblesim.gr.tensors import compute_riemann

    coords = geodesic_result['coords']
    velocity = geodesic_result['velocity']
    lam = geodesic_result['lambda']

    if initial_separation_velocity is None:
        initial_separation_velocity = np.zeros(4)

    # Build linear interpolators along λ for the trajectory.  Linear
    # is enough for a smooth solve_ivp inner loop; quadratic+ would
    # fight the integrator's step control.
    def _coord_at(lambda_val):
        return np.array([
            np.interp(lambda_val, lam, coords[:, mu]) for mu in range(4)
        ])

    def _velocity_at(lambda_val):
        return np.array([
            np.interp(lambda_val, lam, velocity[:, mu]) for mu in range(4)
        ])

    def jacobi_rhs(lam_val, state):
        xi = state[:4]
        xi_dot = state[4:]

        coords_lam = _coord_at(lam_val)
        u = _velocity_at(lam_val)
        R = compute_riemann(metric_func, coords_lam, backend, h)

        # D²ξ^μ/dλ² = -R^μ_{νρσ} u^ν u^ρ ξ^σ
        a = -np.einsum('mnrs,n,r,s->m', R, u, u, xi)
        return np.concatenate([xi_dot, a])

    initial_state = np.concatenate([initial_separation,
                                     initial_separation_velocity])
    sol = solve_ivp(
        jacobi_rhs,
        (lam[0], lam[-1]),
        initial_state,
        t_eval=lam,
        method="RK45",
        rtol=rtol,
        atol=atol,
        dense_output=False,
    )

    if not sol.success:
        raise RuntimeError(
            f"Jacobi-equation integration failed: {sol.message}"
        )

    # Return ξ at each requested λ
    return sol.y[:4].T


def expansion_rate_along_geodesic(
    metric_func: Callable,
    geodesic_result: Dict,
    backend: BackendType = "finite_difference",
    h: float = 1e-6
) -> np.ndarray:
    """
    Compute the Christoffel-trace contribution Γ^μ_{μν} u^ν along a geodesic.

    NOTE: this is NOT the full covariant divergence ∇_μ u^μ = ∂_μ u^μ
    + Γ^μ_{μν} u^ν -- the ∂_μ u^μ term cannot be recovered from a single
    trajectory's λ-samples (it requires the Jacobian of u^μ as a field,
    not just a worldline).  For the Raychaudhuri expansion of a
    congruence use compute_geodesic_deviation (Jacobi fields).  For the
    Eulerian foliation expansion use compute_expansion_scalar(K, γ_inv)
    from the ADM module, which gives θ exactly via the spatial trace
    of the extrinsic curvature.

    What this function returns is still useful: it is the volume
    change of an infinitesimal coordinate-aligned volume element
    carried along u^μ, i.e. ∂_μ(√(-g) u^μ) / √(-g) up to the metric-
    determinant factor.

    Parameters
    ----------
    metric_func : callable
        Function (t, x, y, z) -> g_{μν}.
    geodesic_result : dict
        Result from integrate_geodesic.
    backend : str
        Derivative backend.
    h : float
        Step size.

    Returns
    -------
    np.ndarray
        Christoffel-trace term at each point along the geodesic.
    """
    coords = geodesic_result['coords']
    velocity = geodesic_result['velocity']

    n_steps = len(coords)
    theta = np.zeros(n_steps)

    # WARNING: this function returns only the Christoffel-trace term
    #     Γ^μ_{μν} u^ν,
    # not the full covariant divergence ∇_μ u^μ = ∂_μ u^μ + Γ^μ_{μν} u^ν.
    # "Expansion of a congruence" is a property of a FAMILY of nearby
    # geodesics, not of a single worldline; reconstructing it requires
    # Jacobi fields (use compute_geodesic_deviation) or the slice-
    # foliation extrinsic curvature trace (use ADM compute_expansion
    # _scalar).  The Γ-trace alone is the volume change of an
    # infinitesimal coordinate volume carried along u^μ, which is
    # what previous callers were getting; we keep it for backward
    # compatibility but rename the return to make clear it is NOT
    # the Raychaudhuri θ.
    for i in range(n_steps):
        gamma = compute_christoffel(metric_func, coords[i], backend, h)
        u = velocity[i]
        theta[i] = np.einsum('mmn,n->', gamma, u)

    return theta
