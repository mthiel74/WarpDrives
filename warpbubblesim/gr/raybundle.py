"""
Ray bundle tracing for visualizing light propagation and lensing.

Creates bundles of null geodesics to visualize:
- Gravitational lensing
- Light cone structure
- Horizon-like behavior in warp metrics
"""

import numpy as np
from typing import Callable, List, Dict, Tuple, Optional
from warpbubblesim.gr.geodesics import integrate_null_geodesic, create_initial_velocity


def create_ray_bundle(
    origin: np.ndarray,
    center_direction: np.ndarray,
    half_angle: float,
    n_rays: int,
    pattern: str = "circular"
) -> List[np.ndarray]:
    """
    Create a bundle of ray directions around a central direction.

    Parameters
    ----------
    origin : np.ndarray
        Origin point [t, x, y, z] (for reference, not used in directions).
    center_direction : np.ndarray
        Central direction 3-vector.
    half_angle : float
        Half-angle of the cone in radians.
    n_rays : int
        Number of rays in the bundle.
    pattern : str
        Distribution pattern: 'circular', 'grid', 'random'.

    Returns
    -------
    list
        List of unit 3-vectors representing ray directions.
    """
    # Normalize center direction
    center = center_direction / np.linalg.norm(center_direction)

    # Find perpendicular vectors
    if abs(center[2]) < 0.9:
        perp1 = np.cross(center, [0, 0, 1])
    else:
        perp1 = np.cross(center, [1, 0, 0])
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(center, perp1)

    directions = []

    if pattern == "circular":
        # Rays in a circular pattern around center
        n_rings = max(1, int(np.sqrt(n_rays / np.pi)))
        for ring in range(n_rings + 1):
            if ring == 0:
                directions.append(center.copy())
            else:
                angle_from_center = half_angle * ring / n_rings
                n_in_ring = max(6, int(2 * np.pi * ring))
                for i in range(n_in_ring):
                    phi = 2 * np.pi * i / n_in_ring
                    # Rotate center by angle_from_center in direction phi
                    d = (np.cos(angle_from_center) * center +
                         np.sin(angle_from_center) * (np.cos(phi) * perp1 + np.sin(phi) * perp2))
                    directions.append(d / np.linalg.norm(d))

    elif pattern == "grid":
        # Grid pattern in angular coordinates
        n_side = int(np.sqrt(n_rays))
        for i in range(n_side):
            for j in range(n_side):
                theta = half_angle * (2 * i / (n_side - 1) - 1) if n_side > 1 else 0
                phi_offset = half_angle * (2 * j / (n_side - 1) - 1) if n_side > 1 else 0

                d = (center +
                     np.tan(theta) * perp1 +
                     np.tan(phi_offset) * perp2)
                directions.append(d / np.linalg.norm(d))

    elif pattern == "random":
        # Random directions within cone
        for _ in range(n_rays):
            # Random point in cone
            theta = half_angle * np.sqrt(np.random.random())
            phi = 2 * np.pi * np.random.random()

            d = (np.cos(theta) * center +
                 np.sin(theta) * (np.cos(phi) * perp1 + np.sin(phi) * perp2))
            directions.append(d / np.linalg.norm(d))

    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    return directions


def trace_ray_bundle(
    metric_func: Callable,
    origin: np.ndarray,
    directions: List[np.ndarray],
    lambda_max: float,
    backward: bool = False,
    **kwargs
) -> List[Dict]:
    """
    Trace a bundle of null geodesics.

    Parameters
    ----------
    metric_func : callable
        Function (t, x, y, z) -> g_{μν}.
    origin : np.ndarray
        Starting point [t, x, y, z].
    directions : list
        List of spatial direction 3-vectors.
    lambda_max : float
        Maximum affine parameter.
    backward : bool
        If True, trace backward in time.
    **kwargs
        Additional arguments for integrate_null_geodesic.

    Returns
    -------
    list
        List of geodesic result dictionaries.
    """
    results = []
    for direction in directions:
        result = integrate_null_geodesic(
            metric_func, origin.copy(), direction,
            (0, lambda_max), backward=backward, **kwargs
        )
        results.append(result)
    return results


def create_light_cone(
    metric_func: Callable,
    apex: np.ndarray,
    time_extent: float,
    n_rays: int = 50,
    future: bool = True,
    past: bool = True,
    **kwargs
) -> Dict[str, List[Dict]]:
    """
    Create future and/or past light cones at a point.

    Parameters
    ----------
    metric_func : callable
        Function (t, x, y, z) -> g_{μν}.
    apex : np.ndarray
        Apex of light cone [t, x, y, z].
    time_extent : float
        How far to trace in coordinate time.
    n_rays : int
        Number of rays per cone.
    future : bool
        Whether to compute future light cone.
    past : bool
        Whether to compute past light cone.
    **kwargs
        Additional arguments for ray tracing.

    Returns
    -------
    dict
        Dictionary with 'future' and 'past' lists of geodesics.
    """
    # Create uniform angular distribution
    directions = []
    n_phi = int(np.sqrt(n_rays * np.pi))
    n_theta = max(1, n_rays // n_phi)

    for i in range(n_theta):
        theta = np.pi * (i + 0.5) / n_theta
        n_phi_ring = max(1, int(n_phi * np.sin(theta)))
        for j in range(n_phi_ring):
            phi = 2 * np.pi * j / n_phi_ring
            d = np.array([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta)
            ])
            directions.append(d)

    result = {'future': [], 'past': []}

    if future:
        result['future'] = trace_ray_bundle(
            metric_func, apex, directions, time_extent,
            backward=False, **kwargs
        )

    if past:
        result['past'] = trace_ray_bundle(
            metric_func, apex, directions, time_extent,
            backward=True, **kwargs
        )

    return result


def compute_ray_bundle_area(
    bundle_results: List[Dict],
    lambda_value: float
) -> float:
    """
    Compute the cross-sectional area of a ray bundle at given λ.

    Parameters
    ----------
    bundle_results : list
        List of geodesic results from trace_ray_bundle.
    lambda_value : float
        Affine parameter at which to compute area.

    Returns
    -------
    float
        Approximate cross-sectional area.
    """
    # Interpolate positions at lambda_value
    positions = []
    for result in bundle_results:
        sol = result.get('solution')
        if sol is not None and sol.success:
            if lambda_value <= result['lambda'][-1]:
                # Interpolate
                coords = sol.sol(lambda_value)[:4]
                positions.append(coords[1:4])  # spatial part

    if len(positions) < 3:
        return 0.0

    positions = np.array(positions)

    # Compute approximate area using convex hull or covariance
    # Simple approach: use variance
    cov = np.cov(positions.T)
    # Area ~ sqrt(det(2x2 covariance)) for 2D projection
    # For 3D, use volume^(2/3) as proxy
    volume_proxy = np.sqrt(np.abs(np.linalg.det(cov)))

    return volume_proxy


def compute_magnification(
    bundle_results: List[Dict],
    lambda_initial: float,
    lambda_final: float
) -> float:
    """
    Compute gravitational magnification factor.

    Magnification = (area at λ_initial) / (area at λ_final)

    For demagnification (focus), this is > 1.
    For magnification (defocus), this is < 1.

    Parameters
    ----------
    bundle_results : list
        List of geodesic results.
    lambda_initial : float
        Initial affine parameter.
    lambda_final : float
        Final affine parameter.

    Returns
    -------
    float
        Magnification factor.
    """
    area_i = compute_ray_bundle_area(bundle_results, lambda_initial)
    area_f = compute_ray_bundle_area(bundle_results, lambda_final)

    if area_f < 1e-20:
        return np.inf  # Focus/caustic
    if area_i < 1e-20:
        return 0.0

    return area_i / area_f


def find_caustics(
    bundle_results: List[Dict],
    n_samples: int = 100,
    threshold: float = 0.1
) -> List[float]:
    """
    Find caustic points (where bundle area → 0).

    Parameters
    ----------
    bundle_results : list
        List of geodesic results.
    n_samples : int
        Number of sample points along rays.
    threshold : float
        Relative area threshold for caustic detection.

    Returns
    -------
    list
        List of affine parameter values where caustics occur.
    """
    # Find common lambda range
    lambda_max = min(r['lambda'][-1] for r in bundle_results if r['solution'].success)
    lambda_values = np.linspace(0, lambda_max, n_samples)

    areas = [compute_ray_bundle_area(bundle_results, lam) for lam in lambda_values]
    areas = np.array(areas)

    # Normalize
    if areas.max() > 0:
        areas_norm = areas / areas.max()
    else:
        return []

    # Find minima below threshold
    caustics = []
    for i in range(1, len(areas_norm) - 1):
        if (areas_norm[i] < threshold and
            areas_norm[i] < areas_norm[i-1] and
            areas_norm[i] < areas_norm[i+1]):
            caustics.append(lambda_values[i])

    return caustics


def compute_shear_from_bundle(
    bundle_results: List[Dict],
    lambda_value: float,
    center_direction: np.ndarray
) -> Tuple[float, float]:
    """
    Compute shear (distortion) of a ray bundle.

    Shear measures how circular bundles become elliptical.

    Parameters
    ----------
    bundle_results : list
        List of geodesic results.
    lambda_value : float
        Affine parameter at which to compute shear.
    center_direction : np.ndarray
        Central ray direction for reference.

    Returns
    -------
    tuple
        (shear_magnitude, shear_angle)
    """
    # Get positions
    positions = []
    for result in bundle_results:
        sol = result.get('solution')
        if sol is not None and sol.success and lambda_value <= result['lambda'][-1]:
            coords = sol.sol(lambda_value)[:4]
            positions.append(coords[1:4])

    if len(positions) < 5:
        return 0.0, 0.0

    positions = np.array(positions)
    center = positions.mean(axis=0)
    deviations = positions - center

    # Project onto plane perpendicular to center_direction
    n = center_direction / np.linalg.norm(center_direction)
    projections = deviations - np.outer(deviations @ n, n)

    # Compute 2D covariance in the projected plane
    # Find two orthogonal vectors in the plane
    if abs(n[2]) < 0.9:
        e1 = np.cross(n, [0, 0, 1])
    else:
        e1 = np.cross(n, [1, 0, 0])
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(n, e1)

    # 2D coordinates
    coords_2d = np.column_stack([projections @ e1, projections @ e2])

    if len(coords_2d) < 3:
        return 0.0, 0.0

    cov_2d = np.cov(coords_2d.T)

    # Eigenvalues give semi-axes
    eigvals = np.linalg.eigvalsh(cov_2d)
    eigvals = np.sort(eigvals)[::-1]

    if eigvals[1] > 0:
        # Ellipticity (shear magnitude)
        a, b = np.sqrt(eigvals)
        shear = (a - b) / (a + b)
    else:
        shear = 0.0

    # Shear angle from eigenvector
    _, eigvecs = np.linalg.eigh(cov_2d)
    angle = np.arctan2(eigvecs[1, 1], eigvecs[0, 1])

    return float(shear), float(angle)
