"""
Grid generation utilities for WarpBubbleSim.

Provides functions for creating coordinate grids in 2D, 3D, and 4D
for field evaluation and visualization.
"""

import numpy as np
from typing import Tuple, Optional


def create_grid_2d(
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    nx: int,
    ny: int,
    indexing: str = "xy"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a 2D coordinate grid.

    Parameters
    ----------
    x_range : tuple
        (x_min, x_max) range for x coordinate.
    y_range : tuple
        (y_min, y_max) range for y coordinate.
    nx : int
        Number of points in x direction.
    ny : int
        Number of points in y direction.
    indexing : str
        'xy' for Cartesian indexing, 'ij' for matrix indexing.

    Returns
    -------
    tuple
        (X, Y) meshgrid arrays.
    """
    x = np.linspace(x_range[0], x_range[1], nx)
    y = np.linspace(y_range[0], y_range[1], ny)
    return np.meshgrid(x, y, indexing=indexing)


def create_grid_3d(
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    z_range: Tuple[float, float],
    nx: int,
    ny: int,
    nz: int,
    indexing: str = "ij"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a 3D coordinate grid.

    Parameters
    ----------
    x_range, y_range, z_range : tuple
        (min, max) ranges for each coordinate.
    nx, ny, nz : int
        Number of points in each direction.
    indexing : str
        'xy' for Cartesian indexing, 'ij' for matrix indexing.

    Returns
    -------
    tuple
        (X, Y, Z) meshgrid arrays.
    """
    x = np.linspace(x_range[0], x_range[1], nx)
    y = np.linspace(y_range[0], y_range[1], ny)
    z = np.linspace(z_range[0], z_range[1], nz)
    return np.meshgrid(x, y, z, indexing=indexing)


def create_grid_4d(
    t_range: Tuple[float, float],
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    z_range: Tuple[float, float],
    nt: int,
    nx: int,
    ny: int,
    nz: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a 4D spacetime coordinate grid.

    Uses 'ij' indexing so that output arrays have shape (nt, nx, ny, nz).

    Parameters
    ----------
    t_range, x_range, y_range, z_range : tuple
        (min, max) ranges for each coordinate.
    nt, nx, ny, nz : int
        Number of points in each direction.

    Returns
    -------
    tuple
        (T, X, Y, Z) meshgrid arrays, each with shape (nt, nx, ny, nz).
    """
    t = np.linspace(t_range[0], t_range[1], nt)
    x = np.linspace(x_range[0], x_range[1], nx)
    y = np.linspace(y_range[0], y_range[1], ny)
    z = np.linspace(z_range[0], z_range[1], nz)
    return np.meshgrid(t, x, y, z, indexing='ij')


def meshgrid_to_points(
    *grids: np.ndarray
) -> np.ndarray:
    """
    Convert meshgrid arrays to a flat array of coordinate points.

    Parameters
    ----------
    *grids : np.ndarray
        Meshgrid arrays from np.meshgrid.

    Returns
    -------
    np.ndarray
        Array of shape (n_points, n_dims) containing all grid points.
    """
    return np.stack([g.ravel() for g in grids], axis=-1)


def create_slice_grid(
    slice_coord: str,
    slice_value: float,
    range1: Tuple[float, float],
    range2: Tuple[float, float],
    n1: int,
    n2: int,
    t: float = 0.0
) -> Tuple[np.ndarray, ...]:
    """
    Create a 2D grid representing a slice through 3D or 4D space.

    Parameters
    ----------
    slice_coord : str
        Which coordinate to fix: 'x', 'y', or 'z'.
    slice_value : float
        Value at which to fix the slice coordinate.
    range1, range2 : tuple
        Ranges for the two varying coordinates (in order x, y, z minus slice_coord).
    n1, n2 : int
        Number of points for the two varying coordinates.
    t : float
        Time coordinate value.

    Returns
    -------
    tuple
        (T, X, Y, Z) arrays where the sliced coordinate is constant.
    """
    c1 = np.linspace(range1[0], range1[1], n1)
    c2 = np.linspace(range2[0], range2[1], n2)
    C1, C2 = np.meshgrid(c1, c2, indexing='xy')

    T = np.full_like(C1, t)
    SLICE = np.full_like(C1, slice_value)

    if slice_coord == 'x':
        return T, SLICE, C1, C2
    elif slice_coord == 'y':
        return T, C1, SLICE, C2
    elif slice_coord == 'z':
        return T, C1, C2, SLICE
    else:
        raise ValueError(f"slice_coord must be 'x', 'y', or 'z', got {slice_coord}")


def create_spacetime_slice(
    t_range: Tuple[float, float],
    x_range: Tuple[float, float],
    nt: int,
    nx: int,
    y: float = 0.0,
    z: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a 2D slice in the (t, x) plane for spacetime diagrams.

    Parameters
    ----------
    t_range : tuple
        (t_min, t_max) time range.
    x_range : tuple
        (x_min, x_max) spatial range.
    nt, nx : int
        Number of points in each direction.
    y, z : float
        Fixed values for y and z coordinates.

    Returns
    -------
    tuple
        (T, X, Y, Z) arrays for the slice.
    """
    t = np.linspace(t_range[0], t_range[1], nt)
    x = np.linspace(x_range[0], x_range[1], nx)
    T, X = np.meshgrid(t, x, indexing='ij')
    Y = np.full_like(T, y)
    Z = np.full_like(T, z)
    return T, X, Y, Z


def cylindrical_to_cartesian(
    rho: np.ndarray,
    phi: np.ndarray,
    z: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert cylindrical coordinates to Cartesian.

    Parameters
    ----------
    rho : np.ndarray
        Radial coordinate.
    phi : np.ndarray
        Azimuthal angle.
    z : np.ndarray
        Axial coordinate.

    Returns
    -------
    tuple
        (x, y, z) Cartesian coordinates.
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y, z


def spherical_to_cartesian(
    r: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert spherical coordinates to Cartesian.

    Parameters
    ----------
    r : np.ndarray
        Radial coordinate.
    theta : np.ndarray
        Polar angle (from z-axis).
    phi : np.ndarray
        Azimuthal angle.

    Returns
    -------
    tuple
        (x, y, z) Cartesian coordinates.
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def adaptive_grid_1d(
    x_range: Tuple[float, float],
    n_base: int,
    refine_func: callable,
    max_refine: int = 3
) -> np.ndarray:
    """
    Create an adaptively refined 1D grid.

    Refines regions where refine_func returns high values.

    Parameters
    ----------
    x_range : tuple
        (x_min, x_max) range.
    n_base : int
        Base number of points.
    refine_func : callable
        Function f(x) -> float indicating where to refine (high = refine more).
    max_refine : int
        Maximum refinement levels.

    Returns
    -------
    np.ndarray
        Adaptively refined 1D grid.
    """
    x = np.linspace(x_range[0], x_range[1], n_base)

    for _ in range(max_refine):
        # Evaluate refinement criterion
        vals = refine_func(x)
        threshold = np.percentile(vals, 75)

        # Find regions to refine
        new_points = []
        for i in range(len(x) - 1):
            if vals[i] > threshold or vals[i+1] > threshold:
                # Add midpoint
                new_points.append(0.5 * (x[i] + x[i+1]))

        if not new_points:
            break

        x = np.sort(np.concatenate([x, new_points]))

    return x
