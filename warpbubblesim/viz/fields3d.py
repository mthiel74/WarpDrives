"""
3D visualization for warp bubble spacetimes using PyVista.

Provides isosurface rendering, volume visualization, and streamlines
for shift vectors and other 3D fields.
"""

import numpy as np
from typing import Callable, Optional, Tuple, List, Dict
from pathlib import Path

# Try to import pyvista
try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    pv = None


def _check_pyvista():
    if not PYVISTA_AVAILABLE:
        raise ImportError(
            "PyVista is required for 3D visualization. "
            "Install with: pip install pyvista"
        )


def create_grid_3d(
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    z_range: Tuple[float, float],
    nx: int,
    ny: int,
    nz: int
) -> "pv.StructuredGrid":
    """
    Create a PyVista structured grid.

    Parameters
    ----------
    x_range, y_range, z_range : tuple
        Coordinate ranges.
    nx, ny, nz : int
        Number of points in each direction.

    Returns
    -------
    pv.StructuredGrid
        PyVista grid object.
    """
    _check_pyvista()

    x = np.linspace(x_range[0], x_range[1], nx)
    y = np.linspace(y_range[0], y_range[1], ny)
    z = np.linspace(z_range[0], z_range[1], nz)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    grid = pv.StructuredGrid(X, Y, Z)
    return grid


def compute_field_on_grid(
    grid: "pv.StructuredGrid",
    field_func: Callable,
    t: float = 0.0,
    field_name: str = "field"
) -> "pv.StructuredGrid":
    """
    Compute a scalar field on a PyVista grid.

    Parameters
    ----------
    grid : pv.StructuredGrid
        The grid.
    field_func : callable
        Function (t, x, y, z) -> scalar.
    t : float
        Time coordinate.
    field_name : str
        Name for the field.

    Returns
    -------
    pv.StructuredGrid
        Grid with field values attached.
    """
    _check_pyvista()

    points = grid.points
    values = np.zeros(points.shape[0])

    for i, (x, y, z) in enumerate(points):
        values[i] = field_func(t, x, y, z)

    grid[field_name] = values
    return grid


def plot_isosurface(
    metric,
    field: str = "energy_density",
    x_range: Tuple[float, float] = (-5, 5),
    y_range: Tuple[float, float] = (-5, 5),
    z_range: Tuple[float, float] = (-5, 5),
    nx: int = 50,
    ny: int = 50,
    nz: int = 50,
    t: float = 0.0,
    isovalues: Optional[List[float]] = None,
    opacity: float = 0.7,
    show_edges: bool = False,
    cmap: str = "RdBu_r",
    save_path: Optional[str] = None,
    show: bool = True,
) -> Optional["pv.Plotter"]:
    """
    Plot isosurfaces of a field for a warp metric.

    Parameters
    ----------
    metric : WarpMetric
        The warp metric.
    field : str
        Field to visualize: 'energy_density', 'shape', 'kretschmann'.
    x_range, y_range, z_range : tuple
        Coordinate ranges.
    nx, ny, nz : int
        Grid resolution.
    t : float
        Time coordinate.
    isovalues : list, optional
        Isosurface values. If None, auto-determined.
    opacity : float
        Surface opacity.
    show_edges : bool
        Show mesh edges.
    cmap : str
        Colormap.
    save_path : str, optional
        Path to save screenshot.
    show : bool
        Whether to display interactive plot.

    Returns
    -------
    pv.Plotter or None
        The plotter object if show=False.
    """
    _check_pyvista()

    # Create grid
    grid = create_grid_3d(x_range, y_range, z_range, nx, ny, nz)

    # Select field function
    if field == "energy_density":
        from warpbubblesim.gr.energy import compute_energy_density
        metric_func = metric.get_metric_func()

        def field_func(t, x, y, z):
            coords = np.array([t, x, y, z])
            return compute_energy_density(metric_func, coords)
    elif field == "shape":
        def field_func(t, x, y, z):
            r = metric.r_from_center(t, x, y, z)
            return metric.shape_function(r)
    elif field == "kretschmann":
        from warpbubblesim.gr.invariants import compute_kretschmann
        metric_func = metric.get_metric_func()

        def field_func(t, x, y, z):
            coords = np.array([t, x, y, z])
            return compute_kretschmann(metric_func, coords)
    else:
        raise ValueError(f"Unknown field: {field}")

    # Compute field values
    grid = compute_field_on_grid(grid, field_func, t, field)

    # Determine isovalues if not provided
    if isovalues is None:
        values = grid[field]
        vmin, vmax = values.min(), values.max()
        if vmin * vmax < 0:  # Contains both positive and negative
            # Include both positive and negative isosurfaces
            isovalues = [vmin * 0.5, vmax * 0.5]
        else:
            isovalues = [0.5 * (vmin + vmax)]

    # Create plotter
    plotter = pv.Plotter()

    # Add isosurfaces
    for isovalue in isovalues:
        try:
            surface = grid.contour([isovalue], scalars=field)
            if surface.n_points > 0:
                plotter.add_mesh(
                    surface,
                    opacity=opacity,
                    show_edges=show_edges,
                    cmap=cmap,
                    scalars=field,
                )
        except Exception:
            pass  # Skip if contour fails

    # Add bounding box and axes
    plotter.add_axes()
    plotter.show_bounds(grid=True)

    # Mark bubble center
    x_s = metric.bubble_center(t)
    center = pv.PolyData([x_s, 0.0, 0.0])
    plotter.add_mesh(center, color='red', point_size=20, render_points_as_spheres=True)

    plotter.add_title(f"{metric.name} - {field} (t={t})")

    if save_path:
        plotter.screenshot(save_path)

    if show:
        plotter.show()
        return None
    else:
        return plotter


def plot_energy_density_3d(
    metric,
    x_range: Tuple[float, float] = (-5, 5),
    y_range: Tuple[float, float] = (-5, 5),
    z_range: Tuple[float, float] = (-5, 5),
    nx: int = 40,
    ny: int = 40,
    nz: int = 40,
    t: float = 0.0,
    opacity: str = "sigmoid",
    cmap: str = "RdBu_r",
    **kwargs
) -> Optional["pv.Plotter"]:
    """
    3D volume rendering of energy density.

    Parameters
    ----------
    metric : WarpMetric
        The warp metric.
    x_range, y_range, z_range : tuple
        Coordinate ranges.
    nx, ny, nz : int
        Grid resolution.
    t : float
        Time coordinate.
    opacity : str
        Opacity transfer function.
    cmap : str
        Colormap.
    **kwargs
        Additional arguments for plot_isosurface.

    Returns
    -------
    pv.Plotter or None
    """
    return plot_isosurface(
        metric, field="energy_density",
        x_range=x_range, y_range=y_range, z_range=z_range,
        nx=nx, ny=ny, nz=nz, t=t, cmap=cmap, **kwargs
    )


def plot_streamlines_3d(
    metric,
    x_range: Tuple[float, float] = (-5, 5),
    y_range: Tuple[float, float] = (-5, 5),
    z_range: Tuple[float, float] = (-5, 5),
    nx: int = 30,
    ny: int = 30,
    nz: int = 30,
    t: float = 0.0,
    n_streams: int = 50,
    source_radius: float = 1.0,
    tube_radius: float = 0.05,
    cmap: str = "viridis",
    save_path: Optional[str] = None,
    show: bool = True,
) -> Optional["pv.Plotter"]:
    """
    Plot streamlines of the shift vector field.

    Parameters
    ----------
    metric : WarpMetric
        The warp metric.
    x_range, y_range, z_range : tuple
        Coordinate ranges.
    nx, ny, nz : int
        Grid resolution.
    t : float
        Time coordinate.
    n_streams : int
        Number of streamlines.
    source_radius : float
        Radius of seed sphere.
    tube_radius : float
        Radius of streamline tubes.
    cmap : str
        Colormap.
    save_path : str, optional
        Path to save screenshot.
    show : bool
        Whether to display.

    Returns
    -------
    pv.Plotter or None
    """
    _check_pyvista()

    # Create grid
    grid = create_grid_3d(x_range, y_range, z_range, nx, ny, nz)

    # Compute shift vector field
    points = grid.points
    vectors = np.zeros_like(points)

    for i, (x, y, z) in enumerate(points):
        beta = metric.shift(t, x, y, z)
        vectors[i] = beta

    grid["shift"] = vectors

    # Create seed points
    x_s = metric.bubble_center(t)
    seed = pv.Sphere(radius=source_radius, center=(x_s, 0, 0), theta_resolution=10, phi_resolution=10)

    # Compute streamlines
    try:
        streams = grid.streamlines_from_source(
            seed,
            vectors="shift",
            max_time=10.0,
            n_points=n_streams,
            source_radius=source_radius,
        )
    except Exception:
        # Fallback if streamlines fail
        streams = None

    # Create plotter
    plotter = pv.Plotter()

    if streams is not None and streams.n_points > 0:
        tubes = streams.tube(radius=tube_radius)
        plotter.add_mesh(tubes, cmap=cmap, scalars="shift", component=0)

    # Add bubble center marker
    center = pv.Sphere(radius=0.2, center=(x_s, 0, 0))
    plotter.add_mesh(center, color='red')

    plotter.add_axes()
    plotter.add_title(f"Shift Vector Streamlines - {metric.name}")

    if save_path:
        plotter.screenshot(save_path)

    if show:
        plotter.show()
        return None
    else:
        return plotter


def plot_bubble_surface(
    metric,
    t: float = 0.0,
    f_threshold: float = 0.5,
    resolution: int = 50,
    color: str = "lightblue",
    opacity: float = 0.5,
    save_path: Optional[str] = None,
    show: bool = True,
) -> Optional["pv.Plotter"]:
    """
    Plot the warp bubble surface (where f = threshold).

    Parameters
    ----------
    metric : WarpMetric
        The warp metric.
    t : float
        Time coordinate.
    f_threshold : float
        Shape function threshold defining the bubble surface.
    resolution : int
        Angular resolution.
    color : str
        Surface color.
    opacity : float
        Surface opacity.
    save_path : str, optional
        Path to save screenshot.
    show : bool
        Whether to display.

    Returns
    -------
    pv.Plotter or None
    """
    _check_pyvista()

    # Find the radius where f = threshold
    # Binary search
    R_guess = metric.params.get('R', 1.0)
    r_low, r_high = 0.0, 3 * R_guess

    for _ in range(50):
        r_mid = (r_low + r_high) / 2
        f_mid = metric.shape_function(r_mid)
        if f_mid > f_threshold:
            r_low = r_mid
        else:
            r_high = r_mid

    R_surface = r_mid

    # Create sphere at bubble radius
    x_s = metric.bubble_center(t)
    sphere = pv.Sphere(radius=R_surface, center=(x_s, 0, 0),
                        theta_resolution=resolution, phi_resolution=resolution)

    # Create plotter
    plotter = pv.Plotter()
    plotter.add_mesh(sphere, color=color, opacity=opacity, smooth_shading=True)
    plotter.add_axes()
    plotter.add_title(f"Bubble Surface (f={f_threshold}) - {metric.name}")

    if save_path:
        plotter.screenshot(save_path)

    if show:
        plotter.show()
        return None
    else:
        return plotter


def create_warp_visualization(
    metric,
    t: float = 0.0,
    show_bubble: bool = True,
    show_energy: bool = True,
    show_streamlines: bool = False,
    resolution: int = 30,
    save_path: Optional[str] = None,
    show: bool = True,
) -> Optional["pv.Plotter"]:
    """
    Create a comprehensive 3D visualization of a warp metric.

    Parameters
    ----------
    metric : WarpMetric
        The warp metric.
    t : float
        Time coordinate.
    show_bubble : bool
        Show bubble surface.
    show_energy : bool
        Show energy density isosurfaces.
    show_streamlines : bool
        Show shift vector streamlines.
    resolution : int
        Grid resolution.
    save_path : str, optional
        Path to save screenshot.
    show : bool
        Whether to display.

    Returns
    -------
    pv.Plotter or None
    """
    _check_pyvista()

    plotter = pv.Plotter()
    R = metric.params.get('R', 1.0)
    extent = 3 * R

    x_s = metric.bubble_center(t)

    if show_bubble:
        # Add bubble surface
        sphere = pv.Sphere(radius=R, center=(x_s, 0, 0),
                           theta_resolution=30, phi_resolution=30)
        plotter.add_mesh(sphere, color='lightblue', opacity=0.3, smooth_shading=True)

    if show_energy:
        # Add energy density isosurfaces
        grid = create_grid_3d(
            (-extent, extent), (-extent, extent), (-extent, extent),
            resolution, resolution, resolution
        )

        from warpbubblesim.gr.energy import compute_energy_density
        metric_func = metric.get_metric_func()

        def rho_func(t, x, y, z):
            coords = np.array([t, x, y, z])
            try:
                return compute_energy_density(metric_func, coords, h=0.01)
            except Exception:
                return 0.0

        grid = compute_field_on_grid(grid, rho_func, t, "rho")

        values = grid["rho"]
        rho_min, rho_max = values.min(), values.max()

        # Negative energy isosurface (if present)
        if rho_min < -1e-10:
            try:
                neg_surface = grid.contour([rho_min * 0.5], scalars="rho")
                if neg_surface.n_points > 0:
                    plotter.add_mesh(neg_surface, color='red', opacity=0.5,
                                     label='Negative Energy')
            except Exception:
                pass

    # Add bubble center marker
    center = pv.Sphere(radius=0.1, center=(x_s, 0, 0))
    plotter.add_mesh(center, color='yellow')

    # Add arrow showing direction of motion
    arrow = pv.Arrow(start=(x_s - R, 0, 0), direction=(1, 0, 0), scale=R)
    plotter.add_mesh(arrow, color='green')

    plotter.add_axes()
    plotter.add_title(f"{metric.name} (t={t})")

    if save_path:
        plotter.screenshot(save_path)

    if show:
        plotter.show()
        return None
    else:
        return plotter
