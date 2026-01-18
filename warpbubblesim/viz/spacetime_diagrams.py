"""
Spacetime diagram visualization for warp bubble geodesics.

Provides tools for plotting worldlines, light cones, and geodesic bundles
in (t, x) spacetime diagrams.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.patches import Polygon
from matplotlib.collections import LineCollection
from typing import Callable, Optional, Tuple, List, Dict
from pathlib import Path


def plot_spacetime_diagram(
    metric,
    t_range: Tuple[float, float] = (0, 10),
    x_range: Tuple[float, float] = (-10, 10),
    show_bubble_center: bool = True,
    show_null_cones: bool = True,
    n_cones: int = 5,
    figsize: Tuple[float, float] = (10, 8),
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """
    Create a spacetime diagram for a warp metric.

    Parameters
    ----------
    metric : WarpMetric
        The warp metric.
    t_range : tuple
        (t_min, t_max) time range.
    x_range : tuple
        (x_min, x_max) spatial range.
    show_bubble_center : bool
        Show the bubble center worldline.
    show_null_cones : bool
        Show local light cones.
    n_cones : int
        Number of light cones to draw.
    figsize : tuple
        Figure size.
    ax : Axes, optional
        Existing axes.

    Returns
    -------
    tuple
        (fig, ax)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Draw bubble center worldline
    if show_bubble_center:
        t_values = np.linspace(t_range[0], t_range[1], 100)
        x_center = np.array([metric.bubble_center(t) for t in t_values])

        # Only plot visible portion
        mask = (x_center >= x_range[0]) & (x_center <= x_range[1])
        ax.plot(x_center[mask], t_values[mask], 'r-', linewidth=2,
                label='Bubble center')

    # Draw local light cones at selected points
    if show_null_cones:
        t_cone = np.linspace(t_range[0], t_range[1], n_cones + 2)[1:-1]
        x_cone = np.linspace(x_range[0], x_range[1], n_cones + 2)[1:-1]

        cone_size = min(t_range[1] - t_range[0], x_range[1] - x_range[0]) / (2 * n_cones)

        for t0 in t_cone:
            x0 = metric.bubble_center(t0)
            if x_range[0] <= x0 <= x_range[1]:
                # Local light cone at bubble center
                _draw_light_cone(ax, t0, x0, cone_size, metric, color='blue', alpha=0.3)

    # Set labels and limits
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_xlim(x_range)
    ax.set_ylim(t_range)
    ax.set_title(f'Spacetime Diagram - {metric.name}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax


def _draw_light_cone(
    ax: Axes,
    t0: float,
    x0: float,
    size: float,
    metric,
    color: str = 'blue',
    alpha: float = 0.3
):
    """
    Draw a local light cone at (t0, x0).

    The light cone shape is determined by the metric's shift vector.
    """
    # Get shift at this point
    beta = metric.shift(t0, x0, 0, 0)
    beta_x = beta[0]

    # For Alcubierre-type with α=1:
    # Null vectors satisfy: dt² = (dx + β dt)²
    # So dx/dt = 1 - β (right-going) or dx/dt = -1 - β (left-going)

    # Future light cone edges
    dt = size
    dx_right = (1 - beta_x) * dt  # Right-going null
    dx_left = (-1 - beta_x) * dt  # Left-going null

    # Draw cone as filled polygon
    vertices = [
        (x0, t0),
        (x0 + dx_right, t0 + dt),
        (x0 + dx_left, t0 + dt),
    ]
    cone = Polygon(vertices, alpha=alpha, facecolor=color, edgecolor=color)
    ax.add_patch(cone)

    # Also draw past cone
    vertices_past = [
        (x0, t0),
        (x0 - dx_right, t0 - dt),
        (x0 - dx_left, t0 - dt),
    ]
    cone_past = Polygon(vertices_past, alpha=alpha/2, facecolor=color, edgecolor=color)
    ax.add_patch(cone_past)


def plot_geodesics(
    geodesic_results: List[Dict],
    t_range: Optional[Tuple[float, float]] = None,
    x_range: Optional[Tuple[float, float]] = None,
    colors: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (10, 8),
    ax: Optional[Axes] = None,
    show_initial_points: bool = True,
    linewidth: float = 1.5,
) -> Tuple[Figure, Axes]:
    """
    Plot geodesics on a spacetime diagram.

    Parameters
    ----------
    geodesic_results : list
        List of geodesic result dictionaries from integrate_geodesic.
    t_range : tuple, optional
        Time axis limits.
    x_range : tuple, optional
        Space axis limits.
    colors : list, optional
        Colors for each geodesic.
    labels : list, optional
        Labels for legend.
    figsize : tuple
        Figure size.
    ax : Axes, optional
        Existing axes.
    show_initial_points : bool
        Mark initial points.
    linewidth : float
        Line width.

    Returns
    -------
    tuple
        (fig, ax)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    n_geodesics = len(geodesic_results)

    if colors is None:
        cmap = plt.cm.viridis
        colors = [cmap(i / max(1, n_geodesics - 1)) for i in range(n_geodesics)]

    if labels is None:
        labels = [None] * n_geodesics

    for i, result in enumerate(geodesic_results):
        coords = result['coords']
        t = coords[:, 0]
        x = coords[:, 1]

        # Determine linestyle based on geodesic type
        if result.get('is_timelike', True):
            linestyle = '-'
        else:
            linestyle = '--'

        ax.plot(x, t, color=colors[i], linewidth=linewidth,
                linestyle=linestyle, label=labels[i])

        if show_initial_points:
            ax.plot(x[0], t[0], 'o', color=colors[i], markersize=8)

    # Set limits
    if t_range:
        ax.set_ylim(t_range)
    if x_range:
        ax.set_xlim(x_range)

    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_title('Geodesics in Spacetime')
    ax.grid(True, alpha=0.3)

    if any(labels):
        ax.legend()

    return fig, ax


def plot_light_cones(
    metric,
    points: List[Tuple[float, float]],
    cone_size: float = 1.0,
    t_range: Optional[Tuple[float, float]] = None,
    x_range: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (10, 8),
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """
    Plot light cones at specified spacetime points.

    Parameters
    ----------
    metric : WarpMetric
        The warp metric.
    points : list
        List of (t, x) points where to draw cones.
    cone_size : float
        Size of light cones.
    t_range, x_range : tuple, optional
        Axis limits.
    figsize : tuple
        Figure size.
    ax : Axes, optional
        Existing axes.

    Returns
    -------
    tuple
        (fig, ax)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    cmap = plt.cm.coolwarm
    n_points = len(points)

    for i, (t0, x0) in enumerate(points):
        color = cmap(i / max(1, n_points - 1))
        _draw_light_cone(ax, t0, x0, cone_size, metric, color=color, alpha=0.4)

        # Mark the point
        ax.plot(x0, t0, 'ko', markersize=4)

    if t_range:
        ax.set_ylim(t_range)
    if x_range:
        ax.set_xlim(x_range)

    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_title(f'Light Cones - {metric.name}')
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_worldlines(
    metric,
    initial_positions: List[Tuple[float, float, float]],
    t_range: Tuple[float, float] = (0, 10),
    x_range: Optional[Tuple[float, float]] = None,
    velocity_type: str = "static",
    n_steps: int = 200,
    figsize: Tuple[float, float] = (10, 8),
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """
    Plot worldlines of test particles.

    Parameters
    ----------
    metric : WarpMetric
        The warp metric.
    initial_positions : list
        List of initial (x, y, z) positions.
    t_range : tuple
        Time range.
    x_range : tuple, optional
        x-axis limits.
    velocity_type : str
        'static' for coordinate-stationary, 'comoving' for moving with bubble.
    n_steps : int
        Number of integration steps.
    figsize : tuple
        Figure size.
    ax : Axes, optional
        Existing axes.

    Returns
    -------
    tuple
        (fig, ax)
    """
    from warpbubblesim.gr.geodesics import integrate_geodesic

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    metric_func = metric.get_metric_func()

    cmap = plt.cm.viridis
    n_particles = len(initial_positions)

    for i, (x0, y0, z0) in enumerate(initial_positions):
        color = cmap(i / max(1, n_particles - 1))

        # Initial coordinates
        initial_coords = np.array([t_range[0], x0, y0, z0])

        # Initial velocity (static in coordinates)
        if velocity_type == "static":
            u0 = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            # Comoving with bubble
            beta = metric.shift(t_range[0], x0, y0, z0)
            u0 = np.array([1.0, -beta[0], -beta[1], -beta[2]])

        # Normalize
        g = metric_func(*initial_coords)
        norm_sq = np.einsum('mn,m,n->', g, u0, u0)
        if norm_sq < 0:
            u0 = u0 / np.sqrt(-norm_sq)

        # Integrate
        try:
            result = integrate_geodesic(
                metric_func, initial_coords, u0,
                t_range, max_step=0.1, rtol=1e-6
            )

            coords = result['coords']
            ax.plot(coords[:, 1], coords[:, 0], color=color, linewidth=1.5)
            ax.plot(x0, t_range[0], 'o', color=color, markersize=6)

        except Exception as e:
            # Mark failed integration
            ax.plot(x0, t_range[0], 'x', color='red', markersize=10)

    # Add bubble center worldline
    t_values = np.linspace(t_range[0], t_range[1], 100)
    x_center = np.array([metric.bubble_center(t) for t in t_values])
    ax.plot(x_center, t_values, 'r--', linewidth=2, label='Bubble center')

    if x_range:
        ax.set_xlim(x_range)
    ax.set_ylim(t_range)

    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_title(f'Worldlines - {metric.name}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_null_geodesic_bundle(
    metric,
    origin: Tuple[float, float],
    n_rays: int = 20,
    lambda_max: float = 5.0,
    forward: bool = True,
    backward: bool = True,
    figsize: Tuple[float, float] = (10, 8),
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """
    Plot a bundle of null geodesics from a point.

    Parameters
    ----------
    metric : WarpMetric
        The warp metric.
    origin : tuple
        (t0, x0) origin point in spacetime.
    n_rays : int
        Number of rays.
    lambda_max : float
        Maximum affine parameter.
    forward : bool
        Include forward-directed rays.
    backward : bool
        Include backward-directed rays.
    figsize : tuple
        Figure size.
    ax : Axes, optional
        Existing axes.

    Returns
    -------
    tuple
        (fig, ax)
    """
    from warpbubblesim.gr.geodesics import integrate_null_geodesic

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    t0, x0 = origin
    metric_func = metric.get_metric_func()

    # Directions: mix of forward and backward in x
    angles = np.linspace(-np.pi/3, np.pi/3, n_rays)

    for angle in angles:
        direction = np.array([np.cos(angle), np.sin(angle), 0.0])

        if forward:
            try:
                result = integrate_null_geodesic(
                    metric_func,
                    np.array([t0, x0, 0.0, 0.0]),
                    direction,
                    (0, lambda_max),
                    backward=False,
                    max_step=0.1
                )
                coords = result['coords']
                ax.plot(coords[:, 1], coords[:, 0], 'b-', alpha=0.5, linewidth=1)
            except Exception:
                pass

        if backward:
            try:
                result = integrate_null_geodesic(
                    metric_func,
                    np.array([t0, x0, 0.0, 0.0]),
                    direction,
                    (0, lambda_max),
                    backward=True,
                    max_step=0.1
                )
                coords = result['coords']
                ax.plot(coords[:, 1], coords[:, 0], 'r-', alpha=0.5, linewidth=1)
            except Exception:
                pass

    # Mark origin
    ax.plot(x0, t0, 'ko', markersize=8)

    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_title(f'Null Geodesic Bundle - {metric.name}')
    ax.grid(True, alpha=0.3)

    return fig, ax
