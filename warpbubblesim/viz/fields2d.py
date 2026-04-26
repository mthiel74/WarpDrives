"""
2D field visualization for warp bubble spacetimes.

Provides functions for creating heatmaps, contour plots, and
vector field visualizations of metric components, energy densities,
and curvature quantities.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.colors import Normalize, TwoSlopeNorm
from typing import Callable, Optional, Tuple, List, Union
from pathlib import Path


def plot_field_2d(
    field_func: Callable,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    nx: int = 128,
    ny: int = 128,
    t: float = 0.0,
    z: float = 0.0,
    title: str = "Field",
    cmap: str = "RdBu_r",
    symmetric_colorbar: bool = True,
    add_contours: bool = True,
    n_contours: int = 10,
    figsize: Tuple[float, float] = (8, 6),
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """
    Plot a 2D slice of a scalar field.

    Parameters
    ----------
    field_func : callable
        Function (t, x, y, z) -> scalar value.
    x_range, y_range : tuple
        (min, max) ranges for coordinates.
    nx, ny : int
        Number of grid points.
    t : float
        Fixed time coordinate.
    z : float
        Fixed z coordinate (for x-y slice).
    title : str
        Plot title.
    cmap : str
        Colormap name.
    symmetric_colorbar : bool
        If True, center colorbar on zero.
    add_contours : bool
        If True, overlay contour lines.
    n_contours : int
        Number of contour levels.
    figsize : tuple
        Figure size.
    ax : Axes, optional
        Existing axes to plot on.

    Returns
    -------
    tuple
        (fig, ax) matplotlib figure and axes.
    """
    # Create grid
    x = np.linspace(x_range[0], x_range[1], nx)
    y = np.linspace(y_range[0], y_range[1], ny)
    X, Y = np.meshgrid(x, y, indexing='xy')

    # Evaluate field
    Z = np.zeros_like(X)
    for i in range(ny):
        for j in range(nx):
            Z[i, j] = field_func(t, X[i, j], Y[i, j], z)

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Determine normalization
    if symmetric_colorbar:
        vmax = max(abs(Z.max()), abs(Z.min()))
        vmin = -vmax
        if vmax < 1e-10:
            vmin, vmax = -1, 1
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    else:
        norm = None

    # Plot heatmap
    im = ax.pcolormesh(X, Y, Z, cmap=cmap, norm=norm, shading='auto')
    plt.colorbar(im, ax=ax, label=title)

    # Add contours
    if add_contours:
        levels = np.linspace(Z.min(), Z.max(), n_contours)
        levels = levels[levels != 0]  # Remove zero level if present
        if len(levels) > 0:
            ax.contour(X, Y, Z, levels=levels, colors='k', alpha=0.3, linewidths=0.5)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.set_aspect('equal')

    return fig, ax


def plot_energy_density(
    metric,
    x_range: Tuple[float, float] = (-5, 5),
    y_range: Tuple[float, float] = (-5, 5),
    nx: int = 128,
    ny: int = 128,
    t: float = 0.0,
    z: float = 0.0,
    observer: str = "eulerian",
    backend: str = "finite_difference",
    h: float = 1e-5,
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Plot energy density for a warp metric.

    Parameters
    ----------
    metric : WarpMetric
        The warp drive metric.
    x_range, y_range : tuple
        Coordinate ranges.
    nx, ny : int
        Grid resolution.
    t : float
        Time coordinate.
    z : float
        Z-coordinate for slice.
    observer : str
        Observer type: 'eulerian' or 'static'.
    backend : str
        Derivative backend.
    h : float
        Step size for derivatives.
    **kwargs
        Additional arguments for plot_field_2d.

    Returns
    -------
    tuple
        (fig, ax)
    """
    from warpbubblesim.gr.energy import compute_energy_density

    metric_func = metric.get_metric_func()

    def rho_func(t, x, y, z):
        coords = np.array([t, x, y, z])
        return compute_energy_density(metric_func, coords, backend=backend, h=h)

    return plot_field_2d(
        rho_func, x_range, y_range, nx, ny, t, z,
        title=r"Energy Density $\rho$",
        cmap="RdBu_r",
        symmetric_colorbar=True,
        **kwargs
    )


def plot_expansion_scalar(
    metric,
    x_range: Tuple[float, float] = (-5, 5),
    y_range: Tuple[float, float] = (-5, 5),
    nx: int = 128,
    ny: int = 128,
    t: float = 0.0,
    z: float = 0.0,
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Plot expansion scalar θ for a warp metric.

    Uses the analytic formula if available, otherwise numerical.
    """
    # Check for analytic formula
    if hasattr(metric, 'expansion_scalar_analytic'):
        def theta_func(t, x, y, z):
            return metric.expansion_scalar_analytic(t, x, y, z)
    else:
        # Numerical fallback.  For an ADM metric with flat spatial slice,
        # the Eulerian-observer expansion is theta = -D_i beta^i (the
        # minus sign comes from n^i = -beta^i / alpha; see Wald sec 10).
        # ShiftDivergence returns +D_i beta^i, so flip the sign here so
        # the fallback agrees with the analytic formulas in sign.
        from warpbubblesim.gr.adm import compute_shift_divergence
        shift_func = metric.get_shift_func()

        def theta_func(t, x, y, z):
            coords = np.array([t, x, y, z])
            return -compute_shift_divergence(shift_func, coords)

    return plot_field_2d(
        theta_func, x_range, y_range, nx, ny, t, z,
        title=r"Expansion Scalar $\theta$",
        cmap="RdBu_r",
        symmetric_colorbar=True,
        **kwargs
    )


def plot_metric_component(
    metric,
    component: Tuple[int, int] = (0, 0),
    x_range: Tuple[float, float] = (-5, 5),
    y_range: Tuple[float, float] = (-5, 5),
    nx: int = 128,
    ny: int = 128,
    t: float = 0.0,
    z: float = 0.0,
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Plot a specific metric component g_{μν}.

    Parameters
    ----------
    metric : WarpMetric
        The metric.
    component : tuple
        (μ, ν) indices.
    **kwargs
        Additional plotting arguments.

    Returns
    -------
    tuple
        (fig, ax)
    """
    mu, nu = component
    idx_names = ['t', 'x', 'y', 'z']

    def g_component(t, x, y, z):
        g = metric.metric(t, x, y, z)
        return g[mu, nu]

    return plot_field_2d(
        g_component, x_range, y_range, nx, ny, t, z,
        title=f"$g_{{{idx_names[mu]}{idx_names[nu]}}}$",
        **kwargs
    )


def plot_shape_function(
    metric,
    r_max: float = 5.0,
    n_points: int = 500,
    figsize: Tuple[float, float] = (8, 4),
    ax: Optional[Axes] = None
) -> Tuple[Figure, Axes]:
    """
    Plot the shape function f(r) for a warp metric.

    Parameters
    ----------
    metric : WarpMetric
        The warp metric with shape_function method.
    r_max : float
        Maximum radius to plot.
    n_points : int
        Number of points.
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

    r = np.linspace(0, r_max, n_points)
    f = np.array([metric.shape_function(ri) for ri in r])

    ax.plot(r, f, 'b-', linewidth=2, label='f(r)')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)

    # Mark characteristic radius if available
    if hasattr(metric, 'R'):
        R = metric.R
        ax.axvline(R, color='r', linestyle='--', alpha=0.5, label=f'R = {R}')

    ax.set_xlabel('r')
    ax.set_ylabel('f(r)')
    ax.set_title(f'Shape Function - {metric.name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, r_max)
    ax.set_ylim(-0.1, 1.1)

    return fig, ax


def plot_multiple_fields(
    metric,
    fields: List[str] = ['energy_density', 'expansion', 'shape'],
    x_range: Tuple[float, float] = (-5, 5),
    y_range: Tuple[float, float] = (-5, 5),
    nx: int = 64,
    ny: int = 64,
    t: float = 0.0,
    figsize: Tuple[float, float] = (15, 4),
    save_path: Optional[str] = None,
) -> Figure:
    """
    Create a multi-panel plot of various fields.

    Parameters
    ----------
    metric : WarpMetric
        The warp metric.
    fields : list
        List of field names: 'energy_density', 'expansion', 'shape',
        'g00', 'g01', 'kretschmann'.
    x_range, y_range : tuple
        Coordinate ranges.
    nx, ny : int
        Grid resolution.
    t : float
        Time coordinate.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save figure.

    Returns
    -------
    Figure
        The matplotlib figure.
    """
    n_fields = len(fields)
    fig, axes = plt.subplots(1, n_fields, figsize=figsize)
    if n_fields == 1:
        axes = [axes]

    for i, field in enumerate(fields):
        ax = axes[i]

        if field == 'energy_density':
            plot_energy_density(metric, x_range, y_range, nx, ny, t, ax=ax)
        elif field == 'expansion':
            plot_expansion_scalar(metric, x_range, y_range, nx, ny, t, ax=ax)
        elif field == 'shape':
            plot_shape_function(metric, ax=ax)
        elif field == 'g00':
            plot_metric_component(metric, (0, 0), x_range, y_range, nx, ny, t, ax=ax)
        elif field == 'g01':
            plot_metric_component(metric, (0, 1), x_range, y_range, nx, ny, t, ax=ax)
        else:
            ax.text(0.5, 0.5, f"Unknown: {field}", ha='center', va='center',
                    transform=ax.transAxes)

    plt.suptitle(f'{metric.name} (t = {t})', y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_grid_distortion(
    metric,
    x_range: Tuple[float, float] = (-5, 5),
    y_range: Tuple[float, float] = (-5, 5),
    n_lines: int = 20,
    n_steps: int = 40,
    t: float = 0.0,
    z: float = 0.0,
    figsize: Tuple[float, float] = (8, 8),
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """
    Plot a regular Cartesian grid cumulatively advected by the
    metric's shift field from t=0 to time `t`.

    Each grid intersection is treated as a comoving test particle
    integrated forward by Euler steps under
        dx^i / dt = beta^i
    (the comoving-coordinate advection law).  After the bubble passes
    a grid node it does not return -- the shift field provides no
    restoring force -- so the grid stays deformed behind the bubble.
    For t=0 the grid is undistorted.

    Mirrors the visual in Section 6 of the Wolfram Community post.

    Parameters
    ----------
    metric : WarpMetric
        The warp metric.  Must expose ``shift(t, x, y, z)`` returning
        the 3-vector beta^i.
    x_range, y_range : tuple
        Coordinate ranges.
    n_lines : int
        Number of grid lines per direction.
    n_steps : int
        Number of Euler steps used to advect each grid node from
        t=0 to t.
    t : float
        Final time at which to draw the grid.
    z : float
        Fixed z slice.
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

    def advect(x0: float, y0: float) -> Tuple[float, float]:
        """Forward-Euler advect (x0, y0) from time 0 to time t."""
        if t == 0.0 or n_steps <= 0:
            return x0, y0
        dt = t / n_steps
        xc, yc = x0, y0
        for k in range(n_steps):
            tk = k * dt
            beta = metric.shift(tk, xc, yc, z)
            xc = xc + dt * float(beta[0])
            yc = yc + dt * float(beta[1])
        return xc, yc

    # Sample resolution along each line: enough points that the
    # advected curve looks smooth.
    n_samples = max(n_lines, 60)
    sample_xs = np.linspace(x_range[0], x_range[1], n_samples)
    sample_ys = np.linspace(y_range[0], y_range[1], n_samples)
    line_xs = np.linspace(x_range[0], x_range[1], n_lines)
    line_ys = np.linspace(y_range[0], y_range[1], n_lines)

    # Lines of constant initial y, swept across x
    for y_init in line_ys:
        pts = [advect(x_init, y_init) for x_init in sample_xs]
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                'b-', alpha=0.4, linewidth=0.6)

    # Lines of constant initial x, swept across y
    for x_init in line_xs:
        pts = [advect(x_init, y_init) for y_init in sample_ys]
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                'b-', alpha=0.4, linewidth=0.6)

    # Mark bubble center at the final time t
    x_s = metric.bubble_center(t)
    ax.plot(x_s, 0, 'r*', markersize=15, label='Bubble center')

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Coordinate grid advected to t = {t} -- {metric.name}')
    ax.set_aspect('equal')
    ax.legend()

    return fig, ax


def plot_shift_vector_field(
    metric,
    x_range: Tuple[float, float] = (-5, 5),
    y_range: Tuple[float, float] = (-5, 5),
    nx: int = 20,
    ny: int = 20,
    t: float = 0.0,
    z: float = 0.0,
    figsize: Tuple[float, float] = (8, 6),
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """
    Plot the shift vector field β^i as arrows.

    Parameters
    ----------
    metric : WarpMetric
        The warp metric.
    x_range, y_range : tuple
        Coordinate ranges.
    nx, ny : int
        Number of arrow points per direction.
    t, z : float
        Fixed coordinates.
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

    x = np.linspace(x_range[0], x_range[1], nx)
    y = np.linspace(y_range[0], y_range[1], ny)
    X, Y = np.meshgrid(x, y)

    # Compute shift components
    U = np.zeros_like(X)
    V = np.zeros_like(X)

    for i in range(ny):
        for j in range(nx):
            beta = metric.shift(t, X[i, j], Y[i, j], z)
            U[i, j] = beta[0]
            V[i, j] = beta[1]

    # Magnitude for coloring
    M = np.sqrt(U**2 + V**2)

    # Plot quiver
    q = ax.quiver(X, Y, U, V, M, cmap='viridis', alpha=0.8)
    plt.colorbar(q, ax=ax, label=r'$|\beta|$')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Shift Vector Field - {metric.name}')
    ax.set_aspect('equal')

    return fig, ax
