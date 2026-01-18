"""
Animation utilities for WarpBubbleSim.

Provides functions for creating animated visualizations of:
- Time evolution of fields
- Geodesic motion
- Grid distortion
- Light cone propagation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.figure import Figure
from matplotlib.colors import TwoSlopeNorm
from typing import Callable, Optional, Tuple, List, Dict
from pathlib import Path
from tqdm import tqdm


def animate_field_evolution(
    metric,
    field: str = "energy_density",
    x_range: Tuple[float, float] = (-5, 5),
    y_range: Tuple[float, float] = (-5, 5),
    t_range: Tuple[float, float] = (0, 10),
    nx: int = 64,
    ny: int = 64,
    n_frames: int = 100,
    fps: int = 30,
    z: float = 0.0,
    cmap: str = "RdBu_r",
    figsize: Tuple[float, float] = (8, 6),
    save_path: Optional[str] = None,
    show_progress: bool = True,
) -> FuncAnimation:
    """
    Create an animation of field evolution in time.

    Parameters
    ----------
    metric : WarpMetric
        The warp metric.
    field : str
        Field to animate: 'energy_density', 'shape', 'expansion'.
    x_range, y_range : tuple
        Coordinate ranges.
    t_range : tuple
        Time range.
    nx, ny : int
        Grid resolution.
    n_frames : int
        Number of animation frames.
    fps : int
        Frames per second.
    z : float
        Z coordinate for slice.
    cmap : str
        Colormap.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save animation.
    show_progress : bool
        Show progress bar.

    Returns
    -------
    FuncAnimation
        The animation object.
    """
    # Create grid
    x = np.linspace(x_range[0], x_range[1], nx)
    y = np.linspace(y_range[0], y_range[1], ny)
    X, Y = np.meshgrid(x, y, indexing='xy')
    t_values = np.linspace(t_range[0], t_range[1], n_frames)

    # Select field function
    if field == "energy_density":
        from warpbubblesim.gr.energy import compute_energy_density
        metric_func = metric.get_metric_func()

        def field_func(t, x, y, z):
            coords = np.array([t, x, y, z])
            try:
                return compute_energy_density(metric_func, coords, h=0.01)
            except Exception:
                return 0.0
    elif field == "shape":
        def field_func(t, x, y, z):
            r = metric.r_from_center(t, x, y, z)
            return metric.shape_function(r)
    elif field == "expansion":
        if hasattr(metric, 'expansion_scalar_analytic'):
            field_func = metric.expansion_scalar_analytic
        else:
            from warpbubblesim.gr.adm import compute_shift_divergence
            shift_func = metric.get_shift_func()

            def field_func(t, x, y, z):
                coords = np.array([t, x, y, z])
                return compute_shift_divergence(shift_func, coords)
    else:
        raise ValueError(f"Unknown field: {field}")

    # Precompute all frames for consistent colorscale
    if show_progress:
        print("Precomputing frames...")

    frames_data = []
    vmin_global = np.inf
    vmax_global = -np.inf

    iterator = tqdm(range(n_frames)) if show_progress else range(n_frames)
    for frame_idx in iterator:
        t = t_values[frame_idx]
        Z = np.zeros_like(X)

        for i in range(ny):
            for j in range(nx):
                Z[i, j] = field_func(t, X[i, j], Y[i, j], z)

        frames_data.append(Z)
        vmin_global = min(vmin_global, Z.min())
        vmax_global = max(vmax_global, Z.max())

    # Set up figure
    fig, ax = plt.subplots(figsize=figsize)

    # Symmetric colorbar around zero
    if vmin_global * vmax_global < 0:
        vlim = max(abs(vmin_global), abs(vmax_global))
        norm = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)
    else:
        norm = None

    # Initial plot
    im = ax.pcolormesh(X, Y, frames_data[0], cmap=cmap, norm=norm, shading='auto')
    plt.colorbar(im, ax=ax, label=field)

    # Bubble center marker
    x_s = metric.bubble_center(t_values[0])
    center_marker, = ax.plot([x_s], [0], 'r*', markersize=15)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    title = ax.set_title(f'{metric.name} - {field} (t = {t_values[0]:.2f})')

    def update(frame):
        t = t_values[frame]
        im.set_array(frames_data[frame].ravel())

        # Update bubble center
        x_s = metric.bubble_center(t)
        center_marker.set_data([x_s], [0])

        title.set_text(f'{metric.name} - {field} (t = {t:.2f})')
        return [im, center_marker, title]

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000/fps, blit=True)

    if save_path:
        save_animation(anim, save_path, fps=fps)
        if show_progress:
            print(f"Animation saved to {save_path}")

    return anim


def animate_geodesics(
    metric,
    initial_positions: List[Tuple[float, float, float]],
    t_range: Tuple[float, float] = (0, 10),
    x_range: Optional[Tuple[float, float]] = None,
    n_frames: int = 100,
    fps: int = 30,
    figsize: Tuple[float, float] = (10, 8),
    save_path: Optional[str] = None,
    show_progress: bool = True,
) -> FuncAnimation:
    """
    Animate geodesic worldlines being traced.

    Parameters
    ----------
    metric : WarpMetric
        The warp metric.
    initial_positions : list
        List of (x, y, z) initial positions.
    t_range : tuple
        Time range.
    x_range : tuple, optional
        X-axis limits.
    n_frames : int
        Number of frames.
    fps : int
        Frames per second.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save animation.
    show_progress : bool
        Show progress.

    Returns
    -------
    FuncAnimation
    """
    from warpbubblesim.gr.geodesics import integrate_geodesic

    metric_func = metric.get_metric_func()

    # Integrate all geodesics
    if show_progress:
        print("Integrating geodesics...")

    geodesic_data = []
    for x0, y0, z0 in tqdm(initial_positions) if show_progress else initial_positions:
        initial_coords = np.array([t_range[0], x0, y0, z0])
        u0 = np.array([1.0, 0.0, 0.0, 0.0])

        # Normalize
        g = metric_func(*initial_coords)
        norm_sq = np.einsum('mn,m,n->', g, u0, u0)
        if norm_sq < 0:
            u0 = u0 / np.sqrt(-norm_sq)

        try:
            result = integrate_geodesic(
                metric_func, initial_coords, u0,
                t_range, max_step=0.05
            )
            geodesic_data.append(result['coords'])
        except Exception:
            geodesic_data.append(None)

    # Set up figure
    fig, ax = plt.subplots(figsize=figsize)

    # Determine x range
    if x_range is None:
        all_x = []
        for coords in geodesic_data:
            if coords is not None:
                all_x.extend(coords[:, 1])
        if all_x:
            x_min, x_max = min(all_x), max(all_x)
            margin = 0.1 * (x_max - x_min)
            x_range = (x_min - margin, x_max + margin)
        else:
            x_range = (-10, 10)

    ax.set_xlim(x_range)
    ax.set_ylim(t_range)
    ax.set_xlabel('x')
    ax.set_ylabel('t')

    # Time values for frames
    t_values = np.linspace(t_range[0], t_range[1], n_frames)

    # Plot bubble center
    t_full = np.linspace(t_range[0], t_range[1], 200)
    x_center = [metric.bubble_center(t) for t in t_full]
    ax.plot(x_center, t_full, 'r--', alpha=0.5, label='Bubble center')

    # Initialize lines for geodesics
    cmap = plt.cm.viridis
    lines = []
    markers = []
    for i, coords in enumerate(geodesic_data):
        color = cmap(i / max(1, len(geodesic_data) - 1))
        line, = ax.plot([], [], color=color, linewidth=1.5)
        marker, = ax.plot([], [], 'o', color=color, markersize=6)
        lines.append(line)
        markers.append(marker)

    title = ax.set_title(f'{metric.name} - Geodesics (t = {t_values[0]:.2f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    def update(frame):
        t_current = t_values[frame]

        for i, coords in enumerate(geodesic_data):
            if coords is None:
                continue

            # Find points up to current time
            mask = coords[:, 0] <= t_current
            if mask.any():
                lines[i].set_data(coords[mask, 1], coords[mask, 0])
                # Current position
                idx = np.argmax(coords[:, 0][mask])
                markers[i].set_data([coords[mask][idx, 1]], [coords[mask][idx, 0]])
            else:
                lines[i].set_data([], [])
                markers[i].set_data([], [])

        title.set_text(f'{metric.name} - Geodesics (t = {t_current:.2f})')
        return lines + markers + [title]

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000/fps, blit=True)

    if save_path:
        save_animation(anim, save_path, fps=fps)

    return anim


def animate_grid_distortion(
    metric,
    x_range: Tuple[float, float] = (-5, 5),
    y_range: Tuple[float, float] = (-5, 5),
    t_range: Tuple[float, float] = (0, 10),
    n_lines: int = 15,
    n_frames: int = 100,
    fps: int = 30,
    figsize: Tuple[float, float] = (8, 8),
    save_path: Optional[str] = None,
) -> FuncAnimation:
    """
    Animate coordinate grid showing proper distance distortion.

    Parameters
    ----------
    metric : WarpMetric
        The warp metric.
    x_range, y_range : tuple
        Coordinate ranges.
    t_range : tuple
        Time range.
    n_lines : int
        Number of grid lines per direction.
    n_frames : int
        Number of frames.
    fps : int
        Frames per second.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save.

    Returns
    -------
    FuncAnimation
    """
    fig, ax = plt.subplots(figsize=figsize)

    t_values = np.linspace(t_range[0], t_range[1], n_frames)

    # Grid lines
    x_lines_coord = np.linspace(x_range[0], x_range[1], n_lines)
    y_lines_coord = np.linspace(y_range[0], y_range[1], n_lines)
    n_points = 50

    xs = np.linspace(x_range[0], x_range[1], n_points)
    ys = np.linspace(y_range[0], y_range[1], n_points)

    # Initialize line collections
    v_lines = []
    for x0 in x_lines_coord:
        line, = ax.plot([], [], 'b-', alpha=0.5, linewidth=0.8)
        v_lines.append(line)

    h_lines = []
    for y0 in y_lines_coord:
        line, = ax.plot([], [], 'b-', alpha=0.5, linewidth=0.8)
        h_lines.append(line)

    # Bubble center marker
    center_marker, = ax.plot([], [], 'r*', markersize=20)

    # Background showing shape function
    X, Y = np.meshgrid(xs, ys)
    im = ax.pcolormesh(X, Y, np.zeros_like(X), cmap='Greys', alpha=0.3,
                       vmin=0, vmax=1, shading='auto')

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    title = ax.set_title(f'{metric.name} - Grid (t = 0.00)')

    def update(frame):
        t = t_values[frame]
        x_s = metric.bubble_center(t)

        # Update vertical lines
        for i, x0 in enumerate(x_lines_coord):
            v_lines[i].set_data([x0] * n_points, ys)

        # Update horizontal lines
        for i, y0 in enumerate(y_lines_coord):
            h_lines[i].set_data(xs, [y0] * n_points)

        # Update bubble center
        center_marker.set_data([x_s], [0])

        # Update background (shape function)
        Z = np.zeros_like(X)
        for i in range(len(ys)):
            for j in range(len(xs)):
                r_s = metric.r_from_center(t, xs[j], ys[i], 0)
                Z[i, j] = metric.shape_function(r_s)
        im.set_array(Z.ravel())

        title.set_text(f'{metric.name} - Grid (t = {t:.2f})')

        return v_lines + h_lines + [center_marker, im, title]

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000/fps, blit=True)

    if save_path:
        save_animation(anim, save_path, fps=fps)

    return anim


def save_animation(
    anim: FuncAnimation,
    filepath: str,
    fps: int = 30,
    dpi: int = 150,
    writer: Optional[str] = None,
) -> None:
    """
    Save an animation to file.

    Parameters
    ----------
    anim : FuncAnimation
        The animation object.
    filepath : str
        Output file path (.mp4 or .gif).
    fps : int
        Frames per second.
    dpi : int
        Resolution.
    writer : str, optional
        Writer to use ('ffmpeg', 'pillow'). Auto-detected from extension if None.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if writer is None:
        if filepath.suffix.lower() == '.gif':
            writer = 'pillow'
        else:
            writer = 'ffmpeg'

    if writer == 'ffmpeg':
        try:
            writer_obj = FFMpegWriter(fps=fps, metadata={'title': 'WarpBubbleSim'})
            anim.save(str(filepath), writer=writer_obj, dpi=dpi)
        except Exception:
            # Fallback to pillow for GIF
            if filepath.suffix.lower() != '.gif':
                filepath = filepath.with_suffix('.gif')
            writer_obj = PillowWriter(fps=fps)
            anim.save(str(filepath), writer=writer_obj, dpi=dpi)
    else:
        writer_obj = PillowWriter(fps=fps)
        anim.save(str(filepath), writer=writer_obj, dpi=dpi)


def create_comparison_animation(
    metrics: List,
    field: str = "energy_density",
    x_range: Tuple[float, float] = (-5, 5),
    y_range: Tuple[float, float] = (-5, 5),
    t_range: Tuple[float, float] = (0, 10),
    nx: int = 50,
    ny: int = 50,
    n_frames: int = 50,
    fps: int = 15,
    figsize: Tuple[float, float] = (15, 5),
    save_path: Optional[str] = None,
) -> FuncAnimation:
    """
    Create side-by-side animation comparing multiple metrics.

    Parameters
    ----------
    metrics : list
        List of WarpMetric instances.
    field : str
        Field to visualize.
    Other parameters as in animate_field_evolution.

    Returns
    -------
    FuncAnimation
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]

    x = np.linspace(x_range[0], x_range[1], nx)
    y = np.linspace(y_range[0], y_range[1], ny)
    X, Y = np.meshgrid(x, y, indexing='xy')
    t_values = np.linspace(t_range[0], t_range[1], n_frames)

    # Set up each panel
    images = []
    markers = []
    titles = []

    for i, (metric, ax) in enumerate(zip(metrics, axes)):
        Z = np.zeros_like(X)
        im = ax.pcolormesh(X, Y, Z, cmap='RdBu_r', shading='auto')
        images.append(im)

        marker, = ax.plot([0], [0], 'r*', markersize=10)
        markers.append(marker)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        title = ax.set_title(metric.name)
        titles.append(title)

    plt.tight_layout()

    def update(frame):
        t = t_values[frame]

        for i, metric in enumerate(metrics):
            # Compute field
            Z = np.zeros_like(X)
            for j in range(ny):
                for k in range(nx):
                    if field == "shape":
                        r = metric.r_from_center(t, X[j, k], Y[j, k], 0)
                        Z[j, k] = metric.shape_function(r)
                    elif field == "energy_density":
                        if hasattr(metric, 'eulerian_energy_density_analytic'):
                            Z[j, k] = metric.eulerian_energy_density_analytic(t, X[j, k], Y[j, k], 0)
                        else:
                            r = metric.r_from_center(t, X[j, k], Y[j, k], 0)
                            Z[j, k] = metric.shape_function(r)

            images[i].set_array(Z.ravel())

            x_s = metric.bubble_center(t)
            markers[i].set_data([x_s], [0])
            titles[i].set_text(f'{metric.name} (t={t:.1f})')

        return images + markers + titles

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000/fps, blit=True)

    if save_path:
        save_animation(anim, save_path, fps=fps)

    return anim
