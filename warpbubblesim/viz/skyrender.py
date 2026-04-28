"""
Backward null-geodesic ray tracing through warp-bubble spacetimes.

For each pixel of a virtual "windshield" attached to the spacecraft we
fire a past-pointing null geodesic, integrate it through the warp metric
until it leaves the bubble's sphere of influence, and look up the colour
of the celestial sphere in the asymptotic escape direction.  This gives
us aberration + warp lensing as seen by the crew, without any
small-velocity approximations: the Alcubierre/Natário/etc. metrics are
fed straight into the geodesic integrator.

Key pieces
----------
:func:`build_orthonormal_tetrad`
    Gram-Schmidt an orthonormal frame {e_t̂, e_x̂, e_ŷ, e_ẑ} from the
    observer's 4-velocity using the spacetime metric.  Pixels are mapped
    to past-null directions in this frame; that's where aberration
    "lives".
:func:`pixel_directions`
    Convert image-space (i, j) → unit local 3-vector for a perspective
    camera, given FOV / orientation.
:func:`trace_pixel`
    Integrate one backward null geodesic and return its asymptotic
    spatial direction (or ``None`` if the ray fell into a problematic
    region — caustics, λ_max exceeded, etc.).
:func:`render_sky_view`
    End-to-end: build tetrad, march all pixels, sample the sky, return
    an (H, W, 3) image in [0, 1].
:func:`render_velocity_sweep`
    Convenience: render a series of frames as v_bubble varies from 0 to
    superluminal.  Returns a list of images suitable for
    ``imageio.mimsave``.

Conventions
-----------
- Camera looks along +x by default (the bubble's direction of motion).
- +z is "up" in the local rest frame.
- Backward integration: the past-pointing tangent has k^t < 0 in the
  observer's frame, so the integrator's λ runs forward but coordinate
  time ``state[0]`` decreases.
- "Asymptotic" direction = spatial part of the ray's coordinate
  4-momentum after it has escaped to r_s > escape_radius.  In the
  asymptotically Minkowski region the metric is flat so the spatial
  direction is conserved and equals where the ray came from on the
  celestial sphere.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

from warpbubblesim.gr.geodesics import integrate_geodesic
from warpbubblesim.metrics.base import WarpMetric
from warpbubblesim.viz.skybackground import SkyFunc


# ---------------------------------------------------------------------------
# Tetrads and pixel rays
# ---------------------------------------------------------------------------

def _metric_inner(g: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return float(np.einsum("mn,m,n->", g, a, b))


def build_orthonormal_tetrad(
    metric_func: Callable,
    coords: np.ndarray,
    u: np.ndarray,
    forward_axis: np.ndarray = np.array([0.0, 1.0, 0.0, 0.0]),
    up_axis: np.ndarray = np.array([0.0, 0.0, 0.0, 1.0]),
) -> np.ndarray:
    """Gram-Schmidt an orthonormal tetrad attached to observer u.

    Returns a (4, 4) array whose rows are e_t̂, e_x̂, e_ŷ, e_ẑ in the
    coordinate basis.  e_t̂ = u (assumed timelike, normalised to -1).
    e_x̂ is constructed from ``forward_axis`` (the camera "look"
    direction in coordinate components), and e_ẑ from ``up_axis``;
    e_ŷ closes the right-handed frame.

    Parameters
    ----------
    metric_func : callable
        ``(t, x, y, z) -> g_{μν}``.
    coords : np.ndarray, shape (4,)
        Spacetime point.
    u : np.ndarray, shape (4,)
        4-velocity of the observer.  Must be timelike with respect to
        ``metric_func(*coords)``.
    forward_axis, up_axis : np.ndarray, shape (4,)
        Coordinate-basis vectors used as seeds for the spatial legs
        before Gram-Schmidt.  The default makes the camera look along
        +x with +z as up — the natural choice for bubble flight.

    Returns
    -------
    np.ndarray, shape (4, 4)
        Tetrad rows ``[e_t̂, e_x̂, e_ŷ, e_ẑ]``.
    """
    g = metric_func(*coords)

    # Normalize u to be timelike unit
    u_norm_sq = _metric_inner(g, u, u)
    if u_norm_sq >= 0:
        raise ValueError(
            f"Observer 4-velocity is not timelike (g(u,u) = {u_norm_sq:.3e})"
        )
    e_t = np.asarray(u, dtype=float) / np.sqrt(-u_norm_sq)

    def gs(seed: np.ndarray, prev: List[np.ndarray]) -> np.ndarray:
        v = np.asarray(seed, dtype=float).copy()
        for p in prev:
            inner_pp = _metric_inner(g, p, p)
            inner_pv = _metric_inner(g, p, v)
            v = v - (inner_pv / inner_pp) * p
        norm_sq = _metric_inner(g, v, v)
        if norm_sq <= 1e-12:
            raise ValueError(
                "Gram-Schmidt produced a non-spacelike vector; pick a "
                "different forward_axis/up_axis seed."
            )
        return v / np.sqrt(norm_sq)

    e_x = gs(forward_axis, [e_t])
    e_z = gs(up_axis, [e_t, e_x])
    # e_ŷ is fixed by right-handedness.  We can't take a coordinate cross
    # product (the metric isn't δ), so we Gram-Schmidt against a seed and
    # then flip its sign if the resulting frame is left-handed (det of
    # the spatial 3x3 in coordinate basis is negative).
    seed_y = np.array([0.0, 0.0, 1.0, 0.0])
    e_y = gs(seed_y, [e_t, e_x, e_z])

    spatial = np.column_stack([e_x[1:], e_y[1:], e_z[1:]])
    if np.linalg.det(spatial) < 0:
        e_y = -e_y

    return np.vstack([e_t, e_x, e_y, e_z])


@dataclass
class Camera:
    """Pinhole camera attached to an observer's local rest frame.

    Conventions:
    - The camera looks along its local ``+x̂`` (forward leg of the
      tetrad).
    - Image rows go from top (highest +ẑ) to bottom (-ẑ).
    - Image columns go from left (-ŷ) to right (+ŷ).

    Parameters
    ----------
    width, height : int
        Image dimensions in pixels.
    fov_deg : float
        Horizontal field of view in degrees.
    """

    width: int = 256
    height: int = 256
    fov_deg: float = 90.0

    def pixel_directions(self) -> np.ndarray:
        """Return (H, W, 3) array of unit ray directions in tetrad-local axes.

        Each direction is in the orthonormal frame as
        ``(n_x̂, n_ŷ, n_ẑ)`` — purely spatial, length 1.
        """
        H, W = self.height, self.width
        fov = np.deg2rad(self.fov_deg)
        aspect = W / H
        # Image plane at unit distance: half-width = tan(fov/2)
        half_w = np.tan(fov / 2)
        half_h = half_w / aspect

        i = np.arange(W)
        j = np.arange(H)
        u = (2 * (i + 0.5) / W - 1.0) * half_w   # left-right → +ŷ
        v = (1.0 - 2 * (j + 0.5) / H) * half_h   # top-bottom → +ẑ
        uu, vv = np.meshgrid(u, v)
        n_x = np.ones_like(uu)
        n_y = uu
        n_z = vv
        d = np.stack([n_x, n_y, n_z], axis=-1)
        d = d / np.linalg.norm(d, axis=-1, keepdims=True)
        return d


# ---------------------------------------------------------------------------
# Geodesic ray tracing
# ---------------------------------------------------------------------------

def _make_escape_event(
    metric: WarpMetric,
    escape_radius: float,
):
    """Return a solve_ivp event that fires when the ray leaves the bubble.

    The ray is "outside" the bubble influence when its distance from
    the bubble centre at the *current* coordinate time exceeds
    ``escape_radius``.  Since the bubble centre moves with t, we use
    ``metric.r_from_center`` rather than a static threshold.
    """

    def event(lam, state):
        t, x, y, z = state[:4]
        r = metric.r_from_center(t, x, y, z)
        return escape_radius - r  # > 0 inside, < 0 outside; root at boundary

    event.terminal = True
    event.direction = -1  # crossing from inside to outside
    return event


def _local_to_coord_null_tangent(
    tetrad: np.ndarray, n_local: np.ndarray
) -> np.ndarray:
    """Convert a unit local direction ``n_local`` (in observer frame) to
    a past-pointing null 4-vector in the coordinate basis.

    Convention: a photon observed *coming from* direction n̂ in the
    observer's frame has past-pointing tangent ``k = -e_t̂ + n̂^î e_î``
    (so it travels backward in the observer's time toward the source).
    """
    e_t = tetrad[0]
    e_x = tetrad[1]
    e_y = tetrad[2]
    e_z = tetrad[3]
    return -e_t + n_local[0] * e_x + n_local[1] * e_y + n_local[2] * e_z


def trace_pixel(
    metric: WarpMetric,
    coords: np.ndarray,
    tetrad: np.ndarray,
    n_local: np.ndarray,
    escape_radius: float,
    lambda_max: float,
    rtol: float = 3e-4,
    atol: float = 3e-6,
    max_step: float = 0.5,
    method: str = "RK23",
    backend: str = "finite_difference",
    h: float = 5e-4,
) -> Optional[np.ndarray]:
    """Trace one backward null geodesic and return its escape direction.

    Returns
    -------
    np.ndarray (3,) or None
        Unit spatial direction of the ray's 4-momentum once it has
        escaped the bubble (``r_s > escape_radius``).  ``None`` if the
        ray exceeded ``lambda_max`` without escaping (e.g. trapped, or
        bubble too large for the chosen budget).
    """
    metric_func = metric.get_metric_func()
    k = _local_to_coord_null_tangent(tetrad, n_local)
    event = _make_escape_event(metric, escape_radius)

    try:
        result = integrate_geodesic(
            metric_func,
            np.asarray(coords, dtype=float),
            k,
            (0.0, lambda_max),
            backend=backend,
            h=h,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            method=method,
            dense_output=False,
            renormalize=False,
            events=[event],
        )
    except Exception:
        return None

    sol = result["solution"]
    fired = (
        sol.t_events is not None
        and len(sol.t_events) > 0
        and len(sol.t_events[0]) > 0
    )
    if fired:
        evt_state = sol.y_events[0][0]
        k_final = evt_state[4:]
    else:
        # Didn't escape; use last sample as a best effort
        k_final = result["velocity"][-1]

    # Asymptotic spatial direction: the ray is outside the bubble in
    # asymptotically flat space.  For a past-pointing null tangent
    # k = (k^t, k^i) with k^t < 0, the photon's spatial direction of
    # *origin on the sky* coincides with the spatial part k^i (because
    # the source lies in the direction the past-tangent points).
    spatial = k_final[1:4]
    norm = np.linalg.norm(spatial)
    if norm < 1e-12 or not np.isfinite(norm):
        return None
    return spatial / norm


# ---------------------------------------------------------------------------
# Full-frame rendering
# ---------------------------------------------------------------------------

@dataclass
class RenderConfig:
    """Tunables for :func:`render_sky_view`."""
    escape_radius_factor: float = 6.0
    """Stop a ray when ``r_s > escape_radius_factor * R``."""
    lambda_max_factor: float = 30.0
    """Cap affine parameter at ``lambda_max_factor * R``."""
    rtol: float = 3e-4
    atol: float = 3e-6
    max_step: float = 0.5
    method: str = "RK23"
    h: float = 5e-4
    n_jobs: int = 1
    """Process pool size for ray parallelism.  1 = serial."""
    show_progress: bool = True


def _render_chunk(args):
    """Worker for multiprocessing render — picklable top-level fn."""
    (metric, coords, tetrad, dirs, escape_r, lam_max, cfg) = args
    out = np.full((dirs.shape[0], 3), np.nan)
    for k, n_local in enumerate(dirs):
        d = trace_pixel(
            metric, coords, tetrad, n_local,
            escape_radius=escape_r,
            lambda_max=lam_max,
            rtol=cfg.rtol, atol=cfg.atol,
            max_step=cfg.max_step, method=cfg.method,
            h=cfg.h,
        )
        if d is not None:
            out[k] = d
    return out


def render_sky_view(
    metric: WarpMetric,
    sky: SkyFunc,
    camera: Camera = Camera(),
    t: float = 0.0,
    config: RenderConfig = RenderConfig(),
    fallback_color: Tuple[float, float, float] = (0.4, 0.0, 0.0),
) -> np.ndarray:
    """Render the view from the bubble centre at coordinate time ``t``.

    The observer is placed at the bubble centre with 4-velocity
    co-moving with the bubble (so the spacecraft "sits still" in its
    own frame and the universe streams past as v_bubble varies).

    Parameters
    ----------
    metric : WarpMetric
        Any subclass exposing ``bubble_center(t)``, ``bubble_velocity(t)``
        and a metric function.
    sky : SkyFunc
        Sky background callable; see ``warpbubblesim.viz.skybackground``.
    camera : Camera
        Image dimensions and FOV.  Defaults to 256×256 / 90°.
    t : float
        Coordinate time at which to render.
    config : RenderConfig
        Numerical tunables.
    fallback_color : tuple
        Pixel colour used when a ray fails to escape (caustics, budget
        exhausted).  Default rust-red so failures are visible.

    Returns
    -------
    np.ndarray, shape (H, W, 3) float in [0, 1]
        Rendered RGB frame.
    """
    x_s = metric.bubble_center(t)
    v_s = metric.bubble_velocity(t)
    coords = np.array([t, x_s, 0.0, 0.0], dtype=float)

    # Co-moving 4-velocity at the bubble centre.  For Alcubierre/Natário/
    # similar with α=1, γ_ij=δ_ij, β^x = -v_s · f(0), with f(0)=1 we
    # have β^x = -v_s and the choice u = (1, v_s, 0, 0) gives g(u,u)=-1
    # (algebra in the docstring).  This is the natural "spacecraft at
    # rest in the bubble" worldline.
    u_seed = np.array([1.0, v_s, 0.0, 0.0])
    metric_func = metric.get_metric_func()
    g0 = metric_func(*coords)
    u_norm_sq = _metric_inner(g0, u_seed, u_seed)
    if u_norm_sq >= 0:
        # Numerical edge: at very high v with non-trivial shape function
        # the "static observer at centre" worldline can become non-timelike
        # in some metrics that don't have f(0)=1 exactly.  Fall back to
        # a normalised t-axis tangent if available.
        u_seed = np.array([1.0, 0.0, 0.0, 0.0])
        u_norm_sq = _metric_inner(g0, u_seed, u_seed)
        if u_norm_sq >= 0:
            raise RuntimeError(
                "No timelike observer worldline found at bubble centre "
                f"for v_s={v_s}; metric={metric.name}"
            )
    u = u_seed / np.sqrt(-u_norm_sq)

    tetrad = build_orthonormal_tetrad(metric_func, coords, u)

    dirs = camera.pixel_directions()
    H, W, _ = dirs.shape
    flat_dirs = dirs.reshape(-1, 3)

    R = float(metric.params.get("R", 1.0))
    escape_r = config.escape_radius_factor * R
    lam_max = config.lambda_max_factor * R

    asymptotic = np.full((flat_dirs.shape[0], 3), np.nan)

    if config.n_jobs > 1:
        from multiprocessing import Pool, cpu_count
        n_jobs = min(config.n_jobs, cpu_count())
        # Split flat_dirs into n_jobs chunks
        chunks = np.array_split(flat_dirs, n_jobs)
        args = [
            (metric, coords, tetrad, ch, escape_r, lam_max, config)
            for ch in chunks
        ]
        with Pool(n_jobs) as pool:
            results = pool.map(_render_chunk, args)
        asymptotic = np.vstack(results)
    else:
        iterator = range(flat_dirs.shape[0])
        if config.show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc=f"v={v_s:.2f}", unit="ray")
            except ImportError:
                pass
        for k in iterator:
            d = trace_pixel(
                metric, coords, tetrad, flat_dirs[k],
                escape_radius=escape_r, lambda_max=lam_max,
                rtol=config.rtol, atol=config.atol,
                max_step=config.max_step, method=config.method,
                h=config.h,
            )
            if d is not None:
                asymptotic[k] = d

    # Sample the sky.  Replace NaN rows with fallback before sampling
    # to keep the SkyFunc free of validity logic.
    valid_mask = np.isfinite(asymptotic).all(axis=-1)
    safe_dirs = asymptotic.copy()
    safe_dirs[~valid_mask] = np.array([1.0, 0.0, 0.0])  # arbitrary
    rgb = sky(safe_dirs)
    rgb[~valid_mask] = np.asarray(fallback_color)
    return rgb.reshape(H, W, 3)


def render_velocity_sweep(
    metric_factory: Callable[[float], WarpMetric],
    velocities: Sequence[float],
    sky: SkyFunc,
    camera: Camera = Camera(),
    config: RenderConfig = RenderConfig(),
    t: float = 0.0,
) -> List[np.ndarray]:
    """Render a sequence of frames as the bubble velocity varies.

    Parameters
    ----------
    metric_factory : callable
        ``v -> WarpMetric``.  Receives the velocity for each frame and
        returns a freshly-parameterised metric.  Lets the caller pick
        which warp drive (Alcubierre, Natário, ...) and which other
        knobs are constant across the sweep.
    velocities : sequence of float
        Bubble velocities (in units of c) — anywhere from 0 to many c.
    sky : SkyFunc
        Celestial-sphere background.
    camera, config, t :
        Forwarded to :func:`render_sky_view`.

    Returns
    -------
    list of np.ndarray
        One ``(H, W, 3)`` frame per velocity, in input order.
    """
    frames: List[np.ndarray] = []
    for v in velocities:
        metric = metric_factory(v)
        frame = render_sky_view(metric, sky, camera=camera, t=t, config=config)
        frames.append(frame)
    return frames


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def save_frames_as_animation(
    frames: Sequence[np.ndarray],
    path: str,
    fps: int = 12,
    loop: int = 0,
) -> None:
    """Save a list of (H, W, 3) float frames as a GIF/MP4 via imageio.

    Modern imageio's Pillow backend dropped ``fps=`` in favour of
    ``duration=`` (ms per frame); we forward whichever the installed
    backend accepts.  MP4 still uses ``fps``.
    """
    import imageio.v2 as iio
    arr = [np.clip(f * 255, 0, 255).astype(np.uint8) for f in frames]
    if path.lower().endswith(".gif"):
        try:
            iio.mimsave(path, arr, duration=1000.0 / fps, loop=loop)
        except TypeError:
            # Older imageio: fall back to fps=
            iio.mimsave(path, arr, fps=fps, loop=loop)
    else:
        iio.mimsave(path, arr, fps=fps)
