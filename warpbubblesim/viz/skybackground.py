"""
Celestial-sphere backgrounds for warp-bubble sky rendering.

A "sky" here is any callable that maps a unit 3-vector (the asymptotic
spatial direction of an outgoing/escaping null ray) to an RGB colour in
[0, 1]^3.  Two kinds are provided:

- :func:`make_procedural_starfield` — a deterministic random star field
  with magnitudes/colours.  Useful for testing the geometry of the
  renderer without requiring any external assets.
- :func:`make_image_sky` — equirectangular image sampler.  Drop in a
  Milky Way panorama (e.g. ESO/NASA "GigaGalaxy" or any 2:1 sky map)
  and the renderer will pick up the pixel under each ray's escape
  direction.

Conventions
-----------
- Direction vectors are 3-component, will be normalised internally.
- Spherical coordinates: θ ∈ [0, π] is polar angle from +z, φ ∈ [-π, π]
  is azimuth measured from +x toward +y.
- The renderer's coordinate-grid +z axis points "up" relative to the
  bubble; +x is the direction of motion.  The default camera looks
  along +x.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Callable, Optional, Tuple, Union


SkyFunc = Callable[[np.ndarray], np.ndarray]


def _direction_to_spherical(directions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert (..., 3) unit vectors to (theta, phi).

    theta ∈ [0, π] from +z axis, phi ∈ [-π, π] in xy-plane from +x to +y.
    """
    d = np.asarray(directions, dtype=float)
    norm = np.linalg.norm(d, axis=-1, keepdims=True)
    norm = np.where(norm < 1e-30, 1.0, norm)
    d = d / norm
    theta = np.arccos(np.clip(d[..., 2], -1.0, 1.0))
    phi = np.arctan2(d[..., 1], d[..., 0])
    return theta, phi


def make_procedural_starfield(
    n_stars: int = 6000,
    seed: int = 1,
    background: Tuple[float, float, float] = (0.0, 0.0, 0.015),
    star_radius_px: float = 0.7,
    fov_scale: float = 1.0,
    include_milky_way: bool = True,
    bright_star_fraction: float = 0.04,
    soft_blend_top_k: int = 1,
) -> SkyFunc:
    """Build a procedural starfield as a callable sky function.

    Each pixel softly blends the ``soft_blend_top_k`` brightest stars
    whose Gaussian footprints cover it (rather than picking exactly one
    winner-takes-all).  That smooths the temporal flicker you'd
    otherwise get when individual sub-pixel stars pop into and out of a
    pixel between animation frames as the warp deflects rays.

    Starts as point-like at small ``star_radius_px``, becomes a soft
    glow as the kernel grows.  A subset of "bright" stars get amplified
    to give the image visual punch.

    Parameters
    ----------
    n_stars : int
        Number of stars.
    seed : int
        RNG seed (output is deterministic).
    background : tuple
        Sky background RGB (deep, to give black-of-space feel).
    star_radius_px : float
        Star Gaussian sigma, in pixels.  About 0.5–1.0 is right for a
        clean point-like look; larger values give a soft glow.
    fov_scale : float
        Multiplier converting "pixel" units to *radians on the sphere*.
        Pass ``fov_scale = fov_radians / image_size`` from the renderer
        so ``star_radius_px`` literally is pixels.
    include_milky_way : bool
        If True, add a faint Milky-Way-like band.
    bright_star_fraction : float
        Fraction of stars boosted to "bright" (~ +1.0 magnitude) so the
        image has noticeable focal points.
    """
    rng = np.random.default_rng(seed)

    # Uniform points on the unit sphere via inverse CDF
    u = rng.uniform(-1, 1, size=n_stars)
    phi_s = rng.uniform(-np.pi, np.pi, size=n_stars)
    sint = np.sqrt(np.clip(1 - u**2, 0, None))
    star_dirs = np.column_stack([sint * np.cos(phi_s), sint * np.sin(phi_s), u])

    # Magnitudes: heavy-tailed so most stars are dim, a few are obvious.
    base_brightness = np.exp(-rng.exponential(scale=0.7, size=n_stars))
    bright_mask = rng.random(size=n_stars) < bright_star_fraction
    base_brightness[bright_mask] *= rng.uniform(2.0, 5.0, size=bright_mask.sum())
    base_brightness = np.clip(base_brightness, 0.0, 6.0)

    # Colour scatter (B-V analogue → RGB tint)
    bv = rng.normal(loc=0.0, scale=0.5, size=n_stars)
    r_tint = np.clip(1.0 + 0.30 * bv, 0.5, 1.6)
    g_tint = np.clip(1.0 + 0.05 * bv, 0.8, 1.3)
    b_tint = np.clip(1.0 - 0.35 * bv, 0.4, 1.6)
    star_rgb = np.column_stack([r_tint, g_tint, b_tint])

    # Star angular size on the sphere
    sigma_ang = max(star_radius_px * fov_scale, 1e-5)
    inv_two_sigma2 = 1.0 / (2 * sigma_ang ** 2)
    # Effective cutoff radius at which Gaussian = exp(-half) ≈ 0.6 (still
    # contributes); we don't actually clip but the np.where below keeps
    # only contributions above a tiny floor so out-of-range stars drop
    # out of the max.
    bg = np.asarray(background, dtype=float)

    def sky(directions: np.ndarray) -> np.ndarray:
        d = np.asarray(directions, dtype=float)
        flat = d.reshape(-1, 3)
        norm = np.linalg.norm(flat, axis=1, keepdims=True)
        norm = np.where(norm < 1e-30, 1.0, norm)
        flat = flat / norm

        P = flat.shape[0]
        out = np.tile(bg, (P, 1))

        # Process in chunks to bound memory.
        k = max(1, int(soft_blend_top_k))
        chunk = max(256, 2_000_000 // max(n_stars, 1))
        for i in range(0, P, chunk):
            cos_sim = flat[i:i + chunk] @ star_dirs.T          # (chunk, N)
            ang_sq = 2.0 * np.clip(1.0 - cos_sim, 0.0, 2.0)    # ≈ θ²
            kernel = np.exp(-ang_sq * inv_two_sigma2)          # (chunk, N)
            contrib = kernel * base_brightness[None, :]        # (chunk, N)
            # Soft top-k blend: take the k brightest contributions and
            # sum them.  k=1 reproduces the old winner-takes-all; k≥3-5
            # smooths frame-to-frame flicker as rays sweep between
            # nearest-star Voronoi cells.
            if k >= contrib.shape[1]:
                top = contrib
                top_idx = np.broadcast_to(np.arange(contrib.shape[1]),
                                          contrib.shape)
            else:
                top_idx = np.argpartition(-contrib, k - 1, axis=1)[:, :k]
                rows = np.arange(contrib.shape[0])[:, None]
                top = contrib[rows, top_idx]                   # (chunk, k)
            chunk_rgb = (star_rgb[top_idx] * top[..., None]).sum(axis=1)
            out[i:i + chunk] = out[i:i + chunk] + chunk_rgb

        if include_milky_way:
            # Faint band centred on z = 0, slightly tilted (decorative,
            # not real galactic-coordinate alignment).  Uses additive
            # falloff in offset from a great circle whose normal is the
            # tilted +z axis.
            tilt = 0.4
            normal = np.array([0.05, np.sin(tilt), np.cos(tilt)])
            normal = normal / np.linalg.norm(normal)
            offset = flat @ normal
            # Two-component band: a wider cool envelope plus a brighter core.
            wide = np.exp(-(offset / 0.30) ** 2)
            core = np.exp(-(offset / 0.10) ** 2)
            tint_wide = np.array([0.06, 0.08, 0.14])
            tint_core = np.array([0.18, 0.16, 0.22])
            out = out + 0.4 * wide[:, None] * tint_wide
            out = out + 0.5 * core[:, None] * tint_core

        out = np.clip(out, 0.0, 1.0)
        return out.reshape(d.shape)

    return sky


def make_image_sky(
    path: Union[str, Path],
    rotation_deg: float = 0.0,
    flip_horizontal: bool = False,
    gain: float = 1.0,
    interpolation: str = "bilinear",
) -> SkyFunc:
    """Equirectangular sky-image sampler.

    Parameters
    ----------
    path : str | Path
        Path to a 2:1 equirectangular sky image (jpg/png/exr/etc.).  The
        image is read with ``imageio`` if available, else ``matplotlib``.
    rotation_deg : float
        Rotate the sky by this much around the z-axis (azimuthal shift),
        useful for orienting a Milky Way map relative to the camera.
    flip_horizontal : bool
        Flip the longitude axis (some Milky Way panoramas are mirrored).
    gain : float
        Scalar multiplier applied to the sampled colour before clipping.
    interpolation : str
        ``"bilinear"`` (default — smooth, no temporal flicker as rays
        shift between frames) or ``"nearest"`` (faster but aliases).

    Returns
    -------
    SkyFunc
        Callable mapping (..., 3) directions → (..., 3) RGB in [0, 1].
    """
    try:
        import imageio.v2 as iio  # type: ignore
        img = iio.imread(str(path))
    except ImportError:
        from matplotlib.pyplot import imread
        img = imread(str(path))

    img = np.asarray(img, dtype=float)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    if img.max() > 1.5:
        img = img / 255.0

    H, W, _ = img.shape
    rot_rad = np.deg2rad(rotation_deg)

    def sample_nearest(row_f: np.ndarray, col_f: np.ndarray) -> np.ndarray:
        col = np.clip(col_f.astype(int), 0, W - 1)
        row = np.clip(row_f.astype(int), 0, H - 1)
        return img[row, col, :3]

    def sample_bilinear(row_f: np.ndarray, col_f: np.ndarray) -> np.ndarray:
        # Wrap longitude (col_f) so col0 = -1 maps to W-1
        col_f = col_f % W
        col0 = np.floor(col_f).astype(int) % W
        col1 = (col0 + 1) % W
        cf = col_f - np.floor(col_f)

        row0 = np.clip(np.floor(row_f).astype(int), 0, H - 1)
        row1 = np.clip(row0 + 1, 0, H - 1)
        rf = np.clip(row_f - np.floor(row_f), 0.0, 1.0)

        c00 = img[row0, col0, :3]
        c01 = img[row0, col1, :3]
        c10 = img[row1, col0, :3]
        c11 = img[row1, col1, :3]
        cf_col = cf[..., None]
        rf_col = rf[..., None]
        top = c00 * (1 - cf_col) + c01 * cf_col
        bot = c10 * (1 - cf_col) + c11 * cf_col
        return top * (1 - rf_col) + bot * rf_col

    sampler = sample_bilinear if interpolation == "bilinear" else sample_nearest

    def sky(directions: np.ndarray) -> np.ndarray:
        theta, phi = _direction_to_spherical(np.asarray(directions))
        phi = phi + rot_rad
        if flip_horizontal:
            phi = -phi
        phi = (phi + np.pi) % (2 * np.pi) - np.pi

        u = (phi + np.pi) / (2 * np.pi)
        v = theta / np.pi

        col_f = u * W
        row_f = v * H
        out = sampler(row_f, col_f) * gain
        return np.clip(out, 0.0, 1.0)

    return sky


def make_grid_sky(
    spacing_deg: float = 10.0,
    line_width_deg: float = 0.4,
    background: Tuple[float, float, float] = (0.0, 0.0, 0.05),
    line_color: Tuple[float, float, float] = (0.4, 0.6, 0.9),
    pole_color: Tuple[float, float, float] = (1.0, 0.4, 0.4),
) -> SkyFunc:
    """Diagnostic latitude/longitude grid on the celestial sphere.

    Useful for debugging aberration and lensing — the warping of the
    grid lines makes the geometry of the rendering immediately legible.

    Parameters
    ----------
    spacing_deg : float
        Angular spacing of grid lines.
    line_width_deg : float
        Half-width of grid lines (Gaussian-tapered).
    background : tuple
        Background colour where there are no lines.
    line_color : tuple
        Colour of grid lines.
    pole_color : tuple
        Colour applied near +z and -z.
    """
    spacing = np.deg2rad(spacing_deg)
    width = np.deg2rad(line_width_deg)
    bg = np.asarray(background)
    lc = np.asarray(line_color)
    pc = np.asarray(pole_color)

    def sky(directions: np.ndarray) -> np.ndarray:
        theta, phi = _direction_to_spherical(np.asarray(directions))
        # Distance to nearest latitude/longitude line
        d_lat = np.abs(((theta + spacing / 2) % spacing) - spacing / 2)
        d_lon = np.abs(((phi + spacing / 2) % spacing) - spacing / 2)
        line_strength = np.exp(-(d_lat / width) ** 2) + np.exp(-(d_lon / width) ** 2)
        line_strength = np.clip(line_strength, 0.0, 1.0)

        out = np.broadcast_to(bg, theta.shape + (3,)).copy()
        out = out + line_strength[..., None] * (lc - bg)

        # Highlight poles
        pole_strength = np.exp(-(theta / np.deg2rad(8)) ** 2) + np.exp(
            -((np.pi - theta) / np.deg2rad(8)) ** 2
        )
        out = out + pole_strength[..., None] * (pc - bg)
        return np.clip(out, 0.0, 1.0)

    return sky
