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
    n_stars: int = 4000,
    seed: int = 1,
    background: Tuple[float, float, float] = (0.005, 0.008, 0.02),
    star_radius_px: float = 1.5,
    fov_scale: float = 1.0,
    include_milky_way: bool = True,
) -> SkyFunc:
    """Build a procedural starfield as a callable sky function.

    Stars are sampled uniformly on the unit sphere with a power-law
    magnitude distribution and modest temperature/colour scatter.  A
    soft Gaussian "Milky Way" band is optionally drawn across the
    equator (z ≈ 0) to give a sense of orientation in animations.

    Parameters
    ----------
    n_stars : int
        Number of stars.
    seed : int
        RNG seed (output is deterministic).
    background : tuple
        Sky background RGB.
    star_radius_px : float
        Approximate visual radius of each star, in *angular* units
        scaled by ``fov_scale``.  Stars are blurred by a 3D Gaussian on
        the sphere, so this is a smoothing length, not a hard radius.
    fov_scale : float
        Set to (rendered FOV in radians) / 1 so the angular size of
        stars stays constant when the camera FOV changes.  Defaults to
        1.0; a sensible choice when called from the renderer is
        ``fov_scale = fov_radians / image_size`` so that
        ``star_radius_px`` is genuinely "pixels".
    include_milky_way : bool
        If True, add a faint Gaussian band as a background galaxy.

    Returns
    -------
    SkyFunc
        Callable mapping (..., 3) directions → (..., 3) RGB.
    """
    rng = np.random.default_rng(seed)

    # Uniform points on the unit sphere via inverse CDF
    u = rng.uniform(-1, 1, size=n_stars)
    phi = rng.uniform(-np.pi, np.pi, size=n_stars)
    sint = np.sqrt(np.clip(1 - u**2, 0, None))
    star_dirs = np.column_stack([sint * np.cos(phi), sint * np.sin(phi), u])

    # Magnitudes: dimmer stars are vastly more numerous (rough power law).
    # We use a brightness multiplier in [bright_min, 1].
    log_mag = rng.exponential(scale=0.6, size=n_stars)
    brightness = np.exp(-log_mag)  # ∈ (0, 1]

    # Colour temperature scatter — bias toward white with some red/blue tail.
    bv = rng.normal(loc=0.0, scale=0.4, size=n_stars)  # B-V analogue
    # Map BV to RGB heuristically: positive BV = cooler/redder
    r_tint = np.clip(1.0 + 0.3 * bv, 0.6, 1.4)
    g_tint = np.clip(1.0 + 0.05 * bv, 0.8, 1.2)
    b_tint = np.clip(1.0 - 0.4 * bv, 0.4, 1.4)
    star_rgb = np.column_stack([r_tint, g_tint, b_tint]) * brightness[:, None]

    # Smoothing length on the sphere, in radians.  We treat this as the
    # 1-sigma of a 3D Gaussian in (n̂_pixel · n̂_star).
    # cos(θ) ≈ 1 - θ²/2 → equivalent variance in cos-space is σ_θ² / 2 (approx).
    # We expose star_radius_px as an angular scale times fov_scale.
    sigma_ang = max(star_radius_px * fov_scale, 1e-4)
    inv_two_sigma2 = 1.0 / (2 * sigma_ang**2)

    bg = np.asarray(background, dtype=float)

    def sky(directions: np.ndarray) -> np.ndarray:
        d = np.asarray(directions, dtype=float)
        flat = d.reshape(-1, 3)
        norm = np.linalg.norm(flat, axis=1, keepdims=True)
        norm = np.where(norm < 1e-30, 1.0, norm)
        flat = flat / norm

        # Cosine-similarity to every star: shape (P, N)
        # For large N this would be a memory hog — but for n_stars≈4000 and
        # P up to ~65k it's still fine (≈260M floats worst-case → 1 GB).
        # We chunk over pixels to keep memory bounded.
        out = np.tile(bg, (flat.shape[0], 1))
        chunk = max(256, 4_000_000 // max(n_stars, 1))
        for i in range(0, flat.shape[0], chunk):
            cos_sim = flat[i:i + chunk] @ star_dirs.T
            ang_sq = 2.0 * np.clip(1.0 - cos_sim, 0.0, 2.0)  # ≈ θ²
            weights = np.exp(-ang_sq * inv_two_sigma2)
            out[i:i + chunk] += weights @ star_rgb

        if include_milky_way:
            # Faint Gaussian band centred on z = 0 (the "equator" of the
            # sky), with axes tilted slightly so it doesn't overlap the
            # +x camera direction perfectly.  This is an aesthetic prop,
            # not a real galactic-coordinate transform.
            tilt = 0.35
            normal = np.array([0.0, np.sin(tilt), np.cos(tilt)])
            offset = flat @ normal
            band = np.exp(-(offset / 0.18) ** 2)
            tint = np.array([0.10, 0.12, 0.18])
            out += 0.25 * band[:, None] * tint

        out = np.clip(out, 0.0, 1.0)
        return out.reshape(d.shape)

    return sky


def make_image_sky(
    path: Union[str, Path],
    rotation_deg: float = 0.0,
    flip_horizontal: bool = False,
    gain: float = 1.0,
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

    def sky(directions: np.ndarray) -> np.ndarray:
        theta, phi = _direction_to_spherical(np.asarray(directions))
        phi = phi + rot_rad
        if flip_horizontal:
            phi = -phi
        # Wrap phi to [-π, π]
        phi = (phi + np.pi) % (2 * np.pi) - np.pi

        u = (phi + np.pi) / (2 * np.pi)
        v = theta / np.pi

        col = np.clip((u * W).astype(int), 0, W - 1)
        row = np.clip((v * H).astype(int), 0, H - 1)
        out = img[row, col, :3] * gain
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
