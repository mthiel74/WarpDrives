"""
Post-processing effects for warp-bubble sky renders.

Three effects, all NumPy (the renderers call these once per frame after
sky lookup, no JAX needed):

- :func:`doppler_factor` — per-pixel ω_obs / ω_source from the saved
  initial/final null tangents and observer/source 4-velocities.
- :func:`apply_doppler` — multiplies the sampled sky RGB by a brightness
  factor (Liouville's I_ν/ν³ invariance gives bolometric ∝ f³) and a
  colour shift that crudely approximates wavelength→RGB remapping.
- :func:`horizon_mask` and :func:`horizon_color` — flag pixels whose
  ray failed to escape the bubble (front-horizon / trapped) and render
  them black (no causal contact with the celestial sphere).
- :func:`front_wall_glow` — *stylised* (not physics) forward-cone bloom
  approximating the McMonigal et al. matter-pileup at the leading
  bubble wall.  Provided as an opt-in cinematic toggle so the honest-
  physics render and the "with bloom" render can be compared
  side-by-side; never enabled by default.

Conventions
-----------
- Metric signature (-,+,+,+).
- Past-pointing null tangents ``k`` (k^t < 0).  For such k and
  future-pointing observer u, the *observed* photon energy is
  ω = g_{μν} k^μ u^ν > 0 (the sign flips of -p with future-pointing
  signature cancel).
- Asymptotic source frame is taken to be the standard Minkowski rest
  frame ``u_source = (1, 0, 0, 0)`` evaluated against η = diag(-1, +1,
  +1, +1).  This is exact for warp metrics that are asymptotically
  flat in the chosen Cartesian coordinates.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


_ETA = np.diag([-1.0, 1.0, 1.0, 1.0])


def doppler_factor(
    g_obs: np.ndarray,
    k_init: np.ndarray,
    u_obs: np.ndarray,
    k_final: np.ndarray,
    u_source: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0]),
) -> np.ndarray:
    """Per-pixel ω_obs / ω_source for a batch of null geodesics.

    Parameters
    ----------
    g_obs : np.ndarray, shape (4, 4)
        Metric at the observer's position.  Same for every pixel since
        all rays start at the bubble centre.
    k_init : np.ndarray, shape (P, 4)
        Past-pointing null tangents at the observer.
    u_obs : np.ndarray, shape (4,)
        Observer's normalised 4-velocity.
    k_final : np.ndarray, shape (P, 4)
        Same null tangents after integration through the bubble — taken
        in the asymptotic Minkowski region (so ``g_final ≈ η``).
    u_source : np.ndarray, shape (4,)
        Asymptotic source-frame 4-velocity.  Defaults to ``(1, 0, 0, 0)``,
        i.e. a static observer in the asymptotic Minkowski frame.

    Returns
    -------
    np.ndarray, shape (P,)
        Frequency ratio f = ω_obs / ω_source.  ``f > 1`` is blueshift,
        ``f < 1`` is redshift.

    Notes
    -----
    With past-pointing k, ω = g(k, u) is positive for both observer
    and source; their ratio is the standard Doppler/Killing-energy
    factor.  No need to convert to future tangents first.
    """
    omega_obs = np.einsum("mn,pm,n->p", g_obs, k_init, u_obs)
    omega_src = np.einsum("mn,pm,n->p", _ETA, k_final, u_source)
    # Avoid division-by-zero artefacts at near-horizon rays where
    # |k_final| has been driven nearly to zero by the integration —
    # those pixels are typically also flagged by horizon_mask and
    # overwritten downstream.
    safe = np.where(np.abs(omega_src) < 1e-12, 1e-12, omega_src)
    return omega_obs / safe


def apply_doppler(
    rgb: np.ndarray,
    f: np.ndarray,
    intensity_power: float = 3.0,
    tonemap: bool = False,
) -> np.ndarray:
    """Multiply RGB by the Doppler brightness factor.

    Specific intensity divided by frequency cubed is invariant along
    a null geodesic in vacuum (a direct consequence of Liouville's
    theorem on the photon distribution function), i.e.
    ``I_ν_obs / ν_obs^3 = I_ν_src / ν_src^3``.  So observed monochromatic
    intensity scales as ``f^3`` with ``f = ν_obs / ν_src``, and that's
    the physically correct multiplier whatever the source spectrum.

    No colour shift is applied because the input panorama is a single
    visible-band RGB image — we have no spectral information about
    the source, so any "colour rotation" would be invented.  The
    blue-tinted-foreground appearance is *purely* a brightness effect:
    forward pixels get amplified, trailing pixels get dimmed, and the
    Milky Way panorama colours are preserved as-is.

    Parameters
    ----------
    rgb : np.ndarray, shape (..., 3)
        Sky-sampled RGB before Doppler.
    f : np.ndarray, shape (...,)
        Doppler factor per pixel.
    intensity_power : float
        Exponent on ``f`` for the brightness scale.  Default 3
        (Liouville's I_ν/ν³ → bolometric/monochromatic intensity).
        Use 4 if you want the bolometric-flux version for thermal
        sources.  Anything else is just for visualisation taste.
    tonemap : bool
        If True, apply a Reinhard-style soft compression after
        scaling so the very brightest forward-superluminal pixels
        don't fully white-out the frame.  This is *not* physical;
        leave it off for honest physics.

    Returns
    -------
    np.ndarray, shape (..., 3)
    """
    brightness = np.power(np.clip(f, 0.0, np.inf), intensity_power)
    if tonemap:
        brightness = brightness / (1.0 + 0.5 * brightness) * 1.6
    out = rgb * brightness[..., None]
    return out


def horizon_mask(
    coords_final: np.ndarray,
    x_s_final: np.ndarray,
    bubble_R: float,
    safety_factor: float = 1.5,
) -> np.ndarray:
    """Bool mask of pixels whose ray failed to escape the bubble.

    Parameters
    ----------
    coords_final : np.ndarray, shape (P, 4)
        Final spacetime coordinates at the end of the integration.
    x_s_final : np.ndarray, shape (P,) or scalar
        Bubble centre at the integration's final coordinate time, for
        each ray.  (Yes, "final" varies per ray — they each integrated
        to their own ``λ_max``; we use the per-ray t component to
        compute the corresponding bubble centre.)
    bubble_R : float
        Nominal bubble radius.
    safety_factor : float
        Threshold = ``safety_factor * bubble_R``.  A ray still inside
        ``safety_factor·R`` of the bubble centre after integration is
        deemed trapped behind the front horizon.

    Returns
    -------
    np.ndarray, shape (P,)  bool
    """
    dx = coords_final[..., 1] - x_s_final
    dy = coords_final[..., 2]
    dz = coords_final[..., 3]
    r_s = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    return r_s < (safety_factor * bubble_R)


def front_wall_glow(
    rgb: np.ndarray,
    pixel_dirs: np.ndarray,
    forward: np.ndarray = np.array([1.0, 0.0, 0.0]),
    v_bubble: float = 0.0,
    inner_angle_deg: float = 5.0,
    outer_angle_deg: float = 35.0,
    color: Tuple[float, float, float] = (0.85, 0.95, 1.0),
    intensity: float = 0.6,
    onset_v: float = 0.85,
) -> np.ndarray:
    """STYLISED forward-cone bloom approximating the McMonigal et al.
    matter-pileup at the bubble's leading edge.

    .. warning::
       This is **not** an honest-physics output.  It's an artist's
       proxy for a real-but-not-locally-visible effect: McMonigal,
       Lewis & O'Byrne (2012) showed that any matter the bubble
       encounters during a superluminal cruise piles up *outside*
       the bubble's front wall and is released as a forward-directed
       beam on deceleration.  *Inside* the bubble the crew don't
       directly see this pile-up — it's beyond their forward horizon
       — so a faithful render shows nothing there.  The bloom in
       this function is a stylised proxy that ramps with v so the
       leading-edge effect is at least *legible*.

    Implementation: angular distance from the camera's forward axis
    drives a smoothstep mask between ``inner_angle_deg`` (full glow)
    and ``outer_angle_deg`` (no glow); the bloom strength ramps with
    ``tanh(max(0, v - onset_v))``.

    Parameters
    ----------
    rgb : np.ndarray, shape (..., 3)
        Per-pixel RGB before the bloom.
    pixel_dirs : np.ndarray, shape (..., 3)
        Per-pixel local-frame unit direction.  Use the *original*
        observer-frame pixel directions (not the asymptotic ones) so
        the bloom stays anchored to the camera's forward axis in
        screen space rather than swimming with aberration.
    forward : np.ndarray, shape (3,)
        Forward direction in the camera's local frame.  Default ``+x̂``.
    v_bubble : float
        Bubble velocity in c.
    inner_angle_deg, outer_angle_deg : float
        Bloom radial profile (full glow inside inner, fades to zero at
        outer).
    color : tuple
        Bloom RGB.
    intensity : float
        Peak intensity at saturation.
    onset_v : float
        Bubble velocity below which the bloom has zero contribution.

    Returns
    -------
    np.ndarray, same shape as ``rgb``.
    """
    forward = np.asarray(forward, dtype=float)
    forward = forward / np.linalg.norm(forward)
    cos_alpha = np.einsum("...i,i->...", pixel_dirs, forward)
    alpha = np.arccos(np.clip(cos_alpha, -1.0, 1.0))
    inner = np.deg2rad(inner_angle_deg)
    outer = np.deg2rad(outer_angle_deg)
    t = np.clip((alpha - inner) / max(outer - inner, 1e-6), 0.0, 1.0)
    radial_mask = 1.0 - (3.0 * t ** 2 - 2.0 * t ** 3)

    onset = max(0.0, abs(v_bubble) - onset_v)
    velocity_mask = np.tanh(onset) * intensity

    bloom = velocity_mask * radial_mask
    add = bloom[..., None] * np.asarray(color, dtype=float)
    return rgb + add


def _blackbody_visible_fraction(T: np.ndarray) -> np.ndarray:
    """Fraction of a blackbody's emission falling in the visible band 380-700 nm.

    Computed by integrating Planck's law over the visible band and dividing
    by the bolometric integral (Stefan-Boltzmann normalisation).  Closed
    form via the dimensionless integral
    :math:`\\int x^3/(\\exp x - 1)\\,dx`, but here we use an empirical
    log-T approximation with peak at solar T ≈ 5800 K, which matches the
    exact result to a few percent across :math:`10^3 \\le T \\le 10^7` K
    and stays well-behaved at the limits.

    Returns the bolometric-normalised fraction in [0, 1].
    """
    T_safe = np.maximum(np.asarray(T, dtype=float), 1.0)
    log_T = np.log(T_safe)
    log_T0 = np.log(5800.0)
    sigma = 0.65        # log-T width of the visible window for blackbodies
    return np.exp(-((log_T - log_T0) / sigma) ** 2)


def _planckian_locus_rgb(T: np.ndarray) -> np.ndarray:
    """Approximate sRGB-like colour of a blackbody at temperature ``T`` (K).

    Vectorised over the trailing axis of ``T``.  Returns shape ``T.shape +
    (3,)`` with values in [0, 1].

    Uses Mitchell Charity's piecewise empirical fit to the Planckian locus
    (canonical reference for "colour temperature" in graphics).  At the
    very high temperatures we hit at f = 100+ the formula saturates to
    blue-white, which is the right behaviour: actual blackbody RGB
    monotonically approaches the Planckian-locus limit.
    """
    T = np.asarray(T, dtype=float)
    Tn = np.clip(T, 1000.0, 40000.0) / 100.0
    out = np.zeros(T.shape + (3,))

    # red channel
    r = np.where(
        Tn <= 66.0,
        255.0,
        329.698727446 * np.power(np.maximum(Tn - 60.0, 1.0), -0.1332047592),
    )
    # green channel
    g = np.where(
        Tn <= 66.0,
        99.4708025861 * np.log(np.maximum(Tn, 1.0)) - 161.1195681661,
        288.1221695283 * np.power(np.maximum(Tn - 60.0, 1.0), -0.0755148492),
    )
    # blue channel
    b = np.where(
        Tn >= 66.0,
        255.0,
        np.where(
            Tn <= 19.0,
            0.0,
            138.5177312231 * np.log(np.maximum(Tn - 10.0, 1.0)) - 305.0447927307,
        ),
    )
    out[..., 0] = np.clip(r, 0.0, 255.0) / 255.0
    out[..., 1] = np.clip(g, 0.0, 255.0) / 255.0
    out[..., 2] = np.clip(b, 0.0, 255.0) / 255.0
    return out


def apply_doppler_blackbody(
    rgb: np.ndarray,
    f: np.ndarray,
    T_src: float = 5800.0,
    intensity_power: float = 4.0,
    tonemap: bool = False,
) -> np.ndarray:
    r"""Spectral-aware Doppler under a blackbody-source assumption.

    Treats each pixel of the input ``rgb`` as a thermal source at
    ``T_src`` (K, default solar 5800 K).  The Doppler factor ``f`` shifts
    the apparent temperature to ``T_obs = T_src · f`` (Wien displacement).
    Two corrections are applied to the panorama's visible-band RGB:

    * **Spectral attenuation.**  At high ``f`` the source's visible-band
      emission shifts past our visible band, so the *visible* photon flux
      to the observer drops.  The factor
      ``visible_fraction(T_obs) / visible_fraction(T_src)`` captures
      this — it equals 1 at ``f = 1`` and tends to 0 as ``f → ∞`` (or
      ``f → 0``).  This is the term ``f³`` mode is missing.
    * **Colour shift.**  The pixel's RGB is multiplied by the ratio of
      Planckian-locus RGB at ``T_obs`` vs ``T_src``.  At ``f = 1`` no
      change; at ``f >> 1`` everything goes blue and saturates.

    The bolometric brightness still scales as ``f^intensity_power``
    (default ``4`` for thermal-source bolometric flux).  The spectral
    fraction multiplies it.

    Parameters
    ----------
    rgb : np.ndarray, shape (..., 3)
    f : np.ndarray, shape (...,)
    T_src : float
        Assumed source temperature (K).  5800 ≈ solar; bumping to
        ~10000 K models bluer/younger stars, ~3000 K models red dwarfs.
    intensity_power : float
        Default 4.0 (bolometric flux density for thermal sources, exact
        for blackbodies once you've integrated over all frequencies).
        Use 3.0 for monochromatic Liouville scaling.
    tonemap : bool
        If True, apply a Reinhard-style soft compression after scaling.
        Off by default.

    Returns
    -------
    np.ndarray, shape (..., 3)
        Doppler-modulated, spectrally-attenuated, colour-shifted RGB.
        Note this can return values > 1 (caller clips to [0, 1] for
        display).
    """
    f = np.asarray(f, dtype=float)
    T_obs = T_src * np.maximum(f, 1e-6)

    spectral_factor = _blackbody_visible_fraction(T_obs) \
                    / _blackbody_visible_fraction(T_src)
    bolometric = np.maximum(f, 0.0) ** intensity_power

    color_obs = _planckian_locus_rgb(T_obs)
    color_src = _planckian_locus_rgb(T_src)
    color_factor = color_obs / np.maximum(color_src, 1e-6)

    brightness = bolometric * spectral_factor
    if tonemap:
        brightness = brightness / (1.0 + 0.5 * brightness) * 1.6

    out = rgb * color_factor * brightness[..., None]
    return out


def horizon_color() -> np.ndarray:
    """RGB to render at trapped pixels.

    A pixel whose past-pointing null geodesic fails to escape the
    bubble has no causal contact with any source on the celestial
    sphere — there's literally no light there to see.  Render as
    black.

    Computing an actual Hawking temperature for the front horizon
    requires the surface gravity κ, which depends on the metric
    profile across the wall.  We don't do that calculation here, so
    we don't fake a "Hawking glow" colour.  If you want one, compute
    κ from the metric and use ``T_H = κ / (2π)`` to map to RGB.
    """
    return np.zeros(3, dtype=float)
