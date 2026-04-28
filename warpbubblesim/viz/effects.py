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
