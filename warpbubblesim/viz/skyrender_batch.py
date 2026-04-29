"""
NumPy-vectorised batch ray tracer for warp-bubble sky rendering.

Runs in pure NumPy (no JAX), so it works on any standard install.  All
pixels are advanced together with a fixed-step RK4 integrator and an
analytic, vectorised metric/Christoffel implementation.  This is the
"fast preview" path:

- 30-100x faster than the per-ray scipy tracer (no Python overhead per
  step, no per-pixel solve_ivp setup, no symbolic Christoffel code)
- still slower than the JAX/GPU tracer for very high resolutions, but
  needs no JAX install

Currently supports the Alcubierre and Natário metrics with the four
shape functions in :mod:`warpbubblesim.metrics.base`.

The integration uses a fixed step Δλ for ``n_steps`` steps.  Choose
them so that ``n_steps * Δλ`` exceeds the time the slowest ray needs
to escape the bubble influence — once outside, the metric is η, the
ray travels in a straight line in coordinate space, and the spatial
direction of its 4-momentum (which is what the renderer reads off) is
conserved, so over-integration is harmless.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

import numpy as np

from warpbubblesim.metrics.base import (
    SHAPE_FUNCTIONS,
    tanh_shape,
)


# ---------------------------------------------------------------------------
# Vectorised shape functions and derivatives
# ---------------------------------------------------------------------------

def _tanh_shape_vec(r, R, sigma):
    arg_plus = np.clip(sigma * (r + R), -20.0, 20.0)
    arg_minus = np.clip(sigma * (r - R), -20.0, 20.0)
    arg_R = np.clip(sigma * R, -20.0, 20.0)
    den = 2.0 * np.tanh(arg_R)
    den = np.where(np.abs(den) < 1e-10, 1e-10, den)
    return (np.tanh(arg_plus) - np.tanh(arg_minus)) / den


def _tanh_shape_deriv(r, R, sigma):
    """Analytic d f/dr for tanh shape."""
    arg_plus = np.clip(sigma * (r + R), -20.0, 20.0)
    arg_minus = np.clip(sigma * (r - R), -20.0, 20.0)
    arg_R = np.clip(sigma * R, -20.0, 20.0)
    den = 2.0 * np.tanh(arg_R)
    den = np.where(np.abs(den) < 1e-10, 1e-10, den)
    sech_p = 1.0 / np.cosh(arg_plus)
    sech_m = 1.0 / np.cosh(arg_minus)
    return sigma * (sech_p ** 2 - sech_m ** 2) / den


def _gaussian_shape_vec(r, R, sigma):
    return np.exp(-0.5 * (r / (R * sigma)) ** 2)


def _gaussian_shape_deriv(r, R, sigma):
    s = R * sigma
    return -(r / s ** 2) * np.exp(-0.5 * (r / s) ** 2)


def _polynomial_shape_vec(r, R, sigma):
    R_eff = R + sigma
    x = r / R_eff
    val = (1 - x ** 2) ** 3
    return np.where(r >= R_eff, 0.0, val)


def _polynomial_shape_deriv(r, R, sigma):
    R_eff = R + sigma
    x = r / R_eff
    val = -6.0 * x * (1 - x ** 2) ** 2 / R_eff
    return np.where(r >= R_eff, 0.0, val)


def _smoothstep_shape_vec(r, R, sigma):
    delta = sigma
    t = np.clip((r - R + delta) / (2 * delta), 0.0, 1.0)
    return 1 - (3 * t ** 2 - 2 * t ** 3)


def _smoothstep_shape_deriv(r, R, sigma):
    delta = sigma
    t_raw = (r - R + delta) / (2 * delta)
    t = np.clip(t_raw, 0.0, 1.0)
    inner = -(6 * t - 6 * t ** 2) / (2 * delta)
    return np.where((t_raw <= 0) | (t_raw >= 1), 0.0, inner)


SHAPE_VEC = {
    "tanh": (_tanh_shape_vec, _tanh_shape_deriv),
    "gaussian": (_gaussian_shape_vec, _gaussian_shape_deriv),
    "polynomial": (_polynomial_shape_vec, _polynomial_shape_deriv),
    "smoothstep": (_smoothstep_shape_vec, _smoothstep_shape_deriv),
}


# ---------------------------------------------------------------------------
# Alcubierre metric — vectorised analytic Christoffel
# ---------------------------------------------------------------------------

def _alcubierre_metric_and_derivs(coords, v0, R, sigma, shape_name, x0):
    """Compute (g_{μν}, ∂_α g_{μν}) for a batch of points.

    Parameters
    ----------
    coords : np.ndarray, shape (P, 4)
        Spacetime points [t, x, y, z] for P pixels.
    v0, R, sigma, x0 : float
    shape_name : str

    Returns
    -------
    g : np.ndarray, shape (P, 4, 4)
    dg : np.ndarray, shape (P, 4, 4, 4)  — ``dg[..., α, μ, ν] = ∂_α g_{μν}``
    """
    f_vec, df_vec = SHAPE_VEC[shape_name]

    t = coords[:, 0]
    x = coords[:, 1]
    y = coords[:, 2]
    z = coords[:, 3]
    P = coords.shape[0]

    x_s = x0 + v0 * t
    dxs = x - x_s  # (P,)
    r2 = dxs ** 2 + y ** 2 + z ** 2 + 1e-30
    r = np.sqrt(r2)
    f = f_vec(r, R, sigma)
    df = df_vec(r, R, sigma)
    vs = v0

    # Partial derivatives of r w.r.t. (t, x, y, z)
    # r = sqrt((x - x0 - v0 t)² + y² + z²)
    # ∂_t r = -v0 (x - x_s) / r
    dr_dt = -vs * dxs / r
    dr_dx = dxs / r
    dr_dy = y / r
    dr_dz = z / r

    df_dt = df * dr_dt
    df_dx = df * dr_dx
    df_dy = df * dr_dy
    df_dz = df * dr_dz

    g = np.zeros((P, 4, 4))
    g[:, 0, 0] = -(1.0 - vs ** 2 * f ** 2)
    g[:, 0, 1] = -vs * f
    g[:, 1, 0] = -vs * f
    g[:, 1, 1] = 1.0
    g[:, 2, 2] = 1.0
    g[:, 3, 3] = 1.0

    dg = np.zeros((P, 4, 4, 4))  # axes (P, α, μ, ν)
    # ∂_α g_{00} = 2 v² f ∂_α f
    for a, df_a in enumerate([df_dt, df_dx, df_dy, df_dz]):
        dg[:, a, 0, 0] = 2 * vs ** 2 * f * df_a
        dg[:, a, 0, 1] = -vs * df_a
        dg[:, a, 1, 0] = -vs * df_a
    return g, dg


# ---------------------------------------------------------------------------
# Natário metric — vectorised, derivatives via central differences
# (the analytic divergence-free shift in natario.py is involved enough
#  that a 4-stencil per RK step is just as cheap and avoids algebra bugs)
# ---------------------------------------------------------------------------

def _natario_metric(coords, v0, R, sigma, x0):
    """Vectorised Natário metric (axisymmetric form), shape (P, 4, 4)."""
    t = coords[:, 0]
    x = coords[:, 1]
    y = coords[:, 2]
    z = coords[:, 3]
    P = coords.shape[0]

    x_s = x0 + v0 * t
    dxs = x - x_s
    r2 = dxs ** 2 + y ** 2 + z ** 2 + 1e-30
    r = np.sqrt(r2)
    n = _tanh_shape_vec(r, R, sigma)
    dn = _tanh_shape_deriv(r, R, sigma)

    bx = -v0 * n
    prefac = -v0 * dn * dxs / r2
    by = prefac * y
    bz = prefac * z

    g = np.zeros((P, 4, 4))
    g[:, 0, 0] = -1.0 + bx * bx + by * by + bz * bz
    g[:, 0, 1] = bx
    g[:, 1, 0] = bx
    g[:, 0, 2] = by
    g[:, 2, 0] = by
    g[:, 0, 3] = bz
    g[:, 3, 0] = bz
    g[:, 1, 1] = 1.0
    g[:, 2, 2] = 1.0
    g[:, 3, 3] = 1.0
    return g


def _natario_metric_and_derivs(coords, v0, R, sigma, x0, h=1e-3):
    """Natário metric + derivatives via centred finite differences."""
    g0 = _natario_metric(coords, v0, R, sigma, x0)
    P = coords.shape[0]
    dg = np.zeros((P, 4, 4, 4))
    for a in range(4):
        cp = coords.copy()
        cm = coords.copy()
        cp[:, a] += h
        cm[:, a] -= h
        gp = _natario_metric(cp, v0, R, sigma, x0)
        gm = _natario_metric(cm, v0, R, sigma, x0)
        dg[:, a, :, :] = (gp - gm) / (2 * h)
    return g0, dg


# ---------------------------------------------------------------------------
# Christoffel and geodesic RHS
# ---------------------------------------------------------------------------

def _christoffel_from_metric(g, dg):
    """Γ^μ_{αβ} from g and ∂g, vectorised over the leading batch axis.

    g : (P, 4, 4)
    dg : (P, 4, 4, 4)   axes (P, α, μ, ν)
    Returns: (P, 4, 4, 4)  axes (P, μ, α, β)
    """
    g_inv = np.linalg.inv(g)
    # T1 = ∂_α g_{βρ} = dg[α, β, ρ]                   axes ↔ dg
    # T2 = ∂_β g_{αρ} = dg[β, α, ρ]                   = dg.transpose(0, 2, 1, 3)
    # T3 = ∂_ρ g_{αβ} = dg[ρ, α, β]                   = dg.transpose(0, 2, 3, 1)
    T1 = dg
    T2 = np.transpose(dg, (0, 2, 1, 3))
    T3 = np.transpose(dg, (0, 2, 3, 1))
    bracket = T1 + T2 - T3  # (P, α, β, ρ)
    gamma = 0.5 * np.einsum("pmr,pabr->pmab", g_inv, bracket)
    return gamma


def _geodesic_rhs(state, metric_kind, params):
    """RHS for the (P, 8) batched state."""
    coords = state[:, :4]
    k = state[:, 4:]

    if metric_kind == "alcubierre":
        g, dg = _alcubierre_metric_and_derivs(
            coords, params["v0"], params["R"], params["sigma"],
            params.get("shape", "tanh"), params.get("x0", 0.0),
        )
    elif metric_kind == "natario":
        g, dg = _natario_metric_and_derivs(
            coords, params["v0"], params["R"], params["sigma"],
            params.get("x0", 0.0),
        )
    else:
        raise ValueError(f"unsupported metric kind: {metric_kind}")

    gamma = _christoffel_from_metric(g, dg)  # (P, μ, α, β)
    a = -np.einsum("pmab,pa,pb->pm", gamma, k, k)
    return np.concatenate([k, a], axis=1)


def _rk4_step(state, dlam, metric_kind, params):
    k1 = _geodesic_rhs(state, metric_kind, params)
    k2 = _geodesic_rhs(state + 0.5 * dlam * k1, metric_kind, params)
    k3 = _geodesic_rhs(state + 0.5 * dlam * k2, metric_kind, params)
    k4 = _geodesic_rhs(state + dlam * k3, metric_kind, params)
    return state + (dlam / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# ---------------------------------------------------------------------------
# Tetrad (single-point, NumPy)
# ---------------------------------------------------------------------------

def _tetrad_at_centre(metric_fn, coords, u, forward=(0, 1, 0, 0), up=(0, 0, 0, 1)):
    g = np.asarray(metric_fn(coords))
    nn = float(np.einsum("mn,m,n->", g, u, u))
    if nn >= 0:
        raise ValueError(f"observer not timelike (g(u,u)={nn})")
    e_t = np.asarray(u, dtype=float) / np.sqrt(-nn)

    def gs(seed, prev):
        v = np.asarray(seed, dtype=float).copy()
        for p in prev:
            inner_pp = float(np.einsum("mn,m,n->", g, p, p))
            inner_pv = float(np.einsum("mn,m,n->", g, p, v))
            v = v - (inner_pv / inner_pp) * p
        norm_sq = float(np.einsum("mn,m,n->", g, v, v))
        return v / np.sqrt(norm_sq)

    e_x = gs(np.array(forward, dtype=float), [e_t])
    e_z = gs(np.array(up, dtype=float), [e_t, e_x])
    seed_y = np.array([0.0, 0.0, 1.0, 0.0])
    e_y = gs(seed_y, [e_t, e_x, e_z])
    spatial = np.column_stack([e_x[1:], e_y[1:], e_z[1:]])
    if np.linalg.det(spatial) < 0:
        e_y = -e_y
    return np.vstack([e_t, e_x, e_y, e_z])


def _pixel_directions(width, height, fov_deg):
    fov = np.deg2rad(fov_deg)
    aspect = width / height
    half_w = np.tan(fov / 2)
    half_h = half_w / aspect
    i = np.arange(width)
    j = np.arange(height)
    u = (2 * (i + 0.5) / width - 1.0) * half_w
    v = (1.0 - 2 * (j + 0.5) / height) * half_h
    uu, vv = np.meshgrid(u, v)
    n = np.stack([np.ones_like(uu), uu, vv], axis=-1)
    n = n / np.linalg.norm(n, axis=-1, keepdims=True)
    return n


# ---------------------------------------------------------------------------
# Top-level renderer
# ---------------------------------------------------------------------------

@dataclass
class BatchRenderConfig:
    """Tunables for :func:`render_frame_batch`."""
    width: int = 256
    height: int = 256
    fov_deg: float = 90.0
    n_steps: int = 240
    dlam: float = 0.15
    progress: bool = True
    enable_doppler: bool = False
    doppler_mode: str = "monochromatic"
    """'monochromatic' (default, f^3 Liouville) or 'blackbody' (thermal-
    spectrum-aware: dims forward at high v as visible shifts to UV)."""
    doppler_intensity_power: float = 3.0
    doppler_T_src: float = 5800.0
    doppler_tonemap: bool = False
    enable_horizon_mask: bool = False
    horizon_safety_factor: float = 1.5
    enable_front_wall_glow: bool = False
    front_wall_onset_v: float = 0.85
    front_wall_intensity: float = 0.6
    front_wall_inner_deg: float = 5.0
    front_wall_outer_deg: float = 35.0


def _make_metric_fn(metric_kind, params):
    """Single-point numpy metric for the tetrad (not vectorised)."""
    if metric_kind == "alcubierre":
        v0 = params["v0"]; R = params["R"]; sigma = params["sigma"]
        shape = params.get("shape", "tanh")
        x0 = params.get("x0", 0.0)
        f_vec, _ = SHAPE_VEC[shape]
        def gfn(coords):
            t, x, y, z = coords
            x_s = x0 + v0 * t
            dxs = x - x_s
            r = float(np.sqrt(dxs ** 2 + y ** 2 + z ** 2 + 1e-30))
            f = float(f_vec(np.array([r]), R, sigma)[0])
            g = np.zeros((4, 4))
            g[0, 0] = -(1.0 - v0 ** 2 * f ** 2)
            g[0, 1] = -v0 * f
            g[1, 0] = -v0 * f
            g[1, 1] = 1.0; g[2, 2] = 1.0; g[3, 3] = 1.0
            return g
        return gfn

    if metric_kind == "natario":
        v0 = params["v0"]; R = params["R"]; sigma = params["sigma"]
        x0 = params.get("x0", 0.0)
        def gfn(coords):
            cs = np.asarray(coords, dtype=float).reshape(1, 4)
            return _natario_metric(cs, v0, R, sigma, x0)[0]
        return gfn

    raise ValueError(f"unsupported metric kind: {metric_kind}")


def render_frame_batch(
    metric_kind: str,
    params: dict,
    sky_fn: Callable[[np.ndarray], np.ndarray],
    config: BatchRenderConfig = BatchRenderConfig(),
    t: float = 0.0,
    fallback_color: Tuple[float, float, float] = (0.4, 0.0, 0.0),
) -> np.ndarray:
    """Render one frame of the windshield view via the NumPy batch tracer.

    Drop-in replacement for :func:`warpbubblesim.viz.skyrender.render_sky_view`,
    much faster but only Alcubierre/Natário for now.
    """
    v0 = float(params["v0"])
    x0 = float(params.get("x0", 0.0))
    x_s = x0 + v0 * t
    coords0 = np.array([t, x_s, 0.0, 0.0], dtype=float)

    metric_fn = _make_metric_fn(metric_kind, params)
    u_seed = np.array([1.0, v0, 0.0, 0.0])
    g0 = metric_fn(coords0)
    nn = float(np.einsum("mn,m,n->", g0, u_seed, u_seed))
    if nn >= 0:
        u_seed = np.array([1.0, 0.0, 0.0, 0.0])
        nn = float(np.einsum("mn,m,n->", g0, u_seed, u_seed))
        if nn >= 0:
            raise RuntimeError(f"no timelike observer at v={v0}")
    u = u_seed / np.sqrt(-nn)
    tetrad = _tetrad_at_centre(metric_fn, coords0, u)
    e_t, e_x, e_y, e_z = tetrad

    dirs_local = _pixel_directions(config.width, config.height, config.fov_deg)
    H, W, _ = dirs_local.shape
    flat = dirs_local.reshape(-1, 3)

    k_init = (-e_t[None, :]
              + flat[:, 0:1] * e_x[None, :]
              + flat[:, 1:2] * e_y[None, :]
              + flat[:, 2:3] * e_z[None, :])  # (P, 4)

    state = np.concatenate(
        [np.broadcast_to(coords0, k_init.shape).copy(), k_init], axis=1
    )

    iterator = range(config.n_steps)
    if config.progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc=f"v={v0:.2f}", unit="step")
        except ImportError:
            pass
    for _ in iterator:
        state = _rk4_step(state, config.dlam, metric_kind, params)

    coords_final = state[:, :4]
    k_final = state[:, 4:]
    spatial = k_final[:, 1:]
    norms = np.linalg.norm(spatial, axis=1, keepdims=True)
    valid = (norms[:, 0] > 1e-12) & np.isfinite(norms[:, 0])
    safe = np.where(norms < 1e-12, 1.0, norms)
    dirs_asym = spatial / safe
    dirs_asym[~valid] = np.array([1.0, 0.0, 0.0])

    rgb = np.asarray(sky_fn(dirs_asym))
    rgb[~valid] = np.asarray(fallback_color)

    # Optional post-processing effects (Doppler / horizon mask).
    from warpbubblesim.viz.effects import (
        doppler_factor, apply_doppler, apply_doppler_blackbody,
        horizon_mask, horizon_color,
        front_wall_glow,
    )
    if config.enable_doppler:
        f = doppler_factor(g0, k_init, u, k_final)
        if config.doppler_mode == "blackbody":
            rgb = apply_doppler_blackbody(
                rgb, f,
                T_src=config.doppler_T_src,
                intensity_power=config.doppler_intensity_power,
                tonemap=config.doppler_tonemap,
            )
        else:
            rgb = apply_doppler(
                rgb, f,
                intensity_power=config.doppler_intensity_power,
                tonemap=config.doppler_tonemap,
            )
    if config.enable_horizon_mask and abs(v0) > 1.0:
        x_s_per_ray = x0 + v0 * coords_final[:, 0]
        R_bubble = float(params.get("R", 1.0))
        trapped = horizon_mask(
            coords_final, x_s_per_ray, R_bubble,
            safety_factor=config.horizon_safety_factor,
        )
        rgb[trapped] = horizon_color()
    if config.enable_front_wall_glow:
        rgb = front_wall_glow(
            rgb, flat,
            v_bubble=v0,
            inner_angle_deg=config.front_wall_inner_deg,
            outer_angle_deg=config.front_wall_outer_deg,
            onset_v=config.front_wall_onset_v,
            intensity=config.front_wall_intensity,
        )
    return rgb.reshape(H, W, 3)


def render_velocity_sweep_batch(
    metric_kind: str,
    base_params: dict,
    velocities: Sequence[float],
    sky_fn: Callable[[np.ndarray], np.ndarray],
    config: BatchRenderConfig = BatchRenderConfig(),
    t: float = 0.0,
):
    """Render a velocity sweep with the batch tracer."""
    frames = []
    for v in velocities:
        params = dict(base_params)
        params["v0"] = float(v)
        frame = render_frame_batch(metric_kind, params, sky_fn, config=config, t=t)
        frames.append(frame)
    return frames
