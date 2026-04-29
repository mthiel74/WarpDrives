"""Tests for the warp-bubble sky renderer."""

import numpy as np
import pytest

from warpbubblesim.metrics.alcubierre import AlcubierreMetric
from warpbubblesim.viz.skybackground import (
    make_grid_sky,
    make_procedural_starfield,
)
from warpbubblesim.viz.skyrender import (
    Camera,
    RenderConfig,
    build_orthonormal_tetrad,
    render_sky_view,
    trace_pixel,
)
from warpbubblesim.viz.skyrender_batch import (
    BatchRenderConfig,
    render_frame_batch,
)


def test_tetrad_orthonormal_minkowski():
    """Tetrad construction in Minkowski (v=0) should yield η as Gram matrix."""
    metric = AlcubierreMetric(v0=0.0, R=1.0, sigma=8.0)
    mf = metric.get_metric_func()
    coords = np.array([0.0, 0.0, 0.0, 0.0])
    u = np.array([1.0, 0.0, 0.0, 0.0])
    T = build_orthonormal_tetrad(mf, coords, u)
    g = mf(*coords)
    gram = np.einsum("mn,am,bn->ab", g, T, T)
    expected = np.diag([-1.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(gram, expected, atol=1e-12)


def test_tetrad_orthonormal_alcubierre_at_centre():
    """Tetrad must be orthonormal at the bubble centre even at high v."""
    for v in [0.5, 1.5, 2.5, 5.0]:
        metric = AlcubierreMetric(v0=v, R=1.0, sigma=8.0)
        mf = metric.get_metric_func()
        x_s = metric.bubble_center(0.0)
        coords = np.array([0.0, x_s, 0.0, 0.0])
        u_seed = np.array([1.0, v, 0.0, 0.0])
        nn = float(np.einsum("mn,m,n->", mf(*coords), u_seed, u_seed))
        u = u_seed / np.sqrt(-nn)
        T = build_orthonormal_tetrad(mf, coords, u)
        g = mf(*coords)
        gram = np.einsum("mn,am,bn->ab", g, T, T)
        expected = np.diag([-1.0, 1.0, 1.0, 1.0])
        np.testing.assert_allclose(gram, expected, atol=1e-10,
                                   err_msg=f"v={v}")


def test_minkowski_limit_preserves_direction_scipy():
    """At v=0, every backward null ray must asymptote to its initial n_local."""
    metric = AlcubierreMetric(v0=0.0, R=1.0, sigma=8.0)
    mf = metric.get_metric_func()
    coords = np.array([0.0, 0.0, 0.0, 0.0])
    u = np.array([1.0, 0.0, 0.0, 0.0])
    T = build_orthonormal_tetrad(mf, coords, u)

    targets = [
        (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, 0, 1),
        (0.7071, 0.7071, 0), (0.5773, 0.5773, 0.5773),
    ]
    for n in targets:
        n = np.array(n) / np.linalg.norm(n)
        d = trace_pixel(metric, coords, T, n,
                        escape_radius=4.0, lambda_max=20.0,
                        h=1e-3, rtol=1e-3, atol=1e-5)
        assert d is not None, f"trace_pixel returned None for n={n}"
        np.testing.assert_allclose(d, n, atol=1e-3,
                                   err_msg=f"n={n}, asym={d}")


def test_minkowski_limit_preserves_direction_batch():
    """Same Minkowski sanity check, but through the batch tracer."""
    cfg = BatchRenderConfig(width=8, height=8, fov_deg=90.0,
                            n_steps=80, dlam=0.2, progress=False)
    sky = make_grid_sky(spacing_deg=20.0)
    img = render_frame_batch(
        "alcubierre",
        {"v0": 0.0, "R": 1.0, "sigma": 8.0, "shape": "tanh"},
        sky, cfg,
    )
    assert img.shape == (8, 8, 3)
    # Should be entirely valid (no fallback red); fallback_color is rust
    # red (0.4, 0, 0) so detect by comparing to any pixel having no green/blue
    fallback_match = np.all(np.isclose(img, [0.4, 0.0, 0.0], atol=1e-6), axis=-1)
    assert not fallback_match.any(), "some pixels fell back to error colour"


def test_procedural_starfield_smoke():
    """Procedural starfield must produce finite RGB values in [0, 1]."""
    sky = make_procedural_starfield(
        n_stars=300, seed=0, star_radius_px=1.0, fov_scale=0.01,
    )
    dirs = np.array([
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    rgb = sky(dirs)
    assert rgb.shape == (4, 3)
    assert np.isfinite(rgb).all()
    assert (rgb >= 0).all() and (rgb <= 1).all()


def test_grid_sky_smoke():
    """Diagnostic grid sky must be smooth and bounded."""
    sky = make_grid_sky(spacing_deg=15.0, line_width_deg=0.5)
    rng = np.random.default_rng(0)
    dirs = rng.normal(size=(100, 3))
    rgb = sky(dirs)
    assert rgb.shape == (100, 3)
    assert np.isfinite(rgb).all()


def test_camera_pixel_directions_are_unit():
    cam = Camera(width=16, height=12, fov_deg=80.0)
    d = cam.pixel_directions()
    assert d.shape == (12, 16, 3)
    norms = np.linalg.norm(d, axis=-1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-12)


def test_render_sky_view_returns_image():
    """End-to-end smoke test: low-res render returns a valid (H,W,3) image."""
    metric = AlcubierreMetric(v0=0.0, R=1.0, sigma=8.0)
    cam = Camera(width=8, height=8, fov_deg=90.0)
    sky = make_grid_sky(spacing_deg=20.0, line_width_deg=1.0)
    cfg = RenderConfig(escape_radius_factor=4.0, lambda_max_factor=15.0,
                       show_progress=False, h=2e-3, rtol=1e-3, atol=1e-5)
    img = render_sky_view(metric, sky, camera=cam, config=cfg)
    assert img.shape == (8, 8, 3)
    assert np.isfinite(img).all()
    assert (img >= 0).all() and (img <= 1).all()


def test_batch_renderer_natario_smoke():
    """Natário branch of the batch tracer should at least run."""
    cfg = BatchRenderConfig(width=6, height=6, fov_deg=90.0,
                            n_steps=60, dlam=0.2, progress=False)
    sky = make_grid_sky(spacing_deg=20.0)
    img = render_frame_batch(
        "natario", {"v0": 0.5, "R": 1.0, "sigma": 8.0}, sky, cfg,
    )
    assert img.shape == (6, 6, 3)
    assert np.isfinite(img).all()


def test_doppler_factor_minkowski_unity():
    """A static observer in Minkowski must see f = 1 for every direction."""
    from warpbubblesim.viz.effects import doppler_factor

    eta = np.diag([-1.0, 1.0, 1.0, 1.0])
    u_obs = np.array([1.0, 0.0, 0.0, 0.0])
    # Past-pointing null tangents in random directions
    rng = np.random.default_rng(0)
    dirs = rng.normal(size=(50, 3))
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    k = np.column_stack([-np.ones(50), dirs])  # (-1, n_x, n_y, n_z)
    f = doppler_factor(eta, k, u_obs, k)
    np.testing.assert_allclose(f, 1.0, atol=1e-12)


def test_doppler_factor_alcubierre_forward_blueshift():
    """Looking forward inside an Alcubierre bubble must blueshift."""
    from warpbubblesim.viz.effects import doppler_factor
    from warpbubblesim.viz.skyrender_batch import (
        _make_metric_fn, _tetrad_at_centre, _rk4_step,
    )

    v = 0.5
    params = {"v0": v, "R": 1.0, "sigma": 8.0, "shape": "tanh"}
    mfn = _make_metric_fn("alcubierre", params)
    coords0 = np.array([0.0, 0.0, 0.0, 0.0])
    g0 = mfn(coords0)
    u = np.array([1.0, v, 0.0, 0.0])
    nn = float(np.einsum("mn,m,n->", g0, u, u))
    u = u / np.sqrt(-nn)
    T = _tetrad_at_centre(mfn, coords0, u)

    # Forward ray
    n_local = np.array([1.0, 0.0, 0.0])
    k_init = -T[0] + n_local[0] * T[1] + n_local[1] * T[2] + n_local[2] * T[3]
    state = np.concatenate([coords0, k_init]).reshape(1, 8)
    for _ in range(160):
        state = _rk4_step(state, 0.15, "alcubierre", params)
    k_final = state[0, 4:]

    f = doppler_factor(g0, k_init.reshape(1, 4), u, k_final.reshape(1, 4))
    assert f[0] > 1.05, f"expected blueshift > 1, got {f[0]:.3f}"


def test_apply_doppler_finite_and_unbounded_without_tonemap():
    """Honest f^3 brightness is unbounded; tonemap=False must respect that."""
    from warpbubblesim.viz.effects import apply_doppler

    rgb = np.tile(np.array([0.3, 0.4, 0.5]), (200, 1))
    f = np.linspace(0.05, 20.0, 200)
    out = apply_doppler(rgb, f, intensity_power=3.0, tonemap=False)
    assert np.isfinite(out).all()
    # f=20, intensity 0.3 ⇒ 20³·0.3 = 2400; brightness MUST be allowed
    # to exceed 1, since clipping it would silently rewrite physics.
    assert out.max() > 1000.0, (
        f"Honest Doppler brightness should scale unboundedly with f^3; "
        f"max={out.max():.1f}"
    )


def test_apply_doppler_tonemap_compresses():
    """Optional tonemap=True must compress the unbounded brightness."""
    from warpbubblesim.viz.effects import apply_doppler

    rgb = np.tile(np.array([0.3, 0.4, 0.5]), (200, 1))
    f = np.linspace(0.05, 20.0, 200)
    out = apply_doppler(rgb, f, intensity_power=3.0, tonemap=True)
    assert np.isfinite(out).all()
    # Reinhard asymptote * max channel: 1.6 / 0.5 * 0.5 = 1.6
    assert out.max() < 5.0, f"tonemap should compress, got max={out.max():.1f}"


def test_horizon_mask_basic():
    from warpbubblesim.viz.effects import horizon_mask

    # Three rays: one inside (r_s = 0.5 < R), one outside (r_s = 5 > R), one boundary
    coords = np.array([
        [0.0,  0.5, 0.0, 0.0],   # at x=0.5, bubble at 0 → r_s=0.5 inside
        [0.0,  5.0, 0.0, 0.0],   # outside
        [0.0,  1.4, 0.0, 0.0],   # r_s=1.4, just below safety_factor*R
    ])
    x_s = np.array([0.0, 0.0, 0.0])
    mask = horizon_mask(coords, x_s, bubble_R=1.0, safety_factor=1.5)
    assert mask[0]      # inside
    assert not mask[1]  # outside
    assert mask[2]      # within safety factor


def test_render_with_effects_runs():
    """End-to-end render with all honest effects on at superluminal v."""
    cfg = BatchRenderConfig(
        width=8, height=8, fov_deg=90.0, n_steps=80, dlam=0.2,
        progress=False, enable_doppler=True, enable_horizon_mask=True,
    )
    sky = make_grid_sky(spacing_deg=20.0, line_width_deg=1.0)
    img = render_frame_batch(
        "alcubierre", {"v0": 1.5, "R": 1.0, "sigma": 8.0, "shape": "tanh"},
        sky, cfg,
    )
    assert img.shape == (8, 8, 3)
    assert np.isfinite(img).all()


def test_doppler_brightness_invariant():
    """At v=0 (Minkowski), every pixel should have f=1 → brightness unchanged."""
    from warpbubblesim.viz.effects import apply_doppler
    rgb = np.tile(np.array([0.3, 0.4, 0.5]), (10, 1))
    f = np.ones(10)
    out = apply_doppler(rgb, f, intensity_power=3.0, tonemap=False)
    np.testing.assert_allclose(out, rgb, atol=1e-12)


def test_doppler_no_color_shift():
    """Honest mode applies brightness scaling only — no channel rotation."""
    from warpbubblesim.viz.effects import apply_doppler
    rgb = np.tile(np.array([1.0, 0.0, 0.0]), (5, 1))  # pure red
    f = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
    out = apply_doppler(rgb, f, intensity_power=3.0, tonemap=False)
    # Green and blue channels should remain zero — no synthetic colour shift
    np.testing.assert_allclose(out[:, 1:], 0.0, atol=1e-12)
    # Red scales as f^3
    np.testing.assert_allclose(out[:, 0], f ** 3, rtol=1e-12)


def test_front_wall_glow_off_below_onset():
    """The bloom must contribute zero below ``onset_v``."""
    from warpbubblesim.viz.effects import front_wall_glow
    rgb = np.zeros((10, 3))
    pixel_dirs = np.tile(np.array([1.0, 0.0, 0.0]), (10, 1))
    out = front_wall_glow(rgb, pixel_dirs, v_bubble=0.5, onset_v=0.85)
    np.testing.assert_allclose(out, 0.0, atol=1e-12)


def test_front_wall_glow_active_above_onset_only_forward():
    """Above ``onset_v`` the bloom adds intensity ONLY in the forward cone."""
    from warpbubblesim.viz.effects import front_wall_glow
    rgb = np.zeros((4, 3))
    pixel_dirs = np.array([
        [1.0, 0.0, 0.0],   # forward — should glow
        [0.0, 1.0, 0.0],   # 90°  — outside outer angle, no glow
        [-1.0, 0.0, 0.0],  # backward — no glow
        [0.0, 0.0, 1.0],   # 90°
    ])
    out = front_wall_glow(rgb, pixel_dirs, v_bubble=2.0, onset_v=0.85,
                          inner_angle_deg=5.0, outer_angle_deg=35.0,
                          intensity=0.6)
    assert out[0].sum() > 0.5, "forward pixel should glow"
    np.testing.assert_allclose(out[1], 0.0, atol=1e-9)
    np.testing.assert_allclose(out[2], 0.0, atol=1e-9)
    np.testing.assert_allclose(out[3], 0.0, atol=1e-9)


def test_blackbody_doppler_invariant_at_unity():
    """At f=1 the blackbody-aware Doppler must be the identity."""
    from warpbubblesim.viz.effects import apply_doppler_blackbody
    rgb = np.tile(np.array([0.4, 0.6, 0.8]), (10, 1))
    f = np.ones(10)
    out = apply_doppler_blackbody(rgb, f, T_src=5800.0, intensity_power=4.0)
    np.testing.assert_allclose(out, rgb, atol=1e-9)


def test_blackbody_doppler_dims_at_high_f():
    """At very high f the visible-band emission shifts past visible —
    the blackbody-corrected forward pixel should be DIMMER than the
    f^4 monochromatic prediction (and dimmer than the f=1 source).
    This is the very fix that addresses the SR-Doppler "invisible
    forward" criticism.
    """
    from warpbubblesim.viz.effects import (
        apply_doppler, apply_doppler_blackbody,
    )
    rgb = np.tile(np.array([0.4, 0.4, 0.4]), (5, 1))
    f = np.array([2.0, 5.0, 10.0, 50.0, 200.0])
    bb = apply_doppler_blackbody(rgb, f, T_src=5800.0, intensity_power=4.0)
    mono = apply_doppler(rgb, f, intensity_power=3.0, tonemap=False)
    bb_total = bb.sum(axis=-1)
    mono_total = mono.sum(axis=-1)
    # At very high f the blackbody correction should dim relative to f^3:
    assert bb_total[-1] < mono_total[-1] * 1e-3, (
        f"blackbody mode should be MUCH dimmer than f^3 at f=200; "
        f"bb={bb_total[-1]:.3e} mono={mono_total[-1]:.3e}"
    )
    # And tend toward zero as f → very large:
    assert bb_total[-1] < bb_total[0]


def test_blackbody_visible_fraction_peaks_solar():
    """The visible-fraction approximation should peak near solar T."""
    from warpbubblesim.viz.effects import _blackbody_visible_fraction
    T = np.array([1000.0, 3000.0, 5800.0, 10000.0, 30000.0, 100000.0])
    fr = _blackbody_visible_fraction(T)
    assert np.argmax(fr) == 2, f"visible fraction should peak at solar T, got {fr}"
    # Tails decay
    assert fr[0] < fr[2] and fr[-1] < fr[2]


def test_render_bloom_toggle_changes_output():
    """Enabling the bloom should change the rendered frame at v > onset."""
    cfg_off = BatchRenderConfig(
        width=8, height=8, fov_deg=90.0, n_steps=80, dlam=0.2,
        progress=False, enable_doppler=True,
        enable_front_wall_glow=False,
    )
    cfg_on = BatchRenderConfig(
        width=8, height=8, fov_deg=90.0, n_steps=80, dlam=0.2,
        progress=False, enable_doppler=True,
        enable_front_wall_glow=True, front_wall_intensity=0.6,
        front_wall_onset_v=0.5,
    )
    sky = make_grid_sky(spacing_deg=20.0)
    img_off = render_frame_batch(
        "alcubierre", {"v0": 1.5, "R": 1.0, "sigma": 8.0, "shape": "tanh"},
        sky, cfg_off,
    )
    img_on = render_frame_batch(
        "alcubierre", {"v0": 1.5, "R": 1.0, "sigma": 8.0, "shape": "tanh"},
        sky, cfg_on,
    )
    # Bloom adds non-negative contribution only — `on` ≥ `off` per channel
    assert (img_on >= img_off - 1e-9).all()
    # And differs by a non-zero amount somewhere
    assert (img_on > img_off + 1e-3).any()
