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
