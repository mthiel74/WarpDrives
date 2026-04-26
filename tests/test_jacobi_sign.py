"""
Regression test: Jacobi equation sign.

D²ξ^α/dλ² = -R^α_{βγδ} u^β ξ^γ u^δ  (MTW eq. 11.10 / Wald eq. 3.3.18)

Three audit rounds let a sign error slip through `compute_geodesic_deviation`
and `compute_tidal_acceleration` because every existing test runs in
Minkowski spacetime where R = 0.  These tests force a non-flat
configuration and check the sign explicitly.

Strategy: pick a small initial deviation orthogonal to the geodesic
direction in a region where the curvature is non-trivial (the
Alcubierre wall).  Verify three independent properties:

1. The acceleration changes sign when we swap which two Riemann
   slots ξ goes into -- equivalent to the round-9 brutal-critic
   diagnosis.
2. Comparing against an analytically-evaluated Jacobi acceleration
   computed by direct index ordering as written in MTW.
3. Order-of-magnitude check that the deviation does not blow up
   over a short integration -- forward Euler on a wrong-sign Jacobi
   equation diverges fast.
"""

import numpy as np
import pytest

from warpbubblesim.metrics.alcubierre import AlcubierreMetric
from warpbubblesim.gr.geodesics import (
    integrate_geodesic,
    create_initial_velocity,
    compute_geodesic_deviation,
)
from warpbubblesim.gr.observers import compute_tidal_acceleration
from warpbubblesim.gr.tensors import compute_riemann


def _alcubierre_curved_setup():
    m = AlcubierreMetric(v0=0.5, R=1.0, sigma=8.0)
    metric_func = m.get_metric_func()
    # Pick a coord point in the wall where Riemann is non-trivial.
    coords = np.array([0.0, 1.0, 0.2, 0.0])
    return m, metric_func, coords


def test_tidal_acceleration_matches_mtw_index_order():
    """compute_tidal_acceleration should equal -R^μ_{νρσ} u^ν ξ^ρ u^σ
    computed directly from the Riemann tensor."""
    _, metric_func, coords = _alcubierre_curved_setup()
    R = compute_riemann(metric_func, coords)

    # Pick a non-trivial timelike u and a transverse ξ.
    u = create_initial_velocity(
        metric_func, coords, np.array([1.0, 0.0, 0.0]),
        speed=0.0, timelike=True
    )
    xi = np.array([0.0, 0.0, 1e-3, 0.0])

    # Direct MTW formula: a^μ = -R^μ_{νρσ} u^ν ξ^ρ u^σ
    a_direct = -np.einsum('mnrs,n,r,s->m', R, u, xi, u)

    a_func = compute_tidal_acceleration(metric_func, coords, u, xi)

    np.testing.assert_allclose(
        a_func, a_direct, rtol=1e-9, atol=1e-15,
        err_msg=("compute_tidal_acceleration index order does not match "
                 "MTW eq. 11.10")
    )


def test_jacobi_acceleration_correct_sign():
    """The deviation acceleration must be opposite sign to what you get
    from contracting Riemann with (u, u, ξ).  Riemann is antisymmetric
    in its last pair, so the two contractions differ by exactly -1."""
    _, metric_func, coords = _alcubierre_curved_setup()
    R = compute_riemann(metric_func, coords)
    u = create_initial_velocity(
        metric_func, coords, np.array([1.0, 0.0, 0.0]),
        speed=0.0, timelike=True
    )
    xi = np.array([0.0, 0.0, 1e-3, 0.0])

    # Wrong order (the bug we fixed): swaps the last two contractions.
    a_wrong = -np.einsum('mnrs,n,r,s->m', R, u, u, xi)
    # Correct order (MTW).
    a_correct = -np.einsum('mnrs,n,r,s->m', R, u, xi, u)

    # The two should differ by exactly a sign because Riemann is
    # antisymmetric in the last pair (γ ↔ δ).  Verify the package's
    # implementation matches a_correct, NOT a_wrong.
    a_func = compute_tidal_acceleration(metric_func, coords, u, xi)

    # If a_correct ~ 0 the test is uninformative; require a real signal.
    assert np.linalg.norm(a_correct) > 1e-9, (
        "Test setup is degenerate: Riemann acceleration is zero, "
        "so the sign cannot be checked.  Choose a different point."
    )

    # The function should match a_correct, not a_wrong.
    err_correct = np.linalg.norm(a_func - a_correct)
    err_wrong = np.linalg.norm(a_func - a_wrong)
    assert err_correct < err_wrong, (
        f"compute_tidal_acceleration matches the WRONG-sign formula: "
        f"|a_func - a_correct| = {err_correct:.3e}, "
        f"|a_func - a_wrong| = {err_wrong:.3e}"
    )


def test_geodesic_deviation_finite_over_short_lambda():
    """A proper-sign Jacobi solver returns bounded ξ over a short
    integration.  The wrong-sign equation flips the geometric meaning
    (focusing ↔ defocusing) but ξ stays bounded too -- so this is a
    smoke test, not a sign discriminator.  Sign is enforced by the
    other two tests above."""
    _, metric_func, coords = _alcubierre_curved_setup()
    u = create_initial_velocity(
        metric_func, coords, np.array([1.0, 0.0, 0.0]),
        speed=0.0, timelike=True
    )
    g = integrate_geodesic(metric_func, coords, u, (0.0, 1.0))
    xi0 = np.array([0.0, 0.0, 1e-3, 0.0])
    xi = compute_geodesic_deviation(metric_func, g, xi0)
    assert np.all(np.isfinite(xi)), "Jacobi integration produced non-finite ξ"
    # Trajectory shouldn't blow up by orders of magnitude over λ=1.
    assert np.linalg.norm(xi[-1]) < 1.0, (
        f"|ξ(end)| = {np.linalg.norm(xi[-1]):.3e} -- Jacobi integration "
        f"diverged, suggesting a sign error or instability."
    )
