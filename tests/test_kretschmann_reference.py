"""
Regression test: Kretschmann scalar against a known analytic value.

The codebase's Minkowski limit tests verify all curvature scalars
vanish at flat points, but never check that they reproduce a known
non-zero analytic value.  Without that, a coefficient error in the
Kretschmann or Riemann pipeline could pass the entire suite.

Schwarzschild is the canonical reference.  In Schwarzschild
coordinates with G = c = 1:

    K = R_{αβγδ} R^{αβγδ} = 48 M^2 / r^6

This test instantiates an inline Schwarzschild metric in spherical-ish
Cartesian coordinates (t, x, 0, 0) along the +x axis (so r = x), and
checks the Kretschmann scalar at a point well outside the event
horizon (r >> 2M) where finite-difference accuracy is good.
"""

import numpy as np
import pytest

from warpbubblesim.gr.invariants import compute_kretschmann


def schwarzschild_metric(M=0.1):
    """
    Schwarzschild metric in standard (t, r, θ, φ) coordinates.

    Line element:
        ds² = -(1 - 2M/r) dt² + dr²/(1 - 2M/r)
              + r² (dθ² + sin²θ dφ²)

    The compute_kretschmann function expects metric_func(t, x, y, z)
    -> 4x4 — it just finite-differences with respect to those four
    arguments, so identifying x↔r, y↔θ, z↔φ works as long as the
    finite-difference step doesn't cross the event horizon at r=2M.

    Returns: callable matching the package's metric_func protocol.
    """
    def g(t, r, theta, phi):
        del t, phi  # static, spherically symmetric
        if r <= 2.0 * M:
            raise ValueError("inside event horizon")
        f = 1.0 - 2.0 * M / float(r)
        out = np.zeros((4, 4))
        out[0, 0] = -f
        out[1, 1] = 1.0 / f
        out[2, 2] = float(r) ** 2
        out[3, 3] = float(r) ** 2 * float(np.sin(theta)) ** 2
        return out
    return g


@pytest.mark.parametrize("r,M", [(10.0, 0.1), (20.0, 0.2), (50.0, 0.5)])
def test_schwarzschild_kretschmann(r, M):
    """K_Schwarzschild = 48 M² / r⁶ at any point off the horizon.

    Test at θ = π/2 (equatorial plane) so all metric components are
    well-conditioned and the finite differences along the angular
    directions are clean.  φ choice is irrelevant by symmetry.
    """
    metric = schwarzschild_metric(M=M)
    coords = np.array([0.0, r, np.pi / 2.0, 0.0])

    # Step size for Christoffel finite-difference.  Riemann uses
    # nested differences so cumulative error scales as ~h; for h=1e-3
    # at r ~ 10, the relative error on K should be O(1%).
    K_numeric = compute_kretschmann(metric, coords, h=1e-3)
    K_analytic = 48.0 * M**2 / r**6

    rel_err = abs(K_numeric - K_analytic) / abs(K_analytic)
    # 5% tolerance reflects the cumulative truncation+roundoff error
    # of nested second-difference Riemann finite differencing.
    assert rel_err < 5e-2, (
        f"Schwarzschild Kretschmann mismatch at (r={r}, M={M}): "
        f"numeric = {K_numeric:.4e}, analytic = {K_analytic:.4e}, "
        f"rel err = {rel_err:.2e}"
    )
