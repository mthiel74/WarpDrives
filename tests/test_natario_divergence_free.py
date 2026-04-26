"""
Regression test: Natário shift vector must be divergence-free.

The defining physical property of the Natário (2002) warp drive is
∇·β = 0 -- the bubble carries the ship without locally expanding or
contracting space, so all the geometric distortion is shear.  An
earlier version of NatarioMetric.shift had a divergence of order
unity in the wall, which silently broke every comparison against
Alcubierre and made the metric pedagogically misleading.
"""

import numpy as np
import pytest

from warpbubblesim.metrics.natario import (
    NatarioMetric,
    NatarioVectorPotentialMetric,
)


SAMPLE_POINTS = [
    (0.5, 0.3, 0.0),
    (1.0, 0.5, 0.0),
    (0.8, 0.2, 0.4),
    (1.2, 0.0, 0.7),
    (-0.5, 0.4, 0.4),
    (1.5, 0.6, 0.3),
]


def _numeric_divergence(shift_func, x, y, z, h=1e-5):
    """Central-difference divergence ∇·β at (x, y, z) at t=0."""
    bx_p = shift_func(0.0, x + h, y, z)[0]
    bx_m = shift_func(0.0, x - h, y, z)[0]
    by_p = shift_func(0.0, x, y + h, z)[1]
    by_m = shift_func(0.0, x, y - h, z)[1]
    bz_p = shift_func(0.0, x, y, z + h)[2]
    bz_m = shift_func(0.0, x, y, z - h)[2]
    return ((bx_p - bx_m) + (by_p - by_m) + (bz_p - bz_m)) / (2.0 * h)


@pytest.mark.parametrize("point", SAMPLE_POINTS)
def test_natario_metric_divergence_free(point):
    """NatarioMetric.shift must satisfy ∇·β ≈ 0 (curl-of-vector-potential form)."""
    metric = NatarioMetric(v0=1.0, R=1.0, sigma=8.0)
    div = metric.verify_divergence_free(0.0, *point)
    # h=1e-5 second-order FD has truncation error ~h^2 = 1e-10, but
    # in practice the shift derivative is also computed by FD with
    # h=1e-6 inside shape_derivative, so the realistic noise floor
    # is ~5e-7.
    assert abs(div) < 1e-5, (
        f"NatarioMetric.shift has nonzero divergence "
        f"{div:.3e} at {point} -- the metric is no longer "
        f"expansion-free"
    )


@pytest.mark.parametrize("point", SAMPLE_POINTS)
def test_vector_potential_metric_divergence_free(point):
    """NatarioVectorPotentialMetric.shift must satisfy ∇·β ≈ 0."""
    metric = NatarioVectorPotentialMetric(v0=1.0, R=1.0, sigma=8.0)
    div = _numeric_divergence(metric.shift, *point)
    assert abs(div) < 1e-5, (
        f"NatarioVectorPotentialMetric.shift has nonzero divergence "
        f"{div:.3e} at {point}"
    )


def test_natario_classes_agree():
    """Both Natario classes implement the same curl construction; their
    shift vectors should agree at every point, including the bubble centre
    where each class takes its r_s ≈ 0 branch."""
    m1 = NatarioMetric(v0=1.0, R=1.0, sigma=8.0)
    m2 = NatarioVectorPotentialMetric(v0=1.0, R=1.0, sigma=8.0)
    # Original off-axis sample plus the bubble centre, which
    # exercises each class's r_s < 1e-10 short-circuit.  Using the
    # exact origin makes both classes hit that branch deterministically.
    points_with_centre = SAMPLE_POINTS + [(0.0, 0.0, 0.0)]
    for point in points_with_centre:
        b1 = m1.shift(0.0, *point)
        b2 = m2.shift(0.0, *point)
        np.testing.assert_allclose(
            b1, b2, rtol=1e-9, atol=1e-12,
            err_msg=f"NatarioMetric and VectorPotential disagree at {point}: "
                    f"{b1} vs {b2}"
        )
