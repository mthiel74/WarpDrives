"""
Test Minkowski limit: when v=0 or f=0, spacetime should be flat.

Validates that all tensors vanish for flat spacetime.
"""

import numpy as np
import pytest

from warpbubblesim.metrics import AlcubierreMetric
from warpbubblesim.gr.tensors import (
    compute_christoffel,
    compute_riemann,
    compute_ricci,
    compute_ricci_scalar,
    compute_einstein,
)
from warpbubblesim.gr.invariants import compute_kretschmann


class TestMinkowskiLimit:
    """Tests for the v=0 Minkowski limit."""

    def test_alcubierre_v0_is_minkowski(self):
        """Alcubierre with v=0 should give Minkowski metric."""
        metric = AlcubierreMetric(v0=0.0, R=1.0, sigma=8.0)

        # Check metric at various points
        test_points = [
            (0, 0, 0, 0),
            (1, 2, 0, 0),
            (0, 0, 1, 1),
        ]

        for point in test_points:
            g = metric.metric(*point)

            # Should be Minkowski: diag(-1, 1, 1, 1)
            expected = np.diag([-1, 1, 1, 1])
            np.testing.assert_allclose(g, expected, atol=1e-10,
                                       err_msg=f"Metric not Minkowski at {point}")

    def test_minkowski_christoffel_vanishes(self):
        """Christoffel symbols should vanish for Minkowski."""
        metric = AlcubierreMetric(v0=0.0)
        metric_func = metric.get_metric_func()

        coords = np.array([0.0, 1.0, 0.5, 0.5])
        gamma = compute_christoffel(metric_func, coords, h=1e-5)

        # All Christoffel symbols should be zero
        np.testing.assert_allclose(gamma, 0, atol=1e-8,
                                   err_msg="Christoffel not zero for Minkowski")

    def test_minkowski_riemann_vanishes(self):
        """Riemann tensor should vanish for Minkowski."""
        metric = AlcubierreMetric(v0=0.0)
        metric_func = metric.get_metric_func()

        coords = np.array([0.0, 0.5, 0.5, 0.0])
        riemann = compute_riemann(metric_func, coords, h=1e-5)

        np.testing.assert_allclose(riemann, 0, atol=1e-6,
                                   err_msg="Riemann not zero for Minkowski")

    def test_minkowski_ricci_vanishes(self):
        """Ricci tensor should vanish for Minkowski."""
        metric = AlcubierreMetric(v0=0.0)
        metric_func = metric.get_metric_func()

        coords = np.array([0.0, 0.0, 0.0, 0.0])
        ricci = compute_ricci(metric_func, coords, h=1e-5)

        np.testing.assert_allclose(ricci, 0, atol=1e-6,
                                   err_msg="Ricci not zero for Minkowski")

    def test_minkowski_einstein_vanishes(self):
        """Einstein tensor should vanish for Minkowski."""
        metric = AlcubierreMetric(v0=0.0)
        metric_func = metric.get_metric_func()

        coords = np.array([0.0, 0.0, 0.0, 0.0])
        einstein = compute_einstein(metric_func, coords, h=1e-5)

        np.testing.assert_allclose(einstein, 0, atol=1e-6,
                                   err_msg="Einstein not zero for Minkowski")

    def test_minkowski_kretschmann_vanishes(self):
        """Kretschmann scalar should vanish for Minkowski."""
        metric = AlcubierreMetric(v0=0.0)
        metric_func = metric.get_metric_func()

        coords = np.array([0.0, 0.0, 0.0, 0.0])
        K = compute_kretschmann(metric_func, coords, h=1e-5)

        assert abs(K) < 1e-6, f"Kretschmann not zero for Minkowski: {K}"

    def test_far_from_bubble_is_minkowski(self):
        """Far from the bubble, spacetime should be approximately Minkowski."""
        metric = AlcubierreMetric(v0=1.0, R=1.0, sigma=8.0)

        # Point far from bubble center
        coords = np.array([0.0, 100.0, 0.0, 0.0])

        g = metric.metric(*coords)
        expected = np.diag([-1, 1, 1, 1])

        np.testing.assert_allclose(g, expected, atol=1e-6,
                                   err_msg="Metric not Minkowski far from bubble")


class TestMinkowskiMetricProperties:
    """Test basic metric properties in Minkowski limit."""

    def test_metric_symmetry(self):
        """Metric tensor should be symmetric."""
        metric = AlcubierreMetric(v0=0.5, R=1.0)

        points = [(0, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0)]
        for point in points:
            g = metric.metric(*point)
            np.testing.assert_allclose(g, g.T, atol=1e-15,
                                       err_msg=f"Metric not symmetric at {point}")

    def test_metric_inverse_identity(self):
        """g @ g_inv should give identity."""
        metric = AlcubierreMetric(v0=0.5, R=1.0)

        point = (0, 0.5, 0.5, 0)
        g = metric.metric(*point)
        g_inv = metric.metric_inverse(*point)

        product = g @ g_inv
        np.testing.assert_allclose(product, np.eye(4), atol=1e-10,
                                   err_msg="g @ g_inv not identity")

    def test_metric_signature(self):
        """Metric should have (-,+,+,+) signature."""
        metric = AlcubierreMetric(v0=0.5, R=1.0)

        point = (0, 0, 0, 0)
        g = metric.metric(*point)

        eigenvalues = np.linalg.eigvalsh(g)
        signs = np.sign(eigenvalues)

        # Should have one negative and three positive eigenvalues
        assert np.sum(signs < 0) == 1, "Metric doesn't have one timelike direction"
        assert np.sum(signs > 0) == 3, "Metric doesn't have three spacelike directions"
