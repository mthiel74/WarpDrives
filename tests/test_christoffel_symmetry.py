"""
Test Christoffel symbol symmetry properties.

Christoffel symbols have symmetry in their lower indices:
Γ^μ_{αβ} = Γ^μ_{βα}
"""

import numpy as np
import pytest

from warpbubblesim.metrics import (
    AlcubierreMetric,
    NatarioMetric,
    VanDenBroeckMetric,
    BobrickMartireMetric,
)
from warpbubblesim.gr.tensors import compute_christoffel


class TestChristoffelSymmetry:
    """Tests for Christoffel symbol symmetry."""

    @pytest.fixture(params=[
        AlcubierreMetric(v0=1.0),
        AlcubierreMetric(v0=0.5),
        NatarioMetric(v0=0.5),
        BobrickMartireMetric(v0=0.3),
    ])
    def metric(self, request):
        """Parameterized metric fixture."""
        return request.param

    def test_lower_index_symmetry(self, metric):
        """Christoffel Γ^μ_{αβ} should equal Γ^μ_{βα}."""
        metric_func = metric.get_metric_func()

        # Test at several points
        test_points = [
            np.array([0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.5, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.5, 0.0]),
            np.array([0.0, 0.5, 0.5, 0.5]),
        ]

        for coords in test_points:
            gamma = compute_christoffel(metric_func, coords, h=1e-5)

            # Check symmetry in lower indices
            for mu in range(4):
                for alpha in range(4):
                    for beta in range(4):
                        np.testing.assert_allclose(
                            gamma[mu, alpha, beta],
                            gamma[mu, beta, alpha],
                            atol=1e-8,
                            err_msg=f"Christoffel not symmetric at {coords}: "
                                    f"Γ^{mu}_{alpha}{beta} != Γ^{mu}_{beta}{alpha}"
                        )

    def test_christoffel_from_metric_derivative(self):
        """Test Christoffel formula against explicit computation."""
        metric = AlcubierreMetric(v0=0.5, R=1.0)
        metric_func = metric.get_metric_func()

        coords = np.array([0.0, 0.5, 0.2, 0.0])
        h = 1e-5

        gamma = compute_christoffel(metric_func, coords, h=h)

        # Compute Christoffel directly from metric and derivatives
        g = metric_func(*coords)
        g_inv = np.linalg.inv(g)

        # Compute metric derivatives
        dg = np.zeros((4, 4, 4))  # dg[α, μ, ν] = ∂_α g_{μν}
        for alpha in range(4):
            coords_plus = coords.copy()
            coords_minus = coords.copy()
            coords_plus[alpha] += h
            coords_minus[alpha] -= h

            g_plus = metric_func(*coords_plus)
            g_minus = metric_func(*coords_minus)

            dg[alpha] = (g_plus - g_minus) / (2 * h)

        # Compute Christoffel from formula
        gamma_check = np.zeros((4, 4, 4))
        for mu in range(4):
            for alpha in range(4):
                for beta in range(4):
                    for nu in range(4):
                        gamma_check[mu, alpha, beta] += 0.5 * g_inv[mu, nu] * (
                            dg[alpha, beta, nu] + dg[beta, alpha, nu] - dg[nu, alpha, beta]
                        )

        np.testing.assert_allclose(gamma, gamma_check, atol=1e-6,
                                   err_msg="Christoffel doesn't match formula")


class TestSpecificChristoffelValues:
    """Test specific Christoffel values for known cases."""

    def test_minkowski_all_zero(self):
        """All Christoffel symbols should be zero for Minkowski."""
        metric = AlcubierreMetric(v0=0.0)
        metric_func = metric.get_metric_func()

        coords = np.array([0.0, 0.0, 0.0, 0.0])
        gamma = compute_christoffel(metric_func, coords, h=1e-5)

        assert np.allclose(gamma, 0, atol=1e-8), "Christoffel not zero for Minkowski"

    def test_christoffel_bounded(self):
        """Christoffel symbols should be bounded for physical metrics."""
        metric = AlcubierreMetric(v0=1.0, R=1.0)
        metric_func = metric.get_metric_func()

        # Sample multiple points
        for x in np.linspace(-2, 2, 5):
            for y in np.linspace(-1, 1, 3):
                coords = np.array([0.0, x, y, 0.0])
                gamma = compute_christoffel(metric_func, coords, h=1e-5)

                # Values should be bounded (not NaN or Inf)
                assert np.all(np.isfinite(gamma)), f"Non-finite Christoffel at {coords}"

                # Reasonable magnitude bound
                assert np.max(np.abs(gamma)) < 1000, f"Unreasonably large Christoffel at {coords}"
