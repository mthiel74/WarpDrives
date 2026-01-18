"""
Test that Alcubierre metric has negative energy density in the wall region.

This is a fundamental property of the Alcubierre warp drive that
demonstrates the need for "exotic matter" (negative energy).
"""

import numpy as np
import pytest

from warpbubblesim.metrics import AlcubierreMetric
from warpbubblesim.gr.energy import compute_energy_density


class TestAlcubierreEnergyDensity:
    """Tests for Alcubierre energy density properties."""

    def test_negative_energy_in_wall(self):
        """Energy density should be negative in the bubble wall region."""
        metric = AlcubierreMetric(v0=1.0, R=1.0, sigma=8.0)
        metric_func = metric.get_metric_func()

        # Sample points in the wall region (around r = R)
        R = metric.params['R']
        wall_points = [
            (0, R, 0.5, 0),      # In wall, slightly off-axis
            (0, R, 0, 0.5),      # In wall, different off-axis
            (0, R, 0.3, 0.3),    # In wall
        ]

        negative_found = False
        for point in wall_points:
            coords = np.array(point)
            rho = compute_energy_density(metric_func, coords, h=0.01)

            if rho < -1e-10:
                negative_found = True
                break

        assert negative_found, "No negative energy density found in wall region"

    def test_analytic_energy_density_negative(self):
        """Analytic formula should give negative energy in wall."""
        metric = AlcubierreMetric(v0=1.0, R=1.0, sigma=8.0)

        # Sample off-axis points in wall
        R = metric.params['R']
        test_points = []
        for y in np.linspace(0.1, 0.9, 5):
            test_points.append((0, R, y, 0))

        negative_count = 0
        for point in test_points:
            rho = metric.eulerian_energy_density_analytic(*point)
            if rho < 0:
                negative_count += 1

        # At least some points should have negative energy
        assert negative_count > 0, "Analytic formula gives no negative energy"

    def test_energy_density_zero_at_center(self):
        """Energy density should be approximately zero at bubble center."""
        metric = AlcubierreMetric(v0=1.0, R=1.0, sigma=8.0)

        # At center (x=x_s, y=0, z=0), energy density should be small
        rho = metric.eulerian_energy_density_analytic(0, 0, 0, 0)

        # The formula has y² + z² factor, so at y=z=0 it should be zero
        assert abs(rho) < 1e-10, f"Energy density at center not zero: {rho}"

    def test_energy_density_far_away(self):
        """Energy density should be approximately zero far from bubble."""
        metric = AlcubierreMetric(v0=1.0, R=1.0, sigma=8.0)

        # Far from bubble
        rho = metric.eulerian_energy_density_analytic(0, 10, 0.1, 0)

        # Should be very small (exponentially suppressed)
        assert abs(rho) < 1e-6, f"Energy density too large far from bubble: {rho}"

    def test_energy_density_scales_with_velocity_squared(self):
        """Energy density should scale approximately as v²."""
        R = 1.0
        sigma = 8.0

        metric_v1 = AlcubierreMetric(v0=1.0, R=R, sigma=sigma)
        metric_v2 = AlcubierreMetric(v0=2.0, R=R, sigma=sigma)

        # Test point in wall
        point = (0, R, 0.5, 0)

        rho_v1 = metric_v1.eulerian_energy_density_analytic(*point)
        rho_v2 = metric_v2.eulerian_energy_density_analytic(*point)

        # Should scale as v²
        ratio = rho_v2 / rho_v1 if abs(rho_v1) > 1e-15 else 0
        expected_ratio = 4.0  # (2/1)²

        np.testing.assert_allclose(ratio, expected_ratio, rtol=0.1,
                                   err_msg="Energy doesn't scale as v²")

    def test_energy_distribution_symmetric(self):
        """Energy distribution should be axially symmetric around x-axis."""
        metric = AlcubierreMetric(v0=1.0, R=1.0, sigma=8.0)

        R = metric.params['R']

        # Points at same r but different angles
        r_perp = 0.5
        point1 = (0, R, r_perp, 0)
        point2 = (0, R, 0, r_perp)
        point3 = (0, R, r_perp / np.sqrt(2), r_perp / np.sqrt(2))

        rho1 = metric.eulerian_energy_density_analytic(*point1)
        rho2 = metric.eulerian_energy_density_analytic(*point2)
        rho3 = metric.eulerian_energy_density_analytic(*point3)

        # Should all be equal (axial symmetry)
        np.testing.assert_allclose([rho1, rho2, rho3], [rho1, rho1, rho1], rtol=1e-10,
                                   err_msg="Energy distribution not axially symmetric")


class TestDifferentShapeFunctions:
    """Test energy density properties for different shape functions."""

    @pytest.mark.parametrize("shape", ["tanh", "gaussian", "polynomial", "smoothstep"])
    def test_negative_energy_exists(self, shape):
        """All shape functions should produce negative energy regions."""
        metric = AlcubierreMetric(v0=1.0, R=1.0, sigma=0.5, shape=shape)
        metric_func = metric.get_metric_func()

        # Sample many points
        negative_found = False
        for x in np.linspace(0.5, 1.5, 10):
            for y in np.linspace(0.1, 0.5, 5):
                coords = np.array([0, x, y, 0])
                try:
                    rho = compute_energy_density(metric_func, coords, h=0.02)
                    if rho < -1e-10:
                        negative_found = True
                        break
                except Exception:
                    continue
            if negative_found:
                break

        assert negative_found, f"No negative energy found for shape={shape}"
