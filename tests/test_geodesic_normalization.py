"""
Test geodesic normalization is preserved during integration.

For timelike geodesics: g_{μν} u^μ u^ν = -1
For null geodesics: g_{μν} k^μ k^ν = 0

These constraints should be approximately preserved during integration.
"""

import numpy as np
import pytest

from warpbubblesim.metrics import AlcubierreMetric, NatarioMetric
from warpbubblesim.gr.geodesics import (
    integrate_geodesic,
    integrate_null_geodesic,
    normalize_velocity,
    create_initial_velocity,
)


class TestGeodesicNormalization:
    """Tests for geodesic normalization preservation."""

    def test_timelike_normalization_preserved(self):
        """Timelike geodesic should maintain g_{μν}u^μu^ν ≈ -1."""
        metric = AlcubierreMetric(v0=0.5, R=1.0)
        metric_func = metric.get_metric_func()

        # Initial conditions: particle at rest at origin
        initial_coords = np.array([0.0, 0.0, 0.0, 0.0])
        initial_velocity = np.array([1.0, 0.0, 0.0, 0.0])

        # Normalize
        g0 = metric_func(*initial_coords)
        initial_velocity = normalize_velocity(initial_velocity, g0, timelike=True)

        # Check initial normalization
        norm_sq_initial = np.einsum('mn,m,n->', g0, initial_velocity, initial_velocity)
        np.testing.assert_allclose(norm_sq_initial, -1.0, atol=1e-10,
                                   err_msg="Initial velocity not properly normalized")

        # Integrate geodesic
        result = integrate_geodesic(
            metric_func, initial_coords, initial_velocity,
            (0, 5), rtol=1e-8, atol=1e-10
        )

        # Check normalization drift
        drift = result['normalization_drift']
        assert drift < 1e-4, f"Normalization drift too large: {drift}"

    def test_timelike_normalization_at_each_step(self):
        """Check normalization at intermediate steps."""
        metric = AlcubierreMetric(v0=0.3, R=1.0)
        metric_func = metric.get_metric_func()

        initial_coords = np.array([0.0, 1.0, 0.0, 0.0])
        initial_velocity = np.array([1.0, 0.1, 0.0, 0.0])

        g0 = metric_func(*initial_coords)
        initial_velocity = normalize_velocity(initial_velocity, g0, timelike=True)

        result = integrate_geodesic(
            metric_func, initial_coords, initial_velocity,
            (0, 3), max_step=0.1
        )

        coords = result['coords']
        velocity = result['velocity']

        # Check normalization at each step
        max_deviation = 0
        for i in range(len(coords)):
            g = metric_func(*coords[i])
            norm_sq = np.einsum('mn,m,n->', g, velocity[i], velocity[i])
            deviation = abs(norm_sq + 1)  # Should be close to 0
            max_deviation = max(max_deviation, deviation)

        assert max_deviation < 1e-3, f"Max normalization deviation: {max_deviation}"

    def test_null_geodesic_stays_null(self):
        """Null geodesic should maintain g_{μν}k^μk^ν ≈ 0."""
        metric = AlcubierreMetric(v0=0.5, R=1.0)
        metric_func = metric.get_metric_func()

        # Create null initial velocity
        initial_coords = np.array([0.0, -2.0, 0.0, 0.0])
        direction = np.array([1.0, 0.0, 0.0])  # +x direction

        initial_velocity = create_initial_velocity(
            metric_func, initial_coords, direction, timelike=False
        )

        # Check initial nullity
        g0 = metric_func(*initial_coords)
        norm_sq_initial = np.einsum('mn,m,n->', g0, initial_velocity, initial_velocity)
        assert abs(norm_sq_initial) < 1e-10, f"Initial velocity not null: {norm_sq_initial}"

        # Integrate
        result = integrate_geodesic(
            metric_func, initial_coords, initial_velocity,
            (0, 5), rtol=1e-8
        )

        # Check at final point
        coords = result['coords']
        velocity = result['velocity']

        g_final = metric_func(*coords[-1])
        norm_sq_final = np.einsum('mn,m,n->', g_final, velocity[-1], velocity[-1])

        assert abs(norm_sq_final) < 1e-3, f"Final velocity not null: {norm_sq_final}"


class TestGeodesicIntegration:
    """Tests for geodesic integration correctness."""

    def test_minkowski_straight_lines(self):
        """In Minkowski, geodesics should be straight lines."""
        metric = AlcubierreMetric(v0=0.0)  # Minkowski
        metric_func = metric.get_metric_func()

        # Particle with initial spatial velocity
        initial_coords = np.array([0.0, 0.0, 0.0, 0.0])
        initial_velocity = np.array([1.0, 0.3, 0.2, 0.0])

        # Normalize
        g0 = metric_func(*initial_coords)
        initial_velocity = normalize_velocity(initial_velocity, g0, timelike=True)

        result = integrate_geodesic(
            metric_func, initial_coords, initial_velocity,
            (0, 5)
        )

        coords = result['coords']

        # Check that x(t) is linear
        t = coords[:, 0]
        x = coords[:, 1]

        # Fit line
        coeffs = np.polyfit(t, x, 1)
        x_fit = np.polyval(coeffs, t)

        # Should be very close to linear
        residuals = x - x_fit
        max_residual = np.max(np.abs(residuals))

        assert max_residual < 1e-4, f"Geodesic not straight in Minkowski: max_residual={max_residual}"

    def test_geodesic_enters_bubble(self):
        """Test particle geodesic interacting with warp bubble."""
        metric = AlcubierreMetric(v0=1.0, R=1.0)
        metric_func = metric.get_metric_func()

        # Particle starting outside bubble
        initial_coords = np.array([0.0, 5.0, 0.0, 0.0])
        initial_velocity = np.array([1.0, 0.0, 0.0, 0.0])

        g0 = metric_func(*initial_coords)
        initial_velocity = normalize_velocity(initial_velocity, g0, timelike=True)

        result = integrate_geodesic(
            metric_func, initial_coords, initial_velocity,
            (0, 10)
        )

        # Should complete without error
        assert result['solution'].success, "Geodesic integration failed"

        # Normalization should be preserved
        assert result['normalization_drift'] < 0.01


class TestNullGeodesicBundle:
    """Tests for null geodesic (light ray) behavior."""

    def test_light_cone_structure(self):
        """Light rays should have appropriate causal structure."""
        metric = AlcubierreMetric(v0=0.0)  # Minkowski
        metric_func = metric.get_metric_func()

        origin = np.array([0.0, 0.0, 0.0, 0.0])

        # Right-going ray
        result_right = integrate_null_geodesic(
            metric_func, origin,
            np.array([1.0, 0.0, 0.0]),  # +x direction
            (0, 3)
        )

        # Left-going ray
        result_left = integrate_null_geodesic(
            metric_func, origin,
            np.array([-1.0, 0.0, 0.0]),  # -x direction
            (0, 3)
        )

        # In Minkowski, light travels at 45° in spacetime diagram
        coords_right = result_right['coords']
        coords_left = result_left['coords']

        # dx/dt should be ≈ ±1
        dt_right = coords_right[-1, 0] - coords_right[0, 0]
        dx_right = coords_right[-1, 1] - coords_right[0, 1]

        if dt_right > 0:
            speed_right = dx_right / dt_right
            assert abs(abs(speed_right) - 1) < 0.1, f"Light speed not 1: {speed_right}"
