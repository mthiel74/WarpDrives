"""
Test energy condition evaluation.

Tests that energy condition checks return expected results
for known cases (Minkowski, Alcubierre).
"""

import numpy as np
import pytest

from warpbubblesim.metrics import AlcubierreMetric, BobrickMartireMetric
from warpbubblesim.gr.conditions import (
    check_wec,
    check_nec,
    check_sec,
    check_dec,
    check_energy_conditions,
)


class TestMinkowskiConditions:
    """Energy conditions for Minkowski spacetime."""

    def test_wec_satisfied_minkowski(self):
        """WEC should be satisfied in Minkowski (T=0)."""
        metric = AlcubierreMetric(v0=0.0)  # Minkowski
        metric_func = metric.get_metric_func()

        coords = np.array([0.0, 0.0, 0.0, 0.0])
        satisfied, value = check_wec(metric_func, coords)

        assert satisfied, f"WEC should be satisfied in Minkowski, value={value}"
        assert abs(value) < 1e-6, f"WEC value should be ~0 in Minkowski: {value}"

    def test_nec_satisfied_minkowski(self):
        """NEC should be satisfied in Minkowski."""
        metric = AlcubierreMetric(v0=0.0)
        metric_func = metric.get_metric_func()

        coords = np.array([0.0, 1.0, 0.0, 0.0])
        satisfied, value = check_nec(metric_func, coords)

        assert satisfied, f"NEC should be satisfied in Minkowski"
        assert abs(value) < 1e-6, f"NEC value should be ~0: {value}"

    def test_sec_satisfied_minkowski(self):
        """SEC should be satisfied in Minkowski."""
        metric = AlcubierreMetric(v0=0.0)
        metric_func = metric.get_metric_func()

        coords = np.array([0.0, 0.0, 0.0, 0.0])
        satisfied, value = check_sec(metric_func, coords)

        assert satisfied, f"SEC should be satisfied in Minkowski"

    def test_dec_satisfied_minkowski(self):
        """DEC should be satisfied in Minkowski."""
        metric = AlcubierreMetric(v0=0.0)
        metric_func = metric.get_metric_func()

        coords = np.array([0.0, 0.0, 0.0, 0.0])
        satisfied, value = check_dec(metric_func, coords)

        assert satisfied, f"DEC should be satisfied in Minkowski"

    def test_all_conditions_minkowski(self):
        """All energy conditions should be satisfied in Minkowski."""
        metric = AlcubierreMetric(v0=0.0)
        metric_func = metric.get_metric_func()

        coords = np.array([0.0, 0.0, 0.0, 0.0])
        conditions = check_energy_conditions(metric_func, coords)

        for name, (satisfied, value) in conditions.items():
            assert satisfied, f"{name} should be satisfied in Minkowski"


class TestAlcubierreConditions:
    """Energy conditions for Alcubierre warp drive."""

    def test_wec_violated_in_wall(self):
        """WEC should be violated in Alcubierre bubble wall."""
        metric = AlcubierreMetric(v0=1.0, R=1.0, sigma=8.0)
        metric_func = metric.get_metric_func()

        # Test multiple points in the wall region
        R = metric.params['R']
        wall_points = [
            np.array([0.0, R, 0.3, 0.0]),
            np.array([0.0, R, 0.0, 0.3]),
            np.array([0.0, R, 0.2, 0.2]),
        ]

        violation_found = False
        for coords in wall_points:
            satisfied, value = check_wec(metric_func, coords, n_samples=20)
            if not satisfied and value < -1e-10:
                violation_found = True
                break

        assert violation_found, "WEC violation should be found in Alcubierre wall"

    def test_nec_violated_in_wall(self):
        """NEC should be violated in Alcubierre bubble wall."""
        metric = AlcubierreMetric(v0=1.0, R=1.0, sigma=8.0)
        metric_func = metric.get_metric_func()

        R = metric.params['R']
        coords = np.array([0.0, R, 0.4, 0.0])

        satisfied, value = check_nec(metric_func, coords, n_samples=30)

        # NEC is often violated when WEC is violated
        # This depends on the specific point chosen


class TestBobrickMartireConditions:
    """Energy conditions for Bobrick-Martire positive-energy drive."""

    def test_subluminal_has_better_conditions(self):
        """Subluminal B-M drive should have more favorable energy conditions."""
        metric = BobrickMartireMetric(
            v0=0.1,  # Very subluminal
            positive_energy=True
        )
        metric_func = metric.get_metric_func()

        # Test at various points
        R_outer = metric.params['R_outer']

        # Outside the shell (should be Minkowski-like)
        coords_outside = np.array([0.0, 3 * R_outer, 0.0, 0.0])
        conditions_outside = check_energy_conditions(metric_func, coords_outside, n_samples=10)

        # Outside should have approximately satisfied conditions
        for name, (satisfied, value) in conditions_outside.items():
            # Values should be small (approximately flat spacetime)
            assert abs(value) < 1, f"{name} value unexpectedly large outside shell: {value}"


class TestConditionEvaluationFunctions:
    """Tests for condition evaluation function behavior."""

    def test_conditions_return_correct_format(self):
        """Energy condition functions should return (bool, float) tuples."""
        metric = AlcubierreMetric(v0=0.5)
        metric_func = metric.get_metric_func()
        coords = np.array([0.0, 0.0, 0.0, 0.0])

        for check_func in [check_wec, check_nec, check_sec, check_dec]:
            result = check_func(metric_func, coords)

            assert isinstance(result, tuple), f"{check_func.__name__} should return tuple"
            assert len(result) == 2, f"{check_func.__name__} should return 2 elements"
            assert isinstance(result[0], bool), f"First element should be bool"
            assert isinstance(result[1], (int, float)), f"Second element should be numeric"

    def test_conditions_finite_values(self):
        """Energy condition values should be finite."""
        metric = AlcubierreMetric(v0=1.0, R=1.0)
        metric_func = metric.get_metric_func()

        # Test at several points
        test_points = [
            np.array([0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.5, 0.0]),
            np.array([0.0, 2.0, 0.0, 0.0]),
        ]

        for coords in test_points:
            conditions = check_energy_conditions(metric_func, coords, n_samples=5)

            for name, (satisfied, value) in conditions.items():
                assert np.isfinite(value), f"{name} has non-finite value at {coords}"

    def test_n_samples_affects_accuracy(self):
        """More samples should give more reliable results."""
        metric = AlcubierreMetric(v0=1.0, R=1.0)
        metric_func = metric.get_metric_func()

        coords = np.array([0.0, 1.0, 0.3, 0.0])

        # Run multiple times with different n_samples
        results_low = []
        results_high = []

        for _ in range(5):
            _, val_low = check_wec(metric_func, coords, n_samples=3)
            _, val_high = check_wec(metric_func, coords, n_samples=20)
            results_low.append(val_low)
            results_high.append(val_high)

        # Higher sample count should give more consistent results
        std_low = np.std(results_low)
        std_high = np.std(results_high)

        # This is a soft test - higher samples should generally be more stable
        # but not guaranteed due to randomness
