"""
White toroidal warp drive modification (heuristic).

Harold "Sonny" White proposed modifications to the Alcubierre metric
using a toroidal (donut-shaped) energy distribution instead of the
spherical shell. The motivation was to reduce total energy requirements.

[ASSUMPTION] The exact published metric form is not available in a
standard paper format. This implementation follows the general idea
described in White's NASA talks and papers. The specific parameterization
is a reasonable interpretation but may differ from internal NASA work.

The modification replaces the spherical r_s with a toroidal distance:
- Uses cylindrical coordinates (x, ρ, φ) where ρ = √(y² + z²)
- The torus has major radius R_major (distance from x-axis to tube center)
- The torus has minor radius R_minor (tube radius)
- Distance to torus surface: d = √((ρ - R_major)² + (x - x_s)²) - R_minor

Reference:
    White, H. (2011). "Warp Field Mechanics 101." AIAA Paper.
    White, H. (2013). "Warp Field Mechanics 102." AIAA Paper.

Note: These AIAA papers discuss the toroidal concept but may not
provide complete metric specifications.
"""

import numpy as np
from typing import Callable
from warpbubblesim.metrics.base import WarpMetric, tanh_shape


class WhiteToroidalMetric(WarpMetric):
    """
    White toroidal warp drive modification.

    [ASSUMPTION] This implementation creates a toroidal energy
    distribution by modifying the shape function to depend on
    distance from a torus rather than from a point.

    The torus is centered on the bubble center, oriented with
    the tube axis along the direction of motion (x-axis).

    Parameters
    ----------
    v0 : float
        Bubble velocity.
    R_major : float
        Major radius of torus (from x-axis to tube center).
    R_minor : float
        Minor radius of torus (tube radius).
    sigma : float
        Wall thickness parameter.
    x0 : float
        Initial position.
    """

    @property
    def name(self) -> str:
        return "White Toroidal Warp Drive [HEURISTIC]"

    @property
    def citation(self) -> str:
        return "White2011_AIAA"

    def _default_params(self) -> dict:
        return {
            'v0': 1.0,
            'R_major': 2.0,    # Major radius
            'R_minor': 0.5,    # Minor radius (tube)
            'sigma': 8.0,      # Transition sharpness
            'x0': 0.0,
        }

    @property
    def R(self) -> float:
        """Characteristic radius for base class compatibility."""
        return self.params['R_major']

    def toroidal_distance(self, t: float, x: float, y: float, z: float) -> float:
        """
        Distance from the torus surface.

        For a torus with:
        - Major radius R_major (distance from axis to tube center)
        - Minor radius R_minor (tube radius)
        - Oriented with axis along x direction
        - Centered at (x_s, 0, 0)

        Distance to surface = |d| where:
        d = √((ρ - R_major)² + (x - x_s)²) - R_minor

        ρ = √(y² + z²) is the cylindrical radius

        Parameters
        ----------
        t, x, y, z : float
            Spacetime coordinates.

        Returns
        -------
        float
            Signed distance from torus surface (negative inside tube).
        """
        x_s = self.bubble_center(t)
        R_major = self.params['R_major']
        R_minor = self.params['R_minor']

        # Cylindrical radius
        rho = np.sqrt(y**2 + z**2)

        # Distance from torus tube center (a circle of radius R_major)
        d_from_tube_center = np.sqrt((rho - R_major)**2 + (x - x_s)**2)

        # Distance from torus surface
        return d_from_tube_center - R_minor

    def shape_function(self, r: float) -> float:
        """
        Shape function based on toroidal distance.

        [ASSUMPTION] We use a tanh-based transition around the torus.

        Note: r here is the toroidal distance, not spherical r_s.
        """
        R_minor = self.params['R_minor']
        sigma = self.params['sigma']

        # f = 1 inside torus, 0 outside
        # Transition happens around d = 0
        # Use tanh to smoothly transition
        return 0.5 * (1 - np.tanh(sigma * r / R_minor))

    def shape_function_from_coords(self, t: float, x: float, y: float, z: float) -> float:
        """
        Evaluate shape function at given coordinates.

        Parameters
        ----------
        t, x, y, z : float
            Spacetime coordinates.

        Returns
        -------
        float
            Shape function value.
        """
        d = self.toroidal_distance(t, x, y, z)
        return self.shape_function(d)

    def shift(self, t: float, x: float, y: float, z: float) -> np.ndarray:
        """
        Shift vector for toroidal warp drive.

        [ASSUMPTION] Same functional form as Alcubierre but with
        the toroidal shape function.

        β^i = (-v_s f_torus, 0, 0)

        Parameters
        ----------
        t, x, y, z : float
            Spacetime coordinates.

        Returns
        -------
        np.ndarray
            Shift vector.
        """
        v_s = self.bubble_velocity(t)
        f = self.shape_function_from_coords(t, x, y, z)

        return np.array([-v_s * f, 0.0, 0.0])

    def metric(self, t: float, x: float, y: float, z: float) -> np.ndarray:
        """
        Full 4-metric with toroidal shape.

        Same ADM form as Alcubierre:
        ds² = -dt² + (dx - β^x dt)² + dy² + dz²

        but with toroidally-distributed β^x.
        """
        v_s = self.bubble_velocity(t)
        f = self.shape_function_from_coords(t, x, y, z)

        g = np.zeros((4, 4))
        g[0, 0] = -(1 - v_s**2 * f**2)
        g[0, 1] = -v_s * f
        g[1, 0] = -v_s * f
        g[1, 1] = 1.0
        g[2, 2] = 1.0
        g[3, 3] = 1.0

        return g

    def is_inside_torus(self, t: float, x: float, y: float, z: float) -> bool:
        """Check if point is inside the torus tube."""
        return self.toroidal_distance(t, x, y, z) < 0

    def torus_parameters(self) -> dict:
        """Return torus geometry parameters."""
        R_major = self.params['R_major']
        R_minor = self.params['R_minor']

        # Torus volume
        V_torus = 2 * np.pi**2 * R_major * R_minor**2

        # Torus surface area
        A_torus = 4 * np.pi**2 * R_major * R_minor

        return {
            'R_major': R_major,
            'R_minor': R_minor,
            'volume': V_torus,
            'surface_area': A_torus,
            'aspect_ratio': R_major / R_minor,
        }

    def description(self) -> str:
        """Description of this metric."""
        params = self.torus_parameters()
        return (
            f"[ASSUMPTION] White Toroidal Warp Drive (Heuristic Implementation)\n"
            f"This is an interpretation based on White's AIAA papers.\n"
            f"The exact published form may differ.\n\n"
            f"Torus parameters:\n"
            f"  Major radius: {params['R_major']:.2f}\n"
            f"  Minor radius: {params['R_minor']:.2f}\n"
            f"  Aspect ratio: {params['aspect_ratio']:.2f}\n"
            f"  Tube volume: {params['volume']:.3f}\n"
        )


class WhiteOscillatingMetric(WarpMetric):
    """
    White's oscillating bubble modification.

    [ASSUMPTION] White proposed that oscillating the bubble thickness
    might reduce energy requirements. This implements a simple
    oscillation of the wall thickness parameter.

    σ(t) = σ_0 (1 + A sin(ωt))

    This is speculative and for exploration only.
    """

    @property
    def name(self) -> str:
        return "White Oscillating Warp Drive [SPECULATIVE]"

    @property
    def citation(self) -> str:
        return "White2011_AIAA"

    def _default_params(self) -> dict:
        return {
            'v0': 1.0,
            'R': 1.0,
            'sigma_0': 8.0,       # Base wall parameter
            'amplitude': 0.3,     # Oscillation amplitude (fraction)
            'omega': 1.0,         # Angular frequency
            'x0': 0.0,
        }

    def sigma_effective(self, t: float) -> float:
        """Time-dependent wall thickness parameter."""
        sigma_0 = self.params['sigma_0']
        A = self.params['amplitude']
        omega = self.params['omega']

        return sigma_0 * (1 + A * np.sin(omega * t))

    def shape_function(self, r: float) -> float:
        """Shape function with default sigma (for API compatibility)."""
        R = self.params['R']
        sigma = self.params['sigma_0']
        return tanh_shape(r, R, sigma)

    def shape_function_t(self, r: float, t: float) -> float:
        """Time-dependent shape function."""
        R = self.params['R']
        sigma = self.sigma_effective(t)
        return tanh_shape(r, R, sigma)

    def shift(self, t: float, x: float, y: float, z: float) -> np.ndarray:
        v_s = self.bubble_velocity(t)
        r_s = self.r_from_center(t, x, y, z)
        f = self.shape_function_t(r_s, t)
        return np.array([-v_s * f, 0.0, 0.0])
