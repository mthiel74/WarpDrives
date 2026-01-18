"""
Van Den Broeck "pocket" warp drive modification (1999).

Chris Van Den Broeck proposed a modification to reduce the total
negative energy required. The idea is to create a "pocket" of
expanded space inside the bubble while keeping the external
bubble radius very small.

The geometry has:
- A tiny external bubble (microscopic R_ext)
- A large internal volume (macroscopic R_int)
- The negative energy scales with R_ext, not R_int

This is achieved with a conformal factor that varies with position.

The metric has the form:
ds² = -dt² + B(r)² [(dx - v_s f(r) dt)² + dy² + dz²]

where B(r) is a "warp factor" that:
- B ≈ B_int >> 1 for r < R_int (expanded interior)
- B ≈ 1 for r > R_ext (normal exterior)
- Smooth transition in between

Reference:
    Van Den Broeck, C. (1999). "A 'warp drive' with more reasonable
    total energy requirements." Classical and Quantum Gravity, 16(12), 3973.
"""

import numpy as np
from typing import Callable
from warpbubblesim.metrics.base import WarpMetric, tanh_shape


class VanDenBroeckMetric(WarpMetric):
    """
    Van Den Broeck pocket warp drive.

    Combines Alcubierre warp effect with a spatial expansion factor
    to reduce energy requirements.

    Parameters
    ----------
    v0 : float
        Bubble velocity.
    R_ext : float
        External bubble radius (the "small" radius seen from outside).
    R_int : float
        Internal bubble radius (sets the usable interior volume).
    B_int : float
        Internal expansion factor (B >> 1 for large interior).
    sigma : float
        Wall thickness parameter.
    sigma_B : float
        Wall thickness for B transition.
    x0 : float
        Initial position.
    """

    @property
    def name(self) -> str:
        return "Van Den Broeck Pocket Warp Drive"

    @property
    def citation(self) -> str:
        return "VanDenBroeck1999"

    def _default_params(self) -> dict:
        return {
            'v0': 1.0,
            'R_ext': 0.1,    # External radius (small)
            'R_int': 1.0,    # Internal radius (large)
            'B_int': 10.0,   # Expansion factor
            'sigma': 8.0,    # Shape function parameter
            'sigma_B': 5.0,  # B transition steepness
            'x0': 0.0,
        }

    @property
    def R(self) -> float:
        """Use external radius as the characteristic radius."""
        return self.params['R_ext']

    def shape_function(self, r: float) -> float:
        """
        Alcubierre-type shape function f(r) for the shift.

        This uses R_ext as the characteristic radius since the
        shift is what creates the "bubble" motion.
        """
        R = self.params['R_ext']
        sigma = self.params['sigma']
        return tanh_shape(r, R, sigma)

    def expansion_factor(self, r: float) -> float:
        """
        Spatial expansion factor B(r).

        B(r) = 1 + (B_int - 1) * g(r)

        where g(r) transitions from 1 inside to 0 outside.

        Parameters
        ----------
        r : float
            Distance from bubble center.

        Returns
        -------
        float
            Expansion factor B.
        """
        B_int = self.params['B_int']
        R_int = self.params['R_int']
        sigma_B = self.params['sigma_B']

        # g(r) = 1 for r < R_int, 0 for r >> R_int
        g = tanh_shape(r, R_int, sigma_B)

        return 1 + (B_int - 1) * g

    def expansion_factor_derivative(self, r: float, h: float = 1e-6) -> float:
        """Compute dB/dr numerically."""
        return (self.expansion_factor(r + h) - self.expansion_factor(r - h)) / (2 * h)

    def shift(self, t: float, x: float, y: float, z: float) -> np.ndarray:
        """
        Shift vector for Van Den Broeck metric.

        Same form as Alcubierre but with effective velocity
        modified by the expansion factor.
        """
        v_s = self.bubble_velocity(t)
        r_s = self.r_from_center(t, x, y, z)
        f = self.shape_function(r_s)

        # The shift is the same form as Alcubierre
        return np.array([-v_s * f, 0.0, 0.0])

    def spatial_metric(self, t: float, x: float, y: float, z: float) -> np.ndarray:
        """
        Spatial metric with expansion factor.

        γ_{ij} = B(r)² δ_{ij}

        Parameters
        ----------
        t, x, y, z : float
            Spacetime coordinates.

        Returns
        -------
        np.ndarray
            Spatial metric γ_{ij}, shape (3, 3).
        """
        r_s = self.r_from_center(t, x, y, z)
        B = self.expansion_factor(r_s)

        return B**2 * np.eye(3)

    def metric(self, t: float, x: float, y: float, z: float) -> np.ndarray:
        """
        Full 4-metric with pocket geometry.

        ds² = -dt² + B(r)² [(dx - v_s f dt)² + dy² + dz²]

        Parameters
        ----------
        t, x, y, z : float
            Spacetime coordinates.

        Returns
        -------
        np.ndarray
            4-metric g_{μν}.
        """
        v_s = self.bubble_velocity(t)
        r_s = self.r_from_center(t, x, y, z)
        f = self.shape_function(r_s)
        B = self.expansion_factor(r_s)
        B2 = B**2

        g = np.zeros((4, 4))

        # g_{tt} = -1 + B² v² f²
        g[0, 0] = -1 + B2 * v_s**2 * f**2

        # g_{tx} = -B² v_s f
        g[0, 1] = -B2 * v_s * f
        g[1, 0] = -B2 * v_s * f

        # Spatial: γ_{ij} = B² δ_{ij}
        g[1, 1] = B2
        g[2, 2] = B2
        g[3, 3] = B2

        return g

    def internal_volume(self) -> float:
        """
        Estimate the internal proper volume of the pocket.

        For a sphere of coordinate radius R_int with expansion B:
        V_proper = (4/3) π R_int³ B_int³

        Returns
        -------
        float
            Internal proper volume.
        """
        R_int = self.params['R_int']
        B_int = self.params['B_int']

        return (4/3) * np.pi * R_int**3 * B_int**3

    def external_radius(self) -> float:
        """
        The effective radius as seen from outside.

        Returns
        -------
        float
            External radius.
        """
        return self.params['R_ext']

    def energy_reduction_factor(self) -> float:
        """
        Estimate of energy reduction compared to Alcubierre.

        Van Den Broeck showed that energy scales with R_ext
        while useful volume scales with B³R_int³.

        The ratio (R_ext / R_int)² gives rough reduction factor.

        Returns
        -------
        float
            Approximate energy reduction factor.
        """
        R_ext = self.params['R_ext']
        R_int = self.params['R_int']

        return (R_ext / R_int)**2

    def description(self) -> str:
        """Return description of this metric's physics."""
        B_int = self.params['B_int']
        R_ext = self.params['R_ext']
        R_int = self.params['R_int']

        V_int = self.internal_volume()
        V_ext = (4/3) * np.pi * R_ext**3

        return (
            f"Van Den Broeck pocket warp drive:\n"
            f"  External radius: {R_ext:.3f}\n"
            f"  Internal radius: {R_int:.3f}\n"
            f"  Expansion factor: {B_int:.1f}\n"
            f"  External volume: {V_ext:.3f}\n"
            f"  Internal proper volume: {V_int:.1f}\n"
            f"  Volume ratio: {V_int/V_ext:.1f}x\n"
            f"  Energy reduction: ~{self.energy_reduction_factor():.2e}"
        )


class VanDenBroeckSimplified(WarpMetric):
    """
    Simplified Van Den Broeck metric for easier computation.

    Uses a single transition region instead of separate
    internal and external radii.

    [ASSUMPTION] This is a pedagogical simplification.
    """

    @property
    def name(self) -> str:
        return "Van Den Broeck (Simplified)"

    @property
    def citation(self) -> str:
        return "VanDenBroeck1999"

    def _default_params(self) -> dict:
        return {
            'v0': 1.0,
            'R': 1.0,
            'sigma': 8.0,
            'B_max': 5.0,  # Maximum expansion
            'x0': 0.0,
        }

    def shape_function(self, r: float) -> float:
        R = self.params['R']
        sigma = self.params['sigma']
        return tanh_shape(r, R, sigma)

    def expansion_factor(self, r: float) -> float:
        """B(r) = 1 + (B_max - 1) * f(r)."""
        B_max = self.params['B_max']
        f = self.shape_function(r)
        return 1 + (B_max - 1) * f

    def shift(self, t: float, x: float, y: float, z: float) -> np.ndarray:
        v_s = self.bubble_velocity(t)
        r_s = self.r_from_center(t, x, y, z)
        f = self.shape_function(r_s)
        return np.array([-v_s * f, 0.0, 0.0])

    def spatial_metric(self, t: float, x: float, y: float, z: float) -> np.ndarray:
        r_s = self.r_from_center(t, x, y, z)
        B = self.expansion_factor(r_s)
        return B**2 * np.eye(3)
