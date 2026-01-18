"""
Alcubierre warp drive metric (1994).

The original warp drive metric proposed by Miguel Alcubierre.

The metric in ADM form:
ds² = -dt² + (dx - v_s f(r_s) dt)² + dy² + dz²

where:
- v_s(t) = dx_s/dt is the velocity of the bubble center
- r_s = sqrt((x - x_s)² + y² + z²) is distance from bubble center
- f(r_s) is a shape function with f(0) = 1, f(∞) = 0

ADM decomposition:
- Lapse: α = 1
- Shift: β^i = (-v_s f, 0, 0)
- Spatial metric: γ_{ij} = δ_{ij}

Reference:
    Alcubierre, M. (1994). "The warp drive: hyper-fast travel within
    general relativity." Classical and Quantum Gravity, 11(5), L73.
"""

import numpy as np
from typing import Callable, Optional
from warpbubblesim.metrics.base import (
    WarpMetric,
    tanh_shape,
    gaussian_shape,
    compact_polynomial_shape,
    smooth_step_shape,
    get_shape_function,
)


class AlcubierreMetric(WarpMetric):
    """
    Alcubierre warp drive metric.

    Parameters
    ----------
    v0 : float
        Warp bubble velocity in units of c.
    R : float
        Characteristic bubble radius.
    sigma : float
        Wall thickness parameter (for tanh: 1/thickness).
    x0 : float
        Initial bubble center position.
    shape : str
        Shape function type: 'tanh' (default), 'gaussian', 'polynomial', 'smoothstep'.

    Examples
    --------
    >>> metric = AlcubierreMetric(v0=2.0, R=1.0, sigma=8.0)
    >>> g = metric.metric(t=0, x=0, y=0, z=0)  # At bubble center
    >>> print(g[0, 0])  # Should be ~ -1 + v²
    """

    @property
    def name(self) -> str:
        return "Alcubierre Warp Drive"

    @property
    def citation(self) -> str:
        return "Alcubierre1994"

    def _default_params(self) -> dict:
        return {
            'v0': 1.0,      # Bubble velocity (in c)
            'R': 1.0,       # Bubble radius
            'sigma': 8.0,   # Wall steepness (higher = sharper)
            'x0': 0.0,      # Initial center position
            'shape': 'tanh' # Shape function type
        }

    def shape_function(self, r: float) -> float:
        """
        Evaluate the shape function f(r_s).

        The default is Alcubierre's tanh form:
        f(r) = (tanh(σ(r + R)) - tanh(σ(r - R))) / (2 tanh(σR))

        Parameters
        ----------
        r : float
            Distance from bubble center.

        Returns
        -------
        float
            Shape function value in [0, 1].
        """
        R = self.params['R']
        sigma = self.params['sigma']
        shape_type = self.params.get('shape', 'tanh')

        shape_func = get_shape_function(shape_type)
        return shape_func(r, R, sigma)

    def shape_function_derivative(self, r: float, h: float = 1e-6) -> float:
        """
        Compute df/dr numerically.

        Parameters
        ----------
        r : float
            Distance from center.
        h : float
            Step size for finite difference.

        Returns
        -------
        float
            Derivative df/dr.
        """
        return (self.shape_function(r + h) - self.shape_function(r - h)) / (2 * h)

    def shift(self, t: float, x: float, y: float, z: float) -> np.ndarray:
        """
        Shift vector β^i = (-v_s f(r_s), 0, 0).

        The negative sign is for the convention ds² = -dt² + (dx - β^x dt)² + ...
        so that the bubble moves in the +x direction.

        Parameters
        ----------
        t, x, y, z : float
            Spacetime coordinates.

        Returns
        -------
        np.ndarray
            Shift vector [β^x, β^y, β^z].
        """
        v_s = self.bubble_velocity(t)
        r_s = self.r_from_center(t, x, y, z)
        f = self.shape_function(r_s)

        return np.array([-v_s * f, 0.0, 0.0])

    def metric(self, t: float, x: float, y: float, z: float) -> np.ndarray:
        """
        Full 4-metric in the explicit Alcubierre form.

        ds² = -dt² + (dx - v_s f dt)² + dy² + dz²
            = -(1 - v²f²)dt² - 2v_s f dt dx + dx² + dy² + dz²

        Parameters
        ----------
        t, x, y, z : float
            Spacetime coordinates.

        Returns
        -------
        np.ndarray
            Metric tensor g_{μν}, shape (4, 4).
        """
        v_s = self.bubble_velocity(t)
        r_s = self.r_from_center(t, x, y, z)
        f = self.shape_function(r_s)

        g = np.zeros((4, 4))

        # g_{tt} = -(1 - v²f²) = -1 + v²f²
        g[0, 0] = -(1 - v_s**2 * f**2)

        # g_{tx} = g_{xt} = -v_s f
        g[0, 1] = -v_s * f
        g[1, 0] = -v_s * f

        # Spatial metric is flat
        g[1, 1] = 1.0
        g[2, 2] = 1.0
        g[3, 3] = 1.0

        return g

    def volume_integral_weight(self, t: float, x: float, y: float, z: float) -> float:
        """
        Return sqrt(-det(g)) for volume integrals.

        For Alcubierre metric, det(g) = -1, so sqrt(-det(g)) = 1.

        Parameters
        ----------
        t, x, y, z : float
            Spacetime coordinates.

        Returns
        -------
        float
            Volume element weight.
        """
        return 1.0

    def eulerian_energy_density_analytic(self, t: float, x: float, y: float, z: float) -> float:
        """
        Analytic formula for Eulerian observer energy density.

        For the Alcubierre metric with flat spatial sections:
        ρ = -v²/(32π) * (df/dr_s)² * [(y² + z²)/r_s²]

        This is derived from the Einstein tensor and shows:
        - ρ < 0 (negative energy density required)
        - Maximum |ρ| in the bubble wall
        - Zero at bubble center and far away

        Parameters
        ----------
        t, x, y, z : float
            Spacetime coordinates.

        Returns
        -------
        float
            Energy density (negative in wall region).
        """
        v_s = self.bubble_velocity(t)
        r_s = self.r_from_center(t, x, y, z)

        # Avoid singularity at center
        if r_s < 1e-10:
            return 0.0

        df_dr = self.shape_function_derivative(r_s)
        rho_perp_sq = y**2 + z**2  # ρ² = y² + z² in cylindrical coords about x-axis

        # Energy density formula from Alcubierre paper
        rho = -(v_s**2 / (32 * np.pi)) * df_dr**2 * (rho_perp_sq / r_s**2)

        return rho

    def expansion_scalar_analytic(self, t: float, x: float, y: float, z: float) -> float:
        """
        Analytic formula for expansion scalar θ of Eulerian observers.

        For Alcubierre metric:
        θ = v_s * (x - x_s) / r_s * df/dr_s

        This shows:
        - Expansion (θ > 0) behind the bubble
        - Contraction (θ < 0) in front of the bubble
        - Zero at y-z plane through center

        Parameters
        ----------
        t, x, y, z : float
            Spacetime coordinates.

        Returns
        -------
        float
            Expansion scalar.
        """
        v_s = self.bubble_velocity(t)
        x_s = self.bubble_center(t)
        r_s = self.r_from_center(t, x, y, z)

        if r_s < 1e-10:
            return 0.0

        df_dr = self.shape_function_derivative(r_s)

        return v_s * (x - x_s) / r_s * df_dr

    def total_energy_estimate(self, n_points: int = 100) -> float:
        """
        Estimate total negative energy required.

        Integrates the energy density over the bubble volume.

        For Alcubierre with tanh shape:
        E ≈ -v² R / (4G σ)

        In geometric units (G=c=1), this gives E in units of length.

        Parameters
        ----------
        n_points : int
            Number of integration points per dimension.

        Returns
        -------
        float
            Estimated total energy (negative).
        """
        R = self.params['R']
        sigma = self.params['sigma']
        v = self.params['v0']

        # Rough estimate based on analytic form
        # More accurate would be numerical integration
        return -(v**2 * R) / (4 * sigma)


class AlcubierreMetricWithAcceleration(AlcubierreMetric):
    """
    Alcubierre metric with time-dependent velocity.

    Extends the basic Alcubierre metric to allow acceleration
    of the bubble.

    Parameters
    ----------
    v0 : float
        Initial velocity.
    a : float
        Constant acceleration.
    v_max : float
        Maximum velocity (velocity is capped at this).
    """

    def _default_params(self) -> dict:
        params = super()._default_params()
        params.update({
            'a': 0.0,       # Acceleration
            'v_max': 10.0,  # Maximum velocity cap
        })
        return params

    def bubble_velocity(self, t: float) -> float:
        """Velocity with acceleration: v(t) = v0 + a*t, capped at v_max."""
        v0 = self.params['v0']
        a = self.params['a']
        v_max = self.params['v_max']

        v = v0 + a * t
        return min(v, v_max)

    def bubble_center(self, t: float) -> float:
        """Position with acceleration: x_s(t) = x0 + v0*t + 0.5*a*t²."""
        x0 = self.params['x0']
        v0 = self.params['v0']
        a = self.params['a']

        return x0 + v0 * t + 0.5 * a * t**2
