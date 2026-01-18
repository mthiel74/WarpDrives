"""
Base class for warp drive metrics.

All warp metrics share common structure:
- ADM (3+1) decomposition with lapse, shift, and spatial metric
- Methods for computing the full 4-metric
- Shape functions that control the bubble geometry
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Dict, Any, Optional, Type
from dataclasses import dataclass, field


class MetricRegistry:
    """Registry for metric classes."""
    _metrics: Dict[str, Type["WarpMetric"]] = {}

    @classmethod
    def register(cls, name: str, metric_class: Type["WarpMetric"]):
        """Register a metric class."""
        cls._metrics[name.lower()] = metric_class

    @classmethod
    def create(cls, name: str, **kwargs) -> "WarpMetric":
        """Create a metric instance by name."""
        name_lower = name.lower()
        if name_lower not in cls._metrics:
            raise ValueError(f"Unknown metric: {name}. Available: {list(cls._metrics.keys())}")
        return cls._metrics[name_lower](**kwargs)

    @classmethod
    def list_metrics(cls) -> list[str]:
        """List available metric names."""
        return list(cls._metrics.keys())


@dataclass
class WarpMetricParams:
    """
    Base parameters for warp metrics.

    Attributes
    ----------
    v0 : float
        Characteristic velocity of the warp bubble (in units of c).
    R : float
        Characteristic radius of the warp bubble.
    sigma : float
        Wall thickness parameter (smaller = sharper transition).
    x0 : float
        Initial x-position of bubble center.
    """
    v0: float = 1.0
    R: float = 1.0
    sigma: float = 0.5
    x0: float = 0.0


class WarpMetric(ABC):
    """
    Abstract base class for warp drive metrics.

    All warp metrics are expressed in ADM form:
    ds² = -α² dt² + γ_{ij}(dx^i + β^i dt)(dx^j + β^j dt)

    Most warp metrics have:
    - α = 1 (unit lapse)
    - γ_{ij} = δ_{ij} (flat spatial metric)
    - Non-trivial shift vector β^i encoding the "warp"

    The bubble moves along the x-axis with center at x_s(t).
    """

    def __init__(self, **params):
        """
        Initialize the metric with parameters.

        Parameters can be passed as keyword arguments or a params dict.
        """
        self.params = self._default_params()
        self.params.update(params)
        self._validate_params()

    @abstractmethod
    def _default_params(self) -> dict:
        """Return default parameters for this metric."""
        pass

    def _validate_params(self):
        """Validate parameters. Override for specific checks."""
        if self.params.get('R', 1) <= 0:
            raise ValueError("Bubble radius R must be positive")
        if self.params.get('sigma', 1) <= 0:
            raise ValueError("Wall thickness sigma must be positive")

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this metric."""
        pass

    @property
    @abstractmethod
    def citation(self) -> str:
        """BibTeX key or short citation for this metric."""
        pass

    def bubble_center(self, t: float) -> float:
        """
        Position of bubble center at time t.

        Default: constant velocity motion x_s(t) = x0 + v0 * t

        Parameters
        ----------
        t : float
            Coordinate time.

        Returns
        -------
        float
            x-coordinate of bubble center.
        """
        v0 = self.params.get('v0', 1.0)
        x0 = self.params.get('x0', 0.0)
        return x0 + v0 * t

    def bubble_velocity(self, t: float) -> float:
        """
        Velocity of bubble center at time t.

        Default: constant v0.

        Parameters
        ----------
        t : float
            Coordinate time.

        Returns
        -------
        float
            dx_s/dt
        """
        return self.params.get('v0', 1.0)

    @abstractmethod
    def shape_function(self, r: float) -> float:
        """
        Shape function f(r) that defines the bubble geometry.

        f(r) should be:
        - ≈ 1 for r < R (inside bubble)
        - ≈ 0 for r > R (outside bubble)
        - Smooth transition in the wall region

        Parameters
        ----------
        r : float
            Distance from bubble center.

        Returns
        -------
        float
            Value of shape function in [0, 1].
        """
        pass

    def lapse(self, t: float, x: float, y: float, z: float) -> float:
        """
        Lapse function α(t, x, y, z).

        Default: α = 1 (as in original Alcubierre).

        Parameters
        ----------
        t, x, y, z : float
            Spacetime coordinates.

        Returns
        -------
        float
            Lapse function value.
        """
        return 1.0

    @abstractmethod
    def shift(self, t: float, x: float, y: float, z: float) -> np.ndarray:
        """
        Shift vector β^i(t, x, y, z).

        This encodes the "warping" of spacetime.

        Parameters
        ----------
        t, x, y, z : float
            Spacetime coordinates.

        Returns
        -------
        np.ndarray
            Shift vector [β^x, β^y, β^z], shape (3,).
        """
        pass

    def spatial_metric(self, t: float, x: float, y: float, z: float) -> np.ndarray:
        """
        Spatial metric γ_{ij}(t, x, y, z).

        Default: flat spatial metric γ_{ij} = δ_{ij}.

        Parameters
        ----------
        t, x, y, z : float
            Spacetime coordinates.

        Returns
        -------
        np.ndarray
            Spatial metric, shape (3, 3).
        """
        return np.eye(3)

    def metric(self, t: float, x: float, y: float, z: float) -> np.ndarray:
        """
        Full 4-metric g_{μν}(t, x, y, z).

        Constructs the metric from ADM variables:
        g_{00} = -α² + β_i β^i
        g_{0i} = β_i
        g_{ij} = γ_{ij}

        Parameters
        ----------
        t, x, y, z : float
            Spacetime coordinates.

        Returns
        -------
        np.ndarray
            Metric tensor, shape (4, 4).
        """
        alpha = self.lapse(t, x, y, z)
        beta = self.shift(t, x, y, z)
        gamma = self.spatial_metric(t, x, y, z)

        g = np.zeros((4, 4))

        # Lower shift index: β_i = γ_{ij} β^j
        beta_lower = gamma @ beta

        # g_{00} = -α² + β_i β^i
        g[0, 0] = -alpha**2 + np.dot(beta_lower, beta)

        # g_{0i} = β_i
        g[0, 1:] = beta_lower
        g[1:, 0] = beta_lower

        # g_{ij} = γ_{ij}
        g[1:, 1:] = gamma

        return g

    def metric_inverse(self, t: float, x: float, y: float, z: float) -> np.ndarray:
        """
        Inverse metric g^{μν}(t, x, y, z).

        For ADM form:
        g^{00} = -1/α²
        g^{0i} = β^i/α²
        g^{ij} = γ^{ij} - β^i β^j/α²

        Parameters
        ----------
        t, x, y, z : float
            Spacetime coordinates.

        Returns
        -------
        np.ndarray
            Inverse metric, shape (4, 4).
        """
        alpha = self.lapse(t, x, y, z)
        beta = self.shift(t, x, y, z)
        gamma = self.spatial_metric(t, x, y, z)
        gamma_inv = np.linalg.inv(gamma)

        g_inv = np.zeros((4, 4))
        alpha_sq = alpha**2

        g_inv[0, 0] = -1.0 / alpha_sq
        g_inv[0, 1:] = beta / alpha_sq
        g_inv[1:, 0] = beta / alpha_sq
        g_inv[1:, 1:] = gamma_inv - np.outer(beta, beta) / alpha_sq

        return g_inv

    def __call__(self, t: float, x: float, y: float, z: float) -> np.ndarray:
        """Allow metric to be called as a function."""
        return self.metric(t, x, y, z)

    def get_metric_func(self) -> Callable:
        """Return a function (t, x, y, z) -> g_{μν} for use with GR module."""
        return lambda t, x, y, z: self.metric(t, x, y, z)

    def get_shift_func(self) -> Callable:
        """Return a function (t, x, y, z) -> β^i for use with ADM module."""
        return lambda t, x, y, z: self.shift(t, x, y, z)

    def r_from_center(self, t: float, x: float, y: float, z: float) -> float:
        """
        Distance from bubble center.

        Parameters
        ----------
        t, x, y, z : float
            Spacetime coordinates.

        Returns
        -------
        float
            Distance r_s = sqrt((x - x_s)² + y² + z²).
        """
        x_s = self.bubble_center(t)
        return np.sqrt((x - x_s)**2 + y**2 + z**2)

    def info(self) -> dict:
        """Return metadata about this metric."""
        return {
            'name': self.name,
            'citation': self.citation,
            'params': self.params.copy(),
        }


# Shape function implementations that can be shared

def tanh_shape(r: float, R: float, sigma: float) -> float:
    """
    Alcubierre's original tanh shape function.

    f(r) = (tanh(σ(r + R)) - tanh(σ(r - R))) / (2 tanh(σR))

    Parameters
    ----------
    r : float
        Distance from center.
    R : float
        Bubble radius.
    sigma : float
        Wall steepness (1/thickness).

    Returns
    -------
    float
        Shape function value.
    """
    # Avoid numerical issues for large arguments
    arg_plus = sigma * (r + R)
    arg_minus = sigma * (r - R)
    arg_R = sigma * R

    # Clip to avoid overflow
    arg_plus = np.clip(arg_plus, -20, 20)
    arg_minus = np.clip(arg_minus, -20, 20)
    arg_R = np.clip(arg_R, -20, 20)

    numerator = np.tanh(arg_plus) - np.tanh(arg_minus)
    denominator = 2 * np.tanh(arg_R)

    if abs(denominator) < 1e-10:
        return 1.0 if r < R else 0.0

    return numerator / denominator


def gaussian_shape(r: float, R: float, sigma: float) -> float:
    """
    Gaussian shape function.

    f(r) = exp(-(r/R)^2 / (2σ²)) for smooth Gaussian profile.

    Parameters
    ----------
    r : float
        Distance from center.
    R : float
        Characteristic radius.
    sigma : float
        Width parameter.

    Returns
    -------
    float
        Shape function value.
    """
    return np.exp(-0.5 * (r / (R * sigma))**2)


def compact_polynomial_shape(r: float, R: float, sigma: float) -> float:
    """
    Compact-support C² polynomial shape function.

    f(r) = (1 - (r/R_eff)²)³ for r < R_eff, 0 otherwise.
    R_eff = R + σ to control transition region.

    This has compact support (exactly zero outside R_eff).

    Parameters
    ----------
    r : float
        Distance from center.
    R : float
        Nominal radius.
    sigma : float
        Controls effective radius.

    Returns
    -------
    float
        Shape function value.
    """
    R_eff = R + sigma
    if r >= R_eff:
        return 0.0
    x = r / R_eff
    return (1 - x**2)**3


def smooth_step_shape(r: float, R: float, sigma: float) -> float:
    """
    Smooth step (smoothstep) shape function.

    f(r) = 1 - S(r - R + δ, δ) where S is smoothstep over width 2δ.

    Parameters
    ----------
    r : float
        Distance from center.
    R : float
        Bubble radius.
    sigma : float
        Half-width of transition.

    Returns
    -------
    float
        Shape function value.
    """
    delta = sigma
    t = (r - R + delta) / (2 * delta)
    t = np.clip(t, 0, 1)
    # Smoothstep: 3t² - 2t³
    return 1 - (3 * t**2 - 2 * t**3)


# Dictionary of available shape functions
SHAPE_FUNCTIONS = {
    'tanh': tanh_shape,
    'gaussian': gaussian_shape,
    'polynomial': compact_polynomial_shape,
    'smoothstep': smooth_step_shape,
}


def get_shape_function(name: str) -> Callable:
    """Get a shape function by name."""
    if name not in SHAPE_FUNCTIONS:
        raise ValueError(f"Unknown shape function: {name}. Available: {list(SHAPE_FUNCTIONS.keys())}")
    return SHAPE_FUNCTIONS[name]
