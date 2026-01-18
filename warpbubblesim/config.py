"""
Global configuration for WarpBubbleSim.

Defines conventions, units, and default parameters used throughout the package.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Any
import numpy as np


class MetricSignature(Enum):
    """Metric signature conventions."""
    MOSTLY_PLUS = (-1, 1, 1, 1)  # (-,+,+,+) - Our default
    MOSTLY_MINUS = (1, -1, -1, -1)  # (+,-,-,-) - Alternative


class DerivativeBackend(Enum):
    """Backend for computing derivatives."""
    JAX = "jax"
    FINITE_DIFFERENCE = "finite_difference"
    SYMPY = "sympy"


@dataclass
class Config:
    """
    Global configuration class.

    Conventions used throughout this package:
    - Metric signature: (-,+,+,+) by default
    - Index ordering: (t,x,y,z) = (0,1,2,3)
    - Units: G = c = 1 (geometric units)
    - Einstein equations: G_{μν} = 8π T_{μν}, so T_{μν} = G_{μν}/(8π)
    """

    # Signature and index conventions
    signature: MetricSignature = MetricSignature.MOSTLY_PLUS
    index_order: tuple[str, ...] = ("t", "x", "y", "z")

    # Units (G=c=1)
    G: float = 1.0
    c: float = 1.0

    # Numerical parameters
    derivative_backend: DerivativeBackend = DerivativeBackend.JAX
    finite_diff_step: float = 1e-6
    geodesic_rtol: float = 1e-8
    geodesic_atol: float = 1e-10

    # Grid defaults
    default_resolution: int = 128
    default_extent: tuple[float, float] = (-10.0, 10.0)

    # Visualization
    colormap: str = "RdBu_r"
    dpi: int = 150
    animation_fps: int = 30

    # Output
    output_format: Literal["png", "pdf", "svg"] = "png"
    video_format: Literal["mp4", "gif"] = "mp4"

    def __post_init__(self):
        """Validate configuration."""
        if self.finite_diff_step <= 0:
            raise ValueError("finite_diff_step must be positive")
        if self.default_resolution < 8:
            raise ValueError("default_resolution must be at least 8")

    @property
    def eta(self) -> np.ndarray:
        """Return the Minkowski metric with current signature."""
        return np.diag(list(self.signature.value))

    @property
    def eight_pi(self) -> float:
        """Return 8π for Einstein equations."""
        return 8.0 * np.pi


# Global default configuration
_default_config = Config()


def get_config() -> Config:
    """Get the current global configuration."""
    return _default_config


def set_config(config: Config) -> None:
    """Set the global configuration."""
    global _default_config
    _default_config = config


def reset_config() -> None:
    """Reset to default configuration."""
    global _default_config
    _default_config = Config()


# Physical constants for unit conversions (SI)
CONSTANTS = {
    "c": 299792458.0,  # m/s
    "G": 6.67430e-11,  # m^3 kg^-1 s^-2
    "M_sun": 1.98892e30,  # kg
    "pc": 3.08567758149137e16,  # m (parsec)
    "ly": 9.4607304725808e15,  # m (light-year)
}
