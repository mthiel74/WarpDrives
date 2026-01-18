"""
WarpBubbleSim - GR Warp Bubble Spacetime Simulator

A Python package for simulating and visualizing general relativistic
warp bubble spacetimes, including Alcubierre, Natário, Van Den Broeck,
White toroidal, Bobrick-Martire, and Lentz soliton configurations.

Conventions:
- Metric signature: (-,+,+,+)
- Index ordering: (t,x,y,z) = (0,1,2,3)
- Units: G = c = 1 (geometric units)
"""

__version__ = "0.1.0"
__author__ = "WarpBubbleSim Contributors"

from warpbubblesim.config import Config
from warpbubblesim.gr.tensors import (
    compute_christoffel,
    compute_riemann,
    compute_ricci,
    compute_ricci_scalar,
    compute_einstein,
)
from warpbubblesim.gr.invariants import compute_kretschmann
from warpbubblesim.gr.energy import compute_stress_energy
from warpbubblesim.gr.conditions import check_energy_conditions
from warpbubblesim.gr.geodesics import integrate_geodesic, integrate_null_geodesic

__all__ = [
    "Config",
    "compute_christoffel",
    "compute_riemann",
    "compute_ricci",
    "compute_ricci_scalar",
    "compute_einstein",
    "compute_kretschmann",
    "compute_stress_energy",
    "check_energy_conditions",
    "integrate_geodesic",
    "integrate_null_geodesic",
]
