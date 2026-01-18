"""
General Relativity computational modules for WarpBubbleSim.

This package provides tools for computing:
- Metric tensors and their inverses
- Christoffel symbols
- Riemann curvature tensor
- Ricci tensor and scalar
- Einstein tensor
- Curvature invariants
- Stress-energy tensor
- Energy conditions
- Geodesics
"""

from warpbubblesim.gr.signature import (
    get_signature,
    validate_metric_signature,
    raise_index,
    lower_index,
)
from warpbubblesim.gr.tensors import (
    compute_metric_inverse,
    compute_christoffel,
    compute_riemann,
    compute_ricci,
    compute_ricci_scalar,
    compute_einstein,
)
from warpbubblesim.gr.invariants import (
    compute_kretschmann,
    compute_chern_pontryagin,
    compute_weyl_squared,
)
from warpbubblesim.gr.energy import (
    compute_stress_energy,
    compute_energy_density,
    compute_pressure,
)
from warpbubblesim.gr.conditions import (
    check_energy_conditions,
    check_wec,
    check_nec,
    check_sec,
    check_dec,
)
from warpbubblesim.gr.adm import (
    adm_to_4metric,
    metric_to_adm,
    compute_extrinsic_curvature,
    compute_expansion_scalar,
)
from warpbubblesim.gr.geodesics import (
    integrate_geodesic,
    integrate_null_geodesic,
    geodesic_rhs,
)
from warpbubblesim.gr.raybundle import (
    create_ray_bundle,
    trace_ray_bundle,
)

__all__ = [
    "get_signature",
    "validate_metric_signature",
    "raise_index",
    "lower_index",
    "compute_metric_inverse",
    "compute_christoffel",
    "compute_riemann",
    "compute_ricci",
    "compute_ricci_scalar",
    "compute_einstein",
    "compute_kretschmann",
    "compute_chern_pontryagin",
    "compute_weyl_squared",
    "compute_stress_energy",
    "compute_energy_density",
    "compute_pressure",
    "check_energy_conditions",
    "check_wec",
    "check_nec",
    "check_sec",
    "check_dec",
    "adm_to_4metric",
    "metric_to_adm",
    "compute_extrinsic_curvature",
    "compute_expansion_scalar",
    "integrate_geodesic",
    "integrate_null_geodesic",
    "geodesic_rhs",
    "create_ray_bundle",
    "trace_ray_bundle",
]
