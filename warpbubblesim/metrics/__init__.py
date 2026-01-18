"""
Warp drive metric implementations.

This package provides implementations of various warp drive spacetimes:
- Alcubierre (1994): Classic warp bubble with shift vector construction
- Natário (2002): Divergence-free shift vector (expansion-free)
- Van Den Broeck (1999): Pocket modification for energy reduction
- White toroidal: Heuristic toroidal energy distribution
- Bobrick & Martire (2021): Physical warp drives classification
- Lentz (2021): Soliton warp drive

All metrics inherit from WarpMetric base class and provide:
- metric tensor g_{μν}(t, x, y, z)
- ADM decomposition (lapse, shift, spatial metric)
- Optional analytic curvature expressions
"""

from warpbubblesim.metrics.base import WarpMetric, MetricRegistry
from warpbubblesim.metrics.alcubierre import AlcubierreMetric
from warpbubblesim.metrics.natario import NatarioMetric
from warpbubblesim.metrics.vdbroek import VanDenBroeckMetric
from warpbubblesim.metrics.white_toroidal import WhiteToroidalMetric
from warpbubblesim.metrics.bobrick_martire import BobrickMartireMetric
from warpbubblesim.metrics.lentz import LentzMetric

# Register all metrics
MetricRegistry.register("alcubierre", AlcubierreMetric)
MetricRegistry.register("natario", NatarioMetric)
MetricRegistry.register("vdbroek", VanDenBroeckMetric)
MetricRegistry.register("white_toroidal", WhiteToroidalMetric)
MetricRegistry.register("bobrick_martire", BobrickMartireMetric)
MetricRegistry.register("lentz", LentzMetric)

__all__ = [
    "WarpMetric",
    "MetricRegistry",
    "AlcubierreMetric",
    "NatarioMetric",
    "VanDenBroeckMetric",
    "WhiteToroidalMetric",
    "BobrickMartireMetric",
    "LentzMetric",
]


def get_metric(name: str, **kwargs):
    """
    Get a metric instance by name.

    Parameters
    ----------
    name : str
        Metric name (e.g., 'alcubierre', 'natario').
    **kwargs
        Parameters for the metric.

    Returns
    -------
    WarpMetric
        Configured metric instance.
    """
    return MetricRegistry.create(name, **kwargs)


def list_metrics():
    """List all available metrics."""
    return MetricRegistry.list_metrics()
