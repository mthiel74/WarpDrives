"""
Visualization modules for WarpBubbleSim.

Provides tools for:
- 2D field visualizations (heatmaps, contours)
- 3D isosurfaces and volume rendering
- Spacetime diagrams
- Animations of time evolution
"""

from warpbubblesim.viz.fields2d import (
    plot_field_2d,
    plot_energy_density,
    plot_expansion_scalar,
    plot_metric_component,
    plot_shape_function,
    plot_multiple_fields,
)
from warpbubblesim.viz.fields3d import (
    plot_isosurface,
    plot_energy_density_3d,
    plot_streamlines_3d,
)
from warpbubblesim.viz.spacetime_diagrams import (
    plot_spacetime_diagram,
    plot_geodesics,
    plot_light_cones,
    plot_worldlines,
)
from warpbubblesim.viz.animations import (
    animate_field_evolution,
    animate_geodesics,
    animate_grid_distortion,
    save_animation,
)

__all__ = [
    "plot_field_2d",
    "plot_energy_density",
    "plot_expansion_scalar",
    "plot_metric_component",
    "plot_shape_function",
    "plot_multiple_fields",
    "plot_isosurface",
    "plot_energy_density_3d",
    "plot_streamlines_3d",
    "plot_spacetime_diagram",
    "plot_geodesics",
    "plot_light_cones",
    "plot_worldlines",
    "animate_field_evolution",
    "animate_geodesics",
    "animate_grid_distortion",
    "save_animation",
]
