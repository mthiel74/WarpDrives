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
from warpbubblesim.viz.skybackground import (
    make_procedural_starfield,
    make_image_sky,
    make_grid_sky,
)
from warpbubblesim.viz.skyrender import (
    Camera,
    RenderConfig,
    build_orthonormal_tetrad,
    render_sky_view,
    render_velocity_sweep,
    save_frames_as_animation,
    trace_pixel,
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
    "make_procedural_starfield",
    "make_image_sky",
    "make_grid_sky",
    "Camera",
    "RenderConfig",
    "build_orthonormal_tetrad",
    "render_sky_view",
    "render_velocity_sweep",
    "save_frames_as_animation",
    "trace_pixel",
]
