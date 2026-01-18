# Visualization Guide

This document describes the visualization capabilities of WarpBubbleSim.

## 2D Field Plots

### Basic Field Plotting

```python
from warpbubblesim.metrics import AlcubierreMetric
from warpbubblesim.viz.fields2d import plot_field_2d

metric = AlcubierreMetric(v0=1.0)

# Plot any scalar field
def my_field(t, x, y, z):
    return metric.shape_function(metric.r_from_center(t, x, y, z))

fig, ax = plot_field_2d(
    my_field,
    x_range=(-5, 5),
    y_range=(-5, 5),
    nx=128, ny=128,
    t=0.0, z=0.0,
    title="Shape Function",
    cmap="viridis"
)
```

### Energy Density

```python
from warpbubblesim.viz.fields2d import plot_energy_density

fig, ax = plot_energy_density(
    metric,
    x_range=(-5, 5),
    y_range=(-5, 5),
    nx=128, ny=128,
    t=0.0
)
```

The colormap is symmetric around zero, with:
- Red: negative energy density (exotic matter)
- Blue: positive energy density
- White: zero

### Expansion Scalar

```python
from warpbubblesim.viz.fields2d import plot_expansion_scalar

fig, ax = plot_expansion_scalar(metric, t=0.0)
```

Shows:
- Red: expansion (space stretching) - behind bubble
- Blue: contraction (space compressing) - in front of bubble

### Combined Plots

```python
from warpbubblesim.viz.fields2d import plot_multiple_fields

fig = plot_multiple_fields(
    metric,
    fields=['energy_density', 'expansion', 'shape'],
    x_range=(-5, 5), y_range=(-5, 5),
    nx=64, ny=64,
    save_path='combined_fields.png'
)
```

### Grid Distortion

```python
from warpbubblesim.viz.fields2d import plot_grid_distortion

fig, ax = plot_grid_distortion(
    metric,
    x_range=(-5, 5),
    y_range=(-5, 5),
    n_lines=20,
    t=0.0
)
```

Shows coordinate grid overlaid on shape function.

### Shift Vector Field

```python
from warpbubblesim.viz.fields2d import plot_shift_vector_field

fig, ax = plot_shift_vector_field(
    metric,
    x_range=(-5, 5),
    y_range=(-5, 5),
    nx=20, ny=20
)
```

## 3D Visualizations

Requires PyVista: `pip install pyvista`

### Isosurfaces

```python
from warpbubblesim.viz.fields3d import plot_isosurface

# Energy density isosurfaces
plot_isosurface(
    metric,
    field='energy_density',
    nx=50, ny=50, nz=50,
    isovalues=[-0.01, 0.01],
    opacity=0.7,
    save_path='isosurface.png'
)
```

### Comprehensive 3D View

```python
from warpbubblesim.viz.fields3d import create_warp_visualization

create_warp_visualization(
    metric,
    show_bubble=True,
    show_energy=True,
    show_streamlines=False,
    save_path='warp_3d.png'
)
```

## Spacetime Diagrams

### Basic Diagram

```python
from warpbubblesim.viz.spacetime_diagrams import plot_spacetime_diagram

fig, ax = plot_spacetime_diagram(
    metric,
    t_range=(0, 10),
    x_range=(-10, 10),
    show_bubble_center=True,
    show_null_cones=True,
    n_cones=5
)
```

### Geodesics

```python
from warpbubblesim.viz.spacetime_diagrams import plot_geodesics
from warpbubblesim.gr.geodesics import integrate_geodesic

# Integrate some geodesics
results = []
for x0 in [-3, 0, 3]:
    coords = np.array([0, x0, 0, 0])
    u = np.array([1, 0, 0, 0])
    result = integrate_geodesic(metric.get_metric_func(), coords, u, (0, 10))
    results.append(result)

fig, ax = plot_geodesics(
    results,
    t_range=(0, 10),
    x_range=(-10, 10)
)
```

### Light Cones

```python
from warpbubblesim.viz.spacetime_diagrams import plot_light_cones

# Draw light cones at several spacetime points
points = [(0, 0), (2, 1), (4, 2), (6, 3)]

fig, ax = plot_light_cones(
    metric,
    points,
    cone_size=1.0,
    t_range=(0, 10),
    x_range=(-5, 10)
)
```

### Worldlines

```python
from warpbubblesim.viz.spacetime_diagrams import plot_worldlines

# Initial positions of test particles
initial_positions = [(-3, 0, 0), (-1, 0, 0), (1, 0, 0), (3, 0, 0)]

fig, ax = plot_worldlines(
    metric,
    initial_positions,
    t_range=(0, 10),
    velocity_type='static'  # or 'comoving'
)
```

## Animations

### Field Evolution

```python
from warpbubblesim.viz.animations import animate_field_evolution

anim = animate_field_evolution(
    metric,
    field='shape',  # or 'energy_density', 'expansion'
    x_range=(-5, 5),
    y_range=(-5, 5),
    t_range=(0, 10),
    nx=64, ny=64,
    n_frames=100,
    fps=30,
    save_path='field_evolution.mp4'
)
```

### Geodesic Animation

```python
from warpbubblesim.viz.animations import animate_geodesics

initial_positions = [(x, 0, 0) for x in np.linspace(-3, 3, 10)]

anim = animate_geodesics(
    metric,
    initial_positions,
    t_range=(0, 10),
    n_frames=100,
    fps=30,
    save_path='geodesics.mp4'
)
```

### Grid Distortion Animation

```python
from warpbubblesim.viz.animations import animate_grid_distortion

anim = animate_grid_distortion(
    metric,
    x_range=(-5, 5),
    y_range=(-5, 5),
    t_range=(0, 10),
    n_lines=15,
    n_frames=100,
    save_path='grid_distortion.gif'
)
```

### Metric Comparison

```python
from warpbubblesim.viz.animations import create_comparison_animation
from warpbubblesim.metrics import AlcubierreMetric, NatarioMetric

metrics = [
    AlcubierreMetric(v0=1.0),
    NatarioMetric(v0=1.0)
]

anim = create_comparison_animation(
    metrics,
    field='shape',
    t_range=(0, 10),
    save_path='comparison.gif'
)
```

## Saving Figures

### Static Figures

```python
import matplotlib.pyplot as plt

fig, ax = plot_energy_density(metric)

# PNG (default)
fig.savefig('energy.png', dpi=150, bbox_inches='tight')

# PDF (vector)
fig.savefig('energy.pdf', bbox_inches='tight')

# SVG (vector, web-friendly)
fig.savefig('energy.svg', bbox_inches='tight')

plt.close(fig)  # Free memory
```

### Animations

```python
from warpbubblesim.viz.animations import save_animation

# MP4 (requires ffmpeg)
save_animation(anim, 'output.mp4', fps=30, dpi=150)

# GIF (uses pillow)
save_animation(anim, 'output.gif', fps=15, dpi=100)
```

## Customization

### Color Maps

Recommended colormaps:
- `'RdBu_r'`: Symmetric fields (energy density)
- `'viridis'`: Positive-only fields
- `'coolwarm'`: Symmetric with good contrast
- `'Greys'`: Background/overlay

### Figure Sizes

```python
# Standard single plot
figsize=(8, 6)

# Spacetime diagram
figsize=(10, 8)

# Multi-panel
figsize=(15, 4)  # 3 horizontal panels

# 3D visualization (PyVista)
# Set window size in plotter
plotter = pv.Plotter(window_size=[1200, 900])
```

### Matplotlib Style

```python
import matplotlib.pyplot as plt

# Use a clean style
plt.style.use('seaborn-whitegrid')

# Or customize
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'lines.linewidth': 1.5,
})
```
