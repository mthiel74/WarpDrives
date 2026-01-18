#!/usr/bin/env python3
"""Generate all visualizations for the WarpDrives README."""

import numpy as np
import matplotlib.pyplot as plt
import os

# Create output directory
os.makedirs('images', exist_ok=True)

print("Importing WarpDrives modules...")
from warpbubblesim.metrics import (
    AlcubierreMetric, NatarioMetric, VanDenBroeckMetric,
    WhiteToroidalMetric, BobrickMartireMetric, LentzMetric
)
from warpbubblesim.gr import compute_energy_density, check_energy_conditions
from warpbubblesim.gr.geodesics import integrate_geodesic

plt.rcParams['figure.figsize'] = [10, 8]
plt.rcParams['font.size'] = 12
plt.style.use('dark_background')

# =============================================================================
# 1. Alcubierre Energy Density
# =============================================================================
print("1. Generating Alcubierre energy density plot...")

metric = AlcubierreMetric(v0=1.0, R=1.0, sigma=8.0)
metric_func = metric.get_metric_func()

x = np.linspace(-3, 3, 150)
y = np.linspace(-3, 3, 150)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

for i in range(len(y)):
    for j in range(len(x)):
        Z[i, j] = metric.eulerian_energy_density_analytic(0, x[j], y[i], 0)

fig, ax = plt.subplots(figsize=(10, 8))
vmax = np.max(np.abs(Z))
im = ax.contourf(X, Y, Z, levels=50, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
plt.colorbar(im, ax=ax, label='Energy Density')
ax.set_xlabel('x (direction of travel)')
ax.set_ylabel('y')
ax.set_title('Alcubierre Warp Bubble Energy Density\n(Red = Negative/Exotic Matter Required)')
ax.set_aspect('equal')

# Add bubble outline
theta = np.linspace(0, 2*np.pi, 100)
ax.plot(np.cos(theta), np.sin(theta), 'w--', linewidth=2, alpha=0.7, label='Bubble boundary')
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig('images/alcubierre_energy_density.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: images/alcubierre_energy_density.png")

# =============================================================================
# 2. Shape Function Comparison
# =============================================================================
print("2. Generating shape function comparison...")

fig, ax = plt.subplots(figsize=(10, 6))
r = np.linspace(0, 3, 200)

shapes = ['tanh', 'gaussian', 'polynomial', 'smoothstep']
colors = ['#00ff88', '#ff6b6b', '#4ecdc4', '#ffe66d']

for shape, color in zip(shapes, colors):
    m = AlcubierreMetric(v0=1.0, R=1.0, sigma=8.0, shape=shape)
    f = [m.shape_function(ri) for ri in r]
    ax.plot(r, f, label=shape.capitalize(), linewidth=2.5, color=color)

ax.axhline(y=0.5, color='white', linestyle='--', alpha=0.3)
ax.axvline(x=1.0, color='white', linestyle='--', alpha=0.3, label='R = 1.0')
ax.set_xlabel('Distance from bubble center ($r_s$)')
ax.set_ylabel('Shape function $f(r_s)$')
ax.set_title('Warp Bubble Shape Functions')
ax.legend()
ax.grid(True, alpha=0.2)
ax.set_xlim(0, 3)
ax.set_ylim(-0.05, 1.05)

plt.tight_layout()
plt.savefig('images/shape_functions.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: images/shape_functions.png")

# =============================================================================
# 3. Metric Comparison (Energy Density)
# =============================================================================
print("3. Generating metric comparison...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

metrics_list = [
    (AlcubierreMetric(v0=1.0), 'Alcubierre'),
    (NatarioMetric(v0=1.0), 'Natário'),
    (VanDenBroeckMetric(v0=1.0, R_ext=0.5, R_int=1.0, B_int=3.0), 'Van Den Broeck'),
    (WhiteToroidalMetric(v0=1.0, R_major=1.5, R_minor=0.4), 'White Toroidal'),
    (BobrickMartireMetric(v0=0.3, positive_energy=True), 'Bobrick-Martire'),
    (LentzMetric(v0=0.5), 'Lentz Soliton'),
]

x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)

for ax, (metric, name) in zip(axes, metrics_list):
    metric_func = metric.get_metric_func()
    Z = np.zeros_like(X)

    for i in range(len(y)):
        for j in range(len(x)):
            coords = np.array([0, x[j], y[i], 0])
            Z[i, j] = compute_energy_density(metric_func, coords)

    vmax = np.max(np.abs(Z)) if np.max(np.abs(Z)) > 1e-10 else 1e-10
    im = ax.contourf(X, Y, Z, levels=50, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax)
    ax.set_title(name)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')

plt.suptitle('Energy Density Comparison Across Warp Drive Metrics', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('images/metric_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: images/metric_comparison.png")

# =============================================================================
# 4. Geodesics in Alcubierre Spacetime
# =============================================================================
print("4. Generating geodesics plot...")

metric = AlcubierreMetric(v0=1.0)
metric_func = metric.get_metric_func()

fig, ax = plt.subplots(figsize=(12, 8))

colors = plt.cm.plasma(np.linspace(0.1, 0.9, 11))

for i, x0 in enumerate(np.linspace(-5, 5, 11)):
    initial_coords = np.array([0.0, x0, 0.0, 0.0])
    initial_velocity = np.array([1.0, 0.0, 0.0, 0.0])

    try:
        result = integrate_geodesic(
            metric_func, initial_coords, initial_velocity,
            lambda_span=(0, 15), max_step=0.1
        )
        coords = result['coords']
        ax.plot(coords[:, 1], coords[:, 0], color=colors[i], linewidth=1.5, alpha=0.8)
    except Exception as e:
        print(f"   Warning: geodesic from x0={x0} failed: {e}")

# Bubble trajectory
t_range = np.linspace(0, 15, 100)
x_bubble = [metric.bubble_center(t) for t in t_range]
ax.plot(x_bubble, t_range, 'w--', linewidth=3, label='Warp bubble center')

# Shade bubble region
R = metric.params['R']
ax.fill_betweenx(t_range,
                  [xb - R for xb in x_bubble],
                  [xb + R for xb in x_bubble],
                  alpha=0.15, color='cyan', label='Bubble interior')

ax.set_xlabel('x (space)')
ax.set_ylabel('t (time)')
ax.set_title('Test Particle Geodesics in Alcubierre Warp Bubble Spacetime')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.2)
ax.set_xlim(-8, 20)
ax.set_ylim(0, 15)

plt.tight_layout()
plt.savefig('images/geodesics.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: images/geodesics.png")

# =============================================================================
# 5. Expansion Scalar
# =============================================================================
print("5. Generating expansion scalar plot...")

metric = AlcubierreMetric(v0=1.0)

x = np.linspace(-3, 3, 150)
y = np.linspace(-3, 3, 150)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

for i in range(len(y)):
    for j in range(len(x)):
        Z[i, j] = metric.expansion_scalar_analytic(0, x[j], y[i], 0)

fig, ax = plt.subplots(figsize=(10, 8))
vmax = np.max(np.abs(Z))
im = ax.contourf(X, Y, Z, levels=50, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
plt.colorbar(im, ax=ax, label='Expansion Scalar θ')
ax.set_xlabel('x (direction of travel)')
ax.set_ylabel('y')
ax.set_title('Expansion Scalar: Space Stretching (Red) and Compressing (Blue)')
ax.set_aspect('equal')

# Add annotations
ax.annotate('Expansion\n(behind)', xy=(-1.5, 0), fontsize=12, ha='center', color='white')
ax.annotate('Contraction\n(in front)', xy=(1.5, 0), fontsize=12, ha='center', color='white')

plt.tight_layout()
plt.savefig('images/expansion_scalar.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: images/expansion_scalar.png")

# =============================================================================
# 6. Energy Conditions Visualization
# =============================================================================
print("6. Generating energy conditions plot...")

metric = AlcubierreMetric(v0=1.0)
metric_func = metric.get_metric_func()

# Sample along a radial line
r = np.linspace(0, 3, 50)
y_offset = 0.3  # Off-axis to see energy density

wec_values = []
nec_values = []
energies = []

for ri in r:
    coords = np.array([0, ri, y_offset, 0])
    conditions = check_energy_conditions(metric_func, coords, n_samples=10)
    wec_values.append(conditions['WEC'][1])
    nec_values.append(conditions['NEC'][1])
    energies.append(compute_energy_density(metric_func, coords))

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Energy density
axes[0].plot(r, energies, 'c-', linewidth=2, label='Energy density ρ')
axes[0].axhline(y=0, color='white', linestyle='--', alpha=0.5)
axes[0].fill_between(r, energies, 0, where=[e < 0 for e in energies],
                      alpha=0.3, color='red', label='Exotic matter region')
axes[0].set_ylabel('Energy Density')
axes[0].set_title('Energy Density and Conditions Along Radial Direction (y=0.3)')
axes[0].legend()
axes[0].grid(True, alpha=0.2)

# Energy conditions
axes[1].plot(r, wec_values, 'g-', linewidth=2, label='WEC (T_μν u^μ u^ν)')
axes[1].plot(r, nec_values, 'm-', linewidth=2, label='NEC (T_μν k^μ k^ν)')
axes[1].axhline(y=0, color='white', linestyle='--', alpha=0.5)
axes[1].fill_between(r, wec_values, 0, where=[w < 0 for w in wec_values],
                      alpha=0.3, color='red', label='Condition violated')
axes[1].set_xlabel('Distance from bubble center')
axes[1].set_ylabel('Condition Value')
axes[1].legend()
axes[1].grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('images/energy_conditions.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: images/energy_conditions.png")

# =============================================================================
# 7. Bobrick-Martire Subluminal vs Superluminal
# =============================================================================
print("7. Generating Bobrick-Martire comparison...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)

# Subluminal (positive energy possible)
bm_sub = BobrickMartireMetric(v0=0.2, positive_energy=True)
metric_func = bm_sub.get_metric_func()
Z1 = np.zeros_like(X)
for i in range(len(y)):
    for j in range(len(x)):
        Z1[i, j] = compute_energy_density(metric_func, np.array([0, x[j], y[i], 0]))

vmax = np.max(np.abs(Z1)) if np.max(np.abs(Z1)) > 1e-10 else 1e-10
im = axes[0].contourf(X, Y, Z1, levels=50, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
plt.colorbar(im, ax=axes[0])
axes[0].set_title('Subluminal (v=0.2c)\nPositive Energy Possible!')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_aspect('equal')

# Higher velocity (needs exotic matter)
alc = AlcubierreMetric(v0=1.5)
metric_func = alc.get_metric_func()
Z2 = np.zeros_like(X)
for i in range(len(y)):
    for j in range(len(x)):
        Z2[i, j] = compute_energy_density(metric_func, np.array([0, x[j], y[i], 0]))

vmax = np.max(np.abs(Z2))
im = axes[1].contourf(X, Y, Z2, levels=50, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
plt.colorbar(im, ax=axes[1])
axes[1].set_title('Superluminal (v=1.5c)\nExotic Matter Required')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].set_aspect('equal')

plt.suptitle('Bobrick & Martire (2021): Subluminal Warp Drives Can Use Normal Matter', fontsize=13)
plt.tight_layout()
plt.savefig('images/subluminal_vs_superluminal.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: images/subluminal_vs_superluminal.png")

print("\n✓ All static visualizations generated!")
print("\nGenerating animations...")

# =============================================================================
# 8. Geodesic Animation
# =============================================================================
print("8. Generating geodesic animation...")

import matplotlib.animation as animation

metric = AlcubierreMetric(v0=1.0)
metric_func = metric.get_metric_func()

# Pre-compute geodesics
geodesics = []
for x0 in np.linspace(-4, 4, 9):
    initial_coords = np.array([0.0, x0, 0.0, 0.0])
    initial_velocity = np.array([1.0, 0.0, 0.0, 0.0])
    try:
        result = integrate_geodesic(
            metric_func, initial_coords, initial_velocity,
            lambda_span=(0, 12), max_step=0.05
        )
        geodesics.append(result['coords'])
    except:
        pass

fig, ax = plt.subplots(figsize=(12, 8))

def init():
    ax.clear()
    ax.set_xlim(-6, 18)
    ax.set_ylim(0, 12)
    ax.set_xlabel('x (space)')
    ax.set_ylabel('t (time)')
    ax.set_title('Geodesics in Alcubierre Warp Bubble')
    ax.grid(True, alpha=0.2)
    return []

def animate(frame):
    ax.clear()
    ax.set_xlim(-6, 18)
    ax.set_ylim(0, 12)
    ax.set_xlabel('x (space)')
    ax.set_ylabel('t (time)')
    ax.set_title('Geodesics in Alcubierre Warp Bubble')
    ax.grid(True, alpha=0.2)

    # Draw bubble trajectory up to current time
    t_current = frame * 0.1
    t_range = np.linspace(0, min(t_current, 12), 100)
    x_bubble = [metric.bubble_center(t) for t in t_range]
    ax.plot(x_bubble, t_range, 'w--', linewidth=2, alpha=0.8)

    # Shade bubble at current time
    if t_current <= 12:
        x_center = metric.bubble_center(t_current)
        circle = plt.Circle((x_center, t_current), metric.params['R'], color='cyan', alpha=0.3)
        ax.add_patch(circle)

    # Draw geodesics up to current time
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(geodesics)))
    for geo, color in zip(geodesics, colors):
        mask = geo[:, 0] <= t_current
        if np.any(mask):
            ax.plot(geo[mask, 1], geo[mask, 0], color=color, linewidth=1.5)
            # Current position marker
            idx = np.sum(mask) - 1
            ax.plot(geo[idx, 1], geo[idx, 0], 'o', color=color, markersize=6)

    return []

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=120, interval=50, blit=True)
anim.save('images/geodesics_animation.gif', writer='pillow', fps=20)
plt.close()
print("   Saved: images/geodesics_animation.gif")

# =============================================================================
# 9. Grid Distortion Animation
# =============================================================================
print("9. Generating grid distortion animation...")

metric = AlcubierreMetric(v0=1.0)

fig, ax = plt.subplots(figsize=(10, 8))

def animate_grid(frame):
    ax.clear()
    t = frame * 0.1

    # Create grid
    n_lines = 15
    x_lines = np.linspace(-4, 4, n_lines)
    y_lines = np.linspace(-4, 4, n_lines)

    # Draw distorted vertical lines
    for x_line in x_lines:
        y_points = np.linspace(-4, 4, 50)
        x_distorted = []
        for y in y_points:
            beta = metric.shift(t, x_line, y, 0)
            # Distortion effect (simplified visualization)
            x_dist = x_line + beta[0] * 0.5
            x_distorted.append(x_dist)
        ax.plot(x_distorted, y_points, 'c-', linewidth=0.8, alpha=0.7)

    # Draw distorted horizontal lines
    for y_line in y_lines:
        x_points = np.linspace(-4, 4, 50)
        y_distorted = []
        x_result = []
        for x in x_points:
            beta = metric.shift(t, x, y_line, 0)
            x_dist = x + beta[0] * 0.5
            x_result.append(x_dist)
            y_distorted.append(y_line)
        ax.plot(x_result, y_distorted, 'c-', linewidth=0.8, alpha=0.7)

    # Draw bubble
    x_center = metric.bubble_center(t)
    circle = plt.Circle((x_center, 0), metric.params['R'], fill=False, color='yellow', linewidth=2)
    ax.add_patch(circle)
    ax.plot(x_center, 0, 'yo', markersize=10)

    ax.set_xlim(-4, 8)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Coordinate Grid Distortion by Warp Bubble (t={t:.1f})')
    ax.grid(False)

    return []

anim = animation.FuncAnimation(fig, animate_grid, frames=80, interval=50)
anim.save('images/grid_distortion.gif', writer='pillow', fps=20)
plt.close()
print("   Saved: images/grid_distortion.gif")

# =============================================================================
# 10. Bubble Evolution Animation
# =============================================================================
print("10. Generating bubble evolution animation...")

metric = AlcubierreMetric(v0=1.0)

fig, ax = plt.subplots(figsize=(12, 6))

def animate_bubble(frame):
    ax.clear()
    t = frame * 0.1

    x = np.linspace(-3, 10, 200)
    y = np.linspace(-3, 3, 150)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(len(y)):
        for j in range(len(x)):
            Z[i, j] = metric.eulerian_energy_density_analytic(t, x[j], y[i], 0)

    vmax = 0.015
    im = ax.contourf(X, Y, Z, levels=50, cmap='RdBu_r', vmin=-vmax, vmax=vmax)

    # Bubble center marker
    x_center = metric.bubble_center(t)
    ax.plot(x_center, 0, 'w*', markersize=15)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Warp Bubble Propagation (t={t:.1f}, v=c)')
    ax.set_aspect('equal')

    return []

anim = animation.FuncAnimation(fig, animate_bubble, frames=60, interval=80)
anim.save('images/bubble_evolution.gif', writer='pillow', fps=15)
plt.close()
print("   Saved: images/bubble_evolution.gif")

print("\n" + "="*50)
print("✓ ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
print("="*50)
print("\nStatic images:")
print("  - images/alcubierre_energy_density.png")
print("  - images/shape_functions.png")
print("  - images/metric_comparison.png")
print("  - images/geodesics.png")
print("  - images/expansion_scalar.png")
print("  - images/energy_conditions.png")
print("  - images/subluminal_vs_superluminal.png")
print("\nAnimations:")
print("  - images/geodesics_animation.gif")
print("  - images/grid_distortion.gif")
print("  - images/bubble_evolution.gif")
