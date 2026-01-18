#!/usr/bin/env python3
"""
Generate an impressive 3D warp drive visualization for social media.

Features:
- 3D grid distortion showing spacetime warping
- Warp bubble energy density overlay with transparency
- Stylized spacecraft inside the bubble
- Smooth camera movement
- High-quality rendering
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import os

# Create output directory
os.makedirs('images', exist_ok=True)

print("Importing WarpDrives modules...")
from warpbubblesim.metrics import AlcubierreMetric

# Set up high-quality rendering
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10

# =============================================================================
# Create custom colormaps
# =============================================================================

# Warp bubble colormap: cyan/blue glow
warp_colors = [
    (0.0, 0.0, 0.0, 0.0),      # Transparent
    (0.0, 0.8, 1.0, 0.2),      # Light cyan
    (0.0, 0.5, 1.0, 0.5),      # Cyan-blue
    (0.2, 0.2, 1.0, 0.7),      # Blue
    (0.5, 0.0, 1.0, 0.9),      # Purple
]
warp_cmap = LinearSegmentedColormap.from_list('warp', warp_colors)

# Energy density colormap: red for exotic matter
energy_colors = [
    (0.0, 0.0, 0.0, 0.0),      # Transparent (zero)
    (1.0, 0.3, 0.0, 0.3),      # Orange
    (1.0, 0.0, 0.0, 0.6),      # Red
    (1.0, 0.0, 0.5, 0.8),      # Magenta
]
energy_cmap = LinearSegmentedColormap.from_list('energy', energy_colors)


def create_spacecraft():
    """
    Create a stylized spacecraft mesh.
    Returns vertices and faces for a sleek spacecraft shape.
    """
    # Main hull - elongated diamond/shuttle shape
    hull_length = 0.6
    hull_width = 0.15
    hull_height = 0.08

    # Vertices for main hull (elongated octahedron-like shape)
    vertices = [
        # Nose
        [hull_length * 0.6, 0, 0],                    # 0: nose tip
        # Front section
        [hull_length * 0.2, hull_width * 0.3, hull_height * 0.5],   # 1
        [hull_length * 0.2, -hull_width * 0.3, hull_height * 0.5],  # 2
        [hull_length * 0.2, -hull_width * 0.3, -hull_height * 0.5], # 3
        [hull_length * 0.2, hull_width * 0.3, -hull_height * 0.5],  # 4
        # Mid section (widest)
        [0, hull_width * 0.5, hull_height * 0.7],     # 5
        [0, -hull_width * 0.5, hull_height * 0.7],    # 6
        [0, -hull_width * 0.5, -hull_height * 0.5],   # 7
        [0, hull_width * 0.5, -hull_height * 0.5],    # 8
        # Rear section
        [-hull_length * 0.3, hull_width * 0.4, hull_height * 0.5],  # 9
        [-hull_length * 0.3, -hull_width * 0.4, hull_height * 0.5], # 10
        [-hull_length * 0.3, -hull_width * 0.4, -hull_height * 0.4],# 11
        [-hull_length * 0.3, hull_width * 0.4, -hull_height * 0.4], # 12
        # Engine section
        [-hull_length * 0.5, hull_width * 0.25, hull_height * 0.3], # 13
        [-hull_length * 0.5, -hull_width * 0.25, hull_height * 0.3],# 14
        [-hull_length * 0.5, -hull_width * 0.25, -hull_height * 0.2],# 15
        [-hull_length * 0.5, hull_width * 0.25, -hull_height * 0.2],# 16
        # Nacelle tips (warp engines)
        [-hull_length * 0.4, hull_width * 0.8, 0],    # 17: left nacelle
        [-hull_length * 0.4, -hull_width * 0.8, 0],   # 18: right nacelle
    ]

    vertices = np.array(vertices)

    # Faces (triangles)
    faces = [
        # Nose cone
        [0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1],
        # Front to mid (top)
        [1, 5, 6], [1, 6, 2],
        # Front to mid (bottom)
        [3, 7, 8], [3, 8, 4],
        # Front to mid (sides)
        [2, 6, 7], [2, 7, 3],
        [4, 8, 5], [4, 5, 1],
        # Mid to rear (top)
        [5, 9, 10], [5, 10, 6],
        # Mid to rear (bottom)
        [7, 11, 12], [7, 12, 8],
        # Mid to rear (sides)
        [6, 10, 11], [6, 11, 7],
        [8, 12, 9], [8, 9, 5],
        # Rear to engine
        [9, 13, 14], [9, 14, 10],
        [11, 15, 16], [11, 16, 12],
        [10, 14, 15], [10, 15, 11],
        [12, 16, 13], [12, 13, 9],
        # Engine back
        [13, 16, 15], [13, 15, 14],
    ]

    # Nacelle struts and pods
    nacelle_verts = [
        # Left nacelle
        [[-0.1, hull_width * 0.5, 0], [-0.1, hull_width * 0.8, 0],
         [-0.3, hull_width * 0.8, 0.03], [-0.3, hull_width * 0.5, 0.03]],
        # Right nacelle
        [[-0.1, -hull_width * 0.5, 0], [-0.1, -hull_width * 0.8, 0],
         [-0.3, -hull_width * 0.8, 0.03], [-0.3, -hull_width * 0.5, 0.03]],
    ]

    return vertices, faces, nacelle_verts


def draw_spacecraft(ax, x_offset=0, y_offset=0, z_offset=0, scale=1.0, color='silver'):
    """Draw the spacecraft at given position."""
    vertices, faces, nacelles = create_spacecraft()

    # Scale and offset
    vertices = vertices * scale
    vertices[:, 0] += x_offset
    vertices[:, 1] += y_offset
    vertices[:, 2] += z_offset

    # Create face collection
    face_verts = [[vertices[idx] for idx in face] for face in faces]

    # Main hull
    hull = Poly3DCollection(face_verts, alpha=0.9)
    hull.set_facecolor(color)
    hull.set_edgecolor('darkgray')
    hull.set_linewidth(0.3)
    ax.add_collection3d(hull)

    # Nacelles
    for nacelle in nacelles:
        nacelle = np.array(nacelle) * scale
        nacelle[:, 0] += x_offset
        nacelle[:, 1] += y_offset
        nacelle[:, 2] += z_offset
        nac = Poly3DCollection([nacelle], alpha=0.9)
        nac.set_facecolor('#4080ff')  # Blue glow for warp nacelles
        nac.set_edgecolor('blue')
        ax.add_collection3d(nac)

    # Engine glow (simple point)
    ax.scatter([x_offset - 0.3*scale], [y_offset], [z_offset],
               c='cyan', s=50*scale, alpha=0.8, marker='o')

    return hull


def create_warp_visualization(n_frames=120, save_gif=True, save_mp4=True):
    """
    Create the main warp drive visualization.
    """
    print("\nCreating hero warp drive animation...")

    # Metric setup
    metric = AlcubierreMetric(v0=1.0, R=1.5, sigma=6.0)

    # Grid parameters
    grid_extent = 8
    n_grid_lines = 25
    grid_resolution = 80

    # Create figure with dark background
    fig = plt.figure(figsize=(16, 9), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')

    # Pre-compute some values
    x_base = np.linspace(-grid_extent, grid_extent * 1.5, grid_resolution)
    y_base = np.linspace(-grid_extent/2, grid_extent/2, grid_resolution)

    def animate(frame):
        ax.clear()

        # Time and bubble position
        t = frame * 0.12
        x_bubble = metric.bubble_center(t)

        # Set dark theme
        ax.set_facecolor('black')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('none')
        ax.yaxis.pane.set_edgecolor('none')
        ax.zaxis.pane.set_edgecolor('none')
        ax.grid(False)

        # Create distorted grid
        X, Y = np.meshgrid(x_base, y_base)
        Z_base = np.zeros_like(X)

        # Calculate warp field distortion
        Z_warp = np.zeros_like(X)
        energy_field = np.zeros_like(X)
        shape_field = np.zeros_like(X)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                x, y = X[i, j], Y[i, j]

                # Distance from bubble center
                r_s = np.sqrt((x - x_bubble)**2 + y**2)

                # Shape function value
                f = metric.shape_function(r_s)
                shape_field[i, j] = f

                # Shift vector effect (visual distortion)
                beta = metric.shift(t, x, y, 0)

                # Create z-displacement based on warp field
                # This shows the "warping" of space
                Z_warp[i, j] = -f * 0.8 + beta[0] * 0.3

                # Energy density for coloring
                energy_field[i, j] = metric.eulerian_energy_density_analytic(t, x, y, 0)

        # Normalize energy field for visualization
        energy_max = np.max(np.abs(energy_field))
        if energy_max > 0:
            energy_norm = -energy_field / energy_max  # Negative because exotic matter
        else:
            energy_norm = np.zeros_like(energy_field)

        # Draw base grid plane (subtle)
        ax.plot_surface(X, Y, Z_base - 1.5, alpha=0.1, color='gray',
                       linewidth=0, antialiased=True)

        # Draw grid lines with distortion
        line_indices = np.linspace(0, grid_resolution-1, n_grid_lines, dtype=int)

        # X-direction lines (along travel direction)
        for i in line_indices:
            y_line = Y[i, :]
            x_line = X[i, :]
            z_line = Z_warp[i, :]

            # Color based on energy density
            colors = plt.cm.cool(np.linspace(0.3, 0.7, len(x_line)))
            for k in range(len(x_line)-1):
                alpha = 0.4 + 0.4 * shape_field[i, k]
                ax.plot([x_line[k], x_line[k+1]],
                       [y_line[k], y_line[k+1]],
                       [z_line[k], z_line[k+1]],
                       color='cyan', alpha=alpha, linewidth=0.8)

        # Y-direction lines (perpendicular)
        for j in line_indices:
            y_line = Y[:, j]
            x_line = X[:, j]
            z_line = Z_warp[:, j]

            for k in range(len(x_line)-1):
                alpha = 0.4 + 0.4 * shape_field[k, j]
                ax.plot([x_line[k], x_line[k+1]],
                       [y_line[k], y_line[k+1]],
                       [z_line[k], z_line[k+1]],
                       color='cyan', alpha=alpha, linewidth=0.8)

        # Draw warp bubble as translucent surface
        theta = np.linspace(0, 2*np.pi, 50)
        phi = np.linspace(0, np.pi, 25)
        THETA, PHI = np.meshgrid(theta, phi)

        R_bubble = metric.params['R']
        X_bubble = x_bubble + R_bubble * np.sin(PHI) * np.cos(THETA)
        Y_bubble = R_bubble * np.sin(PHI) * np.sin(THETA)
        Z_bubble = R_bubble * np.cos(PHI) * 0.5  # Slightly flattened

        # Bubble surface with energy glow
        bubble_colors = np.zeros((*X_bubble.shape, 4))
        for i in range(X_bubble.shape[0]):
            for j in range(X_bubble.shape[1]):
                # Glow intensity based on position (stronger at edges)
                r_local = np.sqrt((X_bubble[i,j] - x_bubble)**2 + Y_bubble[i,j]**2)
                intensity = np.exp(-((r_local - R_bubble*0.8)**2) / (R_bubble*0.3)**2)
                bubble_colors[i, j] = [0.3, 0.7, 1.0, intensity * 0.4]

        ax.plot_surface(X_bubble, Y_bubble, Z_bubble,
                       facecolors=bubble_colors,
                       linewidth=0, antialiased=True, shade=False)

        # Draw energy density ring (exotic matter region)
        r_ring = np.linspace(R_bubble * 0.7, R_bubble * 1.3, 20)
        theta_ring = np.linspace(0, 2*np.pi, 60)
        R_RING, THETA_RING = np.meshgrid(r_ring, theta_ring)

        X_ring = x_bubble + R_RING * np.cos(THETA_RING)
        Y_ring = R_RING * np.sin(THETA_RING)

        # Energy density on ring
        Z_ring = np.zeros_like(X_ring)
        ring_colors = np.zeros((*X_ring.shape, 4))

        for i in range(X_ring.shape[0]):
            for j in range(X_ring.shape[1]):
                rho = metric.eulerian_energy_density_analytic(t, X_ring[i,j], Y_ring[i,j], 0)
                # Exotic matter glow (red/magenta for negative energy)
                intensity = min(1.0, -rho * 50) if rho < 0 else 0
                Z_ring[i, j] = -0.3  # Slightly below main plane
                ring_colors[i, j] = [1.0, 0.2, 0.5, intensity * 0.6]

        ax.plot_surface(X_ring, Y_ring, Z_ring,
                       facecolors=ring_colors,
                       linewidth=0, antialiased=True, shade=False)

        # Draw spacecraft
        draw_spacecraft(ax, x_offset=x_bubble, y_offset=0, z_offset=0.05,
                       scale=0.8, color='#c0c0c0')

        # Add engine trail/wake effect
        trail_length = 30
        trail_x = np.linspace(x_bubble - 0.5, x_bubble - 3, trail_length)
        trail_alpha = np.linspace(0.5, 0, trail_length)
        for i, (tx, ta) in enumerate(zip(trail_x, trail_alpha)):
            size = 20 * (1 - i/trail_length)
            ax.scatter([tx], [0], [0], c='cyan', s=size, alpha=ta, marker='o')

        # Add starfield background
        np.random.seed(42)  # Consistent stars
        n_stars = 200
        star_x = np.random.uniform(-grid_extent, grid_extent*2, n_stars)
        star_y = np.random.uniform(-grid_extent, grid_extent, n_stars)
        star_z = np.random.uniform(-2, 2, n_stars)
        star_sizes = np.random.uniform(0.5, 3, n_stars)
        ax.scatter(star_x, star_y, star_z, c='white', s=star_sizes, alpha=0.6, marker='.')

        # Camera follows bubble with smooth motion
        ax.set_xlim(x_bubble - 6, x_bubble + 4)
        ax.set_ylim(-5, 5)
        ax.set_zlim(-2, 2)

        # Dynamic camera angle
        elevation = 25 + 5 * np.sin(frame * 0.05)
        azimuth = -60 + 20 * np.sin(frame * 0.03)
        ax.view_init(elev=elevation, azim=azimuth)

        # Title with physics info
        ax.set_title(f'Alcubierre Warp Drive  |  v = c  |  t = {t:.1f}',
                    color='white', fontsize=14, pad=10)

        # Remove axis labels for cleaner look
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        # Add physics annotation
        ax.text2D(0.02, 0.98, 'Space contracts (front)', transform=ax.transAxes,
                 color='cyan', fontsize=9, verticalalignment='top')
        ax.text2D(0.02, 0.94, 'Space expands (rear)', transform=ax.transAxes,
                 color='cyan', fontsize=9, verticalalignment='top')
        ax.text2D(0.02, 0.90, 'Red: Exotic matter required', transform=ax.transAxes,
                 color='#ff3366', fontsize=9, verticalalignment='top')

        # Attribution
        ax.text2D(0.98, 0.02, 'WarpDrives Simulator', transform=ax.transAxes,
                 color='gray', fontsize=8, ha='right')

        if frame % 10 == 0:
            print(f"   Frame {frame}/{n_frames}")

        return []

    print(f"   Rendering {n_frames} frames...")

    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=n_frames,
                                   interval=50, blit=True)

    # Save as GIF
    if save_gif:
        print("   Saving GIF (this may take a while)...")
        anim.save('images/warp_drive_hero.gif', writer='pillow', fps=24)
        print("   Saved: images/warp_drive_hero.gif")

    # Save as MP4
    if save_mp4:
        print("   Saving MP4...")
        try:
            anim.save('images/warp_drive_hero.mp4', writer='ffmpeg', fps=24,
                     extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
            print("   Saved: images/warp_drive_hero.mp4")
        except Exception as e:
            print(f"   MP4 save failed (ffmpeg may not be available): {e}")
            # Try with different writer
            try:
                anim.save('images/warp_drive_hero.mp4', writer='imagemagick', fps=24)
                print("   Saved: images/warp_drive_hero.mp4 (via imagemagick)")
            except:
                print("   Could not save MP4 - saving additional GIF instead")

    plt.close()

    return anim


def create_top_down_view(n_frames=90):
    """
    Create a top-down view animation showing the grid distortion clearly.
    """
    print("\nCreating top-down warp animation...")

    metric = AlcubierreMetric(v0=1.0, R=1.5, sigma=6.0)

    fig, ax = plt.subplots(figsize=(12, 8), facecolor='black')
    ax.set_facecolor('black')

    def animate(frame):
        ax.clear()
        ax.set_facecolor('black')

        t = frame * 0.1
        x_bubble = metric.bubble_center(t)

        # Grid
        n_lines = 30
        x = np.linspace(-4, 12, 100)
        y_lines = np.linspace(-4, 4, n_lines)

        # Draw distorted horizontal lines
        for y_val in y_lines:
            x_distorted = []
            y_distorted = []
            alphas = []

            for xi in x:
                r_s = np.sqrt((xi - x_bubble)**2 + y_val**2)
                f = metric.shape_function(r_s)
                beta = metric.shift(t, xi, y_val, 0)

                x_dist = xi + beta[0] * 0.5  # Visual distortion
                x_distorted.append(x_dist)
                y_distorted.append(y_val)
                alphas.append(0.3 + 0.5 * f)

            # Draw line segments with varying alpha
            for i in range(len(x_distorted)-1):
                ax.plot([x_distorted[i], x_distorted[i+1]],
                       [y_distorted[i], y_distorted[i+1]],
                       color='cyan', alpha=alphas[i], linewidth=0.8)

        # Draw distorted vertical lines
        x_lines = np.linspace(-4, 12, n_lines)
        y = np.linspace(-4, 4, 100)

        for x_val in x_lines:
            x_distorted = []
            y_distorted = []
            alphas = []

            for yi in y:
                r_s = np.sqrt((x_val - x_bubble)**2 + yi**2)
                f = metric.shape_function(r_s)
                beta = metric.shift(t, x_val, yi, 0)

                x_dist = x_val + beta[0] * 0.5
                x_distorted.append(x_dist)
                y_distorted.append(yi)
                alphas.append(0.3 + 0.5 * f)

            for i in range(len(x_distorted)-1):
                ax.plot([x_distorted[i], x_distorted[i+1]],
                       [y_distorted[i], y_distorted[i+1]],
                       color='cyan', alpha=alphas[i], linewidth=0.8)

        # Draw energy density as background
        X, Y = np.meshgrid(np.linspace(-4, 12, 150), np.linspace(-4, 4, 100))
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = metric.eulerian_energy_density_analytic(t, X[i,j], Y[i,j], 0)

        # Show exotic matter region
        vmax = np.max(np.abs(Z))
        ax.contourf(X, Y, Z, levels=50, cmap='RdBu_r', alpha=0.3,
                   vmin=-vmax, vmax=vmax)

        # Draw bubble outline
        theta = np.linspace(0, 2*np.pi, 100)
        R = metric.params['R']
        ax.plot(x_bubble + R * np.cos(theta), R * np.sin(theta),
               'white', linewidth=2, alpha=0.7, linestyle='--')

        # Draw spacecraft (top view - triangle)
        ship_size = 0.4
        ship = plt.Polygon([
            [x_bubble + ship_size, 0],
            [x_bubble - ship_size*0.5, ship_size*0.4],
            [x_bubble - ship_size*0.5, -ship_size*0.4],
        ], facecolor='silver', edgecolor='white', linewidth=1)
        ax.add_patch(ship)

        # Engine glow
        ax.scatter([x_bubble - ship_size*0.5], [0], c='cyan', s=100, alpha=0.8, zorder=10)

        # Nacelles
        ax.scatter([x_bubble - ship_size*0.3], [ship_size*0.5], c='#4080ff', s=50, alpha=0.9)
        ax.scatter([x_bubble - ship_size*0.3], [-ship_size*0.5], c='#4080ff', s=50, alpha=0.9)

        ax.set_xlim(-4, 12)
        ax.set_ylim(-4, 4)
        ax.set_aspect('equal')

        ax.set_title(f'Warp Bubble - Top View  |  v = c  |  t = {t:.1f}',
                    color='white', fontsize=14)
        ax.set_xlabel('x (direction of travel)', color='gray')
        ax.set_ylabel('y', color='gray')
        ax.tick_params(colors='gray')

        # Annotations
        ax.annotate('Space\ncontracts', xy=(x_bubble + 2, 0), color='white',
                   fontsize=10, ha='center', alpha=0.7)
        ax.annotate('Space\nexpands', xy=(x_bubble - 2.5, 0), color='white',
                   fontsize=10, ha='center', alpha=0.7)

        if frame % 10 == 0:
            print(f"   Frame {frame}/{n_frames}")

        return []

    print(f"   Rendering {n_frames} frames...")

    anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=50)

    print("   Saving GIF...")
    anim.save('images/warp_drive_topdown.gif', writer='pillow', fps=20)
    print("   Saved: images/warp_drive_topdown.gif")

    plt.close()


if __name__ == '__main__':
    print("="*60)
    print("GENERATING HERO WARP DRIVE ANIMATIONS")
    print("="*60)

    # Create the main 3D visualization
    create_warp_visualization(n_frames=120, save_gif=True, save_mp4=True)

    # Create the top-down view
    create_top_down_view(n_frames=90)

    print("\n" + "="*60)
    print("ALL HERO ANIMATIONS COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  - images/warp_drive_hero.gif (3D view)")
    print("  - images/warp_drive_hero.mp4 (3D view, if ffmpeg available)")
    print("  - images/warp_drive_topdown.gif (top-down view)")
