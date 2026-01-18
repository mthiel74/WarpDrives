"""
Command-line interface for WarpBubbleSim.

Usage:
    warpsim list-metrics
    warpsim render --metric alcubierre --scenario scenarios/demo.yaml
    warpsim geodesics --metric alcubierre --output out/geodesics.mp4
    warpsim sweep --metric alcubierre --param v0 --values 0.1,0.5,1.0,2.0
"""

import click
import numpy as np
from pathlib import Path
import yaml
from typing import Optional, List
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for CLI
import matplotlib.pyplot as plt


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """WarpBubbleSim - GR Warp Bubble Spacetime Simulator"""
    pass


@cli.command()
def list_metrics():
    """List all available warp drive metrics."""
    from warpbubblesim.metrics import MetricRegistry

    click.echo("Available warp drive metrics:")
    click.echo("-" * 40)

    for name in MetricRegistry.list_metrics():
        metric_class = MetricRegistry._metrics[name]
        metric = metric_class()
        click.echo(f"\n{name}:")
        click.echo(f"  Name: {metric.name}")
        click.echo(f"  Citation: {metric.citation}")
        click.echo(f"  Default params: {metric._default_params()}")


@cli.command()
@click.option('--metric', '-m', required=True, help='Metric name')
@click.option('--scenario', '-s', required=True, type=click.Path(exists=True),
              help='Path to scenario YAML file')
@click.option('--output', '-o', default='out/', help='Output directory')
@click.option('--resolution', '-r', default=128, type=int, help='Grid resolution')
@click.option('--format', '-f', default='png', type=click.Choice(['png', 'pdf', 'svg']),
              help='Output format')
def render(metric: str, scenario: str, output: str, resolution: int, format: str):
    """Render field visualizations for a warp metric."""
    from warpbubblesim.metrics import get_metric
    from warpbubblesim.viz.fields2d import (
        plot_energy_density, plot_expansion_scalar,
        plot_shape_function, plot_grid_distortion
    )
    from warpbubblesim.utils.io import load_yaml_config

    click.echo(f"Loading scenario from {scenario}...")
    config = load_yaml_config(scenario)

    # Merge scenario params with metric defaults
    metric_params = config.get('metric_params', {})
    viz_params = config.get('visualization', {})

    click.echo(f"Creating {metric} metric...")
    warp_metric = get_metric(metric, **metric_params)

    # Create output directory
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get visualization parameters
    x_range = tuple(viz_params.get('x_range', [-5, 5]))
    y_range = tuple(viz_params.get('y_range', [-5, 5]))
    t = viz_params.get('t', 0.0)

    click.echo(f"Rendering energy density...")
    fig, ax = plot_energy_density(
        warp_metric, x_range, y_range, resolution, resolution, t
    )
    fig.savefig(output_path / f"{metric}_energy_density.{format}", dpi=150, bbox_inches='tight')
    plt.close(fig)

    click.echo(f"Rendering expansion scalar...")
    fig, ax = plot_expansion_scalar(
        warp_metric, x_range, y_range, resolution, resolution, t
    )
    fig.savefig(output_path / f"{metric}_expansion.{format}", dpi=150, bbox_inches='tight')
    plt.close(fig)

    click.echo(f"Rendering shape function...")
    fig, ax = plot_shape_function(warp_metric)
    fig.savefig(output_path / f"{metric}_shape.{format}", dpi=150, bbox_inches='tight')
    plt.close(fig)

    click.echo(f"Rendering grid distortion...")
    fig, ax = plot_grid_distortion(warp_metric, x_range, y_range, t=t)
    fig.savefig(output_path / f"{metric}_grid.{format}", dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Create combined fields image
    from warpbubblesim.viz.fields2d import plot_multiple_fields
    click.echo(f"Rendering combined fields...")
    fig = plot_multiple_fields(
        warp_metric,
        fields=['energy_density', 'expansion', 'shape'],
        x_range=x_range, y_range=y_range,
        nx=resolution // 2, ny=resolution // 2, t=t,
        save_path=output_path / f"{metric}_fields.{format}"
    )
    plt.close(fig)

    click.echo(f"Output saved to {output_path}/")


@cli.command()
@click.option('--metric', '-m', required=True, help='Metric name')
@click.option('--output', '-o', required=True, help='Output file path')
@click.option('--n-particles', '-n', default=10, type=int, help='Number of test particles')
@click.option('--t-max', default=10.0, type=float, help='Maximum time')
@click.option('--fps', default=30, type=int, help='Frames per second')
def geodesics(metric: str, output: str, n_particles: int, t_max: float, fps: int):
    """Generate geodesic animation for a warp metric."""
    from warpbubblesim.metrics import get_metric
    from warpbubblesim.viz.animations import animate_geodesics, save_animation

    click.echo(f"Creating {metric} metric...")
    warp_metric = get_metric(metric)

    R = warp_metric.params.get('R', 1.0)

    # Generate initial positions
    click.echo(f"Setting up {n_particles} test particles...")
    initial_positions = []
    for i in range(n_particles):
        x0 = -3 * R + (6 * R) * i / (n_particles - 1) if n_particles > 1 else 0
        initial_positions.append((x0, 0.0, 0.0))

    click.echo("Integrating geodesics and creating animation...")
    anim = animate_geodesics(
        warp_metric,
        initial_positions,
        t_range=(0, t_max),
        fps=fps,
        save_path=output,
        show_progress=True
    )

    click.echo(f"Animation saved to {output}")


@cli.command()
@click.option('--metric', '-m', required=True, help='Metric name')
@click.option('--param', '-p', required=True, help='Parameter to sweep')
@click.option('--values', '-v', required=True, help='Comma-separated parameter values')
@click.option('--output', '-o', default='out/sweep/', help='Output directory')
@click.option('--resolution', '-r', default=64, type=int, help='Grid resolution')
def sweep(metric: str, param: str, values: str, output: str, resolution: int):
    """Perform parameter sweep for a metric."""
    from warpbubblesim.metrics import get_metric
    from warpbubblesim.viz.fields2d import plot_energy_density

    # Parse values
    param_values = [float(v.strip()) for v in values.split(',')]

    click.echo(f"Parameter sweep: {param} = {param_values}")

    output_path = Path(output) / f"{metric}_{param}_sweep"
    output_path.mkdir(parents=True, exist_ok=True)

    for value in param_values:
        click.echo(f"  {param} = {value}")

        params = {param: value}
        warp_metric = get_metric(metric, **params)

        fig, ax = plot_energy_density(
            warp_metric,
            nx=resolution, ny=resolution
        )
        ax.set_title(f'{warp_metric.name}\n{param} = {value}')

        filename = f"{metric}_{param}_{value:.3f}.png"
        fig.savefig(output_path / filename, dpi=150, bbox_inches='tight')
        plt.close(fig)

    click.echo(f"Sweep results saved to {output_path}/")


@cli.command()
@click.option('--metric', '-m', required=True, help='Metric name')
@click.option('--output', '-o', default='out/', help='Output directory')
@click.option('--resolution', '-r', default=64, type=int, help='Grid resolution')
@click.option('--n-frames', '-f', default=50, type=int, help='Number of frames')
@click.option('--format', default='gif', type=click.Choice(['gif', 'mp4']),
              help='Animation format')
def animate(metric: str, output: str, resolution: int, n_frames: int, format: str):
    """Create field evolution animation."""
    from warpbubblesim.metrics import get_metric
    from warpbubblesim.viz.animations import animate_field_evolution, animate_grid_distortion

    click.echo(f"Creating {metric} metric...")
    warp_metric = get_metric(metric)

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Field evolution
    click.echo("Creating field evolution animation...")
    anim = animate_field_evolution(
        warp_metric,
        field="shape",
        nx=resolution, ny=resolution,
        n_frames=n_frames,
        save_path=output_path / f"{metric}_evolution.{format}",
        show_progress=True
    )
    plt.close()

    # Grid distortion
    click.echo("Creating grid distortion animation...")
    anim = animate_grid_distortion(
        warp_metric,
        n_frames=n_frames,
        save_path=output_path / f"{metric}_grid_distortion.{format}"
    )
    plt.close()

    click.echo(f"Animations saved to {output_path}/")


@cli.command()
@click.option('--metric', '-m', required=True, help='Metric name')
@click.option('--point', '-p', default='0,0,0,0', help='Spacetime point (t,x,y,z)')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed output')
def check_conditions(metric: str, point: str, verbose: bool):
    """Check energy conditions at a point."""
    from warpbubblesim.metrics import get_metric
    from warpbubblesim.gr.conditions import check_energy_conditions
    from warpbubblesim.gr.energy import compute_energy_density, decompose_stress_energy

    coords = np.array([float(x.strip()) for x in point.split(',')])

    click.echo(f"Checking energy conditions for {metric} at {coords}")

    warp_metric = get_metric(metric)
    metric_func = warp_metric.get_metric_func()

    # Check conditions
    conditions = check_energy_conditions(metric_func, coords)

    click.echo("\nEnergy Conditions:")
    click.echo("-" * 40)
    for name, (satisfied, value) in conditions.items():
        status = "✓ Satisfied" if satisfied else "✗ Violated"
        click.echo(f"  {name}: {status} (value: {value:.2e})")

    if verbose:
        # Compute energy density
        rho = compute_energy_density(metric_func, coords)
        click.echo(f"\nEnergy density: {rho:.4e}")

        # Stress-energy decomposition
        decomp = decompose_stress_energy(metric_func, coords)
        click.echo(f"Pressure: {decomp['pressure']:.4e}")


@cli.command()
@click.option('--metric', '-m', required=True, help='Metric name')
def info(metric: str):
    """Show detailed information about a metric."""
    from warpbubblesim.metrics import get_metric

    warp_metric = get_metric(metric)

    click.echo(f"\n{warp_metric.name}")
    click.echo("=" * len(warp_metric.name))
    click.echo(f"\nCitation: {warp_metric.citation}")
    click.echo(f"\nParameters:")
    for key, value in warp_metric.params.items():
        click.echo(f"  {key}: {value}")

    # Metric-specific info
    if hasattr(warp_metric, 'description'):
        click.echo(f"\n{warp_metric.description()}")

    if hasattr(warp_metric, 'energy_condition_type'):
        click.echo(f"\nEnergy conditions: {warp_metric.energy_condition_type()}")


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
