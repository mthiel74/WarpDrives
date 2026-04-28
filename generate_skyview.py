#!/usr/bin/env python3
"""Render the warp-drive "windshield view" — what stars look like from
inside the bubble as the spacecraft accelerates from rest to >c.

Produces:

- ``images/skyview_alcubierre_v{...}.png`` for several velocities
- ``images/skyview_alcubierre_sweep.gif`` velocity-sweep animation
- ``images/skyview_natario_sweep.gif`` (optional, slower)

Run:
    python generate_skyview.py                      # defaults (small, fast)
    python generate_skyview.py --resolution 256 \\
        --frames 32 --jobs 8                        # high-quality

Use ``--sky-image path/to/milky_way.jpg`` to drop in an equirectangular
panorama instead of the procedural starfield.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from warpbubblesim.metrics import AlcubierreMetric, NatarioMetric
from warpbubblesim.viz.skybackground import (
    make_grid_sky,
    make_image_sky,
    make_procedural_starfield,
)
from warpbubblesim.viz.skyrender import (
    Camera,
    RenderConfig,
    render_sky_view,
    render_velocity_sweep,
    save_frames_as_animation,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--resolution", type=int, default=96,
                   help="Image side length in pixels (default 96).")
    p.add_argument("--frames", type=int, default=12,
                   help="Number of velocity-sweep frames (default 12).")
    p.add_argument("--v-max", type=float, default=2.5,
                   help="Maximum bubble velocity in c (default 2.5).")
    p.add_argument("--fov-deg", type=float, default=90.0,
                   help="Camera horizontal FOV in degrees (default 90).")
    p.add_argument("--jobs", type=int, default=1,
                   help="Number of worker processes (default 1).")
    p.add_argument("--sky", choices=["stars", "grid", "image"], default="stars",
                   help="Sky background to use.")
    p.add_argument("--sky-image", type=str, default=None,
                   help="Path to equirectangular sky image (used with --sky image).")
    p.add_argument("--output-dir", type=str, default="images",
                   help="Output directory.")
    p.add_argument("--metric", choices=["alcubierre", "natario", "both"],
                   default="alcubierre")
    p.add_argument("--R", type=float, default=1.0)
    p.add_argument("--sigma", type=float, default=8.0)
    p.add_argument("--still-frames", action="store_true",
                   help="Also write individual still PNGs at v∈{0, 0.5, 0.99, 1.5, v_max}.")
    return p.parse_args()


def build_sky(args: argparse.Namespace, fov_rad: float, resolution: int):
    if args.sky == "grid":
        return make_grid_sky(spacing_deg=10.0, line_width_deg=0.3)
    if args.sky == "image":
        if not args.sky_image:
            raise SystemExit("--sky image requires --sky-image PATH")
        return make_image_sky(args.sky_image)
    # default: stars.  Tune star size to ~1.5 pixels independent of FOV/res.
    return make_procedural_starfield(
        n_stars=4500,
        seed=42,
        star_radius_px=1.5,
        fov_scale=fov_rad / resolution,
    )


def render_for_metric(name: str, factory, args, sky, cam, cfg, out_dir: Path):
    print(f"\n=== Rendering {name} sweep (frames={args.frames}, "
          f"res={args.resolution}, jobs={args.jobs}) ===")
    velocities = np.linspace(0.0, args.v_max, args.frames)
    frames = render_velocity_sweep(factory, velocities, sky, camera=cam, config=cfg)

    gif_path = out_dir / f"skyview_{name}_sweep.gif"
    save_frames_as_animation(frames, str(gif_path), fps=10)
    print(f"  wrote {gif_path}")

    if args.still_frames:
        # Pick characteristic velocities and either reuse or rerender
        characteristic = sorted(set([0.0, 0.5, 0.99, 1.5, args.v_max]))
        for v in characteristic:
            # Find closest sweep frame, or render afresh if outside range
            idx = int(np.argmin(np.abs(velocities - v)))
            if abs(velocities[idx] - v) < 1e-3:
                img = frames[idx]
            else:
                img = render_sky_view(factory(v), sky, camera=cam, config=cfg)
            png = out_dir / f"skyview_{name}_v{v:.2f}.png"
            plt.imsave(png, np.clip(img, 0, 1))
            print(f"  wrote {png}")


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fov_rad = np.deg2rad(args.fov_deg)
    cam = Camera(width=args.resolution, height=args.resolution,
                 fov_deg=args.fov_deg)
    sky = build_sky(args, fov_rad, args.resolution)
    cfg = RenderConfig(
        escape_radius_factor=5.0,
        lambda_max_factor=20.0,
        rtol=3e-4, atol=3e-6,
        max_step=0.5, method="RK23",
        h=2e-3, n_jobs=args.jobs,
        show_progress=False,
    )

    R, sigma = args.R, args.sigma

    metrics = []
    if args.metric in ("alcubierre", "both"):
        metrics.append(("alcubierre",
                        lambda v, R=R, s=sigma: AlcubierreMetric(v0=v, R=R, sigma=s)))
    if args.metric in ("natario", "both"):
        metrics.append(("natario",
                        lambda v, R=R, s=sigma: NatarioMetric(v0=v, R=R, sigma=s)))

    for name, factory in metrics:
        render_for_metric(name, factory, args, sky, cam, cfg, out_dir)

    print("\nAll renders complete.")


if __name__ == "__main__":
    main()
