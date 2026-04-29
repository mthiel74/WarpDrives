# SkyRender (Wolfram)

Wolfram Language port of the Alcubierre / Natário windshield ray tracer
that the Python notebook (`notebooks/07_warp_skyflight_colab.ipynb`)
uses. Same physics, same numerical scheme:

- ADM-form 4-metric, central-difference Christoffels
- Backward null geodesic via fixed-step RK4
- Orthonormal tetrad on the bubble-centre observer (Gram–Schmidt against `g`)
- Asymptotic spatial direction → celestial-sphere lookup
- Liouville `I_ν / ν³` Doppler brightness scaling
- Black mask for pixels behind the front horizon

The maths is documented in `docs/maths/skyrender.pdf`.

## Files

| File | Purpose |
|---|---|
| `SkyRender.wl` | Library: metrics, integrator, tetrad, render, GPU submission helpers |
| `Demo.wls` | One-frame demo. Run `wolframscript -file Demo.wls`. |
| `README.md` | This file |

## Run locally (CPU)

```bash
wolframscript -file Wolfram/skyrender/Demo.wls
```

Renders one 256×144 frame at `v = 0.95c` and writes `skyrender_demo.png` next
to the script. Modify `Demo.wls` for other velocities / resolutions / metrics.

## Run on a Wolfram Cloud GPU

`SkyRender.wl` ships two helpers built on
[`RemoteBatchSubmit`](https://reference.wolfram.com/language/ref/RemoteBatchSubmit.html):

```wolfram
(* one frame on an NVIDIA L40S *)
job = SubmitFrameToGPU[
    "alcubierre",
    <|"v0" -> 2.5, "R" -> 1.0, "sigma" -> 8.0|>,
    "/path/to/milky_way_panorama.jpg",
    {"Width" -> 1280, "Height" -> 720, "NSteps" -> 280, "Dlam" -> 0.15}
];
job["Status"]    (* poll *)
img = job["Result"]    (* once Completed *)
```

```wolfram
(* whole velocity sweep, one job per frame *)
velocities = EaseInOutVelocities[60, 4.0];
jobs = SubmitVelocitySweepToGPU[
    "alcubierre",
    <|"R" -> 1.0, "sigma" -> 8.0|>,
    velocities,
    "/path/to/milky_way_panorama.jpg"
];
```

The functions thread `RemoteMachineClass -> "GPU1xL40S"` (single L40S, 44 GiB
GPU memory) through to `RemoteBatchSubmit`. The other GPU machine class
exposed by Wolfram is `"GPU4xL4"` (4× NVIDIA L4, 89 GiB total) — pass it via
`batchOpts` to override:

```wolfram
SubmitFrameToGPU[
    "alcubierre", params, skyPath, frameOpts,
    {RemoteMachineClass -> "GPU4xL4"}
]
```

## Caveats

1. **Entitlement.** `RemoteBatchSubmit` requires a Wolfram One subscription
   with Service Credits, or an AWS-Batch / Azure-Batch provider configured
   on your end. Local runs (`Demo.wls`) work without any cloud setup.

2. **Performance.** This is straight Wolfram Language, not Compile/CUDA. The
   per-pixel `IntegrateGeodesic` loop is the bottleneck. On a CPU, a 256×144
   subluminal frame takes 1–2 minutes; superluminal a bit longer.
   GPU submission ships the whole code over and runs it remotely; the
   compute itself runs on the cloud kernel.

3. **Natário caveat.** The bubble-centre observer's worldline is timelike
   only at `v < 1`. The Wolfram code uses the same regularised metric
   formulation (eps = R/100) and FD Christoffels as the Python JAX path.

4. **Honest physics only.** Brightness scales as `f^p` (default `p = 3` =
   Liouville monochromatic invariance). No invented colour shift.
   `enableHorizonMask = True` blanks trapped pixels at `|v| > 1`.

## Tying back to the Python pipeline

The Python checkpointing/Drive workflow (`notebooks/07_warp_skyflight_colab.ipynb`)
is the one optimised for high-resolution video assembly. This Wolfram
implementation is the symbolic / GPU-batch counterpart — useful if you'd
rather work in Mathematica, want to inspect intermediate tensors
symbolically, or have a Wolfram cloud GPU entitlement and not Colab.

Both produce the same images modulo numerical noise (RK4 vs. JAX vmap'd RK4
agree to RK4 truncation error).
