(* ::Package:: *)

(* ::Title:: *)
(*SkyRender — Wolfram implementation of the Alcubierre/Natário sky tracer*)

(* ::Text:: *)
(*Mirrors the Python warpbubblesim.viz.skyrender_jax pipeline:*)
(*  1. Build a 4-metric in ADM form for an Alcubierre or Natário warp bubble.*)
(*  2. Construct an orthonormal tetrad on the bubble-centre observer.*)
(*  3. For each pixel of a virtual windshield, fire a backward null geodesic*)
(*     through the metric using a fixed-step RK4 integrator on the standard*)
(*     ODE   d^2 x^μ/dλ² + Γ^μ_{αβ} (dx^α/dλ)(dx^β/dλ) = 0.*)
(*  4. Read the asymptotic spatial direction, look up a celestial-sphere*)
(*     texture, apply Liouville's I_ν/ν³ Doppler brightness scaling, and*)
(*     mask out pixels whose ray failed to escape the bubble (front horizon).*)
(**)
(*GPU offload:  Heavy frame-rendering can be submitted to a Wolfram Cloud*)
(*GPU runtime via RemoteBatchSubmit[..., RemoteMachineClass -> "GPU1xL40S"].*)
(*See `SubmitFrameToGPU` and `SubmitVelocitySweepToGPU` at the bottom.*)
(**)
(*Conventions match the Python code:*)
(*  - signature (-,+,+,+),  (t,x,y,z) = x^0..x^3*)
(*  - past-pointing null tangents (k^t < 0)*)
(*  - geometric units (G = c = 1)*)


(* ============================================================ *)
(* Shape function and warp metrics                              *)
(* ============================================================ *)

(* Alcubierre's tanh shape function *)
TanhShape[r_, R_, sigma_] :=
  Module[{ap, am, aR},
    ap = Clip[sigma (r + R), {-20, 20}];
    am = Clip[sigma (r - R), {-20, 20}];
    aR = Clip[sigma R, {-20, 20}];
    (Tanh[ap] - Tanh[am]) / (2 Tanh[aR])
  ];

(* Analytic dn/dr for tanh shape (used by Natário) *)
TanhShapeD[r_, R_, sigma_] :=
  Module[{ap, am, aR},
    ap = Clip[sigma (r + R), {-20, 20}];
    am = Clip[sigma (r - R), {-20, 20}];
    aR = Clip[sigma R, {-20, 20}];
    sigma (Sech[ap]^2 - Sech[am]^2) / (2 Tanh[aR])
  ];

(* Alcubierre 4-metric at a single point.  Returns 4x4. *)
AlcubierreMetric[t_, x_, y_, z_, opts : OptionsPattern[{
    "v0" -> 1.0, "R" -> 1.0, "sigma" -> 8.0, "x0" -> 0.0
}]] := Module[{v, R, sig, x0, xs, dxs, r, f, g},
  v = OptionValue["v0"]; R = OptionValue["R"]; sig = OptionValue["sigma"];
  x0 = OptionValue["x0"]; xs = x0 + v t; dxs = x - xs;
  r = Sqrt[dxs^2 + y^2 + z^2 + 10^-20];
  f = TanhShape[r, R, sig];
  g = IdentityMatrix[4];
  g[[1, 1]] = -(1 - v^2 f^2);
  g[[1, 2]] = -v f;
  g[[2, 1]] = -v f;
  g
];

(* Natário 4-metric (curl-based divergence-free shift).  Eps regularises the *)
(* radial coordinate so derivatives at the bubble centre are bounded.        *)
NatarioMetric[t_, x_, y_, z_, opts : OptionsPattern[{
    "v0" -> 0.5, "R" -> 1.0, "sigma" -> 8.0, "x0" -> 0.0
}]] := Module[{v, R, sig, x0, eps, xs, dxs, r, n, dn, rho2, bx, by, bz, g},
  v = OptionValue["v0"]; R = OptionValue["R"]; sig = OptionValue["sigma"];
  x0 = OptionValue["x0"]; xs = x0 + v t; dxs = x - xs;
  eps = 0.01 R;
  r = Sqrt[dxs^2 + y^2 + z^2 + eps^2];
  n = TanhShape[r, R, sig];
  dn = TanhShapeD[r, R, sig];
  rho2 = y^2 + z^2;
  bx = -v (2 n + rho2 dn / r);
  by =  v y dxs dn / r;
  bz =  v z dxs dn / r;
  g = IdentityMatrix[4];
  g[[1, 1]] = -1 + bx^2 + by^2 + bz^2;
  g[[1, 2]] = bx; g[[2, 1]] = bx;
  g[[1, 3]] = by; g[[3, 1]] = by;
  g[[1, 4]] = bz; g[[4, 1]] = bz;
  g
];


(* ============================================================ *)
(* Christoffel symbols via central finite differences           *)
(* ============================================================ *)

(* Vectorised central differences on metricFn at the point coords. *)
(* Returns dg with axes (alpha, mu, nu): dg[[alpha, mu, nu]] = ∂_alpha g_{mu nu}. *)
MetricGrad[metricFn_, coords_, h_ : 0.001] := Module[{e, gp, gm},
  e = h IdentityMatrix[4];
  gp = Table[metricFn @@ (coords + e[[k]]), {k, 4}];
  gm = Table[metricFn @@ (coords - e[[k]]), {k, 4}];
  (gp - gm) / (2 h)
];

(* Christoffel symbols of the second kind: Γ^μ_{αβ}. *)
(* Returned indexed as gamma[[mu, alpha, beta]]. *)
Christoffel[metricFn_, coords_, h_ : 0.001] := Module[{g, ginv, dg, T1, T2, T3, br},
  g = metricFn @@ coords;
  ginv = Inverse[g];
  dg = MetricGrad[metricFn, coords, h];
  T1 = dg;                                                    (* ∂_α g_{βρ} *)
  T2 = Transpose[dg, {2, 1, 3}];                              (* ∂_β g_{αρ} *)
  T3 = Transpose[dg, {3, 1, 2}];                              (* ∂_ρ g_{αβ} *)
  br = T1 + T2 - T3;
  0.5 ginv . br // Transpose[#, {2, 3, 1, 4}] & // (* helper-free; use Tensor *)
  (* explicit einsum: gamma[μ, α, β] = (1/2) g^{μρ} br[α, β, ρ] *)
  ((1/2) Sum[ginv[[#, rho]] br[[All, All, rho]], {rho, 4}]) &@1
];

(* Cleaner Christoffel using Transpose / einsum-like contraction *)
Christoffel[metricFn_, coords_, h_ : 0.001] := Module[
  {g, ginv, dg, br, gamma},
  g = metricFn @@ coords;
  ginv = Inverse[g];
  dg = MetricGrad[metricFn, coords, h];
  br = dg + Transpose[dg, {2, 1, 3}] - Transpose[dg, {3, 1, 2}];
  (* gamma[mu, alpha, beta] = (1/2) g^{mu rho} br[alpha, beta, rho] *)
  gamma = (1/2) Transpose[ginv . Transpose[br, {3, 1, 2}], {2, 3, 1}];
  gamma
];


(* ============================================================ *)
(* Geodesic RHS and RK4 integrator                              *)
(* ============================================================ *)

GeodesicRHS[metricFn_, state_, h_ : 0.001] := Module[
  {coords, k, gamma, accel},
  coords = state[[1 ;; 4]];
  k = state[[5 ;; 8]];
  gamma = Christoffel[metricFn, coords, h];
  (* a^μ = -Γ^μ_{αβ} k^α k^β *)
  accel = -Table[Sum[gamma[[mu, a, b]] k[[a]] k[[b]], {a, 4}, {b, 4}], {mu, 4}];
  Join[k, accel]
];

RK4Step[metricFn_, state_, dlam_, h_ : 0.001] := Module[{k1, k2, k3, k4},
  k1 = GeodesicRHS[metricFn, state, h];
  k2 = GeodesicRHS[metricFn, state + (dlam/2) k1, h];
  k3 = GeodesicRHS[metricFn, state + (dlam/2) k2, h];
  k4 = GeodesicRHS[metricFn, state +  dlam     k3, h];
  state + (dlam / 6) (k1 + 2 k2 + 2 k3 + k4)
];

IntegrateGeodesic[metricFn_, initState_, nSteps_, dlam_, h_ : 0.001] :=
  Nest[RK4Step[metricFn, #, dlam, h] &, initState, nSteps];


(* ============================================================ *)
(* Orthonormal tetrad construction                              *)
(* ============================================================ *)

MetricInner[g_, a_, b_] := Sum[g[[mu, nu]] a[[mu]] b[[nu]], {mu, 4}, {nu, 4}];

GramSchmidt[g_, seed_, prev_] := Module[{v, p, ipp, ipv},
  v = seed;
  Do[
    p = prev[[i]];
    ipp = MetricInner[g, p, p];
    ipv = MetricInner[g, p, v];
    v = v - (ipv / ipp) p,
    {i, Length[prev]}
  ];
  v / Sqrt[MetricInner[g, v, v]]
];

OrthonormalTetrad[metricFn_, coords_, u_] := Module[
  {g, et, ex, ez, ey, det3},
  g = metricFn @@ coords;
  et = u / Sqrt[-MetricInner[g, u, u]];
  ex = GramSchmidt[g, {0, 1, 0, 0}, {et}];
  ez = GramSchmidt[g, {0, 0, 0, 1}, {et, ex}];
  ey = GramSchmidt[g, {0, 0, 1, 0}, {et, ex, ez}];
  det3 = Det[Transpose[{ex[[2 ;; 4]], ey[[2 ;; 4]], ez[[2 ;; 4]]}]];
  If[det3 < 0, ey = -ey];
  {et, ex, ey, ez}
];


(* ============================================================ *)
(* Camera and per-pixel ray construction                        *)
(* ============================================================ *)

PixelDirections[width_, height_, fovDeg_] := Module[
  {fov, aspect, halfW, halfH, dirs},
  fov = fovDeg Degree; aspect = width/height;
  halfW = Tan[fov/2]; halfH = halfW / aspect;
  dirs = Table[
    With[{u = (2 (i + 0.5)/width - 1) halfW,
          v = (1 - 2 (j + 0.5)/height) halfH},
      Normalize[{1.0, u, v}]],
    {j, 0, height - 1}, {i, 0, width - 1}];
  dirs
];

PastNullTangent[tetrad_, nLocal_] :=
  -tetrad[[1]] + nLocal[[1]] tetrad[[2]] + nLocal[[2]] tetrad[[3]] + nLocal[[3]] tetrad[[4]];


(* ============================================================ *)
(* Honest-physics post-processing                               *)
(* ============================================================ *)

(* Doppler factor f = ω_obs / ω_src using past-pointing tangents. *)
DopplerFactor[gObs_, kInit_, uObs_, kFinal_, uSrc_ : {1, 0, 0, 0}] :=
  Module[{eta, omegaObs, omegaSrc},
    eta = DiagonalMatrix[{-1.0, 1.0, 1.0, 1.0}];
    omegaObs = MetricInner[gObs, kInit, uObs];
    omegaSrc = MetricInner[eta, kFinal, uSrc];
    omegaObs / If[Abs[omegaSrc] < 10^-12, 10^-12, omegaSrc]
  ];

(* Liouville f^p brightness scaling — no fake colour shift. *)
ApplyDoppler[rgb_, f_, power_ : 3.0] := Max[f, 0]^power rgb;


(* ============================================================ *)
(* End-to-end frame renderer                                    *)
(* ============================================================ *)

(* Build a sky function from an equirectangular image, returning a *)
(* CompiledFunction-like callable that maps a unit 3-vector to RGB.   *)
LoadEquirectangularSky[path_, opts : OptionsPattern[{"Gain" -> 1.0}]] :=
  Module[{img, data, h, w, gain},
    img = Import[path];
    data = ImageData[img, "Real32"];     (* H x W x 3 in [0,1] *)
    {h, w} = Dimensions[data][[1 ;; 2]];
    gain = OptionValue["Gain"];
    Function[{n3},
      Module[{theta, phi, u, v, col, row},
        theta = ArcCos[Clip[n3[[3]], {-1, 1}]];
        phi   = ArcTan[n3[[1]], n3[[2]]];
        u = Mod[(phi + Pi)/(2 Pi), 1.0];
        v = theta / Pi;
        col = Min[Floor[u w] + 1, w];
        row = Min[Floor[v h] + 1, h];
        Min[#, 1.0] & /@ (gain data[[row, col]])
      ]
    ]
  ];

(* Procedural starfield fallback. *)
ProceduralSky[seed_ : 1, nStars_ : 4000] := Module[{rng, dirs, mags},
  SeedRandom[seed];
  dirs = With[{u = RandomReal[{-1, 1}, nStars],
               phi = RandomReal[{-Pi, Pi}, nStars]},
    Transpose[{Sqrt[Clip[1 - u^2, {0, 1}]] Cos[phi],
               Sqrt[Clip[1 - u^2, {0, 1}]] Sin[phi], u}]];
  mags = Exp[-RandomVariate[ExponentialDistribution[1], nStars] 0.7];
  Function[{n3}, Module[{cs, k},
    cs = dirs . n3;                       (* cosine similarity *)
    k = Ordering[-cs, 1][[1]];
    {1, 1, 1} mags[[k]] Exp[-2 (1 - cs[[k]]) / 0.0008]
    + {0.005, 0.008, 0.02}                 (* faint background *)
  ]]
];


(* Render one frame.  metricName ∈ {"alcubierre", "natario"}. *)
RenderFrame[metricName_, params_Association, sky_, opts : OptionsPattern[{
    "Width" -> 320, "Height" -> 180, "FovDeg" -> 90.0,
    "NSteps" -> 240, "Dlam" -> 0.15, "FdStep" -> 0.001,
    "EnableDoppler" -> True, "DopplerPower" -> 3.0,
    "EnableHorizonMask" -> True, "HorizonSafety" -> 1.5,
    "FallbackColor" -> {0.4, 0, 0}
}]] := Module[
  {w, h, fov, n, dl, hf, doppler, dPow, hMask, hSafe, fbCol,
   metricFn, v, R, x0, xsT, coords0, uSeed, g0, nn, u, tetrad,
   dirs, frame},
  w = OptionValue["Width"]; h = OptionValue["Height"];
  fov = OptionValue["FovDeg"]; n = OptionValue["NSteps"];
  dl = OptionValue["Dlam"]; hf = OptionValue["FdStep"];
  doppler = OptionValue["EnableDoppler"]; dPow = OptionValue["DopplerPower"];
  hMask = OptionValue["EnableHorizonMask"]; hSafe = OptionValue["HorizonSafety"];
  fbCol = OptionValue["FallbackColor"];

  (* Closure over params *)
  metricFn = If[metricName === "alcubierre",
    Function[{tt, xx, yy, zz},
      AlcubierreMetric[tt, xx, yy, zz,
        "v0" -> params["v0"], "R" -> params["R"],
        "sigma" -> params["sigma"], "x0" -> Lookup[params, "x0", 0]]],
    Function[{tt, xx, yy, zz},
      NatarioMetric[tt, xx, yy, zz,
        "v0" -> params["v0"], "R" -> params["R"],
        "sigma" -> params["sigma"], "x0" -> Lookup[params, "x0", 0]]]
  ];

  v = params["v0"]; R = params["R"]; x0 = Lookup[params, "x0", 0];
  xsT = x0;                                 (* t = 0 frame *)
  coords0 = {0.0, xsT, 0.0, 0.0};

  uSeed = {1.0, v, 0.0, 0.0};
  g0 = metricFn @@ coords0;
  nn = MetricInner[g0, uSeed, uSeed];
  If[nn >= 0,
    uSeed = {1.0, 0.0, 0.0, 0.0};
    nn = MetricInner[g0, uSeed, uSeed];
    If[nn >= 0, Throw[Failure["NoTimelikeObserver", <|"v" -> v|>]]]
  ];
  u = uSeed / Sqrt[-nn];
  tetrad = OrthonormalTetrad[metricFn, coords0, u];

  dirs = PixelDirections[w, h, fov];     (* H x W x 3 *)

  (* Per-pixel render loop.  In production you'd use Compile or send the *)
  (* whole grid to the GPU; for clarity this is a straight pure-Wolfram   *)
  (* implementation.                                                       *)
  frame = ParallelTable[
    Module[{nLoc, kInit, state, sf, kFin, sp, nrm, asym, rgb, f, rs},
      nLoc = dirs[[j, i]];
      kInit = PastNullTangent[tetrad, nLoc];
      state = Join[coords0, kInit];
      sf = IntegrateGeodesic[metricFn, state, n, dl, hf];
      kFin = sf[[5 ;; 8]];
      sp = kFin[[2 ;; 4]];
      nrm = Norm[sp];
      If[nrm < 10^-12 || ! AllTrue[sp, NumericQ],
        rgb = fbCol,
        asym = sp / nrm;
        rgb = sky[asym];
        If[doppler,
          f = DopplerFactor[g0, kInit, u, kFin];
          rgb = ApplyDoppler[rgb, f, dPow]
        ];
        If[hMask && Abs[v] > 1,
          rs = Sqrt[(sf[[2]] - (x0 + v sf[[1]]))^2 + sf[[3]]^2 + sf[[4]]^2];
          If[rs < hSafe R, rgb = {0, 0, 0}]
        ]
      ];
      Clip[rgb, {0, 1}]
    ],
    {j, h}, {i, w}
  ];
  Image[frame, "Real32"]
];


(* ============================================================ *)
(* GPU offload via RemoteBatchSubmit                            *)
(* ============================================================ *)

(* Submits a single-frame render to a Wolfram Cloud GPU runtime *)
(* (NVIDIA L40S, 44 GiB).  Returns a RemoteBatchJobObject; call *)
(* job["Result"] when complete to get the rendered Image[].     *)
(*                                                              *)
(* IMPORTANT: this requires (1) a Wolfram One / Service Credits *)
(* account with batch-job entitlement, and (2) `IncludeDefinitions *)
(* -> True` so that all the symbols defined above ship with the *)
(* job.  RemoteMachineClass values come from the Wolfram docs:  *)
(* "GPU1xL40S" (single L40S) or "GPU4xL4" (4x L4 GPUs).         *)

SubmitFrameToGPU[metricName_, params_Association, skyPath_String,
                 frameOpts_ : {}, batchOpts_ : {}] := Module[{job},
  job = RemoteBatchSubmit[
    Module[{sky, img},
      Needs["SkyRender`"];                                  (* if installed *)
      sky = LoadEquirectangularSky[skyPath, "Gain" -> 1.6];
      img = RenderFrame[metricName, params, sky, frameOpts];
      img
    ],
    Sequence @@ batchOpts,
    RemoteMachineClass -> "GPU1xL40S",
    IncludeDefinitions -> True,
    TimeConstraint -> 1800,                                 (* 30 min /frame *)
    RemoteInputFiles -> <|skyPath -> skyPath|>
  ];
  job
];

(* Submit the whole velocity sweep at once.                                  *)
(* Each frame is its own remote job so they parallelise across GPUs (if you *)
(* enable that on the provider side); the local function returns a list of  *)
(* RemoteBatchJobObjects.  Pull results by doing                              *)
(*    frames = #["Result"] & /@ jobs                                          *)
(* once they all show "Status" -> "Completed".                               *)

SubmitVelocitySweepToGPU[metricName_, baseParams_Association,
                         velocities_List, skyPath_String,
                         frameOpts_ : {}, batchOpts_ : {}] :=
  Table[
    SubmitFrameToGPU[
      metricName,
      Append[baseParams, "v0" -> v],
      skyPath, frameOpts, batchOpts
    ],
    {v, velocities}
  ];

(* Convenience: cosine ease-in-out velocity schedule, matching the Python *)
(* notebook's make_velocities.                                             *)
EaseInOutVelocities[nFrames_Integer, vMax_] := With[
  {tt = N[Range[0, nFrames - 1] / (nFrames - 1)]},
  vMax (1 - Cos[Pi tt]) / 2
];


(* ============================================================ *)
(* Quick smoke test (run by Tests.wls)                          *)
(* ============================================================ *)

SmokeTest[] := Module[{img},
  img = RenderFrame["alcubierre",
    <|"v0" -> 0.5, "R" -> 1.0, "sigma" -> 8.0|>,
    ProceduralSky[42, 800],
    {"Width" -> 64, "Height" -> 36, "NSteps" -> 80, "Dlam" -> 0.2}
  ];
  Print["SmokeTest: ", ImageDimensions[img]];
  img
];
