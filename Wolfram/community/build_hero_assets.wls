#!/usr/bin/env wolframscript
(* ----------------------------------------------------------------
   Build the "hero" assets that mirror the Python
   generate_hero_animation.py:

     1) anim_hero_3d.gif     -- dark-mode 3D scene: distorted warp
                                surface (cyan wireframe) + translucent
                                bubble + exotic-matter ring + a
                                stylised spacecraft inside the bubble
                                + engine trail + starfield + orbiting
                                camera.
     2) anim_hero_topdown.gif -- top-down 2D view with cyan grid
                                advected by the shift field, the
                                Eulerian energy density as backdrop,
                                a dashed bubble outline, and a
                                triangular spacecraft + nacelles.

   Run:  wolframscript -file build_hero_assets.wls
   Outputs land in ./results/.
   ---------------------------------------------------------------- *)

hereDir = DirectoryName[If[$ScriptCommandLine =!= {}, $InputFileName,
                           NotebookFileName[]]];
resDir  = FileNameJoin[{hereDir, "results"}];
If[!DirectoryQ[resDir], CreateDirectory[resDir]];

Get[FileNameJoin[{hereDir, "WarpBubbleSim.wl"}]];

Print["[hero] writing to ", resDir];

(* ----------------------------------------------------------------
   Closed-form Alcubierre helpers (cheap, vectorisable)
   ---------------------------------------------------------------- *)

shapeFn[r_, R_, sigma_] :=
  WarpBubbleSim`TanhShape[r, R, sigma];

(* Eulerian \[Rho] (analytic) *)
rhoFn[v_, R_, sigma_, t_, x_, y_] :=
  WarpBubbleSim`AlcubierreAnalyticEnergyDensity[
    {v, R, sigma}, {t, x, y, 0.}];

shiftX[v_, R_, sigma_, t_, x_, y_] :=
  -v shapeFn[Sqrt[(x - v t)^2 + y^2], R, sigma];

(* ----------------------------------------------------------------
   Spacecraft mesh, transcribed verbatim from the Python
   create_spacecraft() in generate_hero_animation.py.

   Returns a function pose |-> Graphics3D primitives.
   ---------------------------------------------------------------- *)

hullLength = 0.6; hullWidth = 0.15; hullHeight = 0.08;

shipVerts = {
  {hullLength 0.6, 0,                  0},                  (* 1  nose *)
  {hullLength 0.2, hullWidth 0.3,      hullHeight 0.5},     (* 2 *)
  {hullLength 0.2, -hullWidth 0.3,     hullHeight 0.5},     (* 3 *)
  {hullLength 0.2, -hullWidth 0.3,    -hullHeight 0.5},     (* 4 *)
  {hullLength 0.2, hullWidth 0.3,     -hullHeight 0.5},     (* 5 *)
  {0,             hullWidth 0.5,       hullHeight 0.7},     (* 6 *)
  {0,            -hullWidth 0.5,       hullHeight 0.7},     (* 7 *)
  {0,            -hullWidth 0.5,      -hullHeight 0.5},     (* 8 *)
  {0,             hullWidth 0.5,      -hullHeight 0.5},     (* 9 *)
  {-hullLength 0.3, hullWidth 0.4,     hullHeight 0.5},     (*10 *)
  {-hullLength 0.3, -hullWidth 0.4,    hullHeight 0.5},     (*11 *)
  {-hullLength 0.3, -hullWidth 0.4,   -hullHeight 0.4},     (*12 *)
  {-hullLength 0.3, hullWidth 0.4,    -hullHeight 0.4},     (*13 *)
  {-hullLength 0.5, hullWidth 0.25,    hullHeight 0.3},     (*14 *)
  {-hullLength 0.5, -hullWidth 0.25,   hullHeight 0.3},     (*15 *)
  {-hullLength 0.5, -hullWidth 0.25,  -hullHeight 0.2},     (*16 *)
  {-hullLength 0.5, hullWidth 0.25,   -hullHeight 0.2}};    (*17 *)

(* Triangular faces (1-indexed for Wolfram) *)
shipFaces = {
  (* nose cone *)
  {1,2,3},{1,3,4},{1,4,5},{1,5,2},
  (* front->mid top/bottom/sides *)
  {2,6,7},{2,7,3},
  {4,8,9},{4,9,5},
  {3,7,8},{3,8,4},
  {5,9,6},{5,6,2},
  (* mid->rear top/bottom/sides *)
  {6,10,11},{6,11,7},
  {8,12,13},{8,13,9},
  {7,11,12},{7,12,8},
  {9,13,10},{9,10,6},
  (* rear->engine *)
  {10,14,15},{10,15,11},
  {12,16,17},{12,17,13},
  {11,15,16},{11,16,12},
  {13,17,14},{13,14,10},
  (* engine back *)
  {14,17,16},{14,16,15}};

(* Nacelles: each is a flat quad (z=0.03) sitting alongside the hull *)
shipNacelles = {
  {{-0.1,  hullWidth 0.5, 0}, {-0.1,  hullWidth 0.8, 0},
   {-0.3,  hullWidth 0.8, 0.03}, {-0.3,  hullWidth 0.5, 0.03}},
  {{-0.1, -hullWidth 0.5, 0}, {-0.1, -hullWidth 0.8, 0},
   {-0.3, -hullWidth 0.8, 0.03}, {-0.3, -hullWidth 0.5, 0.03}}};

(* Spacecraft as Graphics3D primitives.
   pose = {xc, yc, zc};  scale uniform *)
spacecraft3D[{xc_, yc_, zc_}, scale_:0.8] :=
  Module[{V},
    V = scale # + {xc, yc, zc} & /@ shipVerts;
    {
      (* Hull *)
      EdgeForm[{Thin, GrayLevel[0.35]}],
      RGBColor[0.78, 0.78, 0.82], Specularity[White, 30],
      GraphicsComplex[V, Polygon[shipFaces]],
      (* Nacelles -- each is a flat quad pair, drawn separately *)
      EdgeForm[{Thin, RGBColor[0.2, 0.4, 1]}],
      Glow[RGBColor[0.25, 0.5, 1]], RGBColor[0.25, 0.5, 1],
      Polygon[scale # + {xc, yc, zc} & /@ #] & /@ shipNacelles,
      (* Engine glow *)
      Glow[Cyan], Cyan,
      Sphere[{xc - 0.3 scale, yc, zc}, 0.05 scale]}];

(* Spacecraft top-down silhouette: a triangle + two nacelle dots
   + an engine glow.  Returns Graphics primitives. *)
spacecraft2D[xc_, scale_:0.4] :=
  {
    EdgeForm[{Thin, White}],
    RGBColor[0.78, 0.78, 0.82],
    Polygon[{{xc + scale, 0},
             {xc - scale 0.5,  scale 0.4},
             {xc - scale 0.5, -scale 0.4}}],
    (* Nacelles *)
    Glow[RGBColor[0.25, 0.5, 1]], RGBColor[0.25, 0.5, 1],
    Disk[{xc - scale 0.3,  scale 0.5}, 0.07],
    Disk[{xc - scale 0.3, -scale 0.5}, 0.07],
    (* Engine glow *)
    Glow[Cyan], Cyan,
    Disk[{xc - scale 0.5, 0}, 0.10]};

(* ----------------------------------------------------------------
   Starfield (consistent across frames)
   ---------------------------------------------------------------- *)

SeedRandom[42];
nStars = 220;
starField3D = Table[{RandomReal[{-8., 16.}],
                     RandomReal[{-8., 8.}],
                     RandomReal[{-2., 2.}],
                     RandomReal[{0.5, 3.}]}, {nStars}];
starGraphics3D := {White,
  PointSize /@ Table[Tooltip[Point[{s[[1]], s[[2]], s[[3]]}], "*"],
                     {s, starField3D}],
  PointSize[0.003],
  Point[{#[[1]], #[[2]], #[[3]]}] & /@ starField3D};

(* Simpler: just fixed-size points *)
starGraphics3D = {White, PointSize[0.003],
  Point[{#[[1]], #[[2]], #[[3]]}] & /@ starField3D};

(* ----------------------------------------------------------------
   1) HERO 3D ANIMATION
   ---------------------------------------------------------------- *)

heroParams = <|"v" -> 1.0, "R" -> 1.5, "sigma" -> 6.0|>;
nFramesHero = 60;
tEndHero = 7.0;
tsHero = Subdivide[0., tEndHero, nFramesHero - 1];

(* z-displacement of the warp surface (cosmetic visualisation):
   z(x,y) = -f(r) * 0.8 + beta_x * 0.3 *)
warpZ[v_, R_, sigma_, t_, x_, y_] :=
  Module[{rs = Sqrt[(x - v t)^2 + y^2], f, bx},
    f  = shapeFn[rs, R, sigma];
    bx = -v f;
    -0.8 f + 0.3 bx];

(* Sample the warp surface on a grid and emit Line[] primitives in
   the cyan style of the Python original. *)
heroSurface[t_, OptionsPattern[{NLines -> 22, NSamp -> 60}]] :=
  Module[{v = heroParams["v"], R = heroParams["R"], sg = heroParams["sigma"],
          xb, xs, ys, nL = OptionValue[NLines], nS = OptionValue[NSamp],
          z, lines, alpha},
    xb = v t;
    xs = Subdivide[xb - 5., xb + 5., nS - 1];
    ys = Subdivide[-3., 3., nS - 1];
    z[i_, j_] := warpZ[v, R, sg, t, xs[[j]], ys[[i]]];
    alpha[i_, j_] :=
      0.30 + 0.60 shapeFn[Sqrt[(xs[[j]] - xb)^2 + ys[[i]]^2], R, sg];
    lines = {};
    (* x-direction lines (constant y) *)
    Do[
      With[{idx = Round[i (nS - 1)/(nL - 1) + 1]},
        lines = Join[lines,
          Flatten[Table[
            {Hue[0.5, 1, 1, alpha[idx, j]],
             Line[{{xs[[j]], ys[[idx]], z[idx, j]},
                   {xs[[j + 1]], ys[[idx]], z[idx, j + 1]}}]},
            {j, 1, nS - 1}], 1]]],
      {i, 0, nL - 1}];
    (* y-direction lines (constant x) *)
    Do[
      With[{idx = Round[j (nS - 1)/(nL - 1) + 1]},
        lines = Join[lines,
          Flatten[Table[
            {Hue[0.5, 1, 1, alpha[i, idx]],
             Line[{{xs[[idx]], ys[[i]], z[i, idx]},
                   {xs[[idx]], ys[[i + 1]], z[i + 1, idx]}}]},
            {i, 1, nS - 1}], 1]]],
      {j, 0, nL - 1}];
    {Thickness[0.0015], lines}];

(* Translucent bubble surface (slightly flattened) *)
heroBubble[t_] :=
  Module[{v = heroParams["v"], R = heroParams["R"], sg = heroParams["sigma"],
          xb},
    xb = v t;
    {Glow[RGBColor[0.2, 0.6, 1, 0.18]],
     EdgeForm[None],
     Specularity[White, 100],
     RGBColor[0.2, 0.6, 1, 0.16],
     Sphere[{xb, 0, 0}, R],
     (* exotic-matter ring slightly below z=0 *)
     Glow[RGBColor[1.0, 0.2, 0.5, 0.55]],
     RGBColor[1.0, 0.2, 0.5, 0.45],
     With[{rIn = R 0.85, rOut = R 1.25, dphi = 2 Pi/64},
       Polygon[
         Table[{
           {xb + rIn  Cos[phi],        rIn  Sin[phi],        -0.2},
           {xb + rIn  Cos[phi + dphi], rIn  Sin[phi + dphi], -0.2},
           {xb + rOut Cos[phi + dphi], rOut Sin[phi + dphi], -0.2},
           {xb + rOut Cos[phi],        rOut Sin[phi],        -0.2}},
           {phi, 0, 2 Pi - dphi, dphi}]]]}];

(* Engine wake: line of fading cyan dots behind the spacecraft *)
heroTrail[t_] :=
  Module[{v = heroParams["v"], xb, n = 24, xs, alphas},
    xb = v t;
    xs = Range[xb - 0.6, xb - 3.5, -(3.5 - 0.6)/(n - 1)];
    alphas = Range[n] / N[n];
    {Glow[Cyan],
     MapIndexed[
       {Cyan, Opacity[(1 - First[#2]/N[n])],
        Sphere[{#1, 0, 0}, Max[0.005, 0.05 (1 - First[#2]/N[n])]]} &,
       xs]}];

heroFrame[t_] :=
  Module[{v = heroParams["v"], xb = heroParams["v"] t,
          elev, azim, vp},
    elev = 25 Degree + 5 Degree Sin[0.05 t / (tEndHero/nFramesHero)];
    azim = -60 Degree + 20 Degree Sin[0.03 t / (tEndHero/nFramesHero)];
    vp = 6.5 {Cos[azim] Cos[elev], Sin[azim] Cos[elev], Sin[elev]};
    Graphics3D[
      {
        (* dark background -- handled by Background option *)
        starGraphics3D,
        heroSurface[t, NLines -> 22, NSamp -> 50],
        heroBubble[t],
        spacecraft3D[{xb, 0, 0.05}, 0.9],
        heroTrail[t]
      },
      Background -> Black,
      Boxed -> False, Axes -> False,
      Lighting -> {{"Ambient", GrayLevel[0.45]},
                   {"Directional", White, ImageScaled[{0.5, -1, 1}]}},
      PlotRange -> {{xb - 6, xb + 4}, {-5, 5}, {-2, 2}},
      ViewVector -> {{xb, 0, 0} + vp, {xb - 0.5, 0, 0}},
      ViewVertical -> {0, 0, 1},
      ViewAngle -> 35 Degree,
      ImageSize -> {640, 360},
      PlotLabel ->
        Style[Row[{"Alcubierre Warp Drive   |   v = c   |   t = ",
                   NumberForm[t, {3, 1}]}],
              White, Bold, 14, FontFamily -> "Helvetica"]]];

Print["[1/2] hero 3D animation (", nFramesHero, " frames)..."];
heroFrames = Table[
  If[Mod[k, 10] === 0, Print["       frame ", k, "/", nFramesHero]];
  heroFrame[tsHero[[k]]],
  {k, 1, nFramesHero}];

Export[FileNameJoin[{resDir, "anim_hero_3d.gif"}],
       heroFrames,
       "DisplayDurations" -> 0.07,
       "AnimationRepetitions" -> Infinity];

Print["       wrote anim_hero_3d.gif (",
      Round[FileByteCount[FileNameJoin[{resDir, "anim_hero_3d.gif"}]]/1024],
      " KB)"];

(* ----------------------------------------------------------------
   2) HERO TOP-DOWN ANIMATION
   ---------------------------------------------------------------- *)

topParams = <|"v" -> 1.0, "R" -> 1.5, "sigma" -> 6.0|>;
nFramesTop = 50;
tEndTop = 6.0;
tsTop = Subdivide[0., tEndTop, nFramesTop - 1];

(* Distorted grid line: each (x, y_const) becomes (x + 0.5 beta, y) *)
distortHorizLines[t_, nLines_, nSamp_] :=
  Module[{v = topParams["v"], R = topParams["R"], sg = topParams["sigma"],
          xs, ys, lines},
    xs = Subdivide[-4., 12., nSamp - 1];
    ys = Subdivide[-4., 4., nLines - 1];
    Table[
      With[{y = ys[[k]]},
        With[{pts = {# + 0.5 shiftX[v, R, sg, t, #, y], y} & /@ xs,
              alphas = (0.30 + 0.60 shapeFn[Sqrt[(# - v t)^2 + y^2], R, sg]) & /@ xs},
          Table[{Hue[0.5, 1, 1, alphas[[j]]],
                 Line[{pts[[j]], pts[[j + 1]]}]},
                {j, 1, nSamp - 1}]]],
      {k, 1, nLines}]];

distortVertLines[t_, nLines_, nSamp_] :=
  Module[{v = topParams["v"], R = topParams["R"], sg = topParams["sigma"],
          xs, ys, lines},
    xs = Subdivide[-4., 12., nLines - 1];
    ys = Subdivide[-4., 4., nSamp - 1];
    Table[
      With[{x = xs[[k]]},
        With[{pts = {x + 0.5 shiftX[v, R, sg, t, x, #], #} & /@ ys,
              alphas = (0.30 + 0.60 shapeFn[Sqrt[(x - v t)^2 + #^2], R, sg]) & /@ ys},
          Table[{Hue[0.5, 1, 1, alphas[[j]]],
                 Line[{pts[[j]], pts[[j + 1]]}]},
                {j, 1, nSamp - 1}]]],
      {k, 1, nLines}]];

topFrame[t_] :=
  Module[{v = topParams["v"], R = topParams["R"], sg = topParams["sigma"],
          xb},
    xb = v t;
    Show[
      DensityPlot[
        rhoFn[v, R, sg, t, x, y],
        {x, -4, 12}, {y, -4, 4},
        PlotPoints -> 100,
        ColorFunction -> (Blend[{RGBColor[1.0, 0.15, 0.30],
                                 RGBColor[0.0, 0.05, 0.15]}, #] &),
        ColorFunctionScaling -> True,
        PlotRangePadding -> 0,
        Frame -> True, FrameStyle -> Directive[GrayLevel[0.55]],
        FrameLabel -> {Style["x  (direction of travel)", GrayLevel[0.7]],
                       Style["y", GrayLevel[0.7]]},
        FrameTicksStyle -> Directive[GrayLevel[0.55], FontSize -> 10],
        Background -> Black,
        AspectRatio -> 4/8],
      Graphics[
        {Thickness[0.0014],
         distortHorizLines[t, 22, 50],
         distortVertLines[t, 32, 50],
         (* dashed bubble outline *)
         White, Thick, Dashing[{0.012, 0.012}], Opacity[0.7],
         Circle[{xb, 0}, R],
         Opacity[1],
         (* spacecraft *)
         spacecraft2D[xb, 0.45],
         (* annotations *)
         Style[Text["Space\ncontracts", {xb + 2.0, 1.0}, {0, 0}],
               White, FontSize -> 11, Opacity[0.8]],
         Style[Text["Space\nexpands", {xb - 2.6, 1.0}, {0, 0}],
               White, FontSize -> 11, Opacity[0.8]]}],
      ImageSize -> {760, 380},
      PlotLabel ->
        Style[Row[{"Warp Bubble \[Dash] Top View   |   v = c   |   t = ",
                   NumberForm[t, {3, 1}]}],
              White, Bold, 13, FontFamily -> "Helvetica"]]];

Print["[2/2] hero top-down animation (", nFramesTop, " frames)..."];
topFrames = Table[
  If[Mod[k, 10] === 0, Print["       frame ", k, "/", nFramesTop]];
  topFrame[tsTop[[k]]],
  {k, 1, nFramesTop}];

Export[FileNameJoin[{resDir, "anim_hero_topdown.gif"}],
       topFrames,
       "DisplayDurations" -> 0.08,
       "AnimationRepetitions" -> Infinity];

Print["       wrote anim_hero_topdown.gif (",
      Round[FileByteCount[FileNameJoin[{resDir, "anim_hero_topdown.gif"}]]/1024],
      " KB)"];

Print["[hero] done."];
