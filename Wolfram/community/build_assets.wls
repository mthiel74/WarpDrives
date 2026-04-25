#!/usr/bin/env wolframscript
(* ----------------------------------------------------------------
   Build all the static images and GIF animations that
   build_notebook.wls embeds into the Wolfram Community notebook
   warpdrives.nb.

   Outputs land in   ./results/   (PNG + GIF).
   Run with         wolframscript -file build_assets.wls
   ---------------------------------------------------------------- *)

hereDir = DirectoryName[If[$ScriptCommandLine =!= {}, $InputFileName,
                           NotebookFileName[]]];
resDir  = FileNameJoin[{hereDir, "results"}];
If[!DirectoryQ[resDir], CreateDirectory[resDir]];

(* Load the consolidated single-file package *)
Get[FileNameJoin[{hereDir, "WarpBubbleSim.wl"}]];

Print["[assets] writing to ", resDir];

(* ----------------------------------------------------------------
   Reusable closed-form Alcubierre quantities.  Using the analytic
   formulas keeps every per-pixel evaluation cheap, so DensityPlot at
   high PlotPoints stays interactive.
   ---------------------------------------------------------------- *)

alcDensity[v_, R_, sigma_, t_, x_, y_, z_] :=
  WarpBubbleSim`AlcubierreAnalyticEnergyDensity[
    {v, R, sigma}, {t, x, y, z}];

alcExpansion[v_, R_, sigma_, t_, x_, y_, z_] :=
  WarpBubbleSim`AlcubierreExpansionScalar[
    {v, R, sigma}, {t, x, y, z}];

alcShiftMag[v_, R_, sigma_, t_, x_, y_] :=
  v WarpBubbleSim`TanhShape[
       Sqrt[(x - v t)^2 + y^2], R, sigma];

(* ----------------------------------------------------------------
   1. Shape functions
   ---------------------------------------------------------------- *)
Print["[1/9] shape functions"];
Export[FileNameJoin[{resDir, "shape_functions.png"}],
  Plot[{WarpBubbleSim`TanhShape[r, 1., 8.],
        WarpBubbleSim`GaussianShape[r, 1., 0.4],
        WarpBubbleSim`CompactPolynomialShape[r, 1., 0.4],
        WarpBubbleSim`SmoothStepShape[r, 1., 0.3]},
       {r, 0, 2.5},
       PlotLegends -> Placed[
         {"Tanh (Alcubierre)", "Gaussian", "Polynomial (compact)", "Smoothstep"},
         {0.7, 0.7}],
       PlotStyle -> {{Thick, ColorData[97][1]},
                     {Thick, ColorData[97][2]},
                     {Thick, ColorData[97][3]},
                     {Thick, ColorData[97][4]}},
       Frame -> True, FrameLabel -> {"r / R", "f(r)"},
       PlotLabel -> Style["Warp-bubble shape functions", Bold, 14],
       PlotRange -> {{0, 2.5}, {-0.05, 1.1}},
       GridLines -> Automatic, GridLinesStyle -> Directive[Gray, Dotted],
       ImageSize -> 720],
  ImageResolution -> 144];

(* ----------------------------------------------------------------
   2. Alcubierre Eulerian energy density (exotic matter map)
   ---------------------------------------------------------------- *)
Print["[2/9] Alcubierre energy density"];
Export[FileNameJoin[{resDir, "alcubierre_density.png"}],
  DensityPlot[
    alcDensity[1., 1., 8., 0., x, y, 0.],
    {x, -2.5, 2.5}, {y, -2.5, 2.5},
    PlotPoints -> 220, ColorFunction -> "TemperatureMap",
    ColorFunctionScaling -> True,
    PlotLegends -> Automatic,
    PlotLabel -> Style[
      "Eulerian energy density (Alcubierre, v=c, R=1, \[Sigma]=8)",
      Bold, 14],
    Frame -> True, FrameLabel -> {"x / R", "y / R"},
    Epilog -> {Black, Thick, Circle[{0, 0}, 1.0]},
    ImageSize -> 720],
  ImageResolution -> 144];

(* ----------------------------------------------------------------
   3. Expansion scalar (space contracts in front, expands behind)
   ---------------------------------------------------------------- *)
Print["[3/9] expansion scalar"];
Export[FileNameJoin[{resDir, "expansion_scalar.png"}],
  DensityPlot[
    alcExpansion[1., 1., 8., 0., x, y, 0.],
    {x, -2.5, 2.5}, {y, -2.5, 2.5},
    PlotPoints -> 220, ColorFunction -> "RedBlueTones",
    ColorFunctionScaling -> True,
    PlotLegends -> Automatic,
    PlotLabel -> Style[
      "Expansion scalar \[Theta] (front contracts, back expands)",
      Bold, 14],
    Frame -> True, FrameLabel -> {"x / R", "y / R"},
    Epilog -> {Black, Thick, Circle[{0, 0}, 1.0]},
    ImageSize -> 720],
  ImageResolution -> 144];

(* ----------------------------------------------------------------
   4. 3D energy-density surface (the bubble wall)
   ---------------------------------------------------------------- *)
Print["[4/9] 3D energy density surface"];
Export[FileNameJoin[{resDir, "energy_density_3d.png"}],
  Plot3D[
    alcDensity[1., 1., 8., 0., x, y, 0.],
    {x, -2.0, 2.0}, {y, -2.0, 2.0},
    PlotPoints -> 80, MaxRecursion -> 1,
    ColorFunction -> Function[{x, y, z}, ColorData["TemperatureMap"][Rescale[z, {-0.0001, 0}]]],
    ColorFunctionScaling -> False,
    Mesh -> None,
    PlotStyle -> Directive[Specularity[White, 30], Opacity[0.92]],
    Lighting -> "Neutral", AxesLabel -> {"x", "y", "\[Rho]"},
    PlotLabel -> Style["Bubble wall: where exotic matter must live",
                       Bold, 14],
    BoxRatios -> {1, 1, 0.55},
    ViewPoint -> {-2.4, -1.6, 1.6},
    ImageSize -> 720],
  ImageResolution -> 144];

(* ----------------------------------------------------------------
   5. Side-by-side comparison of all six metrics
   ---------------------------------------------------------------- *)
Print["[5/9] six-metric comparison panel"];
Export[FileNameJoin[{resDir, "metric_comparison.png"}],
  WarpBubbleSim`PlotMetricComparison[Range -> 2.5, Grid -> 90],
  ImageResolution -> 144];

(* ----------------------------------------------------------------
   6. Grid distortion ANIMATION (bubble swept across the grid)
   ---------------------------------------------------------------- *)
Print["[6/9] grid-distortion animation"];

advectGrid[v_, R_, sigma_, tEnd_, nLines_, nSamp_, nStep_] :=
  Module[{lines, dt = tEnd/nStep, push},
    push[{x_, y_}] := Module[{xc = x, yc = y, t},
      Do[t = (k - 1) dt;
         xc = xc + dt alcShiftMag[v, R, sigma, t, xc, yc],
         {k, 1, nStep}];
      {xc, yc}];
    lines = Join[
      Table[Line[Table[push[{x, yi}], {x, -3, 3, 6/nSamp}]],
            {yi, -3, 3, 6/nLines}],
      Table[Line[Table[push[{xi, y}], {y, -3, 3, 6/nSamp}]],
            {xi, -3, 3, 6/nLines}]];
    lines];

makeGridFrame[v_, R_, sigma_, tEnd_] :=
  Show[
    DensityPlot[
      alcDensity[v, R, sigma, tEnd, x, y, 0.],
      {x, -3, 3}, {y, -2.5, 2.5}, PlotPoints -> 110,
      ColorFunction -> "SunsetColors", ColorFunctionScaling -> True,
      PlotRangePadding -> 0, Frame -> True, FrameLabel -> {"x", "y"},
      PlotLabel -> Style[
        Row[{"Bubble swept across grid  (t = ", NumberForm[tEnd, {3, 2}], ")"}],
        Bold, 13]],
    Graphics[{Lighter[Black, 0.2], Thickness[0.0015],
              advectGrid[v, R, sigma, tEnd, 14, 60, 14]}],
    Graphics[{White, EdgeForm[{Black, Thick}], Disk[{v tEnd, 0}, 0.05]}],
    ImageSize -> {520, 440}];

gridFrames = Table[makeGridFrame[1., 1., 8., t],
                   {t, 0., 2.0, 2.0/29.}];
Export[FileNameJoin[{resDir, "anim_grid_distortion.gif"}],
       gridFrames,
       "DisplayDurations" -> 0.10,
       "AnimationRepetitions" -> Infinity];

(* ----------------------------------------------------------------
   7. Geodesic-bundle animation (test particles caught up by the bubble)
   ---------------------------------------------------------------- *)
Print["[7/9] geodesic-bundle animation"];

(* Numerically integrate static observers being swept along by the
   shift field (Eulerian observer:  dx/dt = -beta^x). *)
swept[v_, R_, sigma_, x0_, y0_, tList_] :=
  Module[{dt, x = x0, y = y0, traj},
    traj = {{0., x0, y0}};
    Do[dt = tList[[k]] - tList[[k - 1]];
       x = x + dt alcShiftMag[v, R, sigma, tList[[k - 1]], x, y];
       AppendTo[traj, {tList[[k]], x, y}],
       {k, 2, Length[tList]}];
    traj];

geoYs    = Range[-1.4, 1.4, 0.35];
tGrid    = Range[0., 4., 0.05];
(* Particles start ahead of the bubble (at x=2, bubble at xs=0) so
   the bubble overtakes them as it moves left-to-right.  The shift
   gives them a +x kick proportional to v_s f(r_s). *)
trajs    = Table[swept[1., 1., 8., 2., y0, tGrid], {y0, geoYs}];

makeGeoFrame[k_] :=
  Show[
    DensityPlot[
      alcShiftMag[1., 1., 8., tGrid[[k]], x, y],
      {x, -1., 6.}, {y, -2., 2.},
      PlotPoints -> 90, ColorFunction -> "BlueGreenYellow",
      ColorFunctionScaling -> True, PlotRangePadding -> 0,
      Frame -> True, FrameLabel -> {"x", "y"},
      PlotLabel -> Style[Row[{"Test particles swept up by warp bubble  (t=",
                              NumberForm[tGrid[[k]], {3, 2}], ")"}],
                         Bold, 13]],
    Graphics[{
      (* trajectories so far *)
      Thickness[0.003], Black,
      Line /@ Table[traj[[1 ;; k, {2, 3}]], {traj, trajs}],
      (* current particle position *)
      Red, PointSize[0.022],
      Point /@ Table[traj[[k, {2, 3}]], {traj, trajs}],
      (* bubble centre *)
      White, EdgeForm[{Black, Thick}],
      Disk[{1.0 tGrid[[k]], 0.}, 0.06]}],
    ImageSize -> {640, 380}];

geoFrames = Table[makeGeoFrame[k], {k, 1, Length[tGrid], 2}];
Export[FileNameJoin[{resDir, "anim_geodesic_bundle.gif"}],
       geoFrames,
       "DisplayDurations" -> 0.08,
       "AnimationRepetitions" -> Infinity];

(* ----------------------------------------------------------------
   8. Density-evolution animation (bubble moving through space)
   ---------------------------------------------------------------- *)
Print["[8/9] bubble-flying-through-space animation"];

makeFlyFrame[t_] :=
  Show[
    DensityPlot[
      alcDensity[1., 1., 8., t, x, y, 0.],
      {x, -3.5, 5.5}, {y, -2.0, 2.0},
      PlotPoints -> 130, ColorFunction -> "TemperatureMap",
      ColorFunctionScaling -> True, PlotRangePadding -> 0,
      Frame -> True, FrameLabel -> {"x", "y"},
      PlotLabel -> Style[Row[{"Alcubierre bubble flying  (t=",
                              NumberForm[t, {3, 2}], ")"}], Bold, 13]],
    Graphics[{Black, Thick, Circle[{1.0 t, 0.}, 1.0]}],
    Graphics[{White, EdgeForm[{Black, Thick}], Disk[{1.0 t, 0.}, 0.07]}],
    ImageSize -> {640, 360}];

flyFrames = Table[makeFlyFrame[t], {t, 0., 4.0, 4.0/29.}];
Export[FileNameJoin[{resDir, "anim_bubble_flying.gif"}],
       flyFrames,
       "DisplayDurations" -> 0.10,
       "AnimationRepetitions" -> Infinity];

(* ----------------------------------------------------------------
   9. 3D wall "rotation" animation (orbit camera around bubble wall)
   ---------------------------------------------------------------- *)
Print["[9/9] 3D rotation animation"];

frame3D[theta_] :=
  Plot3D[
    alcDensity[1., 1., 8., 0., x, y, 0.],
    {x, -1.8, 1.8}, {y, -1.8, 1.8},
    PlotPoints -> 60, MaxRecursion -> 1,
    ColorFunction -> Function[{x, y, z}, ColorData["TemperatureMap"][Rescale[z, {-0.00006, 0}]]],
    ColorFunctionScaling -> False,
    Mesh -> None, Lighting -> "Neutral",
    PlotStyle -> Directive[Specularity[White, 30], Opacity[0.93]],
    Boxed -> False, Axes -> False, BoxRatios -> {1, 1, 0.55},
    ViewPoint -> 2.5 {Cos[theta], Sin[theta], 0.55},
    ViewVertical -> {0, 0, 1},
    ImageSize -> {600, 500},
    PlotLabel -> Style["Bubble wall (3D \[Rho])", Bold, 13]];

rotFrames = Table[frame3D[t], {t, 0, 2 Pi - 2 Pi/24, 2 Pi/24}];
Export[FileNameJoin[{resDir, "anim_density_rotate.gif"}],
       rotFrames,
       "DisplayDurations" -> 0.10,
       "AnimationRepetitions" -> Infinity];

Print["[assets] all done.  Files:"];
Do[Print["  ", FileNameTake[f], "  (",
         IntegerString[Round[FileByteCount[f]/1024]], " KB)"],
   {f, FileNames["*", resDir]}];
