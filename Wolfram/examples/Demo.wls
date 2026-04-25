#!/usr/bin/env wolframscript
(* ::Package:: *)

(* ----------------------------------------------------------------
   WarpBubbleSim demo script
   ----------------------------------------------------------------
   Run with:
     wolframscript -file Demo.wls
   This produces PNG outputs alongside the script and prints a few
   numerical diagnostics to the terminal.
   ---------------------------------------------------------------- *)

SetDirectory[DirectoryName[If[$InputFileName =!= "", $InputFileName, $Input]]];
Get["../WarpBubbleSim.wl"];

Print["--------------------------------------------------------"];
Print["WarpBubbleSim demo"];
Print["--------------------------------------------------------"];

(* ------------  Metric at a point ---------------- *)
gAlc = AlcubierreMetric[{1., 1., 8.}, {t, x, y, z}];
Print["Alcubierre metric at (0, 0, 0, 0) = flat Minkowski shifted:"];
Print[MatrixForm[gAlc /. {t -> 0, x -> 0, y -> 0, z -> 0}]];

(* ------------  Symbolic Einstein tensor ---------- *)
Print["\nComputing Einstein tensor symbolically (may take a moment)..."];
Gmn = EinsteinTensor[gAlc, {t, x, y, z}];
Print["Einstein tensor at (0, 1.0, 0.2, 0) evaluated numerically:"];
Print[MatrixForm[N[Chop[Gmn /. {t -> 0, x -> 1.0, y -> 0.2, z -> 0}]]]];

(* ------------  Analytic \[Rho] formula from Alcubierre 1994 ---------- *)
rho = AlcubierreAnalyticEnergyDensity[{1., 1., 8.}, {0, 1.0, 0.2, 0}];
Print["\nEulerian \[Rho] at (1.0, 0.2, 0) from Alcubierre's closed form = ", N[rho]];

(* ------------  Energy conditions at a wall point ---------- *)
Print["\nChecking classical energy conditions at (x=1, y=0.2, z=0):"];
conds = CheckAllConditions[gAlc, {t, x, y, z}] /.
        {t -> 0, x -> 1.0, y -> 0.2, z -> 0};
KeyValueMap[
  (Print["  ", #1, " : satisfied=", #2[[1]], ",  min=", NumberForm[#2[[2]], 4]]) &,
  conds];

(* ------------  Render figures ---------- *)
out = FileNameJoin[{Directory[], "out"}];
If[!DirectoryQ[out], CreateDirectory[out]];

Print["\nRendering figures to ", out, " ..."];

Export[FileNameJoin[{out, "shape_functions.png"}],
       PlotShapeFunctions[1.0, 8.0],
       ImageResolution -> 150];

Export[FileNameJoin[{out, "energy_density_2d.png"}],
       PlotEnergyDensity2D[{1.0, 1.0, 8.0}, Range -> 3, Grid -> 120],
       ImageResolution -> 150];

Export[FileNameJoin[{out, "expansion_scalar_2d.png"}],
       PlotExpansionScalar2D[{1.0, 1.0, 8.0}, Range -> 3, Grid -> 120],
       ImageResolution -> 150];

Export[FileNameJoin[{out, "metric_comparison.png"}],
       PlotMetricComparison[Range -> 2.5, Grid -> 30],
       ImageResolution -> 150];

Print["Done.  Wrote:"];
Do[Print["  ", FileNameJoin[{out, f}]],
   {f, {"shape_functions.png", "energy_density_2d.png",
        "expansion_scalar_2d.png", "metric_comparison.png"}}];
