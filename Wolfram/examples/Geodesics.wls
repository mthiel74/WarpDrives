#!/usr/bin/env wolframscript
(* ::Package:: *)

(* ----------------------------------------------------------------
   Geodesic bundle through an Alcubierre warp bubble.
   Ports notebooks/01_alcubierre.ipynb style examples.
   ---------------------------------------------------------------- *)

SetDirectory[DirectoryName[If[$InputFileName =!= "", $InputFileName, $Input]]];
Get["../WarpBubbleSim.wl"];

metric = Function[{tt, xx, yy, zz},
  AlcubierreMetric[{1., 1., 8.}, {tt, xx, yy, zz}]];

(* Bundle of static particles at various transverse distances, swept by the bubble *)
x0    = {0., -3., 0., 0.};
ys    = {-0.5, 0., 0.3, 0.6, 1.2};
us    = Table[{1., 0., 0., 0.}, {Length[ys]}];
starts = Table[{0., -3., y, 0.}, {y, ys}];

Print["Integrating ", Length[ys], " test-particle geodesics..."];
results = Table[
  IntegrateGeodesic[metric, starts[[k]], us[[k]], {0, 8}],
  {k, Length[ys]}];

(* Plot the bundle in the (t, x)-plane *)
paths = Table[
  ParametricPlot[Evaluate[{res["Coords"][l][[1]], res["Coords"][l][[2]]}],
                 {l, 0, 8}],
  {res, results}];

fig = Show[paths,
           PlotRange -> {{0, 8}, {-3.5, 6}},
           FrameLabel -> {"t", "x"},
           PlotLabel -> "Alcubierre geodesic bundle (transverse distances)",
           Frame -> True, ImageSize -> 600];

out = FileNameJoin[{Directory[], "out"}];
If[!DirectoryQ[out], CreateDirectory[out]];
Export[FileNameJoin[{out, "geodesics.png"}], fig, ImageResolution -> 150];
Print["Wrote ", FileNameJoin[{out, "geodesics.png"}]];
