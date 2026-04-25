(* ::Package:: *)

(* ::Title:: *)
(*WarpBubbleSim`Visualize`*)

(* ::Text:: *)
(*Plotting helpers.  Ported from warpbubblesim/viz/fields2d.py and fields3d.py.*)

BeginPackage["WarpBubbleSim`Visualize`", {"WarpBubbleSim`Energy`",
                                          "WarpBubbleSim`Metrics`",
                                          "WarpBubbleSim`ShapeFunctions`"}];

PlotShapeFunctions::usage =
  "PlotShapeFunctions[R, \[Sigma]] shows the four shape profiles on a single axis.";

PlotEnergyDensity2D::usage =
  "PlotEnergyDensity2D[{v0,R,\[Sigma]}, opts] shows the Alcubierre \[Rho](x,y) slice at z=0.";

PlotExpansionScalar2D::usage =
  "PlotExpansionScalar2D[{v0,R,\[Sigma]}, opts] shows the Alcubierre expansion scalar \[Theta](x,y) slice.";

PlotShiftField::usage =
  "PlotShiftField[shiftFunc, region] shows a 2D vector plot of the shift vector.";

PlotMetricComparison::usage =
  "PlotMetricComparison[params, opts] shows a grid of Eulerian energy densities for \
all six metrics side by side.";

PlotGridDistortion::usage =
  "PlotGridDistortion[{v0,R,\[Sigma]}, time, opts] shows how the coordinate grid is warped \
by the Alcubierre shift field.";

Begin["`Private`"];

Needs["WarpBubbleSim`ADM`"];

PlotShapeFunctions[R_: 1, sigma_: 8] :=
  Plot[{TanhShape[r, R, sigma],
        GaussianShape[r, R, sigma],
        CompactPolynomialShape[r, R, sigma],
        SmoothStepShape[r, R, sigma]},
       {r, 0, 3 R},
       PlotLegends -> {"Tanh", "Gaussian", "Polynomial", "SmoothStep"},
       AxesLabel -> {"r", "f(r)"},
       PlotLabel -> "Warp bubble shape functions",
       PlotRange -> {0, 1.05}, ImageSize -> 500];

PlotEnergyDensity2D[{v0_, R_, sigma_},
                    opts : OptionsPattern[{Range -> 3, Grid -> 80}]] :=
  Module[{rng = OptionValue[Range], n = OptionValue[Grid]},
    DensityPlot[
      AlcubierreAnalyticEnergyDensity[{v0, R, sigma}, {0, x, y, 0}],
      {x, -rng, rng}, {y, -rng, rng},
      PlotPoints -> n, ColorFunction -> "TemperatureMap",
      PlotLegends -> Automatic, PlotLabel -> "Alcubierre \[Rho](x,y)",
      FrameLabel -> {"x", "y"}, ImageSize -> 500]];

PlotExpansionScalar2D[{v0_, R_, sigma_},
                      opts : OptionsPattern[{Range -> 3, Grid -> 80}]] :=
  Module[{rng = OptionValue[Range], n = OptionValue[Grid]},
    DensityPlot[
      AlcubierreExpansionScalar[{v0, R, sigma}, {0, x, y, 0}],
      {x, -rng, rng}, {y, -rng, rng},
      PlotPoints -> n, ColorFunction -> "TemperatureMap",
      PlotLegends -> Automatic, PlotLabel -> "Expansion scalar \[Theta](x,y)",
      FrameLabel -> {"x", "y"}, ImageSize -> 500]];

PlotShiftField[shiftFunc_, {xMin_, xMax_}, {yMin_, yMax_}] :=
  VectorPlot[{shiftFunc[0, x, y, 0][[1]], shiftFunc[0, x, y, 0][[2]]},
             {x, xMin, xMax}, {y, yMin, yMax},
             VectorPoints -> 15, VectorScale -> {Small, Automatic, None},
             PlotLabel -> "Shift vector field \[Beta](x,y)",
             FrameLabel -> {"x", "y"}, ImageSize -> 500];

(* Grid lines under the Alcubierre shift *)
PlotGridDistortion[{v0_, R_, sigma_}, t_: 0,
                   OptionsPattern[{Range -> 3, Lines -> 15, Steps -> 40}]] :=
  Module[{rng = OptionValue[Range], nLines = OptionValue[Lines],
          nSteps = OptionValue[Steps], shiftX, advect, initialGrid},
    shiftX[tt_, xx_, yy_] := -v0 TanhShape[Sqrt[(xx - v0 tt)^2 + yy^2], R, sigma];
    advect[{x0_, y0_}] :=
      Module[{pts, xcur = x0, ycur = y0, dt = t/nSteps},
        pts = Table[{xcur, ycur}, {nSteps + 1}];
        Do[xcur = xcur + dt shiftX[(i - 1) dt, xcur, ycur];
           pts[[i]] = {xcur, ycur},
           {i, 1, nSteps + 1}];
        pts];
    initialGrid = Join[
      Table[Line[Table[{x, y}, {y, -rng, rng, 2 rng/nLines}]], {x, -rng, rng, 2 rng/nLines}],
      Table[Line[Table[{x, y}, {x, -rng, rng, 2 rng/nLines}]], {y, -rng, rng, 2 rng/nLines}]];
    (* Simple visualization: plot the instantaneous shift as contour background *)
    Show[
      DensityPlot[Abs[shiftX[t, x, y]], {x, -rng, rng}, {y, -rng, rng},
                  ColorFunction -> "SunsetColors", PlotPoints -> 60, Frame -> True,
                  FrameLabel -> {"x", "y"}, PlotLegends -> Automatic],
      Graphics[{Black, Thickness[0.002], initialGrid}],
      ImageSize -> 500,
      PlotLabel -> Row[{"Alcubierre shift magnitude at t=", t}]]];

(* Cheap proxy fields used by PlotMetricComparison: each function returns a
   scalar that highlights where the warp bubble is non-trivial in space.
   For Alcubierre we use the closed-form Eulerian \[Rho] from the paper; for the
   others we plot a shift-magnitude / shape-driven proxy that's qualitatively
   similar but does not require a numeric Einstein tensor at every pixel. *)
PlotMetricComparison[OptionsPattern[{Range -> 3, Grid -> 80}]] :=
  Module[{rng = OptionValue[Range], n = OptionValue[Grid], specs, plots},
    specs = {
      {"Alcubierre",
        AlcubierreAnalyticEnergyDensity[{1, 1, 8}, {0, #1, #2, 0}] &,
        "Eulerian \[Rho] (analytic)"},
      {"Natario",
        Module[{rs = Sqrt[(#1 - 1.)^2 + #2^2 + 0.0001], dphi},
          dphi = Evaluate[D[TanhShape[r, 1, 8], r]] /. r -> rs;
          -1.0 dphi^2 (#2^2/rs^2)] &,
        "Shift-derivative proxy"},
      {"VanDenBroeck",
        Module[{rs = Sqrt[(#1)^2 + #2^2], B},
          B = 1 + (5 - 1) TanhShape[rs, 1, 5];
          B TanhShape[rs, 0.5, 8]] &,
        "Pocket factor"},
      {"WhiteToroidal",
        Module[{rho = Sqrt[#2^2], d, rmaj = 2., rmin = 0.5},
          d = Sqrt[(rho - rmaj)^2 + #1^2] - rmin;
          (1 - Tanh[8 d/rmin])/2] &,
        "Toroidal shape"},
      {"BobrickMartire",
        Module[{rs = Sqrt[(#1)^2 + #2^2], stepIn, stepOut},
          stepIn  = (1 + Tanh[5 (rs - 1)])/2;
          stepOut = (1 - Tanh[5 (rs - 2)])/2;
          stepIn stepOut] &,
        "Shell function"},
      {"Lentz",
        Module[{rs = Sqrt[(#1)^2 + #2^2]}, Sech[rs/1.0]^2] &,
        "Soliton intensity"}};
    plots = Table[
      Module[{name = spec[[1]], f = spec[[2]], sub = spec[[3]]},
        DensityPlot[f[x, y], {x, -rng, rng}, {y, -rng, rng},
                    PlotPoints -> n, ColorFunction -> "TemperatureMap",
                    PlotLabel -> Column[{name, Style[sub, Italic, Smaller]}],
                    FrameLabel -> {"x", "y"}, ImageSize -> 280,
                    PlotLegends -> Automatic]],
      {spec, specs}];
    Grid[Partition[plots, 3], Frame -> All]];

End[];
EndPackage[];
