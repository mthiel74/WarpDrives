(* ::Package:: *)

(* ::Title:: *)
(*WarpBubbleSim`ShapeFunctions`*)

(* ::Text:: *)
(*Shape functions f(r) that define warp-bubble geometries.*)
(*f(r) = 1 inside the bubble, 0 outside, with a smooth transition.*)
(*Ported from warpbubblesim/metrics/base.py.*)

BeginPackage["WarpBubbleSim`ShapeFunctions`"];

TanhShape::usage =
  "TanhShape[r, R, \[Sigma]] is Alcubierre's original shape: \
(Tanh[\[Sigma](r+R)] - Tanh[\[Sigma](r-R)])/(2 Tanh[\[Sigma] R]).";

GaussianShape::usage =
  "GaussianShape[r, R, \[Sigma]] is the Gaussian profile Exp[-(r/(R \[Sigma]))^2/2].";

CompactPolynomialShape::usage =
  "CompactPolynomialShape[r, R, \[Sigma]] is the C^2 polynomial (1-(r/(R+\[Sigma]))^2)^3 \
with compact support.";

SmoothStepShape::usage =
  "SmoothStepShape[r, R, \[Sigma]] uses a smoothstep (3t^2-2t^3) over a 2\[Sigma]-wide wall.";

SechShape::usage =
  "SechShape[r, R] = Sech[r/R] — the soliton profile used in Lentz-type drives.";

ShapeFunction::usage =
  "ShapeFunction[name][r, R, \[Sigma]] looks up one of the registered shape functions \
by symbol name: \"tanh\", \"gaussian\", \"polynomial\", \"smoothstep\", \"sech\".";

Begin["`Private`"];

TanhShape[r_, R_, sigma_] :=
  With[{num = Tanh[sigma (r + R)] - Tanh[sigma (r - R)],
        den = 2 Tanh[sigma R]},
    If[Abs[N[den]] < 10^-12,
      If[NumericQ[r] && r < R, 1., 0.],
      num/den]];

GaussianShape[r_, R_, sigma_] := Exp[-(r/(R sigma))^2/2];

CompactPolynomialShape[r_, R_, sigma_] :=
  Module[{Reff = R + sigma, x},
    If[NumericQ[r] && r >= Reff, 0.,
      x = r/Reff; (1 - x^2)^3]];

SmoothStepShape[r_, R_, sigma_] :=
  Module[{delta = sigma, t},
    t = Clip[(r - R + delta)/(2 delta), {0, 1}];
    1 - (3 t^2 - 2 t^3)];

SechShape[r_, R_] := Sech[r/R];

ShapeFunction["tanh"]       = TanhShape;
ShapeFunction["gaussian"]   = GaussianShape;
ShapeFunction["polynomial"] = CompactPolynomialShape;
ShapeFunction["smoothstep"] = SmoothStepShape;
ShapeFunction["sech"]       = Function[{r, R, sigma}, SechShape[r, R]];

End[];
EndPackage[];
