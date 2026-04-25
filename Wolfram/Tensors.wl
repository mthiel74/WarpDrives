(* ::Package:: *)

(* ::Title:: *)
(*WarpBubbleSim`Tensors`*)

(* ::Text:: *)
(*Christoffel, Riemann, Ricci, Einstein, and Weyl tensors computed via symbolic*)
(*differentiation.  Index conventions: metric signature (-,+,+,+), coordinates*)
(*(t, x, y, z) = (0, 1, 2, 3).*)

BeginPackage["WarpBubbleSim`Tensors`"];

MetricInverse::usage =
  "MetricInverse[g] returns g^{\[Mu]\[Nu]}.";

ChristoffelSymbols::usage =
  "ChristoffelSymbols[g, {t,x,y,z}] returns \[CapitalGamma]^{\[Mu]}_{\[Alpha]\[Beta]} as a 4x4x4 array.";

RiemannTensor::usage =
  "RiemannTensor[g, {t,x,y,z}] returns R^{\[Rho]}_{\[Sigma]\[Mu]\[Nu]} as a 4x4x4x4 array.";

RiemannTensorLower::usage =
  "RiemannTensorLower[g, {t,x,y,z}] returns R_{\[Rho]\[Sigma]\[Mu]\[Nu]}.";

RicciTensor::usage =
  "RicciTensor[g, {t,x,y,z}] returns R_{\[Mu]\[Nu]}.";

RicciScalar::usage =
  "RicciScalar[g, {t,x,y,z}] returns the scalar curvature R.";

EinsteinTensor::usage =
  "EinsteinTensor[g, {t,x,y,z}] returns G_{\[Mu]\[Nu]} = R_{\[Mu]\[Nu]} - (1/2) g_{\[Mu]\[Nu]} R.";

WeylTensor::usage =
  "WeylTensor[g, {t,x,y,z}] returns C_{\[Rho]\[Sigma]\[Mu]\[Nu]}.";

AllTensors::usage =
  "AllTensors[g, {t,x,y,z}] returns an Association containing metric, inverse, \
Christoffel, Riemann (both forms), Ricci, Ricci scalar, Einstein, and Weyl tensors.";

NumericTensors::usage =
  "NumericTensors[metricFunc, {t0,x0,y0,z0}] computes all tensors numerically at a \
specific point.  metricFunc[t,x,y,z] should return a 4x4 matrix.";

Begin["`Private`"];

MetricInverse[g_?MatrixQ] := Inverse[g];
MetricInverse[g_] := Inverse[g];

(* ----  Christoffel \[CapitalGamma]^\[Mu]_{\[Alpha]\[Beta]}  ---- *)
ChristoffelSymbols[g_, coords_List] :=
  Module[{ginv, dg, dim = 4, mu, al, be, nu},
    ginv = Simplify @ Inverse[g];
    (* dg[[\[Alpha],\[Mu],\[Nu]]] = \[PartialD]_\[Alpha] g_{\[Mu]\[Nu]} *)
    dg = Table[D[g, coords[[alpha]]], {alpha, dim}];
    Table[
      (1/2) Sum[ginv[[mu, nu]] (dg[[al, be, nu]] + dg[[be, al, nu]] - dg[[nu, al, be]]),
                {nu, dim}],
      {mu, dim}, {al, dim}, {be, dim}]];

(* ----  Riemann R^\[Rho]_{\[Sigma]\[Mu]\[Nu]}  ---- *)
RiemannTensor[g_, coords_List] :=
  Module[{gamma, dgamma, dim = 4, rho, sig, mu, nu, lam},
    gamma = ChristoffelSymbols[g, coords];
    (* dgamma[[\[Mu],\[Rho],\[Alpha],\[Beta]]] = \[PartialD]_\[Mu] \[CapitalGamma]^\[Rho]_{\[Alpha]\[Beta]} *)
    dgamma = Table[D[gamma, coords[[mu]]], {mu, dim}];
    Table[
      dgamma[[mu, rho, nu, sig]] - dgamma[[nu, rho, mu, sig]]
        + Sum[gamma[[rho, mu, lam]] gamma[[lam, nu, sig]]
              - gamma[[rho, nu, lam]] gamma[[lam, mu, sig]], {lam, dim}],
      {rho, dim}, {sig, dim}, {mu, dim}, {nu, dim}]];

RiemannTensorLower[g_, coords_List] :=
  Module[{R = RiemannTensor[g, coords]},
    (* R_{\[Rho]\[Sigma]\[Mu]\[Nu]} = g_{\[Rho]\[Lambda]} R^\[Lambda]_{\[Sigma]\[Mu]\[Nu]} *)
    TensorContract[TensorProduct[g, R], {{2, 3}}]];

(* ----  Ricci R_{\[Mu]\[Nu]} = R^\[Rho]_{\[Mu]\[Rho]\[Nu]}  ---- *)
RicciTensor[g_, coords_List] :=
  Module[{R = RiemannTensor[g, coords]},
    Table[Sum[R[[rho, mu, rho, nu]], {rho, 4}], {mu, 4}, {nu, 4}]];

RicciScalar[g_, coords_List] :=
  Module[{ginv = Inverse[g], Rmn = RicciTensor[g, coords]},
    Sum[ginv[[mu, nu]] Rmn[[mu, nu]], {mu, 4}, {nu, 4}]];

EinsteinTensor[g_, coords_List] :=
  Module[{Rmn = RicciTensor[g, coords], ginv = Inverse[g], Rscal},
    Rscal = Sum[ginv[[m, n]] Rmn[[m, n]], {m, 4}, {n, 4}];
    Rmn - (1/2) g Rscal];

WeylTensor[g_, coords_List] :=
  Module[{Rlow, Rmn, Rscal, ginv = Inverse[g], C, rho, sig, mu, nu},
    Rlow = RiemannTensorLower[g, coords];
    Rmn  = RicciTensor[g, coords];
    Rscal = Sum[ginv[[m, n]] Rmn[[m, n]], {m, 4}, {n, 4}];
    Table[
      Rlow[[rho, sig, mu, nu]]
        - (1/2) (g[[rho, mu]] Rmn[[sig, nu]] - g[[rho, nu]] Rmn[[sig, mu]]
                 + g[[sig, nu]] Rmn[[rho, mu]] - g[[sig, mu]] Rmn[[rho, nu]])
        + (Rscal/6) (g[[rho, mu]] g[[sig, nu]] - g[[rho, nu]] g[[sig, mu]]),
      {rho, 4}, {sig, 4}, {mu, 4}, {nu, 4}]];

AllTensors[g_, coords_List] :=
  Module[{ginv, gam, Rup, Rlow, Rmn, Rscal, Gmn, Cmn},
    ginv  = Inverse[g];
    gam   = ChristoffelSymbols[g, coords];
    Rup   = RiemannTensor[g, coords];
    Rlow  = TensorContract[TensorProduct[g, Rup], {{2, 3}}];
    Rmn   = Table[Sum[Rup[[rho, mu, rho, nu]], {rho, 4}], {mu, 4}, {nu, 4}];
    Rscal = Sum[ginv[[m, n]] Rmn[[m, n]], {m, 4}, {n, 4}];
    Gmn   = Rmn - (1/2) g Rscal;
    Cmn   = WeylTensor[g, coords];
    <|"Metric" -> g,
      "Inverse" -> ginv,
      "Christoffel" -> gam,
      "Riemann" -> Rup,
      "RiemannLower" -> Rlow,
      "Ricci" -> Rmn,
      "RicciScalar" -> Rscal,
      "Einstein" -> Gmn,
      "Weyl" -> Cmn|>];

(* Numeric shortcut: substitute point then evaluate.  Symbols t,x,y,z are used. *)
NumericTensors[metricFunc_, {t0_?NumericQ, x0_?NumericQ, y0_?NumericQ, z0_?NumericQ},
               OptionsPattern[{Simplify -> False}]] :=
  Module[{t, x, y, z, g, res, rules},
    g = metricFunc[t, x, y, z];
    res = AllTensors[g, {t, x, y, z}];
    rules = {t -> t0, x -> x0, y -> y0, z -> z0};
    res = If[OptionValue[Simplify], Simplify[# /. rules] & /@ res, # /. rules & /@ res];
    N[res]];

End[];
EndPackage[];
