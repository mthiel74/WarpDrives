(* ::Package:: *)

(* ::Title:: *)
(*WarpBubbleSim`ADM`*)

(* ::Text:: *)
(*ADM 3+1 decomposition: ds^2 = -\[Alpha]^2 dt^2 + \[Gamma]_{ij}(dx^i+\[Beta]^i dt)(dx^j+\[Beta]^j dt).*)

BeginPackage["WarpBubbleSim`ADM`"];

MetricToADM::usage =
  "MetricToADM[g] extracts {lapse, shift, spatialMetric} from a 4-metric.";

ADMInverse::usage =
  "ADMInverse[\[Alpha], \[Beta], \[Gamma]] returns g^{\[Mu]\[Nu]} from ADM variables.";

ExtrinsicCurvatureFlat::usage =
  "ExtrinsicCurvatureFlat[\[Beta]func, coords] returns K_{ij} = (\[PartialD]_i \[Beta]_j + \[PartialD]_j \[Beta]_i)/(2) \
for flat spatial metric, \[Alpha]=1.";

ExpansionScalar::usage =
  "ExpansionScalar[K, \[Gamma]inv] = \[Gamma]^{ij} K_{ij}.";

ShearTensor::usage =
  "ShearTensor[K, \[Gamma], \[Gamma]inv] = K - (1/3) \[Gamma] Tr.";

ShiftDivergence::usage =
  "ShiftDivergence[\[Beta]func, coords] returns \[PartialD]_i \[Beta]^i for flat spatial metric.";

NormalVector::usage =
  "NormalVector[\[Alpha], \[Beta]] returns n^{\[Mu]} = (1/\[Alpha])(1, -\[Beta]^i).";

EulerianVelocity::usage =
  "EulerianVelocity[\[Beta]] returns u^{\[Mu]} = (1, -\[Beta]^i) for \[Alpha]=1.";

Begin["`Private`"];

MetricToADM[g_?MatrixQ] :=
  Module[{gamma, gammaInv, betaLower, shift, lapseSq, lapse},
    gamma      = g[[2 ;; 4, 2 ;; 4]];
    gammaInv   = Inverse[gamma];
    betaLower  = g[[1, 2 ;; 4]];
    shift      = gammaInv . betaLower;
    lapseSq    = -g[[1, 1]] + betaLower . shift;
    lapse      = Sqrt[Max[lapseSq, 0]];
    {lapse, shift, gamma}];

ADMInverse[alpha_, beta_?VectorQ, gamma_?MatrixQ] :=
  Module[{gInv, ainvSq = 1/alpha^2, gammaInv = Inverse[gamma]},
    gInv = ConstantArray[0, {4, 4}];
    gInv[[1, 1]] = -ainvSq;
    gInv[[1, 2 ;; 4]] = beta ainvSq;
    gInv[[2 ;; 4, 1]] = beta ainvSq;
    gInv[[2 ;; 4, 2 ;; 4]] = gammaInv - ainvSq Outer[Times, beta, beta];
    gInv];

ExtrinsicCurvatureFlat[shiftFunc_, coords_List] :=
  Module[{dbeta, i, j, beta, spatial = coords[[2 ;; 4]]},
    beta = shiftFunc @@ coords;
    dbeta = Table[D[beta, spatial[[i]]], {i, 3}];
    (* K_{ij} = (\[PartialD]_i \[Beta]_j + \[PartialD]_j \[Beta]_i)/2 *)
    Table[(dbeta[[i, j]] + dbeta[[j, i]])/2, {i, 3}, {j, 3}]];

ExpansionScalar[K_?MatrixQ, gammaInv_?MatrixQ] :=
  Sum[gammaInv[[i, j]] K[[i, j]], {i, 3}, {j, 3}];

ShearTensor[K_?MatrixQ, gamma_?MatrixQ, gammaInv_?MatrixQ] :=
  K - (1/3) gamma ExpansionScalar[K, gammaInv];

ShiftDivergence[shiftFunc_, coords_List] :=
  Module[{beta = shiftFunc @@ coords, spatial = coords[[2 ;; 4]]},
    Sum[D[beta[[i]], spatial[[i]]], {i, 3}]];

NormalVector[alpha_, beta_?VectorQ] := Prepend[-beta/alpha, 1/alpha];

EulerianVelocity[beta_?VectorQ] := Prepend[-beta, 1];

End[];
EndPackage[];
