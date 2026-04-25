(* ::Package:: *)

(* ::Title:: *)
(*WarpBubbleSim`Energy`*)

(* ::Text:: *)
(*Stress-energy tensor and its decomposition, obtained from*)
(*T_{\[Mu]\[Nu]} = G_{\[Mu]\[Nu]}/(8\[Pi]).*)

BeginPackage["WarpBubbleSim`Energy`", {"WarpBubbleSim`Tensors`"}];

StressEnergyTensor::usage =
  "StressEnergyTensor[g, coords] = EinsteinTensor[g, coords]/(8 Pi).";

EnergyDensity::usage =
  "EnergyDensity[g, coords, u] = T_{\[Mu]\[Nu]} u^{\[Mu]} u^{\[Nu]}. Default observer = static (1,0,0,0).";

EulerianEnergyDensity::usage =
  "EulerianEnergyDensity[g, coords, shift] = T_{\[Mu]\[Nu]} n^{\[Mu]} n^{\[Nu]} with n the Eulerian \
observer (\[Alpha]=1 assumed).";

MomentumDensity::usage =
  "MomentumDensity[g, coords, u] = -T^{\[Mu]}_{\[Nu]} u^{\[Nu]}.";

PressureDecomposition::usage =
  "PressureDecomposition[g, coords] returns the isotropic pressure and principal pressures.";

EnergyFlux::usage =
  "EnergyFlux[g, coords] = T^{0i}.";

DecomposeStressEnergy::usage =
  "DecomposeStressEnergy[g, coords, u] returns <|\"rho\"->..., \"pressure\"->..., \
\"heatFlux\"->..., \"anisotropicStress\"->...|>.";

AlcubierreAnalyticEnergyDensity::usage =
  "AlcubierreAnalyticEnergyDensity[{v0,R,\[Sigma]}, {t,x,y,z}] returns the closed-form \
Eulerian energy density derived in Alcubierre (1994): -v^2/(32\[Pi]) (df/dr)^2 (\[Rho]^2/r^2).";

AlcubierreExpansionScalar::usage =
  "AlcubierreExpansionScalar[{v0,R,\[Sigma]}, {t,x,y,z}] = v_s (x-x_s)/r_s df/dr_s.";

Begin["`Private`"];

Needs["WarpBubbleSim`ShapeFunctions`"];
Needs["WarpBubbleSim`Metrics`"];

StressEnergyTensor[g_, coords_List] := EinsteinTensor[g, coords]/(8 Pi);

EnergyDensity[g_, coords_List, u_:{1, 0, 0, 0}] :=
  Module[{T = StressEnergyTensor[g, coords]},
    Sum[T[[m, n]] u[[m]] u[[n]], {m, 4}, {n, 4}]];

EulerianEnergyDensity[g_, coords_List, shiftVec_?VectorQ] :=
  Module[{T = StressEnergyTensor[g, coords], n4 = Prepend[-shiftVec, 1]},
    Sum[T[[m, nn]] n4[[m]] n4[[nn]], {m, 4}, {nn, 4}]];

MomentumDensity[g_, coords_List, u_:{1, 0, 0, 0}] :=
  Module[{ginv = Inverse[g], T = StressEnergyTensor[g, coords], Tmixed},
    Tmixed = ginv . T;
    -Tmixed . u];

PressureDecomposition[g_, coords_List] :=
  Module[{ginv = Inverse[g], T = StressEnergyTensor[g, coords], Tmixed, Tsp, piso, pr},
    Tmixed = ginv . T;
    Tsp    = Tmixed[[2 ;; 4, 2 ;; 4]];
    piso   = Tr[Tsp]/3;
    pr     = Eigenvalues[Tsp];
    <|"Isotropic" -> piso, "Principal" -> pr|>];

EnergyFlux[g_, coords_List] :=
  Module[{ginv = Inverse[g], T = StressEnergyTensor[g, coords], Tup},
    Tup = ginv . T . Transpose[ginv];
    Tup[[1, 2 ;; 4]]];

DecomposeStressEnergy[g_, coords_List, u_:{1, 0, 0, 0}] :=
  Module[{ginv = Inverse[g], T = StressEnergyTensor[g, coords],
          uLower, hProj, hUpper, rho, p, hMixed, q, pi},
    uLower = g . u;
    hProj  = g + Outer[Times, uLower, uLower];
    hUpper = ginv + Outer[Times, u, u];
    rho    = Sum[T[[m, n]] u[[m]] u[[n]], {m, 4}, {n, 4}];
    p      = Sum[hUpper[[m, n]] T[[m, n]], {m, 4}, {n, 4}]/3;
    hMixed = IdentityMatrix[4] + Outer[Times, uLower, u];
    q      = -hMixed . T . u;
    pi     = hMixed . T . Transpose[hMixed] - p hProj;
    <|"rho" -> rho, "pressure" -> p, "heatFlux" -> q, "anisotropicStress" -> pi|>];

(* -- Closed-form Alcubierre Eulerian \[Rho] (Alcubierre 1994) -- *)
AlcubierreAnalyticEnergyDensity[{v0_, R_, sigma_}, {t_, x_, y_, z_},
                                OptionsPattern[{X0 -> 0}]] :=
  Module[{x0 = OptionValue[X0], xs, rs, rho2, df, r},
    xs = x0 + v0 t;
    rs = Sqrt[(x - xs)^2 + y^2 + z^2];
    rho2 = y^2 + z^2;
    df = D[TanhShape[r, R, sigma], r] /. r -> rs;
    -(v0^2/(32 Pi)) df^2 (rho2/rs^2)];

AlcubierreExpansionScalar[{v0_, R_, sigma_}, {t_, x_, y_, z_},
                          OptionsPattern[{X0 -> 0}]] :=
  Module[{x0 = OptionValue[X0], xs, rs, df, r},
    xs = x0 + v0 t;
    rs = Sqrt[(x - xs)^2 + y^2 + z^2];
    df = D[TanhShape[r, R, sigma], r] /. r -> rs;
    v0 (x - xs)/rs df];

End[];
EndPackage[];
