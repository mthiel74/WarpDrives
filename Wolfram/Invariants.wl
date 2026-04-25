(* ::Package:: *)

(* ::Title:: *)
(*WarpBubbleSim`Invariants`*)

(* ::Text:: *)
(*Coordinate-invariant scalars built from curvature tensors.*)

BeginPackage["WarpBubbleSim`Invariants`", {"WarpBubbleSim`Tensors`"}];

KretschmannScalar::usage =
  "KretschmannScalar[g, coords] = R_{\[Alpha]\[Beta]\[Gamma]\[Delta]} R^{\[Alpha]\[Beta]\[Gamma]\[Delta]}.";

WeylSquared::usage =
  "WeylSquared[g, coords] = C_{\[Alpha]\[Beta]\[Gamma]\[Delta]} C^{\[Alpha]\[Beta]\[Gamma]\[Delta]}.";

RicciSquared::usage =
  "RicciSquared[g, coords] = R_{\[Mu]\[Nu]} R^{\[Mu]\[Nu]}.";

EulerDensity4D::usage =
  "EulerDensity4D[g, coords] computes the 4D Euler density \
R^2 - 4 R_{\[Mu]\[Nu]} R^{\[Mu]\[Nu]} + R_{\[Alpha]\[Beta]\[Gamma]\[Delta]} R^{\[Alpha]\[Beta]\[Gamma]\[Delta]}.";

Begin["`Private`"];

raiseRiemann[Rlow_, ginv_] :=
  Module[{step1, step2, step3, step4},
    (* raise each index in turn *)
    step1 = Table[Sum[ginv[[a, i]] Rlow[[i, b, c, d]], {i, 4}],
                  {a, 4}, {b, 4}, {c, 4}, {d, 4}];
    step2 = Table[Sum[ginv[[b, i]] step1[[a, i, c, d]], {i, 4}],
                  {a, 4}, {b, 4}, {c, 4}, {d, 4}];
    step3 = Table[Sum[ginv[[c, i]] step2[[a, b, i, d]], {i, 4}],
                  {a, 4}, {b, 4}, {c, 4}, {d, 4}];
    step4 = Table[Sum[ginv[[d, i]] step3[[a, b, c, i]], {i, 4}],
                  {a, 4}, {b, 4}, {c, 4}, {d, 4}];
    step4];

KretschmannScalar[g_, coords_List] :=
  Module[{Rlow = RiemannTensorLower[g, coords], ginv = Inverse[g], Rup},
    Rup = raiseRiemann[Rlow, ginv];
    Sum[Rlow[[a, b, c, d]] Rup[[a, b, c, d]], {a, 4}, {b, 4}, {c, 4}, {d, 4}]];

WeylSquared[g_, coords_List] :=
  Module[{Clow = WeylTensor[g, coords], ginv = Inverse[g], Cup},
    Cup = raiseRiemann[Clow, ginv];
    Sum[Clow[[a, b, c, d]] Cup[[a, b, c, d]], {a, 4}, {b, 4}, {c, 4}, {d, 4}]];

RicciSquared[g_, coords_List] :=
  Module[{Rmn = RicciTensor[g, coords], ginv = Inverse[g], Rup},
    Rup = ginv . Rmn . ginv;
    Sum[Rmn[[m, n]] Rup[[m, n]], {m, 4}, {n, 4}]];

EulerDensity4D[g_, coords_List] :=
  Module[{Rscal = RicciScalar[g, coords],
          R2    = RicciSquared[g, coords],
          K     = KretschmannScalar[g, coords]},
    Rscal^2 - 4 R2 + K];

End[];
EndPackage[];
