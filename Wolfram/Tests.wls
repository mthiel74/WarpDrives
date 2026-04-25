#!/usr/bin/env wolframscript
(* ::Package:: *)

(* ----------------------------------------------------------------
   Basic sanity checks for the WarpBubbleSim Wolfram port.
   Mirrors a subset of tests/test_minkowski_limit.py.
   ---------------------------------------------------------------- *)

SetDirectory[DirectoryName[If[$InputFileName =!= "", $InputFileName, $Input]]];
Get["WarpBubbleSim.wl"];

pass[label_, cond_] :=
  If[TrueQ[cond],
     Print["  [PASS] ", label],
     Print["  [FAIL] ", label]; $passed = False];

$passed = True;
Print["\n=== WarpBubbleSim tests ==="];

(* --- 1. Minkowski limit: v0 -> 0 gives \[Eta]_{\[Mu]\[Nu]} ------------------------- *)
g0 = AlcubierreMetric[{0, 1, 8}, {t, x, y, z}];
pass["Alcubierre at v=0 reduces to diag(-1,1,1,1)",
     g0 === DiagonalMatrix[{-1, 1, 1, 1}]];

(* --- 2. At bubble center (r_s = 0, f = 1), g_{tt} = -1 + v^2 ----- *)
gCenter = AlcubierreMetric[{1.0, 1.0, 8.0}, {0., 0., 0., 0.}];
pass["Alcubierre g_{tt} at center \[TildeTilde] -1 + v^2 = 0",
     Abs[gCenter[[1, 1]] + 1 - 1.0^2] < 10^-4];

(* --- 3. Determinant of Alcubierre metric = -1 everywhere -------- *)
detSym = Simplify[Det[AlcubierreMetric[{v, R, s}, {t, x, y, z}]]];
pass["Det[Alcubierre] = -1",
     Simplify[detSym + 1] === 0];

(* --- 4. ADM reconstruction round-trip --------------------------- *)
<< "ADM`";
gOrig = AlcubierreMetric[{0.7, 1.2, 5.0}, {0, 0.5, 0.3, 0}];
{alpha, beta, gamma} = MetricToADM[N[gOrig]];
gBack = ConstantArray[0., {4, 4}];
betaLower = gamma . beta;
gBack[[1, 1]] = -alpha^2 + betaLower . beta;
gBack[[1, 2 ;; 4]] = betaLower;
gBack[[2 ;; 4, 1]] = betaLower;
gBack[[2 ;; 4, 2 ;; 4]] = gamma;
pass["ADM \[ShortRightArrow] 4-metric round-trip",
     Max[Abs[Flatten[gBack - N[gOrig]]]] < 10^-8];

(* --- 5. Shape-function limits ----------------------------------- *)
pass["TanhShape[0, R, \[Sigma]] = 1",        Abs[TanhShape[0., 1., 8.] - 1.] < 10^-6];
pass["TanhShape[10R, R, \[Sigma]] \[TildeTilde] 0",  Abs[TanhShape[10., 1., 8.]] < 10^-6];
pass["CompactPolyShape[R+\[Sigma]+1, R, \[Sigma]] = 0",
     CompactPolynomialShape[3., 1., 1.] === 0.];

(* --- 6. Alcubierre analytic \[Rho] < 0 in the wall ---------------- *)
rhoWall = N @ AlcubierreAnalyticEnergyDensity[{1., 1., 8.}, {0, 1.0, 0.3, 0}];
pass["Alcubierre wall \[Rho] is negative (exotic matter)",
     rhoWall < 0];

(* --- 7. Flat-spacetime Christoffel symbols are zero -------------- *)
gFlat = DiagonalMatrix[{-1, 1, 1, 1}];
pass["Christoffel symbols of Minkowski vanish",
     ChristoffelSymbols[gFlat, {t, x, y, z}] === ConstantArray[0, {4, 4, 4}]];

Print["==========================================="];
If[$passed, Print["All tests passed."], Print["Some tests FAILED."]; Exit[1]];
