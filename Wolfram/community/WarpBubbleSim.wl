(* ::Package:: *)

(* ::Title:: *)
(*WarpBubbleSim  \[Dash]  single-file Wolfram Language package*)

(* ::Text:: *)
(*Self-contained consolidation of the 9 sub-packages*)
(*    ShapeFunctions, Metrics, Tensors, ADM, Energy, Conditions,*)
(*    Invariants, Geodesics, Visualize*)
(*so the user only has to load this one file:*)
(**)
(*    Get["WarpBubbleSim.wl"]*)
(**)
(*Conventions:*)
(*  - Metric signature (-,+,+,+)*)
(*  - Coordinate ordering {t, x, y, z} = {0, 1, 2, 3}*)
(*  - Geometric units G = c = 1.*)
(**)
(*All public symbols live in the WarpBubbleSim` context, so a notebook*)
(*can call AlcubierreMetric[...], EinsteinTensor[...], etc. without*)
(*qualifying any sub-context.*)

BeginPackage["WarpBubbleSim`"];

(* ============================================================
   Public API
   ============================================================ *)

(* ---- Shape functions --------------------------------------- *)
TanhShape::usage =
  "TanhShape[r, R, \[Sigma]] is Alcubierre's original shape: \
(Tanh[\[Sigma](r+R)] - Tanh[\[Sigma](r-R)])/(2 Tanh[\[Sigma] R]).";
GaussianShape::usage =
  "GaussianShape[r, R, \[Sigma]] is Exp[-(r/(R \[Sigma]))^2/2].";
CompactPolynomialShape::usage =
  "CompactPolynomialShape[r, R, \[Sigma]] is the C^2 polynomial \
(1-(r/(R+\[Sigma]))^2)^3 with compact support.";
SmoothStepShape::usage =
  "SmoothStepShape[r, R, \[Sigma]] uses a smoothstep over a 2\[Sigma]-wide wall.";
SechShape::usage =
  "SechShape[r, R] = Sech[r/R] \[Dash] the soliton profile used in Lentz drives.";
ShapeFunction::usage =
  "ShapeFunction[name][r, R, \[Sigma]] looks up one of the registered shape \
functions by symbol name: \"tanh\", \"gaussian\", \"polynomial\", \"smoothstep\", \"sech\".";

(* ---- Metrics ----------------------------------------------- *)
ADMToMetric::usage =
  "ADMToMetric[\[Alpha], \[Beta]vec, \[Gamma]mat] builds the 4-metric from ADM variables.";
BubbleCenter::usage =
  "BubbleCenter[v0, x0, t] returns x0 + v0 t.";
RFromCenter::usage =
  "RFromCenter[{t,x,y,z}, v0, x0] = Sqrt[(x-x0-v0 t)^2 + y^2 + z^2].";
AlcubierreMetric::usage =
  "AlcubierreMetric[{v0, R, \[Sigma]}, {t, x, y, z}] returns g_{\[Mu]\[Nu]} for the 1994 metric.";
NatarioMetric::usage =
  "NatarioMetric[{v0, R, \[Sigma]}, {t, x, y, z}] returns g_{\[Mu]\[Nu]} for Nat\[AAcute]rio's \
expansion-free warp drive.";
VanDenBroeckMetric::usage =
  "VanDenBroeckMetric[{v0, Rext, Rint, Bint, \[Sigma], \[Sigma]B}, {t, x, y, z}] returns \
g_{\[Mu]\[Nu]} for Van Den Broeck's pocket geometry.";
WhiteToroidalMetric::usage =
  "WhiteToroidalMetric[{v0, Rmajor, Rminor, \[Sigma]}, {t, x, y, z}] returns g_{\[Mu]\[Nu]} \
for the toroidal energy distribution.";
BobrickMartireMetric::usage =
  "BobrickMartireMetric[{v0, Rinner, Router, amp, \[Sigma]}, {t, x, y, z}] returns \
g_{\[Mu]\[Nu]} for the subluminal warp-shell construction.";
LentzMetric::usage =
  "LentzMetric[{v0, R, amp, h}, {t, x, y, z}] returns g_{\[Mu]\[Nu]} for Lentz's soliton drive.";
MetricRegistry::usage =
  "MetricRegistry[name] returns the metric function for: \"alcubierre\", \"natario\", \
\"vdbroek\", \"white\", \"bobrick_martire\", \"lentz\".";

(* ---- Tensors ----------------------------------------------- *)
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
  "EinsteinTensor[g, {t,x,y,z}] = R_{\[Mu]\[Nu]} - (1/2) g_{\[Mu]\[Nu]} R.";
WeylTensor::usage =
  "WeylTensor[g, {t,x,y,z}] returns C_{\[Rho]\[Sigma]\[Mu]\[Nu]}.";
AllTensors::usage =
  "AllTensors[g, {t,x,y,z}] returns an Association of all standard GR tensors.";
NumericTensors::usage =
  "NumericTensors[metricFunc, {t0,x0,y0,z0}] computes all tensors numerically at a point.";

(* ---- ADM --------------------------------------------------- *)
MetricToADM::usage =
  "MetricToADM[g] extracts {lapse, shift, spatialMetric}.";
ADMInverse::usage =
  "ADMInverse[\[Alpha], \[Beta], \[Gamma]] returns g^{\[Mu]\[Nu]} from ADM variables.";
ExtrinsicCurvatureFlat::usage =
  "ExtrinsicCurvatureFlat[\[Beta]func, coords] = -(\[PartialD]_i \[Beta]_j + \[PartialD]_j \[Beta]_i)/2 \
for flat \[Gamma]_{ij}, \[Alpha]=1 (Wald sign so that trace K = Eulerian \[Theta]).";
ExpansionScalar::usage =
  "ExpansionScalar[K, \[Gamma]inv] = \[Gamma]^{ij} K_{ij}.";
ShearTensor::usage =
  "ShearTensor[K, \[Gamma], \[Gamma]inv] = K - (1/3) \[Gamma] Tr.";
ShiftDivergence::usage =
  "ShiftDivergence[\[Beta]func, coords] returns \[PartialD]_i \[Beta]^i.";
NormalVector::usage =
  "NormalVector[\[Alpha], \[Beta]] returns n^{\[Mu]} = (1/\[Alpha])(1, -\[Beta]^i).";
EulerianVelocity::usage =
  "EulerianVelocity[\[Beta]] returns u^{\[Mu]} = (1, -\[Beta]^i).";

(* ---- Energy ------------------------------------------------ *)
StressEnergyTensor::usage =
  "StressEnergyTensor[g, coords] = G_{\[Mu]\[Nu]}/(8 Pi).";
EnergyDensity::usage =
  "EnergyDensity[g, coords, u] = T_{\[Mu]\[Nu]} u^{\[Mu]} u^{\[Nu]}.";
EulerianEnergyDensity::usage =
  "EulerianEnergyDensity[g, coords] returns T_{\[Mu]\[Nu]} n^{\[Mu]} n^{\[Nu]} with n the Eulerian observer; \
lapse and shift are extracted from g via MetricToADM so the result is correct \
for any ADM metric.  EulerianEnergyDensity[g, coords, shift] is kept for \
backward compatibility and assumes \[Alpha]=1.";
MomentumDensity::usage =
  "MomentumDensity[g, coords, u] = -T^{\[Mu]}_{\[Nu]} u^{\[Nu]}.";
PressureDecomposition::usage =
  "PressureDecomposition[g, coords] returns isotropic + principal pressures.";
EnergyFlux::usage =
  "EnergyFlux[g, coords] = T^{0i}.";
DecomposeStressEnergy::usage =
  "DecomposeStressEnergy[g, coords, u] returns <|rho, pressure, heatFlux, anisotropicStress|>.";
AlcubierreAnalyticEnergyDensity::usage =
  "AlcubierreAnalyticEnergyDensity[{v0,R,\[Sigma]}, {t,x,y,z}] returns the closed-form \
Eulerian \[Rho] from Alcubierre (1994).";
AlcubierreExpansionScalar::usage =
  "AlcubierreExpansionScalar[{v0,R,\[Sigma]}, {t,x,y,z}] returns the analytic expansion scalar.";

(* ---- Conditions -------------------------------------------- *)
TimelikeVelocities::usage =
  "TimelikeVelocities[n, g] samples n normalised timelike 4-velocities.";
NullVectors::usage =
  "NullVectors[n, g] samples n null 4-vectors of the metric g (each k satisfies g.k.k = 0).";
CheckWEC::usage =
  "CheckWEC[g, coords] -> {satisfied, min}.";
CheckNEC::usage =
  "CheckNEC[g, coords] -> {satisfied, min}.";
CheckSEC::usage =
  "CheckSEC[g, coords] -> {satisfied, min}.";
CheckDEC::usage =
  "CheckDEC[g, coords] -> {satisfied, max}.";
CheckAllConditions::usage =
  "CheckAllConditions[g, coords] -> Association of WEC/NEC/SEC/DEC.";

EvalAt::usage =
  "EvalAt is an option for the Check* functions.  Pass a list of coordinate \
substitution rules (e.g. EvalAt -> {t -> 0., x -> 1.}) to evaluate metric and \
stress-energy at a numerical point before random observer sampling.  Default \
None leaves both symbolic; sampling then degenerates to the static observer.";

NSamples::usage =
  "NSamples is an option for the Check* functions controlling the number of \
random observer directions sampled.  Defaults: 10 for WEC/SEC/DEC, 20 for NEC.";

Options[CheckWEC]           = {NSamples -> 10, Tolerance -> 10^-10, EvalAt -> None};
Options[CheckNEC]           = {NSamples -> 20, Tolerance -> 10^-10, EvalAt -> None};
Options[CheckSEC]           = {NSamples -> 10, Tolerance -> 10^-10, EvalAt -> None};
Options[CheckDEC]           = {NSamples -> 10, Tolerance -> 10^-10, EvalAt -> None};
Options[CheckAllConditions] = {NSamples -> 10, Tolerance -> 10^-10, EvalAt -> None};

(* ---- Invariants -------------------------------------------- *)
KretschmannScalar::usage =
  "KretschmannScalar[g, coords] = R_{\[Alpha]\[Beta]\[Gamma]\[Delta]} R^{\[Alpha]\[Beta]\[Gamma]\[Delta]}.";
WeylSquared::usage =
  "WeylSquared[g, coords] = C_{\[Alpha]\[Beta]\[Gamma]\[Delta]} C^{\[Alpha]\[Beta]\[Gamma]\[Delta]}.";
RicciSquared::usage =
  "RicciSquared[g, coords] = R_{\[Mu]\[Nu]} R^{\[Mu]\[Nu]}.";
EulerDensity4D::usage =
  "EulerDensity4D[g, coords] = R^2 - 4 R_{\[Mu]\[Nu]} R^{\[Mu]\[Nu]} + Kretschmann.";

(* ---- Geodesics --------------------------------------------- *)
NormalizeVelocity::usage =
  "NormalizeVelocity[u, g, \"Timelike\"|\"Null\"] rescales a 4-velocity.";
CreateInitialVelocity::usage =
  "CreateInitialVelocity[g0, dir, speed, \"Timelike\"|\"Null\"] builds an initial 4-velocity.";
IntegrateGeodesic::usage =
  "IntegrateGeodesic[metricFunc, x0, u0, {\[Lambda]min, \[Lambda]max}] integrates a geodesic via NDSolve.";
IntegrateGeodesicBundle::usage =
  "IntegrateGeodesicBundle[metricFunc, x0, velocities, span] integrates several geodesics.";

Options[IntegrateGeodesic] = {Method -> "StiffnessSwitching",
                              MaxSteps -> 10^5,
                              AccuracyGoal -> 8,
                              PrecisionGoal -> 6};

(* ---- Visualize --------------------------------------------- *)
PlotShapeFunctions::usage =
  "PlotShapeFunctions[R, \[Sigma]] shows the four shape profiles on a single axis.";
PlotEnergyDensity2D::usage =
  "PlotEnergyDensity2D[{v0,R,\[Sigma]}] shows the Alcubierre \[Rho](x,y) slice at z=0.";
PlotExpansionScalar2D::usage =
  "PlotExpansionScalar2D[{v0,R,\[Sigma]}] shows the expansion scalar slice.";
PlotShiftField::usage =
  "PlotShiftField[shiftFunc, xrange, yrange] shows the 2D shift vector field.  \
shiftFunc must be a function of {t,x,y,z} that returns a length-3 vector \
{\[Beta]^x, \[Beta]^y, \[Beta]^z}; pass MetricToADM[g][[2]] (with g a metric \
function) if you only have the metric.";
PlotMetricComparison::usage =
  "PlotMetricComparison[opts] shows a 2x3 grid comparing all six metric families.";
PlotGridDistortion::usage =
  "PlotGridDistortion[{v0,R,\[Sigma]}, t] shows how the coordinate grid is warped.";

(* ============================================================
   Implementation
   ============================================================ *)

Begin["`Private`"];

(* ----- 1. Shape functions ----------------------------------- *)

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

(* ----- 2. Metrics ------------------------------------------- *)

ADMToMetric[alpha_, beta_?VectorQ, gamma_?MatrixQ] :=
  Module[{betaLower = gamma . beta, g},
    g = ConstantArray[0, {4, 4}];
    g[[1, 1]] = -alpha^2 + betaLower . beta;
    g[[1, 2 ;; 4]] = betaLower;
    g[[2 ;; 4, 1]] = betaLower;
    g[[2 ;; 4, 2 ;; 4]] = gamma;
    g];

BubbleCenter[v0_, x0_, t_] := x0 + v0 t;

RFromCenter[{t_, x_, y_, z_}, v0_, x0_] :=
  Sqrt[(x - BubbleCenter[v0, x0, t])^2 + y^2 + z^2];

AlcubierreMetric[{v0_, R_, sigma_}, {t_, x_, y_, z_},
                 OptionsPattern[{Shape -> "tanh", X0 -> 0}]] :=
  Module[{rs, f, shapeFn = ShapeFunction[OptionValue[Shape]],
          x0v = OptionValue[X0]},
    rs = RFromCenter[{t, x, y, z}, v0, x0v];
    f = shapeFn[rs, R, sigma];
    {{-(1 - v0^2 f^2), -v0 f, 0, 0},
     {-v0 f,            1,    0, 0},
     {0,                0,    1, 0},
     {0,                0,    0, 1}}];

NatarioMetric[{v0_, R_, sigma_}, {t_, x_, y_, z_},
              OptionsPattern[{X0 -> 0}]] :=
  Module[{x0v = OptionValue[X0], rs, g, dg, dx, rho2, bx, by, bz, beta, r},
    dx = x - BubbleCenter[v0, x0v, t];
    rs = Sqrt[dx^2 + y^2 + z^2];
    g = TanhShape[rs, R, sigma];
    dg = D[TanhShape[r, R, sigma], r] /. r -> rs;
    rho2 = y^2 + z^2;
    bx = -v0 (2 g + rho2 dg / rs);
    by =  v0 y dg dx / rs;
    bz =  v0 z dg dx / rs;
    beta = {bx, by, bz};
    ADMToMetric[1, beta, IdentityMatrix[3]]];

VanDenBroeckMetric[{v0_, Rext_, Rint_, Bint_, sigma_, sigmaB_},
                   {t_, x_, y_, z_},
                   OptionsPattern[{X0 -> 0}]] :=
  Module[{x0v = OptionValue[X0], rs, f, B, B2},
    rs = RFromCenter[{t, x, y, z}, v0, x0v];
    f  = TanhShape[rs, Rext, sigma];
    B  = 1 + (Bint - 1) TanhShape[rs, Rint, sigmaB];
    B2 = B^2;
    {{-1 + B2 v0^2 f^2, -B2 v0 f, 0,  0},
     {-B2 v0 f,          B2,      0,  0},
     {0,                 0,      B2,  0},
     {0,                 0,       0, B2}}];

toroidalDistance[{t_, x_, y_, z_}, v0_, x0v_, Rmajor_, Rminor_] :=
  Module[{xs, rho, d},
    xs = BubbleCenter[v0, x0v, t];
    rho = Sqrt[y^2 + z^2];
    d = Sqrt[(rho - Rmajor)^2 + (x - xs)^2];
    d - Rminor];

WhiteToroidalMetric[{v0_, Rmajor_, Rminor_, sigma_}, {t_, x_, y_, z_},
                    OptionsPattern[{X0 -> 0}]] :=
  Module[{x0v = OptionValue[X0], d, f},
    d = toroidalDistance[{t, x, y, z}, v0, x0v, Rmajor, Rminor];
    f = (1 - Tanh[sigma d / Rminor])/2;
    {{-(1 - v0^2 f^2), -v0 f, 0, 0},
     {-v0 f,            1,    0, 0},
     {0,                0,    1, 0},
     {0,                0,    0, 1}}];

BobrickMartireMetric::superluminal =
  "Subluminal Bobrick-Martire requires v0 < 1, got ``.";
BobrickMartireMetric[{v0_, Rinner_, Router_, amp_, sigma_}, {t_, x_, y_, z_},
                     OptionsPattern[{X0 -> 0, PositiveEnergy -> True}]] :=
  Module[{x0v = OptionValue[X0], pos = OptionValue[PositiveEnergy],
          rs, stepIn, stepOut, chi, f, lapse, correction, betaX, B2},
    If[v0 >= 1,
      Message[BobrickMartireMetric::superluminal, v0];
      Return[$Failed]];
    rs = RFromCenter[{t, x, y, z}, v0, x0v];
    stepIn  = (1 + Tanh[sigma (rs - Rinner)])/2;
    stepOut = (1 - Tanh[sigma (rs - Router)])/2;
    chi = stepIn stepOut;
    f = TanhShape[rs, Router, sigma];
    lapse = If[pos, 1 + amp chi v0^2 / 2, 1];
    correction = If[pos, 1 - amp chi, 1];
    betaX = -v0 f correction;
    B2 = If[pos, 1 + amp chi (1 - v0^2), 1];
    ADMToMetric[lapse, {betaX, 0, 0}, B2 IdentityMatrix[3]]];

LentzMetric[{v0_, R_, amp_, hparam_}, {t_, x_, y_, z_},
            OptionsPattern[{X0 -> 0}]] :=
  Module[{x0v = OptionValue[X0], rs, phi, Nsq, lapse, betaMag, beta},
    rs = RFromCenter[{t, x, y, z}, v0, x0v];
    phi = Sech[rs/R];
    Nsq = 1 + amp v0^2 phi^2 Cosh[hparam];
    lapse = Sqrt[Max[Nsq, 0.01]];
    betaMag = v0 phi Tanh[hparam] (1 + amp phi);
    beta = {-betaMag, 0, 0};
    ADMToMetric[lapse, beta, IdentityMatrix[3]]];

MetricRegistry::unknown =
  "Unknown metric. Known names: alcubierre, natario, vdbroek, white, bobrick_martire, lentz.";
MetricRegistry["alcubierre"]      = AlcubierreMetric;
MetricRegistry["natario"]         = NatarioMetric;
MetricRegistry["vdbroek"]         = VanDenBroeckMetric;
MetricRegistry["white"]           = WhiteToroidalMetric;
MetricRegistry["bobrick_martire"] = BobrickMartireMetric;
MetricRegistry["lentz"]           = LentzMetric;
MetricRegistry[_] := (Message[MetricRegistry::unknown]; $Failed);

(* ----- 3. Tensors ------------------------------------------- *)

MetricInverse[g_?MatrixQ] := Inverse[g];
MetricInverse[g_] := Inverse[g];

ChristoffelSymbols[g_, coords_List] :=
  Module[{ginv, dg, dim = 4, mu, al, be, nu, alpha},
    ginv = Simplify @ Inverse[g];
    dg = Table[D[g, coords[[alpha]]], {alpha, dim}];
    Table[
      (1/2) Sum[ginv[[mu, nu]] (dg[[al, be, nu]] + dg[[be, al, nu]]
                                - dg[[nu, al, be]]),
                {nu, dim}],
      {mu, dim}, {al, dim}, {be, dim}]];

RiemannTensor[g_, coords_List] :=
  Module[{gamma, dgamma, dim = 4, rho, sig, mu, nu, lam},
    gamma = ChristoffelSymbols[g, coords];
    dgamma = Table[D[gamma, coords[[mu]]], {mu, dim}];
    Table[
      dgamma[[mu, rho, nu, sig]] - dgamma[[nu, rho, mu, sig]]
        + Sum[gamma[[rho, mu, lam]] gamma[[lam, nu, sig]]
              - gamma[[rho, nu, lam]] gamma[[lam, mu, sig]], {lam, dim}],
      {rho, dim}, {sig, dim}, {mu, dim}, {nu, dim}]];

RiemannTensorLower[g_, coords_List] :=
  Module[{R = RiemannTensor[g, coords]},
    TensorContract[TensorProduct[g, R], {{2, 3}}]];

RicciTensor[g_, coords_List] :=
  Module[{R = RiemannTensor[g, coords], rho, mu, nu},
    Table[Sum[R[[rho, mu, rho, nu]], {rho, 4}], {mu, 4}, {nu, 4}]];

RicciScalar[g_, coords_List] :=
  Module[{ginv = Inverse[g], Rmn = RicciTensor[g, coords], mu, nu},
    Sum[ginv[[mu, nu]] Rmn[[mu, nu]], {mu, 4}, {nu, 4}]];

EinsteinTensor[g_, coords_List] :=
  Module[{Rmn = RicciTensor[g, coords], ginv = Inverse[g], Rscal, m, n},
    Rscal = Sum[ginv[[m, n]] Rmn[[m, n]], {m, 4}, {n, 4}];
    Rmn - (1/2) g Rscal];

WeylTensor[g_, coords_List] :=
  Module[{Rlow, Rmn, Rscal, ginv = Inverse[g], rho, sig, mu, nu, m, n},
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
  Module[{ginv, gam, Rup, Rlow, Rmn, Rscal, Gmn, Cmn, m, n, rho, mu, nu},
    ginv  = Inverse[g];
    gam   = ChristoffelSymbols[g, coords];
    Rup   = RiemannTensor[g, coords];
    Rlow  = TensorContract[TensorProduct[g, Rup], {{2, 3}}];
    Rmn   = Table[Sum[Rup[[rho, mu, rho, nu]], {rho, 4}], {mu, 4}, {nu, 4}];
    Rscal = Sum[ginv[[m, n]] Rmn[[m, n]], {m, 4}, {n, 4}];
    Gmn   = Rmn - (1/2) g Rscal;
    Cmn   = WeylTensor[g, coords];
    <|"Metric" -> g, "Inverse" -> ginv, "Christoffel" -> gam,
      "Riemann" -> Rup, "RiemannLower" -> Rlow,
      "Ricci" -> Rmn, "RicciScalar" -> Rscal,
      "Einstein" -> Gmn, "Weyl" -> Cmn|>];

NumericTensors[metricFunc_, {t0_?NumericQ, x0_?NumericQ,
                              y0_?NumericQ, z0_?NumericQ},
               OptionsPattern[{Simplify -> False}]] :=
  Module[{t, x, y, z, g, res, rules},
    g = metricFunc[t, x, y, z];
    res = AllTensors[g, {t, x, y, z}];
    rules = {t -> t0, x -> x0, y -> y0, z -> z0};
    res = If[OptionValue[Simplify],
             Simplify[# /. rules] & /@ res,
             # /. rules & /@ res];
    N[res]];

(* ----- 4. ADM ---------------------------------------------- *)

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
    Table[-(dbeta[[i, j]] + dbeta[[j, i]])/2, {i, 3}, {j, 3}]];

ExpansionScalar[K_?MatrixQ, gammaInv_?MatrixQ] :=
  Sum[gammaInv[[i, j]] K[[i, j]], {i, 3}, {j, 3}];

ShearTensor[K_?MatrixQ, gamma_?MatrixQ, gammaInv_?MatrixQ] :=
  K - (1/3) gamma ExpansionScalar[K, gammaInv];

ShiftDivergence[shiftFunc_, coords_List] :=
  Module[{beta = shiftFunc @@ coords, spatial = coords[[2 ;; 4]], i},
    Sum[D[beta[[i]], spatial[[i]]], {i, 3}]];

NormalVector[alpha_, beta_?VectorQ] := Prepend[-beta/alpha, 1/alpha];

EulerianVelocity[beta_?VectorQ] := Prepend[-beta, 1];

(* ----- 5. Energy ------------------------------------------- *)

StressEnergyTensor[g_, coords_List] := EinsteinTensor[g, coords]/(8 Pi);

EnergyDensity[g_, coords_List, u_:{1, 0, 0, 0}] :=
  Module[{T = StressEnergyTensor[g, coords], m, n},
    Sum[T[[m, n]] u[[m]] u[[n]], {m, 4}, {n, 4}]];

EulerianEnergyDensity[g_, coords_List] :=
  Module[{T = StressEnergyTensor[g, coords], lapse, shift, gamma, n4,
          m, nn},
    {lapse, shift, gamma} = MetricToADM[g];
    n4 = Prepend[-shift, 1]/lapse;
    Sum[T[[m, nn]] n4[[m]] n4[[nn]], {m, 4}, {nn, 4}]];

EulerianEnergyDensity[g_, coords_List, shiftVec_?VectorQ] :=
  Module[{T = StressEnergyTensor[g, coords], n4 = Prepend[-shiftVec, 1],
          m, nn},
    Sum[T[[m, nn]] n4[[m]] n4[[nn]], {m, 4}, {nn, 4}]];

MomentumDensity[g_, coords_List, u_:{1, 0, 0, 0}] :=
  Module[{ginv = Inverse[g], T = StressEnergyTensor[g, coords], Tmixed},
    Tmixed = ginv . T;
    -Tmixed . u];

PressureDecomposition[g_, coords_List] :=
  Module[{ginv = Inverse[g], T = StressEnergyTensor[g, coords],
          Tmixed, Tsp, piso, pr},
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
          uLower, hProj, hUpper, rho, p, hMixed, q, pi, m, n},
    uLower = g . u;
    hProj  = g + Outer[Times, uLower, uLower];
    hUpper = ginv + Outer[Times, u, u];
    rho    = Sum[T[[m, n]] u[[m]] u[[n]], {m, 4}, {n, 4}];
    p      = Sum[hUpper[[m, n]] T[[m, n]], {m, 4}, {n, 4}]/3;
    hMixed = IdentityMatrix[4] + Outer[Times, uLower, u];
    q      = -hMixed . T . u;
    pi     = hMixed . T . Transpose[hMixed] - p hProj;
    <|"rho" -> rho, "pressure" -> p,
      "heatFlux" -> q, "anisotropicStress" -> pi|>];

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

(* ----- 6. Conditions --------------------------------------- *)

randomUnit3[] :=
  Module[{v = RandomVariate[NormalDistribution[], 3]}, v/Norm[v]];

TimelikeVelocities[n_Integer, g_?MatrixQ] :=
  Module[{out = {}, attempts = 0, vm, dir, u, nrm},
    While[Length[out] < n && attempts < 50 n,
      attempts++;
      vm = 0.9 RandomReal[];
      dir = randomUnit3[];
      u = Prepend[vm dir, 1];
      nrm = u . g . u;
      If[NumericQ[nrm] && nrm < 0,
        AppendTo[out, Re[u/Sqrt[-nrm]]]]];
    If[out === {}, {{1., 0., 0., 0.}}, out]];

NullVectors[n_Integer, g_?MatrixQ] :=
  Module[{out = {}, attempts = 0, dir, a, b, c, disc, vp, vm, i, j},
    While[Length[out] < n && attempts < 50 n,
      attempts++;
      dir = randomUnit3[];
      a = Sum[g[[1 + i, 1 + j]] dir[[i]] dir[[j]], {i, 3}, {j, 3}];
      b = 2 Sum[g[[1, 1 + i]] dir[[i]], {i, 3}];
      c = g[[1, 1]];
      disc = b^2 - 4 a c;
      If[NumericQ[disc] && disc >= 0 && a != 0,
        vp = (-b + Sqrt[disc])/(2 a);
        vm = (-b - Sqrt[disc])/(2 a);
        AppendTo[out, Re @ Prepend[vp dir, 1.]];
        If[Length[out] < n,
          AppendTo[out, Re @ Prepend[vm dir, 1.]]]]];
    If[out === {}, {{1., 0., 0., 1.}}, out]];

applyEvalAt[expr_, None] := expr;
applyEvalAt[expr_, rule_Rule] := expr /. rule;
applyEvalAt[expr_, rules_List] := expr /. rules;

CheckWEC[g_, coords_List, OptionsPattern[]] :=
  Module[{T, vals, gEval, n = OptionValue[NSamples],
          tol = OptionValue[Tolerance], ev = OptionValue[EvalAt]},
    T = N @ applyEvalAt[StressEnergyTensor[g, coords], ev];
    gEval = N @ applyEvalAt[g, ev];
    vals = Table[u . T . u,
                 {u, Append[TimelikeVelocities[n, gEval], {1., 0, 0, 0}]}];
    {Min[vals] >= -tol, Min[vals]}];

CheckNEC[g_, coords_List, OptionsPattern[]] :=
  Module[{T, gEval, vals, n = OptionValue[NSamples],
          tol = OptionValue[Tolerance], ev = OptionValue[EvalAt]},
    T = N @ applyEvalAt[StressEnergyTensor[g, coords], ev];
    gEval = N @ applyEvalAt[g, ev];
    vals = Table[k . T . k, {k, NullVectors[n, gEval]}];
    {Min[vals] >= -tol, Min[vals]}];

CheckSEC[g_, coords_List, OptionsPattern[]] :=
  Module[{gEval, T, ginv, tr, Tsec, vals,
          n = OptionValue[NSamples], tol = OptionValue[Tolerance],
          ev = OptionValue[EvalAt], m, nn},
    gEval = N @ applyEvalAt[g, ev];
    T    = N @ applyEvalAt[StressEnergyTensor[g, coords], ev];
    ginv = Inverse[gEval];
    tr   = Sum[ginv[[m, nn]] T[[m, nn]], {m, 4}, {nn, 4}];
    Tsec = T - (1/2) tr gEval;
    vals = Table[u . Tsec . u,
                 {u, Append[TimelikeVelocities[n, gEval], {1., 0, 0, 0}]}];
    {Min[vals] >= -tol, Min[vals]}];

CheckDEC[g_, coords_List, OptionsPattern[]] :=
  Module[{gEval, T, ginv, Tmixed, vals, wec,
          n = OptionValue[NSamples], tol = OptionValue[Tolerance],
          ev = OptionValue[EvalAt]},
    wec = First @ CheckWEC[g, coords,
                           NSamples -> n, Tolerance -> tol, EvalAt -> ev];
    If[!wec, Return[{False, -Infinity}]];
    gEval = N @ applyEvalAt[g, ev];
    T      = N @ applyEvalAt[StressEnergyTensor[g, coords], ev];
    ginv   = Inverse[gEval];
    Tmixed = ginv . T;
    vals   = Table[Module[{J = -Tmixed . u}, J . gEval . J],
                   {u, Append[TimelikeVelocities[n, gEval], {1., 0, 0, 0}]}];
    {Max[vals] <= tol, Max[vals]}];

CheckAllConditions[g_, coords_List, opts : OptionsPattern[]] :=
  <|"WEC" -> CheckWEC[g, coords, opts],
    "NEC" -> CheckNEC[g, coords, opts],
    "SEC" -> CheckSEC[g, coords, opts],
    "DEC" -> CheckDEC[g, coords, opts]|>;

(* ----- 7. Invariants --------------------------------------- *)

raiseRiemann[Rlow_, ginv_] :=
  Module[{step1, step2, step3, step4, a, b, c, d, i},
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
  Module[{Rlow = RiemannTensorLower[g, coords],
          ginv = Inverse[g], Rup, a, b, c, d},
    Rup = raiseRiemann[Rlow, ginv];
    Sum[Rlow[[a, b, c, d]] Rup[[a, b, c, d]],
        {a, 4}, {b, 4}, {c, 4}, {d, 4}]];

WeylSquared[g_, coords_List] :=
  Module[{Clow = WeylTensor[g, coords], ginv = Inverse[g], Cup,
          a, b, c, d},
    Cup = raiseRiemann[Clow, ginv];
    Sum[Clow[[a, b, c, d]] Cup[[a, b, c, d]],
        {a, 4}, {b, 4}, {c, 4}, {d, 4}]];

RicciSquared[g_, coords_List] :=
  Module[{Rmn = RicciTensor[g, coords], ginv = Inverse[g], Rup, m, n},
    Rup = ginv . Rmn . ginv;
    Sum[Rmn[[m, n]] Rup[[m, n]], {m, 4}, {n, 4}]];

EulerDensity4D[g_, coords_List] :=
  Module[{Rscal = RicciScalar[g, coords],
          R2    = RicciSquared[g, coords],
          K     = KretschmannScalar[g, coords]},
    Rscal^2 - 4 R2 + K];

(* ----- 8. Geodesics ---------------------------------------- *)

NormalizeVelocity::notTimelike = "Vector not timelike.";
NormalizeVelocity[u_?VectorQ, g_?MatrixQ, "Timelike"] :=
  Module[{n = u . g . u},
    If[n >= 0, Message[NormalizeVelocity::notTimelike]; u,
      u/Sqrt[-n]]];
NormalizeVelocity[u_?VectorQ, _?MatrixQ, "Null"] := u;

CreateInitialVelocity::noNull = "Cannot construct null vector in this direction.";
CreateInitialVelocity[g0_?MatrixQ, direction_?VectorQ, speed_, type_:"Timelike"] :=
  Module[{n = direction/Norm[direction], uRaw, disc, a, b, c, v, i, j},
    Switch[type,
      "Timelike",
        uRaw = Prepend[speed n, 1.];
        NormalizeVelocity[uRaw, g0, "Timelike"],
      "Null",
        a = Sum[g0[[1 + i, 1 + j]] n[[i]] n[[j]], {i, 3}, {j, 3}];
        b = 2 Sum[g0[[1, 1 + i]] n[[i]], {i, 3}];
        c = g0[[1, 1]];
        disc = b^2 - 4 a c;
        If[disc < 0, Message[CreateInitialVelocity::noNull]; Return[$Failed]];
        v = (-b + Sqrt[disc])/(2 a);
        Prepend[v n, 1.]]];

IntegrateGeodesic[metricFunc_, x0_?VectorQ, u0_?VectorQ,
                  {lamMin_?NumericQ, lamMax_?NumericQ},
                  opts : OptionsPattern[]] :=
  Module[{t, x, y, z, lam, coordSym, chris, chrisN,
          T, X, Y, Z, U0, U1, U2, U3, sol, eqs, ics, a, b},
    coordSym = {t, x, y, z};
    chris = ChristoffelSymbols[metricFunc @@ coordSym, coordSym];
    chrisN = With[{expr = chris, vars = coordSym},
                  Function @@ {vars, expr}];
    eqs = {
      T'[lam]  == U0[lam],
      X'[lam]  == U1[lam],
      Y'[lam]  == U2[lam],
      Z'[lam]  == U3[lam],
      U0'[lam] == -Sum[chrisN[T[lam], X[lam], Y[lam], Z[lam]][[1, a, b]] *
                         ({U0[lam], U1[lam], U2[lam], U3[lam]}[[a]]) *
                         ({U0[lam], U1[lam], U2[lam], U3[lam]}[[b]]),
                       {a, 4}, {b, 4}],
      U1'[lam] == -Sum[chrisN[T[lam], X[lam], Y[lam], Z[lam]][[2, a, b]] *
                         ({U0[lam], U1[lam], U2[lam], U3[lam]}[[a]]) *
                         ({U0[lam], U1[lam], U2[lam], U3[lam]}[[b]]),
                       {a, 4}, {b, 4}],
      U2'[lam] == -Sum[chrisN[T[lam], X[lam], Y[lam], Z[lam]][[3, a, b]] *
                         ({U0[lam], U1[lam], U2[lam], U3[lam]}[[a]]) *
                         ({U0[lam], U1[lam], U2[lam], U3[lam]}[[b]]),
                       {a, 4}, {b, 4}],
      U3'[lam] == -Sum[chrisN[T[lam], X[lam], Y[lam], Z[lam]][[4, a, b]] *
                         ({U0[lam], U1[lam], U2[lam], U3[lam]}[[a]]) *
                         ({U0[lam], U1[lam], U2[lam], U3[lam]}[[b]]),
                       {a, 4}, {b, 4}]};
    ics = {T[lamMin]  == x0[[1]], X[lamMin]  == x0[[2]],
           Y[lamMin]  == x0[[3]], Z[lamMin]  == x0[[4]],
           U0[lamMin] == u0[[1]], U1[lamMin] == u0[[2]],
           U2[lamMin] == u0[[3]], U3[lamMin] == u0[[4]]};
    sol = NDSolve[Join[eqs, ics],
                  {T, X, Y, Z, U0, U1, U2, U3},
                  {lam, lamMin, lamMax},
                  Sequence @@ FilterRules[{opts}, Options[NDSolve]]];
    <|"Solution" -> First[sol],
      "LambdaRange" -> {lamMin, lamMax},
      "Coords" -> Function[l, {T[l], X[l], Y[l], Z[l]} /. First[sol]],
      "Velocity" -> Function[l, {U0[l], U1[l], U2[l], U3[l]} /. First[sol]]|>];

IntegrateGeodesicBundle[metricFunc_, x0_?VectorQ, us_List, span_List,
                        opts : OptionsPattern[]] :=
  Table[IntegrateGeodesic[metricFunc, x0, u, span, opts], {u, us}];

(* ----- 9. Visualize ---------------------------------------- *)

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
                    OptionsPattern[{Range -> 3, Grid -> 80}]] :=
  Module[{rng = OptionValue[Range], np = OptionValue[Grid]},
    DensityPlot[
      AlcubierreAnalyticEnergyDensity[{v0, R, sigma}, {0, x, y, 0}],
      {x, -rng, rng}, {y, -rng, rng},
      PlotPoints -> np, ColorFunction -> "TemperatureMap",
      PlotLegends -> Automatic,
      PlotLabel -> "Alcubierre \[Rho](x,y)",
      FrameLabel -> {"x", "y"}, ImageSize -> 500]];

PlotExpansionScalar2D[{v0_, R_, sigma_},
                      OptionsPattern[{Range -> 3, Grid -> 80}]] :=
  Module[{rng = OptionValue[Range], np = OptionValue[Grid]},
    DensityPlot[
      AlcubierreExpansionScalar[{v0, R, sigma}, {0, x, y, 0}],
      {x, -rng, rng}, {y, -rng, rng},
      PlotPoints -> np, ColorFunction -> "TemperatureMap",
      PlotLegends -> Automatic,
      PlotLabel -> "Expansion scalar \[Theta](x,y)",
      FrameLabel -> {"x", "y"}, ImageSize -> 500]];

PlotShiftField::not3vec =
  "PlotShiftField expected shiftFunc to return a length-3 vector but got `1`.";
PlotShiftField[shiftFunc_, {xMin_, xMax_}, {yMin_, yMax_}] :=
  Module[{probe = shiftFunc[0., 0., 0., 0.]},
    If[!(VectorQ[probe] && Length[probe] === 3),
      Message[PlotShiftField::not3vec, probe]; Return[$Failed]];
    VectorPlot[{shiftFunc[0, x, y, 0][[1]], shiftFunc[0, x, y, 0][[2]]},
               {x, xMin, xMax}, {y, yMin, yMax},
               VectorPoints -> 15,
               VectorScale -> {Small, Automatic, None},
               PlotLabel -> "Shift vector field \[Beta](x,y)",
               FrameLabel -> {"x", "y"}, ImageSize -> 500]];

PlotMetricComparison[OptionsPattern[{Range -> 3, Grid -> 80}]] :=
  Module[{rng = OptionValue[Range], np = OptionValue[Grid], specs, plots, r},
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
                    PlotPoints -> np, ColorFunction -> "TemperatureMap",
                    PlotLabel ->
                      Column[{name, Style[sub, Italic, Smaller]}],
                    FrameLabel -> {"x", "y"}, ImageSize -> 280,
                    PlotLegends -> Automatic]],
      {spec, specs}];
    Grid[Partition[plots, 3], Frame -> All]];

PlotGridDistortion[{v0_, R_, sigma_}, t_: 0,
                   OptionsPattern[{Range -> 3, Lines -> 15, Steps -> 40}]] :=
  Module[{rng = OptionValue[Range], nLines = OptionValue[Lines],
          nSteps = OptionValue[Steps], shiftXFn, advect, step, gridLines, i},
    shiftXFn[tt_, xx_, yy_] :=
      -v0 TanhShape[Sqrt[(xx - v0 tt)^2 + yy^2], R, sigma];
    (* Forward-Euler advect a comoving particle from (x0, y0) at t=0
       to its position at time t under the Alcubierre x-shift. *)
    advect[{x0_, y0_}] :=
      Module[{xcur = x0, ycur = y0, dt},
        If[t == 0, {x0, y0},
          dt = t/nSteps;
          Do[xcur = xcur + dt shiftXFn[(i - 1) dt, xcur, ycur],
             {i, 1, nSteps}];
          {xcur, ycur}]];
    step = 2 rng/nLines;
    (* Lines of constant initial y, advected: each curve is a row of
       comoving particles connected after advection. *)
    gridLines = Join[
      Table[Line[Table[advect[{x, y}], {x, -rng, rng, step}]],
            {y, -rng, rng, step}],
      Table[Line[Table[advect[{x, y}], {y, -rng, rng, step}]],
            {x, -rng, rng, step}]];
    Show[
      DensityPlot[Abs[shiftXFn[t, x, y]],
                  {x, -rng, rng}, {y, -rng, rng},
                  ColorFunction -> "SunsetColors", PlotPoints -> 60,
                  Frame -> True, FrameLabel -> {"x", "y"},
                  PlotLegends -> Automatic],
      Graphics[{Black, Thickness[0.002], gridLines}],
      ImageSize -> 500,
      PlotLabel ->
        Row[{"Alcubierre shift magnitude at t=", t}]]];

End[];
EndPackage[];
